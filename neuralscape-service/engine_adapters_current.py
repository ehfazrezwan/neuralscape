"""Adapters that present the current Qdrant and Graphiti stack behind engine ports.

These wrap the access the service already uses: the mem0 ``QdrantVectorStore``
(``user_id``/``data`` payload convention, ``vector_store.client`` for filtered
queries) and the Graphiti instance behind the mem0 graph adapter (episodes,
entity nodes and edges, native ``remove_episode``).  Nothing here is imported by
the running service and ``memory/`` is not modified.

Manifests are honest declarations: operations are supported and configured, but
nothing is qualified, and health is only ``healthy`` after a live probe.  The
adapters add the guards the stack does not enforce itself (tenant scoping,
revision fencing) and otherwise preserve native behavior, so differences from
the port contract surface in the conformance suite rather than being hidden.

Known native properties that the adapters cannot repair:

* Qdrant read-compare-write revision fencing is not atomic across concurrent
  writers.
* Graphiti ``remove_episode`` deletes only edges first created by that episode
  and does not detach a deleted episode from edges it shares with others.  The
  graph adapter therefore does not use it: removal is implemented here, in one
  transaction, as support-aware detachment (see ``GraphitiGraphAdapter``).
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

import third_party_telemetry  # noqa: F401 — must precede any mem0/Graphiti import (defaults their telemetry off)

from contracts_engines import (
    CapabilityHealth,
    CapabilityManifest,
    CapabilityOperationState,
    EngineAdapterKind,
)
from contracts_errors import InternalErrorCode
from engine_ports import (
    CONTRACT_VERSION,
    OPERATIONS_BY_KIND,
    AdapterRegistry,
    EngineRecord,
    GraphEdge,
    GraphEpisode,
    OperationContext,
    PortFailure,
    ProjectionReceipt,
    SearchHit,
    TenantFilter,
    VectorStorePort,
)

ADAPTER_VERSION = "current-stack-v1"
_NAMESPACE = uuid.UUID("5d6f1c1e-6f0a-5b39-9a53-2b6d3d6c0d11")
_FAKE_EMBEDDING_DIMS = 8


def _stable_uuid(*parts: str) -> str:
    """Deterministic UUID for a tenant-scoped canonical identity."""

    return str(uuid.uuid5(_NAMESPACE, "\x1f".join(parts)))


def _manifest(
    implementation_id: str,
    kind: EngineAdapterKind,
    *,
    health: CapabilityHealth,
    unsupported: frozenset[str] = frozenset(),
) -> CapabilityManifest:
    operations = []
    for name in OPERATIONS_BY_KIND[kind]:
        supported = name not in unsupported
        operations.append(
            CapabilityOperationState(
                operation=name,
                supported=supported,
                configured=supported,
                health=health if supported else CapabilityHealth.UNKNOWN,
                qualification=None,
            )
        )
    return CapabilityManifest(
        schema_version=CONTRACT_VERSION,
        implementation_id=implementation_id,
        implementation_version=ADAPTER_VERSION,
        adapter_kind=kind,
        contract_schema_versions=(CONTRACT_VERSION,),
        operations=tuple(operations),
    )


def _check_tenant(ctx: OperationContext, tenant_id: str) -> None:
    if ctx.tenant_id != tenant_id:
        raise PortFailure(InternalErrorCode.DENIED, "context tenant mismatch")


def _receipt(
    engine_id: str, object_id: str, revision: int, operation: str
) -> ProjectionReceipt:
    return ProjectionReceipt(
        object_id=object_id,
        revision=revision,
        operation=operation,
        engine_id=engine_id,
        engine_version=ADAPTER_VERSION,
        index_version=ADAPTER_VERSION,
    )


class QdrantVectorAdapter:
    """``VectorStorePort`` over a mem0 ``QdrantVectorStore``.

    ``embed`` maps text to the collection's dense vector; it is injected so the
    adapter never chooses a provider.  A canonical record maps to one point
    whose id is a deterministic UUID of (tenant, record id); the canonical id,
    revision and labels live in the payload next to the service's ``user_id``
    (tenant) and ``data`` (text) keys.
    """

    def __init__(
        self,
        vector_store: Any,
        embed: Callable[[str], list[float]],
        implementation_id: str = "current-qdrant",
    ) -> None:
        self._store = vector_store
        self._embed = embed
        self._implementation_id = implementation_id

    # -- capability ---------------------------------------------------------

    def destroy(self) -> None:
        """Delete this generation's collection and every point in it (used when a rollback target is released)."""

        self._store.client.delete_collection(self._store.collection_name)

    def manifest(self) -> CapabilityManifest:
        health = CapabilityHealth.UNHEALTHY
        try:
            self._store.client.get_collection(self._store.collection_name)
            health = CapabilityHealth.HEALTHY
        except Exception:
            pass
        return _manifest(self._implementation_id, EngineAdapterKind.VECTOR_STORE, health=health)

    # -- helpers ------------------------------------------------------------

    def _point_id(self, tenant_id: str, record_id: str) -> str:
        return _stable_uuid("vector", tenant_id, record_id)

    def _row(self, tenant_id: str, record_id: str) -> Optional[tuple[int, bool, dict]]:
        """(revision, removed, payload) of the point, or None when unknown."""

        points = self._store.client.retrieve(
            collection_name=self._store.collection_name,
            ids=[self._point_id(tenant_id, record_id)],
            with_payload=True,
        )
        if not points:
            return None
        payload = points[0].payload or {}
        if payload.get("user_id") != tenant_id or payload.get("canonical_id") != record_id:
            return None
        return payload["revision"], bool(payload.get("removed")), payload

    def _stored(self, tenant_id: str, record_id: str) -> Optional[EngineRecord]:
        row = self._row(tenant_id, record_id)
        if row is None or row[1]:
            return None
        revision, _removed, payload = row
        return EngineRecord(
            record_id=record_id,
            revision=revision,
            tenant_id=tenant_id,
            text=payload.get("data", ""),
            labels=tuple(payload.get("labels", ())),
        )

    def _receipt(self, record_id: str, revision: int, operation: str) -> ProjectionReceipt:
        return _receipt(self._implementation_id, record_id, revision, operation)

    # -- port ---------------------------------------------------------------

    def apply(self, ctx: OperationContext, record: EngineRecord) -> ProjectionReceipt:
        _check_tenant(ctx, record.tenant_id)
        row = self._row(record.tenant_id, record.record_id)
        if row is not None:
            stored_revision, removed, _payload = row
            if record.revision < stored_revision:
                raise PortFailure(InternalErrorCode.STALE_REVISION)
            if record.revision == stored_revision:
                stored = None if removed else self._stored(record.tenant_id, record.record_id)
                if stored is None or stored.text != record.text:
                    raise PortFailure(InternalErrorCode.IDEMPOTENCY_CONFLICT)
                if stored.labels != tuple(record.labels):
                    # Same revision and text, other labels: a relabel.  Only the payload changes; the
                    # vector (and so any embedding call) is left alone.
                    self._relabel(record)
                return self._receipt(record.record_id, record.revision, "applied")
        self._store.insert(
            vectors=[self._embed(record.text)],
            ids=[self._point_id(record.tenant_id, record.record_id)],
            payloads=[
                {
                    "user_id": record.tenant_id,
                    "data": record.text,
                    "canonical_id": record.record_id,
                    "revision": record.revision,
                    "labels": list(record.labels),
                }
            ],
        )
        return self._receipt(record.record_id, record.revision, "applied")

    def _relabel(self, record: EngineRecord) -> None:
        """Replace the labels of the point at exactly this revision, fenced in the store.

        The revision check in ``apply`` and this write are two calls, so a newer revision (or a removal)
        can land in between, from this process or another.  The write therefore carries its own
        condition: the point must still be at ``record.revision`` and not removed, evaluated by the
        store in the same call.  When the condition no longer holds nothing is written, and the point
        is read again to say why: a newer revision is stale for this write, a removal is a conflict.
        """

        from qdrant_client import models

        point_id = self._point_id(record.tenant_id, record.record_id)
        self._store.client.set_payload(
            collection_name=self._store.collection_name,
            payload={"labels": list(record.labels)},
            points=models.Filter(
                must=[
                    models.HasIdCondition(has_id=[point_id]),
                    models.FieldCondition(key="revision", match=models.MatchValue(value=record.revision)),
                ],
                must_not=[models.FieldCondition(key="removed", match=models.MatchValue(value=True))],
            ),
        )
        now = self._row(record.tenant_id, record.record_id)
        if now is None or now[1]:
            raise PortFailure(InternalErrorCode.IDEMPOTENCY_CONFLICT)
        if now[0] > record.revision:
            raise PortFailure(InternalErrorCode.STALE_REVISION)

    def remove(
        self, ctx: OperationContext, record_id: str, revision: int
    ) -> ProjectionReceipt:
        row = self._row(ctx.tenant_id, record_id)
        if row is None or row[1]:
            raise PortFailure(InternalErrorCode.MISSING_RESOURCE)
        if revision < row[0]:
            raise PortFailure(InternalErrorCode.STALE_REVISION)
        # Overwrite the point with a content-free tombstone (no text, labels or
        # meaningful vector) that keeps only the removal revision, so an older
        # revision cannot be re-applied.  Searches and reads exclude it.
        dims = getattr(self._store, "embedding_model_dims", None) or len(self._embed(""))
        self._store.insert(
            vectors=[[0.0] * dims],
            ids=[self._point_id(ctx.tenant_id, record_id)],
            payloads=[
                {
                    "user_id": ctx.tenant_id,
                    "canonical_id": record_id,
                    "revision": revision,
                    "removed": True,
                }
            ],
        )
        return self._receipt(record_id, revision, "removed")

    def get(self, ctx: OperationContext, record_id: str) -> Optional[EngineRecord]:
        return self._stored(ctx.tenant_id, record_id)

    def search(
        self, ctx: OperationContext, query: str, flt: TenantFilter, limit: int
    ) -> tuple[SearchHit, ...]:
        from qdrant_client import models

        _check_tenant(ctx, flt.tenant_id)
        if limit <= 0 or flt.authorized_record_ids == ():
            return ()
        must = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=flt.tenant_id))
        ]
        for label in flt.labels_all:
            must.append(
                models.FieldCondition(key="labels", match=models.MatchValue(value=label))
            )
        if flt.authorized_record_ids is not None:
            must.append(
                models.FieldCondition(
                    key="canonical_id",
                    match=models.MatchAny(any=list(flt.authorized_record_ids)),
                )
            )
        response = self._store.client.query_points(
            collection_name=self._store.collection_name,
            query=self._embed(query),
            query_filter=models.Filter(
                must=must,
                must_not=[
                    models.FieldCondition(key="removed", match=models.MatchValue(value=True))
                ],
            ),
            limit=limit,
            with_payload=True,
        )
        return tuple(
            SearchHit(
                record_id=point.payload["canonical_id"],
                revision=point.payload["revision"],
                score=float(point.score),
            )
            for point in response.points
        )


_GRAPH_UNSUPPORTED: frozenset[str] = frozenset()


class GraphitiGraphAdapter:
    """``GraphStorePort`` over a Graphiti instance.

    ``run`` executes a coroutine on the loop the Graphiti driver is bound to
    (the mem0 graph adapter's bridge ``run``).  The tenant maps to the Graphiti
    ``group_id``, matching the service's legacy ``user_id``-as-group convention.
    Episodes are written structurally (nodes, edges, mentions) with no LLM.

    Removal is support-aware and tenant-scoped, in a single transaction: the
    removed episodes are detached from every edge's supporting-episode list, an
    edge is deleted only when that list becomes empty, entity nodes are deleted
    only when no remaining edge or mention references them, and a content-free
    fence (``removed_revision`` on a dedicated ``NsFence`` node, keyed by a
    one-way hash of the source id) records the removal revision so an older
    episode for that source cannot be re-applied, even when the removal arrives
    before the source was ever applied.  Removing an already removed source is
    idempotent, and a removal older than the fence is ``stale_revision``.  Applies
    and removals are atomic and serialized per source, edge and entity with
    ordered locks (see "concurrency control" below); concurrent applies of
    episodes that share an edge keep all of their support.  Graphiti's own
    ``remove_episode`` is not used (it is support-blind).  ``authorized_record_ids``
    in a filter are *source* record ids: an edge is visible when at least one of
    its live supporting sources is authorized.
    """

    def __init__(
        self,
        graphiti: Any,
        run: Callable[[Awaitable[Any]], Any],
        implementation_id: str = "current-graphiti",
    ) -> None:
        self._graphiti = graphiti
        self._run = run
        self._implementation_id = implementation_id
        self._indexed = False
        self._index_lock = threading.Lock()

    def manifest(self) -> CapabilityManifest:
        health = CapabilityHealth.UNHEALTHY
        try:
            async def probe() -> None:
                await self._graphiti.driver.execute_query("RETURN 1", routing_="r")

            self._run(probe())
            health = CapabilityHealth.HEALTHY
        except Exception:
            pass
        return _manifest(
            self._implementation_id,
            EngineAdapterKind.GRAPH_STORE,
            health=health,
            unsupported=_GRAPH_UNSUPPORTED,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _source_prefix(source_record_id: str) -> str:
        return '{"ns_source":' + json.dumps(source_record_id) + ","

    @staticmethod
    def _source_description(source_record_id: str, revision: int) -> str:
        return json.dumps(
            {"ns_source": source_record_id, "revision": revision}, separators=(",", ":")
        )

    def _ensure_indices(self) -> None:
        if not self._indexed:
            with self._index_lock:
                if not self._indexed:
                    self._run(self._graphiti.build_indices_and_constraints())
                    # A uniqueness constraint makes concurrent MERGE of one lock
                    # node create exactly one node.
                    self._run(
                        self._graphiti.driver.execute_query(
                            "CREATE CONSTRAINT ns_lock_key IF NOT EXISTS "
                            "FOR (l:NsLock) REQUIRE l.key IS UNIQUE"
                        )
                    )
                    self._run(
                        self._graphiti.driver.execute_query(
                            "CREATE CONSTRAINT ns_fence_key IF NOT EXISTS "
                            "FOR (f:NsFence) REQUIRE f.key IS UNIQUE"
                        )
                    )
                    self._indexed = True

    def _receipt(self, object_id: str, revision: int, operation: str) -> ProjectionReceipt:
        return _receipt(self._implementation_id, object_id, revision, operation)

    @staticmethod
    def _fake_embedding() -> list[float]:
        return [0.0] * _FAKE_EMBEDDING_DIMS

    # -- port ---------------------------------------------------------------

    # -- concurrency control --------------------------------------------------
    #
    # Applies and removals run in one write transaction each.  Conflicting
    # writers are serialized by exclusive-write locks on ``NsLock`` nodes
    # (unique key), always taken in one global order: the source fence first,
    # then edges, then entity nodes, each sorted.  With the locks held the
    # read-modify-write of an edge's supporting-episode list, the fence check
    # and the node/edge creation are atomic with respect to every other apply
    # or removal that could touch the same source, edge or entity.  A deadlock
    # between independent transactions is retried by the driver.  The fence is
    # stored on a dedicated ``NsFence`` node (``removed_revision``) and is written
    # even when a removal finds nothing, so a delayed older apply cannot
    # resurrect a source whose removal arrived first.  Every ``NsLock`` node is
    # only a mutex: it is deleted before its transaction commits, so the graph
    # keeps one content-free, hashed fence per removed source (the tombstone,
    # bounded by the number of erasures) and no lock node at all.  Lock
    # keys use a one-way hash of the identity, never the identity itself.

    @staticmethod
    def _lock_key(kind: str, group: str, identity: str) -> str:
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"{kind}\x1f{group}\x1f{digest}"

    @staticmethod
    async def _lock(tx: Any, keys: list[str], held: list[str]) -> None:
        if not keys:
            return
        held.extend(keys)
        result = await tx.run(
            "UNWIND range(0, size($keys) - 1) AS i "
            "MERGE (l:NsLock {key: $keys[i]}) "
            "SET l.version = coalesce(l.version, 0) + 1",
            keys=keys,
        )
        await result.consume()

    @staticmethod
    async def _release(tx: Any, held: list[str]) -> None:
        """Delete every mutex node this transaction took; the removal fence is a separate node."""

        if not held:
            return
        result = await tx.run(
            "MATCH (l:NsLock) WHERE l.key IN $keys DETACH DELETE l",
            keys=held,
        )
        await result.consume()

    async def _run_query(self, tx: Any, query: str, **params: Any) -> list[dict]:
        result = await tx.run(query, **params)
        rows = await result.data()
        return rows

    def apply_episode(self, ctx: OperationContext, episode: GraphEpisode) -> ProjectionReceipt:
        _check_tenant(ctx, episode.tenant_id)
        self._ensure_indices()
        self._run(self._apply(episode))
        return self._receipt(episode.episode_id, episode.revision, "applied")

    async def _apply(self, episode: GraphEpisode) -> None:
        from graphiti_core.models.edges.edge_db_queries import (
            EPISODIC_EDGE_SAVE,
            get_entity_edge_save_query,
        )
        from graphiti_core.models.nodes.node_db_queries import (
            get_entity_node_save_query,
            get_episode_node_save_query,
        )

        driver = self._graphiti.driver
        provider = driver.provider
        group = episode.tenant_id
        episode_uuid = _stable_uuid("episode", group, episode.episode_id)
        fence_key = self._lock_key("0fence", group, episode.source_record_id)
        now = datetime.now(timezone.utc)
        node_uuids = {
            node_id: _stable_uuid("node", group, node_id)
            for edge in episode.edges
            for node_id in (edge.subject_id, edge.object_id)
        }
        edge_uuids = {edge.edge_id: _stable_uuid("edge", group, edge.edge_id) for edge in episode.edges}

        async def work(tx: Any) -> None:
            held: list[str] = []
            await self._lock(tx, [fence_key], held)
            rows = await self._run_query(
                tx, "MATCH (f:NsFence {key: $k}) RETURN f.removed_revision AS revision", k=fence_key
            )
            fence = rows[0]["revision"] if rows else None
            if fence is not None and episode.revision <= fence:
                raise PortFailure(InternalErrorCode.STALE_REVISION)
            existing = await self._run_query(
                tx,
                "MATCH (e:Episodic {uuid: $u, group_id: $g}) "
                "RETURN e.source_description AS description",
                u=episode_uuid,
                g=group,
            )
            supersede = False
            old_edges: list[str] = []
            old_nodes: list[str] = []
            if existing:
                stored_revision = json.loads(existing[0]["description"]).get("revision", 0)
                if episode.revision < stored_revision:
                    raise PortFailure(InternalErrorCode.STALE_REVISION)
                if episode.revision > stored_revision:
                    supersede = True
                    old_edges, old_nodes = await self._footprint(tx, group, [episode_uuid])
            # The fence lock makes this source's episodes stable; now lock every
            # edge and entity the apply (and a supersede) can touch, in order.
            await self._lock(
                tx,
                [self._lock_key("1edge", group, u) for u in sorted({*edge_uuids.values(), *old_edges})]
                + [self._lock_key("2node", group, u) for u in sorted({*node_uuids.values(), *old_nodes})],
                held,
            )
            if supersede:
                await self._detach_locked(tx, group, [episode_uuid])

            for node_id, node_uuid in node_uuids.items():
                entity = {
                    "uuid": node_uuid,
                    "name": node_id,
                    "name_embedding": self._fake_embedding(),
                    "group_id": group,
                    "summary": "",
                    "created_at": now,
                }
                await (
                    await tx.run(get_entity_node_save_query(provider, "Entity"), entity_data=entity)
                ).consume()
            for edge in episode.edges:
                edge_uuid = edge_uuids[edge.edge_id]
                prior = await self._run_query(
                    tx,
                    "MATCH (:Entity {group_id: $g})-[r:RELATES_TO {uuid: $u, group_id: $g}]->(:Entity) "
                    "RETURN r.episodes AS episodes",
                    g=group,
                    u=edge_uuid,
                )
                episodes = list(prior[0]["episodes"] or []) if prior else []
                if episode_uuid not in episodes:
                    episodes.append(episode_uuid)
                edge_data = {
                    "source_uuid": node_uuids[edge.subject_id],
                    "target_uuid": node_uuids[edge.object_id],
                    "uuid": edge_uuid,
                    "name": edge.predicate,
                    "group_id": group,
                    "fact": f"{edge.subject_id} {edge.predicate} {edge.object_id}",
                    "fact_embedding": self._fake_embedding(),
                    "episodes": episodes,
                    "created_at": now,
                    "expired_at": None,
                    "valid_at": None,
                    "invalid_at": None,
                    "reference_time": None,
                    "ns_edge_id": edge.edge_id,
                    "ns_subject": edge.subject_id,
                    "ns_predicate": edge.predicate,
                    "ns_object": edge.object_id,
                }
                await (
                    await tx.run(get_entity_edge_save_query(provider), edge_data=edge_data)
                ).consume()
            await (
                await tx.run(
                    get_episode_node_save_query(provider),
                    uuid=episode_uuid,
                    name=episode.episode_id,
                    group_id=group,
                    source_description=self._source_description(
                        episode.source_record_id, episode.revision
                    ),
                    content="",
                    entity_edges=sorted(edge_uuids.values()),
                    created_at=now,
                    valid_at=now,
                    source="text",
                )
            ).consume()
            for node_uuid in node_uuids.values():
                await (
                    await tx.run(
                        EPISODIC_EDGE_SAVE,
                        episode_uuid=episode_uuid,
                        entity_uuid=node_uuid,
                        uuid=_stable_uuid("mention", episode_uuid, node_uuid),
                        group_id=group,
                        created_at=now,
                    )
                ).consume()
            await self._release(tx, held)

        async with driver.session() as session:
            await session.execute_write(work)

    def remove_by_source(
        self, ctx: OperationContext, source_record_id: str, revision: int
    ) -> ProjectionReceipt:
        self._ensure_indices()
        outcome = self._run(self._remove(ctx.tenant_id, source_record_id, revision))
        if outcome == "missing":
            # The fence was committed above, so a delayed older apply is still
            # refused; the contract's "nothing was ever there" signal is kept.
            raise PortFailure(InternalErrorCode.MISSING_RESOURCE)
        return self._receipt(source_record_id, revision, "removed")

    async def _remove(self, group: str, source_record_id: str, revision: int) -> str:
        """Return ``removed``, ``already`` (idempotent repeat) or ``missing`` (never applied)."""

        fence_key = self._lock_key("0fence", group, source_record_id)

        async def work(tx: Any) -> str:
            held: list[str] = []
            await self._lock(tx, [fence_key], held)
            fence_rows = await self._run_query(
                tx, "MATCH (f:NsFence {key: $k}) RETURN f.removed_revision AS revision", k=fence_key
            )
            prior = fence_rows[0]["revision"] if fence_rows else None
            rows = await self._run_query(
                tx,
                "MATCH (e:Episodic {group_id: $group_id}) "
                "WHERE e.source_description STARTS WITH $prefix "
                "RETURN e.uuid AS uuid, e.source_description AS description",
                group_id=group,
                prefix=self._source_prefix(source_record_id),
            )
            for row in rows:
                if revision < json.loads(row["description"]).get("revision", 0):
                    raise PortFailure(InternalErrorCode.STALE_REVISION)
            uuids = [row["uuid"] for row in rows]
            if not uuids and prior is not None and revision < prior:
                raise PortFailure(InternalErrorCode.STALE_REVISION)  # older than a removal already recorded
            if uuids:
                edges, nodes = await self._footprint(tx, group, uuids)
                await self._lock(
                    tx,
                    [self._lock_key("1edge", group, u) for u in sorted(edges)]
                    + [self._lock_key("2node", group, u) for u in sorted(nodes)],
                    held,
                )
                await self._detach_locked(tx, group, uuids)
            # The fence only ever rises, and is written even when nothing was found.
            await (
                await tx.run(
                    "MERGE (f:NsFence {key: $k}) "
                    "SET f.removed_revision = CASE WHEN f.removed_revision IS NULL "
                    "OR f.removed_revision < $revision THEN $revision ELSE f.removed_revision END",
                    k=fence_key,
                    revision=revision,
                )
            ).consume()
            await self._release(tx, held)
            if uuids:
                return "removed"
            return "already" if prior is not None else "missing"

        async with self._graphiti.driver.session() as session:
            return await session.execute_write(work)

    async def _footprint(
        self, tx: Any, group: str, episode_uuids: list[str]
    ) -> tuple[list[str], list[str]]:
        """Edge and entity uuids that the given episodes support or mention."""

        edges = await self._run_query(
            tx,
            "MATCH (:Entity {group_id: $g})-[r:RELATES_TO {group_id: $g}]->(:Entity {group_id: $g}) "
            "WHERE any(u IN r.episodes WHERE u IN $uuids) RETURN r.uuid AS uuid",
            g=group,
            uuids=episode_uuids,
        )
        nodes = await self._run_query(
            tx,
            "MATCH (e:Episodic {group_id: $g})-[:MENTIONS]->(n:Entity {group_id: $g}) "
            "WHERE e.uuid IN $uuids RETURN DISTINCT n.uuid AS uuid",
            g=group,
            uuids=episode_uuids,
        )
        return [row["uuid"] for row in edges], [row["uuid"] for row in nodes]

    async def _detach_locked(self, tx: Any, group: str, episode_uuids: list[str]) -> dict[str, int]:
        """Support-aware removal of episodes; the caller holds every needed lock.

        Returns ``{"edges_kept", "edges_removed", "nodes_removed"}``.
        """

        candidates = (await self._footprint(tx, group, episode_uuids))[1]
        # Remaining support counts only episodes that still exist: a list may hold the ids of
        # episodes another path deleted without detaching them, and those support nothing.
        edges = await self._run_query(
            tx,
            "MATCH (:Entity {group_id: $g})-[r:RELATES_TO {group_id: $g}]->(:Entity {group_id: $g}) "
            "WHERE any(u IN r.episodes WHERE u IN $uuids) "
            "RETURN r.uuid AS uuid, [u IN r.episodes WHERE NOT u IN $uuids "
            "AND EXISTS { MATCH (:Episodic {group_id: $g, uuid: u}) }] AS remaining",
            g=group,
            uuids=episode_uuids,
        )
        dead = [row["uuid"] for row in edges if not row["remaining"]]
        for row in edges:
            if row["remaining"]:
                await (
                    await tx.run(
                        "MATCH (:Entity {group_id: $g})-[r:RELATES_TO {uuid: $uuid, group_id: $g}]->(:Entity) "
                        "SET r.episodes = $remaining",
                        g=group,
                        uuid=row["uuid"],
                        remaining=row["remaining"],
                    )
                ).consume()
        if dead:
            await (
                await tx.run(
                    "MATCH (:Entity {group_id: $g})-[r:RELATES_TO {group_id: $g}]->(:Entity) "
                    "WHERE r.uuid IN $dead DELETE r",
                    g=group,
                    dead=dead,
                )
            ).consume()
        await (
            await tx.run(
                "MATCH (e:Episodic {group_id: $g}) WHERE e.uuid IN $uuids DETACH DELETE e",
                g=group,
                uuids=episode_uuids,
            )
        ).consume()
        if dead:
            await (
                await tx.run(
                    "MATCH (e:Episodic {group_id: $g}) "
                    "WHERE any(x IN e.entity_edges WHERE x IN $dead) "
                    "SET e.entity_edges = [x IN e.entity_edges WHERE NOT x IN $dead]",
                    g=group,
                    dead=dead,
                )
            ).consume()
        removed_nodes = 0
        if candidates:
            rows = await self._run_query(
                tx,
                "MATCH (n:Entity {group_id: $g}) WHERE n.uuid IN $candidates "
                "AND NOT (n)-[:RELATES_TO]-() AND NOT (:Episodic)-[:MENTIONS]->(n) "
                "WITH collect(n) AS orphans "
                "FOREACH (n IN orphans | DETACH DELETE n) "
                "RETURN size(orphans) AS removed",
                g=group,
                candidates=candidates,
            )
            removed_nodes = int(rows[0]["removed"]) if rows else 0
        return {
            "edges_kept": sum(1 for row in edges if row["remaining"]),
            "edges_removed": len(dead),
            "nodes_removed": removed_nodes,
        }

    def remove_episodes(
        self, group_id: str, episode_uuids: list[str], *, clear_summaries: bool = True
    ) -> dict[str, int]:
        """Support-aware removal of existing episodes by uuid, for the service's delete and correct paths.

        The same single-transaction, ordered-lock detachment as ``remove_by_source``, applied to
        episodes the service's enrichment wrote (identified by uuid, not by a source marker): an
        edge another live episode supports keeps that support, an edge loses its existence only
        with its last support, and an entity is deleted only when no edge or mention references it.
        With ``clear_summaries`` the summary of every entity a removed episode mentions is cleared
        in the same transaction (a summary aggregates every mention and cannot be partly undone).
        Returns ``{"edges_kept", "edges_removed", "nodes_removed", "summaries_cleared"}``.
        """

        uuids = sorted({u for u in episode_uuids if u})
        zero = {"edges_kept": 0, "edges_removed": 0, "nodes_removed": 0, "summaries_cleared": 0}
        if not uuids:
            return dict(zero)
        self._ensure_indices()

        async def work(tx: Any) -> dict[str, int]:
            held: list[str] = []
            present = await self._run_query(
                tx,
                "MATCH (e:Episodic {group_id: $g}) WHERE e.uuid IN $uuids RETURN e.uuid AS uuid",
                g=group_id,
                uuids=uuids,
            )
            found = [row["uuid"] for row in present]
            if not found:
                return dict(zero)
            edges, nodes = await self._footprint(tx, group_id, found)
            await self._lock(
                tx,
                [self._lock_key("1edge", group_id, u) for u in sorted(edges)]
                + [self._lock_key("2node", group_id, u) for u in sorted(nodes)],
                held,
            )
            cleared = 0
            if clear_summaries and nodes:
                rows = await self._run_query(
                    tx,
                    "MATCH (n:Entity {group_id: $g}) WHERE n.uuid IN $nodes SET n.summary = '' "
                    "RETURN count(n) AS cleared",
                    g=group_id,
                    nodes=nodes,
                )
                cleared = int(rows[0]["cleared"]) if rows else 0
            counts = await self._detach_locked(tx, group_id, found)
            await self._release(tx, held)
            return dict(counts, summaries_cleared=max(cleared - counts["nodes_removed"], 0))

        async def transaction() -> dict[str, int]:
            async with self._graphiti.driver.session() as session:
                return await session.execute_write(work)

        return self._run(transaction())

    @staticmethod
    def _to_edge(edge: Any) -> Optional[GraphEdge]:
        attributes = edge.attributes or {}
        if "ns_edge_id" not in attributes:
            return None
        return GraphEdge(
            edge_id=attributes["ns_edge_id"],
            subject_id=attributes["ns_subject"],
            predicate=attributes["ns_predicate"],
            object_id=attributes["ns_object"],
        )

    def resolve_edge(self, ctx: OperationContext, edge_id: str) -> Optional[GraphEdge]:
        self._ensure_indices()

        async def resolve() -> Optional[GraphEdge]:
            from graphiti_core.edges import EntityEdge
            from graphiti_core.errors import EdgeNotFoundError

            try:
                edge = await EntityEdge.get_by_uuid(
                    self._graphiti.driver, _stable_uuid("edge", ctx.tenant_id, edge_id)
                )
            except EdgeNotFoundError:
                return None
            # Graphiti reads by uuid without a group scope; enforce it here.
            if edge.group_id != ctx.tenant_id:
                return None
            return self._to_edge(edge)

        return self._run(resolve())

    def neighbors(
        self, ctx: OperationContext, node_id: str, flt: TenantFilter
    ) -> tuple[GraphEdge, ...]:
        _check_tenant(ctx, flt.tenant_id)
        self._ensure_indices()

        async def collect() -> tuple[GraphEdge, ...]:
            from graphiti_core.edges import EntityEdge

            if flt.authorized_record_ids == ():
                return ()
            edges = await EntityEdge.get_by_node_uuid(
                self._graphiti.driver, _stable_uuid("node", flt.tenant_id, node_id)
            )
            edges = [edge for edge in edges if edge.group_id == flt.tenant_id]
            if flt.authorized_record_ids is not None:
                allowed = set(flt.authorized_record_ids)
                supporting = sorted({u for edge in edges for u in edge.episodes})
                records, _, _ = await self._graphiti.driver.execute_query(
                    "MATCH (e:Episodic {group_id: $group_id}) WHERE e.uuid IN $uuids "
                    "RETURN e.uuid AS uuid, e.source_description AS description",
                    group_id=flt.tenant_id,
                    uuids=supporting,
                    routing_="r",
                )
                authorized_episodes = {
                    record["uuid"]
                    for record in records
                    if json.loads(record["description"]).get("ns_source") in allowed
                }
                edges = [
                    edge for edge in edges if authorized_episodes.intersection(edge.episodes)
                ]
            found: dict[str, GraphEdge] = {}
            for edge in edges:
                converted = self._to_edge(edge)
                if converted is not None:
                    found.setdefault(converted.edge_id, converted)
            return tuple(sorted(found.values(), key=lambda e: e.edge_id))

        return self._run(collect())


class CurrentMemoryEngineAdapter:
    """``MemoryEnginePort`` composing a vector half and a graph half.

    The vector half is any ``VectorStorePort``: the Qdrant adapter (current
    stack), the pgvector adapter, or an ``EngineRouter`` port that switches
    between them by generation.  The composite never inspects which one it has.

    The current engine keeps records in Qdrant and, when an ``edge_extractor``
    is supplied, a graph projection in Graphiti keyed by the record as its
    source.  Without an extractor ingestion is vector-only (the service's real
    extraction needs a model, which this adapter never calls).  A failure in one
    half after the other succeeded is reported as ``PARTIAL_RESULT`` rather than
    hidden.
    """

    def __init__(
        self,
        vector: VectorStorePort,
        graph: Optional[GraphitiGraphAdapter] = None,
        edge_extractor: Optional[Callable[[EngineRecord], tuple[GraphEdge, ...]]] = None,
        implementation_id: str = "current-engine",
    ) -> None:
        if edge_extractor is not None and graph is None:
            raise ValueError("an edge extractor requires a graph adapter")
        self._vector = vector
        self._graph = graph
        self._extractor = edge_extractor
        self._implementation_id = implementation_id

    def manifest(self) -> CapabilityManifest:
        parts = [self._vector.manifest()]
        if self._graph is not None:
            parts.append(self._graph.manifest())
        health = CapabilityHealth.HEALTHY
        for part in parts:
            if any(state.health is not CapabilityHealth.HEALTHY for state in part.operations if state.supported):
                health = CapabilityHealth.UNHEALTHY
        return _manifest(self._implementation_id, EngineAdapterKind.MEMORY_ENGINE, health=health)

    def ingest(self, ctx: OperationContext, record: EngineRecord) -> ProjectionReceipt:
        receipt = self._vector.apply(ctx, record)
        if self._graph is not None and self._extractor is not None:
            try:
                edges = self._extractor(record)
                self._graph.apply_episode(
                    ctx,
                    GraphEpisode(
                        episode_id=record.record_id,
                        revision=record.revision,
                        tenant_id=record.tenant_id,
                        source_record_id=record.record_id,
                        edges=edges,
                    ),
                )
            except PortFailure:
                raise PortFailure(InternalErrorCode.PARTIAL_RESULT, "graph half failed")
            except Exception:
                raise PortFailure(InternalErrorCode.PARTIAL_RESULT, "graph half failed")
        return receipt

    def delete(
        self, ctx: OperationContext, record_id: str, revision: int
    ) -> ProjectionReceipt:
        receipt = self._vector.remove(ctx, record_id, revision)
        if self._graph is not None:
            try:
                self._graph.remove_by_source(ctx, record_id, revision)
            except PortFailure as failure:
                if failure.code is not InternalErrorCode.MISSING_RESOURCE:
                    raise PortFailure(InternalErrorCode.PARTIAL_RESULT, "graph half failed")
            except Exception:
                raise PortFailure(InternalErrorCode.PARTIAL_RESULT, "graph half failed")
        return _receipt(self._implementation_id, record_id, revision, "removed")

    def resolve(self, ctx: OperationContext, record_id: str) -> Optional[EngineRecord]:
        return self._vector.get(ctx, record_id)

    def retrieve(
        self, ctx: OperationContext, query: str, flt: TenantFilter, limit: int
    ) -> tuple[SearchHit, ...]:
        return self._vector.search(ctx, query, flt, limit)


def register_current_adapters(
    registry: AdapterRegistry,
    vector: VectorStorePort,
    graph: Optional[GraphitiGraphAdapter] = None,
    engine: Optional[CurrentMemoryEngineAdapter] = None,
) -> dict[EngineAdapterKind, str]:
    """Register the current-stack adapters; activation stays a separate step."""

    ids = {EngineAdapterKind.VECTOR_STORE: registry.register(vector, EngineAdapterKind.VECTOR_STORE)}
    if graph is not None:
        ids[EngineAdapterKind.GRAPH_STORE] = registry.register(graph, EngineAdapterKind.GRAPH_STORE)
    if engine is not None:
        ids[EngineAdapterKind.MEMORY_ENGINE] = registry.register(
            engine, EngineAdapterKind.MEMORY_ENGINE
        )
    return ids


def hashed_bag_embedder(dims: int = 64) -> Callable[[str], list[float]]:
    """Deterministic token-hash embedder for tests; not a quality embedder."""

    import hashlib

    def embed(text: str) -> list[float]:
        vector = [0.0] * dims
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % dims] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else [1.0] + [0.0] * (dims - 1)

    return embed


__all__ = [
    "ADAPTER_VERSION",
    "CurrentMemoryEngineAdapter",
    "GraphitiGraphAdapter",
    "QdrantVectorAdapter",
    "hashed_bag_embedder",
    "register_current_adapters",
]
