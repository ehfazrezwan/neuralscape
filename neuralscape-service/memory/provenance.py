"""Episode-precise graph provenance cascade.

Security fix: a memory's visibility flip (e.g. shared → private) or a
delete must not leave the memory's contribution to the shared knowledge
graph live. ``_expire_graph_edges_for_memory`` (memory/delete.py) tries to
find those edges by a hybrid search + a literal lowercase substring match
against ``edge.fact`` — but Graphiti's extracted facts are an LLM
PARAPHRASE of the source memory, so the substring almost never matches and
the edges silently stay live in the shared pool.

This module resolves the memory's actual Graphiti *episode* (by uuid, by
deterministic name, or by verbatim content match) and cascades the
expiry from there instead of guessing by text similarity:

- every edge the episode created OR reaffirmed is soft-expired
  (``expired_at = now``) — most-restrictive-wins: an edge is expired even
  when a *different* live episode also asserts it (locked design decision;
  transient recall loss is accepted over a wrongly-live sensitive fact),
- every entity node the episode mentions is either removed (if this
  episode was its only mention) or has its ``summary`` cleared (if other
  live episodes still mention it, so it gets a clean re-summary on the
  next enrichment pass),
- the stale episode node itself is hard-deleted so nothing re-extracts
  from it.

Matches the raw-Cypher-through-``self._run_on_bridge`` dispatch style used
throughout ``memory/graph_admin.py`` and ``extensions/dreaming/graph_patcher.py``:
each step is a small, independently-testable helper issuing exactly one
Cypher statement, wrapped in the same "build the coroutine outside the
try, close() it if the bridge call fails" idiom used by
``write.py::_graph_episode_exists`` — so a mocked/half-initialized bridge
in unit tests degrades to a clean no-op instead of leaking an unawaited
coroutine warning.

Every entry point here is best-effort: a graph failure must never break
the memory write/edit/delete path it was called from. Failures are logged
at WARNING (not swallowed quietly) because a failed cascade is a
potential data-exposure event, not routine noise.
"""

import logging

from datetime import datetime, timezone

from config import settings
from schemas import MemoryVisibility
from memory.groups import _build_group_id

logger = logging.getLogger(__name__)

# Row metadata key: episodes a correction superseded that could not be removed yet (retried later).
SUPERSEDED_EPISODES_KEY = "graph_episode_superseded"
# Row metadata key: set on a surviving sibling whose shared episode was removed, until its
# re-enrichment into a fresh episode has run (the removed episode's uuid; cleared after).
REENRICH_PENDING_KEY = "graph_reenrich_pending"
# The ledger job that re-enriches one memory into a fresh episode (see legacy_jobs).
GRAPH_REENRICH_OPERATION = "graph.reenrich"



def ledger_reenrich_dispatch(producer):
    """Survivor re-enrichment as a durable ledger job (``graph.reenrich``), keyed per memory and
    removed episode so a repeated removal replays the same intent."""

    def dispatch(job: dict):
        return producer.accept_job(
            operation=GRAPH_REENRICH_OPERATION,
            payload={"memory_id": job["memory_id"]},
            user_id=job["user_id"],
            idempotency_key=f"graph-reenrich:{job['memory_id']}:{job['removed_episode']}",
            visibility=job["visibility"],
            project_id=job["project_id"],
            workspace=job["workspace"],
        )

    return dispatch


def queue_reenrich_dispatch(pool, loop):
    """Survivor re-enrichment as a graph-queue job (``process_graph_reenrich``) on ``loop``.

    Called from a worker thread it waits for the enqueue; called on ``loop`` itself it
    schedules the enqueue (the marker on the row shows it until the job ran).
    """
    import asyncio

    def dispatch(job: dict) -> None:
        coro = pool.enqueue_job(
            "process_graph_reenrich",
            job["memory_id"],
            _job_id=f"graph-reenrich:{job['memory_id']}:{job['removed_episode']}",
            _queue_name=settings.graph_queue_name,
        )
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            loop.create_task(coro)
        else:
            asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=30)

    return dispatch


class GraphReadError(RuntimeError):
    """A read-only graph lookup could not run (bridge/Neo4j down).

    Only raised when a caller asks for ``fail_closed=True`` — audit and
    remediation reads must never degrade a broken lookup into
    "episode not found" (which would produce a false-clean result).
    """

class ProvenanceMixin:
    """ProvenanceMixin for MemoryService — episode-precise graph cascade."""

    # ──────────────────────────────────────────────
    # Public entry points
    # ──────────────────────────────────────────────

    def _cascade_expire_episode(
        self,
        group_id: str,
        *,
        episode_name: str | None = None,
        episode_uuid: str | None = None,
        content: str | None = None,
        support_aware: bool | None = None,
        exclude_ids: tuple[str, ...] | None = None,
    ) -> dict:
        """Resolve a memory's Graphiti episode in ``group_id`` and expire
        everything it contributed: edges (soft), entity nodes (removed if
        solely mentioned, else summary-cleared), and the episode itself
        (hard-deleted).

        Resolution order: ``episode_uuid`` (exact; the durable link
        persisted at write time by ``write.py::_persist_graph_episode_ref``
        — primary mechanism) → ``content`` (exact verbatim match against
        the episode's raw body — Graphiti stores episode content
        byte-for-byte, so this works for single-fact writes even without a
        persisted uuid, e.g. rows written before this fix existed) →
        ``episode_name`` (exact; last resort — the single-fact write path
        does NOT name its episode deterministically, so this only helps
        the conversation path, or a single-fact row whose persisted name
        survived but whose content later diverged). Every match is an
        EXACT equality check, never a prefix/fuzzy match, so two episodes
        minted in the same second (or sharing a hash prefix) can't collide.

        Returns zeroed counts with ``resolved: False`` when no episode
        could be found — this is not an error, just "nothing to cascade"
        (e.g. the write predates this fix, or the caller has none of the
        three identifiers). Never raises.

        ``support_aware`` (default: ``GRAPH_SUPPORT_AWARE_REMOVAL_ENABLED``) replaces the
        most-restrictive edge expiry with support-aware removal: an edge another live episode still supports keeps that support and
        stays live; an edge whose last support was this episode is removed; an entity is
        removed only when no edge or mention references it; every entity the episode
        mentions has its summary cleared; all in one transaction.  If that transaction
        fails, the legacy most-restrictive steps run instead (never less restrictive).

        ``exclude_ids`` names the memory rows going away with this removal (the deleted or
        migrated memory, and any batch it is deleted with).  Conversation extraction stamps
        every fact it creates with the same episode, so in support-aware mode the episode may
        still be recorded by surviving sibling rows.  It is removed anyway (a removed memory's
        facts must not stay retrievable; facts other episodes support are kept), and every
        surviving sibling is re-enriched into a fresh episode of its own through the durable
        path (``survivors_reenriched``), so its facts return and the removed memory's do not.
        Until that runs the siblings' facts may be absent (availability, never exposure).  If
        the rows cannot be checked the removal still proceeds and no sibling is known.  With
        ``None`` no row check is made.
        """
        if support_aware is None:
            support_aware = bool(getattr(settings, "graph_support_aware_removal_enabled", False))
        zero = {
            "resolved": False,
            "episode_uuid": None,
            "edges_expired": 0,
            "nodes_removed": 0,
            "summaries_cleared": 0,
        }
        if not group_id or not (episode_uuid or episode_name or content):
            return dict(zero)

        try:
            g = self._get_graphiti()
        except Exception:
            g = None
        if g is None or not self._bridge:
            return dict(zero)

        try:
            resolved_uuid = None
            if episode_uuid:
                resolved_uuid = self._lookup_episode_uuid(group_id, "uuid", episode_uuid)
            # Content before name: the single-fact write path (write.py
            # ::enrich_graph) does NOT name its episode deterministically —
            # MemoryGraph.add() mints `mem0_episode_{now.isoformat()}` when no
            # episode_name is supplied, so a caller can never RECOMPUTE the
            # name, only replay whatever got persisted onto the Qdrant row at
            # write time (write.py::_persist_graph_episode_ref). Content is
            # the more universally trustworthy signal — it's a plain exact
            # match (never a prefix/fuzzy match, so two episodes written in
            # the same session can't collide) and works even for rows
            # written before this fix existed (no persisted name at all).
            # Name is still tried as the last resort for the conversation
            # path, where it IS deterministic (sha256 of raw_text+group_id)
            # but the Qdrant row's `content` is the distilled fact, not the
            # raw transcript the episode actually stored.
            if not resolved_uuid and content:
                resolved_uuid = self._lookup_episode_uuid(group_id, "content", content)
            if not resolved_uuid and episode_name:
                resolved_uuid = self._lookup_episode_uuid(group_id, "name", episode_name)

            if not resolved_uuid:
                logger.warning(
                    "Episode cascade: could not resolve an episode in "
                    "group_id=%r (uuid=%r, name=%r, content_len=%s)",
                    group_id, episode_uuid, episode_name,
                    len(content) if content else 0,
                )
                return dict(zero)

            survivors: list = []
            if support_aware and exclude_ids is not None:
                survivors = self._episode_survivors(resolved_uuid, episode_name, exclude_ids) or []
            if support_aware:
                counts = self._support_aware_episode_removal(group_id, [resolved_uuid])
                if counts is not None:
                    logger.info(
                        "Support-aware removal of episode=%s group_id=%r: edges_removed=%d "
                        "edges_kept=%d nodes_removed=%d summaries_cleared=%d",
                        resolved_uuid, group_id, counts["edges_removed"], counts["edges_kept"],
                        counts["nodes_removed"], counts["summaries_cleared"],
                    )
                    return {
                        "resolved": True,
                        "episode_uuid": resolved_uuid,
                        "mode": "support_aware",
                        "edges_expired": counts["edges_removed"],
                        "edges_kept": counts["edges_kept"],
                        "nodes_removed": counts["nodes_removed"],
                        "summaries_cleared": counts["summaries_cleared"],
                        "survivors_reenriched": self._reenrich_survivors(survivors, resolved_uuid),
                    }

            now_iso = datetime.now(timezone.utc).isoformat()
            edges_expired = self._expire_episode_edges(group_id, resolved_uuid, now_iso)
            node_counts = self._clear_or_remove_episode_entities(resolved_uuid)
            self._delete_episode_node(resolved_uuid)
            if survivors:  # the most-restrictive fallback removed the shared episode too
                self._reenrich_survivors(survivors, resolved_uuid)

            logger.info(
                "Episode cascade expired episode=%s group_id=%r: "
                "edges_expired=%d nodes_removed=%d summaries_cleared=%d",
                resolved_uuid, group_id, edges_expired,
                node_counts["nodes_removed"], node_counts["summaries_cleared"],
            )
            return {
                "resolved": True,
                "episode_uuid": resolved_uuid,
                "edges_expired": edges_expired,
                "nodes_removed": node_counts["nodes_removed"],
                "summaries_cleared": node_counts["summaries_cleared"],
            }
        except Exception as e:
            logger.warning(
                "Episode cascade failed for group_id=%r (non-critical): %s",
                group_id, e, exc_info=True,
            )
            return dict(zero)

    def _cascade_or_fallback_expire(
        self,
        mem: dict,
        *,
        group_id: str | None = None,
        memory_id: str = "",
        removing: tuple[str, ...] = (),
        whole_user: bool = False,
    ) -> dict:
        """Expire a memory's graph provenance: cascade first, legacy
        substring heuristic only as a last resort.

        ``mem`` is the ``{"memory": content, "metadata": meta, ...}`` shape
        already used by :meth:`_expire_graph_edges_for_memory` (memory/delete.py).
        When ``group_id`` is omitted it's derived from ``mem``'s own
        metadata (owner/visibility/project_id/workspace) — the same
        derivation ``_expire_graph_edges_for_memory`` does internally.
        Pass ``group_id`` explicitly for a partition-migration edit, where
        the episode to expire lives in the OLD group, not whatever the
        (already-mutated) metadata would compute.

        Never raises — every failure degrades to a logged no-op, matching
        the non-critical graph-cleanup convention used across the
        delete/edit paths.

        The returned dict always carries ``memory_id`` (echoing the input)
        alongside the cascade's own keys, so a bulk caller can collect
        ``memory_id`` for every unresolved row into a reportable list
        instead of silently absorbing it (audit: the pre-fix substring
        routine's failure mode was exactly this — a per-item try/except
        that logged one debug-level line and moved on, so an operator
        polling task status saw a clean ``completed`` with no indication
        entire rows were never touched in the graph).

        ``removing`` lists the other rows of a batch delete that cascades before it deletes the
        rows, so siblings of one conversation do not keep each other's shared episode alive;
        ``whole_user`` (every memory of the user goes) skips the row check altogether.  In
        support-aware mode episodes an earlier correction could not remove
        (``graph_episode_superseded``) are retried here as well.
        """
        metadata = mem.get("metadata", {}) or {}
        if isinstance(metadata.get("metadata"), dict):
            metadata = metadata["metadata"]

        if group_id is None:
            owner = metadata.get("owner_user_id") or mem.get("user_id", "")
            visibility = metadata.get("visibility") or MemoryVisibility.PRIVATE.value
            group_id = _build_group_id(
                visibility, owner, metadata.get("project_id"), metadata.get("workspace")
            )

        exclude_ids = None
        if memory_id and not whole_user:
            exclude_ids = tuple(dict.fromkeys(i for i in (memory_id, *removing) if i))
        result = self._cascade_expire_episode(
            group_id,
            episode_uuid=metadata.get("graph_episode_uuid"),
            episode_name=metadata.get("graph_episode_name"),
            content=mem.get("memory", ""),
            exclude_ids=exclude_ids,
        )
        superseded = metadata.get(SUPERSEDED_EPISODES_KEY)
        if ((memory_id or whole_user) and isinstance(superseded, list) and superseded
                and getattr(settings, "graph_support_aware_removal_enabled", False)):
            left = self._retire_episodes(exclude_ids, [u for u in superseded if isinstance(u, str)])
            if left:
                logger.warning(
                    "Superseded graph episodes of memory=%s could not be removed: %s", memory_id, left,
                )
        result = dict(result, memory_id=memory_id or None)
        if not result.get("resolved"):
            logger.warning(
                "Episode cascade unresolved for memory=%s group_id=%r — "
                "falling back to substring edge-expiry heuristic (may leave "
                "stale/paraphrased facts live in the graph — cannot verify "
                "cleanup succeeded)",
                memory_id or "?", group_id,
            )
            try:
                self._expire_graph_edges_for_memory(mem)
            except Exception as e:
                logger.warning(
                    "Fallback graph edge expiration failed for memory=%s "
                    "(non-critical): %s",
                    memory_id or "?", e,
                )
        return result

    def _support_aware_episode_removal(self, group_id: str, episode_uuids: list[str]) -> dict | None:
        """Remove episodes support-aware through the current-stack graph adapter; ``None`` on failure."""

        try:
            from engine_adapters_current import GraphitiGraphAdapter

            adapter = GraphitiGraphAdapter(
                self._graphiti, lambda coro: self._run_on_bridge(coro, timeout=30.0)
            )
            return adapter.remove_episodes(group_id, episode_uuids)
        except Exception:
            logger.warning(
                "Support-aware graph removal failed for group_id=%r episodes=%s; "
                "falling back to the most-restrictive cascade",
                group_id, episode_uuids, exc_info=True,
            )
            return None

    def _episode_survivors(
        self, episode_uuid: str, episode_name: str | None, exclude_ids: tuple[str, ...]
    ) -> list[tuple[str, dict]] | None:
        """The memory rows outside ``exclude_ids`` that record this episode (by uuid or name).

        Pages through the rows that reference the episode (the facts of one conversation, so
        few; both reference fields carry a keyword payload index) and compares ids here, so a
        large batch never becomes a large filter.  Returns ``(id, payload)`` pairs, or ``None``
        when the row store cannot be asked.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._ensure_filter_indexes()
        should = [FieldCondition(key="metadata.graph_episode_uuid", match=MatchValue(value=episode_uuid))]
        if episode_name:
            should.append(FieldCondition(key="metadata.graph_episode_name", match=MatchValue(value=episode_name)))
        excluded = set(exclude_ids)
        survivors: list[tuple[str, dict]] = []
        offset = None
        try:
            while True:
                points, offset = self._memory.vector_store.client.scroll(
                    collection_name=settings.qdrant_collection,
                    scroll_filter=Filter(should=should),
                    limit=64,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                survivors += [(str(p.id), p.payload or {}) for p in points if str(p.id) not in excluded]
                if offset is None or not points:
                    return survivors
        except Exception:
            logger.warning("Episode reference check failed for episode=%s", episode_uuid, exc_info=True)
            return None

    def _reenrich_survivors(self, survivors: list[tuple[str, dict]], removed_episode: str) -> int:
        """Re-enrich each surviving sibling of a removed shared episode into a fresh episode.

        The row's reference to the removed episode is cleared and ``graph_reenrich_pending``
        set (it stays until the re-enrichment ran, so a lost dispatch is visible), then one
        job per row goes through the durable path installed for this process (a ledger job,
        or the graph enrichment queue); with none installed the row is enriched inline.  A
        tombstoned row is not re-enriched.  Returns how many rows were scheduled.
        """
        scheduled = 0
        for memory_id, payload in survivors:
            meta = payload.get("metadata") or {}
            if isinstance(meta.get("metadata"), dict):
                meta = meta["metadata"]
            if meta.get("dream_tombstoned"):
                continue
            job = {
                "memory_id": memory_id,
                "content": payload.get("data", ""),
                "user_id": meta.get("owner_user_id") or payload.get("user_id", ""),
                "project_id": meta.get("project_id"),
                "visibility": meta.get("visibility") or MemoryVisibility.PRIVATE.value,
                "workspace": meta.get("workspace"),
                "removed_episode": removed_episode,
            }
            try:
                self._memory.vector_store.client.set_payload(
                    collection_name=settings.qdrant_collection,
                    payload={"graph_episode_uuid": None, "graph_episode_name": None,
                             REENRICH_PENDING_KEY: removed_episode},
                    points=[memory_id],
                    key="metadata",
                )
                dispatch = getattr(self, "_graph_reenrich_dispatch", None)
                if dispatch is not None:
                    dispatch(job)
                else:
                    self.reenrich_memory_graph(memory_id)
                scheduled += 1
            except Exception:
                logger.warning(
                    "Re-enrichment of memory=%s after removing its shared episode=%s could not be "
                    "scheduled (graph_reenrich_pending stays on the row)",
                    memory_id, removed_episode, exc_info=True,
                )
        if scheduled:
            logger.info("Removed shared episode=%s; re-enriching %d surviving memories", removed_episode, scheduled)
        return scheduled

    def set_graph_reenrich_dispatch(self, dispatch) -> None:
        """Install the durable path for survivor re-enrichment (``dispatch(job)``; ``None``: inline)."""

        self._graph_reenrich_dispatch = dispatch

    def reenrich_memory_graph(self, memory_id: str) -> dict:
        """Re-enrich one memory into a fresh episode (the ``graph.reenrich`` job).

        A memory deleted meanwhile is skipped; an enrichment that does not complete raises, so
        the job is retried.
        """
        row = self._memory.get(memory_id)
        if not row:
            return {"skipped": True, "reason": "memory_missing"}
        meta = row.get("metadata") or {}
        if isinstance(meta.get("metadata"), dict):
            meta = meta["metadata"]
        if meta.get("dream_tombstoned"):
            return {"skipped": True, "reason": "memory_tombstoned"}
        enriched = self.enrich_graph(
            content=row.get("memory", ""),
            user_id=meta.get("owner_user_id") or row.get("user_id", ""),
            project_id=meta.get("project_id"),
            visibility=meta.get("visibility") or MemoryVisibility.PRIVATE.value,
            memory_id=memory_id,
            occurred_at=meta.get("occurred_at"),
            workspace=meta.get("workspace"),
        )
        if not enriched:
            raise RuntimeError(f"graph re-enrichment of {memory_id} did not complete")
        return {"memory_ids": [memory_id]}

    def _episode_groups(self, episode_uuids: list[str]) -> dict[str, tuple[str, str | None]] | None:
        """The group and name of each existing episode (missing ones are absent); ``None`` on failure."""

        async def _run():
            async with self._graphiti.driver.session() as session:
                result = await session.run(
                    "MATCH (e:Episodic) WHERE e.uuid IN $uuids "
                    "RETURN e.uuid AS uuid, e.group_id AS group_id, e.name AS name",
                    uuids=episode_uuids,
                )
                return await result.data()

        coro = _run()
        try:
            records = self._run_on_bridge(coro, timeout=10.0) or []
        except Exception:
            coro.close()
            logger.warning("Episode group lookup failed for %s", episode_uuids, exc_info=True)
            return None
        return {r["uuid"]: (r["group_id"], r.get("name")) for r in records}

    def _retire_episodes(self, exclude_ids: tuple[str, ...] | None, episode_uuids: list[str]) -> list[str]:
        """Support-aware removal of episodes the rows in ``exclude_ids`` no longer stand on.

        Each episode is removed in the group it lives in (a correction may have moved the memory
        to another group); an episode already gone is done.  An episode other memories still
        record is removed as well and those memories are re-enriched into fresh episodes, as on
        delete; if the rows cannot be checked the episode is still removed (fail closed) and no
        sibling is known (``exclude_ids`` ``None``: no row check).  Returns the episodes that
        could not be removed, for a later retry.
        """
        candidates = list(dict.fromkeys(u for u in episode_uuids if u))
        if not candidates:
            return []
        groups = self._episode_groups(candidates)
        if groups is None:
            return candidates
        left = []
        for episode in candidates:
            if episode not in groups:
                continue
            group, name = groups[episode]
            survivors = [] if exclude_ids is None else (self._episode_survivors(episode, name, exclude_ids) or [])
            if self._support_aware_episode_removal(group, [episode]) is None:
                left.append(episode)
                continue
            self._reenrich_survivors(survivors, episode)
        return left

    def _persist_graph_episode_ref(
        self,
        memory_id: str,
        episode_uuid: str | None,
        episode_name: str | None,
    ) -> None:
        """Stamp the resolved Graphiti episode identity onto a Qdrant row.

        Durable two-way link (memory → episode) so a later visibility flip
        or delete can find the EXACT episode instead of relying on a
        verbatim content match (which only works for single-fact writes —
        a conversation episode's body is the raw transcript, not the
        extracted fact stored in Qdrant) or the lossy substring-search
        fallback.

        Additive nested-key merge (mirrors ``write.py::_bump_times_derived``)
        so a concurrent metadata patch can't be clobbered by a stale full
        rewrite. Best-effort: never blocks or fails the write path this is
        called from, and never raises.
        """
        if not memory_id or not (episode_uuid or episode_name):
            return
        try:
            patch: dict = {}
            if episode_uuid:
                patch["graph_episode_uuid"] = episode_uuid
            if episode_name:
                patch["graph_episode_name"] = episode_name
            client = self._memory.vector_store.client
            client.set_payload(
                collection_name=settings.qdrant_collection,
                payload=patch,
                points=[memory_id],
                key="metadata",
            )
        except Exception as e:
            logger.warning(
                f"Persisting graph episode ref failed for {memory_id} (non-fatal): {e}"
            )

    # ──────────────────────────────────────────────
    # Single-Cypher-statement helpers (each one _run_on_bridge round trip)
    # ──────────────────────────────────────────────

    def _lookup_episode_uuid(
        self, group_id: str, field: str, value: str, *, fail_closed: bool = False
    ) -> str | None:
        """One cheap Cypher lookup: find an episode's uuid in ``group_id``
        by ``field`` ('uuid' | 'name' | 'content') == ``value``.

        Default fail-OPEN like ``write.py::_graph_episode_exists``: any error
        (bridge down, Neo4j hiccup, mocked/half-initialized bridge in
        tests) returns None rather than raising, so a broken lookup
        degrades to "episode not found" — the cascade then reports the row
        as unresolved / ``migration_incomplete`` (visible, not silent).

        ``fail_closed=True`` (audit/remediation): the same error RAISES
        :class:`GraphReadError` instead, because there "episode not found"
        would read as "no leak" — a false-clean result.
        """
        if field == "uuid":
            cypher = (
                "MATCH (e:Episodic {group_id: $group_id, uuid: $value}) "
                "RETURN e.uuid AS uuid LIMIT 1"
            )
        elif field == "name":
            cypher = (
                "MATCH (e:Episodic {group_id: $group_id, name: $value}) "
                "RETURN e.uuid AS uuid LIMIT 1"
            )
        elif field == "content":
            # Verbatim match — Graphiti stores episode content byte-for-byte.
            # Multiple episodes could theoretically share identical content;
            # pick the most recent one.
            cypher = (
                "MATCH (e:Episodic {group_id: $group_id, content: $value}) "
                "RETURN e.uuid AS uuid ORDER BY e.created_at DESC LIMIT 1"
            )
        else:
            return None

        async def _run():
            async with self._graphiti.driver.session() as session:
                result = await session.run(cypher, group_id=group_id, value=value)
                return await result.data()

        coro = _run()
        try:
            records = self._run_on_bridge(coro, timeout=10.0) or []
        except Exception as e:
            coro.close()
            if fail_closed:
                logger.error(
                    "episode lookup by %s failed for group_id=%r (failing CLOSED)", field, group_id, exc_info=True,
                )
                raise GraphReadError(f"episode lookup by {field} failed in {group_id}: {e}") from e
            logger.debug(
                "episode lookup by %s failed for group_id=%r (fail-open)", field, group_id, exc_info=True,
            )
            return None
        return records[0]["uuid"] if records else None

    def _expire_episode_edges(self, group_id: str, episode_uuid: str, now_iso: str) -> int:
        """Soft-expire every edge this episode created OR reaffirmed.

        Belt-and-braces in ONE Cypher statement: matches by the episode's
        ``entity_edges`` list property AND by edges whose own ``episodes``
        array contains this episode's uuid (catches edges that reference
        the episode but were somehow missing from ``entity_edges``).

        MOST-RESTRICTIVE-WINS (locked design decision): an edge is expired
        even when it has other, still-live parent episodes — mixed
        parentage does not protect it. ``expired_at IS NULL`` in the WHERE
        clause makes this idempotent: a second run finds nothing left to
        expire.

        F6: ``$now`` arrives as an ISO string (``now_iso``); ``datetime()``
        parses it into Neo4j's native temporal type on the way in, matching
        what Graphiti itself stores for ``expired_at``/``invalid_at``
        elsewhere. Storing the raw string instead would desync this write
        from every other bi-temporal stamp in the graph — a downstream
        reader that expects a temporal value (e.g. calling ``.isoformat()``
        on the deserialized property, as the graph listing endpoints do)
        would break on exactly the edges this cascade just touched.
        """
        cypher = """
        MATCH (ep:Episodic {uuid: $episode_uuid})
        WITH ep, coalesce(ep.entity_edges, []) AS listed_edge_uuids
        MATCH ()-[r:RELATES_TO {group_id: $group_id}]->()
        WHERE r.expired_at IS NULL
          AND (r.uuid IN listed_edge_uuids OR $episode_uuid IN coalesce(r.episodes, []))
        SET r.expired_at = datetime($now)
        RETURN count(r) AS edges_expired
        """

        async def _run():
            async with self._graphiti.driver.session() as session:
                result = await session.run(
                    cypher, episode_uuid=episode_uuid, group_id=group_id, now=now_iso
                )
                return await result.data()

        coro = _run()
        try:
            records = self._run_on_bridge(coro, timeout=15.0) or []
        except Exception:
            coro.close()
            logger.warning(
                "Edge expiry cascade failed for episode=%s group_id=%r (non-critical)",
                episode_uuid, group_id, exc_info=True,
            )
            return 0
        return int(records[0]["edges_expired"]) if records else 0

    def _clear_or_remove_episode_entities(self, episode_uuid: str) -> dict:
        """For every entity node this episode mentions: UNCONDITIONALLY
        clear its ``summary`` — then, additionally, remove the node
        entirely if this episode is its ONLY mention.

        Summary-clearing is deliberately NOT gated behind the mention-count
        check (orchestrator amendment, live-repro-confirmed): a node's
        ``summary`` is an LLM-synthesized AGGREGATE over every episode that
        mentions it, and there is no safe way to subtract just one
        episode's contribution from that aggregate. A node mentioned by
        several OTHER live episodes still had this episode's content
        folded into its summary — leaving that summary untouched (the
        old "only clear if sole-mentioned" framing) can leave sensitive
        text readable through a node that never looked like the "owner"
        of the leak. So every mentioned node's summary is cleared, full
        stop, and left to regenerate cleanly on the next enrichment pass.
        Nodes mentioned ONLY by this episode go further and are removed
        outright (their clear-then-delete is a harmless, cheap no-op
        ordering, not a correctness dependency).

        Mirrors the ``episode_count == 1`` check in
        ``graphiti_core.graphiti.Graphiti.remove_episode`` (read as a
        pattern reference only — that method hard-deletes edges by sole
        first-parent, which is NOT the semantics wanted here).

        Returns ``{"nodes_removed": int, "summaries_cleared": int}`` where
        the two counts are disjoint: ``nodes_removed`` = sole-mentioned
        nodes (gone entirely), ``summaries_cleared`` = every OTHER
        mentioned node (survives, summary now empty). Idempotent: once
        run, the episode's MENTIONS edges are gone (the episode itself
        gets hard-deleted next), so a second cascade attempt on the same
        episode finds nothing to mention.
        """
        cypher = """
        MATCH (ep:Episodic {uuid: $episode_uuid})-[:MENTIONS]->(n:Entity)
        WITH DISTINCT n
        SET n.summary = ''
        WITH n
        OPTIONAL MATCH (other:Episodic)-[:MENTIONS]->(n)
        WITH n, count(other) AS mention_count
        FOREACH (_ IN CASE WHEN mention_count <= 1 THEN [1] ELSE [] END | DETACH DELETE n)
        RETURN
          sum(CASE WHEN mention_count <= 1 THEN 1 ELSE 0 END) AS nodes_removed,
          sum(CASE WHEN mention_count > 1 THEN 1 ELSE 0 END) AS summaries_cleared
        """

        async def _run():
            async with self._graphiti.driver.session() as session:
                result = await session.run(cypher, episode_uuid=episode_uuid)
                return await result.data()

        coro = _run()
        try:
            records = self._run_on_bridge(coro, timeout=15.0) or []
        except Exception:
            coro.close()
            logger.warning(
                "Entity node cascade failed for episode=%s (non-critical)",
                episode_uuid, exc_info=True,
            )
            return {"nodes_removed": 0, "summaries_cleared": 0}
        if not records:
            return {"nodes_removed": 0, "summaries_cleared": 0}
        rec = records[0]
        return {
            "nodes_removed": int(rec.get("nodes_removed") or 0),
            "summaries_cleared": int(rec.get("summaries_cleared") or 0),
        }

    def _delete_episode_node(self, episode_uuid: str) -> None:
        """Hard-delete the stale episode node (and its relationships) so
        nothing re-extracts from it. Idempotent — deleting an
        already-deleted uuid matches zero nodes and no-ops.
        """
        cypher = "MATCH (ep:Episodic {uuid: $episode_uuid}) DETACH DELETE ep"

        async def _run():
            async with self._graphiti.driver.session() as session:
                await session.run(cypher, episode_uuid=episode_uuid)

        coro = _run()
        try:
            self._run_on_bridge(coro, timeout=15.0)
        except Exception:
            coro.close()
            logger.warning(
                "Episode node deletion failed for episode=%s (non-critical)",
                episode_uuid, exc_info=True,
            )
