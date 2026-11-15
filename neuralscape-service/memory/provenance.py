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
        """
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
            found = None
            if episode_uuid:
                found = self._lookup_episode(group_id, "uuid", episode_uuid)
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
            if not found and content:
                found = self._lookup_episode(group_id, "content", content)
            if not found and episode_name:
                found = self._lookup_episode(group_id, "name", episode_name)

            if not found:
                logger.warning(
                    "Episode cascade: could not resolve an episode in "
                    "group_id=%r (uuid=%r, name=%r, content_len=%s)",
                    group_id, episode_uuid, episode_name,
                    len(content) if content else 0,
                )
                return dict(zero)

            resolved_uuid = found["uuid"]
            resolved_name = found.get("name")
            now_iso = datetime.now(timezone.utc).isoformat()
            edges_expired = self._expire_episode_edges(group_id, resolved_uuid, now_iso)
            node_counts = self._clear_or_remove_episode_entities(resolved_uuid)
            self._delete_episode_node(resolved_uuid)

            logger.info(
                "Episode cascade expired episode=%s group_id=%r: "
                "edges_expired=%d nodes_removed=%d summaries_cleared=%d",
                resolved_uuid, group_id, edges_expired,
                node_counts["nodes_removed"], node_counts["summaries_cleared"],
            )
            return {
                "resolved": True,
                "episode_uuid": resolved_uuid,
                "episode_name": resolved_name,
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

        result = self._cascade_expire_episode(
            group_id,
            episode_uuid=metadata.get("graph_episode_uuid"),
            episode_name=metadata.get("graph_episode_name"),
            content=mem.get("memory", ""),
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
    # One episode, N rows: cascade once, repair the siblings
    # ──────────────────────────────────────────────
    #
    # A conversation write produces ONE Graphiti episode for N extracted
    # facts and stamps the same ``graph_episode_uuid``/``graph_episode_name``
    # on all N Qdrant rows. The episode body contains every fact verbatim and
    # episode recall serves it, so when ANY of those rows is deleted (or
    # migrated to another partition) the whole episode must go
    # (most-restrictive-wins, unchanged). What must not happen is the other
    # N-1 rows silently losing their graph evidence while their vectors live
    # on with a dangling ref. ``_cascade_rows_once`` therefore (a) cascades
    # each distinct episode ONCE per call and (b) repairs every live sibling:
    # clears its stale ref and builds a single-fact re-ingest job into its own
    # group — the same job shape ``patch_memory`` returns as ``graph_job``,
    # consumed by ``task_manager.enqueue_graph_enrichment``.

    @staticmethod
    def _row_meta(mem: dict) -> dict:
        meta = mem.get("metadata", {}) or {}
        if isinstance(meta.get("metadata"), dict):
            meta = meta["metadata"]
        return meta

    def _cascade_rows_once(
        self,
        rows: list[dict],
        *,
        deleted_ids=(),
        skip_job_ids=(),
    ) -> dict:
        """Cascade a batch of rows' graph provenance, one cascade per episode.

        ``rows``: ``[{"id": str, "mem": {"memory", "metadata", ...},
        "group_id": str | None}]`` — ``mem``/``group_id`` are exactly what
        :meth:`_cascade_or_fallback_expire` takes (``group_id`` only when the
        caller needs to pin the OLD partition, e.g. a migration).

        ``deleted_ids``: rows that no longer exist (or are being deleted in
        this call). They are never repaired or re-enriched.
        ``skip_job_ids``: live rows the caller re-enriches itself (the row a
        partition migration moved): their stale ref is cleared but no sibling
        job is built for them.

        Rows sharing an episode ref with an already-cascaded row reuse that
        cascade's result instead of re-resolving a node that is already gone
        (which used to read as "unresolved" and fire the lossy substring
        fallback). Unresolved rows are NOT merged — the substring fallback is
        content-specific, so each runs its own.

        Never raises. Returns::

            {"results": {id: cascade_result}, "unresolved_ids": [...],
             "graph_jobs": [job, ...],        # sibling re-enrichment jobs
             "sibling_unrepaired_ids": [...], # siblings that could not be
                                              # repaired (reported, not dropped)
             "sibling_repair_failed": bool,   # the sibling lookup itself failed
             "episodes_cascaded": int}
        """
        deleted = {str(i) for i in deleted_ids}
        skip_jobs = {str(i) for i in skip_job_ids}
        results: dict[str, dict] = {}
        unresolved: list[str] = []
        covered: dict[tuple, dict] = {}   # (group, ref) -> resolved cascade result
        episodes: list[dict] = []         # distinct cascaded episodes, for repair

        for row in rows:
            mid = str(row.get("id") or "")
            mem = row.get("mem") or {}
            pinned_group = row.get("group_id")
            meta = self._row_meta(mem)
            group = pinned_group or _build_group_id(
                meta.get("visibility") or MemoryVisibility.PRIVATE.value,
                meta.get("owner_user_id") or mem.get("user_id", ""),
                meta.get("project_id"),
                meta.get("workspace"),
            )
            row_uuid = meta.get("graph_episode_uuid")
            row_name = meta.get("graph_episode_name")
            hit = None
            for ref in (row_uuid, row_name):
                if ref and (group, ref) in covered:
                    hit = covered[(group, ref)]
                    break
            if hit is not None:
                results[mid] = dict(hit, memory_id=mid or None)
                continue

            try:
                if pinned_group:
                    res = self._cascade_or_fallback_expire(mem, group_id=pinned_group, memory_id=mid)
                else:
                    res = self._cascade_or_fallback_expire(mem, memory_id=mid)
            except Exception as e:
                logger.warning(f"Graph cascade failed for {mid} (non-critical): {e}")
                res = {"resolved": False, "memory_id": mid or None}
            results[mid] = res
            if not res.get("resolved"):
                unresolved.append(mid)
                continue
            refs = {r for r in (row_uuid, row_name, res.get("episode_uuid"), res.get("episode_name")) if r}
            for ref in refs:
                covered[(group, ref)] = res
            episodes.append({
                "owner": meta.get("owner_user_id") or mem.get("user_id", ""),
                "uuids": {r for r in (row_uuid, res.get("episode_uuid")) if r},
                "names": {r for r in (row_name, res.get("episode_name")) if r},
            })

        # Merge episodes per owner so the sibling scroll runs once per owner.
        by_owner: dict[str, dict] = {}
        for ep in episodes:
            agg = by_owner.setdefault(ep["owner"], {"uuids": set(), "names": set()})
            agg["uuids"] |= ep["uuids"]
            agg["names"] |= ep["names"]
        jobs: list[dict] = []
        unrepaired: list[str] = []
        repair_failed = False
        for owner, agg in by_owner.items():
            if not owner or not (agg["uuids"] or agg["names"]):
                continue
            repaired = self._repair_episode_siblings(
                owner, agg["uuids"], agg["names"], deleted_ids=deleted, skip_job_ids=skip_jobs,
            )
            jobs.extend(repaired["graph_jobs"])
            unrepaired.extend(repaired["unrepaired_ids"])
            repair_failed = repair_failed or bool(repaired.get("lookup_failed"))
        return {
            "results": results,
            "unresolved_ids": unresolved,
            "graph_jobs": jobs,
            "sibling_unrepaired_ids": unrepaired,
            # True when the sibling lookup itself failed: siblings may exist
            # with dangling refs and no re-enrichment job (reported, not hidden).
            "sibling_repair_failed": repair_failed,
            "episodes_cascaded": len(episodes),
        }

    def _repair_episode_siblings(
        self,
        owner: str,
        uuids: set[str],
        names: set[str],
        *,
        deleted_ids: set[str],
        skip_job_ids: set[str],
    ) -> dict:
        """Clear the dangling episode ref on every live row of ``owner``
        pointing at a just-cascaded episode and build its re-enrichment job.

        Scoped by ``user_id`` (a cross-user row can never be touched) and
        bounded to the rows that carry one of the cascaded refs. The ref is
        cleared with an atomic nested-key ``set_payload(key="metadata")`` (no
        read-modify-write of the whole payload). Any failure is returned in
        ``unrepaired_ids`` — a sibling is never silently dropped. Never raises.
        """
        out: dict = {"graph_jobs": [], "unrepaired_ids": []}
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

            self._ensure_filter_indexes()
            client = self._memory.vector_store.client
            collection = settings.qdrant_collection
            should = []
            for prefix in ("metadata.", "metadata.metadata."):
                if uuids:
                    should.append(FieldCondition(
                        key=prefix + "graph_episode_uuid", match=MatchAny(any=sorted(uuids))))
                if names:
                    should.append(FieldCondition(
                        key=prefix + "graph_episode_name", match=MatchAny(any=sorted(names))))
            scroll_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=owner))],
                should=should,
            )
            siblings: list[tuple[str, dict, dict, bool]] = []
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=collection,
                    scroll_filter=scroll_filter,
                    limit=200,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for pt in points:
                    sid = str(pt.id)
                    payload = pt.payload or {}
                    if sid in deleted_ids or payload.get("user_id") != owner:
                        continue
                    raw = payload.get("metadata", {}) or {}
                    wrapped = isinstance(raw.get("metadata"), dict)
                    meta = raw["metadata"] if wrapped else raw
                    # Re-check in Python: only rows that really point at a
                    # cascaded episode (the filter is a candidate net).
                    if meta.get("graph_episode_uuid") not in uuids and (
                        meta.get("graph_episode_name") not in names
                    ):
                        continue
                    siblings.append((sid, payload, meta, wrapped))
                if offset is None:
                    break
        except Exception as e:
            logger.warning(
                "Sibling lookup for cascaded episode failed (owner=%r): %s", owner, e,
                exc_info=True,
            )
            out["lookup_failed"] = True
            return out

        cleared_payload = {"graph_episode_uuid": None, "graph_episode_name": None}
        failed: set[str] = set()
        for want_wrapped in (False, True):
            ids = [sid for sid, _, _, w in siblings if w == want_wrapped]
            if not ids:
                continue
            key = "metadata.metadata" if want_wrapped else "metadata"
            try:
                client.set_payload(
                    collection_name=collection, payload=cleared_payload, points=ids, key=key,
                )
                continue
            except Exception as e:
                logger.warning(
                    "Batched clear of stale episode refs failed for %d rows (%s); retrying per row",
                    len(ids), e,
                )
            # One bad point must not strand the rest: retry each on its own.
            for sid in ids:
                try:
                    client.set_payload(
                        collection_name=collection, payload=cleared_payload, points=[sid], key=key,
                    )
                except Exception as e:
                    logger.warning("Clearing stale episode ref failed for %s: %s", sid, e)
                    failed.add(sid)
        if failed:
            out["unrepaired_ids"].extend(sorted(failed))
            siblings = [t for t in siblings if t[0] not in failed]

        for sid, payload, meta, _ in siblings:
            if sid in skip_job_ids or meta.get("dream_tombstoned"):
                continue  # tombstoned rows are not live: ref cleared, no re-ingest
            content = payload.get("data") or ""
            workspace = meta.get("workspace")
            if not content or (workspace and workspace != "memory"):
                # The enrichment job shape carries no workspace, so a
                # workspace-partitioned row would land in the wrong group;
                # empty content has nothing to ingest. Report, don't drop.
                out["unrepaired_ids"].append(sid)
                continue
            out["graph_jobs"].append({
                "memory_id": sid,
                "content": content,
                "user_id": meta.get("owner_user_id") or owner,
                "project_id": meta.get("project_id"),
                "visibility": meta.get("visibility") or MemoryVisibility.PRIVATE.value,
                "source_ref": meta.get("source_ref"),
            })
        return out

    # ──────────────────────────────────────────────
    # Single-Cypher-statement helpers (each one _run_on_bridge round trip)
    # ──────────────────────────────────────────────

    def _lookup_episode_uuid(
        self, group_id: str, field: str, value: str, *, fail_closed: bool = False
    ) -> str | None:
        """uuid-only view of :meth:`_lookup_episode` (kept for the audit /
        remediation callers, which only need the uuid)."""
        found = self._lookup_episode(group_id, field, value, fail_closed=fail_closed)
        return found["uuid"] if found else None

    def _lookup_episode(
        self, group_id: str, field: str, value: str, *, fail_closed: bool = False
    ) -> dict | None:
        """One cheap Cypher lookup: find an episode (``{"uuid", "name"}``) in
        ``group_id`` by ``field`` ('uuid' | 'name' | 'content') == ``value``.
        The name rides along in the same round trip so the cascade can repair
        sibling rows that reference the episode by name only.

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
                "RETURN e.uuid AS uuid, e.name AS name LIMIT 1"
            )
        elif field == "name":
            cypher = (
                "MATCH (e:Episodic {group_id: $group_id, name: $value}) "
                "RETURN e.uuid AS uuid, e.name AS name LIMIT 1"
            )
        elif field == "content":
            # Verbatim match — Graphiti stores episode content byte-for-byte.
            # Multiple episodes could theoretically share identical content;
            # pick the most recent one.
            cypher = (
                "MATCH (e:Episodic {group_id: $group_id, content: $value}) "
                "RETURN e.uuid AS uuid, e.name AS name ORDER BY e.created_at DESC LIMIT 1"
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
        if not records:
            return None
        return {"uuid": records[0]["uuid"], "name": records[0].get("name")}

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
