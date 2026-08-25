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

            now_iso = datetime.now(timezone.utc).isoformat()
            edges_expired = self._expire_episode_edges(group_id, resolved_uuid, now_iso)
            node_counts = self._clear_or_remove_episode_entities(resolved_uuid)
            self._delete_episode_node(resolved_uuid)

            logger.warning(
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
    # Single-Cypher-statement helpers (each one _run_on_bridge round trip)
    # ──────────────────────────────────────────────

    def _lookup_episode_uuid(self, group_id: str, field: str, value: str) -> str | None:
        """One cheap Cypher lookup: find an episode's uuid in ``group_id``
        by ``field`` ('uuid' | 'name' | 'content') == ``value``.

        Fail-OPEN like ``write.py::_graph_episode_exists``: any error
        (bridge down, Neo4j hiccup, mocked/half-initialized bridge in
        tests) returns None rather than raising, so a broken lookup
        degrades to "episode not found" — the caller's next resolution
        attempt (or the substring fallback) still runs.
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
        except Exception:
            coro.close()
            logger.debug(
                "episode lookup by %s failed for group_id=%r (fail-open)",
                field, group_id, exc_info=True,
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
        """
        cypher = """
        MATCH (ep:Episodic {uuid: $episode_uuid})
        WITH ep, coalesce(ep.entity_edges, []) AS listed_edge_uuids
        MATCH ()-[r:RELATES_TO {group_id: $group_id}]->()
        WHERE r.expired_at IS NULL
          AND (r.uuid IN listed_edge_uuids OR $episode_uuid IN coalesce(r.episodes, []))
        SET r.expired_at = $now
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
