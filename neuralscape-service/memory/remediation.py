"""Backward-looking remediation for the private-graph-leakage bug.

Security fix (part 2 of 2): a memory written while ``visibility=shared`` was
ingested into the Graphiti group ``shared`` (or ``shared--project--{pid}``).
When the owner later flipped it to ``private``, the vector row moved but the
GRAPH artifacts stayed behind in the shared group, readable by every other
seat. ``memory/provenance.py`` (``_cascade_expire_episode``) fixed the
GOING-FORWARD path — every visibility flip / delete from here on cascades
correctly. This module is the BACKWARD-looking half: find and clean the
already-leaked artifacts for rows written before that fix existed, and let
an operator prove afterwards that none remain.

Three leak surfaces (confirmed against a live pre-fix instance — a leaked
memory can hit any subset of these, never all three reliably):

  (a) a still-live ``RELATES_TO`` edge whose ``fact`` text (or provenance)
      traces back to the private memory,
  (b) an ``Entity.summary`` (LLM-aggregated, can restate figures from the
      private memory) on a node the leaked episode mentions,
  (c) the raw ``Episodic.content`` itself, byte-identical to the Qdrant
      memory text on the single-fact write path — exposed via episode
      listing / fulltext search.

Resolution mechanism: exactly the one ``memory/provenance.py`` already
proved out — resolve the memory's Graphiti episode by persisted uuid or by
verbatim content match, then act only on what that specific episode
touches. Never a text-similarity guess, never an unfiltered group wipe
(issue #176's known bug is exactly that: an unfiltered private-wipe crashed
the graph worker). Every write in this module is scoped to one resolved
episode uuid.

Matches the raw-Cypher-through-``self._run_on_bridge`` dispatch style used
throughout ``memory/graph_admin.py`` and ``memory/provenance.py``: each
step is a small, independently-testable helper issuing exactly one Cypher
statement. The read-only helpers here additionally share one dispatch
wrapper (``_run_read_cypher``) since none of them ever SET or DELETE —
unlike ``provenance.py``'s helpers, which are deliberately kept separate
because each one performs a distinct mutation.
"""

import logging
import re

from schemas import MemoryVisibility
from memory.groups import _build_group_id
from memory.provenance import GraphReadError

logger = logging.getLogger(__name__)

# Belt-and-braces heuristic token extraction (NOT provenance-based — see
# ``_extract_distinctive_tokens``). Conservative on purpose: currency
# amounts and long digit runs are distinctive enough that a coincidental
# match across two unrelated memories is unlikely, unlike short words or
# common phrases.
_CURRENCY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
_LONG_DIGIT_RE = re.compile(r"\d{6,}")



class RemediationReadError(GraphReadError):
    """A read-only audit/remediation query could not run (bridge/Neo4j down).

    Raised by ``_run_read_cypher`` and deliberately propagated by
    ``audit_private_leakage`` / ``rescope_private_derivatives`` so a failed
    read is surfaced as an error — never as a false-clean ``total == 0``.
    """

class RemediationMixin:
    """RemediationMixin for MemoryService — backward-looking cleanup of
    already-leaked private-graph derivatives, plus a read-only audit that
    proves whether any remain.
    """

    # ──────────────────────────────────────────────
    # Public entry points
    # ──────────────────────────────────────────────

    def rescope_private_derivatives(self, user_id: str, *, dry_run: bool = True) -> dict:
        """Find and cascade-expire shared-group graph derivatives of every
        PRIVATE memory ``user_id`` owns.

        For each private memory, derives the shared-side group ids it
        could previously have lived in (:meth:`_candidate_shared_groups`),
        resolves the memory's episode in each candidate group by persisted
        uuid (``metadata.graph_episode_uuid`` — only present on rows W1
        stamped going forward) or verbatim content match (the reliable
        signal for pre-fix rows — Graphiti stores episode content
        byte-for-byte), and cascades the expiry from there
        (``memory/provenance.py::_cascade_expire_episode``).

        ``dry_run=True`` (the default) performs the SAME resolution
        lookups but never calls the cascade — no ``SET``/``DELETE`` Cypher
        runs at all, only ``MATCH ... RETURN``. A second non-dry run after
        a first is idempotent: the episode node is already hard-deleted,
        so resolution finds nothing left and every count is zero.

        Re-enrichment: when a cascade actually removed a shared derivation
        AND the private group has no episode of its own for that memory
        yet (content lookup comes up empty), one graph-enrichment job is
        queued in the returned ``graph_jobs`` list, shaped exactly like
        ``memory/edit.py::patch_memory``'s ``graph_job`` — the caller
        (REST/MCP) is responsible for enqueuing it via
        ``TaskManager.enqueue_graph_enrichment``, mirroring how the edit
        path's caller enqueues its single job. Graphiti work is
        minutes-slow and must never run inline here.

        Returns a dict with per-run counts, the resolved edge/episode
        uuids, and an ``unresolved`` / ``unresolved_memory_ids``
        accounting: a private memory for which NO candidate group
        resolves an episode but the belt-and-braces heuristic
        (:meth:`_heuristic_any_hit`) still finds a suspicious text match
        in one is counted there instead of being silently skipped — it
        means "something may still be there and we can't safely act on
        it," which must never be conflated with "clean."
        """
        memories_checked = 0
        episodes_found = 0
        edges_expired = 0
        nodes_removed = 0
        summaries_cleared = 0
        edge_uuids: list[str] = []
        episode_uuids: list[str] = []
        unresolved = 0
        unresolved_memory_ids: list[str] = []
        graph_jobs: list[dict] = []

        try:
            rows = self._scroll_all_user_memories(user_id)
        except GraphReadError:
            raise  # infra read failure: never degrade to a false-clean result
        except Exception as e:
            logger.warning(
                "rescope_private_derivatives: failed to scroll memories for user=%r: %s",
                user_id, e,
            )
            rows = []

        for row in rows:
            payload = row.get("payload", {}) or {}
            metadata = payload.get("metadata", {}) or {}
            if isinstance(metadata.get("metadata"), dict):
                metadata = metadata["metadata"]
            visibility = metadata.get("visibility") or MemoryVisibility.PRIVATE.value
            if visibility != MemoryVisibility.PRIVATE.value:
                continue  # only private rows can have a leaked shared derivative

            memories_checked += 1
            memory_id = row.get("id", "")
            content = payload.get("data", "") or ""
            owner = metadata.get("owner_user_id") or user_id
            project_id = metadata.get("project_id")
            workspace = metadata.get("workspace")
            persisted_uuid = metadata.get("graph_episode_uuid")
            source_ref = metadata.get("source_ref")

            candidate_groups = self._candidate_shared_groups(project_id, workspace)

            resolved_any = False
            for group_id in candidate_groups:
                resolved_uuid = self._resolve_episode_uuid(group_id, persisted_uuid, content)
                if not resolved_uuid:
                    continue

                resolved_any = True
                episodes_found += 1
                episode_uuids.append(resolved_uuid)
                # Preview BEFORE any mutation — after the cascade runs, the
                # `expired_at IS NULL` predicate this query shares with
                # `_expire_episode_edges` would return nothing.
                preview_edges = self._preview_episode_edges(group_id, resolved_uuid)

                if dry_run:
                    edge_uuids.extend(preview_edges)
                    continue

                # Pass the already-resolved uuid straight through so the
                # cascade doesn't re-run its own (redundant but harmless)
                # resolution — this guarantees we expire the exact episode
                # we just previewed, not a second lookup that could in
                # principle land on a different row.
                cascade = self._cascade_expire_episode(group_id, episode_uuid=resolved_uuid)
                if cascade.get("resolved"):
                    edges_expired += cascade["edges_expired"]
                    nodes_removed += cascade["nodes_removed"]
                    summaries_cleared += cascade["summaries_cleared"]
                    edge_uuids.extend(preview_edges)

            if not resolved_any:
                tokens = self._extract_distinctive_tokens(content)
                if tokens and self._heuristic_any_hit(candidate_groups, tokens):
                    # We attempted resolution and it failed, but the
                    # heuristic backstop still sees suspicious text in a
                    # shared group — count it, never silently skip it.
                    unresolved += 1
                    unresolved_memory_ids.append(memory_id)
                continue

            if dry_run:
                continue

            # Re-enrichment: only when we actually removed a shared
            # derivation AND the private side has no episode of its own.
            private_group = _build_group_id(
                MemoryVisibility.PRIVATE.value, owner, project_id, workspace
            )
            private_uuid = self._resolve_episode_uuid(private_group, None, content)
            if not private_uuid:
                graph_jobs.append({
                    "memory_id": memory_id,
                    "content": content,
                    "user_id": owner,
                    "project_id": project_id,
                    "visibility": MemoryVisibility.PRIVATE.value,
                    "source_ref": source_ref,
                })

        return {
            "user_id": user_id,
            "dry_run": dry_run,
            "memories_checked": memories_checked,
            "episodes_found": episodes_found,
            "edges_expired": edges_expired,
            "nodes_removed": nodes_removed,
            "summaries_cleared": summaries_cleared,
            "edge_uuids": edge_uuids,
            "episode_uuids": episode_uuids,
            "unresolved": unresolved,
            "unresolved_memory_ids": unresolved_memory_ids,
            "graph_jobs": graph_jobs,
        }

    def audit_private_leakage(self, user_id: str) -> dict:
        """Read-only proof of whether any of ``user_id``'s private memories
        still have a live derivative in a shared group. Never writes.

        For every private memory, resolves its episode (uuid/content, same
        as :meth:`rescope_private_derivatives`) in each candidate shared
        group. A resolved episode is itself surface (c) — raw episode
        content exposed in the shared group — and its still-live edges
        (surface a) and non-empty-summary mentioned entities (surface b)
        are read alongside it. A memory can surface through any subset —
        e.g. zero live edges but a leaked entity summary, or vice versa —
        so all three are checked independently per resolved episode
        rather than short-circuiting on the first hit.

        PLUS a heuristic backstop (:meth:`_heuristic_any_hit`'s sibling,
        :meth:`_heuristic_scan_group`): any still-live edge/entity/episode
        in a candidate group whose text contains a distinctive token
        (currency figure, long digit run) from the private memory, found
        independently of episode resolution. This is a substring match,
        not provenance — kept in its own ``heuristic`` bucket, clearly
        labeled, and never conflated with the provenance-backed findings.

        ``total`` after a successful :meth:`rescope_private_derivatives`
        run must be ``0``.
        """
        edges_found: list[dict] = []
        node_summaries_found: list[dict] = []
        episodes_found: list[dict] = []
        heuristic_found: list[dict] = []

        try:
            rows = self._scroll_all_user_memories(user_id)
        except GraphReadError:
            raise  # infra read failure: never degrade to a false-clean result
        except Exception as e:
            logger.warning(
                "audit_private_leakage: failed to scroll memories for user=%r: %s",
                user_id, e,
            )
            rows = []

        for row in rows:
            payload = row.get("payload", {}) or {}
            metadata = payload.get("metadata", {}) or {}
            if isinstance(metadata.get("metadata"), dict):
                metadata = metadata["metadata"]
            visibility = metadata.get("visibility") or MemoryVisibility.PRIVATE.value
            if visibility != MemoryVisibility.PRIVATE.value:
                continue

            memory_id = row.get("id", "")
            content = payload.get("data", "") or ""
            project_id = metadata.get("project_id")
            workspace = metadata.get("workspace")
            persisted_uuid = metadata.get("graph_episode_uuid")

            candidate_groups = self._candidate_shared_groups(project_id, workspace)
            tokens = self._extract_distinctive_tokens(content)

            for group_id in candidate_groups:
                resolved_uuid = self._resolve_episode_uuid(group_id, persisted_uuid, content)
                if resolved_uuid:
                    episodes_found.append({
                        "memory_id": memory_id,
                        "group_id": group_id,
                        "episode_uuid": resolved_uuid,
                    })
                    for edge_uuid in self._preview_episode_edges(group_id, resolved_uuid):
                        edges_found.append({
                            "memory_id": memory_id,
                            "group_id": group_id,
                            "episode_uuid": resolved_uuid,
                            "edge_uuid": edge_uuid,
                        })
                    for entity in self._preview_episode_entity_summaries(resolved_uuid):
                        node_summaries_found.append({
                            "memory_id": memory_id,
                            "group_id": group_id,
                            "episode_uuid": resolved_uuid,
                            **entity,
                        })

                if tokens:
                    for hit in self._heuristic_scan_group(group_id, tokens):
                        heuristic_found.append({
                            "memory_id": memory_id,
                            "group_id": group_id,
                            **hit,
                        })

        total = (
            len(edges_found) + len(node_summaries_found)
            + len(episodes_found) + len(heuristic_found)
        )
        return {
            "user_id": user_id,
            "leaked": total > 0,
            "total": total,
            "by_surface": {
                "edges": edges_found,
                "node_summaries": node_summaries_found,
                "episodes": episodes_found,
                "heuristic": heuristic_found,
            },
        }

    # ──────────────────────────────────────────────
    # Candidate-group derivation
    # ──────────────────────────────────────────────

    def _candidate_shared_groups(
        self, project_id: str | None, workspace: str | None
    ) -> list[str]:
        """Shared-tier group ids a private memory could previously have
        lived in, derived from ``_build_group_id`` (memory/groups.py)
        rather than hardcoded.

        Always includes the base pool (``shared``) and, when the memory
        has a project_id, the project-scoped pool
        (``shared--project--{project_id}``) — per ``_build_group_id``'s
        own table, that's every shared group_id shape that exists
        independent of workspace.

        ALSO includes the workspace-suffixed variant of each
        (``--ws--{workspace}``) when the memory carries a non-default
        workspace, because ``_build_group_id`` appends that suffix
        uniformly across every visibility/project combination once
        workspace is set and isn't the default ``"memory"``. Both the
        suffixed AND unsuffixed forms are checked (superset, not
        either/or): workspace partitioning (WT6) postdates some of these
        leaked rows, so a pre-partition leak can still be sitting in the
        unsuffixed group even though the memory's CURRENT metadata now
        carries a workspace value.
        """
        candidates = ["shared"]
        if project_id:
            candidates.append(_build_group_id(MemoryVisibility.SHARED.value, "", project_id, None))
        if workspace and workspace != "memory":
            candidates.append(_build_group_id(MemoryVisibility.SHARED.value, "", None, workspace))
            if project_id:
                candidates.append(
                    _build_group_id(MemoryVisibility.SHARED.value, "", project_id, workspace)
                )
        # De-dup while preserving order (project_id == None + workspace
        # unset can't collide here, but keep this robust regardless).
        seen: set[str] = set()
        out: list[str] = []
        for g in candidates:
            if g not in seen:
                seen.add(g)
                out.append(g)
        return out

    def _resolve_episode_uuid(
        self, group_id: str, persisted_uuid: str | None, content: str
    ) -> str | None:
        """Resolve a memory's episode in ``group_id`` WITHOUT mutating
        anything — uuid first (exact, the durable link W1 persists going
        forward), then verbatim content match (the reliable signal for
        pre-fix rows). Same order and same underlying lookup
        (``memory/provenance.py::_lookup_episode_uuid``) as
        ``_cascade_expire_episode`` uses for its own resolution; kept
        separate here so dry-run and non-dry-run share one resolve step
        before either decides whether to mutate.
        """
        # fail_closed: a broken lookup must surface as an error here, never
        # as "episode not found" (which an audit would count as no leak).
        resolved = None
        if persisted_uuid:
            resolved = self._lookup_episode_uuid(group_id, "uuid", persisted_uuid, fail_closed=True)
        if not resolved and content:
            resolved = self._lookup_episode_uuid(group_id, "content", content, fail_closed=True)
        return resolved

    # ──────────────────────────────────────────────
    # Read-only Cypher helpers (never SET/DELETE — safe for dry-run and audit)
    # ──────────────────────────────────────────────

    def _run_read_cypher(self, cypher: str, *, timeout: float = 15.0, **params) -> list[dict]:
        """Shared read-only Cypher dispatch for this module: run one
        ``MATCH ... RETURN`` statement via ``self._run_on_bridge`` and
        return its records as plain dicts.

        FAILS CLOSED: any error (bridge down, Neo4j hiccup) raises instead
        of returning an empty list. An audit/remediation read that cannot
        run must never degrade to "nothing found" — a false-clean audit
        (``total == 0``) would mask ongoing exposure. The REST/MCP admin
        wrappers turn the exception into a non-ok result. Every helper
        below issues exactly one statement through this — none of them
        ever SET or DELETE.
        """
        async def _run():
            async with self._graphiti.driver.session() as session:
                result = await session.run(cypher, **params)
                return await result.data()

        coro = _run()
        try:
            return self._run_on_bridge(coro, timeout=timeout) or []
        except Exception as e:
            coro.close()
            logger.error(
                "remediation read query failed (failing CLOSED): %r", params, exc_info=True,
            )
            raise RemediationReadError(f"remediation read query failed: {e}") from e

    def _preview_episode_edges(self, group_id: str, episode_uuid: str) -> list[str]:
        """Read-only: uuids of the still-live ``RELATES_TO`` edges this
        episode created or reaffirmed in ``group_id`` — the exact same
        match ``provenance.py::_expire_episode_edges`` uses, minus the
        ``SET``. Used both to preview what a dry-run WOULD expire and to
        capture the uuid list right before a real cascade expires them
        (afterward the ``expired_at IS NULL`` predicate would return
        nothing).
        """
        cypher = """
        MATCH (ep:Episodic {uuid: $episode_uuid})
        WITH ep, coalesce(ep.entity_edges, []) AS listed_edge_uuids
        MATCH ()-[r:RELATES_TO {group_id: $group_id}]->()
        WHERE r.expired_at IS NULL
          AND (r.uuid IN listed_edge_uuids OR $episode_uuid IN coalesce(r.episodes, []))
        RETURN r.uuid AS uuid
        """
        records = self._run_read_cypher(cypher, episode_uuid=episode_uuid, group_id=group_id)
        return [r["uuid"] for r in records if r.get("uuid")]

    def _preview_episode_entity_summaries(self, episode_uuid: str) -> list[dict]:
        """Read-only: entities this episode mentions that carry a
        non-empty ``summary`` — surface (b) of the leakage audit. Never
        mutates (contrast ``provenance.py::_clear_or_remove_episode_entities``,
        which clears/removes)."""
        cypher = """
        MATCH (ep:Episodic {uuid: $episode_uuid})-[:MENTIONS]->(n:Entity)
        WHERE n.summary IS NOT NULL AND n.summary <> ''
        RETURN DISTINCT n.uuid AS entity_uuid, n.name AS name, n.summary AS summary
        """
        records = self._run_read_cypher(cypher, episode_uuid=episode_uuid)
        return [
            {"entity_uuid": r.get("entity_uuid"), "name": r.get("name"), "summary": r.get("summary")}
            for r in records
        ]

    # ──────────────────────────────────────────────
    # Heuristic backstop (NOT provenance-based — see module docstring)
    # ──────────────────────────────────────────────

    @staticmethod
    def _extract_distinctive_tokens(content: str) -> list[str]:
        """Conservative candidate-token extraction for the heuristic
        backstop: currency amounts (``$1,234.56``) and long digit runs
        (6+ consecutive digits — account/order/ticket numbers etc.).
        Deliberately narrow — short words or common phrases would make
        the heuristic noisy and turn it into a second, unreliable
        resolution mechanism instead of a backstop. Returns ``[]`` for
        content with no such tokens (the common case), which short-circuits
        the heuristic scan entirely for that memory.
        """
        if not content:
            return []
        tokens: set[str] = set()
        tokens.update(m.group(0) for m in _CURRENCY_RE.finditer(content))
        tokens.update(m.group(0) for m in _LONG_DIGIT_RE.finditer(content))
        return sorted(tokens)

    def _heuristic_any_hit(self, group_ids: list[str], tokens: list[str]) -> bool:
        """True if any candidate group has a still-live artifact whose
        text contains one of ``tokens``. Used only to decide the
        ``unresolved`` accounting in :meth:`rescope_private_derivatives` —
        never to drive a cascade decision."""
        return any(self._heuristic_scan_group(g, tokens) for g in group_ids)

    def _heuristic_scan_group(self, group_id: str, tokens: list[str]) -> list[dict]:
        """Read-only belt-and-braces text scan across edges, entity
        summaries, and episode content in ``group_id`` for a substring
        match against any of ``tokens``. NOT provenance-based — a plain
        text backstop for when episode-uuid/content resolution misses
        (e.g. the private memory's content was edited after the shared
        write, so the verbatim-content lookup no longer matches). Labeled
        ``heuristic`` in the audit output; never used to decide what to
        mutate.
        """
        hits: list[dict] = []
        hits.extend(self._heuristic_scan_edges(group_id, tokens))
        hits.extend(self._heuristic_scan_entities(group_id, tokens))
        hits.extend(self._heuristic_scan_episodes(group_id, tokens))
        return hits

    def _heuristic_scan_edges(self, group_id: str, tokens: list[str]) -> list[dict]:
        cypher = """
        MATCH ()-[r:RELATES_TO {group_id: $group_id}]->()
        WHERE r.expired_at IS NULL AND r.fact IS NOT NULL
          AND ANY(t IN $tokens WHERE r.fact CONTAINS t)
        RETURN r.uuid AS uuid, r.fact AS text
        """
        records = self._run_read_cypher(cypher, group_id=group_id, tokens=tokens)
        return [self._heuristic_hit("edge", r, tokens) for r in records]

    def _heuristic_scan_entities(self, group_id: str, tokens: list[str]) -> list[dict]:
        cypher = """
        MATCH (n:Entity {group_id: $group_id})
        WHERE n.summary IS NOT NULL AND n.summary <> ''
          AND ANY(t IN $tokens WHERE n.summary CONTAINS t)
        RETURN n.uuid AS uuid, n.summary AS text
        """
        records = self._run_read_cypher(cypher, group_id=group_id, tokens=tokens)
        return [self._heuristic_hit("entity", r, tokens) for r in records]

    def _heuristic_scan_episodes(self, group_id: str, tokens: list[str]) -> list[dict]:
        cypher = """
        MATCH (e:Episodic {group_id: $group_id})
        WHERE e.content IS NOT NULL
          AND ANY(t IN $tokens WHERE e.content CONTAINS t)
        RETURN e.uuid AS uuid, e.content AS text
        """
        records = self._run_read_cypher(cypher, group_id=group_id, tokens=tokens)
        return [self._heuristic_hit("episode", r, tokens) for r in records]

    @staticmethod
    def _heuristic_hit(kind: str, record: dict, tokens: list[str]) -> dict:
        """Shape one heuristic-scan record into a reportable hit,
        including which token matched (recomputed in Python — simpler
        than round-tripping it out of Cypher's ANY(...))."""
        text = record.get("text") or ""
        matched_token = next((t for t in tokens if t in text), None)
        return {
            "kind": kind,
            "uuid": record.get("uuid"),
            "matched_token": matched_token,
            "snippet": text[:200],
        }
