"""Read path: hybrid vector+graph search, keyword search, dedup and merge.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

import logging

from config import settings
from schemas import MemoryResponse, MemoryVisibility
from memory.groups import _edge_is_invalidated, _get_group_ids, _live_edges_filter
from memory.ranking import RRF_K, _dense_score_floor, _reinforcement_boost, _rrf_fuse, _salience_tiebreak, _unit_cosine

logger = logging.getLogger(__name__)


def _dt_to_iso(value):
    """Coerce a Graphiti datetime (or already-string) to a canonical ISO string.

    Graphiti edge/episode temporal fields are ``datetime`` objects, but the
    NS envelope (``MemoryResponse.created_at`` etc.) is ``str | None`` and
    Pydantic rejects a datetime. Returns ``value.isoformat()`` for datetimes,
    the value unchanged if it's already a string, else None.
    """
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


# ── Overlapped graph pass (audit 27 #9) ────────────────────────────────
# search() used to run its vector pools to completion and only then start
# the Graphiti pass — wall time was vector + graph even though the two legs
# are independent once the query embed exists. The graph leg now runs on
# this small dedicated pool while the calling thread works the vector legs,
# and is joined right before the weave. Threads here spend their life
# blocked on the Graphiti bridge future, so a modest pool bounds concurrent
# graph fan-out without ever serializing a single search.
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

_GRAPH_SEARCH_POOL = _ThreadPoolExecutor(
    max_workers=16, thread_name_prefix="graph-search"
)
# The bridge call inside _do_graph_search already enforces its own 30s
# timeout; this outer join is belt-and-suspenders so a wedged bridge thread
# can never hang a read indefinitely.
_GRAPH_SEARCH_JOIN_TIMEOUT_S = 45.0

# R3: per-snippet char cap for recall-side episode excerpts. Bounds evidence
# tokens; the COUNT of episode rows is separately capped at 3 (the ask sweet
# spot). A plain char cap (not ask.py's sentence-boundary _clip_content) keeps
# the search mixin free of an ask.py import (which would be circular).
_EPISODE_SNIPPET_CLIP = 600

class SearchMixin:
    """SearchMixin for MemoryService (mechanical split — see memory_service.py)."""

    def _search_shared_pool(
        self,
        m,
        query: str,
        project_id: str | None,
        categories: list[str] | None,
        scope: str | None,
        domain: str | None,
        observation_type: str | None,
        concepts: list[str] | None,
        limit: int,
        query_embedding: list[float] | None = None,
        visibility_value: str = MemoryVisibility.SHARED.value,
        workspaces: list[str] | None = None,
    ) -> list[MemoryResponse]:
        """Search Qdrant for cross-writer memories of a given visibility.

        Used by ``search()`` to deliver team-wide knowledge to all
        authenticated callers. Bypasses mem0's wrapper because that
        wrapper enforces user_id namespacing — for the shared/standard pools we
        explicitly want hits across writers, scoped by
        ``metadata.visibility=<visibility_value>`` plus any other supplied
        filters. ``visibility_value`` selects the pool: ``"shared"`` (default,
        team-wide) or ``"standard"`` (authoritative dictator-written).

        ``query_embedding`` lets the caller pass a precomputed query vector so a
        single ``search()`` doesn't re-embed the same query for every pool/scope
        (the embed round-trip dominates read latency); falls back to embedding
        ``query`` when not provided.

        ``workspaces`` (WT6) filters by workspace partition. Default ``None``
        is treated as ``["memory"]`` (memory-type workspace only — reference
        workspaces fenced out). Pass explicit workspace names to search
        reference content.
        """
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
        )

        client = m.vector_store.client
        embedding = (
            query_embedding
            if query_embedding is not None
            else m.embedding_model.embed(query, memory_action="search")
        )

        must: list = [
            FieldCondition(
                key="metadata.visibility",
                match=MatchValue(value=visibility_value),
            )
        ]
        # Dreaming: exclude reversible consolidation tombstones from recall.
        must_not = [
            FieldCondition(key="metadata.dream_tombstoned", match=MatchValue(value=True))
        ]
        if categories:
            must.append(FieldCondition(key="metadata.category", match=MatchAny(any=categories)))
        if scope:
            must.append(FieldCondition(key="metadata.scope", match=MatchValue(value=scope)))
        if project_id:
            must.append(FieldCondition(key="metadata.project_id", match=MatchValue(value=project_id)))
        if domain:
            must.append(FieldCondition(key="metadata.domain", match=MatchValue(value=domain)))
        if observation_type:
            must.append(FieldCondition(key="metadata.observation_type", match=MatchValue(value=observation_type)))
        if concepts:
            must.append(FieldCondition(key="metadata.concepts", match=MatchAny(any=concepts)))

        # Workspace filter (WT6): default to memory type only (reference fenced out).
        # Explicit workspaces list opens the door to reference content. For now,
        # simple implementation: filter post-retrieval if needed. A future optimization
        # can use Qdrant's should/must filter combinations.
        # When workspaces=None or ["memory"], exclude any rows with non-memory workspace.
        effective_workspaces = workspaces if workspaces is not None else ["memory"]
        if effective_workspaces == ["memory"]:
            # Memory-only: exclude rows with a non-None, non-"memory" workspace.
            # This is done post-retrieval for simplicity (Qdrant's filter syntax
            # doesn't cleanly express "IsEmpty OR IsNull OR MatchValue('memory')").
            pass  # Applied as post-filter below

        # qdrant-client v1.13+ removed `.search()` in favor of `.query_points()`;
        # the response wraps hits in a `.points` attribute.
        qfilter = Filter(must=must, must_not=must_not)
        result = client.query_points(
            collection_name=settings.qdrant_collection,
            query=embedding,
            query_filter=qfilter,
            limit=limit,
            with_payload=True,
        )
        dense_hits = list(getattr(result, "points", result) or [])

        # Hybrid recall (audit 27 #1): BM25 lexical leg under the SAME
        # filter, rank-fused with the dense leg before the pool merge.
        lexical_hits = self._lexical_pool_hits(m, query, qfilter, limit)
        fused = _rrf_fuse(dense_hits, lexical_hits, limit)
        dense_floor = _dense_score_floor(dense_hits)

        out: list[MemoryResponse] = []
        for entry in fused:
            hit = entry["hit"]
            payload = getattr(hit, "payload", None) or {}
            metadata = payload.get("metadata", {})
            # Workspace filter (WT6): post-retrieval filter by workspace list.
            # Absent/None/"memory" all represent the memory workspace.
            hit_workspace = metadata.get("workspace")
            workspace_normalized = hit_workspace if hit_workspace else "memory"
            if workspace_normalized not in effective_workspaces:
                continue  # Skip this hit — it's from a non-requested workspace
            # Lexical-only hits carry a raw BM25 score (not cosine-comparable);
            # impute the pool's dense floor so the cross-pool score sort stays
            # meaningful without ever letting a keyword match outrank a
            # stronger dense hit on score alone.
            raw_score = getattr(hit, "score", None) if entry["dense"] else dense_floor
            mem_dict = {
                "id": str(getattr(hit, "id", "")),
                "memory": payload.get("data", ""),
                "metadata": metadata,
                # Reinforcement-aware recall: memories that survived N dedup
                # collapses rank slightly above one-offs at equal similarity.
                "score": _reinforcement_boost(raw_score, metadata),
                "created_at": payload.get("created_at"),
            }
            out.append(self._mem_to_response(mem_dict))
        return out

    def _lexical_pool_hits(self, m, query: str, query_filter, limit: int) -> list:
        """BM25 lexical leg for one pool (audit 27 #1).

        Runs a sparse-vector keyword query against the collection's
        ``bm25`` named-vector slot using the SAME payload filter as the
        pool's dense pass, so the lexical leg can never widen
        visibility/scope. Degrades to an empty list (dense-only search,
        no error) when:

        - the collection predates the v3 hybrid schema (no ``bm25`` sparse
          slot — the mem0 fork detects this at startup via
          ``_has_bm25_slot``),
        - the configured vector store isn't the NS mem0 Qdrant fork,
        - the query is empty or sparse encoding fails,
        - the Qdrant call errors — recall must never break on the lexical
          extra.
        """
        vs = m.vector_store
        # `is True` (not truthiness): only the fork sets a real bool; any
        # other store lacks the attribute and must skip the sparse leg.
        if not query or getattr(vs, "_has_bm25_slot", False) is not True:
            return []
        try:
            sparse = vs._encode_bm25(query)
            if sparse is None:
                return []
            result = vs.client.query_points(
                collection_name=settings.qdrant_collection,
                query=sparse,
                using="bm25",
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            return list(getattr(result, "points", result) or [])
        except Exception as e:
            logger.debug(f"BM25 lexical leg failed (non-fatal, dense-only): {e}")
            return []

    def _search_standard_pool(
        self,
        m,
        query: str,
        project_id: str | None,
        categories: list[str] | None,
        scope: str | None,
        domain: str | None,
        observation_type: str | None,
        concepts: list[str] | None,
        limit: int,
        query_embedding: list[float] | None = None,
        workspaces: list[str] | None = None,
    ) -> list[MemoryResponse]:
        """Search the authoritative ``standard``-tier pool (dictator-written).

        Thin wrapper over ``_search_shared_pool`` that scopes to
        ``metadata.visibility=standard``. Returned to every caller so binding
        org standards surface in recall regardless of ``include_shared``.
        """
        return self._search_shared_pool(
            m=m,
            query=query,
            project_id=project_id,
            categories=categories,
            scope=scope,
            domain=domain,
            observation_type=observation_type,
            concepts=concepts,
            limit=limit,
            workspaces=workspaces,
            query_embedding=query_embedding,
            visibility_value=MemoryVisibility.STANDARD.value,
        )

    def _search_personal_pool(
        self,
        m,
        user_id: str,
        query_embedding: list[float],
        project_id: str | None,
        categories: list[str] | None,
        scope: str | None,
        domain: str | None,
        observation_type: str | None,
        concepts: list[str] | None,
        limit: int,
        query: str = "",
        workspaces: list[str] | None = None,
    ) -> list[MemoryResponse]:
        """Search Qdrant for the caller's own memories using a precomputed vector.

        Mirrors ``_search_shared_pool`` but scopes by the top-level ``user_id``
        payload field (mem0's namespace) instead of ``visibility=shared``, and
        returns the caller's memories regardless of visibility. We query Qdrant
        directly — like the shared pool — rather than via ``Memory.search`` so a
        single ``search()`` embeds the query ONCE and reuses ``query_embedding``
        across every pool/scope, instead of re-embedding per ``Memory.search``
        call (the embed round-trip dominates read latency).

        ``query`` (raw text) feeds the BM25 lexical leg only — sparse
        encoding is lexical, not an embed round-trip. Empty ⇒ dense-only.
        """
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
        )

        client = m.vector_store.client
        must: list = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        # Dreaming: rows consolidated away by a sweep stay in Qdrant as
        # reversible tombstones but are excluded from live recall.
        must_not = [
            FieldCondition(key="metadata.dream_tombstoned", match=MatchValue(value=True))
        ]
        if categories:
            must.append(FieldCondition(key="metadata.category", match=MatchAny(any=categories)))
        if scope:
            must.append(FieldCondition(key="metadata.scope", match=MatchValue(value=scope)))
        if project_id:
            must.append(FieldCondition(key="metadata.project_id", match=MatchValue(value=project_id)))
        if domain:
            must.append(FieldCondition(key="metadata.domain", match=MatchValue(value=domain)))
        if observation_type:
            must.append(FieldCondition(key="metadata.observation_type", match=MatchValue(value=observation_type)))
        if concepts:
            must.append(FieldCondition(key="metadata.concepts", match=MatchAny(any=concepts)))

        qfilter = Filter(must=must, must_not=must_not)
        result = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_embedding,
            query_filter=qfilter,
            limit=limit,
            with_payload=True,
        )
        dense_hits = list(getattr(result, "points", result) or [])

        # Hybrid recall (audit 27 #1): BM25 lexical leg under the SAME
        # filter, rank-fused with the dense leg before the pool merge.
        lexical_hits = self._lexical_pool_hits(m, query, qfilter, limit)
        fused = _rrf_fuse(dense_hits, lexical_hits, limit)
        dense_floor = _dense_score_floor(dense_hits)

        # Workspace filter (WT6): same post-retrieval filter as shared pool
        effective_workspaces = workspaces if workspaces is not None else ["memory"]

        out: list[MemoryResponse] = []
        for entry in fused:
            hit = entry["hit"]
            payload = getattr(hit, "payload", None) or {}
            metadata = payload.get("metadata", {})
            # Workspace filter (WT6): post-retrieval filter by workspace list
            hit_workspace = metadata.get("workspace")
            workspace_normalized = hit_workspace if hit_workspace else "memory"
            if workspace_normalized not in effective_workspaces:
                continue  # Skip this hit — it's from a non-requested workspace
            # Lexical-only hits: impute the pool's dense floor (see
            # _search_shared_pool for the rationale).
            raw_score = getattr(hit, "score", None) if entry["dense"] else dense_floor
            mem_dict = {
                "id": str(getattr(hit, "id", "")),
                "memory": payload.get("data", ""),
                "metadata": metadata,
                # Same reinforcement boost as the shared/standard pools — the
                # re-rank must be identical across pools or a reinforced
                # private memory would lose to its unreinforced shared twin.
                "score": _reinforcement_boost(raw_score, metadata),
                "created_at": payload.get("created_at"),
            }
            out.append(self._mem_to_response(mem_dict))
        return out

    # ──────────────────────────────────────────────
    # Search operations
    # ──────────────────────────────────────────────

    def search(
        self,
        query: str,
        user_id: str,
        project_id: str | None = None,
        categories: list[str] | None = None,
        scope: str | None = None,
        limit: int = 10,
        # Memory-model v2 filters
        domain: str | None = None,
        observation_type: str | None = None,
        concepts: list[str] | None = None,
        memory_kind: str | None = None,
        # Multi-user pool selection
        visibility: str | None = None,
        include_shared: bool = True,
        # Workspace partition (WT6): default ["memory"] fences out reference content
        workspaces: list[str] | None = None,
        # Internal write-path mode (audit 27 #12): vector pools only — no
        # graph pass, no graph enrichment, no recall-trace logging.
        vector_only: bool = False,
    ) -> list[MemoryResponse]:
        """Semantic search across memories with scope/category filters.

        Multi-user model: returns the union of two pools, dedup'd by id:

        - **Personal pool**: memories owned by `user_id` (regardless of
          visibility — you can always read what you wrote).
        - **Shared pool**: memories with `metadata.visibility=shared`, written
          by anyone in this Neuralscape instance.

        Use ``visibility="private"`` to scope to the personal pool only;
        ``visibility="shared"`` to scope to the shared pool only;
        ``include_shared=False`` to skip the shared pool entirely.

        When project_id is provided, both pools search project+global memories
        for that project (existing dual-scope merge preserved per pool).

        ``vector_only=True`` is the internal write-path mode: the raw-write
        idempotency check needs a cheap semantic near-dupe probe, not a full
        hybrid recall — it skips the graph pass, graph enrichment, and the
        dreaming recall trace (an internal probe must not reinforce
        salience).

        Returns:
            List of matching memory responses sorted by score.
        """
        m = self._get_memory()
        # Lazily ensure keyword/bool payload indexes for the hot-path filters
        # (audit 27 #14) — one attempt per service instance, never fatal.
        self._ensure_filter_indexes()

        # Embed the query ONCE and reuse the vector across every pool/scope
        # below. Both pools query Qdrant directly with this precomputed vector
        # (see _search_personal_pool / _search_shared_pool). Previously each
        # Memory.search + shared-pool call re-embedded the same query — 4-5
        # embeds per recall — and the embed round-trip dominates read latency.
        query_embedding = m.embedding_model.embed(query, memory_action="search")

        # Audit 27 #9: kick the graph pass off NOW, on its own thread, so it
        # overlaps the vector pool queries below (the legs are independent —
        # Graphiti embeds the query itself). Joined right before the weave.
        graph_future = None
        if not vector_only:
            try:
                graph_future = _GRAPH_SEARCH_POOL.submit(
                    self._search_graph_for_visibility,
                    query=query,
                    user_id=user_id,
                    project_id=project_id,
                    limit=limit,
                    visibility=visibility,
                    include_shared=include_shared,
                    include_episodes=settings.graph_episode_recall_enabled,
                )
            except Exception as e:
                logger.warning(f"Graph search submit failed (non-critical): {e}")

        vector_responses: list[MemoryResponse] = []

        # ── Personal pool: the caller's own memories (any visibility) ──
        # Skip when caller restricted to shared-only. Dedup + sort + limit
        # happen once across both pools below.
        want_personal = visibility != MemoryVisibility.SHARED.value
        if want_personal and user_id:
            # Failure isolation: a transient Qdrant error in the personal
            # pool must not abort the whole recall — degrade to shared/graph
            # results instead, matching the shared-pool/graph paths below.
            try:
                if project_id and not scope:
                    # Dual-scope: this user's project-scoped + global memories.
                    vector_responses.extend(
                        self._search_personal_pool(
                            m=m, user_id=user_id, query_embedding=query_embedding,
                            project_id=project_id, categories=categories, scope=None,
                            domain=domain, observation_type=observation_type,
                            concepts=concepts, limit=limit, query=query,
                            workspaces=workspaces,
                        )
                    )
                    vector_responses.extend(
                        self._search_personal_pool(
                            m=m, user_id=user_id, query_embedding=query_embedding,
                            project_id=None, categories=categories, scope="global",
                            domain=domain, observation_type=observation_type,
                            concepts=concepts, limit=limit, query=query,
                            workspaces=workspaces,
                        )
                    )
                else:
                    vector_responses.extend(
                        self._search_personal_pool(
                            m=m, user_id=user_id, query_embedding=query_embedding,
                            project_id=project_id, categories=categories, scope=scope,
                            domain=domain, observation_type=observation_type,
                            concepts=concepts, limit=limit, query=query,
                            workspaces=workspaces,
                        )
                    )
            except Exception as e:
                logger.warning(f"Personal-pool search failed (non-critical): {e}")

        # ── Shared pool: direct Qdrant, no user_id namespace ───────
        # Bypass mem0's wrapper because shared memories span multiple
        # writers; we need a search that returns hits regardless of
        # which user_id wrote them. Only memories with explicit
        # `metadata.visibility=shared` are returned (legacy memories
        # without that field stay de-facto private until migration).
        #
        # Dual-scope merge: when `project_id` is set and `scope` is
        # omitted, mirror the personal-pool's project+global merge —
        # otherwise a project-context search would miss global shared
        # memories that should still be visible (the graph read-set
        # already covers both via `_get_group_ids`). The downstream
        # dedup at line ~1094 collapses any overlap.
        # An explicit `visibility="shared"` selects the shared pool even when
        # `include_shared=False` — otherwise the vector path would suppress the
        # shared pool while the graph path (which keys off `visibility==shared`)
        # still returns it, yielding inconsistent/partial results.
        want_shared = visibility == MemoryVisibility.SHARED.value or (
            include_shared and visibility != MemoryVisibility.PRIVATE.value
        )
        if want_shared:
            try:
                if project_id and not scope:
                    vector_responses.extend(
                        self._search_shared_pool(
                            m=m,
                            query=query,
                            project_id=project_id,
                            categories=categories,
                            scope=None,
                            domain=domain,
                            observation_type=observation_type,
                            concepts=concepts,
                            limit=limit,
                            workspaces=workspaces,
                            query_embedding=query_embedding,
                        )
                    )
                    vector_responses.extend(
                        self._search_shared_pool(
                            m=m,
                            query=query,
                            project_id=None,
                            categories=categories,
                            scope="global",
                            domain=domain,
                            observation_type=observation_type,
                            concepts=concepts,
                            limit=limit,
                            workspaces=workspaces,
                            query_embedding=query_embedding,
                        )
                    )
                else:
                    vector_responses.extend(
                        self._search_shared_pool(
                            m=m,
                            query=query,
                            project_id=project_id,
                            categories=categories,
                            scope=scope,
                            domain=domain,
                            observation_type=observation_type,
                            concepts=concepts,
                            limit=limit,
                            workspaces=workspaces,
                            query_embedding=query_embedding,
                        )
                    )
            except Exception as e:
                logger.warning(f"Shared-pool search failed (non-critical): {e}")

        # ── Standard pool: authoritative dictator-written memories ──────
        # Always included when the tier is enabled (independent of
        # include_shared), because org standards are binding and must surface
        # in recall for everyone. Suppressed only when the caller explicitly
        # narrowed to a different single pool (visibility=private/shared).
        want_standard = settings.standards_enabled and visibility in (
            None,
            MemoryVisibility.STANDARD.value,
        )
        if want_standard:
            try:
                # Standards are ALWAYS written global-scope with no project_id
                # (store_raw forces this), so the standard pool must NOT inherit
                # the caller's scope/project_id — doing so returns zero standards
                # for a project-scoped recall and breaks the everyone-reads-
                # standards guarantee. Query the pool unscoped; it's already
                # filtered to visibility=standard.
                vector_responses.extend(
                    self._search_standard_pool(
                        m=m,
                        query=query,
                        project_id=None,
                        categories=categories,
                        scope=None,
                        domain=domain,
                        observation_type=observation_type,
                        concepts=concepts,
                        limit=limit,
                        workspaces=workspaces,
                        query_embedding=query_embedding,
                    )
                )
            except Exception as e:
                logger.warning(f"Standard-pool search failed (non-critical): {e}")

        # Dedup across the pools (caller's own shared writes match both).
        seen_ids: set[str] = set()
        deduped: list[MemoryResponse] = []
        for r in vector_responses:
            if r.id and r.id not in seen_ids:
                seen_ids.add(r.id)
                deduped.append(r)
        # A4 salience tie-breaker (opt-in; k=0 default returns untouched).
        deduped = _salience_tiebreak(deduped)
        deduped.sort(key=lambda r: r.score or 0.0, reverse=True)
        vector_responses = deduped[:limit]

        # Join the knowledge-graph pass started above and merge edge facts.
        # Multi-user: when caller restricted the visibility, the graph search
        # already restricted its group_ids to match — otherwise the graph
        # would walk the full read-set (caller's private + shared) and we'd
        # have to retroactively filter, which is unreliable for graph
        # rows whose enriched visibility ends up as None.
        graph_responses: list[MemoryResponse] = []
        # Stored fact_embeddings, index-aligned with graph_responses — reused
        # below for the zero-embed twin decoration (query_batch_points with
        # these vectors instead of re-embedding edge facts).
        graph_edge_embeddings: list[list | None] = []
        # R3: episode rows are collected here and appended AFTER edge enrichment
        # so they can never desync graph_edge_embeddings from the edge rows.
        # Initialized at edge scope so the append below is unconditional.
        episodes_for_later: list[MemoryResponse] = []
        if graph_future is not None:
            try:
                graph_results = graph_future.result(
                    timeout=_GRAPH_SEARCH_JOIN_TIMEOUT_S
                )
                for edge in graph_results.get("edges", []):
                    # Graph as a ranked leg: score each edge with its STORED
                    # fact_embedding (piggybacked on the enrichment Cypher in
                    # _enrich_graph_results) against the query vector computed
                    # once above — local cosine in the same Gemini space as
                    # the vector rows, clamped to [0, 1]. Edges without a
                    # stored embedding stay unscored (rank last on merit).
                    emb = edge.get("fact_embedding") or None
                    graph_responses.append(
                        MemoryResponse(
                            id=edge.get("uuid", ""),
                            memory=edge.get("fact", edge.get("name", "")),
                            source="graph",
                            score=(
                                _unit_cosine(query_embedding, emb)
                                if emb is not None
                                else None
                            ),
                            # R4: surface Graphiti's bi-temporal edge validity
                            # metadata so the answer layer can reason about
                            # recency/contradiction (already ISO-stringified).
                            # Intentionally NOT setting created_at from the edge:
                            # graph rows keep getting created_at from their nearest
                            # source memory via enrichment (fills only when None,
                            # _enrich_graph_with_v2). Setting it here would change
                            # graph-row created_at (edge-creation vs storage time)
                            # and shift ask's recency sort — outside R4's scope.
                            valid_at=edge.get("valid_at"),
                            invalid_at=edge.get("invalid_at"),
                        )
                    )
                    graph_edge_embeddings.append(emb)
                # R3: consume episodes from the graph search (when flag enabled).
                # Use the same id/source scheme as ask.py (ep-<uuid12>, source="episode")
                # so deduplication aligns. Episodes are appended AFTER edge
                # enrichment to preserve edge_embeddings index alignment.
                if settings.graph_episode_recall_enabled:
                    for ep in graph_results.get("episodes", []):
                        ep_uuid = str(ep.get("uuid") or "")
                        ep_content = str(ep.get("content") or "")
                        # Clip each recall episode snippet to a bounded length.
                        # (Not the same as ask.py's _clip_content, which clips at
                        # a sentence/whitespace boundary; a plain char cap keeps
                        # the recall leg dependency-free — importing ask into the
                        # search mixin would be circular.)
                        clipped = ep_content[:_EPISODE_SNIPPET_CLIP]
                        episodes_for_later.append(
                            MemoryResponse(
                                id=f"ep-{ep_uuid[:12]}",
                                memory=f"[verbatim session excerpt] {clipped}",
                                source="episode",
                                score=None,  # rank on merit, not score
                                created_at=ep.get("created_at") or None,
                            )
                        )
            except Exception as e:
                logger.warning(f"Graph search failed during recall (non-critical): {e}")
                episodes_for_later = []

        # Enrich graph rows with metadata from their nearest source memory
        # (title/category/created_at/v2 fields + the twin back-reference).
        # Graphiti's edge schema doesn't carry those fields natively — we
        # recover them by semantic match against the Qdrant store.
        #
        # Graph-ranked-leg: decoration is ALWAYS-ON again (audit 27 #7 had
        # gated it behind a v2/passage filter because it cost one embed API
        # call per edge, then #121 one batch call). With the STORED edge
        # embeddings from part 1 the twin lookup is a single
        # query_batch_points round trip and ZERO embed calls, so every
        # search gets decorated graph rows (feeding ask.py's chronological
        # evidence and the recall-index budget logic). Edges without a
        # stored embedding only fall back to a batched embed when a filter
        # actually needs their metadata (v2 filter / passage filter) —
        # otherwise they stay undecorated rather than paying API calls.
        v2_filter_active = bool(domain or observation_type or concepts)
        if graph_responses and v2_filter_active:
            graph_responses = self._enrich_and_filter_graph(
                graph_responses,
                user_id=user_id,
                project_id=project_id,
                domain=domain,
                observation_type=observation_type,
                concepts=concepts,
                visibility=visibility,
                include_shared=include_shared,
                edge_embeddings=graph_edge_embeddings,
            )
        elif graph_responses:
            graph_responses = self._enrich_graph_with_v2(
                graph_responses,
                user_id=user_id,
                project_id=project_id,
                visibility=visibility,
                include_shared=include_shared,
                edge_embeddings=graph_edge_embeddings,
                # The passage filter below reads memory_kind off each row,
                # which for graph rows only exists via source-memory
                # enrichment — worth a batched embed for embedding-less
                # edges. A plain recall is not.
                allow_embed_fallback=(memory_kind == "passage"),
            )

        # R3: append episode rows after edge enrichment to preserve index alignment
        # of graph_edge_embeddings with edge rows (episodes have no embeddings).
        if episodes_for_later:
            graph_responses.extend(episodes_for_later)

        # Multi-user model: post-filter graph rows by enriched visibility.
        # The Graphiti search above already scopes by group_ids, so most
        # rows arrive in the right pool. This pass mops up the edge case
        # where enrichment couldn't find a source memory: when the caller
        # asked for `private`, an unenriched row (visibility=None) is
        # treated as private — it came from the private group_id range
        # we just scoped to. When they asked for `shared`, an unenriched
        # row could only have come from a shared group_id, so we keep it.
        if visibility and graph_responses:
            graph_responses = [
                r for r in graph_responses
                if r.visibility == visibility or r.visibility is None
            ]

        # Deduplicate without applying limit yet — the memory_kind filter
        # below may exclude rows, so we apply limit AFTER filtering to avoid
        # the cap being consumed by filtered-out rows (audit 27 hardening #8).
        combined = self._deduplicate_responses(
            vector_responses, graph_responses, limit=None
        )

        # memory_kind filter (data-layer connectors). Legacy memories have no
        # memory_kind, so a "fact" filter treats null as fact (back-compat);
        # "passage" matches only explicitly-tagged passages. Applied BEFORE
        # the top-k truncation so the cap isn't consumed by filtered-out rows.
        if memory_kind == "fact":
            combined = [r for r in combined if (r.memory_kind or "fact") == "fact"]
        elif memory_kind == "passage":
            combined = [r for r in combined if r.memory_kind == "passage"]

        results = combined[:limit]

        # Dreaming: fire-and-forget recall trace (reinforcement signal for
        # the dream sweep's promotion/retention scoring). Runs on a daemon
        # thread inside log_recall — never blocks or fails the read.
        #
        # Audit 27 #13: skipped entirely unless something CONSUMES the traces
        # — the dream sweep (dreaming enabled) or the bounded salience
        # tie-breaker (salience_recall_k > 0). With both off, log_recall was
        # ~5N+4 Redis writes per search feeding a store nothing ever read.
        # Internal vector_only probes (write-side idempotency checks) never
        # log — they aren't user recalls and must not reinforce salience.
        if not vector_only:
            try:
                from extensions.dreaming.config import dreaming_settings

                if dreaming_settings.enabled or float(dreaming_settings.salience_recall_k) > 0.0:
                    from extensions.dreaming.traces import log_recall

                    log_recall([r.id for r in results if r.id], query)
            except Exception:
                pass

        return results

    # Minimum vector similarity for graph→source enrichment to be trusted.
    # Below this, the "source" is just the nearest unrelated memory and we
    # leave v2 fields as None rather than propagating wrong metadata.
    _GRAPH_ENRICH_THRESHOLD: float = 0.7

    def _enrich_graph_with_v2(
        self,
        graph_responses: list[MemoryResponse],
        user_id: str,
        project_id: str | None,
        visibility: str | None = None,
        include_shared: bool = True,
        edge_embeddings: list | None = None,
        allow_embed_fallback: bool = True,
    ) -> list[MemoryResponse]:
        """For each graph edge, find its nearest Qdrant source memory and
        copy that source's metadata onto the graph response — only when
        the similarity score clears _GRAPH_ENRICH_THRESHOLD. Recovered
        fields: memory-model v2 (domain/observation_type/concepts/…),
        presentation fields (title/category/created_at/token_estimate),
        multi-user fields, and the twin's memory id (as
        ``related_memory_ids``, the graph row's source back-reference).

        Graph-ranked-leg: ``edge_embeddings`` (index-aligned with
        ``graph_responses``) carries the STORED Graphiti ``fact_embedding``
        per row — those rows are looked up with the stored vector and cost
        ZERO embed API calls. Rows without one either fall back to a single
        batched ``embed_batch`` call (``allow_embed_fallback=True``, the
        #121 behavior — used when a v2/passage filter needs their metadata)
        or stay un-enriched (``allow_embed_fallback=False``, the plain-
        search hot path).

        Audit 27 #7 history: this used to embed + query Qdrant once per
        edge, sequentially — 10 edges meant 10 Gemini round-trips + 10
        Qdrant queries (3-12s of the measured hybrid-search latency). At
        most ONE ``embed_batch`` call and exactly ONE Qdrant
        ``query_batch_points`` round trip remain. Any failure leaves the
        rows un-enriched (never breaks the read).

        Multi-user model: the enrichment source filter mirrors the EXACT
        read-set of the calling search (``visibility``/``include_shared``
        select the same pools ``_search_graph_for_visibility`` scoped its
        group_ids to). A private-only or ``include_shared=False`` recall
        must never pick up a shared row's metadata (visibility /
        owner_user_id) for its graph rows — the graph edges themselves
        came from the narrowed group_ids, and enriching them from a wider
        pool would mislabel rows (or get them dropped by the post-filter).
        Restricts to the active project_id when supplied so v2 filter
        parity holds.
        """
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchValue,
            QueryRequest,
        )

        enrichable = [(i, r.memory) for i, r in enumerate(graph_responses) if r.memory]
        if not enrichable:
            return graph_responses

        # Pool selection — parity with _search_graph_for_visibility:
        #   private  → caller's own rows only
        #   shared   → shared pool only
        #   standard → standard pool only
        #   include_shared=False (no explicit visibility) → personal + standard
        #   default  → personal + shared + standard
        want_personal = visibility in (None, MemoryVisibility.PRIVATE.value)
        want_shared = visibility == MemoryVisibility.SHARED.value or (
            visibility is None and include_shared
        )
        want_standard = settings.standards_enabled and visibility in (
            None,
            MemoryVisibility.STANDARD.value,
        )

        try:
            m = self._get_memory()
            client = m.vector_store.client

            # Per-row lookup vector: prefer the STORED edge embedding (zero
            # embed API calls); optionally batch-embed the leftovers.
            provided: dict[int, list] = {}
            if edge_embeddings:
                for i, _ in enrichable:
                    if i < len(edge_embeddings) and edge_embeddings[i]:
                        provided[i] = edge_embeddings[i]
            missing = [(i, text) for i, text in enrichable if i not in provided]
            if missing and not allow_embed_fallback:
                # Plain-search hot path: rows without a stored embedding
                # stay un-enriched rather than paying embed API calls.
                enrichable = [(i, t) for i, t in enrichable if i in provided]
                missing = []
            if missing:
                texts = [text for _, text in missing]
                embed_batch = getattr(m.embedding_model, "embed_batch", None)
                if callable(embed_batch):
                    embeddings = embed_batch(texts, memory_action="search")
                else:  # embedder without a batch API — degrade to per-text embeds
                    embeddings = [
                        m.embedding_model.embed(t, memory_action="search")
                        for t in texts
                    ]
                for (i, _), emb in zip(missing, embeddings):
                    provided[i] = emb
            if not enrichable:
                return graph_responses

            # Enrichment source = OR of per-pool sub-filters (Qdrant `should`
            # accepts nested Filters). Personal + shared are constrained to the
            # active project; the authoritative STANDARD pool is always global
            # (no project_id), so it must NOT carry the project constraint —
            # otherwise standard-origin graph edges never match their source
            # and lose their v2 metadata / get dropped by v2 filters. The
            # filter is identical for every edge, so it is built ONCE.
            # Hardening #9 (HELD FOR BENCH VALIDATION): when project_id is set,
            # include BOTH the project-scoped AND global-scoped pools as
            # fallback — so a graph edge in a project can enrich from global
            # memories when no project memories exist. NOTE: this is ranking-
            # relevant and unverified — Qdrant returns the nearest neighbor by
            # vector distance across all `should` clauses, so a semantically-
            # closer GLOBAL memory can override a project one. Validate the
            # recall/precision impact on the NSBench stack before merging to dev.
            proj = (
                [FieldCondition(key="metadata.project_id", match=MatchValue(value=project_id))]
                if project_id else []
            )
            should_filters: list = []
            if want_personal and user_id:
                should_filters.append(
                    Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))] + proj)
                )
                if project_id:
                    should_filters.append(
                        Filter(must=[
                            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                            FieldCondition(key="metadata.scope", match=MatchValue(value="global")),
                        ])
                    )
            if want_shared:
                should_filters.append(
                    Filter(must=[FieldCondition(
                        key="metadata.visibility",
                        match=MatchValue(value=MemoryVisibility.SHARED.value),
                    )] + proj)
                )
                if project_id:
                    should_filters.append(
                        Filter(must=[
                            FieldCondition(
                                key="metadata.visibility",
                                match=MatchValue(value=MemoryVisibility.SHARED.value),
                            ),
                            FieldCondition(key="metadata.scope", match=MatchValue(value="global")),
                        ])
                    )
            if want_standard:
                should_filters.append(
                    Filter(must=[FieldCondition(
                        key="metadata.visibility",
                        match=MatchValue(value=MemoryVisibility.STANDARD.value),
                    )])
                )
            if not should_filters:
                # No readable enrichment pool (e.g. private-only with no
                # user_id) — leave the rows un-enriched rather than querying
                # unfiltered.
                return graph_responses
            qf = Filter(should=should_filters)

            requests = [
                QueryRequest(query=provided[i], filter=qf, limit=1, with_payload=True)
                for i, _ in enrichable
            ]
            batch_results = client.query_batch_points(
                collection_name=settings.qdrant_collection,
                requests=requests,
            )
        except Exception as e:
            logger.debug(f"Graph enrichment skipped (batch failed): {e}")
            return graph_responses

        for (idx, _), result in zip(enrichable, batch_results):
            resp = graph_responses[idx]
            try:
                hits = getattr(result, "points", result) or []
                if not hits:
                    continue
                hit = hits[0]
                score = getattr(hit, "score", None)
                if score is None and isinstance(hit, dict):
                    score = hit.get("score")
                if score is not None and score < self._GRAPH_ENRICH_THRESHOLD:
                    continue  # too weak a match to trust the metadata link

                payload = getattr(hit, "payload", None)
                if payload is None and isinstance(hit, dict):
                    payload = hit.get("payload", {})
                payload = payload or {}
                src_metadata = payload.get("metadata", {}) or {}
                if isinstance(src_metadata.get("metadata"), dict):
                    src_metadata = src_metadata["metadata"]

                # Presentation decoration (always-on again): timestamps feed
                # ask.py's chronological evidence; title/token_estimate feed
                # the recall-index budget logic; the twin's id is the graph
                # row's source-memory back-reference.
                if resp.created_at is None:
                    resp.created_at = payload.get("created_at")
                if resp.updated_at is None:
                    resp.updated_at = payload.get("updated_at")
                if resp.occurred_at is None:
                    resp.occurred_at = src_metadata.get("occurred_at")
                if resp.title is None:
                    resp.title = src_metadata.get("title")
                if resp.token_estimate is None:
                    resp.token_estimate = src_metadata.get("token_estimate")
                hit_id = getattr(hit, "id", None)
                if hit_id is None and isinstance(hit, dict):
                    hit_id = hit.get("id")
                if resp.related_memory_ids is None and hit_id is not None:
                    resp.related_memory_ids = [str(hit_id)]

                # Copy v2 fields when source has them and graph response doesn't
                if resp.domain is None:
                    resp.domain = src_metadata.get("domain")
                if resp.observation_type is None:
                    resp.observation_type = src_metadata.get("observation_type")
                if resp.concepts is None:
                    resp.concepts = src_metadata.get("concepts")
                if resp.source_type is None:
                    resp.source_type = src_metadata.get("source_type")
                if resp.epistemic_level is None:
                    resp.epistemic_level = src_metadata.get("epistemic_level")
                if resp.confidence is None:
                    resp.confidence = src_metadata.get("confidence")
                if resp.expires_at is None:
                    resp.expires_at = src_metadata.get("expires_at")
                if resp.memory_kind is None:
                    resp.memory_kind = src_metadata.get("memory_kind")
                if resp.source_ref is None:
                    resp.source_ref = src_metadata.get("source_ref")
                if resp.category is None:
                    resp.category = src_metadata.get("category")
                if resp.scope is None:
                    resp.scope = src_metadata.get("scope")
                if resp.project_id is None:
                    resp.project_id = src_metadata.get("project_id")
                if resp.visibility is None:
                    resp.visibility = src_metadata.get("visibility")
                if resp.owner_user_id is None:
                    resp.owner_user_id = src_metadata.get("owner_user_id")
            except Exception as e:
                logger.debug(f"Graph enrichment skipped for {resp.id}: {e}")
        return graph_responses

    def _enrich_and_filter_graph(
        self,
        graph_responses: list[MemoryResponse],
        user_id: str,
        project_id: str | None,
        domain: str | None,
        observation_type: str | None,
        concepts: list[str] | None,
        visibility: str | None = None,
        include_shared: bool = True,
        edge_embeddings: list | None = None,
    ) -> list[MemoryResponse]:
        """Enrich graph rows with v2 metadata, then drop rows that don't match
        the supplied filter. Used when the caller passes domain/observation_type/
        concepts in SearchMemoryRequest. ``visibility``/``include_shared``
        scope the enrichment source to the calling search's read-set.
        ``edge_embeddings`` are the stored per-edge vectors (see
        ``_enrich_graph_with_v2``); embed fallback stays allowed here —
        filter correctness beats latency.
        """
        enriched = self._enrich_graph_with_v2(
            graph_responses,
            user_id=user_id,
            project_id=project_id,
            visibility=visibility,
            include_shared=include_shared,
            edge_embeddings=edge_embeddings,
        )
        out: list[MemoryResponse] = []
        for resp in enriched:
            if domain and resp.domain != domain:
                continue
            if observation_type and resp.observation_type != observation_type:
                continue
            if concepts:
                resp_concepts = set(resp.concepts or [])
                if not (resp_concepts & set(concepts)):
                    continue
            out.append(resp)
        return out

    def _search_graph_for_visibility(
        self,
        query: str,
        user_id: str,
        project_id: str | None,
        limit: int,
        visibility: str | None,
        include_shared: bool,
        include_episodes: bool = False,
    ) -> dict:
        """search_graph with multi-user visibility scoping.

        When the caller restricts visibility to one pool, narrow the
        Graphiti `group_ids` to that pool's namespace. This is
        load-bearing for cross-user isolation: if we walked the full
        group_id set and then filtered by enriched visibility, an
        unenriched row from the shared pool could slip into a
        private-only response.
        """
        def _standard_groups() -> list[str]:
            if not settings.standards_enabled:
                return []
            return ["standard"] + ([f"standard--project--{project_id}"] if project_id else [])

        if visibility == MemoryVisibility.PRIVATE.value:
            group_ids = [f"user--{user_id}"]
            if project_id:
                group_ids.append(f"user--{user_id}--project--{project_id}")
        elif visibility == MemoryVisibility.STANDARD.value:
            group_ids = _standard_groups()
        elif visibility == MemoryVisibility.SHARED.value:
            group_ids = ["shared"]
            if project_id:
                group_ids.append(f"shared--project--{project_id}")
        elif not include_shared:
            # No explicit visibility, but caller opted out of shared pool.
            # Standards remain in-scope — they are binding and independent of
            # the shared opt-out.
            group_ids = [f"user--{user_id}"]
            if project_id:
                group_ids.append(f"user--{user_id}--project--{project_id}")
            group_ids += _standard_groups()
        else:
            # Default: full read-set (caller's private + shared + standard).
            group_ids = _get_group_ids(user_id, project_id)

        return self._do_graph_search(
            query=query, group_ids=group_ids, limit=limit, include_episodes=include_episodes
        )

    def _do_graph_search(
        self,
        query: str,
        group_ids: list[str],
        limit: int,
        search_config: dict | None = None,
        include_episodes: bool = False,
    ) -> dict:
        """Internal: run a graph search across the given group_ids."""
        g = self._get_graphiti()
        if g is None:
            return {"edges": [], "nodes": [], "episodes": [], "communities": []}

        from graphiti_core.search.search_config import (
            EpisodeReranker,
            EpisodeSearchConfig,
            EpisodeSearchMethod,
            SearchConfig,
        )
        from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

        # Audit 27 #10: EDGE_HYBRID_SEARCH_RRF is a module-level singleton
        # shared across every thread — mutating its `.limit` in place let a
        # concurrent delete (limit=5) clamp a live search's graph fan-out.
        # Always work on a deep copy.
        if search_config:
            try:
                config = SearchConfig(**search_config)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid search_config, falling back to default: {e}")
                config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        else:
            config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            # R3: when include_episodes AND using the default recipe, add the
            # bm25 episode leg (no embeddings). Explicit search_configs (e.g.
            # the delete path with limit=5) stay episode-free.
            if include_episodes:
                config.episode_config = EpisodeSearchConfig(
                    search_methods=[EpisodeSearchMethod.bm25],
                    reranker=EpisodeReranker.rrf,
                )
        config.limit = limit

        try:
            results = self._run_on_bridge(
                g.search_(
                    query=query,
                    config=config,
                    group_ids=group_ids,
                    # Audit 27 #3: bi-temporal liveness — edges the dreaming
                    # sweep stamped invalid_at/expired_at (INVALIDATE / PRUNE /
                    # MERGE, see extensions/dreaming/graph_patcher.py) must not
                    # surface as live facts and eat top-k slots.
                    search_filter=_live_edges_filter(),
                )
            )
            edges = [
                {
                    "uuid": e.uuid,
                    "name": e.name,
                    "fact": e.fact,
                    "valid_at": _dt_to_iso(getattr(e, "valid_at", None)),
                    "invalid_at": _dt_to_iso(getattr(e, "invalid_at", None)),
                    "created_at": _dt_to_iso(getattr(e, "created_at", None)),
                }
                for e in results.edges
                # Belt-and-suspenders: some drivers/recipes skip the Cypher
                # filter constructor, so drop stamped edges here too.
                if not _edge_is_invalidated(e)
            ]
            nodes = [
                {"uuid": n.uuid, "name": n.name, "summary": n.summary}
                for n in results.nodes
            ]
            # R3: cap episodes at 3 (ask measured 3 as sweet spot, 5 regressed).
            # Stringify datetimes to ISO — Graphiti hands back datetime objects,
            # but MemoryResponse.created_at is `str | None` and would reject a
            # datetime (silently dropping every episode row via the recall
            # try/except). Keep created_at/valid_at as ISO strings (R4 uses them).
            episodes = [
                {
                    "uuid": ep.uuid,
                    "name": ep.name,
                    "content": ep.content,
                    "created_at": _dt_to_iso(getattr(ep, "created_at", None)),
                    "valid_at": _dt_to_iso(getattr(ep, "valid_at", None)),
                }
                for ep in results.episodes[:3]
            ]
            communities = [
                {"uuid": c.uuid, "name": c.name} for c in results.communities
            ]
            # Enrich nodes/edges/communities with the back-references the
            # synthesizer set (memory_id, wiki_path). Best-effort; a
            # failed enrichment leaves the result as-is.
            self._enrich_graph_results(nodes, edges, communities)
            return {
                "edges": edges,
                "nodes": nodes,
                "episodes": episodes,
                "communities": communities,
            }
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return {"edges": [], "nodes": [], "episodes": [], "communities": []}

    def _enrich_graph_results(
        self,
        nodes: list[dict],
        edges: list[dict],
        communities: list[dict],
    ) -> None:
        """Annotate graph search results with ``memory_id`` + ``wiki_path``,
        and edges additionally with their stored ``fact_embedding``.

        ``memory_id``/``wiki_path`` are added by the wiki synthesizer's
        Cypher patchers (``attach_memory_id`` and ``patch_wiki_path``) as
        top-level Neo4j properties, but Graphiti's ORM doesn't rehydrate
        them. We do one extra Cypher round-trip per result set to fetch the
        values by UUID, then mutate the dicts in place.

        The same round trip piggybacks ``e.fact_embedding`` for RELATES_TO
        edges (graph-as-a-ranked-leg): Graphiti's search return path
        deliberately omits the stored embedding
        (``get_entity_edge_return_query``), and re-embedding edge facts
        would cost one Gemini call per search. The UNION arm below matches
        relationships by uuid (indexed: Graphiti's ``relation_uuid`` range
        index) — zero additional round trips, no Graphiti subtree change.
        Failure logs and leaves the original dicts unchanged.
        """
        all_uuids: list[str] = []
        for collection in (nodes, edges, communities):
            for item in collection:
                u = item.get("uuid")
                if u:
                    all_uuids.append(u)
        if not all_uuids:
            return
        if self._graphiti is None or self._bridge is None:
            return
        cypher = """
        MATCH (n)
        WHERE n.uuid IN $uuids
        RETURN n.uuid AS uuid,
               n.memory_id AS memory_id,
               n.wiki_path AS wiki_path,
               null AS fact_embedding
        UNION ALL
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.uuid IN $uuids
        RETURN e.uuid AS uuid,
               e.memory_id AS memory_id,
               e.wiki_path AS wiki_path,
               e.fact_embedding AS fact_embedding
        """

        async def _run():
            async with self._graphiti.driver.session() as session:
                result = await session.run(cypher, uuids=all_uuids)
                return await result.data()

        try:
            records = self._run_on_bridge(_run(), timeout=10.0) or []
        except Exception:
            logger.warning("graph result enrichment failed (non-critical)", exc_info=True)
            return
        by_uuid = {r["uuid"]: r for r in records if r.get("uuid")}
        for collection in (nodes, edges, communities):
            for item in collection:
                rec = by_uuid.get(item.get("uuid"))
                if not rec:
                    continue
                if rec.get("memory_id"):
                    item["memory_id"] = rec["memory_id"]
                if rec.get("wiki_path"):
                    item["wiki_path"] = rec["wiki_path"]
                if rec.get("fact_embedding"):
                    item["fact_embedding"] = rec["fact_embedding"]

    def search_graph(
        self,
        query: str,
        user_id: str,
        project_id: str | None = None,
        limit: int = 10,
        search_config: dict | None = None,
    ) -> dict:
        """Knowledge graph search via Graphiti.

        Args:
            query: Search query
            user_id: User identifier
            project_id: Optional project to include in search scope
            limit: Maximum results
            search_config: Optional SearchConfig dict override

        Returns:
            Dict with edges, nodes, episodes, communities.
        """
        g = self._get_graphiti()
        if g is None:
            return {"edges": [], "nodes": [], "episodes": [], "communities": []}

        from graphiti_core.search.search_config import SearchConfig
        from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

        # Multi-user: search across the caller's private namespace + the
        # shared pool, plus project-scoped variants. Replaces the prior
        # cross-user `"global"`/`"project--..."` group_ids.
        group_ids = _get_group_ids(user_id, project_id)

        # Audit 27 #10: never mutate the shared module-level recipe singleton
        # — deep-copy before setting the per-call limit.
        if search_config:
            try:
                config = SearchConfig(**search_config)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid search_config, falling back to default: {e}")
                config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        else:
            config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)

        config.limit = limit

        try:
            # Audit 27 (hardening): filter out invalidated/expired edges from
            # graph search results — same live-edge discipline as other paths.
            results = self._run_on_bridge(
                g.search_(
                    query=query,
                    config=config,
                    group_ids=group_ids,
                    search_filter=_live_edges_filter(),
                )
            )

            return {
                "edges": [
                    {
                        "uuid": e.uuid,
                        "name": e.name,
                        "fact": e.fact,
                        "valid_at": _dt_to_iso(getattr(e, "valid_at", None)),
                        "invalid_at": _dt_to_iso(getattr(e, "invalid_at", None)),
                        "created_at": _dt_to_iso(getattr(e, "created_at", None)),
                    }
                    for e in results.edges
                ],
                "nodes": [
                    {"uuid": n.uuid, "name": n.name, "summary": n.summary}
                    for n in results.nodes
                ],
                "episodes": [
                    {"uuid": ep.uuid, "name": ep.name, "content": ep.content}
                    for ep in results.episodes
                ],
                "communities": [
                    {"uuid": c.uuid, "name": c.name}
                    for c in results.communities
                ],
            }
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return {"edges": [], "nodes": [], "episodes": [], "communities": []}

    def keyword_search(
        self,
        user_id: str,
        terms: list[str],
        project_id: str | None = None,
        limit: int = 20,
    ) -> tuple[list[MemoryResponse], bool]:
        """Grep-style exact/keyword scan over caller-visible memories (C3).

        Case-insensitive substring match. A single term must be present;
        with multiple terms a memory must contain at least TWO of them
        (i.e. both when only two are given) — ANY-term matching promoted
        single-common-word noise as "exact" evidence (audit 27 #18). This
        is the dialectic discipline for enumeration/counting questions —
        embeddings under-recall exhaustive "list every X" sets, so the ask
        path runs this exact pass *before* semantic search and dedups the
        union.

        Visibility mirrors search's pool union (caller's own rows at any
        visibility + shared + standard-when-enabled; ``project_id`` adds the
        project+global dual scope) and dream-tombstoned rows are excluded —
        the same rules as ``_timeline_filter``. Bounded scan: pages of
        ``_TIMELINE_FALLBACK_PAGE`` up to ``_TIMELINE_FALLBACK_CAP`` points,
        stopping early once ``limit`` matches are found. Results carry no
        score (exact matches aren't ranked).

        Returns ``(matches, scan_capped)`` — ``scan_capped`` is True when
        the point cap terminated the scan with candidates left unscanned,
        so callers must NOT present the matches as exhaustive. (Follow-up:
        a Qdrant full-text index would make this scan complete and cheap.)
        """
        lowered = [t.strip().lower() for t in terms if t and t.strip()]
        if not lowered:
            return [], False
        # ≥2 terms must match on multi-term queries (ALL when only 2).
        required_matches = min(2, len(lowered))
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = self._get_memory().vector_store.client

        visibility_should = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(
                key="metadata.visibility",
                match=MatchValue(value=MemoryVisibility.SHARED.value),
            ),
        ]
        if settings.standards_enabled:
            visibility_should.append(
                FieldCondition(
                    key="metadata.visibility",
                    match=MatchValue(value=MemoryVisibility.STANDARD.value),
                )
            )
        must: list = [Filter(should=visibility_should)]
        if project_id:
            must.append(
                Filter(
                    should=[
                        FieldCondition(
                            key="metadata.project_id",
                            match=MatchValue(value=project_id),
                        ),
                        FieldCondition(
                            key="metadata.scope", match=MatchValue(value="global")
                        ),
                    ]
                )
            )
        flt = Filter(
            must=must,
            must_not=[
                FieldCondition(
                    key="metadata.dream_tombstoned", match=MatchValue(value=True)
                )
            ],
        )

        matches: list[MemoryResponse] = []
        offset = None
        scanned = 0
        while scanned < self._TIMELINE_FALLBACK_CAP and len(matches) < limit:
            page, offset = client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=flt,
                limit=self._TIMELINE_FALLBACK_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in page or []:
                content = ((getattr(pt, "payload", None) or {}).get("data") or "").lower()
                if sum(1 for t in lowered if t in content) >= required_matches:
                    matches.append(self._point_to_response(pt))
                    if len(matches) >= limit:
                        break
            scanned += len(page or [])
            # `is None`, not truthiness — integer point-id offsets can be 0.
            if offset is None:
                break
        # Partial scan: the point cap fired while Qdrant still had more
        # candidate points (`offset` non-None) — the matches are a sample,
        # not the exhaustive lexical truth.
        scan_capped = offset is not None and scanned >= self._TIMELINE_FALLBACK_CAP
        return matches, scan_capped

    def _deduplicate_responses(
        self,
        vector_responses: list[MemoryResponse],
        graph_responses: list[MemoryResponse],
        limit: int | None = None,
    ) -> list[MemoryResponse]:
        """Fuse the vector and graph legs into one merit-ranked list.

        Graph is a FIRST-CLASS RANKED leg: with part-1 cosine scores on the
        edges (stored ``fact_embedding`` × the query vector), both legs
        carry scores in the SAME Gemini cosine space, so ranks are computed
        on the merged score scale — NO quota and NO cap. Strong graph rows
        take the majority of top-k when they earn it; weak ones sink below
        every stronger vector hit. (Literal per-leg reciprocal-rank fusion
        can't do either: a full vector leg mathematically pins graph at a
        50/50 alternation regardless of magnitude, which both re-creates
        the audit-27 #2 interleave defect for weak edges and forbids a
        graph majority for strong ones.)

        This replaces the interim #120 weave (graph appended after vector,
        capped at ``max(1, limit // 4)``) which itself replaced the 1:1
        positional interleave that let score=None relation strings evict
        ranked vector hits. Unscored graph rows (no stored embedding) still
        rank at the bottom — exactly the old append behavior.

        Content-identity dedup is unchanged: a graph edge whose fact is an
        exact/substring twin of a vector row is dropped (the vector row is
        the richer record) — but the twin still credits its vector row with
        an ``_rrf_fuse``-style reciprocal-rank corroboration bonus
        ``1/(RRF_K + graph_rank + 1)``: leg agreement ranks up. Each row
        keeps its NATIVE score; the fused value only orders the list.
        Ties break deterministically: vector leg first, then id.
        """
        # Graph leg ranked by its cosine scores (strongest first, unscored
        # last, stable) — this rank drives both graph-vs-graph twin
        # preference and the corroboration bonus.
        graph_leg = sorted(
            graph_responses,
            key=lambda r: (r.score is None, -(r.score or 0.0)),
        )

        # entries: {"row", "leg" (0=vector, 1=graph), "fused"}
        entries: list[dict] = []
        # normalized content -> entry index (insertion-ordered so substring
        # twin resolution is deterministic: first matching row wins).
        norm_to_idx: dict[str, int] = {}

        for vr in vector_responses:
            norm_to_idx.setdefault(
                self.normalize_memory_content(vr.memory), len(entries)
            )
            entries.append({"row": vr, "leg": 0, "fused": float(vr.score or 0.0)})

        for rank, gr in enumerate(graph_leg):
            normalized = self.normalize_memory_content(gr.memory)
            twin_norm = self.find_duplicate_content(normalized, norm_to_idx.keys())
            if twin_norm is not None:
                # Content twin: drop the graph row, credit the survivor with
                # this edge's reciprocal-rank weight (corroboration).
                entries[norm_to_idx[twin_norm]]["fused"] += 1.0 / (
                    RRF_K + rank + 1
                )
                continue
            norm_to_idx.setdefault(normalized, len(entries))
            entries.append({"row": gr, "leg": 1, "fused": float(gr.score or 0.0)})

        # Unscored (score=None) rows sort strictly below scored rows even at
        # a fused value of 0.0 — the id tie-break must never lift them.
        entries.sort(
            key=lambda e: (
                e["row"].score is None and e["fused"] == 0.0,
                -e["fused"],
                e["leg"],
                str(e["row"].id or ""),
            )
        )
        fused = [e["row"] for e in entries]
        return fused if limit is None else fused[:limit]

    def _merge_results(self, *result_sets) -> list[dict]:
        """Merge multiple result sets, deduplicate by ID, sort by score descending."""
        seen_ids = set()
        merged = []

        for result_set in result_sets:
            memories = self._extract_memory_list(result_set)
            for mem in memories:
                mem_id = mem.get("id")
                if mem_id and mem_id not in seen_ids:
                    seen_ids.add(mem_id)
                    merged.append(mem)

        # Sort by score (descending) if available
        merged.sort(key=lambda x: x.get("score", 0) or 0, reverse=True)
        return merged
