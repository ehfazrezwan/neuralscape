"""Cluster 1 regression tests — search-recall defects from audit 27.

One test class per defect in ``docs/neuralscape/27-perf-retrieval-audit.md``
(Cluster 1, items 1-6). Every test here was written FIRST against the broken
code (failing) and then made to pass by the fix — the class docstrings note
what the pre-fix failure looked like.
"""

import hashlib
import math
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memory_service import (
    REINFORCEMENT_BOOST_K,
    MemoryService,
    _reinforcement_boost,
)
from schemas import MemoryResponse


# ──────────────────────────────────────────────
# Shared fixtures / helpers
# ──────────────────────────────────────────────


def _qresult(hits):
    """Wrap hits in a qdrant query_points()-style result (.points)."""
    r = MagicMock()
    r.points = hits
    return r


def _make_hit(id, score, payload):
    hit = MagicMock()
    hit.id = id
    hit.score = score
    hit.payload = payload
    return hit


@pytest.fixture
def service():
    """MemoryService with mocked internals (unit tests, no live services)."""
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    svc._memory.embedding_model.embed.return_value = [0.1] * 768
    svc._memory.vector_store.client.query_points.return_value = _qresult([])
    return svc


def _payload(text, **meta):
    return {"data": text, "created_at": "2026-01-01T00:00:00+00:00", "metadata": meta}


# ══════════════════════════════════════════════
# Defect 1 — BM25 lexical leg restored (RRF fusion per pool)
# ══════════════════════════════════════════════


class TestBM25LexicalLeg:
    """Pre-fix: search() ran a dense-only exactly-k query_points per pool —
    the ``using="bm25"`` sparse leg was never queried, so a memory matching
    the query's proper noun lexically but ranking below k densely was
    unrecallable (test_lexical_only_gold_hit_recalled failed: 'gold' absent).
    """

    QUERY = "Where does Margrethe Vestager work?"

    def _dense_hits(self, k=3):
        # k dense hits, none of them the gold memory.
        return [
            _make_hit(f"d{i}", 0.90 - i * 0.05, _payload(f"unrelated dense fact {i}"))
            for i in range(k)
        ]

    def _gold_hit(self):
        # BM25 scores are NOT cosine-comparable (unbounded).
        return _make_hit(
            "gold", 7.42, _payload("Margrethe Vestager works at the Commission")
        )

    def _wire(self, service, dense_hits, lexical_hits, has_slot=True):
        service._memory.vector_store._has_bm25_slot = has_slot
        service._memory.vector_store._encode_bm25 = MagicMock(
            return_value="SPARSE-VEC"
        )

        def _qp(*args, **kwargs):
            if kwargs.get("using") == "bm25":
                return _qresult(list(lexical_hits))
            return _qresult(list(dense_hits))

        service._memory.vector_store.client.query_points.side_effect = _qp

    def test_lexical_only_gold_hit_recalled(self, service):
        """The gold memory matches the proper noun lexically but is outside
        the dense top-k — the restored BM25 leg must surface it."""
        self._wire(service, self._dense_hits(3), [self._gold_hit()])

        results = service.search(
            query=self.QUERY, user_id="u1", visibility="private", limit=3
        )

        assert any(r.id == "gold" for r in results), (
            "BM25 lexical leg is amputated: a proper-noun match outside the "
            "dense top-k never surfaces"
        )

    def test_lexical_leg_reuses_dense_filter(self, service):
        """The sparse pass must run under the SAME payload filter as the
        dense pass — the lexical leg can never widen visibility/scope."""
        self._wire(service, self._dense_hits(2), [self._gold_hit()])

        service.search(query=self.QUERY, user_id="u1", visibility="private", limit=3)

        calls = service._memory.vector_store.client.query_points.call_args_list
        dense_calls = [c for c in calls if c.kwargs.get("using") != "bm25"]
        bm25_calls = [c for c in calls if c.kwargs.get("using") == "bm25"]
        assert bm25_calls, "no BM25 pass was issued"
        assert bm25_calls[0].kwargs["query_filter"] is dense_calls[0].kwargs["query_filter"]

    def test_dense_scores_survive_fusion(self, service):
        """Fusion must not corrupt cosine scores on dense hits, and a
        lexical-only hit gets the pool's dense floor (BM25 magnitudes are
        not comparable to cosine)."""
        dense = self._dense_hits(3)  # 0.90 / 0.85 / 0.80
        self._wire(service, dense, [self._gold_hit()])

        results = service.search(
            query=self.QUERY, user_id="u1", visibility="private", limit=4
        )

        scores = {r.id: r.score for r in results}
        assert scores["d0"] == pytest.approx(0.90)
        assert scores["gold"] is not None
        assert scores["gold"] <= min(0.90, 0.85, 0.80)

    def test_degrades_to_dense_only_without_bm25_slot(self, service):
        """Collections that predate the bm25 sparse slot must keep working
        dense-only — no sparse query, no error."""
        self._wire(service, self._dense_hits(3), [self._gold_hit()], has_slot=False)

        results = service.search(
            query=self.QUERY, user_id="u1", visibility="private", limit=3
        )

        calls = service._memory.vector_store.client.query_points.call_args_list
        assert all(c.kwargs.get("using") != "bm25" for c in calls)
        assert [r.id for r in results] == ["d0", "d1", "d2"]

    def test_lexical_failure_never_breaks_search(self, service):
        """A transient error on the sparse leg degrades to dense-only."""
        self._wire(service, self._dense_hits(2), [])
        service._memory.vector_store._encode_bm25 = MagicMock(
            side_effect=RuntimeError("fastembed exploded")
        )

        results = service.search(
            query=self.QUERY, user_id="u1", visibility="private", limit=3
        )

        assert [r.id for r in results] == ["d0", "d1"]

    def test_rrf_agreement_ranks_up(self):
        """A hit present in BOTH legs outranks a dense-only hit at the same
        dense rank neighborhood (reciprocal-rank sum)."""
        from memory_service import _rrf_fuse

        d1 = _make_hit("d1", 0.9, _payload("a"))
        both = _make_hit("both", 0.85, _payload("b"))
        fused = _rrf_fuse([d1, both], [both], limit=2)

        assert [str(e["hit"].id) for e in fused] == ["both", "d1"]
        assert fused[0]["dense"] is True


# ══════════════════════════════════════════════
# Defect 2 — vector/graph weave (ranked hits keep priority)
# ══════════════════════════════════════════════


class TestVectorGraphWeave:
    """Pre-fix: `_deduplicate_responses` interleaved scored vector hits 1:1
    with score=None graph edges and the caller cut `combined[:limit]` — at
    limit=10 with 6 graph rows, vector ranks 6-10 were evicted by unranked
    relation strings (and the method took no ``limit`` at all, so these
    tests raised TypeError before the fix).
    """

    def setup_method(self):
        self.svc = MemoryService()

    @staticmethod
    def _vector(n):
        return [
            MemoryResponse(
                id=f"v{i}", memory=f"vector fact {i}", score=0.95 - i * 0.01,
                source="vector",
            )
            for i in range(n)
        ]

    @staticmethod
    def _graph(n):
        return [
            MemoryResponse(id=f"g{i}", memory=f"graph relation {i}", source="graph")
            for i in range(n)
        ]

    def test_vector_ranks_survive_and_graph_capped(self):
        """10 scored vector hits + 6 graph rows at limit=10: at least
        ceil(3*limit/4) vector hits survive, in ranked order, and graph
        rows come after them (pre-fix weave kept only v0-v4)."""
        limit = 10
        result = self.svc._deduplicate_responses(
            self._vector(10), self._graph(6), limit=limit
        )

        assert len(result) == limit
        vector_survivors = [r for r in result if r.source == "vector"]
        assert len(vector_survivors) >= math.ceil(3 * limit / 4)  # >= 8
        # ranked order preserved, graph strictly after every vector hit
        assert [r.id for r in vector_survivors] == [f"v{i}" for i in range(8)]
        first_graph = next(i for i, r in enumerate(result) if r.source == "graph")
        assert all(r.source == "graph" for r in result[first_graph:])
        # graph rows capped at max(1, limit // 4)
        assert sum(1 for r in result if r.source == "graph") <= max(1, limit // 4)

    def test_graph_fills_shortfall_when_vector_underfills(self):
        result = self.svc._deduplicate_responses(
            self._vector(3), self._graph(6), limit=10
        )

        assert [r.id for r in result[:3]] == ["v0", "v1", "v2"]
        assert sum(1 for r in result if r.source == "graph") == 6
        assert len(result) == 9

    def test_vector_reclaims_unused_graph_reservation(self):
        result = self.svc._deduplicate_responses(
            self._vector(10), self._graph(1), limit=10
        )

        assert len(result) == 10
        assert sum(1 for r in result if r.source == "vector") == 9
        assert result[-1].source == "graph"

    def test_limit_one_prefers_the_top_vector_hit(self):
        result = self.svc._deduplicate_responses(
            self._vector(1), self._graph(1), limit=1
        )
        assert [r.id for r in result] == ["v0"]

    def test_substring_dedup_still_applies(self):
        vector = [
            MemoryResponse(
                id="v0", memory="User prefers tabs over spaces", score=0.9,
                source="vector",
            )
        ]
        graph = [MemoryResponse(id="g0", memory="prefers tabs", source="graph")]
        result = self.svc._deduplicate_responses(vector, graph, limit=10)
        assert [r.id for r in result] == ["v0"]

    def test_no_limit_appends_graph_after_vector(self):
        """Back-compat: called without a limit, vector still precedes graph."""
        result = self.svc._deduplicate_responses(self._vector(2), self._graph(2))
        assert [r.source for r in result] == ["vector", "vector", "graph", "graph"]


# ══════════════════════════════════════════════
# Defect 3 — bi-temporally invalidated graph edges filtered
# ══════════════════════════════════════════════


def _edge(uuid, fact, invalid_at=None, expired_at=None):
    return SimpleNamespace(
        uuid=uuid, name=fact, fact=fact, invalid_at=invalid_at, expired_at=expired_at
    )


class TestInvalidatedEdgeFilter:
    """Pre-fix: `_do_graph_search` passed no SearchFilters and never looked
    at invalid_at/expired_at, so edges the dreaming sweep bi-temporally
    invalidated kept surfacing (test_invalidated_edges_excluded failed with
    all three uuids present).
    """

    def _wire(self, service, edges):
        results = SimpleNamespace(edges=edges, nodes=[], episodes=[], communities=[])
        service._run_on_bridge = MagicMock(return_value=results)

    def test_invalidated_edges_excluded(self, service):
        live = _edge("e-live", "Alice works at Acme")
        invalidated = _edge(
            "e-dead", "Alice works at OldCorp",
            invalid_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        expired = _edge(
            "e-exp", "stale relation", expired_at="2026-01-01T00:00:00+00:00"
        )
        self._wire(service, [live, invalidated, expired])

        with patch.object(service, "_enrich_graph_results"):
            out = service._do_graph_search(
                query="alice", group_ids=["user--u1"], limit=10
            )

        assert [e["uuid"] for e in out["edges"]] == ["e-live"]

    def test_search_filter_requests_null_invalid_and_expired(self, service):
        """The Graphiti query itself must carry the is-null SearchFilters so
        the DB never returns invalidated edges in the first place."""
        from graphiti_core.search.search_filters import ComparisonOperator

        self._wire(service, [])
        with patch.object(service, "_enrich_graph_results"):
            service._do_graph_search(query="q", group_ids=["shared"], limit=5)

        sf = service._graphiti.search_.call_args.kwargs["search_filter"]
        assert sf is not None
        assert (
            sf.invalid_at[0][0].comparison_operator is ComparisonOperator.is_null
        )
        assert (
            sf.expired_at[0][0].comparison_operator is ComparisonOperator.is_null
        )

