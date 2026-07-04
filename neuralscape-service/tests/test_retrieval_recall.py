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

