"""Unit tests for A5 — surprisal-targeted REM (extensions/dreaming/surprisal).

Pure-math coverage (centroid, cosine distance, anomaly scoring), the top-K
substrate bias, the K=0 byte-identical guarantee against the uniform
reflection substrate, staged-dict annotation, and the Qdrant vector-fetch
normalization (list vs named-vector dict). No running services.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from extensions.dreaming import surprisal
from extensions.dreaming.consolidate import PoolBatch
from extensions.dreaming.prompts import render_memories_block
from extensions.dreaming.surprisal import (
    annotate,
    bias_substrate,
    centroid,
    cosine_distance,
    surprisal_scores,
)


def _mem(mid: str, content: str = "some staged fact", **extra) -> dict:
    return {
        "memory_id": mid,
        "content": content,
        "category": "decision",
        "created_at": "2026-07-01T00:00:00+00:00",
        **extra,
    }


# ── Pure math ─────────────────────────────────────────────────────────


class TestCentroid:
    def test_mean_of_vectors(self):
        assert centroid([[0.0, 2.0], [2.0, 0.0]]) == [1.0, 1.0]

    def test_empty_is_none(self):
        assert centroid([]) is None

    def test_dimension_mismatch_is_none(self):
        assert centroid([[1.0, 2.0], [1.0]]) is None

    def test_zero_dim_is_none(self):
        assert centroid([[], []]) is None


class TestCosineDistance:
    def test_identical_is_zero(self):
        assert cosine_distance([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)

    def test_orthogonal_is_one(self):
        assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_opposite_is_two(self):
        assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)

    def test_zero_norm_is_zero(self):
        assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestSurprisalScores:
    def test_fewer_than_three_vectors_no_scores(self):
        assert surprisal_scores({"a": [1.0, 0.0], "b": [0.0, 1.0]}) == {}

    def test_anomaly_scores_highest(self):
        # Three near-identical vectors + one pointing the other way.
        vectors = {
            "m1": [1.0, 0.0],
            "m2": [0.99, 0.05],
            "m3": [0.98, -0.05],
            "weird": [-1.0, 0.1],
        }
        scores = surprisal_scores(vectors)
        assert set(scores) == set(vectors)
        assert max(scores, key=scores.get) == "weird"
        assert scores["weird"] > scores["m1"]

    def test_scores_bounded_zero_to_two(self):
        vectors = {"a": [1.0, 0.0], "b": [-1.0, 0.0], "c": [0.0, 1.0]}
        for s in surprisal_scores(vectors).values():
            assert 0.0 <= s <= 2.0


# ── Annotation ────────────────────────────────────────────────────────


class TestAnnotate:
    def test_stamps_surprisal_on_staged_dicts(self):
        mems = [_mem("m1"), _mem("m2"), _mem("weird")]
        vectors = {
            "m1": [1.0, 0.0],
            "m2": [0.99, 0.01],
            "weird": [-1.0, 0.0],
        }
        stamped = annotate(mems, vectors)
        assert stamped == 3
        assert all(isinstance(m["surprisal"], float) for m in mems)
        by_id = {m["memory_id"]: m for m in mems}
        assert by_id["weird"]["surprisal"] > by_id["m1"]["surprisal"]

    def test_missing_vector_leaves_dict_untouched(self):
        mems = [_mem("m1"), _mem("m2"), _mem("m3"), _mem("no-vec")]
        vectors = {"m1": [1.0, 0.0], "m2": [0.9, 0.1], "m3": [0.8, 0.0]}
        annotate(mems, vectors)
        assert "surprisal" not in mems[3]

    def test_too_few_vectors_stamps_nothing(self):
        mems = [_mem("m1"), _mem("m2")]
        assert annotate(mems, {"m1": [1.0], "m2": [0.5]}) == 0
        assert all("surprisal" not in m for m in mems)


# ── Substrate bias ────────────────────────────────────────────────────


class TestBiasSubstrate:
    def test_k_zero_returns_same_object(self):
        mems = [_mem("m1", surprisal=0.9), _mem("m2", surprisal=0.1)]
        assert bias_substrate(mems, 0) is mems

    def test_no_scores_returns_same_object(self):
        mems = [_mem("m1"), _mem("m2")]
        assert bias_substrate(mems, 5) is mems

    def test_top_k_anomalies_lead_rest_keeps_order(self):
        mems = [
            _mem("a", surprisal=0.1),
            _mem("b", surprisal=0.8),
            _mem("c", surprisal=0.3),
            _mem("d", surprisal=0.9),
            _mem("e"),  # unscored — never an anomaly
        ]
        out = bias_substrate(mems, 2)
        assert [m["memory_id"] for m in out] == ["d", "b", "a", "c", "e"]
        # bias, not filter: nothing dropped
        assert len(out) == len(mems)

    def test_k_larger_than_pool_sorts_all_scored_first(self):
        mems = [
            _mem("a", surprisal=0.1),
            _mem("b"),
            _mem("c", surprisal=0.5),
        ]
        out = bias_substrate(mems, 99)
        assert [m["memory_id"] for m in out] == ["c", "a", "b"]

    def test_deterministic_tie_break_on_memory_id(self):
        mems = [_mem("z", surprisal=0.5), _mem("a", surprisal=0.5), _mem("m", surprisal=0.5)]
        out = bias_substrate(mems, 3)
        assert [m["memory_id"] for m in out] == ["a", "m", "z"]


# ── K=0 byte-identical substrate through reflect() ────────────────────


class TestReflectSubstrate:
    def _batch(self, mems: list[dict]) -> PoolBatch:
        return PoolBatch(
            pool="user--u1",
            group_id="user--u1",
            visibility="private",
            owner_user_id="u1",
            project_id=None,
            memories=mems,
        )

    @pytest.mark.asyncio
    async def test_k_zero_prompt_byte_identical_to_uniform(self):
        """With top_k=0 (and no annotation, as the sweep skips the fetch),
        the rendered reflection prompt is byte-identical to the legacy path."""
        from extensions.dreaming import reflect as reflect_mod

        mems = [_mem(f"m{i}", content=f"fact number {i}") for i in range(5)]
        prompts: list[str] = []

        async def capture(prompt: str) -> str:
            prompts.append(prompt)
            return ""  # parse_json_response -> no insights

        await reflect_mod.reflect(
            self._batch([dict(m) for m in mems]), capture, max_insights=3
        )
        await reflect_mod.reflect(
            self._batch([dict(m) for m in mems]), capture,
            max_insights=3, surprisal_top_k=0,
        )
        assert prompts[0] == prompts[1]

    @pytest.mark.asyncio
    async def test_top_k_puts_anomaly_first_in_memories_block(self):
        from extensions.dreaming import reflect as reflect_mod

        mems = [
            _mem("m1", content="routine fact one", surprisal=0.05),
            _mem("m2", content="routine fact two", surprisal=0.04),
            _mem("anomaly", content="a wildly novel observation", surprisal=1.4),
            _mem("m3", content="routine fact three", surprisal=0.06),
        ]
        prompts: list[str] = []

        async def capture(prompt: str) -> str:
            prompts.append(prompt)
            return ""

        await reflect_mod.reflect(
            self._batch(mems), capture, max_insights=3, surprisal_top_k=2
        )
        block = prompts[0]
        assert block.index("a wildly novel observation") < block.index("routine fact one")

    def test_bias_matches_render_block_order(self):
        mems = [
            _mem("m1", content="alpha", surprisal=0.1),
            _mem("m2", content="beta", surprisal=0.9),
            _mem("m3", content="gamma", surprisal=0.2),
        ]
        block = render_memories_block(bias_substrate(mems, 1), include_strength=False)
        first_line = block.splitlines()[0]
        assert "beta" in first_line


# ── Vector fetch normalization ────────────────────────────────────────


class TestFetchVectors:
    def _service_with(self, points):
        svc = MagicMock()
        svc._memory.vector_store.client.retrieve.return_value = points
        return svc

    def test_plain_list_vectors(self):
        svc = self._service_with([
            SimpleNamespace(id="m1", vector=[0.1, 0.2]),
            SimpleNamespace(id="m2", vector=[0.3, 0.4]),
        ])
        out = surprisal.fetch_vectors(svc, ["m1", "m2"])
        assert out == {"m1": [0.1, 0.2], "m2": [0.3, 0.4]}
        kwargs = svc._memory.vector_store.client.retrieve.call_args.kwargs
        assert kwargs["with_vectors"] is True
        assert kwargs["with_payload"] is False

    def test_named_vector_dict(self):
        svc = self._service_with([
            SimpleNamespace(id="m1", vector={"default": [1.0, 0.0]}),
        ])
        assert surprisal.fetch_vectors(svc, ["m1"]) == {"m1": [1.0, 0.0]}

    def test_hybrid_bm25_named_vectors_pick_dense(self):
        """mem0's hybrid layout: {'bm25': SparseVector, '': dense}. The
        sparse vector must be skipped regardless of dict order."""
        sparse = SimpleNamespace(indices=[1, 5], values=[0.2, 0.7])  # not a list
        svc = self._service_with([
            SimpleNamespace(id="m1", vector={"bm25": sparse, "": [0.5, 0.5]}),
            SimpleNamespace(id="m2", vector={"": [0.1, 0.9], "bm25": sparse}),
        ])
        out = surprisal.fetch_vectors(svc, ["m1", "m2"])
        assert out == {"m1": [0.5, 0.5], "m2": [0.1, 0.9]}

    def test_missing_or_malformed_vectors_skipped(self):
        svc = self._service_with([
            SimpleNamespace(id="m1", vector=None),
            SimpleNamespace(id="m2", vector={}),
            SimpleNamespace(id="m3", vector=["not", "numbers"]),
        ])
        assert surprisal.fetch_vectors(svc, ["m1", "m2", "m3"]) == {}

    def test_empty_ids_no_call(self):
        svc = MagicMock()
        assert surprisal.fetch_vectors(svc, []) == {}
        svc._memory.vector_store.client.retrieve.assert_not_called()
