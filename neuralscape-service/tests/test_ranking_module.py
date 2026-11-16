"""Direct unit tests for ``memory/ranking.py``.

Every helper is exercised on its own (no MemoryService, no stores):
RRF fusion, dense score floor, fused-order keys, order-key recording and
resolution, cosine / MRL cosine / graph-edge cosine, vector fitting to the
collection dimension, tombstone detection, reinforcement boost and the
salience tie-breaker.
"""

import math
from types import SimpleNamespace

import pytest

from memory import ranking
from memory.ranking import (
    REINFORCEMENT_TIMES_DERIVED_CAP,
    RRF_K,
    _canonical_embedding_model,
    _dense_score_floor,
    _fit_vector_to_collection,
    _fused_order_keys,
    _graph_edge_cosine,
    _graph_vectors_mrl_comparable,
    _mem_is_tombstoned,
    _mrl_cosine,
    _record_order_key,
    _reinforcement_boost,
    _resolve_order_keys,
    _rrf_fuse,
    _salience_tiebreak,
    _times_derived_from_metadata,
    _unit_cosine,
)


def _hit(hid, score=None):
    return SimpleNamespace(id=hid, score=score)


# ── _rrf_fuse ──────────────────────────────────────────────────────────


class TestRrfFuse:
    def test_empty_inputs(self):
        assert _rrf_fuse([], [], 10) == []

    def test_empty_lexical_is_order_preserving_passthrough(self):
        dense = [_hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)]
        out = _rrf_fuse(dense, [], 10)
        assert [e["hit"].id for e in out] == ["a", "b", "c"]
        assert all(e["dense"] for e in out)

    def test_rrf_value_for_rank_zero(self):
        out = _rrf_fuse([_hit("a", 0.9)], [], 5)
        assert out[0]["rrf"] == pytest.approx(1.0 / (RRF_K + 1))

    def test_hit_in_both_legs_accumulates_and_outranks(self):
        dense = [_hit("a", 0.9), _hit("b", 0.8)]
        lexical = [_hit("b", 5.0), _hit("c", 4.0)]
        out = _rrf_fuse(dense, lexical, 10)
        # b: 1/(k+2) + 1/(k+1); a: 1/(k+1); c: 1/(k+2)
        assert [e["hit"].id for e in out] == ["b", "a", "c"]
        assert out[0]["rrf"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
        assert out[0]["dense"] is True

    def test_lexical_only_hit_is_not_dense(self):
        out = _rrf_fuse([_hit("a", 0.9)], [_hit("z", 12.0)], 10)
        by_id = {e["hit"].id: e for e in out}
        assert by_id["a"]["dense"] is True
        assert by_id["z"]["dense"] is False

    def test_dense_hit_object_wins_over_lexical_duplicate(self):
        d = _hit("a", 0.9)
        lx = _hit("a", 33.0)
        out = _rrf_fuse([d], [lx], 10)
        assert len(out) == 1
        assert out[0]["hit"] is d

    def test_limit_truncates_and_zero_limit(self):
        dense = [_hit(str(i), 1.0 - i / 10) for i in range(5)]
        assert len(_rrf_fuse(dense, [], 3)) == 3
        assert _rrf_fuse(dense, [], 0) == []

    def test_ties_keep_dense_insertion_order(self):
        # a is dense rank 0, x is lexical rank 0 -> identical rrf; stable sort
        out = _rrf_fuse([_hit("a", 0.9)], [_hit("x", 9.0)], 10)
        assert out[0]["rrf"] == pytest.approx(out[1]["rrf"])
        assert [e["hit"].id for e in out] == ["a", "x"]

    def test_ids_coerced_to_str_for_matching(self):
        out = _rrf_fuse([_hit(1, 0.9)], [_hit("1", 3.0)], 10)
        assert len(out) == 1
        assert out[0]["rrf"] > 1 / (RRF_K + 1)

    def test_hits_without_id_collapse_to_one_entry(self):
        # getattr default "" -> all id-less hits share the "" key
        a, b = object(), object()
        out = _rrf_fuse([a, b], [], 10)
        assert len(out) == 1


# ── _dense_score_floor ─────────────────────────────────────────────────


class TestDenseScoreFloor:
    def test_empty(self):
        assert _dense_score_floor([]) is None

    def test_min_of_scores(self):
        assert _dense_score_floor([_hit("a", 0.9), _hit("b", 0.4), _hit("c", 0.6)]) == 0.4

    def test_none_scores_ignored(self):
        assert _dense_score_floor([_hit("a", None), _hit("b", 0.5)]) == 0.5

    def test_all_none_returns_none(self):
        assert _dense_score_floor([_hit("a", None), _hit("b", None)]) is None

    def test_object_without_score_attr(self):
        assert _dense_score_floor([object(), _hit("b", 0.3)]) == 0.3

    def test_zero_score_is_a_real_score(self):
        assert _dense_score_floor([_hit("a", 0.0), _hit("b", 0.5)]) == 0.0


# ── _fused_order_keys ──────────────────────────────────────────────────


class TestFusedOrderKeys:
    def test_empty_dense_leg_gives_all_none(self):
        assert _fused_order_keys([{}, {}, {}], []) == [None, None, None]

    def test_dense_leg_with_only_none_scores_gives_all_none(self):
        assert _fused_order_keys([{}, {}], [_hit("a", None)]) == [None, None]

    def test_empty_fused_empty_result(self):
        assert _fused_order_keys([], [_hit("a", 0.5)]) == []

    def test_cosines_handed_out_by_position_descending(self):
        dense = [_hit("a", 0.3), _hit("b", 0.9), _hit("c", 0.6)]
        keys = _fused_order_keys([{}, {}, {}], dense)
        assert keys == [0.9, 0.6, 0.3]

    def test_positions_past_dense_leg_get_floor(self):
        dense = [_hit("a", 0.8), _hit("b", 0.5)]
        keys = _fused_order_keys([{}] * 4, dense)
        assert keys == [0.8, 0.5, 0.5, 0.5]

    def test_keys_monotone_non_increasing(self):
        dense = [_hit(str(i), s) for i, s in enumerate([0.2, 0.9, 0.5, 0.7])]
        keys = _fused_order_keys([{}] * 7, dense)
        assert all(a >= b for a, b in zip(keys, keys[1:]))

    def test_fewer_fused_than_dense(self):
        dense = [_hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)]
        assert _fused_order_keys([{}], dense) == [0.9]

    def test_none_scores_skipped_in_the_scale(self):
        dense = [_hit("a", None), _hit("b", 0.4), _hit("c", 0.8)]
        assert _fused_order_keys([{}] * 3, dense) == [0.8, 0.4, 0.4]


# ── _record_order_key / _resolve_order_keys ────────────────────────────


class TestRecordOrderKey:
    def test_none_dict_is_noop(self):
        _record_order_key(None, "a", 0.5)  # must not raise

    def test_empty_row_id_is_noop(self):
        d: dict = {}
        _record_order_key(d, "", 0.5)
        assert d == {}

    def test_first_write(self):
        d: dict = {}
        _record_order_key(d, "a", 0.5)
        assert d == {"a": 0.5}

    def test_best_key_wins(self):
        d = {"a": 0.5}
        _record_order_key(d, "a", 0.7)
        assert d["a"] == 0.7
        _record_order_key(d, "a", 0.2)
        assert d["a"] == 0.7

    def test_none_registers_id_without_overwriting(self):
        d: dict = {}
        _record_order_key(d, "a", None)
        assert d == {"a": None}
        d["b"] = 0.4
        _record_order_key(d, "b", None)
        assert d["b"] == 0.4

    def test_real_key_replaces_none_placeholder(self):
        d = {"a": None}
        _record_order_key(d, "a", 0.1)
        assert d["a"] == 0.1

    def test_equal_key_keeps_prior(self):
        d = {"a": 0.5}
        _record_order_key(d, "a", 0.5)
        assert d["a"] == 0.5

    def test_zero_key_beats_none(self):
        d = {"a": None}
        _record_order_key(d, "a", 0.0)
        assert d["a"] == 0.0


class TestResolveOrderKeys:
    def test_empty(self):
        assert _resolve_order_keys({}) == ({}, set())

    def test_all_real_no_imputation(self):
        keys, imputed = _resolve_order_keys({"a": 0.9, "b": 0.4})
        assert keys == {"a": 0.9, "b": 0.4}
        assert imputed == set()

    def test_none_imputed_with_lowest_real_key(self):
        keys, imputed = _resolve_order_keys({"a": 0.9, "b": 0.4, "x": None})
        assert keys["x"] == 0.4
        assert imputed == {"x"}

    def test_no_real_keys_falls_back_to_zero(self):
        keys, imputed = _resolve_order_keys({"x": None, "y": None})
        assert keys == {"x": 0.0, "y": 0.0}
        assert imputed == {"x", "y"}

    def test_zero_real_key_is_real(self):
        keys, imputed = _resolve_order_keys({"a": 0.0, "x": None})
        assert keys == {"a": 0.0, "x": 0.0}
        assert imputed == {"x"}

    def test_does_not_mutate_input(self):
        src = {"a": 0.5, "x": None}
        _resolve_order_keys(src)
        assert src == {"a": 0.5, "x": None}


# ── _unit_cosine ───────────────────────────────────────────────────────


class TestUnitCosine:
    def test_identical_vectors(self):
        assert _unit_cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_scale_invariant(self):
        assert _unit_cosine([1.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_is_zero(self):
        assert _unit_cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_clamped_to_zero(self):
        assert _unit_cosine([1.0, 0.0], [-1.0, 0.0]) == 0.0

    def test_partial_overlap_value(self):
        assert _unit_cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1 / math.sqrt(2))

    @pytest.mark.parametrize(
        "a,b",
        [
            ([], [1.0]),
            ([1.0], []),
            ([], []),
            (None, [1.0]),
            ([1.0], None),
        ],
    )
    def test_empty_or_none_returns_none(self, a, b):
        assert _unit_cosine(a, b) is None

    def test_length_mismatch_returns_none(self):
        assert _unit_cosine([1.0, 2.0], [1.0, 2.0, 3.0]) is None

    def test_zero_norm_returns_none(self):
        assert _unit_cosine([0.0, 0.0], [1.0, 1.0]) is None
        assert _unit_cosine([1.0, 1.0], [0.0, 0.0]) is None

    def test_non_numeric_returns_none(self):
        assert _unit_cosine(["a", "b"], [1.0, 2.0]) is None
        assert _unit_cosine([1.0, None], [1.0, 2.0]) is None

    def test_numeric_strings_do_not_silently_pass(self):
        # "1" * "1" is a str repeat -> TypeError on float + str -> None
        assert _unit_cosine(["1", "2"], ["1", "2"]) is None

    def test_clamped_upper_bound(self):
        v = [0.1] * 7
        assert _unit_cosine(v, v) <= 1.0

    def test_ints_accepted(self):
        assert _unit_cosine([1, 0], [1, 0]) == pytest.approx(1.0)


# ── MRL model helpers ──────────────────────────────────────────────────


class TestCanonicalEmbeddingModel:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("gemini-embedding-001", "gemini-embedding-001"),
            ("models/gemini-embedding-001", "gemini-embedding-001"),
            ("google-vertex/gemini-embedding-001", "gemini-embedding-001"),
            ("  Models/Gemini-Embedding-001  ", "gemini-embedding-001"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_normalises(self, name, expected):
        assert _canonical_embedding_model(name) == expected


class TestGraphVectorsMrlComparable:
    def test_default_settings_comparable(self, monkeypatch):
        monkeypatch.setattr(ranking.settings, "gemini_embedder_model", "gemini-embedding-001")
        monkeypatch.setattr(ranking.settings, "llm_gateway_enabled", False)
        assert _graph_vectors_mrl_comparable() is True

    def test_gateway_same_model_still_comparable(self, monkeypatch):
        monkeypatch.setattr(ranking.settings, "gemini_embedder_model", "gemini-embedding-001")
        monkeypatch.setattr(ranking.settings, "llm_gateway_enabled", True)
        monkeypatch.setattr(
            ranking.settings, "llm_gateway_embedder_model", "google-vertex/gemini-embedding-001"
        )
        assert _graph_vectors_mrl_comparable() is True

    def test_gateway_different_model_not_comparable(self, monkeypatch):
        monkeypatch.setattr(ranking.settings, "gemini_embedder_model", "gemini-embedding-001")
        monkeypatch.setattr(ranking.settings, "llm_gateway_enabled", True)
        monkeypatch.setattr(ranking.settings, "llm_gateway_embedder_model", "text-embedding-3-large")
        assert _graph_vectors_mrl_comparable() is False

    def test_non_mrl_model_same_on_both_sides_not_comparable(self, monkeypatch):
        monkeypatch.setattr(ranking.settings, "gemini_embedder_model", "text-embedding-004")
        monkeypatch.setattr(ranking.settings, "llm_gateway_enabled", False)
        assert _graph_vectors_mrl_comparable() is False

    def test_gateway_model_ignored_when_gateway_disabled(self, monkeypatch):
        monkeypatch.setattr(ranking.settings, "gemini_embedder_model", "gemini-embedding-001")
        monkeypatch.setattr(ranking.settings, "llm_gateway_enabled", False)
        monkeypatch.setattr(ranking.settings, "llm_gateway_embedder_model", "something-else")
        assert _graph_vectors_mrl_comparable() is True


@pytest.fixture
def mrl_on(monkeypatch):
    monkeypatch.setattr(ranking.settings, "gemini_embedder_model", "gemini-embedding-001")
    monkeypatch.setattr(ranking.settings, "llm_gateway_enabled", False)


@pytest.fixture
def mrl_off(monkeypatch):
    monkeypatch.setattr(ranking.settings, "gemini_embedder_model", "text-embedding-004")
    monkeypatch.setattr(ranking.settings, "llm_gateway_enabled", False)


class TestMrlCosine:
    def test_equal_length_matches_unit_cosine(self):
        a, b = [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]
        assert _mrl_cosine(a, b) == _unit_cosine(a, b)

    def test_longer_vector_compared_on_common_prefix(self):
        # prefix of b matches a exactly -> cosine 1 despite divergent tail
        assert _mrl_cosine([1.0, 0.0], [1.0, 0.0, 9.0, 9.0]) == pytest.approx(1.0)

    def test_shorter_second_vector(self):
        assert _mrl_cosine([1.0, 0.0, 5.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_prefix_orthogonal(self):
        assert _mrl_cosine([1.0, 0.0, 7.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_empty_inputs_none(self):
        assert _mrl_cosine([], [1.0]) is None
        assert _mrl_cosine([1.0], []) is None
        assert _mrl_cosine([], []) is None

    def test_prefix_zero_norm_none(self):
        assert _mrl_cosine([0.0, 0.0, 1.0], [1.0, 1.0]) is None

    def test_non_numeric_prefix_none(self):
        assert _mrl_cosine(["x", "y", "z"], [1.0, 2.0]) is None

    def test_unsliceable_input_none(self):
        class Weird:
            def __bool__(self):
                return True

            def __len__(self):
                return 3

            def __getitem__(self, _):
                raise TypeError("no slicing")

        assert _mrl_cosine(Weird(), [1.0, 2.0]) is None


class TestGraphEdgeCosine:
    def test_same_length_plain_cosine(self, mrl_off):
        assert _graph_edge_cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_different_length_mrl_prefix_cosine(self, mrl_on):
        assert _graph_edge_cosine([1.0, 0.0], [1.0, 0.0, 4.0]) == pytest.approx(1.0)

    def test_different_length_non_mrl_unscorable(self, mrl_off):
        assert _graph_edge_cosine([1.0, 0.0], [1.0, 0.0, 4.0]) is None

    def test_empty_query_goes_to_cosine_none(self, mrl_on):
        assert _graph_edge_cosine([], [1.0, 2.0]) is None

    def test_empty_edge_goes_to_cosine_none(self, mrl_on):
        assert _graph_edge_cosine([1.0, 2.0], []) is None

    def test_zero_norm_none(self, mrl_on):
        assert _graph_edge_cosine([0.0, 0.0], [0.0, 0.0]) is None


class TestFitVectorToCollection:
    DIMS = ranking.VECTOR_EMBEDDING_DIMS

    def test_none_and_empty(self, mrl_on):
        assert _fit_vector_to_collection(None) is None
        assert _fit_vector_to_collection([]) is None

    def test_exact_dims_passthrough_unchanged(self, mrl_on):
        vec = [0.5] * self.DIMS
        assert _fit_vector_to_collection(vec) is vec

    def test_shorter_returns_none(self, mrl_on):
        assert _fit_vector_to_collection([1.0] * (self.DIMS - 1)) is None

    def test_longer_mrl_truncated_and_renormalised(self, mrl_on):
        vec = [2.0] * self.DIMS + [99.0] * 10
        out = _fit_vector_to_collection(vec)
        assert len(out) == self.DIMS
        assert math.sqrt(sum(x * x for x in out)) == pytest.approx(1.0)
        assert out[0] == pytest.approx(1 / math.sqrt(self.DIMS))

    def test_longer_non_mrl_returns_none(self, mrl_off):
        assert _fit_vector_to_collection([1.0] * (self.DIMS + 5)) is None

    def test_longer_zero_head_returns_none(self, mrl_on):
        assert _fit_vector_to_collection([0.0] * self.DIMS + [1.0]) is None

    def test_longer_non_numeric_head_returns_none(self, mrl_on):
        vec = ["x"] * self.DIMS + [1.0]
        assert _fit_vector_to_collection(vec) is None

    def test_longer_numeric_strings_are_coerced(self, mrl_on):
        # float("1") works: documents the coercion behaviour of the head slice
        out = _fit_vector_to_collection(["1"] * self.DIMS + [0.0])
        assert out is not None and len(out) == self.DIMS


# ── tombstone / times_derived ──────────────────────────────────────────


class TestMemIsTombstoned:
    def test_flat_flag(self):
        assert _mem_is_tombstoned({"metadata": {"dream_tombstoned": True}}) is True

    def test_double_wrapped_flag(self):
        mem = {"metadata": {"metadata": {"dream_tombstoned": True}}}
        assert _mem_is_tombstoned(mem) is True

    def test_missing_metadata(self):
        assert _mem_is_tombstoned({}) is False

    def test_none_metadata(self):
        assert _mem_is_tombstoned({"metadata": None}) is False

    def test_false_flag(self):
        assert _mem_is_tombstoned({"metadata": {"dream_tombstoned": False}}) is False

    def test_inner_dict_shadows_outer_flag(self):
        # once double-wrapped, only the inner dict is consulted
        mem = {"metadata": {"dream_tombstoned": True, "metadata": {}}}
        assert _mem_is_tombstoned(mem) is False

    def test_non_dict_inner_metadata_uses_outer(self):
        mem = {"metadata": {"dream_tombstoned": True, "metadata": "oops"}}
        assert _mem_is_tombstoned(mem) is True

    def test_truthy_non_bool_flag(self):
        assert _mem_is_tombstoned({"metadata": {"dream_tombstoned": "yes"}}) is True


class TestTimesDerived:
    @pytest.mark.parametrize(
        "meta,expected",
        [
            (None, 1),
            ({}, 1),
            ({"times_derived": 5}, 5),
            ({"times_derived": "7"}, 7),
            ({"times_derived": 0}, 1),
            ({"times_derived": -3}, 1),
            ({"times_derived": None}, 1),
            ({"times_derived": "abc"}, 1),
            ({"times_derived": [1]}, 1),
            ({"metadata": {"times_derived": 4}}, 4),
            ({"metadata": "notadict", "times_derived": 3}, 3),
            ({"times_derived": 2.9}, 2),
        ],
    )
    def test_values(self, meta, expected):
        assert _times_derived_from_metadata(meta) == expected


# ── _reinforcement_boost ───────────────────────────────────────────────


class TestReinforcementBoost:
    @pytest.fixture(autouse=True)
    def _k(self, monkeypatch):
        monkeypatch.setattr(ranking.settings, "reinforcement_boost_k", 0.05)

    def test_none_score_passthrough(self):
        assert _reinforcement_boost(None, {"times_derived": 9}) is None

    def test_unreinforced_unchanged(self):
        assert _reinforcement_boost(0.7, {}) == 0.7
        assert _reinforcement_boost(0.7, None) == 0.7
        assert _reinforcement_boost(0.7, {"times_derived": 1}) == 0.7

    def test_boost_formula(self):
        expected = 0.8 * (1 + 0.05 * math.log1p(3 - 1))
        assert _reinforcement_boost(0.8, {"times_derived": 3}) == pytest.approx(expected)

    def test_double_wrapped_metadata(self):
        a = _reinforcement_boost(0.8, {"metadata": {"times_derived": 4}})
        b = _reinforcement_boost(0.8, {"times_derived": 4})
        assert a == pytest.approx(b)

    def test_cap_on_times_derived(self):
        capped = _reinforcement_boost(0.5, {"times_derived": REINFORCEMENT_TIMES_DERIVED_CAP})
        way_over = _reinforcement_boost(0.5, {"times_derived": 10_000})
        assert capped == pytest.approx(way_over)

    def test_score_capped_at_one(self):
        assert _reinforcement_boost(0.99, {"times_derived": 10}) == 1.0

    def test_zero_k_is_byte_identical(self, monkeypatch):
        monkeypatch.setattr(ranking.settings, "reinforcement_boost_k", 0.0)
        assert _reinforcement_boost(0.6, {"times_derived": 9}) == 0.6

    def test_negative_k_disables(self, monkeypatch):
        monkeypatch.setattr(ranking.settings, "reinforcement_boost_k", -1.0)
        assert _reinforcement_boost(0.6, {"times_derived": 9}) == 0.6

    def test_unparseable_k_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(ranking.settings, "reinforcement_boost_k", "garbage")
        expected = 0.5 * (1 + ranking.REINFORCEMENT_BOOST_K * math.log1p(2))
        assert _reinforcement_boost(0.5, {"times_derived": 3}) == pytest.approx(expected)

    def test_more_reinforced_scores_higher(self):
        lo = _reinforcement_boost(0.5, {"times_derived": 2})
        hi = _reinforcement_boost(0.5, {"times_derived": 6})
        assert hi > lo > 0.5

    def test_zero_score_stays_zero(self):
        assert _reinforcement_boost(0.0, {"times_derived": 5}) == 0.0

    def test_over_reinforced_mediocre_does_not_beat_better_hit(self):
        boosted = _reinforcement_boost(0.80, {"times_derived": 30})
        assert boosted < 0.90


# ── _salience_tiebreak ─────────────────────────────────────────────────


def _resp(rid, score):
    return SimpleNamespace(id=rid, score=score)


@pytest.fixture
def salience(monkeypatch):
    """Patch the dreaming config + trace lookups the helper imports lazily."""
    from extensions.dreaming import config as dcfg
    from extensions.dreaming import traces

    def _apply(k, signals=None, raises=False):
        monkeypatch.setattr(dcfg.dreaming_settings, "salience_recall_k", k, raising=False)

        def fake(ids):
            if raises:
                raise RuntimeError("redis down")
            return dict(signals or {})

        monkeypatch.setattr(traces, "get_strength_signals", fake)

    return _apply


class TestSalienceTiebreak:
    def test_disabled_returns_same_list_untouched(self, salience):
        salience(0.0, {"a": 5.0})
        rs = [_resp("a", 0.5)]
        assert _salience_tiebreak(rs) is rs
        assert rs[0].score == 0.5

    def test_negative_k_disabled(self, salience):
        salience(-0.1, {"a": 5.0})
        rs = [_resp("a", 0.5)]
        _salience_tiebreak(rs)
        assert rs[0].score == 0.5

    def test_empty_responses(self, salience):
        salience(0.05, {"a": 5.0})
        assert _salience_tiebreak([]) == []

    def test_boost_applied(self, salience):
        salience(0.05, {"a": 3.0})
        rs = [_resp("a", 0.5), _resp("b", 0.5)]
        _salience_tiebreak(rs)
        assert rs[0].score == pytest.approx(0.5 * (1 + 0.05 * math.log1p(3.0)))
        assert rs[1].score == 0.5

    def test_none_score_not_boosted(self, salience):
        salience(0.05, {"a": 3.0})
        rs = [_resp("a", None)]
        _salience_tiebreak(rs)
        assert rs[0].score is None

    def test_zero_signal_not_boosted(self, salience):
        salience(0.05, {"a": 0.0})
        rs = [_resp("a", 0.5)]
        _salience_tiebreak(rs)
        assert rs[0].score == 0.5

    def test_empty_signals_returns_unchanged(self, salience):
        salience(0.05, {})
        rs = [_resp("a", 0.5)]
        assert _salience_tiebreak(rs) is rs
        assert rs[0].score == 0.5

    def test_order_keys_receive_same_lift(self, salience):
        salience(0.05, {"a": 3.0})
        rs = [_resp("a", 0.5)]
        keys = {"a": 0.8}
        _salience_tiebreak(rs, keys)
        lift = 1 + 0.05 * math.log1p(3.0)
        assert keys["a"] == pytest.approx(0.8 * lift)

    def test_order_key_none_left_alone(self, salience):
        salience(0.05, {"a": 3.0})
        rs = [_resp("a", 0.5)]
        keys = {"a": None}
        _salience_tiebreak(rs, keys)
        assert keys["a"] is None

    def test_order_key_missing_not_created(self, salience):
        salience(0.05, {"a": 3.0})
        keys: dict = {}
        _salience_tiebreak([_resp("a", 0.5)], keys)
        assert keys == {}

    def test_trace_failure_is_swallowed(self, salience):
        salience(0.05, raises=True)
        rs = [_resp("a", 0.5)]
        assert _salience_tiebreak(rs) is rs
        assert rs[0].score == 0.5

    def test_config_failure_returns_responses(self, monkeypatch):
        from extensions.dreaming import config as dcfg

        class Boom:
            @property
            def salience_recall_k(self):
                raise RuntimeError("bad config")

        monkeypatch.setattr(dcfg, "dreaming_settings", Boom())
        rs = [_resp("a", 0.5)]
        assert _salience_tiebreak(rs) is rs
        assert rs[0].score == 0.5

    def test_responses_without_id_skipped_for_lookup(self, salience, monkeypatch):
        from extensions.dreaming import traces

        salience(0.05, {"a": 3.0})
        seen = {}

        def spy(ids):
            seen["ids"] = ids
            return {"a": 3.0}

        monkeypatch.setattr(traces, "get_strength_signals", spy)
        rs = [_resp("a", 0.5), _resp(None, 0.4), _resp("", 0.3)]
        _salience_tiebreak(rs)
        assert seen["ids"] == ["a"]
        assert rs[1].score == 0.4 and rs[2].score == 0.3

    def test_boost_reorders_close_scores_but_not_distant_ones(self, salience):
        salience(0.1, {"a": 50.0})
        close = [_resp("a", 0.50), _resp("b", 0.52)]
        _salience_tiebreak(close)
        assert close[0].score > close[1].score
        far = [_resp("a", 0.30), _resp("b", 0.90)]
        _salience_tiebreak(far)
        assert far[0].score < far[1].score
