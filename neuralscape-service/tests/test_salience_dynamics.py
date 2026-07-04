"""Unit tests for A4 salience dynamics + the A3-lite settling guard.

Pure-function heavy: strength cap / co-recall / repeat-query damping,
spacing effect, stability-modulated decay with the 0.05 floor, the three
contractual recall-safety guardrails (k=0 no-op, faded-but-relevant beats
hot-but-mediocre, prune NOMINATION only), trace-side persistence, scoring
fallback parity, staging wiring, and the settling guard with a fake clock.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from extensions.dreaming import consolidate, dynamics, gate, scoring
from extensions.dreaming.consolidate import PoolBatch
from extensions.dreaming.dynamics import DynamicsState
from tests.test_dreaming import FakeRedis

NOW = 1_800_000_000.0  # fixed fake clock for determinism


# ── dynamics.reinforce: strength ────────────────────────────────────


def test_strength_delta_per_recall_and_hard_cap():
    s = DynamicsState()
    s1 = dynamics.reinforce(s, now=NOW)
    assert s1.strength == pytest.approx(1.05)  # +δ per recall
    assert s1.recall_count == 1

    # guardrail 3: increments saturate — hammer it far past the cap
    for i in range(500):
        s1 = dynamics.reinforce(s1, now=NOW + i)
    assert s1.strength == pytest.approx(dynamics.STRENGTH_CAP)
    assert s1.strength <= dynamics.STRENGTH_CAP


def test_co_recall_earns_extra_delta():
    solo = dynamics.reinforce(DynamicsState(), now=NOW, co_recalled=False)
    pair = dynamics.reinforce(DynamicsState(), now=NOW, co_recalled=True)
    assert solo.strength == pytest.approx(1.05)
    assert pair.strength == pytest.approx(1.10)  # +δ recall, +δ co-recall
    assert pair.co_recall_count == 1 and solo.co_recall_count == 0


def test_repeat_query_hammering_is_damped():
    """Guardrail 3: the query-diversity term — one query repeated N times
    must be worth far less than N distinct queries."""
    hammered = DynamicsState()
    hammered = dynamics.reinforce(hammered, now=NOW, novel_query=True)
    for i in range(9):
        hammered = dynamics.reinforce(hammered, now=NOW + i, novel_query=False)

    diverse = DynamicsState()
    for i in range(10):
        diverse = dynamics.reinforce(diverse, now=NOW + i, novel_query=True)

    assert hammered.strength < diverse.strength
    # 1 novel (+.05) + 9 damped (+.05*.2 each) = 1.14 vs 10 novel = 1.5
    assert hammered.strength == pytest.approx(1.14)
    assert diverse.strength == pytest.approx(1.5)


# ── dynamics.reinforce: stability (spacing effect) ──────────────────


def test_spacing_effect_close_reinforcements_dont_grow_stability():
    s = dynamics.reinforce(DynamicsState(), now=NOW)
    for i in range(1, 10):
        s = dynamics.reinforce(s, now=NOW + i * 600)  # 10 min apart
    assert s.stability == pytest.approx(1.0)  # massed practice: no growth


def test_spacing_effect_spaced_reinforcements_grow_stability():
    s = dynamics.reinforce(DynamicsState(), now=NOW)  # first: nothing to space against
    assert s.stability == pytest.approx(1.0)
    s = dynamics.reinforce(s, now=NOW + 3600)  # exactly 1h later
    assert s.stability == pytest.approx(2.0)
    s = dynamics.reinforce(s, now=NOW + 3600 + 7200)
    assert s.stability == pytest.approx(3.0)

    # stability saturates too (no unbounded decay immunity)
    for i in range(50):
        s = dynamics.reinforce(s, now=NOW + (i + 10) * 7200)
    assert s.stability == pytest.approx(dynamics.STABILITY_CAP)


# ── dynamics.salience: decay ────────────────────────────────────────


def test_decay_floor_never_zero():
    s = DynamicsState(strength=1.05, stability=1.0, last_activated_at=NOW)
    fifty_years = NOW + 50 * 365 * 86400
    val = dynamics.salience(s, now=fifty_years)
    assert val == pytest.approx(dynamics.DECAY_FLOOR)  # dim, don't delete
    assert val > 0.0


def test_decay_is_monotonic_in_age():
    s = DynamicsState(strength=2.0, stability=1.0, last_activated_at=NOW)
    ages = [0, 10, 45, 90, 365]
    vals = [dynamics.salience(s, now=NOW + d * 86400) for d in ages]
    assert vals == sorted(vals, reverse=True)


def test_stability_slows_decay():
    frail = DynamicsState(strength=2.0, stability=1.0, last_activated_at=NOW)
    stable = DynamicsState(strength=2.0, stability=5.0, last_activated_at=NOW)
    later = NOW + 90 * 86400
    assert dynamics.salience(stable, now=later) > dynamics.salience(frail, now=later)
    # same strength, no age → identical
    assert dynamics.salience(stable, now=NOW) == pytest.approx(
        dynamics.salience(frail, now=NOW)
    )


def test_fresh_state_matches_legacy_neutral_seed():
    """Switching a memory onto the dynamics path on its first recall must
    not reclassify it toward pruning: fresh state ≈ the legacy 0.7 seed."""
    assert dynamics.salience(DynamicsState(), now=NOW) == pytest.approx(0.7, abs=1e-9)


def test_state_serialization_roundtrip_and_garbage():
    s = dynamics.reinforce(DynamicsState(), now=NOW, co_recalled=True)
    assert dynamics.from_dict(json.dumps(dynamics.to_dict(s))) == s
    assert dynamics.from_dict(json.dumps(dynamics.to_dict(s)).encode()) == s
    for garbage in (None, "", "not json", b"\xff", 42, ["x"], '{"s": "NaNope"}'):
        assert dynamics.from_dict(garbage) == DynamicsState()


# ── Guardrail 1: the recall tie-breaker ─────────────────────────────


def test_recall_boost_k_zero_is_identity():
    score = 0.87654321
    assert dynamics.recall_boost(score, 4.0, 0.0) is score  # untouched, not recomputed
    assert dynamics.recall_boost(score, 0.0, 0.05) is score  # no signal → untouched
    assert dynamics.recall_boost(None, 4.0, 0.05) is None


@pytest.mark.parametrize("k", [0.0, 0.01, 0.05])
def test_faded_but_relevant_beats_hot_but_mediocre(k):
    """Guardrail 1: relevance always dominates at any k ≤ the conservative
    recommended default (0.05)."""
    faded_relevant = dynamics.recall_boost(0.9, 0.0, k)      # faded: zero signal
    max_signal = dynamics.STRENGTH_CAP - dynamics.INITIAL_STRENGTH  # hottest possible
    hot_mediocre = dynamics.recall_boost(0.5, max_signal, k)
    assert faded_relevant > hot_mediocre


def test_salience_tiebreak_k_zero_is_byte_identical_and_redis_free(monkeypatch):
    """The k=0 default must leave the search path untouched: same list,
    same score objects, and ZERO calls into the trace store."""
    import memory_service
    from extensions.dreaming import traces
    from extensions.dreaming.config import dreaming_settings

    monkeypatch.setattr(dreaming_settings, "salience_recall_k", 0.0)

    def _boom(*a, **kw):  # pragma: no cover - must never run
        raise AssertionError("k=0 must not touch the trace store")

    monkeypatch.setattr(traces, "get_strength_signals", _boom)
    responses = [
        SimpleNamespace(id="a", score=0.9),
        SimpleNamespace(id="b", score=0.5),
    ]
    scores_before = [r.score for r in responses]
    out = memory_service._salience_tiebreak(responses)
    assert out is responses
    assert [r.score for r in out] == scores_before
    assert all(r.score is s for r, s in zip(out, scores_before))


def test_salience_tiebreak_boosts_but_relevance_dominates(monkeypatch):
    import memory_service
    from extensions.dreaming import traces
    from extensions.dreaming.config import dreaming_settings

    monkeypatch.setattr(dreaming_settings, "salience_recall_k", 0.05)
    monkeypatch.setattr(
        traces, "get_strength_signals", lambda ids: {"hot": 4.0, "faded": 0.0}
    )
    faded = SimpleNamespace(id="faded", score=0.9)   # relevant but never recalled
    hot = SimpleNamespace(id="hot", score=0.5)       # mediocre but maxed-out salience
    out = memory_service._salience_tiebreak([faded, hot])
    assert hot.score > 0.5                            # the boost is real...
    assert faded.score == 0.9                         # ...zero signal untouched...
    assert faded.score > hot.score                    # ...and relevance still wins
    assert out[0] is faded


def test_salience_tiebreak_survives_trace_store_failure(monkeypatch):
    import memory_service
    from extensions.dreaming import traces
    from extensions.dreaming.config import dreaming_settings

    monkeypatch.setattr(dreaming_settings, "salience_recall_k", 0.05)

    def _boom(ids):
        raise ConnectionError("redis down")

    monkeypatch.setattr(traces, "get_strength_signals", _boom)
    r = SimpleNamespace(id="a", score=0.9)
    assert memory_service._salience_tiebreak([r]) == [r]
    assert r.score == 0.9


# ── scoring wiring ──────────────────────────────────────────────────


def _row(mid="m1", created="2026-01-01T00:00:00+00:00", **kw):
    return {"memory_id": mid, "content": "x", "created_at": created,
            "confidence": 0.9, **kw}


def test_score_memory_uses_dynamics_state_when_present():
    state = DynamicsState(strength=3.0, stability=4.0, last_activated_at=NOW - 86400)
    scored = scoring.score_memory(
        _row(), {}, now=NOW, dynamics_states={"m1": state}
    )
    expected = dynamics.salience(state, now=NOW, base_half_life_days=45.0)
    assert scored["retention_strength"] == pytest.approx(expected)
    assert scored["salience"] == scored["retention_strength"]


def test_score_memory_fallback_is_bit_identical_to_legacy():
    """No dynamics state (or feature off) ⇒ exactly the old half-life math."""
    row = _row()
    traces = {"m1": {"recall_count": 3, "unique_query_count": 2,
                     "last_recalled_at": NOW - 10 * 86400}}
    for scored in (
        scoring.score_memory(dict(row), traces, now=NOW),                       # no arg
        scoring.score_memory(dict(row), traces, now=NOW, dynamics_states={}),   # empty
        scoring.score_memory(dict(row), traces, now=NOW, dynamics_states={"other": DynamicsState()}),
    ):
        legacy = scoring.retention_strength(
            base_confidence=0.9, recall_count=3,
            last_recalled_at=NOW - 10 * 86400,
            created_at=scoring._parse_ts(row["created_at"]), now=NOW,
            half_life_days=45.0,
        )
        assert scored["retention_strength"] == legacy
        assert scored["salience"] == legacy


# ── staging wiring (guardrail 2: nomination only) ───────────────────


def _batch(memories):
    return PoolBatch(pool="user--u", group_id="user--u", visibility="private",
                     owner_user_id="u", project_id=None, memories=memories)


def test_stage_pool_dynamics_strength_drives_prune_nomination(monkeypatch):
    """An old, co-recalled memory resists the weak-retention path; its
    untouched twin decays into PRUNE candidacy — nomination only."""
    monkeypatch.setattr(
        consolidate, "read_aggregates",
        lambda redis, ids: {i: {"recall_count": 0, "unique_query_count": 0,
                                "last_recalled_at": 0.0} for i in ids},
    )
    now = time.time()
    recalled_state = DynamicsState(
        strength=1.4, stability=2.0, last_activated_at=now - 86400
    )
    monkeypatch.setattr(
        consolidate, "read_dynamics", lambda redis, ids: {"recalled": recalled_state}
    )
    ancient = "2024-01-01T00:00:00+00:00"
    batch = _batch([
        {"memory_id": "recalled", "content": "kept alive by recall",
         "created_at": ancient, "confidence": 0.9, "source_type": "explicit"},
        {"memory_id": "forgotten", "content": "never recalled",
         "created_at": ancient, "confidence": 0.9, "source_type": "explicit"},
    ])
    staged = consolidate.stage_pool(
        batch, FakeRedis(), last_dreamt_at=now - 86400,
        max_memories=100, strength_half_life_days=45.0,
        prune_strength_threshold=0.15,
    )
    ids = {m["memory_id"] for m in staged.memories}
    assert ids == {"forgotten"}  # only the disused twin is nominated
    nominee = staged.memories[0]
    assert nominee["retention_strength"] < 0.15
    assert "salience" in nominee  # exposed for the vault ranking
    # no delete/tombstone anywhere in staging — nomination is all it does
    assert "dream_tombstoned" not in nominee


def test_stage_pool_dynamics_disabled_never_reads_states(monkeypatch):
    monkeypatch.setattr(
        consolidate, "read_aggregates",
        lambda redis, ids: {i: {"recall_count": 0, "unique_query_count": 0,
                                "last_recalled_at": 0.0} for i in ids},
    )

    def _boom(redis, ids):  # pragma: no cover - must never run
        raise AssertionError("dynamics disabled — read_dynamics must not be called")

    monkeypatch.setattr(consolidate, "read_dynamics", _boom)
    batch = _batch([
        {"memory_id": "a", "content": "x", "created_at": "2026-07-04T00:00:00+00:00",
         "confidence": 0.9, "source_type": "explicit"},
    ])
    staged = consolidate.stage_pool(
        batch, FakeRedis(), last_dreamt_at=0, max_memories=100,
        strength_half_life_days=45.0, prune_strength_threshold=0.15,
        dynamics_enabled=False,
    )
    assert {m["memory_id"] for m in staged.memories} == {"a"}
    assert "salience" in staged.memories[0]


# ── trace-side persistence ──────────────────────────────────────────


@pytest.fixture()
def stub_trace_redis(monkeypatch):
    from extensions.dreaming import traces as tr

    fake = FakeRedis()
    monkeypatch.setattr(tr, "_get_redis", lambda: fake)
    return fake


def test_write_trace_persists_dynamics_with_co_recall_and_damping(stub_trace_redis):
    from extensions.dreaming import traces as tr

    tr._write_trace(["a", "b"], "how do we deploy", 30)     # co-recall, novel
    tr._write_trace(["a", "b"], "how do we deploy", 30)     # same query → damped
    tr._write_trace(["a", "b"], "what breaks the relay", 30)  # second novel query

    states = tr.read_dynamics(stub_trace_redis, ["a", "b", "ghost"])
    assert set(states) == {"a", "b"}
    for state in states.values():
        # 1.0 + .10 (novel co) + .02 (damped co) + .10 (novel co) = 1.22
        assert state.strength == pytest.approx(1.22)
        assert state.recall_count == 3
        assert state.co_recall_count == 3
        assert state.last_activated_at > 0

    signals = tr.get_strength_signals(["a", "ghost"])
    assert signals == {"a": pytest.approx(0.22)}


def test_write_trace_single_result_gets_no_co_bonus(stub_trace_redis):
    from extensions.dreaming import traces as tr

    tr._write_trace(["solo"], "a query", 30)
    state = tr.read_dynamics(stub_trace_redis, ["solo"])["solo"]
    assert state.strength == pytest.approx(1.05)
    assert state.co_recall_count == 0


def test_write_trace_respects_dynamics_disabled(stub_trace_redis, monkeypatch):
    from extensions.dreaming import traces as tr
    from extensions.dreaming.config import dreaming_settings

    monkeypatch.setattr(dreaming_settings, "dynamics_enabled", False)
    tr._write_trace(["a"], "q", 30)
    assert tr.read_dynamics(stub_trace_redis, ["a"]) == {}
    # the classic trace aggregates still land
    assert tr.read_aggregates(stub_trace_redis, ["a"])["a"]["recall_count"] == 1


# ── A3-lite settling guard ──────────────────────────────────────────


def _mem_at(ts_iso: str, **kw):
    return {"memory_id": "m", "content": "x", "created_at": ts_iso, **kw}


def test_settling_gate_fake_clock():
    from datetime import datetime, timezone

    now = 1_800_000_000.0

    def iso(offset_s: float) -> str:
        return datetime.fromtimestamp(now - offset_s, tz=timezone.utc).isoformat()

    fresh = [_mem_at(iso(5 * 60))]                       # 5 minutes ago
    settled = [_mem_at(iso(31 * 60))]                    # 31 minutes ago
    updated_fresh = [_mem_at(iso(90 * 60), updated_at=iso(2 * 60))]

    assert not gate.check_settling_gate(fresh, settling_minutes=30, now=now).proceed
    assert "settling" in gate.check_settling_gate(fresh, settling_minutes=30, now=now).reason
    assert gate.check_settling_gate(settled, settling_minutes=30, now=now).proceed
    # updated_at counts as a write too
    assert not gate.check_settling_gate(updated_fresh, settling_minutes=30, now=now).proceed
    # 0 disables; empty/parseless pools never settle
    assert gate.check_settling_gate(fresh, settling_minutes=0, now=now).proceed
    assert gate.check_settling_gate([], settling_minutes=30, now=now).proceed
    assert gate.check_settling_gate([{"memory_id": "x"}], settling_minutes=30, now=now).proceed


@pytest.mark.asyncio
async def test_dream_pool_defers_with_settling_status_and_force_bypasses():
    from datetime import datetime, timezone

    from extensions.dreaming import sweep
    from extensions.dreaming.config import DreamingSettings

    settings = DreamingSettings(
        _env_file=None, enabled=True, min_hours=0.0, min_new_memories=1,
        settling_minutes=30.0, reflection_enabled=False,
        vault_pages_enabled=False, dynamics_enabled=True,
    )
    fresh_iso = datetime.now(timezone.utc).isoformat()
    batch = _batch([
        {"memory_id": "a", "content": "just written", "created_at": fresh_iso,
         "confidence": 0.9, "source_type": "explicit"},
    ])
    llm = AsyncMock(return_value='{"actions": []}')

    report = await sweep._dream_pool(
        service=MagicMock(), settings=settings, redis=FakeRedis(),
        llm_call=llm, batch=batch, dry_run=True, force=False,
    )
    assert report.status == "settling"
    assert "settling" in report.reason
    llm.assert_not_awaited()  # deferred before any expensive work

    # force=true bypasses the settling guard (and the other gates)
    report2 = await sweep._dream_pool(
        service=MagicMock(), settings=settings, redis=FakeRedis(),
        llm_call=llm, batch=batch, dry_run=True, force=True,
    )
    assert report2.status == "dreamt"


@pytest.mark.asyncio
async def test_dream_run_totals_count_settling_as_gated():
    from extensions.dreaming.sweep import DreamRun, PoolReport

    run = DreamRun(run_id="r", started_at="now")
    run.pools = [
        PoolReport(pool="a", status="settling", reason="settling: 1m < 30m"),
        PoolReport(pool="b", status="gated", reason="time"),
        PoolReport(pool="c", status="dreamt"),
    ]
    totals = run.to_dict()["totals"]
    assert totals["pools_gated"] == 2
    assert totals["pools_dreamt"] == 1
