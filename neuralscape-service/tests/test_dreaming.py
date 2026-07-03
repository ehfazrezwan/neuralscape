"""Unit tests for the dreaming extension (no running services needed).

Covers: gate economy + lock, trace aggregation, scoring/strength math,
consolidation decision validation + hybrid adoption split, pool staging
(feedback-loop guard, standard-tier exclusion), reflection validation,
prompt JSON parsing, and the diary renderer.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from extensions.dreaming import consolidate, gate, prompts, reflect, scoring
from extensions.dreaming.consolidate import PoolBatch


# ── Fakes ───────────────────────────────────────────────────────────


class FakeRedis:
    """Minimal in-memory stand-in for the sync redis client."""

    def __init__(self):
        self.kv: dict[str, str] = {}
        self.hashes: dict[str, dict] = {}
        self.hll: dict[str, set] = {}

    def get(self, key):
        return self.kv.get(key)

    def set(self, key, value, nx=False, px=None):
        if nx and key in self.kv:
            return False
        self.kv[key] = value
        return True

    def delete(self, key):
        self.kv.pop(key, None)

    def exists(self, key):
        return key in self.kv

    def pipeline(self, transaction=False):
        return FakePipeline(self)

    def hmget(self, key, fields):
        h = self.hashes.get(key, {})
        return [h.get(f) for f in fields]

    def pfcount(self, key):
        return len(self.hll.get(key, set()))


class FakePipeline:
    def __init__(self, r: FakeRedis):
        self.r = r
        self.ops: list = []

    def hincrby(self, key, field, amount):
        self.ops.append(("hincrby", key, field, amount))

    def hset(self, key, field, value):
        self.ops.append(("hset", key, field, value))

    def pfadd(self, key, value):
        self.ops.append(("pfadd", key, value))

    def expire(self, key, ttl):
        pass

    def hmget(self, key, fields):
        self.ops.append(("hmget", key, tuple(fields)))

    def pfcount(self, key):
        self.ops.append(("pfcount", key))

    def execute(self):
        results = []
        for op in self.ops:
            if op[0] == "hincrby":
                h = self.r.hashes.setdefault(op[1], {})
                h[op[2]] = int(h.get(op[2], 0)) + op[3]
            elif op[0] == "hset":
                self.r.hashes.setdefault(op[1], {})[op[2]] = op[3]
            elif op[0] == "pfadd":
                self.r.hll.setdefault(op[1], set()).add(op[2])
            elif op[0] == "hmget":
                h = self.r.hashes.get(op[1], {})
                results.append([h.get(f) for f in op[2]])
            elif op[0] == "pfcount":
                results.append(len(self.r.hll.get(op[1], set())))
        self.ops = []
        return results


# ── Gate economy ────────────────────────────────────────────────────


def test_time_gate_blocks_recent_dream():
    r = FakeRedis()
    gate.record_completion(r, "shared", now=time.time() - 3600)  # 1h ago
    decision = gate.check_time_gate(r, "shared", min_hours=24)
    assert not decision.proceed
    assert "time:" in decision.reason


def test_time_gate_passes_when_stale_or_never():
    r = FakeRedis()
    assert gate.check_time_gate(r, "never-dreamt", min_hours=24).proceed
    gate.record_completion(r, "shared", now=time.time() - 100 * 3600)
    assert gate.check_time_gate(r, "shared", min_hours=24).proceed


def test_volume_gate():
    assert not gate.check_volume_gate(5, min_new_memories=20).proceed
    assert gate.check_volume_gate(20, min_new_memories=20).proceed


def test_lock_exclusive_and_released():
    r = FakeRedis()
    assert gate.acquire_lock(r, "shared")
    assert not gate.acquire_lock(r, "shared")  # held
    assert gate.is_locked(r, "shared")
    gate.release_lock(r, "shared")
    assert gate.acquire_lock(r, "shared")  # reacquirable


def test_completion_resets_time_gate():
    r = FakeRedis()
    gate.record_completion(r, "shared")
    state = gate.get_gate_state(r, "shared")
    assert state["last_dreamt_at"] > 0
    assert not gate.check_time_gate(r, "shared", min_hours=1).proceed


# ── Traces + scoring ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _stub_trace_redis(monkeypatch):
    """Route the traces module at a fresh FakeRedis per test."""
    from extensions.dreaming import traces as tr

    fake = FakeRedis()
    monkeypatch.setattr(tr, "_get_redis", lambda: fake)
    tr._test_fake_redis = fake
    yield
    if hasattr(tr, "_test_fake_redis"):
        del tr._test_fake_redis


def test_trace_write_then_read_via_stub():
    from extensions.dreaming import traces as tr

    fake = tr._get_redis()
    tr._write_trace(["a"], "q one", 30)
    tr._write_trace(["a"], "q two", 30)
    tr._write_trace(["a"], "q one", 30)  # duplicate query — HLL dedups
    agg = tr.read_aggregates(fake, ["a"])
    assert agg["a"]["recall_count"] == 3
    assert agg["a"]["unique_query_count"] == 2
    assert agg["a"]["last_recalled_at"] > 0


def test_promotion_score_weights_and_bounds():
    now = time.time()
    zero = scoring.promotion_score(now=now)
    assert zero == 0.0
    hot = scoring.promotion_score(
        relevance=1.0, recall_count=50, unique_query_count=20,
        last_recalled_at=now, created_at=now - 86400,
        merged_source_count=10, concept_count=5, now=now,
    )
    assert 0.9 < hot <= 1.0
    # recency decays
    cold = scoring.promotion_score(
        relevance=1.0, recall_count=50, unique_query_count=20,
        last_recalled_at=now - 90 * 86400, created_at=now - 100 * 86400,
        merged_source_count=10, concept_count=5, now=now,
    )
    assert cold < hot


def test_retention_strength_decays_and_reinforces():
    now = time.time()
    fresh = scoring.retention_strength(
        base_confidence=0.9, recall_count=0,
        last_recalled_at=0, created_at=now, now=now,
    )
    old = scoring.retention_strength(
        base_confidence=0.9, recall_count=0,
        last_recalled_at=0, created_at=now - 180 * 86400, now=now,
    )
    assert old < fresh
    recalled_old = scoring.retention_strength(
        base_confidence=0.9, recall_count=10,
        last_recalled_at=now - 5 * 86400, created_at=now - 180 * 86400, now=now,
    )
    assert recalled_old > old  # recall reinforces + re-anchors decay


# ── Prompt JSON parsing ─────────────────────────────────────────────


def test_parse_json_response_tolerates_fences_and_prose():
    raw = 'Sure! Here you go:\n```json\n{"actions": [{"type": "merge"}]}\n```'
    assert prompts.parse_json_response(raw, key="actions") == [{"type": "merge"}]
    assert prompts.parse_json_response("garbage", key="actions") == []
    assert prompts.parse_json_response('{"actions": "not a list"}', key="actions") == []
    assert prompts.parse_json_response("", key="actions") == []


# ── Consolidation: decide validation + posture split ────────────────


def _batch(memories: list[dict]) -> PoolBatch:
    return PoolBatch(
        pool="shared", group_id="shared", visibility="shared",
        owner_user_id=None, project_id=None, memories=memories,
    )


@pytest.mark.asyncio
async def test_decide_validates_ids_and_shapes():
    mems = [
        {"memory_id": "a", "content": "x", "created_at": "2026-07-01T00:00:00+00:00"},
        {"memory_id": "b", "content": "y", "created_at": "2026-07-02T00:00:00+00:00"},
    ]
    llm = AsyncMock(return_value=json.dumps({"actions": [
        {"type": "merge", "memory_ids": ["a", "b"], "survivor_id": "a",
         "content": "merged", "confidence": 0.9},
        {"type": "merge", "memory_ids": ["a"], "survivor_id": "a",
         "content": "solo merge is invalid", "confidence": 0.9},
        {"type": "prune", "memory_ids": ["ghost"], "confidence": 0.99},
        {"type": "rewrite", "memory_ids": ["b"], "content": "", "confidence": 0.9},
        {"type": "explode", "memory_ids": ["a"], "confidence": 1.0},
        {"type": "invalidate", "memory_ids": ["a"],
         "superseded_by_id": "b", "confidence": "high"},
    ]}))
    actions = await consolidate.decide(_batch(mems), llm)
    types = [a["type"] for a in actions]
    assert types == ["merge", "invalidate"]
    assert actions[1]["confidence"] == 0.0  # unparseable → 0.0


def test_split_by_posture_hybrid():
    actions = [
        {"type": "merge", "confidence": 0.1},            # reversible → apply
        {"type": "rewrite", "confidence": 0.0},          # reversible → apply
        {"type": "invalidate", "confidence": 0.9},       # destructive, high → apply
        {"type": "invalidate", "confidence": 0.5},       # destructive, low → report
        {"type": "prune", "confidence": 0.2,
         "contains_secret": True},                        # secret → always apply
        {"type": "prune", "confidence": 0.2},            # low prune → report
    ]
    to_apply, to_report = consolidate.split_by_posture(
        actions, auto_apply_confidence=0.85
    )
    assert len(to_apply) == 4
    assert len(to_report) == 2
    assert all(a["type"] in ("invalidate", "prune") for a in to_report)


@pytest.mark.asyncio
async def test_apply_actions_dry_run_touches_nothing():
    service = MagicMock()
    batch = _batch([{"memory_id": "a", "content": "x"}])
    result = await consolidate.apply_actions(
        service, batch,
        [{"type": "prune", "memory_ids": ["a"], "confidence": 0.9}],
        dry_run=True,
    )
    assert result.applied and result.applied[0]["dry_run"] is True
    service.delete_memory.assert_not_called()
    assert not service.mock_calls  # zero service interaction in dry-run


# ── LIGHT staging ───────────────────────────────────────────────────


def test_stage_pool_guards_and_gates(monkeypatch):
    from extensions.dreaming import consolidate as c

    monkeypatch.setattr(
        c, "read_aggregates",
        lambda redis, ids: {i: {"recall_count": 0, "unique_query_count": 0,
                                "last_recalled_at": 0.0} for i in ids},
    )
    now = time.time()
    recent = "2026-07-04T00:00:00+00:00"
    ancient = "2024-01-01T00:00:00+00:00"
    batch = _batch([
        {"memory_id": "new1", "content": "n", "created_at": recent,
         "confidence": 0.9, "source_type": "conversation"},
        # dream-authored new memory: excluded (feedback-loop guard)
        {"memory_id": "dreamnew", "content": "d", "created_at": recent,
         "confidence": 0.9, "source_type": "dream"},
        # old but strong: not staged
        {"memory_id": "oldstrong", "content": "o", "created_at": ancient,
         "confidence": 0.9, "source_type": "conversation"},
    ])
    staged = c.stage_pool(
        batch, FakeRedis(), last_dreamt_at=now - 86400,
        max_memories=100, strength_half_life_days=100000,  # no decay
        prune_strength_threshold=0.15,
    )
    ids = {m["memory_id"] for m in staged.memories}
    assert ids == {"new1"}
    # dream-authored writes count toward NEITHER staging NOR the volume
    # gate — a sweep's own insights must not trigger the next sweep.
    assert staged.new_count == 1


def test_enumerate_pools_excludes_standard_and_tombstoned():
    from types import SimpleNamespace

    points = [
        SimpleNamespace(id="s1", payload={
            "data": "shared fact", "user_id": "alice", "created_at": "2026-07-01",
            "metadata": {"visibility": "shared", "category": "decision"}}),
        SimpleNamespace(id="p1", payload={
            "data": "private fact", "user_id": "alice", "created_at": "2026-07-01",
            "metadata": {"visibility": "private", "owner_user_id": "alice",
                         "project_id": "proj", "category": "preference"}}),
        SimpleNamespace(id="std", payload={
            "data": "standard", "user_id": "boss",
            "metadata": {"visibility": "standard"}}),
        SimpleNamespace(id="tomb", payload={
            "data": "gone", "user_id": "alice",
            "metadata": {"visibility": "private", "dream_tombstoned": True}}),
    ]
    client = MagicMock()
    client.scroll.return_value = (points, None)
    service = MagicMock()
    service._memory.vector_store.client = client

    pools = consolidate.enumerate_pools(service)
    assert set(pools) == {"shared", "user--alice--project--proj"}
    assert pools["shared"].visibility == "shared"
    private = pools["user--alice--project--proj"]
    assert private.owner_user_id == "alice"
    assert private.project_id == "proj"
    all_ids = {m["memory_id"] for b in pools.values() for m in b.memories}
    assert "std" not in all_ids and "tomb" not in all_ids


# ── Reflection ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reflect_validates_citations_and_caps():
    mems = [
        {"memory_id": m, "content": f"mem {m}", "created_at": "2026-07-01",
         "category": "decision", "source_type": "conversation"}
        for m in ("a", "b", "c")
    ]
    llm = AsyncMock(return_value=json.dumps({"insights": [
        {"content": "Good insight.", "lens": "pattern",
         "category": "preference", "source_memory_ids": ["a", "b"],
         "confidence": 0.8},
        {"content": "Single-source — rejected.", "lens": "pattern",
         "category": "preference", "source_memory_ids": ["a"], "confidence": 0.9},
        {"content": "Unknown cites — rejected.", "lens": "failure",
         "category": "preference", "source_memory_ids": ["x", "y"], "confidence": 0.9},
        {"content": "Bad category falls back.", "lens": "weird",
         "category": "not_a_category", "source_memory_ids": ["b", "c"],
         "confidence": 2.5},
    ]}))
    out = await reflect.reflect(_batch(mems), llm, max_insights=5)
    assert len(out) == 2
    assert out[0]["content"] == "Good insight."
    assert out[1]["category"] == "domain_knowledge"  # fallback
    assert out[1]["lens"] == "pattern"               # fallback
    assert out[1]["confidence"] == 1.0               # clamped


@pytest.mark.asyncio
async def test_reflect_needs_substrate():
    mems = [{"memory_id": "a", "content": "x", "source_type": "conversation"}]
    llm = AsyncMock()
    assert await reflect.reflect(_batch(mems), llm, max_insights=5) == []
    llm.assert_not_awaited()


# ── Diary ───────────────────────────────────────────────────────────


def test_diary_render_and_write(tmp_path):
    entry = reflect.render_diary_entry(
        pool="shared", run_id="drm_test",
        applied=[{"type": "merge", "memory_ids": ["a", "b"], "reason": "dup"}],
        reported=[{"type": "prune", "memory_ids": ["c"], "confidence": 0.4,
                   "reason": "low strength"}],
        insights=[{"lens": "pattern", "content": "An insight.",
                   "source_memory_ids": ["a", "b"]}],
    )
    assert "### Reflections" in entry
    assert "### Consolidated" in entry
    assert "### Proposed (not applied — review)" in entry

    rel = reflect.write_diary(tmp_path, "shared", entry, source_memory_ids=["a", "b"])
    assert rel == "Dreams/shared.md"
    text = (tmp_path / "shared.md").read_text()
    assert "source_memory_ids: [a, b]" in text
    assert "An insight." in text

    # second sweep prepends, keeps the first entry
    entry2 = reflect.render_diary_entry(
        pool="shared", run_id="drm_two", applied=[], reported=[], insights=[]
    )
    reflect.write_diary(tmp_path, "shared", entry2, source_memory_ids=["a", "b", "c"])
    text2 = (tmp_path / "shared.md").read_text()
    assert "Quiet night" in text2
    assert "An insight." in text2
    assert text2.index("Quiet night") < text2.index("An insight.")  # newest first


# ── pool_key ────────────────────────────────────────────────────────


def test_pool_key_shapes():
    assert consolidate.pool_key(visibility="shared", owner_user_id=None, project_id=None) == "shared"
    assert consolidate.pool_key(visibility="shared", owner_user_id=None, project_id="p") == "shared--project--p"
    assert consolidate.pool_key(visibility="private", owner_user_id="u", project_id=None) == "user--u"
    assert consolidate.pool_key(visibility="private", owner_user_id="u", project_id="p") == "user--u--project--p"
