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
    """Minimal in-memory stand-in for the sync redis client.

    Tracks ``px`` expiries with a monotonic-ish clock so the lock's
    stale-reclaim path (``SET NX PX``) is actually testable.
    """

    def __init__(self):
        self.kv: dict[str, str] = {}
        self.expiries: dict[str, float] = {}  # key → unix deadline
        self.hashes: dict[str, dict] = {}
        self.hll: dict[str, set] = {}

    def _alive(self, key) -> bool:
        if key not in self.kv:
            return False
        deadline = self.expiries.get(key)
        if deadline is not None and time.time() >= deadline:
            self.kv.pop(key, None)
            self.expiries.pop(key, None)
            return False
        return True

    def get(self, key):
        return self.kv.get(key) if self._alive(key) else None

    def set(self, key, value, nx=False, px=None):
        if nx and self._alive(key):
            return False
        self.kv[key] = value
        if px is not None:
            self.expiries[key] = time.time() + px / 1000.0
        else:
            self.expiries.pop(key, None)
        return True

    def delete(self, key):
        self.kv.pop(key, None)
        self.expiries.pop(key, None)

    def exists(self, key):
        return self._alive(key)

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
    token = gate.acquire_lock(r, "shared")
    assert token
    assert gate.acquire_lock(r, "shared") is None  # held
    assert gate.is_locked(r, "shared")
    gate.release_lock(r, "shared", token)
    assert gate.acquire_lock(r, "shared")  # reacquirable


def test_lock_release_is_owner_safe():
    """A stale holder's release must not clear a reacquired lock."""
    r = FakeRedis()
    stale_token = gate.acquire_lock(r, "shared")
    r.expiries["dreaming:lock:shared"] = 0.0  # simulate the PX window lapsing
    fresh_token = gate.acquire_lock(r, "shared")  # reclaimed by another worker
    assert fresh_token and fresh_token != stale_token

    gate.release_lock(r, "shared", stale_token)  # zombie sweep finishes late
    assert gate.is_locked(r, "shared")  # fresh owner keeps the lock

    gate.release_lock(r, "shared", fresh_token)
    assert not gate.is_locked(r, "shared")


def test_lock_stale_reclaim_via_px_expiry(monkeypatch):
    """A crashed sweep's lock is reclaimable once the PX window lapses."""
    monkeypatch.setattr(gate, "_LOCK_STALE_MS", 1)
    r = FakeRedis()
    assert gate.acquire_lock(r, "shared")
    time.sleep(0.005)  # outlive the 1ms stale window; no release_lock call
    assert gate.acquire_lock(r, "shared")


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


# ── Librarian (humane vault) ────────────────────────────────────────


def _lib_batch(memories, *, visibility="shared", project_id="scope", owner=None):
    return PoolBatch(
        pool="p", group_id="p", visibility=visibility,
        owner_user_id=owner, project_id=project_id, memories=memories,
    )


def test_pool_dir_routing(tmp_path):
    from extensions.dreaming import librarian as lib

    shared_proj = _lib_batch([], visibility="shared", project_id="scope")
    assert lib.pool_dir(tmp_path, shared_proj, "ehfaz") == tmp_path / "Projects" / "scope"
    shared_global = _lib_batch([], visibility="shared", project_id=None)
    assert lib.pool_dir(tmp_path, shared_global, "ehfaz") == tmp_path / "Knowledge"
    mine = _lib_batch([], visibility="private", project_id=None, owner="ehfaz")
    assert lib.pool_dir(tmp_path, mine, "ehfaz") == tmp_path / "Me"
    theirs = _lib_batch([], visibility="private", project_id=None, owner="someone-else")
    assert lib.pool_dir(tmp_path, theirs, "ehfaz") is None  # never in my vault


def test_slug_title():
    from extensions.dreaming.librarian import _slug_title

    assert _slug_title("TURN & ICE Connectivity") == "TURN and ICE Connectivity"
    assert _slug_title("LoRA: Restyling!") == "LoRA Restyling"
    assert _slug_title("///") == "Untitled"


@pytest.mark.asyncio
async def test_librarian_writes_topics_hub_and_home(tmp_path):
    from extensions.dreaming import librarian as lib

    mems = [
        {"memory_id": "a", "content": "TURN server DNS is broken on RunPod",
         "category": "architecture", "created_at": "2026-07-01"},
        {"memory_id": "b", "content": "Cloudflare TURN is the fallback path",
         "category": "architecture", "created_at": "2026-07-02"},
        {"memory_id": "c", "content": "LoRA restyle needs trigger words",
         "category": "procedure", "created_at": "2026-07-03"},
        {"memory_id": "d", "content": "Use runtime_peft for live LoRA scale",
         "category": "procedure", "created_at": "2026-07-03"},
    ]
    calls = {"n": 0}

    async def llm(prompt):
        calls["n"] += 1
        if "librarian of a personal knowledge vault" in prompt:
            return json.dumps({"topics": [
                {"title": "TURN & ICE Connectivity", "summary": "How TURN works here.",
                 "memory_ids": ["a", "b"]},
                {"title": "LoRA Restyling", "summary": "Restyle workflow.",
                 "memory_ids": ["c", "d"]},
                {"title": "Stray", "summary": "too few ids", "memory_ids": ["a"]},
            ]})
        return "A narrative body linking [[LoRA Restyling]]."

    out = await lib.update_vault(
        _lib_batch(mems), llm, vault=tmp_path, operator_user_id="ehfaz", dry_run=False,
    )
    assert out["pages_written"] == 2
    tdir = tmp_path / "Projects" / "scope"
    page = (tdir / "TURN and ICE Connectivity.md").read_text()
    assert "source_memory_ids: [a, b]" in page
    assert "# TURN & ICE Connectivity" in page
    assert "[[scope]]" in page                      # hub backlink
    hub = (tdir / "scope.md").read_text()
    assert "[[TURN and ICE Connectivity|TURN & ICE Connectivity]]" in hub
    home = (tmp_path / "Home.md").read_text()
    assert "[[scope]] (2 topics)" in home

    # idempotent second pass: same id sets → all skipped, no LLM merges
    before = calls["n"]
    out2 = await lib.update_vault(
        _lib_batch(mems), llm, vault=tmp_path, operator_user_id="ehfaz", dry_run=False,
    )
    assert out2["pages_written"] == 0
    assert out2["pages_skipped"] == 2
    assert calls["n"] == before + 1  # only the cluster call, zero merges


@pytest.mark.asyncio
async def test_librarian_dry_run_writes_nothing(tmp_path):
    from extensions.dreaming import librarian as lib

    async def llm(prompt):
        if "librarian" in prompt:
            return json.dumps({"topics": [{"title": "T", "summary": "s",
                                           "memory_ids": ["a", "b"]}]})
        return "body"

    mems = [{"memory_id": i, "content": f"c{i}", "category": "decision"} for i in "ab"]
    out = await lib.update_vault(
        _lib_batch(mems), llm, vault=tmp_path, operator_user_id="e", dry_run=True,
    )
    assert out["pages_written"] == 1
    assert not (tmp_path / "Projects").exists()
