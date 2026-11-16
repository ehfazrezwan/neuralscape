"""Semantic dedup and the dream sweep never work the same pool at once.

(a) With dreaming on, ``dedup_memories`` HOLDS a pool's dream lock (the sweep's own
key) while it works on that pool's rows, skips a pool it cannot lock, releases the
lock afterwards (also on error), and skips the whole semantic phase when Redis is
down. The exact-hash phase is unaffected. (b) ``apply_actions`` re-reads the
tombstone state of every row an action writes to right before applying it and skips
the action (recorded, not an error) when any was tombstoned after staging. (c) The
skips reach the pool report, the run totals and one INFO line per sweep. (d) The pool
key dedup locks is the key the sweep's pool enumeration produces.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from extensions.dreaming import consolidate, gate, sweep
from extensions.dreaming.config import dreaming_settings
from memory.delete import _dream_pool_of
from tests.test_dedup import _FakeQdrant, _deleted, _row, _tombstoned, service  # noqa: F401
from tests.test_dreaming_rewrite_atomic import _batch, _service

TEXT = "Deploys go through the staging gate before prod"
OTHER = "The nightly report job runs at 02:00 UTC on weekdays"


class _FakeRedis:
    def __init__(self):
        self.keys: dict[str, str] = {}
        self.fail = False

    def ping(self):
        if self.fail:
            raise ConnectionError("redis down")
        return True

    def set(self, key, value, nx=False, px=None):
        if nx and key in self.keys:
            return None
        self.keys[key] = value
        return True

    def get(self, key):
        return self.keys.get(key)

    def delete(self, key):
        self.keys.pop(key, None)

    def exists(self, key):
        return key in self.keys

    def pexpire(self, key, ms):
        return 1 if key in self.keys else 0


@pytest.fixture
def dream_on_tombstone(monkeypatch):
    from config import settings

    monkeypatch.setattr(dreaming_settings, "enabled", True)
    monkeypatch.setattr(settings, "dedup_semantic_tombstone_enabled", True)


@pytest.fixture
def redis(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr("extensions.dreaming.sweep._get_redis", lambda: r)
    return r


def _pair(service, a="old", b="new", text=TEXT):
    return _FakeQdrant(service, [
        _row(a, text, "2025-01-01T00:00:00", scope="global", visibility="private"),
        _row(b, text, "2025-06-01T00:00:00", scope="global", visibility="private"),
    ])


POOL = "user--u1"


@pytest.mark.usefixtures("dream_on_tombstone")
class TestSemanticPhaseHoldsTheDreamLock:
    def test_the_sweeps_key_is_held_while_the_pool_is_worked_and_released_after(
        self, service, redis
    ):
        fake = _pair(service)
        seen = {}
        inner = service._memory.vector_store.search.side_effect

        def spy(*a, **kw):
            seen["held"] = f"dreaming:lock:{POOL}" in redis.keys
            seen["sweep_can_take_it"] = gate.acquire_lock(redis, POOL)
            return inner(*a, **kw)

        service._memory.vector_store.search.side_effect = spy

        result = service.dedup_memories("u1")

        assert seen["held"] is True
        assert seen["sweep_can_take_it"] is None  # the sweep would report "locked"
        assert result["semantic_duplicates_removed"] == 1
        assert _tombstoned(service) == ["old"]
        assert redis.keys == {}  # released after the pass
        assert fake.payloads["new"]["metadata"].get("dream_tombstoned") is None

    def test_lock_taken_by_the_sweep_mid_run_stops_tombstones_in_that_pool(
        self, service, redis, monkeypatch
    ):
        from config import settings

        monkeypatch.setattr(settings, "dedup_batch_size", 2)
        rows = [
            _row("a_old", TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("a_new", TEXT, "2025-06-01T00:00:00", scope="global", visibility="private"),
            _row("b_old", OTHER, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("b_new", OTHER, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ]
        _FakeQdrant(service, rows)
        pages = type(service)._iter_user_memory_pages
        acquired = []
        inner_set = redis.set
        redis.set = lambda k, v, nx=False, px=None: (
            acquired.append(k), inner_set(k, v, nx=nx, px=px))[1]
        sweep_locked = []

        def pages_then_sweep_locks(self_, *a, **kw):
            for page in pages(self_, *a, **kw):
                # the sweep takes the pool once page 1 of the semantic pass was deduped
                if acquired and not sweep_locked:
                    sweep_locked.append(gate.acquire_lock(redis, POOL))
                    assert sweep_locked[0]
                yield page

        monkeypatch.setattr(type(service), "_iter_user_memory_pages", pages_then_sweep_locks)

        result = service.dedup_memories("u1")

        assert _tombstoned(service) == ["a_old"]  # page 1 only; page 2 left to the sweep
        assert result["semantic_skipped_locked_pools"] == 1
        assert f"dreaming:lock:{POOL}" in redis.keys  # the sweep's lock is untouched

    def test_pool_locked_before_the_run_is_skipped_and_reported(self, service, redis, caplog):
        _pair(service)
        assert gate.acquire_lock(redis, POOL)

        with caplog.at_level("INFO"):
            result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 0
        assert result["semantic_skipped_locked_pools"] == 1
        assert _tombstoned(service) == [] and _deleted(service) == []
        assert any("semantic_skipped_locked_pools=1" in r.getMessage() for r in caplog.records)

    def test_another_pools_lock_does_not_block(self, service, redis):
        _pair(service)
        assert gate.acquire_lock(redis, "user--someone-else")

        assert service.dedup_memories("u1")["semantic_duplicates_removed"] == 1

    def test_lock_is_released_when_the_pass_errors(self, service, redis, monkeypatch):
        _pair(service)
        monkeypatch.setattr(
            type(service), "_fetch_dense_vectors",
            lambda self_, ids: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError):
            service.dedup_memories("u1")

        assert redis.keys == {}

    def test_lock_has_a_bounded_expiry(self, service, redis):
        _pair(service)
        calls = []
        inner = redis.set
        redis.set = lambda key, value, nx=False, px=None: (calls.append(px), inner(key, value, nx=nx, px=px))[1]

        service.dedup_memories("u1")

        assert calls and all(px and px <= 15 * 60 * 1000 for px in calls)

    def test_redis_down_skips_the_semantic_phase_and_logs_once(self, service, redis, caplog):
        _pair(service)
        redis.fail = True

        with caplog.at_level("INFO"):
            result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 0
        assert "unavailable" in result["semantic_skipped"]
        assert _tombstoned(service) == [] and _deleted(service) == []
        assert sum("dream lock unavailable" in r.getMessage() and r.levelno == logging.WARNING
                   for r in caplog.records) == 1

    def test_redis_down_does_not_stop_the_exact_phase(self, service, redis):
        fake = _FakeQdrant(service, [
            _row("e1", TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("e2", TEXT, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])
        for pid in ("e1", "e2"):
            fake.payloads[pid]["hash"] = "same-hash"
        redis.fail = True

        result = service.dedup_memories("u1")

        assert result["exact_duplicates_removed"] == 1 and _deleted(service) == ["e1"]

    def test_exact_phase_skips_a_pool_that_is_locked_and_reports_it(self, service, redis):
        fake = _FakeQdrant(service, [
            _row("e1", TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("e2", TEXT, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])
        for pid in ("e1", "e2"):
            fake.payloads[pid]["hash"] = "same-hash"
        assert gate.acquire_lock(redis, POOL)

        result = service.dedup_memories("u1")

        assert result["exact_duplicates_removed"] == 0 and _deleted(service) == []
        assert result["exact_skipped_locked_pools"] == 1

    def test_row_tombstoned_between_page_read_and_lock_is_left_alone(
        self, service, redis, monkeypatch
    ):
        fake = _pair(service)
        orig = type(service)._live_rows

        def sweep_got_there_first(self_, rows):
            fake.payloads["old"].setdefault("metadata", {})["dream_tombstoned"] = True
            return orig(self_, rows)

        monkeypatch.setattr(type(service), "_live_rows", sweep_got_there_first)

        service.dedup_memories("u1")

        assert _tombstoned(service) == []  # nothing re-tombstoned from the stale page

    def test_neighbour_in_another_pool_is_never_merged(self, service, redis):
        # Same dedup partition (user, scope, project, visibility) but a different dream
        # pool: the row's owner_user_id differs. The lock covers one pool, so the pair
        # is left alone.
        _FakeQdrant(service, [
            _row("mine", TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("theirs", TEXT, "2025-06-01T00:00:00", scope="global", visibility="private",
                 owner_user_id="u9"),
        ])

        result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 0 and _tombstoned(service) == []

    def test_dreaming_off_with_redis_down_runs_unlocked(self, service, redis, monkeypatch):
        _pair(service)
        redis.fail = True
        monkeypatch.setattr(dreaming_settings, "enabled", False)

        assert service.dedup_memories("u1")["semantic_duplicates_removed"] == 1


# ── apply_actions: every action re-checks tombstones ────────────────────────────


ACTIONS = {
    "merge": {"type": "merge", "memory_ids": ["s", "l"], "survivor_id": "s",
              "content": "merged fact", "confidence": 0.9},
    "rewrite": {"type": "rewrite", "memory_ids": ["l"], "content": "new text", "confidence": 0.9},
    "temporal_reframe": {"type": "temporal_reframe", "memory_ids": ["l"],
                         "content": "was true in 2024", "confidence": 0.9},
    "invalidate": {"type": "invalidate", "memory_ids": ["l"], "superseded_by_id": "s",
                   "confidence": 0.9},
    "prune": {"type": "prune", "memory_ids": ["l"], "confidence": 0.9},
}


def _payloads(tombstone: str | None):
    p = {
        "s": {"data": "survivor text", "metadata": {}},
        "l": {"data": "loser text", "metadata": {}},
    }
    if tombstone:
        p[tombstone]["metadata"] = {"dream_tombstoned": True, "superseded_by": "dedup-survivor"}
    return p


class TestEveryActionSkipsRowsTombstonedSinceStaging:
    async def _apply(self, action, tombstone):
        service, client, _ = _service(_payloads(tombstone), bm25_slot=False)
        result = await consolidate.apply_actions(
            service, _batch(["s", "l"]), [dict(ACTIONS[action])], dry_run=False
        )
        return result, client

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", sorted(ACTIONS))
    async def test_tombstoned_target_skips_the_action_and_writes_nothing(self, action):
        result, client = await self._apply(action, "l")

        assert not result.errors and not result.applied
        assert len(result.skipped) == 1
        assert result.skipped[0]["type"] == action
        assert "tombstoned since staging" in result.skipped[0]["skip_reason"]
        assert "l" in result.skipped[0]["skip_reason"]
        assert client.calls == []  # nothing was written
        assert client.payloads["l"]["data"] == "loser text"
        assert client.payloads["l"]["metadata"]["superseded_by"] == "dedup-survivor"

    @pytest.mark.asyncio
    async def test_tombstoned_merge_survivor_skips_the_merge(self):
        result, client = await self._apply("merge", "s")

        assert not result.errors and not result.applied and len(result.skipped) == 1
        assert "s" in result.skipped[0]["skip_reason"]
        assert client.calls == []
        assert client.payloads["l"]["data"] == "loser text"
        assert "dream_tombstoned" not in client.payloads["l"]["metadata"]

    @pytest.mark.asyncio
    async def test_double_wrapped_tombstone_is_seen(self):
        service, client, _ = _service(
            {"s": {"data": "a", "metadata": {}},
             "l": {"data": "b", "metadata": {"metadata": {"dream_tombstoned": True}}}},
            bm25_slot=False,
        )
        result = await consolidate.apply_actions(
            service, _batch(["s", "l"]), [dict(ACTIONS["merge"])], dry_run=False
        )
        assert not result.applied and len(result.skipped) == 1 and client.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", sorted(ACTIONS))
    async def test_live_rows_still_apply(self, action):
        result, client = await self._apply(action, None)

        assert not result.errors and not result.skipped and len(result.applied) == 1
        assert client.calls  # something was written

    @pytest.mark.asyncio
    async def test_secret_prune_is_exempt_and_still_deletes(self):
        service, client, _ = _service(_payloads("l"), bm25_slot=False)
        service.delete_memory = MagicMock(return_value={})
        act = {**ACTIONS["prune"], "contains_secret": True, "confidence": 0.99}
        result = await consolidate.apply_actions(
            service, _batch(["s", "l"]), [act], dry_run=False
        )

        assert not result.skipped and len(result.applied) == 1
        service.delete_memory.assert_called_once_with("l")


# ── skips are visible: pool report, run totals, one INFO line ───────────────────


class TestSkipsAreReported:
    @pytest.mark.asyncio
    async def test_pool_report_carries_the_skips_without_memory_text(self, tmp_path, monkeypatch):
        from tests.test_dreaming_safety import FakeRedis, _fresh_batch, _sweep_settings

        skipped = [{"type": "merge", "memory_ids": ["m0", "m1"], "content": "SECRET TEXT",
                    "skip_reason": "tombstoned since staging: m1"}]
        monkeypatch.setattr(
            consolidate, "apply_actions",
            AsyncMock(return_value=consolidate.ActionResult(skipped=skipped)),
        )
        report = await sweep._dream_pool(
            service=MagicMock(), settings=_sweep_settings(tmp_path), redis=FakeRedis(),
            llm_call=AsyncMock(return_value='{"actions": []}'), batch=_fresh_batch(),
            dry_run=False, force=False,
        )

        assert report.skipped == 1
        assert report.skip_reasons == ["merge: tombstoned since staging: m1"]
        assert "SECRET TEXT" not in repr(report)

    def test_run_totals_sum_the_skips(self):
        run = sweep.DreamRun(run_id="r", started_at="t", pools=[
            sweep.PoolReport(pool="a", status="dreamt", skipped=2, skip_reasons=["x", "y"]),
            sweep.PoolReport(pool="b", status="dreamt", skipped=1, skip_reasons=["z"]),
        ])
        assert run.to_dict()["totals"]["actions_skipped"] == 3

    @pytest.mark.asyncio
    async def test_one_info_line_per_sweep_with_count_and_reasons(self, monkeypatch, caplog):
        from extensions.dreaming.config import DreamingSettings

        settings = DreamingSettings(_env_file=None, enabled=True)
        redis = _FakeRedis()
        redis.set = lambda *a, **k: True
        monkeypatch.setattr(sweep, "_get_redis", lambda: redis)
        monkeypatch.setattr(sweep, "_make_llm_call", AsyncMock(return_value=None))
        monkeypatch.setattr(
            consolidate, "enumerate_pools",
            lambda service: {"p1": SimpleNamespace(), "p2": SimpleNamespace()},
        )
        reports = iter([
            sweep.PoolReport(pool="p1", status="dreamt", skipped=2, skip_reasons=[
                "merge: tombstoned since staging: a", "rewrite: tombstoned since staging: b"]),
            sweep.PoolReport(pool="p2", status="dreamt", skipped=1, skip_reasons=[
                "merge: tombstoned since staging: c"]),
        ])
        monkeypatch.setattr(sweep, "_dream_pool", AsyncMock(side_effect=lambda **kw: next(reports)))

        with caplog.at_level("INFO"):
            await sweep.dream_all(service=MagicMock(), settings=settings, dry_run=True)

        lines = [r for r in caplog.records if "skipped" in r.getMessage() and r.levelno == logging.INFO
                 and "staged action" in r.getMessage()]
        assert len(lines) == 1
        msg = lines[0].getMessage()
        assert "skipped 3 staged action" in msg
        assert "merge: tombstoned since staging': 2" in msg
        assert "rewrite: tombstoned since staging': 1" in msg
        assert ": a" not in msg and ": c" not in msg


# ── the key dedup locks is the key the sweep locks ──────────────────────────────

# (payload user_id, metadata) -> the pool key, written out from the documented
# group_id table (memory/groups.py::_build_group_id), not derived with the function.
KEY_CASES = [
    ("u1", {}, "user--u1"),
    ("u1", {"visibility": "private", "project_id": "p1"}, "user--u1--project--p1"),
    ("u1", {"visibility": "private", "project_id": "p2"}, "user--u1--project--p2"),
    ("u2", {"visibility": "private", "project_id": "p1"}, "user--u2--project--p1"),
    ("u1", {"visibility": "private", "workspace": "ref"}, "user--u1--ws--ref"),
    ("u1", {"visibility": "private", "project_id": "p1", "workspace": "ref"},
     "user--u1--project--p1--ws--ref"),
    ("u1", {"visibility": "private", "workspace": "memory"}, "user--u1"),
    ("u1", {"visibility": "private", "owner_user_id": "u7"}, "user--u7"),
    ("u1", {"visibility": "shared"}, "shared"),
    ("u2", {"visibility": "shared"}, "shared"),
    ("u1", {"visibility": "shared", "project_id": "p1"}, "shared--project--p1"),
    ("u1", {"visibility": "shared", "workspace": "ref"}, "shared--ws--ref"),
]


class TestPoolKeyMatchesTheSweeps:
    @pytest.mark.parametrize("user,meta,expected", KEY_CASES)
    def test_key_is_the_literal_expected_string(self, user, meta, expected):
        assert _dream_pool_of({"user_id": user, "metadata": meta}) == expected

    def test_rows_differing_in_one_dimension_get_distinct_keys(self):
        keys = {_dream_pool_of({"user_id": u, "metadata": m}) for u, m, _ in KEY_CASES}
        # 12 cases, two collapse onto an existing key by design (workspace "memory" is the
        # default workspace; every shared row of one project/workspace is one pool).
        assert len(keys) == len(KEY_CASES) - 2

    def test_standard_tier_is_never_a_dream_pool(self):
        assert _dream_pool_of({"user_id": "u1", "metadata": {"visibility": "standard"}}) is None

    def test_double_wrapped_metadata_is_unwrapped(self):
        payload = {"user_id": "u1", "metadata": {"metadata": {"project_id": "p1"}}}
        assert _dream_pool_of(payload) == "user--u1--project--p1"

    def test_equals_what_the_sweeps_pool_enumeration_produces(self):
        points = [
            SimpleNamespace(id=f"m{i}", payload={
                "user_id": user, "created_at": "2025-01-01T00:00:00",
                "metadata": dict(meta),
            })
            for i, (user, meta, _) in enumerate(KEY_CASES)
        ]
        points.append(SimpleNamespace(id="std", payload={
            "user_id": "u1", "metadata": {"visibility": "standard"}}))
        svc = MagicMock()
        svc._memory.vector_store.client.scroll.return_value = (points, None)

        pools = consolidate.enumerate_pools(svc)

        by_row = {m["memory_id"]: key for key, b in pools.items() for m in b.memories}
        assert "std" not in by_row
        for i, (user, meta, expected) in enumerate(KEY_CASES):
            assert by_row[f"m{i}"] == expected == _dream_pool_of(
                {"user_id": user, "metadata": meta}
            )
