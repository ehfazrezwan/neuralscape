"""Dedup's destructive phases and the dream sweep's pool lock: ownership, scope, freshness.

(1) Before every delete or tombstone the pass verifies the pool lock is still its own
(compare-and-renew); a lock that expired and was taken by a sweep stops all writes in
that pool, and the pool is reported. (2) The lock is taken whenever Redis answers,
dreaming on or off (a forced sweep runs with dreaming off). (3) After the lock is
taken only the re-read rows are used, so a sweep merge's summed counter survives.
(4) The exact-hash phase locks the pools of a duplicate group, re-reads the group and
deletes only rows that are still exact duplicates; a page without a duplicate group
never touches Redis. (5) ``gate.refresh_lock`` semantics.
"""

from __future__ import annotations

import copy
import logging

import pytest

from config import settings
from extensions.dreaming import consolidate, gate
from extensions.dreaming.config import dreaming_settings
from tests.test_dedup import _FakeQdrant, _deleted, _make_point, _row, _tombstoned, service  # noqa: F401
from tests.test_dedup_dream_exclusion import _FakeRedis, dream_on_tombstone  # noqa: F401
from tests.test_dreaming_rewrite_atomic import _batch, _service

A = "Deploys go through the staging gate before prod"
B = "The cache layer uses Redis with a five minute TTL"
POOL = "user--u1"
MIN = 60 * 1000


class _ClockRedis(_FakeRedis):
    """Fake Redis whose keys expire against a manual clock (ms) and count round trips."""

    def __init__(self):
        super().__init__()
        self.now = 0
        self.exp: dict[str, int] = {}
        self.pings = 0
        self.sets = 0

    def _gc(self):
        for k, t in list(self.exp.items()):
            if self.now >= t:
                self.keys.pop(k, None)
                self.exp.pop(k, None)

    def ping(self):
        self.pings += 1
        return super().ping()

    def set(self, key, value, nx=False, px=None):
        self._gc()
        self.sets += 1
        if nx and key in self.keys:
            return None
        self.keys[key] = value
        if px:
            self.exp[key] = self.now + px
        return True

    def get(self, key):
        self._gc()
        return self.keys.get(key)

    def exists(self, key):
        self._gc()
        return key in self.keys

    def pexpire(self, key, ms):
        self._gc()
        if key not in self.keys:
            return 0
        self.exp[key] = self.now + ms
        return 1

    def delete(self, key):
        self.keys.pop(key, None)
        self.exp.pop(key, None)


@pytest.fixture
def clock(monkeypatch):
    r = _ClockRedis()
    monkeypatch.setattr("extensions.dreaming.sweep._get_redis", lambda: r)
    return r


@pytest.fixture
def dreaming_off_defaults(monkeypatch):
    monkeypatch.setattr(dreaming_settings, "enabled", False)
    monkeypatch.setattr(settings, "dedup_semantic_tombstone_enabled", False)  # hard delete


def _copying_scroll(fake):
    """Real Qdrant returns a snapshot per scroll; the shared fake returns live dicts."""
    inner = fake.scroll

    def scroll(*a, **kw):
        pts, nxt = inner(*a, **kw)
        return [_make_point(p.id, copy.deepcopy(p.payload)) for p in pts], nxt

    return scroll


def _two_pairs(service):
    return _FakeQdrant(service, [
        _row("a_old", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
        _row("a_new", A, "2025-06-01T00:00:00", scope="global", visibility="private"),
        _row("b_old", B, "2025-01-01T00:00:00", scope="global", visibility="private"),
        _row("b_new", B, "2025-06-01T00:00:00", scope="global", visibility="private"),
    ])


# ── 1. lock ownership is verified before every write ────────────────────────────


@pytest.mark.usefixtures("dream_on_tombstone")
class TestLockLostMidPass:
    def test_lock_expiring_mid_page_stops_every_write_and_reports_the_pool(
        self, service, clock, caplog
    ):
        _two_pairs(service)
        inner = service._memory.vector_store.search.side_effect
        sweep = []

        def slow_search(*a, **kw):
            clock.now += 11 * MIN  # degraded Qdrant / embed fallback: slower than the expiry
            if not sweep:
                sweep.append(gate.acquire_lock(clock, POOL))  # the sweep's tick lands now
            return inner(*a, **kw)

        service._memory.vector_store.search.side_effect = slow_search

        with caplog.at_level("INFO"):
            result = service.dedup_memories("u1")

        assert sweep[0], "the sweep must have been able to take the expired lock"
        assert _tombstoned(service) == [] and _deleted(service) == []
        assert result["semantic_duplicates_removed"] == 0
        assert result["lock_lost_pools"] == 1
        assert clock.get(f"dreaming:lock:{POOL}") == sweep[0]  # the sweep's lock is untouched
        assert any("lock_lost_pools=1" in r.getMessage() for r in caplog.records)

    def test_a_lost_lock_abandons_the_pool_for_the_later_pages_too(
        self, service, clock, monkeypatch
    ):
        monkeypatch.setattr(settings, "dedup_batch_size", 2)
        _two_pairs(service)
        inner = service._memory.vector_store.search.side_effect
        calls = []

        def slow_search(*a, **kw):
            calls.append(1)
            clock.now += 11 * MIN
            gate.acquire_lock(clock, POOL)
            return inner(*a, **kw)

        service._memory.vector_store.search.side_effect = slow_search

        result = service.dedup_memories("u1")

        assert _tombstoned(service) == []
        assert result["lock_lost_pools"] == 1
        assert len(calls) == 1  # page 1 stopped at its first write check; page 2 never started

    def test_lock_still_ours_is_renewed_before_each_write(self, service, clock):
        _two_pairs(service)
        seen_ttls = []
        inner = clock.pexpire
        clock.pexpire = lambda key, ms: (seen_ttls.append(ms), inner(key, ms))[1]

        result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 2 and result["lock_lost_pools"] == 0
        assert len(seen_ttls) == 2 and all(ms <= 15 * MIN for ms in seen_ttls)
        assert clock.keys == {}  # released

    def test_exact_phase_stops_deleting_when_the_lock_is_lost(self, service, clock):
        fake = _FakeQdrant(service, [
            _row("x1", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("x2", A, "2025-03-01T00:00:00", scope="global", visibility="private"),
            _row("x3", A, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])
        for pid in ("x1", "x2", "x3"):
            fake.payloads[pid]["hash"] = "hA"
        inner = service._memory.vector_store.delete.side_effect
        sweep = []

        def delete_then_expire(pid):
            inner(pid)
            clock.now += 11 * MIN
            sweep.append(gate.acquire_lock(clock, POOL))

        service._memory.vector_store.delete.side_effect = delete_then_expire

        result = service.dedup_memories("u1", semantic=False)

        assert _deleted(service) == ["x2"]  # newest-first order: x2 went, x1 was never written
        assert "x1" in fake.payloads
        assert result["lock_lost_pools"] == 1 and result["exact_duplicates_removed"] == 1


# ── 2. the lock is taken whenever Redis answers ─────────────────────────────────


class TestLockWithDreamingOff:
    def test_a_forced_sweeps_pool_is_not_hard_deleted_from(
        self, service, clock, dreaming_off_defaults
    ):
        _FakeQdrant(service, [
            _row("a_old", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("a_new", A, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])
        assert gate.acquire_lock(clock, POOL)  # force=true sweep holds the pool

        result = service.dedup_memories("u1")

        assert _deleted(service) == [] and _tombstoned(service) == []
        assert result["semantic_duplicates_removed"] == 0
        assert result["semantic_skipped_locked_pools"] == 1

    def test_the_pass_holds_the_lock_while_it_works_and_serialises_with_a_sweep(
        self, service, clock, dreaming_off_defaults
    ):
        _FakeQdrant(service, [
            _row("a_old", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("a_new", A, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])
        inner = service._memory.vector_store.search.side_effect
        seen = {}

        def spy(*a, **kw):
            seen["held"] = f"dreaming:lock:{POOL}" in clock.keys
            seen["sweep"] = gate.acquire_lock(clock, POOL)
            return inner(*a, **kw)

        service._memory.vector_store.search.side_effect = spy

        result = service.dedup_memories("u1")

        assert seen == {"held": True, "sweep": None}
        assert result["semantic_duplicates_removed"] == 1 and _deleted(service) == ["a_old"]
        assert clock.keys == {}

    def test_redis_down_skips_the_page_instead_of_running_unlocked(self, service, clock, dreaming_off_defaults):
        _FakeQdrant(service, [
            _row("a_old", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("a_new", A, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])
        clock.fail = True

        res = service.dedup_memories("u1")

        assert res["semantic_duplicates_removed"] == 0 and _deleted(service) == []
        assert "unavailable" in res["semantic_skipped"]

    def test_one_lock_round_trip_per_page_with_duplicates(
        self, service, clock, dreaming_off_defaults
    ):
        _FakeQdrant(service, [
            _row("a_old", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("a_new", A, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])

        service.dedup_memories("u1")

        assert clock.sets == 1  # one SET NX for the one page


# ── 3. only the re-read rows are used after the lock ────────────────────────────


@pytest.mark.usefixtures("dream_on_tombstone")
def test_a_merges_summed_counter_survives_the_reread(service, clock, monkeypatch):
    fake = _FakeQdrant(service, [
        _row("x_old", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
        _row("y_new", A, "2025-06-01T00:00:00", scope="global", visibility="private"),
    ])
    service._memory.vector_store.client.scroll.side_effect = _copying_scroll(fake)
    orig = type(service)._live_rows

    def sweep_ran_before_lock(self_, rows):
        fake.payloads["x_old"]["metadata"]["times_derived"] = 3  # a merge summed into x_old
        return orig(self_, rows)

    monkeypatch.setattr(type(service), "_live_rows", sweep_ran_before_lock)

    service.dedup_memories("u1")

    assert _tombstoned(service) == ["x_old"]
    assert fake.payloads["y_new"]["metadata"].get("times_derived") == 4  # 1 + 3, not 1 + 0


@pytest.mark.usefixtures("dream_on_tombstone")
def test_a_row_deleted_between_read_and_lock_is_not_used(service, clock, monkeypatch):
    fake = _FakeQdrant(service, [
        _row("x_old", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
        _row("y_new", A, "2025-06-01T00:00:00", scope="global", visibility="private"),
    ])
    service._memory.vector_store.client.scroll.side_effect = _copying_scroll(fake)
    orig = type(service)._live_rows

    def gone(self_, rows):
        fake.payloads.pop("x_old")
        fake.vectors.pop("x_old")
        return orig(self_, rows)

    monkeypatch.setattr(type(service), "_live_rows", gone)

    assert service.dedup_memories("u1")["semantic_duplicates_removed"] == 0
    assert _tombstoned(service) == []


# ── 4. the exact-hash phase ─────────────────────────────────────────────────────


def _exact_rows(service, extra=()):
    fake = _FakeQdrant(service, [
        _row("x_old", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
        _row("c_new", A, "2025-06-01T00:00:00", scope="global", visibility="private"),
        *extra,
    ])
    fake.payloads["x_old"]["hash"] = fake.payloads["c_new"]["hash"] = "hA"
    return fake


class TestExactPhaseLocksAndRereads:
    def test_a_merge_landing_after_the_read_is_not_deleted(self, service, clock):
        fake = _exact_rows(service, [
            _row("l", B, "2025-03-01T00:00:00", scope="global", visibility="private"),
        ])
        service._memory.vector_store.client.scroll.side_effect = _copying_scroll(fake)
        pages = type(service)._iter_user_memory_pages
        n = {"calls": 0}

        def pages_then_sweep(self_, *a, **kw):
            n["calls"] += 1
            yield from pages(self_, *a, **kw)
            if n["calls"] == 2:  # the exact phase finished reading; the sweep MERGEs l into x_old
                fake.payloads["x_old"]["data"] = A + " Also: " + B
                fake.payloads["x_old"]["hash"] = "h-merged"
                fake.payloads["l"]["metadata"].update(dream_tombstoned=True, superseded_by="x_old")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(type(service), "_iter_user_memory_pages", pages_then_sweep)
        try:
            result = service.dedup_memories("u1", semantic=False)
        finally:
            monkeypatch.undo()

        assert _deleted(service) == [] and "x_old" in fake.payloads
        assert result["exact_duplicates_removed"] == 0

    def test_a_locked_pool_is_skipped_and_reported(self, service, clock):
        _exact_rows(service)
        assert gate.acquire_lock(clock, POOL)

        result = service.dedup_memories("u1", semantic=False)

        assert _deleted(service) == []
        assert result["exact_skipped_locked_pools"] == 1

    def test_unlocked_group_is_deleted_under_the_lock_and_released(self, service, clock):
        _exact_rows(service)
        inner = service._memory.vector_store.delete.side_effect
        held = []
        service._memory.vector_store.delete.side_effect = lambda pid: (
            held.append(f"dreaming:lock:{POOL}" in clock.keys), inner(pid))[1]

        result = service.dedup_memories("u1", semantic=False)

        assert _deleted(service) == ["x_old"] and held == [True]
        assert result["exact_duplicates_removed"] == 1 and clock.keys == {}

    def test_a_group_spanning_two_pools_needs_both_locks(self, service, clock):
        fake = _FakeQdrant(service, [
            _row("m1", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("m2", A, "2025-06-01T00:00:00", scope="global", visibility="private",
                 owner_user_id="u9"),
        ])
        fake.payloads["m1"]["hash"] = fake.payloads["m2"]["hash"] = "hA"
        assert gate.acquire_lock(clock, "user--u9")  # only the second row's pool is busy

        result = service.dedup_memories("u1", semantic=False)

        assert _deleted(service) == [] and result["exact_skipped_locked_pools"] == 1
        assert f"dreaming:lock:{POOL}" not in clock.keys  # the free pool's lock was let go

    def test_a_page_without_a_duplicate_group_never_touches_redis(self, service, clock):
        _FakeQdrant(service, [
            _row("a", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("b", B, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])

        result = service.dedup_memories("u1", semantic=False)

        assert clock.pings == 0 and clock.sets == 0
        assert result["exact_duplicates_removed"] == 0

    def test_redis_down_skips_the_exact_group_instead_of_running_unlocked(self, service, clock):
        _exact_rows(service)
        clock.fail = True

        res = service.dedup_memories("u1", semantic=False)

        assert res["exact_duplicates_removed"] == 0 and _deleted(service) == []
        assert res["exact_groups_abandoned"] == 1


# ── 5. gate.refresh_lock ────────────────────────────────────────────────────────


class TestRefreshLock:
    def test_owner_renews_and_a_stranger_does_not(self):
        r = _ClockRedis()
        token = gate.acquire_lock(r, POOL, ttl_ms=10 * MIN)
        r.now += 9 * MIN
        assert gate.refresh_lock(r, POOL, token, ttl_ms=10 * MIN) is True
        r.now += 9 * MIN
        assert gate.refresh_lock(r, POOL, token, ttl_ms=10 * MIN) is True  # renewed, not expired
        assert gate.refresh_lock(r, POOL, "someone-else") is False
        assert gate.refresh_lock(r, POOL, None) is False

    def test_expired_lock_taken_by_another_is_not_ours(self):
        r = _ClockRedis()
        token = gate.acquire_lock(r, POOL, ttl_ms=10 * MIN)
        r.now += 11 * MIN
        other = gate.acquire_lock(r, POOL)
        assert other and gate.refresh_lock(r, POOL, token) is False
        assert r.get(f"dreaming:lock:{POOL}") == other

    def test_redis_error_fails_closed(self):
        class Boom(_ClockRedis):
            def get(self, key):
                raise ConnectionError("down")

        assert gate.refresh_lock(Boom(), POOL, "t") is False


# ── 6. an invalidate re-checks its replacement row ──────────────────────────────


@pytest.mark.asyncio
async def test_invalidate_is_skipped_when_its_replacement_row_was_tombstoned():
    payloads = {
        "s": {"data": "replacement", "metadata": {"dream_tombstoned": True,
                                                   "superseded_by": "dedup-survivor"}},
        "l": {"data": "stale fact", "metadata": {}},
    }
    service, client, _ = _service(payloads, bm25_slot=False)
    act = {"type": "invalidate", "memory_ids": ["l"], "superseded_by_id": "s", "confidence": 0.9}

    result = await consolidate.apply_actions(service, _batch(["s", "l"]), [act], dry_run=False)

    assert not result.applied and not result.errors and len(result.skipped) == 1
    assert "s" in result.skipped[0]["skip_reason"]
    assert client.calls == []
