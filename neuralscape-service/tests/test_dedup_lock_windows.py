"""Dedup and the dream sweep's pool lock: cross-pool groups, a Redis that comes back, scripts.

(1) A duplicate group spanning two pools verifies and renews the locks of ALL its pools
before every delete, re-reads its survivor, and writes nothing once any check fails;
``require`` refuses a pool this run holds no lock for. (2) A failed Redis probe is not
remembered: Redis answering again mid-run takes effect at the next group / page. (3) The
exact phase uses only the rows it re-read after taking the lock. (4) The lock's
``refresh`` / ``release`` Lua scripts are EXECUTED here, by a small interpreter for the
script shape, so a wrong key or a wrong compare fails a test.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from config import settings
from extensions.dreaming import gate
from extensions.dreaming.config import dreaming_settings
from memory.delete import _DedupLocks, _PoolLockLost
from tests.test_dedup import _FakeQdrant, _deleted, _row, _tombstoned, service  # noqa: F401
from tests.test_dedup_dream_exclusion import dream_on_tombstone  # noqa: F401
from tests.test_dedup_lock_ownership import MIN, _ClockRedis, _copying_scroll

A = "Deploys go through the staging gate before prod"
B = "The cache layer uses Redis with a five minute TTL"
POOL = "user--u1"
POOL9 = "user--u9"


class _LuaRedis(_ClockRedis):
    """Clock Redis whose ``eval`` interprets the lock scripts (``if C then return X else return Y end``).

    Lua's ``redis.call`` becomes a method call, ``KEYS`` / ``ARGV`` are 1-indexed and every
    argument reaches the script as a string (as on a real server), so a script that reads
    ``KEYS[2]``, compares the wrong value or calls an unknown command fails here.
    """

    def __init__(self):
        super().__init__()
        self.evals: list[str] = []

    def _call(self, cmd, *args):
        if cmd == "get":
            return self.get(args[0]) or False  # nil is false in Lua
        if cmd == "pexpire":
            return self.pexpire(args[0], int(args[1]))
        if cmd == "del":
            existed = args[0] in self.keys
            self.delete(args[0])
            return 1 if existed else 0
        raise AssertionError(f"script called unsupported command {cmd!r}")

    def eval(self, script, numkeys, *keys_and_args):
        self.evals.append(script)
        if self.fail:
            raise ConnectionError("down")
        keys = [str(k) for k in keys_and_args[:numkeys]]
        args = [str(a) for a in keys_and_args[numkeys:]]
        m = re.fullmatch(r"if (.+) then return (.+) else return (.+) end", script.replace("redis.call", "call"))
        assert m, f"script shape not understood: {script!r}"
        cond, then, other = m.groups()
        env = {"KEYS": [None, *keys], "ARGV": [None, *args], "call": self._call}
        return eval(then if eval(cond, env) else other, env)  # noqa: S307 - our own two scripts


@pytest.fixture
def sredis(monkeypatch):
    r = _LuaRedis()
    monkeypatch.setattr("extensions.dreaming.sweep._get_redis", lambda: r)
    return r


@pytest.fixture
def dreaming_off_hard(monkeypatch):
    monkeypatch.setattr(dreaming_settings, "enabled", False)
    monkeypatch.setattr(settings, "dedup_semantic_tombstone_enabled", False)


def _key(pool):
    return f"dreaming:lock:{pool}"


# ── the lock scripts, executed ───────────────────────────────────────────────────


class TestLockScripts:
    def test_refresh_owner_true_and_renews_the_expiry_of_that_pool_only(self):
        r = _LuaRedis()
        t1 = gate.acquire_lock(r, POOL, ttl_ms=10 * MIN)
        t9 = gate.acquire_lock(r, POOL9, ttl_ms=10 * MIN)
        r.now = 8 * MIN

        assert gate.refresh_lock(r, POOL, t1, ttl_ms=10 * MIN) is True

        assert r.exp[_key(POOL)] == 18 * MIN and r.exp[_key(POOL9)] == 10 * MIN
        assert r.evals, "the refresh script must actually run"
        assert gate.refresh_lock(r, POOL9, t1, ttl_ms=10 * MIN) is False  # the other pool's key, a stranger's token
        assert gate.refresh_lock(r, POOL9, t9, ttl_ms=10 * MIN) is True

    def test_refresh_stranger_missing_and_expired_are_false_and_change_nothing(self):
        r = _LuaRedis()
        tok = gate.acquire_lock(r, POOL, ttl_ms=10 * MIN)

        assert gate.refresh_lock(r, POOL, "someone-else", ttl_ms=10 * MIN) is False
        assert r.exp[_key(POOL)] == 10 * MIN and r.keys[_key(POOL)] == tok
        assert gate.refresh_lock(r, "user--nobody", tok, ttl_ms=10 * MIN) is False
        r.now = 11 * MIN
        assert gate.refresh_lock(r, POOL, tok, ttl_ms=10 * MIN) is False
        assert _key(POOL) not in r.keys

    def test_release_only_clears_the_owners_own_key(self):
        r = _LuaRedis()
        t1 = gate.acquire_lock(r, POOL)
        t9 = gate.acquire_lock(r, POOL9)

        gate.release_lock(r, POOL, "stranger")
        assert _key(POOL) in r.keys
        gate.release_lock(r, POOL, t9)  # a token valid for ANOTHER pool
        assert _key(POOL) in r.keys and _key(POOL9) in r.keys
        gate.release_lock(r, POOL, t1)
        assert _key(POOL) not in r.keys and _key(POOL9) in r.keys

    def test_require_refuses_a_pool_it_holds_no_lock_for(self):
        r = _LuaRedis()
        locks = _DedupLocks(SimpleNamespace(_dream_lock_client=lambda: r))
        assert locks.redis() is r and locks.acquire(POOL)

        locks.require(POOL)
        locks.require(None)  # a tier the sweep never touches
        with pytest.raises(_PoolLockLost) as lost:
            locks.require(POOL, POOL9)  # never locked: never a silent pass
        assert lost.value.pool == POOL9
        r.keys[_key(POOL)] = "sweeper"  # expired and taken
        with pytest.raises(_PoolLockLost):
            locks.require(POOL)

    def test_failed_probe_is_not_cached(self):
        r = _LuaRedis()
        r.fail = True
        locks = _DedupLocks(SimpleNamespace(_dream_lock_client=lambda: None if r.fail else r))
        assert locks.redis() is None
        r.fail = False
        assert locks.redis() is r


# ── cross-pool exact groups ──────────────────────────────────────────────────────


def _cross_pool(service, *, survivor_owner="u9"):
    fake = _FakeQdrant(service, [
        _row("m1a", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
        _row("m1b", A, "2025-02-01T00:00:00", scope="global", visibility="private"),
        _row("m1c", A, "2025-03-01T00:00:00", scope="global", visibility="private"),
        _row("m2", A, "2025-06-01T00:00:00", scope="global", visibility="private",
             owner_user_id=survivor_owner),
    ])
    for pid in fake.payloads:
        fake.payloads[pid]["hash"] = "hA"
    return fake


def _live_with(fake, text):
    return sorted(p for p, pl in fake.payloads.items()
                  if pl.get("data") == text and not (pl.get("metadata") or {}).get("dream_tombstoned"))


class TestCrossPoolGroups:
    def test_the_survivors_pool_lock_is_renewed_so_a_sweep_cannot_take_it_mid_group(
        self, service, sredis
    ):
        fake = _cross_pool(service)
        inner = service._memory.vector_store.delete.side_effect
        sweep_tries = []

        def slow_delete(pid):  # each delete's graph cascade is slow; the survivor's lock would age out
            inner(pid)
            sredis.now += 6 * MIN
            sweep_tries.append(gate.acquire_lock(sredis, POOL9))  # a sweep tick lands on the survivor's pool

        service._memory.vector_store.delete.side_effect = slow_delete

        res = service.dedup_memories("u1", semantic=False)

        assert sweep_tries == [None, None, None], "the survivor's pool lock must stay ours across the group"
        assert _deleted(service) == ["m1c", "m1b", "m1a"] and res["lock_lost_pools"] == 0
        assert _live_with(fake, A) == ["m2"]

    def test_survivors_lock_lost_mid_group_stops_every_further_delete(self, service, sredis):
        fake = _cross_pool(service)
        inner = service._memory.vector_store.delete.side_effect

        def delete_then_sweep_takes_survivors_pool(pid):
            inner(pid)
            sredis.keys[_key(POOL9)] = "sweeper"  # lock expired, a sweep took it ...
            fake.payloads["m2"]["metadata"].update(dream_tombstoned=True, pruned=True)  # ... and pruned m2

        service._memory.vector_store.delete.side_effect = delete_then_sweep_takes_survivors_pool

        res = service.dedup_memories("u1", semantic=False)

        assert _deleted(service) == ["m1c"], "no delete once the survivor's pool lock is not ours"
        assert res["lock_lost_pools"] == 1 and res["exact_duplicates_removed"] == 1
        assert _live_with(fake, A) == ["m1a", "m1b"], "the fact must survive in a live row"
        assert fake.payloads["m2"]["metadata"].get("times_derived") is None  # no counter bump after the loss
        assert sredis.keys[_key(POOL9)] == "sweeper" and _key(POOL) not in sredis.keys

    @pytest.mark.parametrize("how", ["key_changed", "tombstoned"])
    def test_a_survivor_that_changed_under_the_lock_abandons_the_group(self, service, sredis, how):
        fake = _cross_pool(service, survivor_owner="u1")
        inner = service._memory.vector_store.delete.side_effect

        def delete_then_survivor_changes(pid):
            inner(pid)
            if how == "key_changed":
                fake.payloads["m2"]["hash"] = "h-merged"
            else:
                fake.payloads["m2"]["metadata"]["dream_tombstoned"] = True

        service._memory.vector_store.delete.side_effect = delete_then_survivor_changes

        res = service.dedup_memories("u1", semantic=False)

        assert _deleted(service) == ["m1c"]
        assert res["exact_groups_abandoned"] == 1 and res["lock_lost_pools"] == 0
        assert _live_with(fake, A) == (["m1a", "m1b"] if how == "tombstoned" else ["m1a", "m1b", "m2"])
        assert fake.payloads["m2"]["metadata"].get("times_derived") is None
        assert sredis.keys == {}, "the group's locks are released"

    def test_a_row_whose_pool_changed_before_the_lock_is_never_deleted_unlocked(self, service, sredis, monkeypatch):
        fake = _FakeQdrant(service, [
            _row("x_old", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("y_new", A, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])
        fake.payloads["x_old"]["hash"] = fake.payloads["y_new"]["hash"] = "hA"
        service._memory.vector_store.client.scroll.side_effect = _copying_scroll(fake)
        orig = type(service)._live_rows

        def owner_changes_before_the_reread(self_, rows):
            fake.payloads["x_old"]["metadata"]["owner_user_id"] = "u7"  # now pool user--u7: not locked
            return orig(self_, rows)

        monkeypatch.setattr(type(service), "_live_rows", owner_changes_before_the_reread)

        res = service.dedup_memories("u1", semantic=False)

        assert _deleted(service) == [] and res["lock_lost_pools"] == 1


# ── a Redis that comes back mid-run ──────────────────────────────────────────────


class TestRedisFlap:
    def test_semantic_pages_are_probed_again_while_the_probe_failed(
        self, service, sredis, dreaming_off_hard, monkeypatch
    ):
        monkeypatch.setattr(settings, "dedup_batch_size", 2)
        _FakeQdrant(service, [
            _row("a_old", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("a_new", A, "2025-06-01T00:00:00", scope="global", visibility="private"),
            _row("b_old", B, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("b_new", B, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])
        sredis.fail = True  # page 1: Redis is down, dreaming off: runs unlocked
        inner = service._memory.vector_store.delete.side_effect
        sweep = []

        def delete_then_redis_back_and_forced_sweep_starts(pid):
            inner(pid)
            sredis.fail = False
            sweep.append(gate.acquire_lock(sredis, POOL))

        service._memory.vector_store.delete.side_effect = delete_then_redis_back_and_forced_sweep_starts

        res = service.dedup_memories("u1")

        assert sweep and sweep[0], "the forced sweep took the pool"
        assert _deleted(service) == ["a_old"], "page 2 must see the sweep's lock, not run unlocked"
        assert res["semantic_skipped_locked_pools"] == 1 and res["semantic_duplicates_removed"] == 1

    def test_with_dreaming_on_a_page_skipped_for_a_down_redis_is_retried_on_the_next(
        self, service, sredis, dream_on_tombstone, monkeypatch  # noqa: F811
    ):
        monkeypatch.setattr(settings, "dedup_batch_size", 2)
        _FakeQdrant(service, [
            _row("a_old", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("a_new", A, "2025-06-01T00:00:00", scope="global", visibility="private"),
            _row("b_old", B, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("b_new", B, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])
        sredis.fail = True
        pages = type(service)._iter_user_memory_pages
        n = {"calls": 0}

        def pages_then_redis_back(self_, *a, **kw):
            n["calls"] += 1  # call 1 is phase 1's key count (no duplicates: no second pass); call 2 is phase 2
            for i, page in enumerate(pages(self_, *a, **kw), start=1):
                if n["calls"] == 2 and i == 2:
                    sredis.fail = False  # Redis is back before phase 2's page 2 is looked at
                yield page

        monkeypatch.setattr(type(service), "_iter_user_memory_pages", pages_then_redis_back)

        res = service.dedup_memories("u1")

        assert _tombstoned(service) == ["b_old"], "page 1 skipped (no lock, never unlocked), page 2 done under the lock"
        assert "1 page" in res["semantic_skipped"] and "unavailable" in res["semantic_skipped"]

    def test_exact_groups_are_probed_again_while_the_probe_failed(self, service, sredis):
        fake = _FakeQdrant(service, [
            _row("x1", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("x2", A, "2025-02-01T00:00:00", scope="global", visibility="private"),
            _row("y1", B, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("y2", B, "2025-02-01T00:00:00", scope="global", visibility="private"),
        ])
        for pid in ("x1", "x2"):
            fake.payloads[pid]["hash"] = "hA"
        for pid in ("y1", "y2"):
            fake.payloads[pid]["hash"] = "hB"
        sredis.fail = True
        inner = service._memory.vector_store.delete.side_effect
        sweep = []

        def delete_then_redis_back_and_sweep_starts(pid):
            inner(pid)
            sredis.fail = False
            sweep.append(gate.acquire_lock(sredis, POOL))

        service._memory.vector_store.delete.side_effect = delete_then_redis_back_and_sweep_starts

        res = service.dedup_memories("u1", semantic=False)

        assert _deleted(service) == ["x1"], "group 2 must meet the sweep's lock"
        assert res["exact_skipped_locked_pools"] == 1


# ── the exact phase uses only the re-read rows ───────────────────────────────────


def test_exact_phase_uses_the_rows_it_re_read_after_taking_the_lock(service, sredis, monkeypatch):
    fake = _FakeQdrant(service, [
        _row("x_old", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
        _row("y_new", A, "2025-06-01T00:00:00", scope="global", visibility="private"),
    ])
    fake.payloads["x_old"]["hash"] = fake.payloads["y_new"]["hash"] = "hA"
    service._memory.vector_store.client.scroll.side_effect = _copying_scroll(fake)  # page rows are snapshots
    orig = type(service)._live_rows

    def counter_bumped_between_the_page_read_and_the_lock(self_, rows):
        fake.payloads["x_old"]["metadata"]["times_derived"] = 3  # e.g. a write-path reinforcement
        return orig(self_, rows)

    monkeypatch.setattr(type(service), "_live_rows", counter_bumped_between_the_page_read_and_the_lock)

    service.dedup_memories("u1", semantic=False)

    assert _deleted(service) == ["x_old"]
    # survivor's own 1 + the dropped row's CURRENT 3 (the page snapshot said 1)
    assert fake.payloads["y_new"]["metadata"].get("times_derived") == 4
