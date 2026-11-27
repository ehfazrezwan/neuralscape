"""Post-compile dedup: a pool skipped for a held or lost dream lock is visible to the caller.

``dedup_memories`` skips a pool whose dream lock a running sweep holds and abandons a pool
whose lock it lost mid-work; it reports both as counters. The compile result used to carry
only the lock-store-down counters, so a compile whose dedup skipped a whole pool said
nothing. Driven through ``compile_date`` with the real ``MemoryService.dedup_memories``
(fake Qdrant, fake Redis holding the sweep's lock) and, for the summing and precedence
rules, through ``_dedup_compiled_users`` with canned dedup results.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from extensions.conversation_compiler.compile import _dedup_compiled_users, compile_date
from extensions.conversation_compiler.obsidian_writer import ObsidianWriter
from extensions.dreaming import gate
from tests.test_dedup import _FakeQdrant, _deleted, _row, _tombstoned, service  # noqa: F401
from tests.test_dedup_dream_exclusion import (  # noqa: F401
    OTHER,
    POOL,
    TEXT,
    _pair,
    dream_on_tombstone,
    redis,
)
from tests.test_dedup_lock_ownership import MIN, _ClockRedis

DATE = "2026-11-17"
HELD = "dream lock held or lost"
STORE_DOWN = "lock store unreachable"


@pytest.fixture
def writer(tmp_path):
    w = ObsidianWriter(vault_path=tmp_path / "vault")
    w.append_daily_log(
        DATE,
        [{"time": "10:00", "category": "decision", "content": "Chose Qdrant", "session_id": "s", "user_id": "u1"}],
    )
    return w


@pytest.fixture(autouse=True)
def gemini():
    with patch(
        "extensions.conversation_compiler.compile._async_call_gemini",
        new=AsyncMock(return_value="summary"),
    ):
        yield


def _fields(result) -> dict:
    return {
        "triggered": result.dedup_triggered,
        "exact": result.dedup_exact_skipped_locked_pools,
        "semantic": result.dedup_semantic_skipped_locked_pools,
        "lost": result.dedup_lock_lost_pools,
        "reason": result.dedup_skipped_reason,
    }


@pytest.mark.usefixtures("dream_on_tombstone")
@pytest.mark.asyncio
async def test_semantic_pool_held_by_a_sweep_is_reported(writer, service, redis):  # noqa: F811
    _pair(service)
    assert gate.acquire_lock(redis, POOL)  # a sweep is working the pool

    result = await compile_date(DATE, service, writer, user_id="u1")

    got = _fields(result)
    assert got["semantic"] == 1 and got["reason"] == HELD and got["triggered"] is False, got
    assert _tombstoned(service) == [] and _deleted(service) == [], "the held pool is untouched"
    assert result.dedup_skipped == 0, "dedup_skipped keeps its lock-store-down meaning"


@pytest.mark.usefixtures("dream_on_tombstone")
@pytest.mark.asyncio
async def test_exact_pool_held_by_a_sweep_is_reported(writer, service, redis):  # noqa: F811
    fake = _FakeQdrant(service, [
        _row("e1", TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
        _row("e2", TEXT, "2025-06-01T00:00:00", scope="global", visibility="private"),
    ])
    for pid in ("e1", "e2"):
        fake.payloads[pid]["hash"] = "same-hash"
    assert gate.acquire_lock(redis, POOL)

    result = await compile_date(DATE, service, writer, user_id="u1")

    got = _fields(result)
    assert got["exact"] == 1 and got["reason"] == HELD and got["triggered"] is False, got
    assert _deleted(service) == [] and _tombstoned(service) == [], "the held pool is untouched"


@pytest.mark.usefixtures("dream_on_tombstone")
@pytest.mark.asyncio
async def test_empty_day_compile_reports_a_held_pool_too(tmp_path, service, redis):  # noqa: F811
    """The explicit compile of a date with no entries builds its result on its own path."""
    _pair(service)
    assert gate.acquire_lock(redis, POOL)
    empty = ObsidianWriter(vault_path=tmp_path / "empty-vault")

    result = await compile_date(DATE, service, empty, user_id="u1")

    got = _fields(result)
    assert result.entries_compiled == 0
    assert got["semantic"] == 1 and got["reason"] == HELD and got["triggered"] is False, got
    assert _tombstoned(service) == [] and _deleted(service) == [], "the held pool is untouched"


@pytest.mark.usefixtures("dream_on_tombstone")
@pytest.mark.asyncio
async def test_pool_whose_lock_was_lost_mid_work_is_reported(writer, service, monkeypatch):  # noqa: F811
    clock = _ClockRedis()
    monkeypatch.setattr("extensions.dreaming.sweep._get_redis", lambda: clock)
    _FakeQdrant(service, [
        _row("a_old", TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
        _row("a_new", TEXT, "2025-06-01T00:00:00", scope="global", visibility="private"),
        _row("b_old", OTHER, "2025-01-01T00:00:00", scope="global", visibility="private"),
        _row("b_new", OTHER, "2025-06-01T00:00:00", scope="global", visibility="private"),
    ])
    inner = service._memory.vector_store.search.side_effect

    def slow_search(*a, **kw):
        clock.now += 11 * MIN  # slower than the lock expiry; a sweep tick takes the pool
        gate.acquire_lock(clock, POOL)
        return inner(*a, **kw)

    service._memory.vector_store.search.side_effect = slow_search

    result = await compile_date(DATE, service, writer, user_id="u1")

    got = _fields(result)
    assert got["lost"] == 1 and got["reason"] == HELD and got["triggered"] is False, got
    assert _tombstoned(service) == [] and _deleted(service) == [], "no write after the lock was lost"


@pytest.mark.usefixtures("dream_on_tombstone")
@pytest.mark.asyncio
async def test_released_pool_is_deduplicated_and_nothing_is_reported(writer, service, redis):  # noqa: F811
    """Control: with no sweep holding the pool the same compile dedups and reports no skip."""
    _pair(service)

    result = await compile_date(DATE, service, writer, user_id="u1")

    got = _fields(result)
    assert got == {"triggered": True, "exact": 0, "semantic": 0, "lost": 0, "reason": None}, got
    assert _tombstoned(service) == ["old"], "the near-duplicate was deduplicated"


@pytest.mark.asyncio
async def test_counters_are_summed_over_users_and_store_down_reason_wins():
    results = {
        "u1": {"exact_skipped_locked_pools": 1, "semantic_skipped_locked_pools": 2, "lock_lost_pools": 3},
        "u2": {"semantic_skipped_locked_pools": 1, "lock_store_unreachable_skipped": 2},
    }
    svc = MagicMock()
    svc.dedup_memories.side_effect = lambda uid, semantic=True: results[uid]
    entries = [{"user_id": "u1"}, {"user_id": "u2"}]

    ok, _, _, skip = await _dedup_compiled_users(svc, entries, "u1")

    assert ok is True
    assert skip["exact_skipped_locked_pools"] == 1
    assert skip["semantic_skipped_locked_pools"] == 3
    assert skip["lock_lost_pools"] == 3
    assert skip["skipped"] == 2
    assert skip["reason"] == STORE_DOWN, "an unreachable lock store is the wider cause and wins"


@pytest.mark.asyncio
async def test_a_dedup_with_no_lock_counters_leaves_the_summary_clean():
    svc = MagicMock()
    svc.dedup_memories.return_value = {"exact_duplicates_removed": 4}

    _, _, _, skip = await _dedup_compiled_users(svc, [{"user_id": "u1"}], "u1")

    assert skip["reason"] is None
    assert (skip["exact_skipped_locked_pools"], skip["semantic_skipped_locked_pools"], skip["lock_lost_pools"]) == (0, 0, 0)
