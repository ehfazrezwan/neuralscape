"""Semantic dedup and the dream sweep never work the same pool at once.

(a) ``dedup_memories`` skips the semantic phase for a pool whose dream lock is
held; the exact-hash phase still runs. (b) ``apply_actions`` re-reads the
tombstone state of a MERGE's rows right before applying it and skips the action
(recorded as skipped, not as an error) when either row was tombstoned after the
sweep staged it, so the loser's content stays readable.
"""

from __future__ import annotations

import pytest

from extensions.dreaming import consolidate, gate
from extensions.dreaming.config import dreaming_settings
from memory.delete import _dream_pool_of
from tests.test_dedup import _FakeQdrant, _deleted, _row, _tombstoned, service  # noqa: F401
from tests.test_dreaming_rewrite_atomic import _batch, _service

TEXT = "Deploys go through the staging gate before prod"


class _FakeRedis:
    def __init__(self):
        self.keys: dict[str, str] = {}

    def set(self, key, value, nx=False, px=None):
        if nx and key in self.keys:
            return None
        self.keys[key] = value
        return True

    def exists(self, key):
        return key in self.keys


@pytest.fixture
def dream_on_tombstone(monkeypatch):
    from config import settings

    monkeypatch.setattr(dreaming_settings, "enabled", True)
    monkeypatch.setattr(settings, "dedup_semantic_tombstone_enabled", True)


def _pair(service):
    return _FakeQdrant(service, [
        _row("old", TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
        _row("new", TEXT, "2025-06-01T00:00:00", scope="global", visibility="private"),
    ])


@pytest.fixture
def redis(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr("extensions.dreaming.sweep._get_redis", lambda: r)
    return r


@pytest.mark.usefixtures("dream_on_tombstone")
class TestSemanticPhaseHonoursTheDreamLock:
    def test_locked_pool_skips_the_semantic_phase(self, service, redis, caplog):
        fake = _pair(service)
        pool = _dream_pool_of(fake.payloads["old"])
        assert gate.acquire_lock(redis, pool)

        with caplog.at_level("INFO"):
            result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 0
        assert result["semantic_skipped_locked_pools"] == 1
        assert _tombstoned(service) == [] and _deleted(service) == []
        assert any("semantic_skipped_locked_pools=1" in r.getMessage() for r in caplog.records)

    def test_free_pool_runs_the_semantic_phase(self, service, redis):
        _pair(service)

        result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 1
        assert result["semantic_skipped_locked_pools"] == 0
        assert _tombstoned(service) == ["old"]

    def test_another_pools_lock_does_not_block(self, service, redis):
        _pair(service)
        assert gate.acquire_lock(redis, "private:someone-else")

        assert service.dedup_memories("u1")["semantic_duplicates_removed"] == 1

    def test_exact_phase_still_runs_when_the_pool_is_locked(self, service, redis):
        fake = _FakeQdrant(service, [
            _row("e1", TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("e2", TEXT, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])
        for pid in ("e1", "e2"):
            fake.payloads[pid]["hash"] = "same-hash"
        assert gate.acquire_lock(redis, _dream_pool_of(fake.payloads["e1"]))

        result = service.dedup_memories("u1")

        assert result["exact_duplicates_removed"] == 1
        assert _deleted(service) == ["e1"]

    def test_dreaming_off_never_consults_the_lock(self, service, redis, monkeypatch):
        fake = _pair(service)
        assert gate.acquire_lock(redis, _dream_pool_of(fake.payloads["old"]))
        monkeypatch.setattr(dreaming_settings, "enabled", False)

        assert service.dedup_memories("u1")["semantic_duplicates_removed"] == 1


class TestMergeSkipsRowsTombstonedSinceStaging:
    MERGE = {"type": "merge", "memory_ids": ["s", "l"], "survivor_id": "s",
             "content": "merged fact", "confidence": 0.9}

    async def _apply(self, payloads):
        service, client, _ = _service(payloads, bm25_slot=False)
        result = await consolidate.apply_actions(
            service, _batch(["s", "l"]), [dict(self.MERGE)], dry_run=False
        )
        return result, client

    @pytest.mark.asyncio
    async def test_tombstoned_survivor_skips_the_merge_and_keeps_the_loser_readable(self):
        payloads = {
            "s": {"data": "survivor text", "metadata": {"dream_tombstoned": True}},
            "l": {"data": "loser text", "metadata": {}},
        }
        result, client = await self._apply(payloads)

        assert not result.errors and not result.applied
        assert len(result.skipped) == 1
        assert "tombstoned since staging" in result.skipped[0]["skip_reason"]
        assert "s" in result.skipped[0]["skip_reason"]
        assert client.payloads["l"]["data"] == "loser text"
        assert "dream_tombstoned" not in client.payloads["l"]["metadata"]
        assert client.calls == []  # nothing was written

    @pytest.mark.asyncio
    async def test_tombstoned_loser_skips_the_merge(self):
        payloads = {
            "s": {"data": "survivor text", "metadata": {}},
            "l": {"data": "loser text", "metadata": {"metadata": {"dream_tombstoned": True}}},
        }
        result, client = await self._apply(payloads)

        assert not result.errors and not result.applied and len(result.skipped) == 1
        assert client.payloads["s"]["data"] == "survivor text"

    @pytest.mark.asyncio
    async def test_live_rows_still_merge(self):
        payloads = {
            "s": {"data": "survivor text", "metadata": {}},
            "l": {"data": "loser text", "metadata": {}},
        }
        result, client = await self._apply(payloads)

        assert not result.errors and not result.skipped and len(result.applied) == 1
        assert client.payloads["l"]["metadata"]["dream_tombstoned"] is True
