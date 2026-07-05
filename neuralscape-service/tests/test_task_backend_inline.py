"""Inline task backend interim behavior (solo engine, pulled forward from
unit 4): with ``task_backend != "redis"`` the TaskManager never touches Redis,
its pool stays a falsy sentinel, and any enqueue raises ConnectionError — the
exact exception type every API/MCP write path's synchronous fallback catches.

Found by the solo e2e boot: previously connect() raised on a missing Redis
(startup crash), and a None pool raised AttributeError from enqueues, which
the fallbacks do NOT catch.
"""

import asyncio

import pytest

from task_manager import TaskManager, _DISABLED_POOL


class TestDisabledPool:
    def test_is_falsy(self):
        assert not _DISABLED_POOL

    def test_any_use_raises_connection_error(self):
        with pytest.raises(ConnectionError, match="task queue unavailable"):
            _DISABLED_POOL.enqueue_job

    def test_fallback_catch_tuple_matches(self):
        """The write paths catch (ConnectionError, OSError) — the sentinel's
        error must land inside that tuple."""
        try:
            _DISABLED_POOL.zadd
        except (ConnectionError, OSError):
            pass


class TestInlineConnect:
    def test_connect_skips_redis_in_inline_mode(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "task_backend", "inline")
        called = {"create_pool": False}

        async def fake_create_pool(*a, **k):
            called["create_pool"] = True

        monkeypatch.setattr("task_manager.create_pool", fake_create_pool)
        tm = TaskManager()
        asyncio.run(tm.connect())
        assert called["create_pool"] is False
        assert not tm.pool

    def test_connect_uses_redis_in_team_mode(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "task_backend", "redis")
        sentinel_pool = object()

        async def fake_create_pool(*a, **k):
            return sentinel_pool

        monkeypatch.setattr("task_manager.create_pool", fake_create_pool)
        tm = TaskManager()
        asyncio.run(tm.connect())
        assert tm.pool is sentinel_pool

    def test_enqueue_raises_connection_error_when_disabled(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "task_backend", "inline")
        tm = TaskManager()
        asyncio.run(tm.connect())
        with pytest.raises(ConnectionError):
            asyncio.run(tm.enqueue_raw("x", "u", "preference"))
