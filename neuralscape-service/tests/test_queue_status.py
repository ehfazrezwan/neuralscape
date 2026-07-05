"""Tests for queue visibility + the queue.empty webhook (roadmap C4).

Covers: per-caller task tracking at enqueue, queue_status aggregation with
stubbed Redis, caught_up semantics, the REST/MCP surfaces, the webhook SSRF
guard (http(s)-only, no redirects, 5s cap, daemon thread), and the worker's
after_job_end hook (fires only on an empty queue, attributes the caller,
never raises).
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import task_manager as task_manager_mod
import webhooks
from config import settings
from task_manager import TaskManager, _task_user_key, _user_tasks_key
from worker import _make_after_job_end


class FakeRedis:
    """Minimal async Redis stub: sorted sets + string keys."""

    def __init__(self):
        self.zsets: dict[str, dict[str, float]] = {}
        self.strings: dict[str, str] = {}

    async def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    async def zremrangebyscore(self, key, lo, hi):
        lo = float("-inf") if lo == "-inf" else float(lo)
        hi = float("inf") if hi == "+inf" else float(hi)
        z = self.zsets.get(key, {})
        for member in [m for m, s in z.items() if lo <= s <= hi]:
            del z[member]

    async def expire(self, key, ttl):
        return True

    async def set(self, key, value, ex=None):
        self.strings[key] = value

    async def get(self, key):
        v = self.strings.get(key)
        return v.encode() if isinstance(v, str) else v

    async def zrevrangebyscore(self, key, max=None, min=None, start=0, num=None):
        z = self.zsets.get(key, {})
        members = [
            m for m, s in sorted(z.items(), key=lambda kv: -kv[1])
            if (min is None or s >= min) and (max is None or s <= max)
        ]
        end = None if num is None else start + num
        return [m.encode() for m in members[start:end]]

    async def zcard(self, key):
        return len(self.zsets.get(key, {}))


@pytest.fixture()
def tm():
    manager = TaskManager()
    manager.pool = FakeRedis()
    return manager


# ──────────────────────────────────────────────
# Tracking at enqueue
# ──────────────────────────────────────────────


class TestTaskTracking:
    @pytest.mark.asyncio
    async def test_track_records_forward_and_reverse_keys(self, tm):
        await tm._track_task("alice", "task-1")
        assert "task-1" in tm.pool.zsets[_user_tasks_key("alice")]
        assert tm.pool.strings[_task_user_key("task-1")] == "alice"

    @pytest.mark.asyncio
    async def test_track_noop_without_user_or_pool(self, tm):
        await tm._track_task(None, "task-1")
        assert tm.pool.zsets == {}
        empty = TaskManager()
        await empty._track_task("alice", "task-1")  # pool None — no crash

    @pytest.mark.asyncio
    async def test_track_swallows_redis_errors(self, tm):
        tm.pool.zadd = AsyncMock(side_effect=RuntimeError("redis down"))
        await tm._track_task("alice", "task-1")  # must not raise

    @pytest.mark.asyncio
    async def test_enqueue_raw_tracks_caller(self, tm):
        job = MagicMock(job_id="job-raw-1")
        tm.pool.enqueue_job = AsyncMock(return_value=job)
        task_id = await tm.enqueue_raw(content="x", user_id="alice", category="preference")
        assert task_id == "job-raw-1"
        assert "job-raw-1" in tm.pool.zsets[_user_tasks_key("alice")]

    @pytest.mark.asyncio
    async def test_enqueue_raw_batch_tracks_each_distinct_user(self, tm):
        tm.pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="job-b"))
        await tm.enqueue_raw_batch([
            {"content": "a", "user_id": "alice", "category": "decision"},
            {"content": "b", "user_id": "bob", "category": "decision"},
            {"content": "c", "user_id": "alice", "category": "decision"},
        ])
        assert "job-b" in tm.pool.zsets[_user_tasks_key("alice")]
        assert "job-b" in tm.pool.zsets[_user_tasks_key("bob")]

    @pytest.mark.asyncio
    async def test_duplicate_enqueue_still_tracked(self, tm):
        """ARQ returning None (duplicate job id) still yields a pollable id —
        it must be tracked for the caller too."""
        tm.pool.enqueue_job = AsyncMock(return_value=None)
        task_id = await tm.enqueue_raw(content="x", user_id="alice", category="preference")
        assert task_id.startswith("ns-")
        assert task_id in tm.pool.zsets[_user_tasks_key("alice")]


# ──────────────────────────────────────────────
# get_queue_status aggregation
# ──────────────────────────────────────────────


class TestQueueStatusAggregation:
    @pytest.mark.asyncio
    async def test_counts_by_live_status(self, tm, monkeypatch):
        now = time.time()
        for i, tid in enumerate(["t-q", "t-p", "t-c", "t-f", "t-x"]):
            await tm.pool.zadd(_user_tasks_key("alice"), {tid: now - i})
        statuses = {
            "t-q": "queued", "t-p": "processing", "t-c": "completed",
            "t-f": "failed", "t-x": "not_found",
        }

        async def fake_status(task_id):
            return {"task_id": task_id, "status": statuses[task_id],
                    "result": None, "error": None}

        monkeypatch.setattr(tm, "get_status", fake_status)
        out = await tm.get_queue_status("alice")
        assert out["tracked"] == 5
        assert out["counts"] == {
            "queued": 1, "processing": 1, "completed": 1, "failed": 1, "expired": 1,
        }
        assert out["caught_up"] is False

    @pytest.mark.asyncio
    async def test_caught_up_when_nothing_pending(self, tm, monkeypatch):
        await tm.pool.zadd(_user_tasks_key("alice"), {"t-1": time.time()})

        async def fake_status(task_id):
            return {"task_id": task_id, "status": "completed", "result": None, "error": None}

        monkeypatch.setattr(tm, "get_status", fake_status)
        out = await tm.get_queue_status("alice")
        assert out["caught_up"] is True
        assert out["counts"]["completed"] == 1

    @pytest.mark.asyncio
    async def test_empty_tracking_is_caught_up(self, tm):
        out = await tm.get_queue_status("nobody")
        assert out["tracked"] == 0
        assert out["caught_up"] is True

    @pytest.mark.asyncio
    async def test_window_excludes_old_tasks(self, tm, monkeypatch):
        now = time.time()
        await tm.pool.zadd(_user_tasks_key("alice"), {
            "recent": now - 10,
            "ancient": now - settings.queue_status_window_s - 100,
        })

        async def fake_status(task_id):
            return {"task_id": task_id, "status": "queued", "result": None, "error": None}

        monkeypatch.setattr(tm, "get_status", fake_status)
        out = await tm.get_queue_status("alice")
        assert out["tracked"] == 1

    @pytest.mark.asyncio
    async def test_reports_instance_queue_depths(self, tm, monkeypatch):
        await tm.pool.zadd(settings.arq_queue_name, {"j1": 1, "j2": 2})
        await tm.pool.zadd(settings.graph_queue_name, {"j3": 1})
        out = await tm.get_queue_status("alice")
        assert out["queues"] == {"main": 2, "graph": 1, "ingest": 0}

    @pytest.mark.asyncio
    async def test_tracked_read_failure_degrades(self, tm, monkeypatch):
        tm.pool.zrevrangebyscore = AsyncMock(side_effect=RuntimeError("boom"))
        out = await tm.get_queue_status("alice")
        assert out["tracked"] == 0 and out["caught_up"] is True


# ──────────────────────────────────────────────
# REST + MCP surfaces
# ──────────────────────────────────────────────


class TestQueueStatusSurfaces:
    def test_rest_route(self, monkeypatch):
        from fastapi.testclient import TestClient

        import main

        tm = MagicMock()
        tm.get_queue_status = AsyncMock(return_value={
            "user_id": "alice", "window_seconds": 3600, "tracked": 2,
            "counts": {"queued": 0, "processing": 0, "completed": 2,
                       "failed": 0, "expired": 0},
            "queues": {"main": 0, "graph": 0, "ingest": 0},
            "caught_up": True,
        })
        monkeypatch.setattr(main, "_task_manager", tm)
        client = TestClient(main.app, raise_server_exceptions=False)
        resp = client.get("/v1/queue/status", params={"user_id": "alice"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["caught_up"] is True and body["tracked"] == 2
        tm.get_queue_status.assert_awaited_once_with("alice")

    @pytest.mark.asyncio
    async def test_mcp_tool(self, monkeypatch):
        import mcp_server

        tm = MagicMock()
        tm.get_queue_status = AsyncMock(return_value={
            "user_id": "alice", "window_seconds": 3600, "tracked": 1,
            "counts": {"queued": 1, "processing": 0, "completed": 0,
                       "failed": 0, "expired": 0},
            "queues": {"main": 1, "graph": 0, "ingest": 0},
            "caught_up": False,
        })
        monkeypatch.setattr(mcp_server, "_task_manager", tm)
        result = await mcp_server.call_tool("queue_status", {"user_id": "alice"})
        data = json.loads(result[0].text)
        assert data["status"] == "ok" and data["caught_up"] is False


# ──────────────────────────────────────────────
# Webhook SSRF guard
# ──────────────────────────────────────────────


class TestWebhookGuard:
    def test_http_and_https_allowed(self):
        assert webhooks.webhook_url_allowed("https://hooks.example.com/queue")
        assert webhooks.webhook_url_allowed("http://127.0.0.1:9999/hook")

    def test_non_http_schemes_rejected(self):
        for url in (
            "ftp://example.com/x",
            "file:///etc/passwd",
            "gopher://example.com",
            "javascript:alert(1)",
            "redis://localhost:6379",
        ):
            assert not webhooks.webhook_url_allowed(url), url

    def test_hostless_and_empty_rejected(self):
        for url in ("", "http://", "https://", "not a url", "/relative/path"):
            assert not webhooks.webhook_url_allowed(url), url

    def test_fire_rejects_bad_url_without_dispatch(self, monkeypatch):
        posted = []
        monkeypatch.setattr(webhooks, "_post", lambda *a: posted.append(a))
        assert webhooks.fire_queue_empty("ftp://internal/x", {"event": "queue.empty"}) is False
        assert posted == []

    def test_fire_dispatches_on_daemon_thread(self, monkeypatch):
        import threading

        done = threading.Event()
        captured = {}

        def fake_post(url, payload):
            captured["url"] = url
            captured["payload"] = payload
            captured["daemon"] = threading.current_thread().daemon
            done.set()

        monkeypatch.setattr(webhooks, "_post", fake_post)
        assert webhooks.fire_queue_empty("https://x.test/hook", {"event": "queue.empty"}) is True
        assert done.wait(timeout=2)
        assert captured["url"] == "https://x.test/hook"
        assert captured["daemon"] is True

    def test_post_never_follows_redirects_and_caps_timeout(self, monkeypatch):
        captured = {}

        def fake_httpx_post(url, **kwargs):
            captured.update(kwargs, url=url)
            return MagicMock(status_code=204)

        import httpx

        monkeypatch.setattr(httpx, "post", fake_httpx_post)
        webhooks._post("https://x.test/hook", {"event": "queue.empty"})
        assert captured["follow_redirects"] is False
        assert captured["timeout"] <= 5.0

    def test_post_swallows_delivery_errors(self, monkeypatch):
        import httpx

        monkeypatch.setattr(
            httpx, "post", MagicMock(side_effect=httpx.ConnectError("refused"))
        )
        webhooks._post("https://x.test/hook", {})  # must not raise


# ──────────────────────────────────────────────
# Worker after_job_end hook
# ──────────────────────────────────────────────


class TestQueueEmptyHook:
    def _ctx(self, depth=0, user=b"alice"):
        redis = MagicMock()
        redis.zcard = AsyncMock(return_value=depth)
        redis.get = AsyncMock(return_value=user)
        return {"redis": redis, "job_id": "job-9"}

    @pytest.mark.asyncio
    async def test_fires_when_queue_empty(self, monkeypatch):
        monkeypatch.setattr(settings, "webhook_queue_empty_url", "https://x.test/hook")
        fired = []
        monkeypatch.setattr(webhooks, "fire_queue_empty", lambda url, p: fired.append((url, p)))
        hook = _make_after_job_end("test:queue")
        await hook(self._ctx(depth=0))
        assert len(fired) == 1
        url, payload = fired[0]
        assert url == "https://x.test/hook"
        assert payload["event"] == "queue.empty"
        assert payload["queue"] == "test:queue"
        assert payload["user_id"] == "alice"
        assert payload["job_id"] == "job-9"

    @pytest.mark.asyncio
    async def test_silent_when_queue_still_has_jobs(self, monkeypatch):
        monkeypatch.setattr(settings, "webhook_queue_empty_url", "https://x.test/hook")
        fired = []
        monkeypatch.setattr(webhooks, "fire_queue_empty", lambda url, p: fired.append(p))
        hook = _make_after_job_end("test:queue")
        await hook(self._ctx(depth=3))
        assert fired == []

    @pytest.mark.asyncio
    async def test_disabled_when_url_empty(self, monkeypatch):
        monkeypatch.setattr(settings, "webhook_queue_empty_url", "")
        ctx = self._ctx()
        hook = _make_after_job_end("test:queue")
        await hook(ctx)
        ctx["redis"].zcard.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_user_still_fires_with_null(self, monkeypatch):
        monkeypatch.setattr(settings, "webhook_queue_empty_url", "https://x.test/hook")
        fired = []
        monkeypatch.setattr(webhooks, "fire_queue_empty", lambda url, p: fired.append(p))
        hook = _make_after_job_end("test:queue")
        await hook(self._ctx(depth=0, user=None))
        assert fired[0]["user_id"] is None

    @pytest.mark.asyncio
    async def test_hook_swallows_errors(self, monkeypatch):
        monkeypatch.setattr(settings, "webhook_queue_empty_url", "https://x.test/hook")
        redis = MagicMock()
        redis.zcard = AsyncMock(side_effect=RuntimeError("redis exploded"))
        hook = _make_after_job_end("test:queue")
        await hook({"redis": redis, "job_id": "j"})  # must not raise

    def test_all_worker_settings_have_hook(self):
        import worker

        for cls in (worker.WorkerSettings, worker.GraphWorkerSettings,
                    worker.IngestWorkerSettings):
            assert "after_job_end" in cls.__dict__
