"""Client request/poll shaping against a mocked transport (no live service)."""

import httpx
import pytest

from neuralscape_bench.client import NeuralscapeClient, TaskTimeout


def _client(handler):
    # Inject a MockTransport-backed client (no overwrite of a real one → no leak).
    http = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler),
                             headers={"Content-Type": "application/json"})
    return NeuralscapeClient("http://test", http=http)


async def test_raw_write_returns_task_id():
    def handler(req):
        assert req.url.path == "/v1/memories/raw"
        return httpx.Response(202, json={"status": "accepted", "task_id": "t1", "poll_url": "/v1/memories/status/t1"})
    c = _client(handler)
    try:
        resp = await c.raw_write("hello", user_id="bench-x", category="convention")
        assert resp["task_id"] == "t1"
    finally:
        await c.aclose()


async def test_wait_for_task_polls_to_completion():
    calls = {"n": 0}

    def handler(req):
        if req.url.path == "/v1/memories/status/t1":
            calls["n"] += 1
            status = "completed" if calls["n"] >= 2 else "processing"
            return httpx.Response(200, json={"task_id": "t1", "status": status, "result": {}, "error": None})
        return httpx.Response(404)
    c = _client(handler)
    try:
        out = await c.wait_for_task("t1", timeout_s=5, interval_s=0.01)
        assert out["status"] == "completed" and calls["n"] >= 2
    finally:
        await c.aclose()


async def test_wait_for_task_treats_404_as_terminal():
    # A 404 from the status endpoint is a terminal "unknown/expired" state and
    # must not raise — it should return not_found so a run isn't aborted.
    def handler(req):
        return httpx.Response(404, json={"detail": "no such task"})
    c = _client(handler)
    try:
        out = await c.wait_for_task("t1", timeout_s=5, interval_s=0.01)
        assert out["status"] == "not_found"
    finally:
        await c.aclose()


async def test_wait_for_task_times_out():
    def handler(req):
        return httpx.Response(200, json={"task_id": "t1", "status": "processing", "result": None, "error": None})
    c = _client(handler)
    try:
        with pytest.raises(TaskTimeout):
            await c.wait_for_task("t1", timeout_s=0.05, interval_s=0.01)
    finally:
        await c.aclose()


async def test_search_returns_results():
    def handler(req):
        assert req.url.path == "/v1/search"
        return httpx.Response(200, json={"status": "ok", "results": [{"id": "m1", "memory": "x"}]})
    c = _client(handler)
    try:
        res = await c.search("query", user_id="bench-x")
        assert len(res["results"]) == 1
    finally:
        await c.aclose()


async def test_delete_bench_data_is_tolerant():
    def handler(req):
        return httpx.Response(404, json={"detail": "nope"})
    c = _client(handler)
    try:
        out = await c.delete_bench_data(user_id="bench-x")
        assert out["status"] == "cleanup_failed"  # no exception raised
    finally:
        await c.aclose()
