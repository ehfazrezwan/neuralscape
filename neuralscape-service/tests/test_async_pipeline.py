"""Integration tests for the async memory pipeline (ARQ + Redis).

Prerequisites — these services must be running:
  - Redis on localhost:6379
  - Qdrant on localhost:6333
  - Neo4j on localhost:7687
  - API server on localhost:8199  (uv run python main.py)
  - ARQ worker                    (uv run arq worker.WorkerSettings)

Run:
  uv run pytest tests/test_async_pipeline.py -v -s

Skip these automatically when the server isn't reachable by using the
`integration` marker:
  uv run pytest tests/test_async_pipeline.py -v -s -m integration
"""

import asyncio
import time

import httpx
import pytest

BASE_URL = "http://localhost:8199"
POLL_INTERVAL = 5  # seconds between status polls
POLL_TIMEOUT = 300  # max seconds to wait for task completion (LLM + Neo4j writes are slow)


def server_is_reachable() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not server_is_reachable(),
        reason="Neuralscape API server not reachable at localhost:8199",
    ),
]


def poll_until_done(task_id: str, base_path: str = "/v1/memories/status") -> dict:
    """Poll a task status endpoint until it completes or times out."""
    url = f"{BASE_URL}{base_path}/{task_id}"
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        resp = httpx.get(url, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"Task {task_id} did not complete within {POLL_TIMEOUT}s")


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────


class TestHealth:
    def test_health_endpoint(self):
        resp = httpx.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ──────────────────────────────────────────────
# POST /v1/memories/raw  (async raw store)
# ──────────────────────────────────────────────


class TestAsyncRawMemoryStore:
    def test_returns_202_with_task_id(self):
        resp = httpx.post(
            f"{BASE_URL}/v1/memories/raw",
            json={
                "content": "Integration test: prefers 4-space indentation",
                "user_id": "test-async-pipeline",
                "category": "preference",
            },
            timeout=10,
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "accepted"
        assert "task_id" in data
        assert data["poll_url"].startswith("/v1/memories/status/")

    def test_task_completes_successfully(self):
        resp = httpx.post(
            f"{BASE_URL}/v1/memories/raw",
            json={
                "content": "Integration test: uses vim keybindings",
                "user_id": "test-async-pipeline",
                "category": "preference",
            },
            timeout=10,
        )
        task_id = resp.json()["task_id"]

        result = poll_until_done(task_id)
        assert result["status"] == "completed"
        assert result["error"] is None
        assert "memories" in result["result"]
        assert len(result["result"]["memories"]) >= 1

        mem = result["result"]["memories"][0]
        assert "id" in mem
        assert mem["category"] == "preference"
        assert mem["scope"] == "global"

    def test_rejects_invalid_category(self):
        resp = httpx.post(
            f"{BASE_URL}/v1/memories/raw",
            json={
                "content": "test",
                "user_id": "test-async-pipeline",
                "category": "invalid_category",
            },
            timeout=10,
        )
        assert resp.status_code == 400
        assert "Invalid category" in resp.json()["detail"]

    def test_rejects_project_scope_without_project_id(self):
        resp = httpx.post(
            f"{BASE_URL}/v1/memories/raw",
            json={
                "content": "test",
                "user_id": "test-async-pipeline",
                "category": "tech_stack",
                "scope": "project",
            },
            timeout=10,
        )
        assert resp.status_code == 400
        assert "project_id" in resp.json()["detail"]


# ──────────────────────────────────────────────
# POST /v1/memories  (async conversation extraction)
# ──────────────────────────────────────────────


class TestAsyncConversationExtraction:
    def test_returns_202_with_task_id(self):
        resp = httpx.post(
            f"{BASE_URL}/v1/memories",
            json={
                "messages": [
                    {"role": "user", "content": "I prefer Python over JavaScript for backend work."},
                    {"role": "assistant", "content": "Noted, Python is your preferred backend language."},
                ],
                "user_id": "test-async-pipeline",
            },
            timeout=10,
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "accepted"
        assert "task_id" in data

    def test_extracts_and_stores_facts(self):
        resp = httpx.post(
            f"{BASE_URL}/v1/memories",
            json={
                "messages": [
                    {"role": "user", "content": "I always use pytest for testing. My timezone is UTC+6."},
                    {"role": "assistant", "content": "Got it, pytest for tests and UTC+6 timezone."},
                ],
                "user_id": "test-async-pipeline",
            },
            timeout=10,
        )
        task_id = resp.json()["task_id"]

        result = poll_until_done(task_id)
        assert result["status"] == "completed"
        assert result["error"] is None
        assert "memories" in result["result"]
        # LLM should extract at least 1 fact
        assert len(result["result"]["memories"]) >= 1


# ──────────────────────────────────────────────
# GET /v1/memories/status/{task_id}
# ──────────────────────────────────────────────


class TestTaskStatusPolling:
    def test_returns_404_for_unknown_task(self):
        resp = httpx.get(
            f"{BASE_URL}/v1/memories/status/nonexistent-task-id-12345",
            timeout=5,
        )
        assert resp.status_code == 404

    def test_status_transitions(self):
        """Enqueue a task and verify status goes from queued/processing to completed."""
        resp = httpx.post(
            f"{BASE_URL}/v1/memories/raw",
            json={
                "content": "Integration test: status transition check",
                "user_id": "test-async-pipeline",
                "category": "interaction",
            },
            timeout=10,
        )
        task_id = resp.json()["task_id"]

        # Immediately check — should be queued or processing
        status_resp = httpx.get(
            f"{BASE_URL}/v1/memories/status/{task_id}", timeout=5
        )
        assert status_resp.status_code == 200
        initial_status = status_resp.json()["status"]
        assert initial_status in ("queued", "processing", "completed")

        # Wait for completion
        result = poll_until_done(task_id)
        assert result["status"] == "completed"


# ──────────────────────────────────────────────
# POST /v1/search  (sync — unaffected by async changes)
# ──────────────────────────────────────────────


class TestSyncSearch:
    def test_search_returns_results(self):
        """Search should still work synchronously and find stored memories."""
        resp = httpx.post(
            f"{BASE_URL}/v1/search",
            json={
                "query": "indentation preference",
                "user_id": "test-async-pipeline",
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert isinstance(data["results"], list)


# ──────────────────────────────────────────────
# Legacy endpoints (backward compat)
# ──────────────────────────────────────────────


class TestLegacyAsyncEndpoint:
    def test_legacy_async_returns_accepted(self):
        resp = httpx.post(
            f"{BASE_URL}/memories/async",
            json={
                "messages": [{"role": "user", "content": "Legacy test memory"}],
                "user_id": "test-async-pipeline",
            },
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert "task_id" in data
        assert "poll_url" in data

    def test_legacy_status_endpoint(self):
        resp = httpx.post(
            f"{BASE_URL}/memories/async",
            json={
                "messages": [{"role": "user", "content": "Legacy status check"}],
                "user_id": "test-async-pipeline",
            },
            timeout=10,
        )
        task_id = resp.json()["task_id"]

        result = poll_until_done(task_id, base_path="/memories/status")
        assert result["status"] in ("completed", "failed")
