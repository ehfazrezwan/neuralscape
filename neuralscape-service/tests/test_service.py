"""Tests for neuralscape-service endpoints using FastAPI TestClient."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main
from main import app


@pytest.fixture(autouse=True)
def mock_memory():
    """Patch the lazy-init globals so no real mem0/Graphiti is created."""
    mock_mem = MagicMock(name="Memory")
    mock_graphiti = MagicMock(name="Graphiti")
    mock_bridge = MagicMock(name="AsyncBridge")

    original_memory = main._memory
    original_graphiti = main._graphiti
    original_bridge = main._bridge
    original_async_memory = main._async_memory
    original_tasks = main._tasks.copy()

    main._memory = mock_mem
    main._graphiti = mock_graphiti
    main._bridge = mock_bridge
    main._async_memory = None
    main._tasks.clear()

    yield mock_mem

    main._memory = original_memory
    main._graphiti = original_graphiti
    main._bridge = original_bridge
    main._async_memory = original_async_memory
    main._tasks.clear()
    main._tasks.update(original_tasks)


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────


class TestHealthEndpoint:
    def test_returns_200_with_status_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "service" in data


# ──────────────────────────────────────────────
# POST /memories (sync)
# ──────────────────────────────────────────────


class TestAddMemorySync:
    def test_returns_result(self, client, mock_memory):
        mock_memory.add.return_value = {"results": [{"memory": "test", "event": "ADD"}]}
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "I work at Acme"}],
            "user_id": "test_user",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "result" in data

    def test_calls_memory_add(self, client, mock_memory):
        mock_memory.add.return_value = {}
        client.post("/memories", json={
            "messages": [{"role": "user", "content": "hello"}],
            "user_id": "u1",
        })
        mock_memory.add.assert_called_once()
        call_kwargs = mock_memory.add.call_args[1]
        assert call_kwargs["user_id"] == "u1"

    def test_uses_default_user_id(self, client, mock_memory):
        mock_memory.add.return_value = {}
        client.post("/memories", json={
            "messages": [{"role": "user", "content": "hello"}],
        })
        call_kwargs = mock_memory.add.call_args[1]
        assert call_kwargs["user_id"] == "default_user"

    def test_returns_500_on_exception(self, client, mock_memory):
        mock_memory.add.side_effect = Exception("LLM error")
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert resp.status_code == 500


# ──────────────────────────────────────────────
# POST /search
# ──────────────────────────────────────────────


class TestSearchEndpoint:
    def test_returns_results(self, client, mock_memory):
        mock_memory.search.return_value = [{"memory": "works at Acme", "score": 0.9}]
        resp = client.post("/search", json={
            "query": "Where do I work?",
            "user_id": "u1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "results" in data


# ──────────────────────────────────────────────
# GET /memories
# ──────────────────────────────────────────────


class TestListMemoriesEndpoint:
    def test_returns_memories(self, client, mock_memory):
        mock_memory.get_all.return_value = [{"memory": "fact1"}, {"memory": "fact2"}]
        resp = client.get("/memories", params={"user_id": "u1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "memories" in data
        assert len(data["memories"]) == 2


# ──────────────────────────────────────────────
# DELETE /memories
# ──────────────────────────────────────────────


class TestDeleteMemoriesEndpoint:
    def test_calls_delete_all(self, client, mock_memory):
        resp = client.delete("/memories", params={"user_id": "u1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "deleted" in data["message"].lower()
        mock_memory.delete_all.assert_called_once()


# ──────────────────────────────────────────────
# POST /memories/async
# ──────────────────────────────────────────────


class TestAsyncAddMemory:
    def test_returns_task_id_with_accepted(self, client):
        resp = client.post("/memories/async", json={
            "messages": [{"role": "user", "content": "I work at Acme"}],
            "user_id": "test_user",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert "task_id" in data
        assert len(data["task_id"]) > 0

    def test_task_appears_in_store(self, client):
        resp = client.post("/memories/async", json={
            "messages": [{"role": "user", "content": "hello"}],
        })
        task_id = resp.json()["task_id"]
        assert task_id in main._tasks
        assert main._tasks[task_id]["status"] in ("processing", "completed", "failed")


# ──────────────────────────────────────────────
# GET /memories/status/{task_id}
# ──────────────────────────────────────────────


class TestAsyncTaskStatus:
    def test_returns_completed_task(self, client):
        task_id = "test-completed-task"
        main._tasks[task_id] = {
            "status": "completed",
            "created_at": "2026-01-01T00:00:00+00:00",
            "result": {"results": []},
            "error": None,
        }
        resp = client.get(f"/memories/status/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["result"] is not None
        assert data["error"] is None

    def test_returns_failed_task(self, client):
        task_id = "test-failed-task"
        main._tasks[task_id] = {
            "status": "failed",
            "created_at": "2026-01-01T00:00:00+00:00",
            "result": None,
            "error": "LLM timeout",
        }
        resp = client.get(f"/memories/status/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert data["error"] == "LLM timeout"

    def test_returns_404_for_unknown(self, client):
        resp = client.get("/memories/status/nonexistent-id")
        assert resp.status_code == 404
