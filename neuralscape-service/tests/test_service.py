"""Tests for neuralscape-service endpoints using FastAPI TestClient.

Tests both legacy (root) and new v1 endpoints.
"""

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
def mock_service():
    """Patch the MemoryService instance for v1 endpoints."""
    mock_svc = MagicMock(name="MemoryService")
    original = main._service
    main._service = mock_svc
    yield mock_svc
    main._service = original


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
# Legacy POST /memories (sync)
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
# Legacy POST /search
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
# Legacy GET /memories
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
# Legacy DELETE /memories
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
# Legacy POST /memories/async
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
# Legacy GET /memories/status/{task_id}
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


# ══════════════════════════════════════════════
# V1 Endpoints
# ══════════════════════════════════════════════


class TestV1StoreRawMemory:
    def test_stores_raw_memory(self, client, mock_service):
        from schemas import MemoryResponse
        mock_service.store_raw.return_value = [
            MemoryResponse(id="m1", memory="Prefers tabs", category="preference", scope="global")
        ]
        resp = client.post("/v1/memories/raw", json={
            "content": "Prefers tabs over spaces",
            "user_id": "ehfaz",
            "category": "preference",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["memories"]) == 1
        assert data["memories"][0]["category"] == "preference"

    def test_returns_400_for_invalid_category(self, client, mock_service):
        mock_service.store_raw.side_effect = ValueError("Invalid category: bogus")
        resp = client.post("/v1/memories/raw", json={
            "content": "test",
            "user_id": "ehfaz",
            "category": "bogus",
        })
        assert resp.status_code == 400


class TestV1StoreMemories:
    def test_extracts_and_stores(self, client, mock_service):
        from schemas import MemoryResponse
        mock_service.extract_and_store.return_value = [
            MemoryResponse(id="m1", memory="Uses Python 3.12", category="technical_skill", scope="global"),
            MemoryResponse(id="m2", memory="Uses FastAPI", category="tech_stack", scope="project"),
        ]
        resp = client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "I use Python 3.12 with FastAPI"}],
            "user_id": "ehfaz",
            "project_id": "my-project",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["memories"]) == 2

    def test_returns_500_on_error(self, client, mock_service):
        mock_service.extract_and_store.side_effect = Exception("LLM down")
        resp = client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "hello"}],
            "user_id": "ehfaz",
        })
        assert resp.status_code == 500


class TestV1Search:
    def test_searches_memories(self, client, mock_service):
        from schemas import MemoryResponse
        mock_service.search.return_value = [
            MemoryResponse(id="m1", memory="Prefers tabs", score=0.95)
        ]
        resp = client.post("/v1/search", json={
            "query": "indentation style",
            "user_id": "ehfaz",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["results"]) == 1

    def test_search_with_project_id(self, client, mock_service):
        from schemas import MemoryResponse
        mock_service.search.return_value = []
        resp = client.post("/v1/search", json={
            "query": "tech stack",
            "user_id": "ehfaz",
            "project_id": "neuralscape",
        })
        assert resp.status_code == 200
        mock_service.search.assert_called_once()
        call_kwargs = mock_service.search.call_args[1]
        assert call_kwargs["project_id"] == "neuralscape"

    def test_search_with_category_filter(self, client, mock_service):
        from schemas import MemoryResponse
        mock_service.search.return_value = []
        resp = client.post("/v1/search", json={
            "query": "preferences",
            "user_id": "ehfaz",
            "categories": ["preference", "personal_fact"],
        })
        assert resp.status_code == 200
        call_kwargs = mock_service.search.call_args[1]
        assert call_kwargs["categories"] == ["preference", "personal_fact"]


class TestV1GraphSearch:
    def test_graph_search(self, client, mock_service):
        mock_service.search_graph.return_value = {
            "edges": [{"uuid": "e1", "name": "uses", "fact": "Uses Python"}],
            "nodes": [],
            "episodes": [],
            "communities": [],
        }
        resp = client.post("/v1/graph/search", json={
            "query": "Python",
            "user_id": "ehfaz",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["edges"]) == 1


class TestV1Context:
    def test_get_global_context(self, client, mock_service):
        from schemas import ContextResponse, MemoryResponse
        mock_service.get_global_context.return_value = ContextResponse(
            user_id="ehfaz",
            categories={
                "preference": [MemoryResponse(id="m1", memory="Prefers dark mode")],
            },
        )
        resp = client.get("/v1/context/global", params={"user_id": "ehfaz"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "ehfaz"
        assert "preference" in data["categories"]

    def test_get_project_context(self, client, mock_service):
        from schemas import ContextResponse, MemoryResponse
        mock_service.get_project_context.return_value = ContextResponse(
            user_id="ehfaz",
            project_id="my-project",
            categories={
                "tech_stack": [MemoryResponse(id="m1", memory="Uses FastAPI")],
                "preference": [MemoryResponse(id="m2", memory="Prefers tabs")],
            },
        )
        resp = client.get("/v1/context/my-project", params={"user_id": "ehfaz"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_id"] == "my-project"
        assert "tech_stack" in data["categories"]
        assert "preference" in data["categories"]


class TestV1ManageMemories:
    def test_list_memories_with_filters(self, client, mock_service):
        from schemas import MemoryResponse
        mock_service.list_memories.return_value = [
            MemoryResponse(id="m1", memory="fact1", category="preference"),
        ]
        resp = client.get("/v1/memories", params={
            "user_id": "ehfaz",
            "scope": "global",
            "category": "preference",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_get_single_memory(self, client, mock_service):
        from schemas import MemoryResponse
        mock_service.get_memory.return_value = MemoryResponse(
            id="m1", memory="Prefers tabs", category="preference"
        )
        resp = client.get("/v1/memories/m1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "m1"

    def test_get_memory_not_found(self, client, mock_service):
        mock_service.get_memory.return_value = None
        resp = client.get("/v1/memories/nonexistent")
        assert resp.status_code == 404

    def test_update_memory(self, client, mock_service):
        mock_service.update_memory.return_value = {"message": "Memory updated successfully"}
        resp = client.put("/v1/memories/m1", json={
            "content": "Prefers 4-space indentation",
        })
        assert resp.status_code == 200

    def test_delete_single_memory(self, client, mock_service):
        mock_service.delete_memory.return_value = {"message": "Memory deleted successfully!"}
        resp = client.delete("/v1/memories/m1")
        assert resp.status_code == 200


class TestV1Categories:
    def test_list_categories(self, client):
        resp = client.get("/v1/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "preference" in data["categories"]
        assert "tech_stack" in data["categories"]
        assert len(data["categories"]) == 13


class TestV1AsyncMemoryStatus:
    def test_returns_completed_task(self, client):
        task_id = "v1-test-task"
        main._tasks[task_id] = {
            "status": "completed",
            "created_at": "2026-01-01T00:00:00+00:00",
            "result": [{"id": "m1", "memory": "test"}],
            "error": None,
        }
        resp = client.get(f"/v1/memories/status/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    def test_returns_404_for_unknown(self, client):
        resp = client.get("/v1/memories/status/nonexistent")
        assert resp.status_code == 404


class TestV1GraphIntrospection:
    def test_graph_nodes(self, client, mock_service):
        mock_service.get_graph_nodes.return_value = [
            {"uuid": "n1", "name": "Python", "summary": "Language", "labels": [], "group_id": "global", "created_at": "2026-01-01"}
        ]
        resp = client.get("/v1/graph/nodes", params={"user_id": "ehfaz"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["nodes"]) == 1

    def test_graph_edges(self, client, mock_service):
        mock_service.get_graph_edges.return_value = []
        resp = client.get("/v1/graph/edges", params={"user_id": "ehfaz"})
        assert resp.status_code == 200

    def test_graph_episodes(self, client, mock_service):
        mock_service.get_graph_episodes.return_value = []
        resp = client.get("/v1/graph/episodes", params={"user_id": "ehfaz"})
        assert resp.status_code == 200

    def test_graph_communities(self, client, mock_service):
        mock_service.get_graph_communities.return_value = []
        resp = client.get("/v1/graph/communities", params={"user_id": "ehfaz"})
        assert resp.status_code == 200
