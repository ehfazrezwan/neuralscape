"""Tests for neuralscape-service endpoints using FastAPI TestClient.

Tests both legacy (root) and new v1 endpoints.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

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

    main._memory = mock_mem
    main._graphiti = mock_graphiti
    main._bridge = mock_bridge
    main._async_memory = None

    yield mock_mem

    main._memory = original_memory
    main._graphiti = original_graphiti
    main._bridge = original_bridge
    main._async_memory = original_async_memory


@pytest.fixture
def mock_service():
    """Patch the MemoryService instance for v1 endpoints."""
    mock_svc = MagicMock(name="MemoryService")
    original = main._service
    main._service = mock_svc
    yield mock_svc
    main._service = original


@pytest.fixture(autouse=True)
def mock_task_manager():
    """Patch the TaskManager so tests don't need Redis."""
    mock_tm = MagicMock(name="TaskManager")
    mock_tm.connect = AsyncMock()
    mock_tm.close = AsyncMock()
    mock_tm.enqueue_store = AsyncMock(return_value="test-task-id-store")
    mock_tm.enqueue_raw = AsyncMock(return_value="test-task-id-raw")
    mock_tm.enqueue_retag = AsyncMock(return_value="test-task-id-retag")
    mock_tm.enqueue_graph_enrichment = AsyncMock(return_value="test-task-id-graph")
    mock_tm.get_status = AsyncMock(return_value={
        "task_id": "test-task-id",
        "status": "completed",
        "result": {"memories": []},
        "error": None,
    })
    original = main._task_manager
    main._task_manager = mock_tm
    yield mock_tm
    main._task_manager = original


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────


class TestHealthEndpoint:
    def test_returns_200_with_service_name(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert data["service"] == "neuralscape-memory"
        assert "checks" in data


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
        assert "poll_url" in data

    def test_enqueues_via_task_manager(self, client, mock_task_manager):
        resp = client.post("/memories/async", json={
            "messages": [{"role": "user", "content": "hello"}],
        })
        mock_task_manager.enqueue_store.assert_called_once()


# ──────────────────────────────────────────────
# Legacy GET /memories/status/{task_id}
# ──────────────────────────────────────────────


class TestAsyncTaskStatus:
    def test_returns_completed_task(self, client, mock_task_manager):
        mock_task_manager.get_status.return_value = {
            "task_id": "test-completed-task",
            "status": "completed",
            "result": {"memories": []},
            "error": None,
        }
        resp = client.get("/memories/status/test-completed-task")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["result"] is not None
        assert data["error"] is None

    def test_returns_failed_task(self, client, mock_task_manager):
        mock_task_manager.get_status.return_value = {
            "task_id": "test-failed-task",
            "status": "failed",
            "result": None,
            "error": "LLM timeout",
        }
        resp = client.get("/memories/status/test-failed-task")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert data["error"] == "LLM timeout"

    def test_returns_404_for_unknown(self, client, mock_task_manager):
        mock_task_manager.get_status.return_value = {
            "task_id": "nonexistent-id",
            "status": "not_found",
            "result": None,
            "error": None,
        }
        resp = client.get("/memories/status/nonexistent-id")
        assert resp.status_code == 404


# ══════════════════════════════════════════════
# V1 Endpoints
# ══════════════════════════════════════════════


class TestV1StoreRawMemory:
    def test_enqueues_and_returns_202(self, client, mock_task_manager):
        resp = client.post("/v1/memories/raw", json={
            "content": "Prefers tabs over spaces",
            "user_id": "ehfaz",
            "category": "preference",
        })
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "accepted"
        assert "task_id" in data
        assert data["poll_url"].startswith("/v1/memories/status/")
        mock_task_manager.enqueue_raw.assert_called_once()

    def test_returns_400_for_invalid_category(self, client):
        resp = client.post("/v1/memories/raw", json={
            "content": "test",
            "user_id": "ehfaz",
            "category": "bogus",
        })
        assert resp.status_code == 400

    def test_returns_400_for_project_scope_without_project_id(self, client):
        resp = client.post("/v1/memories/raw", json={
            "content": "test",
            "user_id": "ehfaz",
            "category": "tech_stack",
            "scope": "project",
        })
        assert resp.status_code == 400


class TestV1StoreMemories:
    def test_enqueues_and_returns_202(self, client, mock_task_manager):
        resp = client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "I use Python 3.12 with FastAPI"}],
            "user_id": "ehfaz",
            "project_id": "my-project",
        })
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "accepted"
        assert "task_id" in data
        mock_task_manager.enqueue_store.assert_called_once()


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

    def test_get_memory_returns_sensitivity_when_set(self, client, mock_service):
        """A memory the write-time sensitivity gate forced private surfaces
        `sensitivity`/`sensitivity_source` on read (memory/sensitivity.py +
        memory/write.py stamp them; memory/convert.py::_mem_to_response maps
        them through to MemoryResponse)."""
        from schemas import MemoryResponse
        mock_service.get_memory.return_value = MemoryResponse(
            id="m1", memory="Approved a $50,000 client contract renewal",
            category="decision", visibility="private",
            sensitivity="financial", sensitivity_source="regex",
        )
        resp = client.get("/v1/memories/m1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sensitivity"] == "financial"
        assert data["sensitivity_source"] == "regex"

    def test_get_memory_omits_sensitivity_when_unset(self, client, mock_service):
        """A plain (non-sensitive) memory's sensitivity fields render null —
        same as every other optional v2 field on this endpoint (which does
        not use exclude_none), so this is byte-identical to pre-change output
        for a memory the gate never touched."""
        from schemas import MemoryResponse
        mock_service.get_memory.return_value = MemoryResponse(
            id="m1", memory="Prefers tabs", category="preference"
        )
        resp = client.get("/v1/memories/m1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sensitivity"] is None
        assert data["sensitivity_source"] is None

    def test_delete_single_memory(self, client, mock_service):
        mock_service.delete_memory.return_value = {"message": "Memory deleted successfully!"}
        resp = client.delete("/v1/memories/m1")
        assert resp.status_code == 200


class TestV1MemoryByIdReadAuthorization:
    """IDOR fix: GET /v1/memories/{id} and its /reasoning_chain twin must pass
    the caller's resolved identity into the service so the read gate
    (private == owner-only, shared/standard == any authenticated caller) is
    actually applied — not just available.

    ``_service`` is fully mocked here (matching the rest of this file's REST
    tests), so these only prove the wiring: the resolved caller reaches the
    service call, and the service's None/missing-marker responses map to 404
    without leaking whether the memory exists. The real gate logic (private
    vs shared vs cross-user) is exercised directly against MemoryService in
    tests/test_memory_service.py's TestGetMemoryReadGate /
    TestReasoningChainReadGate.
    """

    def test_get_memory_passes_resolved_caller_id(self, client, mock_service):
        from schemas import MemoryResponse

        mock_service.get_memory.return_value = MemoryResponse(
            id="m1", memory="Prefers tabs", category="preference"
        )
        resp = client.get("/v1/memories/m1", params={"user_id": "alice"})
        assert resp.status_code == 200
        # (memory_id, resolved_caller_user_id) — no longer a bare memory_id call.
        mock_service.get_memory.assert_called_once_with("m1", "alice")

    def test_get_memory_unreadable_by_caller_maps_to_404(self, client, mock_service):
        """The service reports a caller-unreadable memory the same as a
        nonexistent one (None) — the route must 404, not leak a 403 that
        would confirm the id exists."""
        mock_service.get_memory.return_value = None
        resp = client.get("/v1/memories/m1", params={"user_id": "bob"})
        assert resp.status_code == 404
        mock_service.get_memory.assert_called_once_with("m1", "bob")

    def test_no_token_secret_configured_legacy_query_user_id_used(self, client, mock_service):
        """No per-user token secret configured (this file's autouse fixture
        disables it): the caller identity legacy deployments have always
        used — the query-string ``user_id`` — must still be what's passed to
        the service, unchanged by the read-gate fix."""
        from config import settings

        assert settings.neuralscape_user_token_secret == ""
        mock_service.get_memory.return_value = None
        client.get("/v1/memories/m1", params={"user_id": "legacy-caller"})
        mock_service.get_memory.assert_called_once_with("m1", "legacy-caller")

    def test_get_memory_no_user_id_falls_back_to_default(self, client, mock_service):
        """No token, no query user_id: falls back to settings.default_user_id
        (single-user/legacy deployment mode) — unchanged behavior."""
        mock_service.get_memory.return_value = None
        client.get("/v1/memories/m1")
        mock_service.get_memory.assert_called_once_with("m1", "default_user")

    def test_reasoning_chain_passes_resolved_caller_id(self, client, mock_service):
        mock_service.get_reasoning_chain.return_value = {
            "memory_id": "m1", "content": "x", "epistemic_level": None, "children": [],
        }
        resp = client.get("/v1/memories/m1/reasoning_chain", params={"user_id": "alice"})
        assert resp.status_code == 200
        mock_service.get_reasoning_chain.assert_called_once_with(
            "m1", 3, caller_user_id="alice"
        )

    def test_reasoning_chain_unreadable_root_maps_to_404(self, client, mock_service):
        mock_service.get_reasoning_chain.return_value = None
        resp = client.get("/v1/memories/m1/reasoning_chain", params={"user_id": "bob"})
        assert resp.status_code == 404


def _patch_result(graph="unchanged", graph_job=None):
    from schemas import MemoryResponse

    return {
        "memory": MemoryResponse(id="m1", memory="x", category="preference"),
        "graph_job": graph_job,
        "graph": graph,
    }


class TestV1PatchMemory:
    def test_patch_metadata_only(self, client, mock_service, mock_task_manager):
        mock_service.patch_memory.return_value = _patch_result()
        resp = client.patch("/v1/memories/m1", json={"tags": ["project:bon002"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["graph"] == "unchanged" and data["graph_task_id"] is None
        # PATCH semantics: only the sent field reaches the service
        args = mock_service.patch_memory.call_args.args
        assert args[0] == "m1"
        assert args[2] == {"tags": ["project:bon002"]}
        mock_task_manager.enqueue_graph_enrichment.assert_not_awaited()

    def test_patch_explicit_null_clears(self, client, mock_service):
        mock_service.patch_memory.return_value = _patch_result()
        resp = client.patch("/v1/memories/m1", json={"project_id": None})
        assert resp.status_code == 200
        assert mock_service.patch_memory.call_args.args[2] == {"project_id": None}

    def test_patch_empty_body_400(self, client, mock_service):
        resp = client.patch("/v1/memories/m1", json={})
        assert resp.status_code == 400
        mock_service.patch_memory.assert_not_called()

    def test_patch_permission_maps_403(self, client, mock_service):
        mock_service.patch_memory.side_effect = PermissionError("Only the memory's owner")
        resp = client.patch("/v1/memories/m1", json={"content": "rewrite"})
        assert resp.status_code == 403

    def test_patch_not_found_maps_404(self, client, mock_service):
        mock_service.patch_memory.side_effect = LookupError("Memory m1 not found")
        resp = client.patch("/v1/memories/m1", json={"tags": ["x"]})
        assert resp.status_code == 404

    def test_patch_invalid_maps_400(self, client, mock_service):
        mock_service.patch_memory.side_effect = ValueError("Invalid category: bogus")
        resp = client.patch("/v1/memories/m1", json={"category": "bogus"})
        assert resp.status_code == 400

    def test_put_alias_still_works(self, client, mock_service):
        mock_service.patch_memory.return_value = _patch_result()
        resp = client.put("/v1/memories/m1", json={"content": "Prefers 4-space indentation"})
        assert resp.status_code == 200

    def test_patch_enqueues_graph_job(self, client, mock_service, mock_task_manager):
        job = {
            "memory_id": "m1", "content": "new", "user_id": "ehfaz",
            "project_id": "p1", "visibility": "shared", "source_ref": None,
        }
        mock_service.patch_memory.return_value = _patch_result("reingest_pending", job)
        resp = client.patch("/v1/memories/m1", json={"content": "new"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["graph"] == "reingest_queued"
        assert data["graph_task_id"] == "test-task-id-graph"
        mock_task_manager.enqueue_graph_enrichment.assert_awaited_once_with(**job)

    def test_patch_reports_enqueue_failure_honestly(self, client, mock_service, mock_task_manager):
        job = {"memory_id": "m1", "content": "new", "user_id": "e",
               "project_id": None, "visibility": "private", "source_ref": None}
        mock_service.patch_memory.return_value = _patch_result("migration_pending", job)
        mock_task_manager.enqueue_graph_enrichment.side_effect = ConnectionError("redis down")
        resp = client.patch("/v1/memories/m1", json={"project_id": None})
        assert resp.status_code == 200
        assert resp.json()["graph"] == "enqueue_failed"


class TestV1Retag:
    def test_retag_returns_202(self, client, mock_service, mock_task_manager):
        resp = client.post("/v1/memories/retag", json={
            "filters": {"project_id": "neuralscape"},
            "ops": {"add_tags": ["project:bon002"]},
        })
        assert resp.status_code == 202
        assert resp.json()["task_id"] == "test-task-id-retag"
        _caller, filters, ops = mock_task_manager.enqueue_retag.await_args.args
        assert filters == {"project_id": "neuralscape"}
        assert ops == {"add_tags": ["project:bon002"]}
        mock_service.retag_memories.assert_not_called()

    def test_retag_preserves_explicit_null_project(self, client, mock_task_manager):
        resp = client.post("/v1/memories/retag", json={
            "filters": {"category": "decision"},
            "ops": {"set_project_id": None},
        })
        assert resp.status_code == 202
        _, _, ops = mock_task_manager.enqueue_retag.await_args.args
        assert ops == {"set_project_id": None}

    def test_retag_dry_run_synchronous(self, client, mock_service, mock_task_manager):
        mock_service.retag_memories.return_value = {
            "matched": 5, "updated": 4, "skipped_forbidden": 1,
            "skipped_invalid": 0, "graph_jobs": [], "dry_run": True,
        }
        resp = client.post("/v1/memories/retag", json={
            "filters": {"category": "decision"},
            "ops": {"add_tags": ["t"]},
            "dry_run": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] == 5 and "graph_jobs" not in data
        mock_task_manager.enqueue_retag.assert_not_awaited()

    def test_retag_requires_a_filter(self, client):
        resp = client.post("/v1/memories/retag", json={
            "filters": {}, "ops": {"add_tags": ["t"]},
        })
        assert resp.status_code == 422

    def test_retag_empty_filter_values_rejected(self, client, mock_task_manager):
        """REGRESSION: `tags_contains: []` / `category: ""` produce no Qdrant
        condition downstream — counting them as 'a filter is present' would
        turn a supposedly filtered retag into an unfiltered sweep."""
        for filters in ({"tags_contains": []}, {"category": ""}):
            resp = client.post("/v1/memories/retag", json={
                "filters": filters, "ops": {"add_tags": ["t"]},
            })
            assert resp.status_code == 422, filters
        mock_task_manager.enqueue_retag.assert_not_awaited()

    def test_retag_requires_an_op(self, client):
        resp = client.post("/v1/memories/retag", json={
            "filters": {"category": "decision"}, "ops": {},
        })
        assert resp.status_code == 422

    def test_retag_redis_down_503_with_project_change(self, client, mock_task_manager):
        mock_task_manager.enqueue_retag.side_effect = ConnectionError("redis down")
        resp = client.post("/v1/memories/retag", json={
            "filters": {"category": "decision"},
            "ops": {"set_project_id": "bon002"},
        })
        assert resp.status_code == 503

    def test_retag_redis_down_sync_fallback_without_project_change(
        self, client, mock_service, mock_task_manager
    ):
        mock_task_manager.enqueue_retag.side_effect = ConnectionError("redis down")
        mock_service.retag_memories.return_value = {
            "matched": 2, "updated": 2, "skipped_forbidden": 0,
            "skipped_invalid": 0, "graph_jobs": [], "dry_run": False,
        }
        resp = client.post("/v1/memories/retag", json={
            "filters": {"category": "decision"},
            "ops": {"add_tags": ["t"]},
        })
        assert resp.status_code == 200
        assert resp.json()["updated"] == 2


class TestV1Categories:
    def test_list_categories(self, client):
        resp = client.get("/v1/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "preference" in data["categories"]
        assert "tech_stack" in data["categories"]
        # The core 13 are always present. Knowledge adapters (e.g. the trading
        # adapter) additively register more categories, so assert the core set
        # is a subset rather than an exact count.
        core = {
            "preference", "personal_fact", "technical_skill", "domain_knowledge",
            "tech_stack", "convention", "architecture", "dependency",
            "decision", "interaction", "workflow", "procedure", "task_context",
        }
        assert core <= set(data["categories"])


class TestV1AsyncMemoryStatus:
    def test_returns_completed_task(self, client, mock_task_manager):
        mock_task_manager.get_status.return_value = {
            "task_id": "v1-test-task",
            "status": "completed",
            "result": {"memories": [{"id": "m1", "memory": "test"}]},
            "error": None,
        }
        resp = client.get("/v1/memories/status/v1-test-task")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    def test_returns_404_for_unknown(self, client, mock_task_manager):
        mock_task_manager.get_status.return_value = {
            "task_id": "nonexistent",
            "status": "not_found",
            "result": None,
            "error": None,
        }
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


# ══════════════════════════════════════════════
# V1 Memory model v2 endpoints
# ══════════════════════════════════════════════


class TestV1StoreRawMemoryV2:
    """Memory-model v2 — POST /v1/memories/raw with v2 fields."""

    def test_serializes_expires_at_datetime_for_enqueue(self, client, mock_task_manager):
        """expires_at as datetime gets ISO-stringified before enqueue."""
        resp = client.post("/v1/memories/raw", json={
            "content": "Time-bound fact",
            "user_id": "ehfaz",
            "category": "task_context",
            "domain": "coding",
            "observation_type": "task_plan",
            "concepts": ["next-step"],
            "source_type": "tool_extraction",
            "confidence": 0.85,
            "expires_at": "2026-12-01T00:00:00+00:00",
        })
        assert resp.status_code == 202
        kwargs = mock_task_manager.enqueue_raw.call_args[1]
        assert kwargs["domain"] == "coding"
        assert kwargs["observation_type"] == "task_plan"
        assert kwargs["concepts"] == ["next-step"]
        assert kwargs["source_type"] == "tool_extraction"
        assert kwargs["confidence"] == 0.85
        # ISO string survives the JSON-enqueue serialization
        assert kwargs["expires_at"] == "2026-12-01T00:00:00+00:00"

    def test_redis_unavailable_falls_back_to_sync(self, client, mock_task_manager, mock_service):
        """When Redis is down, the route calls service.store_raw directly."""
        from schemas import MemoryResponse

        mock_task_manager.enqueue_raw.side_effect = ConnectionError("Redis down")
        mock_service.store_raw.return_value = [
            MemoryResponse(id="m1", memory="x", category="preference"),
        ]
        resp = client.post("/v1/memories/raw", json={
            "content": "x",
            "user_id": "ehfaz",
            "category": "preference",
            "expires_at": "2026-12-01T00:00:00+00:00",  # exercises ISO → datetime path
        })
        assert resp.status_code == 200
        # store_raw was called with the deserialized datetime
        call_kwargs = mock_service.store_raw.call_args[1]
        from datetime import datetime
        assert isinstance(call_kwargs["expires_at"], datetime)

    def test_redis_unavailable_no_expires_sync_fallback(self, client, mock_task_manager, mock_service):
        """When Redis is down and no expires_at is supplied, sync fallback still succeeds.

        Note: Pydantic rejects malformed `expires_at` ISO strings at request
        validation (422), so the route's defensive `except ValueError`
        branch in the sync-fallback path is unreachable through the HTTP
        contract. That branch is exercised by the unit test
        `test_handles_iso_string_expires_at` against `store_raw_batch`.
        """
        from schemas import MemoryResponse
        mock_task_manager.enqueue_raw.side_effect = OSError("Connection refused")
        mock_service.store_raw.return_value = [MemoryResponse(id="m1", memory="x")]
        resp = client.post("/v1/memories/raw", json={
            "content": "x",
            "user_id": "ehfaz",
            "category": "preference",
        })
        assert resp.status_code == 200


class TestV1StoreRawBatch:
    """Memory-model v2 — POST /v1/memories/raw/batch."""

    def test_enqueues_batch_returns_202(self, client, mock_task_manager):
        mock_task_manager.enqueue_raw_batch = AsyncMock(return_value="batch-task-id")
        resp = client.post("/v1/memories/raw/batch", json={
            "memories": [
                {"content": "Fact A", "user_id": "ehfaz", "category": "preference"},
                {"content": "Fact B", "user_id": "ehfaz", "category": "personal_fact",
                 "domain": "personal"},
            ],
        })
        assert resp.status_code == 202
        data = resp.json()
        assert data["task_id"] == "batch-task-id"
        mock_task_manager.enqueue_raw_batch.assert_called_once()
        items = mock_task_manager.enqueue_raw_batch.call_args[1]["items"]
        assert len(items) == 2
        assert items[1]["domain"] == "personal"

    def test_serializes_expires_at_in_each_item(self, client, mock_task_manager):
        mock_task_manager.enqueue_raw_batch = AsyncMock(return_value="batch-task-id")
        resp = client.post("/v1/memories/raw/batch", json={
            "memories": [
                {"content": "x", "user_id": "u", "category": "task_context",
                 "expires_at": "2026-12-01T00:00:00+00:00"},
            ],
        })
        assert resp.status_code == 202
        items = mock_task_manager.enqueue_raw_batch.call_args[1]["items"]
        # Stored as ISO string for JSON-enqueue safety
        assert items[0]["expires_at"] == "2026-12-01T00:00:00+00:00"

    def test_invalid_category_in_one_item_400s_whole_batch(self, client, mock_task_manager):
        resp = client.post("/v1/memories/raw/batch", json={
            "memories": [
                {"content": "Fact A", "user_id": "ehfaz", "category": "preference"},
                {"content": "Fact B", "user_id": "ehfaz", "category": "BOGUS"},
            ],
        })
        assert resp.status_code == 400
        assert "Item 1" in resp.json()["detail"]

    def test_project_scope_without_project_id_400s(self, client):
        resp = client.post("/v1/memories/raw/batch", json={
            "memories": [
                {"content": "Fact A", "user_id": "ehfaz", "category": "tech_stack",
                 "scope": "project"},
            ],
        })
        assert resp.status_code == 400
        assert "Item 0" in resp.json()["detail"]

    def test_redis_unavailable_falls_back_to_sync_batch(self, client, mock_task_manager, mock_service):
        from schemas import MemoryResponse
        mock_task_manager.enqueue_raw_batch = AsyncMock(side_effect=ConnectionError("Redis down"))
        mock_service.store_raw_batch.return_value = [
            MemoryResponse(id="m1", memory="A", category="preference"),
            MemoryResponse(id="m2", memory="B", category="personal_fact"),
        ]
        resp = client.post("/v1/memories/raw/batch", json={
            "memories": [
                {"content": "A", "user_id": "ehfaz", "category": "preference"},
                {"content": "B", "user_id": "ehfaz", "category": "personal_fact"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["memories"]) == 2
        mock_service.store_raw_batch.assert_called_once()

    def test_empty_batch_422s(self, client):
        """RawMemoryBatchRequest min_length=1 rejects empty memories list."""
        resp = client.post("/v1/memories/raw/batch", json={"memories": []})
        assert resp.status_code == 422  # Pydantic validation error

    def test_oversized_batch_422s(self, client):
        """RawMemoryBatchRequest max_length=50 rejects > 50 items."""
        items = [
            {"content": f"f{i}", "user_id": "u", "category": "preference"}
            for i in range(51)
        ]
        resp = client.post("/v1/memories/raw/batch", json={"memories": items})
        assert resp.status_code == 422


class TestV1SearchV2:
    """Memory-model v2 — search route honors new optional filters."""

    def test_passes_v2_filters_through(self, client, mock_service):
        mock_service.search.return_value = []
        resp = client.post("/v1/search", json={
            "query": "anything",
            "user_id": "ehfaz",
            "domain": "research",
            "observation_type": "discovery",
            "concepts": ["gotcha", "pattern"],
        })
        assert resp.status_code == 200
        kwargs = mock_service.search.call_args[1]
        assert kwargs["domain"] == "research"
        assert kwargs["observation_type"] == "discovery"
        assert kwargs["concepts"] == ["gotcha", "pattern"]


# ══════════════════════════════════════════════
# Multi-user identity resolution (token vs body)
# ══════════════════════════════════════════════


class TestV1MultiUserIdentityResolution:
    """Routes prefer request.state.user_id (token-derived) over body user_id.

    The TestClient bypasses the actual auth middleware, so for these tests
    we monkey-patch `_resolve_user_id` or simulate a verified token by
    pushing user_id into request state via a small middleware override.
    """

    def _push_token_user_id(self, app_user_id: str):
        """Return a starlette middleware that sets request.state.user_id."""
        from starlette.middleware.base import BaseHTTPMiddleware

        class _UserIdInjector(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.user_id = app_user_id
                return await call_next(request)

        return _UserIdInjector

    def test_route_prefers_token_user_id_over_body(self, mock_service, mock_task_manager):
        """When a token attaches user_id=alice and the body says nothing,
        the route uses alice."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import main
        # Build a fresh app with our identity-injecting middleware in front.
        injector = self._push_token_user_id("alice-from-token")
        sub_app = FastAPI()
        sub_app.add_middleware(injector)
        sub_app.include_router(main.v1_router)
        client = TestClient(sub_app, raise_server_exceptions=False)

        resp = client.post("/v1/memories/raw", json={
            "content": "hello",
            "category": "preference",
            # No user_id in body — route should use the token's
        })
        assert resp.status_code in (200, 202)
        kwargs = mock_task_manager.enqueue_raw.call_args[1]
        assert kwargs["user_id"] == "alice-from-token"

    def test_token_body_mismatch_returns_400(self, mock_service, mock_task_manager):
        """If body has user_id=bob but the token says alice, reject."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import main
        injector = self._push_token_user_id("alice-from-token")
        sub_app = FastAPI()
        sub_app.add_middleware(injector)
        sub_app.include_router(main.v1_router)
        client = TestClient(sub_app, raise_server_exceptions=False)

        resp = client.post("/v1/memories/raw", json={
            "content": "hello",
            "user_id": "bob-impersonator",
            "category": "preference",
        })
        assert resp.status_code == 400
        assert "does not match" in resp.json()["detail"]

    def test_token_body_match_passes(self, mock_service, mock_task_manager):
        """body user_id == token user_id is fine."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import main
        injector = self._push_token_user_id("alice-from-token")
        sub_app = FastAPI()
        sub_app.add_middleware(injector)
        sub_app.include_router(main.v1_router)
        client = TestClient(sub_app, raise_server_exceptions=False)

        resp = client.post("/v1/memories/raw", json={
            "content": "hello",
            "user_id": "alice-from-token",
            "category": "preference",
        })
        assert resp.status_code in (200, 202)

    def test_legacy_no_token_falls_back_to_body(self, client, mock_task_manager):
        """Without a token (legacy shared-key callers), body user_id wins."""
        resp = client.post("/v1/memories/raw", json={
            "content": "hello",
            "user_id": "legacy-alice",
            "category": "preference",
        })
        assert resp.status_code in (200, 202)
        kwargs = mock_task_manager.enqueue_raw.call_args[1]
        assert kwargs["user_id"] == "legacy-alice"

    def test_search_token_body_mismatch_returns_400(self, mock_service):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import main
        injector = self._push_token_user_id("alice")
        sub_app = FastAPI()
        sub_app.add_middleware(injector)
        sub_app.include_router(main.v1_router)
        client = TestClient(sub_app, raise_server_exceptions=False)

        resp = client.post("/v1/search", json={
            "query": "anything",
            "user_id": "bob",
        })
        assert resp.status_code == 400


class TestV1SearchMultiUserFlags:
    """Visibility + include_shared are forwarded to MemoryService.search."""

    def test_visibility_private_forwarded(self, client, mock_service):
        mock_service.search.return_value = []
        resp = client.post("/v1/search", json={
            "query": "x",
            "user_id": "alice",
            "visibility": "private",
        })
        assert resp.status_code == 200
        kwargs = mock_service.search.call_args[1]
        assert kwargs["visibility"] == "private"

    def test_visibility_shared_forwarded(self, client, mock_service):
        mock_service.search.return_value = []
        resp = client.post("/v1/search", json={
            "query": "x",
            "user_id": "alice",
            "visibility": "shared",
        })
        kwargs = mock_service.search.call_args[1]
        assert kwargs["visibility"] == "shared"

    def test_include_shared_false_forwarded(self, client, mock_service):
        mock_service.search.return_value = []
        resp = client.post("/v1/search", json={
            "query": "x",
            "user_id": "alice",
            "include_shared": False,
        })
        kwargs = mock_service.search.call_args[1]
        assert kwargs["include_shared"] is False

    def test_invalid_visibility_value_rejected(self, client):
        resp = client.post("/v1/search", json={
            "query": "x",
            "user_id": "alice",
            "visibility": "not-a-real-vis",
        })
        # Pydantic enum validation produces 422
        assert resp.status_code == 422

    def test_raw_store_visibility_forwarded(self, client, mock_task_manager):
        resp = client.post("/v1/memories/raw", json={
            "content": "Project uses Postgres",
            "user_id": "alice",
            "category": "tech_stack",
            "scope": "project",
            "project_id": "neuralscape",
            "visibility": "shared",
        })
        assert resp.status_code in (200, 202)
        kwargs = mock_task_manager.enqueue_raw.call_args[1]
        assert kwargs["visibility"] == "shared"


class TestV1BatchUserIdBypass:
    """Regression for CR-05: a token-authenticated batch caller can NOT
    sidestep their token's user_id by submitting `item.user_id=""`.

    Pre-fix: items_payload was built via `d.setdefault("user_id", ...)`,
    which preserved an explicitly-empty user_id from the request body.
    Post-fix: when a token is present, every item's user_id is
    overwritten with the token's user_id.
    """

    def _push_token_user_id(self, app_user_id: str):
        from starlette.middleware.base import BaseHTTPMiddleware

        class _UserIdInjector(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.user_id = app_user_id
                return await call_next(request)

        return _UserIdInjector

    def _client_with_token(self, app_user_id: str):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import main
        injector = self._push_token_user_id(app_user_id)
        sub_app = FastAPI()
        sub_app.add_middleware(injector)
        sub_app.include_router(main.v1_router)
        return TestClient(sub_app, raise_server_exceptions=False)

    def test_empty_user_id_in_item_rejected_at_schema(self, mock_task_manager):
        """Empty user_id is rejected by schema validation (pattern requires
        non-empty), so it can't reach the route — first line of defense."""
        client = self._client_with_token("alice-from-token")
        resp = client.post("/v1/memories/raw/batch", json={
            "memories": [
                {"content": "x", "user_id": "", "category": "preference"},
            ],
        })
        assert resp.status_code == 422

    def test_missing_user_id_in_item_filled_from_token(self, mock_task_manager):
        from unittest.mock import AsyncMock
        mock_task_manager.enqueue_raw_batch = AsyncMock(return_value="batch-task")
        client = self._client_with_token("alice-from-token")
        resp = client.post("/v1/memories/raw/batch", json={
            "memories": [
                {"content": "x", "category": "preference"},
            ],
        })
        assert resp.status_code in (200, 202)
        items = mock_task_manager.enqueue_raw_batch.call_args[1]["items"]
        assert items[0]["user_id"] == "alice-from-token"

    def test_matching_user_id_in_item_kept_as_token(self, mock_task_manager):
        from unittest.mock import AsyncMock
        mock_task_manager.enqueue_raw_batch = AsyncMock(return_value="batch-task")
        client = self._client_with_token("alice-from-token")
        resp = client.post("/v1/memories/raw/batch", json={
            "memories": [
                {"content": "x", "user_id": "alice-from-token", "category": "preference"},
            ],
        })
        assert resp.status_code in (200, 202)
        items = mock_task_manager.enqueue_raw_batch.call_args[1]["items"]
        assert items[0]["user_id"] == "alice-from-token"

    def test_mismatching_user_id_in_item_returns_400(self, mock_task_manager):
        client = self._client_with_token("alice-from-token")
        resp = client.post("/v1/memories/raw/batch", json={
            "memories": [
                {"content": "x", "user_id": "bob-impersonator", "category": "preference"},
            ],
        })
        assert resp.status_code == 400
        assert "does not match" in resp.json()["detail"]

    def test_legacy_no_token_keeps_per_item_user_id(self, client, mock_task_manager):
        """Without a token (legacy shared-key path), per-item body user_id
        is trusted as before."""
        from unittest.mock import AsyncMock
        mock_task_manager.enqueue_raw_batch = AsyncMock(return_value="batch-task")
        resp = client.post("/v1/memories/raw/batch", json={
            "memories": [
                {"content": "x", "user_id": "legacy-alice", "category": "preference"},
            ],
        })
        assert resp.status_code in (200, 202)
        items = mock_task_manager.enqueue_raw_batch.call_args[1]["items"]
        assert items[0]["user_id"] == "legacy-alice"
