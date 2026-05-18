"""Tests for MCP tools."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import mcp_server
from schemas import ContextResponse, MemoryResponse


@pytest.fixture(autouse=True)
def mock_mcp_service():
    """Patch the MemoryService in the MCP server module."""
    mock_svc = MagicMock(name="MemoryService")
    original = mcp_server._service
    mcp_server._service = mock_svc
    yield mock_svc
    mcp_server._service = original


@pytest.fixture(autouse=True)
def mock_task_manager():
    """Patch the TaskManager in the MCP server module."""
    mock_tm = MagicMock(name="TaskManager")
    mock_tm.enqueue_raw = AsyncMock(return_value="task-123")
    mock_tm.enqueue_store = AsyncMock(return_value="task-456")
    mock_tm.wait_for_result = AsyncMock(return_value={
        "task_id": "task-123",
        "status": "completed",
        "result": None,
        "error": None,
    })
    original = mcp_server._task_manager
    mcp_server._task_manager = mock_tm
    yield mock_tm
    mcp_server._task_manager = original


# ──────────────────────────────────────────────
# Tool listing
# ──────────────────────────────────────────────


class TestListTools:
    @pytest.mark.asyncio
    async def test_returns_7_tools(self):
        tools = await mcp_server.list_tools()
        assert len(tools) == 7

    @pytest.mark.asyncio
    async def test_tool_names(self):
        tools = await mcp_server.list_tools()
        names = {t.name for t in tools}
        expected = {
            "recall_memories",
            "remember",
            "remember_conversation",
            "get_project_context",
            "search_knowledge_graph",
            "list_memories",
            "delete_memories",
        }
        assert names == expected

    @pytest.mark.asyncio
    async def test_all_tools_have_input_schema(self):
        tools = await mcp_server.list_tools()
        for tool in tools:
            assert tool.inputSchema is not None
            assert "properties" in tool.inputSchema


# ──────────────────────────────────────────────
# Tool execution
# ──────────────────────────────────────────────


class TestRecallMemoriesTool:
    @pytest.mark.asyncio
    async def test_basic_search(self, mock_mcp_service):
        mock_mcp_service.search.return_value = [
            MemoryResponse(id="m1", memory="Prefers tabs", score=0.95, category="preference", source="vector")
        ]
        result = await mcp_server.call_tool("recall_memories", {
            "query": "indentation preferences",
            "user_id": "ehfaz",
        })
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert len(data) == 1
        assert data[0]["memory"] == "Prefers tabs"
        assert data[0]["source"] == "vector"

    @pytest.mark.asyncio
    async def test_search_includes_graph_source(self, mock_mcp_service):
        mock_mcp_service.search.return_value = [
            MemoryResponse(id="v1", memory="Prefers tabs", score=0.95, source="vector"),
            MemoryResponse(id="g1", memory="User uses Python", source="graph"),
        ]
        result = await mcp_server.call_tool("recall_memories", {
            "query": "Python",
            "user_id": "ehfaz",
        })
        data = json.loads(result[0].text)
        sources = [r["source"] for r in data]
        assert "vector" in sources
        assert "graph" in sources

    @pytest.mark.asyncio
    async def test_search_with_project(self, mock_mcp_service):
        mock_mcp_service.search.return_value = []
        await mcp_server.call_tool("recall_memories", {
            "query": "tech stack",
            "user_id": "ehfaz",
            "project_id": "my-project",
        })
        mock_mcp_service.search.assert_called_once()
        call_kwargs = mock_mcp_service.search.call_args[1]
        assert call_kwargs["project_id"] == "my-project"

    @pytest.mark.asyncio
    async def test_search_with_categories(self, mock_mcp_service):
        mock_mcp_service.search.return_value = []
        await mcp_server.call_tool("recall_memories", {
            "query": "coding style",
            "user_id": "ehfaz",
            "categories": ["preference", "convention"],
        })
        call_kwargs = mock_mcp_service.search.call_args[1]
        assert call_kwargs["categories"] == ["preference", "convention"]


class TestRememberTool:
    @pytest.mark.asyncio
    async def test_stores_fact(self, mock_mcp_service, mock_task_manager):
        result = await mcp_server.call_tool("remember", {
            "content": "Prefers tabs over spaces",
            "user_id": "ehfaz",
            "category": "preference",
        })
        data = json.loads(result[0].text)
        assert data["status"] == "accepted"
        assert data["task_id"] == "task-123"
        mock_task_manager.enqueue_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_project_category_sets_project_scope(self, mock_mcp_service, mock_task_manager):
        await mcp_server.call_tool("remember", {
            "content": "Uses FastAPI 0.115",
            "user_id": "ehfaz",
            "category": "tech_stack",
            "project_id": "my-project",
        })
        call_kwargs = mock_task_manager.enqueue_raw.call_args[1]
        assert call_kwargs["scope"] == "project"
        assert call_kwargs["project_id"] == "my-project"

    @pytest.mark.asyncio
    async def test_v2_fields_forwarded_to_enqueue(self, mock_mcp_service, mock_task_manager):
        """Memory-model v2 — domain/observation_type/concepts/etc. flow through to enqueue_raw."""
        await mcp_server.call_tool("remember", {
            "content": "Decided to use feature flags via GrowthBook",
            "user_id": "ehfaz",
            "category": "decision",
            "project_id": "my-project",
            "domain": "coding",
            "observation_type": "decision",
            "concepts": ["why-it-exists", "trade-off"],
            "source_type": "tool_extraction",
            "confidence": 0.85,
        })
        call_kwargs = mock_task_manager.enqueue_raw.call_args[1]
        assert call_kwargs["domain"] == "coding"
        assert call_kwargs["observation_type"] == "decision"
        assert call_kwargs["concepts"] == ["why-it-exists", "trade-off"]
        assert call_kwargs["source_type"] == "tool_extraction"
        assert call_kwargs["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_redis_unavailable_falls_back_to_sync_with_iso_expires(
        self, mock_mcp_service, mock_task_manager
    ):
        """When Redis is down, sync-fallback re-hydrates expires_at ISO string to datetime."""
        from datetime import datetime as _dt

        mock_task_manager.enqueue_raw.side_effect = ConnectionError("Redis down")
        mock_mcp_service.store_raw.return_value = [
            MemoryResponse(id="m1", memory="x", category="task_context"),
        ]

        result = await mcp_server.call_tool("remember", {
            "content": "x",
            "user_id": "ehfaz",
            "category": "task_context",
            "expires_at": "2026-12-01T00:00:00+00:00",
            "domain": "personal",
        })
        data = json.loads(result[0].text)
        assert data["status"] == "completed"
        assert data["fallback"] == "sync"

        # The sync path got expires_at as a real datetime, not ISO string
        call_kwargs = mock_mcp_service.store_raw.call_args[1]
        assert isinstance(call_kwargs["expires_at"], _dt)
        assert call_kwargs["domain"] == "personal"

    @pytest.mark.asyncio
    async def test_redis_unavailable_invalid_iso_drops(self, mock_mcp_service, mock_task_manager):
        """Malformed expires_at ISO string in fallback path drops to None."""
        mock_task_manager.enqueue_raw.side_effect = OSError("Connection refused")
        mock_mcp_service.store_raw.return_value = [
            MemoryResponse(id="m1", memory="x"),
        ]
        await mcp_server.call_tool("remember", {
            "content": "x",
            "user_id": "ehfaz",
            "category": "preference",
            "expires_at": "not-a-date",
        })
        # store_raw was called; expires_at gets dropped (None) so it's not in kwargs
        # at all (we filter `None` values before passing).
        assert "expires_at" not in mock_mcp_service.store_raw.call_args[1]

    @pytest.mark.asyncio
    async def test_wait_true_returns_completed_status(self, mock_mcp_service, mock_task_manager):
        mock_task_manager.wait_for_result.return_value = {
            "task_id": "task-123",
            "status": "completed",
            "result": {"memories": [{"id": "m1"}]},
            "error": None,
        }
        result = await mcp_server.call_tool("remember", {
            "content": "x",
            "user_id": "ehfaz",
            "category": "preference",
            "wait": True,
        })
        data = json.loads(result[0].text)
        assert data["status"] == "completed"
        assert data["task_id"] == "task-123"


class TestRememberConversationTool:
    @pytest.mark.asyncio
    async def test_extracts_from_messages(self, mock_mcp_service, mock_task_manager):
        result = await mcp_server.call_tool("remember_conversation", {
            "messages": [
                {"role": "user", "content": "I use Python and prefer tabs"},
            ],
            "user_id": "ehfaz",
        })
        data = json.loads(result[0].text)
        assert data["status"] == "accepted"
        assert data["task_id"] == "task-456"
        mock_task_manager.enqueue_store.assert_called_once()


class TestGetProjectContextTool:
    @pytest.mark.asyncio
    async def test_returns_organized_context(self, mock_mcp_service):
        mock_mcp_service.get_project_context.return_value = ContextResponse(
            user_id="ehfaz",
            project_id="my-project",
            categories={
                "preference": [MemoryResponse(id="m1", memory="Prefers tabs")],
                "tech_stack": [MemoryResponse(id="m2", memory="Uses FastAPI")],
            },
        )
        result = await mcp_server.call_tool("get_project_context", {
            "user_id": "ehfaz",
            "project_id": "my-project",
        })
        data = json.loads(result[0].text)
        assert data["project_id"] == "my-project"
        assert "preference" in data["categories"]
        assert "tech_stack" in data["categories"]


class TestSearchKnowledgeGraphTool:
    @pytest.mark.asyncio
    async def test_returns_graph_results(self, mock_mcp_service):
        mock_mcp_service.search_graph.return_value = {
            "edges": [{"uuid": "e1", "name": "uses", "fact": "Uses Python"}],
            "nodes": [{"uuid": "n1", "name": "Python", "summary": "Language"}],
            "episodes": [],
            "communities": [],
        }
        result = await mcp_server.call_tool("search_knowledge_graph", {
            "query": "Python",
            "user_id": "ehfaz",
        })
        data = json.loads(result[0].text)
        assert len(data["edges"]) == 1
        assert len(data["nodes"]) == 1


class TestListMemoriesTool:
    @pytest.mark.asyncio
    async def test_lists_memories(self, mock_mcp_service):
        mock_mcp_service.list_memories.return_value = [
            MemoryResponse(id="m1", memory="fact1"),
            MemoryResponse(id="m2", memory="fact2"),
        ]
        result = await mcp_server.call_tool("list_memories", {
            "user_id": "ehfaz",
        })
        data = json.loads(result[0].text)
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_lists_with_filters(self, mock_mcp_service):
        mock_mcp_service.list_memories.return_value = []
        await mcp_server.call_tool("list_memories", {
            "user_id": "ehfaz",
            "scope": "global",
            "category": "preference",
        })
        call_kwargs = mock_mcp_service.list_memories.call_args[1]
        assert call_kwargs["scope"] == "global"
        assert call_kwargs["category"] == "preference"


class TestDeleteMemoriesTool:
    @pytest.mark.asyncio
    async def test_delete_by_id(self, mock_mcp_service):
        mock_mcp_service.delete_memory.return_value = {"message": "Memory deleted successfully!"}
        result = await mcp_server.call_tool("delete_memories", {
            "user_id": "ehfaz",
            "memory_id": "m1",
        })
        data = json.loads(result[0].text)
        assert "deleted" in data["message"].lower()
        mock_mcp_service.delete_memory.assert_called_once_with("m1")

    @pytest.mark.asyncio
    async def test_bulk_delete(self, mock_mcp_service):
        mock_mcp_service.delete_memories.return_value = {"message": "Deleted 5 memories"}
        result = await mcp_server.call_tool("delete_memories", {
            "user_id": "ehfaz",
            "scope": "project",
            "project_id": "my-project",
        })
        data = json.loads(result[0].text)
        assert "deleted" in data["message"].lower()


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, mock_mcp_service):
        result = await mcp_server.call_tool("nonexistent_tool", {
            "user_id": "ehfaz",
        })
        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_mcp_service):
        mock_mcp_service.search.side_effect = Exception("Database connection failed")
        result = await mcp_server.call_tool("recall_memories", {
            "query": "test",
            "user_id": "ehfaz",
        })
        data = json.loads(result[0].text)
        assert "error" in data
        assert "Database connection failed" in data["error"]
