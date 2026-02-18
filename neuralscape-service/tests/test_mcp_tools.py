"""Tests for MCP tools."""

import json
from unittest.mock import MagicMock, patch

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
            MemoryResponse(id="m1", memory="Prefers tabs", score=0.95, category="preference")
        ]
        result = await mcp_server.call_tool("recall_memories", {
            "query": "indentation preferences",
            "user_id": "ehfaz",
        })
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert len(data) == 1
        assert data[0]["memory"] == "Prefers tabs"

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
    async def test_stores_fact(self, mock_mcp_service):
        mock_mcp_service.store_raw.return_value = [
            MemoryResponse(id="m1", memory="Prefers tabs", category="preference")
        ]
        result = await mcp_server.call_tool("remember", {
            "content": "Prefers tabs over spaces",
            "user_id": "ehfaz",
            "category": "preference",
        })
        data = json.loads(result[0].text)
        assert data["status"] == "stored"
        mock_mcp_service.store_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_project_category_sets_project_scope(self, mock_mcp_service):
        mock_mcp_service.store_raw.return_value = []
        await mcp_server.call_tool("remember", {
            "content": "Uses FastAPI 0.115",
            "user_id": "ehfaz",
            "category": "tech_stack",
            "project_id": "my-project",
        })
        call_kwargs = mock_mcp_service.store_raw.call_args[1]
        assert call_kwargs["scope"] == "project"
        assert call_kwargs["project_id"] == "my-project"


class TestRememberConversationTool:
    @pytest.mark.asyncio
    async def test_extracts_from_messages(self, mock_mcp_service):
        mock_mcp_service.extract_and_store.return_value = [
            MemoryResponse(id="m1", memory="Uses Python", category="technical_skill"),
            MemoryResponse(id="m2", memory="Prefers tabs", category="preference"),
        ]
        result = await mcp_server.call_tool("remember_conversation", {
            "messages": [
                {"role": "user", "content": "I use Python and prefer tabs"},
            ],
            "user_id": "ehfaz",
        })
        data = json.loads(result[0].text)
        assert data["status"] == "stored"
        assert data["count"] == 2


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
