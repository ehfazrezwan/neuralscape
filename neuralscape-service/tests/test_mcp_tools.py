"""Tests for MCP tools."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import mcp_server
from config import settings
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
    mock_tm.enqueue_retag = AsyncMock(return_value="task-retag")
    mock_tm.enqueue_graph_enrichment = AsyncMock(return_value="task-graph")
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
    async def test_returns_17_tools(self):
        tools = await mcp_server.list_tools()
        assert len(tools) == 17

    @pytest.mark.asyncio
    async def test_tool_names(self):
        tools = await mcp_server.list_tools()
        names = {t.name for t in tools}
        expected = {
            "recall_memories",
            "get_memories",
            "timeline",
            "remember",
            "remember_conversation",
            "ingest_document",
            "ingest_text",
            "get_project_context",
            "search_knowledge_graph",
            "list_memories",
            "list_projects",
            "list_processes",
            "get_process",
            "delete_memories",
            "edit_memory",
            "retag_memories",
            "get_reasoning_chain",
            "schedule_dream",
            "get_card",
        }
        assert names == expected

    @pytest.mark.asyncio
    async def test_all_tools_have_input_schema(self):
        tools = await mcp_server.list_tools()
        for tool in tools:
            assert tool.inputSchema is not None
            assert "properties" in tool.inputSchema

    @pytest.mark.asyncio
    async def test_visibility_enums_include_standard(self):
        # Regression: the MCP SDK validates arguments against these enums BEFORE
        # the handler runs, so a stale ['private','shared'] enum silently blocks
        # writing/filtering the 'standard' tier even for a dictator.
        tools = {t.name: t for t in await mcp_server.list_tools()}
        for name in ("remember", "recall_memories", "ingest_document", "ingest_text"):
            enum = tools[name].inputSchema["properties"]["visibility"]["enum"]
            assert "standard" in enum, f"{name} visibility enum missing 'standard': {enum}"


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
        # Default page size is bounded (25) so large projects don't overflow the
        # agent tool-result token limit.
        assert mock_mcp_service.get_project_context.call_args.kwargs["limit"] == 25


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


class TestListProjectsTool:
    @pytest.mark.asyncio
    async def test_lists_projects(self, mock_mcp_service):
        mock_mcp_service.list_projects.return_value = ["lightpath", "neuralscape"]
        result = await mcp_server.call_tool("list_projects", {"user_id": "ehfaz"})
        data = json.loads(result[0].text)
        assert data["projects"] == ["lightpath", "neuralscape"]
        mock_mcp_service.list_projects.assert_called_once_with(user_id="ehfaz")

    @pytest.mark.asyncio
    async def test_lists_projects_without_user_id(self, mock_mcp_service):
        """user_id is optional in the schema — omitting it falls back to the
        token identity / default, never errors."""
        mock_mcp_service.list_projects.return_value = []
        result = await mcp_server.call_tool("list_projects", {})
        data = json.loads(result[0].text)
        assert data["projects"] == []
        # Identity contract: with no token and no arg, fall back to default_user_id.
        mock_mcp_service.list_projects.assert_called_once_with(
            user_id=settings.default_user_id
        )


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
        # Caller identity is now passed so standard-tier deletes can be gated.
        mock_mcp_service.delete_memory.assert_called_once_with("m1", "ehfaz")

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


class TestEditMemoryTool:
    @pytest.mark.asyncio
    async def test_edit_metadata(self, mock_mcp_service, mock_task_manager):
        mock_mcp_service.patch_memory.return_value = {
            "memory": MemoryResponse(id="m1", memory="x", category="decision"),
            "graph_job": None,
            "graph": "unchanged",
        }
        result = await mcp_server.call_tool("edit_memory", {
            "memory_id": "m1",
            "user_id": "robb",
            "tags": ["project:bon002"],
        })
        data = json.loads(result[0].text)
        assert data["status"] == "ok" and data["graph"] == "unchanged"
        # Presence-keyed: only the sent field reaches the service
        args = mock_mcp_service.patch_memory.call_args.args
        assert args == ("m1", "robb", {"tags": ["project:bon002"]})
        mock_task_manager.enqueue_graph_enrichment.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_edit_explicit_null_clears(self, mock_mcp_service):
        mock_mcp_service.patch_memory.return_value = {
            "memory": None, "graph_job": None, "graph": "unchanged",
        }
        await mcp_server.call_tool("edit_memory", {
            "memory_id": "m1", "user_id": "e", "project_id": None,
        })
        assert mock_mcp_service.patch_memory.call_args.args[2] == {"project_id": None}

    @pytest.mark.asyncio
    async def test_edit_enqueues_graph_job(self, mock_mcp_service, mock_task_manager):
        job = {"memory_id": "m1", "content": "new", "user_id": "e",
               "project_id": "p1", "visibility": "shared", "source_ref": None}
        mock_mcp_service.patch_memory.return_value = {
            "memory": None, "graph_job": job, "graph": "migration_pending",
        }
        result = await mcp_server.call_tool("edit_memory", {
            "memory_id": "m1", "user_id": "e", "project_id": "p1",
        })
        data = json.loads(result[0].text)
        assert data["graph"] == "migration_queued"
        assert data["graph_task_id"] == "task-graph"
        mock_task_manager.enqueue_graph_enrichment.assert_awaited_once_with(**job)

    @pytest.mark.asyncio
    async def test_edit_permission_error_surfaced(self, mock_mcp_service):
        mock_mcp_service.patch_memory.side_effect = PermissionError(
            "Only the memory's owner may edit its content"
        )
        result = await mcp_server.call_tool("edit_memory", {
            "memory_id": "m1", "user_id": "robb", "content": "rewrite",
        })
        data = json.loads(result[0].text)
        assert "owner" in data["error"]

    @pytest.mark.asyncio
    async def test_edit_no_fields_rejected(self, mock_mcp_service):
        result = await mcp_server.call_tool("edit_memory", {"memory_id": "m1", "user_id": "e"})
        data = json.loads(result[0].text)
        assert "error" in data
        mock_mcp_service.patch_memory.assert_not_called()


class TestRetagMemoriesTool:
    @pytest.mark.asyncio
    async def test_retag_enqueues(self, mock_mcp_service, mock_task_manager):
        result = await mcp_server.call_tool("retag_memories", {
            "user_id": "robb",
            "project_id": "neuralscape",
            "add_tags": ["project:bon002"],
        })
        data = json.loads(result[0].text)
        assert data == {"status": "accepted", "task_id": "task-retag"}
        caller, filters, ops = mock_task_manager.enqueue_retag.await_args.args
        assert caller == "robb"
        assert filters == {"project_id": "neuralscape"}
        assert ops == {"add_tags": ["project:bon002"]}
        mock_mcp_service.retag_memories.assert_not_called()

    @pytest.mark.asyncio
    async def test_retag_dry_run_synchronous(self, mock_mcp_service, mock_task_manager):
        mock_mcp_service.retag_memories.return_value = {
            "matched": 3, "updated": 2, "skipped_forbidden": 1,
            "skipped_invalid": 0, "graph_jobs": [], "dry_run": True,
        }
        result = await mcp_server.call_tool("retag_memories", {
            "user_id": "robb",
            "category": "decision",
            "add_tags": ["t"],
            "dry_run": True,
        })
        data = json.loads(result[0].text)
        assert data["matched"] == 3 and "graph_jobs" not in data
        mock_task_manager.enqueue_retag.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retag_requires_filter_and_op(self, mock_task_manager):
        no_filter = await mcp_server.call_tool("retag_memories", {
            "user_id": "robb", "add_tags": ["t"],
        })
        assert "filter" in json.loads(no_filter[0].text)["error"]
        no_op = await mcp_server.call_tool("retag_memories", {
            "user_id": "robb", "category": "decision",
        })
        assert "operation" in json.loads(no_op[0].text)["error"]
        mock_task_manager.enqueue_retag.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retag_empty_filter_values_treated_as_absent(self, mock_task_manager):
        """REGRESSION: `tags_contains: []` must not satisfy the filter guard —
        it builds no Qdrant condition and would sweep."""
        result = await mcp_server.call_tool("retag_memories", {
            "user_id": "robb", "tags_contains": [], "add_tags": ["t"],
        })
        assert "filter" in json.loads(result[0].text)["error"]
        mock_task_manager.enqueue_retag.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retag_null_set_project_preserved(self, mock_task_manager):
        await mcp_server.call_tool("retag_memories", {
            "user_id": "robb",
            "tags_contains": ["strategy:naked-forex"],
            "set_project_id": None,
        })
        _, _, ops = mock_task_manager.enqueue_retag.await_args.args
        assert ops == {"set_project_id": None}


class TestStandardWriteGate:
    """The `remember` tool must reject non-dictator writes to the standard tier
    BEFORE enqueue, returning an actionable error rather than a lost job."""

    @pytest.mark.asyncio
    async def test_disabled_tier_rejected(self, mock_mcp_service, mock_task_manager, monkeypatch):
        monkeypatch.setattr(settings, "standards_enabled", False)
        result = await mcp_server.call_tool("remember", {
            "content": "All decks use the Opti template.",
            "category": "convention",
            "user_id": "mark",
            "visibility": "standard",
        })
        data = json.loads(result[0].text)
        assert "error" in data
        mock_task_manager.enqueue_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_dictator_rejected(self, mock_mcp_service, mock_task_manager, monkeypatch):
        monkeypatch.setattr(settings, "standards_enabled", True)
        monkeypatch.setattr(settings, "dictator_user_ids", "mark")
        result = await mcp_server.call_tool("remember", {
            "content": "All decks use the Opti template.",
            "category": "convention",
            "user_id": "alice",
            "visibility": "standard",
        })
        data = json.loads(result[0].text)
        assert "error" in data and "not authorized" in data["error"]
        mock_task_manager.enqueue_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_dictator_allowed(self, mock_mcp_service, mock_task_manager, monkeypatch):
        monkeypatch.setattr(settings, "standards_enabled", True)
        monkeypatch.setattr(settings, "dictator_user_ids", "mark")
        result = await mcp_server.call_tool("remember", {
            "content": "All decks use the Opti template.",
            "category": "convention",
            "user_id": "mark",
            "visibility": "standard",
        })
        data = json.loads(result[0].text)
        assert data.get("status") == "accepted"
        mock_task_manager.enqueue_raw.assert_called_once()


class TestProcessTools:
    @pytest.mark.asyncio
    async def test_list_processes(self, mock_mcp_service):
        mock_mcp_service.list_processes.return_value = [{"slug": "qbr", "title": "Quarterly Business Review"}]
        result = await mcp_server.call_tool("list_processes", {"project_id": "svc"})
        data = json.loads(result[0].text)
        assert data["processes"][0]["slug"] == "qbr"
        mock_mcp_service.list_processes.assert_called_once_with(project_id="svc")

    @pytest.mark.asyncio
    async def test_get_process(self, mock_mcp_service):
        mock_mcp_service.get_process.return_value = {
            "slug": "qbr", "title": "QBR", "definition": "Run a QBR", "steps": ["a", "b"],
        }
        result = await mcp_server.call_tool("get_process", {"slug": "qbr"})
        data = json.loads(result[0].text)
        assert data["slug"] == "qbr" and data["steps"] == ["a", "b"]
        mock_mcp_service.get_process.assert_called_once_with("qbr", None)

    @pytest.mark.asyncio
    async def test_get_process_unknown_returns_error(self, mock_mcp_service):
        mock_mcp_service.get_process.return_value = None
        result = await mcp_server.call_tool("get_process", {"slug": "nope"})
        data = json.loads(result[0].text)
        assert "error" in data


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


class TestGetReasoningChainTool:
    @pytest.mark.asyncio
    async def test_returns_chain(self, mock_mcp_service):
        mock_mcp_service.get_reasoning_chain.return_value = {
            "memory_id": "m1", "content": "insight",
            "epistemic_level": "inductive",
            "children": [
                {"memory_id": "p1", "content": "premise",
                 "epistemic_level": "explicit", "children": []},
            ],
        }
        result = await mcp_server.call_tool(
            "get_reasoning_chain", {"memory_id": "m1", "max_depth": 5}
        )
        data = json.loads(result[0].text)
        assert data["status"] == "ok"
        assert data["chain"]["epistemic_level"] == "inductive"
        assert data["chain"]["children"][0]["memory_id"] == "p1"
        mock_mcp_service.get_reasoning_chain.assert_called_once_with("m1", 5)

    @pytest.mark.asyncio
    async def test_clamps_max_depth_and_defaults(self, mock_mcp_service):
        mock_mcp_service.get_reasoning_chain.return_value = {"memory_id": "m1", "children": []}
        await mcp_server.call_tool("get_reasoning_chain", {"memory_id": "m1"})
        assert mock_mcp_service.get_reasoning_chain.call_args[0][1] == 3
        await mcp_server.call_tool(
            "get_reasoning_chain", {"memory_id": "m1", "max_depth": 99}
        )
        assert mock_mcp_service.get_reasoning_chain.call_args[0][1] == 10

    @pytest.mark.asyncio
    async def test_missing_memory_returns_error(self, mock_mcp_service):
        mock_mcp_service.get_reasoning_chain.return_value = None
        result = await mcp_server.call_tool(
            "get_reasoning_chain", {"memory_id": "ghost"}
        )
        data = json.loads(result[0].text)
        assert "error" in data and "ghost" in data["error"]


class TestRememberProvenanceValidation:
    """A1 fields are validated server-side — MCP JSON-schema hints alone
    don't bind a client, and a bad value must fail fast, not as a silent
    background job failure."""

    @pytest.mark.asyncio
    async def test_oversized_derived_from_rejected(self, mock_task_manager):
        result = await mcp_server.call_tool("remember", {
            "content": "x", "user_id": "u", "category": "preference",
            "derived_from": [f"m{i}" for i in range(11)],
        })
        data = json.loads(result[0].text)
        assert "error" in data and "derived_from" in data["error"]
        mock_task_manager.enqueue_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_epistemic_level_rejected(self, mock_task_manager):
        result = await mcp_server.call_tool("remember", {
            "content": "x", "user_id": "u", "category": "preference",
            "epistemic_level": "vibes",
        })
        data = json.loads(result[0].text)
        assert "error" in data and "epistemic_level" in data["error"]
        mock_task_manager.enqueue_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_provenance_forwarded(self, mock_task_manager):
        result = await mcp_server.call_tool("remember", {
            "content": "x", "user_id": "u", "category": "preference",
            "derived_from": ["m1", "m2"], "epistemic_level": "deductive",
        })
        data = json.loads(result[0].text)
        assert data["status"] == "accepted"
        kwargs = mock_task_manager.enqueue_raw.call_args[1]
        assert kwargs["derived_from"] == ["m1", "m2"]
        assert kwargs["epistemic_level"] == "deductive"


# ──────────────────────────────────────────────
# schedule_dream (A3-lite manual trigger)
# ──────────────────────────────────────────────


class TestScheduleDreamTool:
    @pytest.mark.asyncio
    async def test_refuses_when_dreaming_disabled(self, monkeypatch):
        from extensions.dreaming.config import dreaming_settings

        monkeypatch.setattr(dreaming_settings, "enabled", False)
        result = await mcp_server.call_tool("schedule_dream", {})
        data = json.loads(result[0].text)
        assert "error" in data and "DREAMING_ENABLED" in data["error"]

    @pytest.mark.asyncio
    async def test_enqueues_onto_graph_queue(self, monkeypatch):
        """The tool goes through the same arq path as the /run route:
        run_dream_sweep(pool, dry_run, force) on the graph worker queue."""
        from extensions.dreaming.config import dreaming_settings

        monkeypatch.setattr(dreaming_settings, "enabled", True)
        arq_pool = MagicMock()
        arq_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="job-1"))
        arq_pool.close = AsyncMock()
        import arq

        monkeypatch.setattr(arq, "create_pool", AsyncMock(return_value=arq_pool))

        result = await mcp_server.call_tool(
            "schedule_dream", {"pool": "user--alice", "dry_run": True}
        )
        data = json.loads(result[0].text)
        assert data == {
            "status": "enqueued",
            "job_id": "job-1",
            "poll": "/v1/extensions/dreaming/status",
        }
        args, kwargs = arq_pool.enqueue_job.call_args
        assert args == ("run_dream_sweep", "user--alice", True, False)
        assert kwargs["_queue_name"] == settings.graph_queue_name
        arq_pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_force_bypasses_disabled_and_forwards(self, monkeypatch):
        from extensions.dreaming.config import dreaming_settings

        monkeypatch.setattr(dreaming_settings, "enabled", False)
        arq_pool = MagicMock()
        arq_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="job-2"))
        arq_pool.close = AsyncMock()
        import arq

        monkeypatch.setattr(arq, "create_pool", AsyncMock(return_value=arq_pool))

        result = await mcp_server.call_tool("schedule_dream", {"force": True})
        data = json.loads(result[0].text)
        assert data["status"] == "enqueued" and data["job_id"] == "job-2"
        args, _ = arq_pool.enqueue_job.call_args
        assert args == ("run_dream_sweep", None, False, True)
