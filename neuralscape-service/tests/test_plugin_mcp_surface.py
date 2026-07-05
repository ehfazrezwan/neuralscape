"""Audit 27, Cluster 6 (items 31–36) — plugin/MCP surface, Python side.

Covers:
- #31: GET /v1/memories supports ``fields=index`` (index-level rows, no full
  content payloads) so the plugin read gate stops pulling 500 full payloads.
- #33: index-first steering softened from absolutes to guidance in the MCP
  tool descriptions (titles are lossy — full fetches are allowed when a title
  looks relevant).
- #35: one shared MemoryService instance across REST + mounted MCP server;
  Redis/ARQ clients cached at module scope instead of per call.
- #36: MCP category enums advertise the core 13 only (adapter categories stay
  accepted on validation); adapter categories get scope from their adapter
  profile; queued jobs referencing an unregistered adapter fail loudly.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import main
import mcp_server
import schemas
from main import app
from schemas import MemoryResponse, MemoryScope


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def mock_service():
    """Patch main's MemoryService instance."""
    mock_svc = MagicMock(name="MemoryService")
    original = main._service
    main._service = mock_svc
    yield mock_svc
    main._service = original


CORE_13 = {
    "preference", "personal_fact", "technical_skill", "domain_knowledge",
    "tech_stack", "convention", "architecture", "dependency",
    "decision", "interaction", "workflow", "procedure", "task_context",
}


# ──────────────────────────────────────────────
# #31 — GET /v1/memories?fields=index
# ──────────────────────────────────────────────


class TestListMemoriesIndexFields:
    def _rows(self):
        return [
            MemoryResponse(
                id="m1",
                memory="x" * 800,
                category="decision",
                title="Chose ARQ over Celery for the queue",
                token_estimate=210,
                tags=["queue"],
                created_at="2026-07-01T10:00:00Z",
                observation_type="decision",
            ),
            MemoryResponse(id="m2", memory="short but still content"),
        ]

    def test_fields_index_strips_content_payloads(self, mock_service):
        mock_service.list_memories.return_value = self._rows()
        client = TestClient(app)
        resp = client.get("/v1/memories", params={"user_id": "u1", "fields": "index"})
        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 2
        for row in rows:
            assert row["memory"] == ""  # no full content payloads
        # Index-level fields survive
        assert rows[0]["id"] == "m1"
        assert rows[0]["title"] == "Chose ARQ over Celery for the queue"
        assert rows[0]["token_estimate"] == 210
        assert rows[0]["tags"] == ["queue"]
        assert rows[0]["created_at"] == "2026-07-01T10:00:00Z"
        assert rows[0]["observation_type"] == "decision"

    def test_fields_index_backfills_missing_title(self, mock_service):
        """Legacy rows without a stored title get one distilled server-side —
        otherwise an index row is unmatchable/unreadable for the client."""
        mock_service.list_memories.return_value = [
            MemoryResponse(id="m3", memory="Fixed the race in worker.py by locking the pool.")
        ]
        client = TestClient(app)
        resp = client.get("/v1/memories", params={"user_id": "u1", "fields": "index"})
        assert resp.status_code == 200
        row = resp.json()[0]
        assert row["memory"] == ""
        assert row["title"]  # distilled, non-empty
        assert "worker.py" in row["title"]

    def test_default_stays_full(self, mock_service):
        mock_service.list_memories.return_value = self._rows()
        client = TestClient(app)
        resp = client.get("/v1/memories", params={"user_id": "u1"})
        assert resp.status_code == 200
        assert resp.json()[0]["memory"] == "x" * 800

    def test_invalid_fields_value_rejected(self, mock_service):
        mock_service.list_memories.return_value = []
        client = TestClient(app)
        resp = client.get("/v1/memories", params={"user_id": "u1", "fields": "everything"})
        assert resp.status_code == 422


# ──────────────────────────────────────────────
# #33 — index-first steering softened
# ──────────────────────────────────────────────


class TestSteeringSoftened:
    @pytest.mark.asyncio
    async def test_no_absolute_never_fetch_language(self):
        tools = await mcp_server.list_tools()
        by_name = {t.name: t for t in tools}
        for name in ("recall_memories", "get_memories"):
            desc = by_name[name].description
            assert "NEVER fetch full details" not in desc, name

    @pytest.mark.asyncio
    async def test_guidance_acknowledges_lossy_titles(self):
        tools = await mcp_server.list_tools()
        by_name = {t.name: t for t in tools}
        for name in ("recall_memories", "get_memories"):
            desc = by_name[name].description.lower()
            assert "titles are lossy" in desc, name
            # Token-economics rationale is kept
            assert "token" in desc, name


# ──────────────────────────────────────────────
# #35 — shared service + cached clients
# ──────────────────────────────────────────────


class TestSharedServiceAndClients:
    def test_rest_and_mcp_share_one_memory_service(self):
        assert main._service is mcp_server._service

    def test_dreaming_get_redis_is_cached(self):
        from extensions.dreaming import sweep

        original = getattr(sweep, "_redis_client", None)
        sweep._redis_client = None
        try:
            r1 = sweep._get_redis()
            r2 = sweep._get_redis()
            assert r1 is r2
        finally:
            sweep._redis_client = original

    @pytest.mark.asyncio
    async def test_mcp_arq_pool_is_cached(self, monkeypatch):
        fake_pool = MagicMock(name="ArqPool")
        create_calls = []

        async def fake_create_pool(*args, **kwargs):
            create_calls.append(1)
            return fake_pool

        monkeypatch.setattr("arq.create_pool", fake_create_pool)
        original = mcp_server._arq_pool
        mcp_server._arq_pool = None
        try:
            p1 = await mcp_server._get_arq_pool()
            p2 = await mcp_server._get_arq_pool()
            assert p1 is p2 is fake_pool
            assert len(create_calls) == 1
        finally:
            mcp_server._arq_pool = original


# ──────────────────────────────────────────────
# #36 — adapter category hygiene
# ──────────────────────────────────────────────


def _category_enums(node, path=""):
    """Recursively collect every {'category': {... 'enum': [...]}} advertisement."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "category" and isinstance(value, dict) and "enum" in value:
                found.append((path + ".category", value["enum"]))
            found.extend(_category_enums(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            found.extend(_category_enums(item, f"{path}[{i}]"))
    return found


class TestAdapterEnumHygiene:
    @pytest.mark.asyncio
    async def test_mcp_category_enums_are_core_13_only(self):
        # Precondition: at least one adapter taxonomy is registered, so the
        # shared taxonomy is wider than the core 13.
        assert "setup" in schemas.MEMORY_CATEGORIES  # trading adapter registered
        assert len(schemas.MEMORY_CATEGORIES) > 13

        tools = await mcp_server.list_tools()
        all_enums = []
        for tool in tools:
            all_enums.extend(_category_enums(tool.inputSchema, tool.name))
        assert len(all_enums) >= 5  # remember, ingest_document, ingest_text, list_memories, checkpoint
        for where, enum in all_enums:
            assert set(enum) == CORE_13, f"{where} advertises {set(enum) - CORE_13}"

    def test_adapter_categories_still_accepted_on_validation(self):
        # Parser acceptance ≠ enum advertisement: the shared taxonomy (what
        # store_raw / ingest validators check) still contains adapter categories.
        assert "setup" in schemas.MEMORY_CATEGORIES
        assert "entry_rule" in schemas.MEMORY_CATEGORIES


class TestAdapterScopeFromProfile:
    def test_project_scoped_adapter_categories(self):
        from adapters.base import ADAPTER_REGISTRY, KnowledgeAdapter, register_adapter

        cat = "wt6_scope_probe"
        adapter = KnowledgeAdapter(
            name="wt6_probe",
            categories={cat: "probe category"},
            project_categories=frozenset({cat}),
        )
        register_adapter(adapter)
        try:
            assert cat in schemas.MEMORY_CATEGORIES
            assert cat in schemas.PROJECT_CATEGORIES
            assert schemas.default_scope_for_category(cat) == MemoryScope.PROJECT
        finally:
            ADAPTER_REGISTRY.pop("wt6_probe", None)
            schemas.MEMORY_CATEGORIES.pop(cat, None)
            schemas.PROJECT_CATEGORIES.discard(cat)
            schemas.FLEXIBLE_CATEGORIES.discard(cat)

    def test_global_scoped_adapter_categories(self):
        from adapters.base import ADAPTER_REGISTRY, KnowledgeAdapter, register_adapter

        cat = "wt6_global_probe"
        adapter = KnowledgeAdapter(
            name="wt6_probe_g",
            categories={cat: "probe category"},
            global_categories=frozenset({cat}),
        )
        register_adapter(adapter)
        try:
            assert cat in schemas.GLOBAL_CATEGORIES
            assert schemas.default_scope_for_category(cat) == MemoryScope.GLOBAL
        finally:
            ADAPTER_REGISTRY.pop("wt6_probe_g", None)
            schemas.MEMORY_CATEGORIES.pop(cat, None)
            schemas.GLOBAL_CATEGORIES.discard(cat)
            schemas.FLEXIBLE_CATEGORIES.discard(cat)

    def test_code_graph_categories_are_project_scoped(self):
        from adapters.code_graph import code_graph_available

        if not code_graph_available():
            pytest.skip("code-graph extra not installed")
        for cat in ("module", "boundary", "invariant", "rationale", "hotspot"):
            assert cat in schemas.PROJECT_CATEGORIES, cat
            assert schemas.default_scope_for_category(cat) == MemoryScope.PROJECT, cat


class TestStrictAdapterResolutionForQueuedJobs:
    def test_require_adapter_known_and_default(self):
        from adapters import DEFAULT_ADAPTER, require_adapter

        assert require_adapter(None) is DEFAULT_ADAPTER
        assert require_adapter("default") is DEFAULT_ADAPTER
        assert require_adapter("trading_strategy").name == "trading_strategy"

    def test_require_adapter_unknown_raises(self):
        from adapters import UnknownAdapterError, require_adapter

        with pytest.raises(UnknownAdapterError, match="wt6_ghost"):
            require_adapter("wt6_ghost")

    def test_pipeline_rejects_unregistered_adapter(self):
        from adapters import UnknownAdapterError
        from ingest.pipeline import IngestDoc, ingest_document

        doc = IngestDoc(
            content="some document body",
            source={"connector_id": "c1", "connector_type": "mcp"},
            user_id="u1",
            adapter="wt6_ghost",
        )
        service = MagicMock(name="MemoryService")
        with pytest.raises(UnknownAdapterError, match="wt6_ghost"):
            ingest_document(service, doc)
        service.store_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_graph_enrichment_job_fails_on_unregistered_adapter(self):
        from adapters import UnknownAdapterError
        import worker

        service = MagicMock(name="MemoryService")
        service.get_memory.return_value = {"id": "m1"}
        ctx = {"service": service}
        with pytest.raises(UnknownAdapterError, match="wt6_ghost"):
            await worker.process_graph_enrichment(
                ctx, "m1", "content", "u1", adapter="wt6_ghost"
            )
        service.enrich_graph.assert_not_called()
