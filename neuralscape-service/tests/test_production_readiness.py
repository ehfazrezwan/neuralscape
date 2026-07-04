"""Tests for production readiness improvements.

Covers: config validation, Redis URL parser, search dedup/limit,
delete graph cleanup, bridge timeout, Redis fallback, input validation,
global exception handler, thread-safe init, worker idempotency,
and LLM retry with exponential backoff.
"""

import concurrent.futures
import hashlib
import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main
import mcp_server
from config import Settings, parse_redis_settings
from memory_service import MemoryService, _is_transient, retry_transient
from schemas import MemoryResponse
from task_manager import _generate_job_id


# ══════════════════════════════════════════════
# P0-5: Config validation
# ══════════════════════════════════════════════


class TestConfigValidation:
    def test_validate_passes_with_all_fields_set(self):
        s = Settings(
            google_api_key="test-key",
            neo4j_password="test-pass",
            neo4j_uri="neo4j://localhost:7687",
            redis_url="redis://localhost:6379",
        )
        # Should not raise
        s.validate_required()

    def test_validate_fails_when_google_api_key_missing(self):
        s = Settings(
            google_api_key="",
            neo4j_password="test-pass",
        )
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            s.validate_required()

    def test_validate_fails_when_neo4j_password_missing(self):
        s = Settings(
            google_api_key="test-key",
            neo4j_password="",
        )
        with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
            s.validate_required()

    def test_validate_collects_all_errors(self):
        s = Settings(
            google_api_key="",
            neo4j_password="",
            neo4j_uri="",
            redis_url="",
        )
        with pytest.raises(ValueError) as exc_info:
            s.validate_required()
        error_msg = str(exc_info.value)
        assert "GOOGLE_API_KEY" in error_msg
        assert "NEO4J_PASSWORD" in error_msg
        assert "NEO4J_URI" in error_msg
        assert "REDIS_URL" in error_msg


# ══════════════════════════════════════════════
# P0-6: Shared Redis URL parser
# ══════════════════════════════════════════════


class TestParseRedisSettings:
    def test_simple_url(self):
        with patch("config.settings") as mock_settings:
            mock_settings.redis_url = "redis://localhost:6379"
            rs = parse_redis_settings()
            assert rs.host == "localhost"
            assert rs.port == 6379
            assert rs.database == 0
            assert rs.password is None

    def test_url_with_db(self):
        with patch("config.settings") as mock_settings:
            mock_settings.redis_url = "redis://myhost:6380/2"
            rs = parse_redis_settings()
            assert rs.host == "myhost"
            assert rs.port == 6380
            assert rs.database == 2

    def test_url_with_password(self):
        with patch("config.settings") as mock_settings:
            mock_settings.redis_url = "redis://user:s3cret@redis-host:6379/1"
            rs = parse_redis_settings()
            assert rs.host == "redis-host"
            assert rs.port == 6379
            assert rs.database == 1
            assert rs.password == "s3cret"

    def test_url_with_default_port(self):
        with patch("config.settings") as mock_settings:
            mock_settings.redis_url = "redis://myhost"
            rs = parse_redis_settings()
            assert rs.host == "myhost"
            assert rs.port == 6379
            assert rs.database == 0


# ══════════════════════════════════════════════
# P0-2: Search dedup and limit enforcement
# ══════════════════════════════════════════════


class TestDeduplicateResponses:
    def setup_method(self):
        self.svc = MemoryService()

    def test_removes_exact_duplicate(self):
        vector = [MemoryResponse(id="v1", memory="Prefers tabs", source="vector")]
        graph = [MemoryResponse(id="g1", memory="Prefers tabs", source="graph")]
        result = self.svc._deduplicate_responses(vector, graph)
        assert len(result) == 1
        assert result[0].source == "vector"

    def test_removes_substring_duplicate(self):
        vector = [MemoryResponse(id="v1", memory="User prefers tabs over spaces", source="vector")]
        graph = [MemoryResponse(id="g1", memory="prefers tabs", source="graph")]
        result = self.svc._deduplicate_responses(vector, graph)
        assert len(result) == 1

    def test_keeps_non_duplicate_graph_results(self):
        vector = [MemoryResponse(id="v1", memory="Prefers tabs", source="vector")]
        graph = [MemoryResponse(id="g1", memory="Uses Python 3.12", source="graph")]
        result = self.svc._deduplicate_responses(vector, graph)
        assert len(result) == 2
        sources = [r.source for r in result]
        assert "vector" in sources
        assert "graph" in sources

    def test_vector_hits_precede_graph_rows(self):
        """Audit 27 #2: ranked vector hits keep priority — graph rows are
        appended after them, never positionally interleaved (the old 1:1
        weave let unranked relation strings evict ranked vector hits)."""
        vector = [
            MemoryResponse(id="v1", memory="Fact A", source="vector"),
            MemoryResponse(id="v2", memory="Fact B", source="vector"),
        ]
        graph = [
            MemoryResponse(id="g1", memory="Fact C", source="graph"),
            MemoryResponse(id="g2", memory="Fact D", source="graph"),
        ]
        result = self.svc._deduplicate_responses(vector, graph)
        assert len(result) == 4
        # Should be: v1, v2, g1, g2 — vector first, graph after
        assert [r.id for r in result] == ["v1", "v2", "g1", "g2"]

    def test_case_insensitive_dedup(self):
        vector = [MemoryResponse(id="v1", memory="PREFERS TABS", source="vector")]
        graph = [MemoryResponse(id="g1", memory="prefers tabs", source="graph")]
        result = self.svc._deduplicate_responses(vector, graph)
        assert len(result) == 1

    def test_empty_graph_returns_vector_only(self):
        vector = [MemoryResponse(id="v1", memory="Fact", source="vector")]
        result = self.svc._deduplicate_responses(vector, [])
        assert len(result) == 1

    def test_empty_vector_returns_graph_only(self):
        graph = [MemoryResponse(id="g1", memory="Fact", source="graph")]
        result = self.svc._deduplicate_responses([], graph)
        assert len(result) == 1


class TestSearchLimitEnforcement:
    @pytest.fixture
    def service(self):
        svc = MemoryService()
        svc._memory = MagicMock(name="Memory")
        svc._graphiti = MagicMock(name="Graphiti")
        svc._bridge = MagicMock(name="AsyncBridge")
        svc._memory.graph = MagicMock()
        svc._memory.graph.graphiti = svc._graphiti
        svc._memory.graph._bridge = svc._bridge
        return svc

    def test_search_respects_limit(self, service):
        """Search should return at most `limit` results even when both sources return full sets."""
        service._memory.search.return_value = {
            "results": [
                {"id": f"v{i}", "memory": f"Vector fact {i}", "score": 0.9 - i * 0.01, "metadata": {}}
                for i in range(5)
            ]
        }

        # Mock graph search to return unique results
        mock_search_results = MagicMock()
        mock_search_results.edges = [
            MagicMock(uuid=f"g{i}", name=f"edge{i}", fact=f"Graph fact {i}")
            for i in range(5)
        ]
        mock_search_results.nodes = []
        mock_search_results.episodes = []
        mock_search_results.communities = []

        import asyncio
        future = concurrent.futures.Future()
        future.set_result(mock_search_results)
        service._bridge._loop = MagicMock()
        service._bridge._loop.run_coroutine_threadsafe.return_value = future

        results = service.search(query="test", user_id="u1", limit=3)
        assert len(results) <= 3


# ══════════════════════════════════════════════
# P0-1: Delete graph cleanup
# ══════════════════════════════════════════════


class TestDeleteGraphCleanup:
    @pytest.fixture
    def service(self):
        svc = MemoryService()
        svc._memory = MagicMock(name="Memory")
        svc._graphiti = MagicMock(name="Graphiti")
        svc._bridge = MagicMock(name="AsyncBridge")
        svc._memory.graph = MagicMock()
        svc._memory.graph.graphiti = svc._graphiti
        svc._memory.graph._bridge = svc._bridge
        return svc

    def test_delete_memory_gets_content_before_deleting(self, service):
        """delete_memory should fetch the memory content before calling delete."""
        service._memory.get.return_value = {
            "id": "m1", "memory": "Prefers tabs", "metadata": {"category": "preference"}
        }
        service._memory.delete.return_value = {"message": "deleted"}

        service.delete_memory("m1")

        service._memory.get.assert_called_once_with("m1")
        service._memory.delete.assert_called_once_with("m1")

    def test_delete_memory_attempts_graph_expiration(self, service):
        """delete_memory should try to expire graph edges for the deleted memory."""
        service._memory.get.return_value = {
            "id": "m1", "memory": "Prefers tabs", "metadata": {"category": "preference"}
        }
        service._memory.delete.return_value = {"message": "deleted"}

        with patch.object(service, "_expire_graph_edges_for_memory") as mock_expire:
            service.delete_memory("m1")
            mock_expire.assert_called_once()

    def test_delete_memory_works_without_graphiti(self, service):
        """delete_memory should work even without graphiti initialized."""
        service._graphiti = None
        service._bridge = None
        service._memory.get.return_value = {"id": "m1", "memory": "test", "metadata": {}}
        service._memory.delete.return_value = {"message": "deleted"}

        result = service.delete_memory("m1")
        assert "message" in result

    def test_bulk_delete_all_attempts_graph_expiration(self, service):
        """delete_memories with no filters should expire graph edges for the user's groups."""
        # Stub the scroll so the private-only path finds a private memory
        # and reaches the _expire_graph_edges_for_groups call.
        service._memory.vector_store.delete = MagicMock()
        with patch.object(
            service,
            "_scroll_all_user_memories",
            return_value=[{"id": "m1", "payload": {"data": "x", "metadata": {"visibility": "private"}}}],
        ), patch.object(service, "_expire_graph_edges_for_groups") as mock_expire:
            service.delete_memories(user_id="ehfaz")
            mock_expire.assert_called_once()

    def test_graph_expiration_failure_doesnt_break_delete(self, service):
        """Graph cleanup failure should not prevent vector store delete."""
        service._memory.get.return_value = {"id": "m1", "memory": "test", "metadata": {}}
        service._memory.delete.return_value = {"message": "deleted"}

        # Make _expire_graph_edges_for_memory raise — delete_memory should catch it
        original = service._expire_graph_edges_for_memory
        service._expire_graph_edges_for_memory = MagicMock(side_effect=Exception("Graph down"))

        # Should not raise — the try/except in delete_memory handles it
        result = service.delete_memory("m1")
        service._memory.delete.assert_called_once()
        assert "message" in result

        service._expire_graph_edges_for_memory = original


# ══════════════════════════════════════════════
# P1-8: Bridge timeout
# ══════════════════════════════════════════════


class TestBridgeTimeout:
    def test_run_on_bridge_raises_on_timeout(self):
        import asyncio as _asyncio

        svc = MemoryService()
        svc._bridge = MagicMock(name="AsyncBridge")

        # Create a real event loop for run_coroutine_threadsafe
        loop = _asyncio.new_event_loop()
        svc._bridge._loop = loop

        # Create a coroutine that sleeps forever
        async def slow_coro():
            await _asyncio.sleep(100)

        try:
            with pytest.raises(TimeoutError, match="timed out"):
                svc._run_on_bridge(slow_coro(), timeout=0.05)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop.close()

    def test_run_on_bridge_returns_result_within_timeout(self):
        import asyncio as _asyncio

        svc = MemoryService()
        svc._bridge = MagicMock(name="AsyncBridge")

        loop = _asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        svc._bridge._loop = loop

        async def fast_coro():
            return "test_result"

        try:
            result = svc._run_on_bridge(fast_coro(), timeout=5.0)
            assert result == "test_result"
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=2)
            loop.close()

    def test_run_on_bridge_raises_without_bridge(self):
        svc = MemoryService()
        svc._bridge = None
        with pytest.raises(RuntimeError, match="not initialized"):
            svc._run_on_bridge(MagicMock())


# ══════════════════════════════════════════════
# P1-9: Redis fallback (MCP tools)
# ══════════════════════════════════════════════


class TestMCPRedisFallback:
    @pytest.fixture(autouse=True)
    def mock_mcp_service(self):
        mock_svc = MagicMock(name="MemoryService")
        original = mcp_server._service
        mcp_server._service = mock_svc
        yield mock_svc
        mcp_server._service = original

    @pytest.fixture(autouse=True)
    def mock_mcp_task_manager(self):
        mock_tm = MagicMock(name="TaskManager")
        mock_tm.enqueue_raw = AsyncMock(return_value="task-123")
        mock_tm.enqueue_store = AsyncMock(return_value="task-456")
        original = mcp_server._task_manager
        mcp_server._task_manager = mock_tm
        yield mock_tm
        mcp_server._task_manager = original

    @pytest.mark.asyncio
    async def test_remember_falls_back_on_redis_error(self, mock_mcp_service, mock_mcp_task_manager):
        """When Redis is down, remember should fall back to sync store."""
        mock_mcp_task_manager.enqueue_raw = AsyncMock(side_effect=ConnectionError("Redis unavailable"))
        mock_mcp_service.store_raw.return_value = [
            MemoryResponse(id="m1", memory="test fact", category="preference")
        ]

        result = await mcp_server.call_tool("remember", {
            "content": "Prefers tabs",
            "user_id": "ehfaz",
            "category": "preference",
        })

        data = json.loads(result[0].text)
        assert data["status"] == "completed"
        assert data["fallback"] == "sync"
        mock_mcp_service.store_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_remember_conversation_falls_back_on_redis_error(
        self, mock_mcp_service, mock_mcp_task_manager
    ):
        """When Redis is down, remember_conversation should fall back to sync extract_and_store."""
        mock_mcp_task_manager.enqueue_store = AsyncMock(side_effect=OSError("Connection refused"))
        mock_mcp_service.extract_and_store.return_value = [
            MemoryResponse(id="m1", memory="extracted fact")
        ]

        result = await mcp_server.call_tool("remember_conversation", {
            "messages": [{"role": "user", "content": "I use Python"}],
            "user_id": "ehfaz",
        })

        data = json.loads(result[0].text)
        assert data["status"] == "completed"
        assert data["fallback"] == "sync"
        mock_mcp_service.extract_and_store.assert_called_once()


# ══════════════════════════════════════════════
# P1-9: Redis fallback (REST endpoints)
# ══════════════════════════════════════════════


class TestRESTRedisFallback:
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Patch globals for REST endpoint tests."""
        mock_mem = MagicMock(name="Memory")
        mock_graphiti = MagicMock(name="Graphiti")
        mock_bridge = MagicMock(name="AsyncBridge")

        original_memory = main._memory
        original_graphiti = main._graphiti
        original_bridge = main._bridge

        main._memory = mock_mem
        main._graphiti = mock_graphiti
        main._bridge = mock_bridge

        yield

        main._memory = original_memory
        main._graphiti = original_graphiti
        main._bridge = original_bridge

    @pytest.fixture
    def mock_service(self):
        mock_svc = MagicMock(name="MemoryService")
        original = main._service
        main._service = mock_svc
        yield mock_svc
        main._service = original

    @pytest.fixture
    def mock_task_manager_down(self):
        """Task manager that raises ConnectionError on enqueue."""
        mock_tm = MagicMock(name="TaskManager")
        mock_tm.connect = AsyncMock()
        mock_tm.close = AsyncMock()
        mock_tm.enqueue_store = AsyncMock(side_effect=ConnectionError("Redis down"))
        mock_tm.enqueue_raw = AsyncMock(side_effect=ConnectionError("Redis down"))
        mock_tm.get_status = AsyncMock(return_value={
            "task_id": "test", "status": "not_found", "result": None, "error": None,
        })
        original = main._task_manager
        main._task_manager = mock_tm
        yield mock_tm
        main._task_manager = original

    @pytest.fixture
    def client(self):
        return TestClient(main.app, raise_server_exceptions=False)

    def test_v1_store_raw_falls_back_to_sync(self, client, mock_service, mock_task_manager_down):
        mock_service.store_raw.return_value = [
            MemoryResponse(id="m1", memory="test", category="preference")
        ]

        resp = client.post("/v1/memories/raw", json={
            "content": "Prefers tabs",
            "user_id": "ehfaz",
            "category": "preference",
        })
        # Should return 200 with sync result instead of 500
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        mock_service.store_raw.assert_called_once()

    def test_v1_store_memories_falls_back_to_sync(self, client, mock_service, mock_task_manager_down):
        mock_service.extract_and_store.return_value = [
            MemoryResponse(id="m1", memory="extracted")
        ]

        resp = client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": "I use Python"}],
            "user_id": "ehfaz",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        mock_service.extract_and_store.assert_called_once()


# ══════════════════════════════════════════════
# P1-10: Worker job idempotency
# ══════════════════════════════════════════════


class TestJobIdempotency:
    def test_generate_job_id_deterministic(self):
        """Same content + user_id should produce the same job ID."""
        id1 = _generate_job_id("raw:Prefers tabs", "ehfaz")
        id2 = _generate_job_id("raw:Prefers tabs", "ehfaz")
        assert id1 == id2

    def test_generate_job_id_different_content(self):
        """Different content should produce different job IDs."""
        id1 = _generate_job_id("raw:Prefers tabs", "ehfaz")
        id2 = _generate_job_id("raw:Prefers spaces", "ehfaz")
        assert id1 != id2

    def test_generate_job_id_different_user(self):
        """Different user_id should produce different job IDs."""
        id1 = _generate_job_id("raw:Prefers tabs", "ehfaz")
        id2 = _generate_job_id("raw:Prefers tabs", "alice")
        assert id1 != id2

    def test_generate_job_id_format(self):
        """Job ID should have expected prefix."""
        job_id = _generate_job_id("raw:test", "u1")
        assert job_id.startswith("ns-")
        assert len(job_id) == 19  # "ns-" + 16 hex chars


# ══════════════════════════════════════════════
# P2-14: Global exception handler
# ══════════════════════════════════════════════


class TestGlobalExceptionHandler:
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        mock_mem = MagicMock(name="Memory")
        original_memory = main._memory
        main._memory = mock_mem
        yield
        main._memory = original_memory

    @pytest.fixture(autouse=True)
    def mock_task_manager(self):
        mock_tm = MagicMock(name="TaskManager")
        mock_tm.connect = AsyncMock()
        mock_tm.close = AsyncMock()
        mock_tm.get_status = AsyncMock(return_value={
            "task_id": "test", "status": "completed", "result": {}, "error": None,
        })
        original = main._task_manager
        main._task_manager = mock_tm
        yield mock_tm
        main._task_manager = original

    @pytest.fixture
    def client(self):
        return TestClient(main.app, raise_server_exceptions=False)

    def test_500_does_not_leak_internal_details(self, client):
        """Error responses should not contain stack traces or internal paths."""
        main._memory.add.side_effect = Exception(
            "neo4j://admin:secret@localhost:7687 connection refused /app/internal/path.py"
        )
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert resp.status_code == 500
        body = resp.json()
        # Should NOT contain the raw exception with connection strings
        assert "neo4j://" not in json.dumps(body)
        assert "secret" not in json.dumps(body)
        assert "/app/internal" not in json.dumps(body)

    def test_500_returns_generic_message(self, client):
        main._memory.search.side_effect = RuntimeError("internal failure")
        resp = client.post("/search", json={
            "query": "test",
            "user_id": "u1",
        })
        assert resp.status_code == 500
        body = resp.json()
        # Should have a generic error message
        assert "detail" in body or "error" in body


# ══════════════════════════════════════════════
# P2-15: Input validation
# ══════════════════════════════════════════════


class TestInputValidation:
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        mock_mem = MagicMock(name="Memory")
        original_memory = main._memory
        main._memory = mock_mem

        mock_tm = MagicMock(name="TaskManager")
        mock_tm.connect = AsyncMock()
        mock_tm.close = AsyncMock()
        mock_tm.enqueue_raw = AsyncMock(return_value="task-id")
        mock_tm.enqueue_store = AsyncMock(return_value="task-id")
        original_tm = main._task_manager
        main._task_manager = mock_tm

        yield

        main._memory = original_memory
        main._task_manager = original_tm

    @pytest.fixture
    def client(self):
        return TestClient(main.app, raise_server_exceptions=False)

    def test_rejects_empty_user_id(self, client):
        resp = client.post("/v1/memories/raw", json={
            "content": "test fact",
            "user_id": "",
            "category": "preference",
        })
        assert resp.status_code == 422

    def test_rejects_user_id_with_special_chars(self, client):
        resp = client.post("/v1/memories/raw", json={
            "content": "test fact",
            "user_id": "user@evil.com; DROP TABLE",
            "category": "preference",
        })
        assert resp.status_code == 422

    def test_rejects_empty_content(self, client):
        resp = client.post("/v1/memories/raw", json={
            "content": "",
            "user_id": "ehfaz",
            "category": "preference",
        })
        assert resp.status_code == 422

    def test_rejects_oversized_content(self, client):
        resp = client.post("/v1/memories/raw", json={
            "content": "x" * 10001,
            "user_id": "ehfaz",
            "category": "preference",
        })
        assert resp.status_code == 422

    def test_rejects_empty_search_query(self, client):
        resp = client.post("/v1/search", json={
            "query": "",
            "user_id": "ehfaz",
        })
        assert resp.status_code == 422

    def test_accepts_valid_user_id_formats(self, client):
        """user_id with alphanumeric, underscore, hyphen, and dots should be accepted."""
        resp = client.post("/v1/memories/raw", json={
            "content": "test fact",
            "user_id": "user-123_test.name",
            "category": "preference",
        })
        # Should be 202 (accepted) not 422
        assert resp.status_code == 202

    def test_rejects_too_many_messages(self, client):
        resp = client.post("/v1/memories", json={
            "messages": [{"role": "user", "content": f"msg {i}"} for i in range(501)],
            "user_id": "ehfaz",
        })
        assert resp.status_code == 422


# ══════════════════════════════════════════════
# P3-22: Thread-safe lazy init
# ══════════════════════════════════════════════


# ══════════════════════════════════════════════
# Audit P1-2: Health endpoint checks backends
# ══════════════════════════════════════════════


class TestHealthEndpointChecks:
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        mock_mem = MagicMock(name="Memory")
        original_memory = main._memory
        main._memory = mock_mem
        yield
        main._memory = original_memory

    @pytest.fixture
    def client(self):
        return TestClient(main.app, raise_server_exceptions=False)

    def test_health_returns_checks_dict(self, client):
        """Health endpoint should return per-backend check results."""
        mock_tm = MagicMock(name="TaskManager")
        mock_tm.pool = MagicMock()
        mock_tm.pool.ping = AsyncMock(return_value=True)
        mock_tm.connect = AsyncMock()
        mock_tm.close = AsyncMock()
        original = main._task_manager
        main._task_manager = mock_tm

        # Set up service with initialized backends
        mock_svc = MagicMock()
        mock_svc._memory = MagicMock()
        mock_svc._graphiti = MagicMock()
        original_svc = main._service
        main._service = mock_svc

        try:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "checks" in data
            assert "redis" in data["checks"]
            assert "vector_store" in data["checks"]
            assert "graph_store" in data["checks"]
        finally:
            main._task_manager = original
            main._service = original_svc

    def test_health_reports_degraded_when_redis_down(self, client):
        """Health should report degraded when Redis ping fails."""
        mock_tm = MagicMock(name="TaskManager")
        mock_tm.pool = MagicMock()
        mock_tm.pool.ping = AsyncMock(side_effect=ConnectionError("refused"))
        mock_tm.connect = AsyncMock()
        mock_tm.close = AsyncMock()
        original = main._task_manager
        main._task_manager = mock_tm

        mock_svc = MagicMock()
        mock_svc._memory = MagicMock()
        mock_svc._graphiti = MagicMock()
        original_svc = main._service
        main._service = mock_svc

        try:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["checks"]["redis"] == "unreachable"
        finally:
            main._task_manager = original
            main._service = original_svc

    def test_health_returns_503_when_vector_store_unreachable(self, client):
        """Health should return 503 if the vector store is unreachable."""
        mock_tm = MagicMock(name="TaskManager")
        mock_tm.pool = MagicMock()
        mock_tm.pool.ping = AsyncMock(return_value=True)
        mock_tm.connect = AsyncMock()
        mock_tm.close = AsyncMock()
        original = main._task_manager
        main._task_manager = mock_tm

        # Simulate vector store access raising an exception
        mock_svc = MagicMock()
        type(mock_svc)._memory = property(lambda self: (_ for _ in ()).throw(RuntimeError("down")))
        original_svc = main._service
        main._service = mock_svc

        try:
            resp = client.get("/health")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "unhealthy"
            assert data["checks"]["vector_store"] == "unreachable"
        finally:
            main._task_manager = original
            main._service = original_svc


# ══════════════════════════════════════════════
# Audit P1-4: _get_genai_client() thread safety
# ══════════════════════════════════════════════


class TestGenaiClientThreadSafety:
    def test_genai_client_init_uses_lock(self):
        """_get_genai_client should use the init lock for thread safety."""
        svc = MemoryService()

        with patch("memory_service.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()

            # Call from multiple threads
            threads = []
            for _ in range(5):
                t = threading.Thread(target=svc._get_genai_client)
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

            # Should only create one client
            assert mock_genai.Client.call_count == 1


# ══════════════════════════════════════════════
# Audit P1-6: search_config validation
# ══════════════════════════════════════════════


class TestSearchConfigValidation:
    @pytest.fixture
    def service(self):
        svc = MemoryService()
        svc._memory = MagicMock(name="Memory")
        svc._graphiti = MagicMock(name="Graphiti")
        svc._bridge = MagicMock(name="AsyncBridge")
        svc._memory.graph = MagicMock()
        svc._memory.graph.graphiti = svc._graphiti
        svc._memory.graph._bridge = svc._bridge
        return svc

    def test_invalid_search_config_falls_back_to_default(self, service):
        """Invalid search_config should fall back to default instead of crashing."""
        # Mock the bridge to return empty results
        mock_results = MagicMock()
        mock_results.edges = []
        mock_results.nodes = []
        mock_results.episodes = []
        mock_results.communities = []

        import concurrent.futures
        future = concurrent.futures.Future()
        future.set_result(mock_results)
        service._bridge._loop = MagicMock()
        service._bridge._loop.run_coroutine_threadsafe.return_value = future

        # Pass a dict with bogus keys that will fail SearchConfig(**...)
        result = service.search_graph(
            query="test",
            user_id="u1",
            search_config={"nonexistent_field": True, "another_bad_key": 999},
        )

        # Should not crash — falls back to default config
        assert "edges" in result

    def test_valid_search_config_is_used(self, service):
        """A valid search_config dict should be accepted without fallback."""
        mock_results = MagicMock()
        mock_results.edges = []
        mock_results.nodes = []
        mock_results.episodes = []
        mock_results.communities = []

        import concurrent.futures
        future = concurrent.futures.Future()
        future.set_result(mock_results)
        service._bridge._loop = MagicMock()
        service._bridge._loop.run_coroutine_threadsafe.return_value = future

        # Empty dict should be fine (all fields are optional in SearchConfig)
        result = service.search_graph(
            query="test",
            user_id="u1",
            search_config={},
        )
        assert "edges" in result


# ══════════════════════════════════════════════
# P3-22: Thread-safe lazy init
# ══════════════════════════════════════════════


class TestThreadSafeInit:
    def test_get_memory_has_lock(self):
        """MemoryService should have an init lock."""
        svc = MemoryService()
        assert hasattr(svc, "_init_lock")
        assert isinstance(svc._init_lock, type(threading.Lock()))

    def test_concurrent_get_memory_calls_init_once(self):
        """Multiple threads calling _get_memory should only initialize once."""
        svc = MemoryService()
        init_count = 0

        mock_mem = MagicMock()
        mock_mem.graph = MagicMock()
        mock_mem.graph.graphiti = MagicMock()
        mock_mem.graph._bridge = MagicMock()

        def track_init(*args, **kwargs):
            nonlocal init_count
            init_count += 1
            return mock_mem

        # Patch mem0.Memory at the module level where it's imported
        with patch.dict("sys.modules", {"mem0": MagicMock()}) as _:
            import sys
            sys.modules["mem0"].Memory.from_config.side_effect = track_init

            with patch("memory_service.settings") as mock_settings:
                mock_settings.get_mem0_config.return_value = {}

                threads = []
                for _ in range(5):
                    t = threading.Thread(target=svc._get_memory)
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

                assert init_count == 1


# ══════════════════════════════════════════════
# LLM retry with exponential backoff
# ══════════════════════════════════════════════


class TestIsTransient:
    def test_503_is_transient(self):
        assert _is_transient(Exception("503 UNAVAILABLE"))

    def test_429_is_transient(self):
        assert _is_transient(Exception("429 Too Many Requests"))

    def test_resource_exhausted_is_transient(self):
        assert _is_transient(Exception("RESOURCE_EXHAUSTED: quota exceeded"))

    def test_rate_limit_is_transient(self):
        assert _is_transient(Exception("rate limit exceeded"))

    def test_overloaded_is_transient(self):
        assert _is_transient(Exception("model is overloaded"))

    def test_400_is_not_transient(self):
        assert not _is_transient(Exception("400 Bad Request"))

    def test_auth_error_is_not_transient(self):
        assert not _is_transient(Exception("401 UNAUTHENTICATED: invalid API key"))

    def test_generic_error_is_not_transient(self):
        assert not _is_transient(ValueError("invalid input"))


class TestRetryTransient:
    def test_succeeds_on_first_try(self):
        fn = MagicMock(return_value="ok")
        result = retry_transient(fn, "arg1", max_retries=3, base_delay=0, operation="test")
        assert result == "ok"
        assert fn.call_count == 1

    def test_retries_on_transient_then_succeeds(self):
        fn = MagicMock(side_effect=[
            Exception("503 UNAVAILABLE"),
            Exception("429 rate limit"),
            "ok",
        ])
        result = retry_transient(fn, max_retries=3, base_delay=0, operation="test")
        assert result == "ok"
        assert fn.call_count == 3

    def test_raises_immediately_on_non_transient(self):
        fn = MagicMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            retry_transient(fn, max_retries=3, base_delay=0, operation="test")
        assert fn.call_count == 1

    def test_raises_after_max_retries_exhausted(self):
        fn = MagicMock(side_effect=Exception("503 UNAVAILABLE"))
        with pytest.raises(Exception, match="503"):
            retry_transient(fn, max_retries=2, base_delay=0, operation="test")
        assert fn.call_count == 3  # initial + 2 retries

    def test_passes_kwargs_through(self):
        fn = MagicMock(return_value="ok")
        retry_transient(fn, "a", "b", key="val", max_retries=1, base_delay=0, operation="test")
        fn.assert_called_once_with("a", "b", key="val")

    @patch("memory_service.settings")
    def test_uses_config_defaults(self, mock_settings):
        mock_settings.llm_max_retries = 1
        mock_settings.llm_retry_base_delay = 0
        mock_settings.llm_retry_max_delay = 1
        fn = MagicMock(side_effect=[Exception("503 UNAVAILABLE"), "ok"])
        result = retry_transient(fn, operation="test")
        assert result == "ok"
        assert fn.call_count == 2
