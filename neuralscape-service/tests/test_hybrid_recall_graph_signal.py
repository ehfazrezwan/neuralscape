"""Hybrid recall says so when its graph leg did not contribute (issue 576).

The REAL ``MemoryService.search`` runs against qdrant-client's in-process
engine; only the Graphiti bridge is faked (it raises, never answers, or is
absent). REST ``/v1/search``, legacy ``/search`` and MCP ``recall_memories`` are
the real handlers. The signal is additive: absent on a healthy or disabled
graph, ``degraded: true`` plus ``graph_leg`` (failed / timed_out / skipped) when
the leg was lost, and the vector results are returned either way.
"""

import json
import threading
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import main
import mcp_server
from memory_service import MemoryService

COLLECTION = "graph_signal_test"


@pytest.fixture
def make_service(monkeypatch):
    import memory.search as search_mod

    client = QdrantClient(":memory:")
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    client.upsert(COLLECTION, [PointStruct(
        id=1, vector=[1.0, 0.0, 0.0, 0.0],
        payload={"data": "vector row about deploys", "user_id": "u1",
                 "metadata": {"category": "convention", "scope": "global"},
                 "created_at": "2026-01-01T00:00:00+00:00"},
    )])
    monkeypatch.setattr(search_mod.settings, "qdrant_collection", COLLECTION)

    def build(*, graphiti=True, fail: Exception | None = None, hang: threading.Event | None = None):
        svc = MemoryService()
        svc._memory = MagicMock(name="Memory")
        svc._memory.embedding_model.embed.return_value = [1.0, 0.0, 0.0, 0.0]
        svc._memory.vector_store.client = client
        svc._memory.vector_store._has_bm25_slot = False
        svc._ensure_filter_indexes = MagicMock()
        if not graphiti:
            svc._graphiti = None
            svc._bridge = None
            return svc
        svc._graphiti = MagicMock(name="Graphiti")
        svc._bridge = MagicMock(name="AsyncBridge")
        results = MagicMock()
        results.edges = results.nodes = results.episodes = results.communities = []

        def run(coro, timeout=30.0):
            if hang is not None:
                hang.wait(10)
            if fail is not None:
                raise fail
            return results

        svc._run_on_bridge = run
        return svc

    return build


@pytest.fixture
def swap_services():
    old_main, old_mcp = main._service, mcp_server._service
    yield lambda svc: (setattr(main, "_service", svc), setattr(mcp_server, "_service", svc))
    main._service, mcp_server._service = old_main, old_mcp


def _signal(body: dict):
    return body.get("degraded"), body.get("graph_leg")


class TestService:
    def test_graph_exception_is_reported_as_failed(self, make_service):
        out = make_service(fail=ConnectionError("neo4j down")).search("q", "u1")
        assert [r.memory for r in out] == ["vector row about deploys"]
        assert out.graph_leg == "failed"

    def test_graph_configured_but_init_failed_is_reported_as_failed(self, make_service):
        svc = make_service(graphiti=False)
        svc._graph_init_error = "ConnectionError: neo4j down at startup"
        assert svc.search("q", "u1").graph_leg == "failed"

    def test_healthy_empty_graph_has_no_signal(self, make_service):
        assert make_service().search("q", "u1").graph_leg is None

    def test_graph_not_configured_has_no_signal(self, make_service):
        assert make_service(graphiti=False).search("q", "u1").graph_leg is None

    def test_vector_only_probe_has_no_signal(self, make_service):
        assert make_service(fail=ConnectionError("x")).search("q", "u1", vector_only=True).graph_leg is None

    def test_join_timeout_is_reported_as_timed_out(self, make_service, monkeypatch):
        import memory.search as search_mod

        monkeypatch.setattr(search_mod, "_GRAPH_SEARCH_JOIN_TIMEOUT_S", 0.05)
        monkeypatch.setattr(search_mod.settings, "graph_leg_budget_enabled", False)
        release = threading.Event()
        try:
            out = make_service(hang=release).search("q", "u1")
        finally:
            release.set()
        assert [r.memory for r in out] == ["vector row about deploys"]
        assert out.graph_leg == "timed_out"

    def test_budget_expiry_is_reported_as_timed_out(self, make_service, monkeypatch):
        import memory.search as search_mod

        monkeypatch.setattr(search_mod.settings, "graph_leg_budget_enabled", True)
        monkeypatch.setattr(search_mod.settings, "graph_leg_budget_s", 0.05)
        release = threading.Event()
        try:
            out = make_service(hang=release).search("q", "u1")
        finally:
            release.set()
        assert out.graph_leg == "timed_out"

    def test_submit_failure_is_reported_as_skipped(self, make_service, monkeypatch):
        import memory.search as search_mod

        pool = MagicMock()
        pool.submit.side_effect = RuntimeError("cannot schedule new futures after shutdown")
        monkeypatch.setattr(search_mod, "_GRAPH_SEARCH_POOL", pool)
        out = make_service().search("q", "u1")
        assert [r.memory for r in out] == ["vector row about deploys"]
        assert out.graph_leg == "skipped"


class TestRest:
    @pytest.mark.parametrize("path", ["/v1/search", "/search"])
    def test_outage_is_visible_in_the_body(self, make_service, swap_services, path):
        swap_services(make_service(fail=ConnectionError("neo4j down")))
        resp = TestClient(main.app, raise_server_exceptions=False).post(path, json={"query": "q", "user_id": "u1"})
        assert resp.status_code == 200
        body = resp.json()
        assert _signal(body) == (True, "failed")
        assert [r["memory"] for r in body["results"]] == ["vector row about deploys"]

    def test_index_only_carries_the_signal(self, make_service, swap_services):
        swap_services(make_service(fail=ConnectionError("neo4j down")))
        body = TestClient(main.app, raise_server_exceptions=False).post(
            "/v1/search", json={"query": "q", "user_id": "u1", "index_only": True}
        ).json()
        assert body["index_only"] is True and _signal(body) == (True, "failed")

    @pytest.mark.parametrize("path", ["/v1/search", "/search"])
    @pytest.mark.parametrize("graphiti", [True, False])
    def test_healthy_and_disabled_bodies_are_unchanged(self, make_service, swap_services, path, graphiti):
        swap_services(make_service(graphiti=graphiti))
        body = TestClient(main.app, raise_server_exceptions=False).post(
            path, json={"query": "q", "user_id": "u1"}
        ).json()
        assert "degraded" not in body and "graph_leg" not in body


class TestMcp:
    @pytest.mark.asyncio
    async def test_outage_visible_to_agent_without_changing_the_result_block(self, make_service, swap_services):
        swap_services(make_service(fail=ConnectionError("neo4j down")))
        result = await mcp_server.call_tool("recall_memories", {"query": "q", "user_id": "u1"})
        rows = json.loads(result[0].text)
        assert [r["memory"] for r in rows] == ["vector row about deploys"]  # block 0 keeps its shape
        assert len(result) == 2
        assert _signal(json.loads(result[1].text)) == (True, "failed")

    @pytest.mark.asyncio
    async def test_index_only_carries_the_signal(self, make_service, swap_services):
        swap_services(make_service(fail=ConnectionError("neo4j down")))
        result = await mcp_server.call_tool("recall_memories", {"query": "q", "user_id": "u1", "index_only": True})
        assert len(result) == 1 and _signal(json.loads(result[0].text)) == (True, "failed")

    @pytest.mark.asyncio
    async def test_healthy_recall_is_unchanged(self, make_service, swap_services):
        swap_services(make_service())
        result = await mcp_server.call_tool("recall_memories", {"query": "q", "user_id": "u1"})
        assert len(result) == 1 and isinstance(json.loads(result[0].text), list)
