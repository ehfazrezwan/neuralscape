"""A graph outage must not read as "no facts" on graph search (REST + MCP).

The REAL ``MemoryService.search_graph`` runs; only the Graphiti bridge is
faked (it raises, returns nothing, or is absent).  REST and MCP handlers are
the real ones, so the marker is shown to survive each surface.
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import main
import mcp_server
from memory_service import MemoryService


def _service(*, graphiti=True, fail: Exception | None = None):
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    if not graphiti:
        svc._graphiti = None
        svc._bridge = None
        return svc
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    if fail is not None:
        def run(coro, timeout=30.0):
            raise fail
    else:
        results = MagicMock()
        results.edges = results.nodes = results.episodes = results.communities = []

        def run(coro, timeout=30.0):
            return results

    svc._run_on_bridge = run  # the real bridge needs a live event loop
    return svc


EMPTY = {"edges": [], "nodes": [], "episodes": [], "communities": []}


class TestServiceMarker:
    def test_graph_configured_but_init_failed_is_degraded(self):
        svc = _service(graphiti=False)
        svc._graph_init_error = "ConnectionError: neo4j down at startup"
        out = svc.search_graph("q", "u1")
        assert out["degraded"] is True and out["error"] == "graph_unavailable"

    def test_mapping_bug_is_not_reported_as_an_outage(self):
        svc = _service()
        bad = MagicMock()
        bad.edges = [object()]  # no .uuid etc: a result-mapping defect
        bad.nodes = bad.episodes = bad.communities = []
        svc._run_on_bridge = lambda coro, timeout=30.0: bad
        with pytest.raises(AttributeError):
            svc.search_graph("q", "u1")

    def test_graph_failure_is_marked_degraded(self):
        out = _service(fail=ConnectionError("neo4j down")).search_graph("q", "u1")
        assert out["degraded"] is True
        assert out["error"] == "graph_unavailable"
        assert out["edges"] == [] and out["nodes"] == []

    def test_healthy_empty_graph_has_no_marker(self):
        out = _service().search_graph("q", "u1")
        assert out == EMPTY

    def test_graph_not_configured_is_unchanged(self):
        out = _service(graphiti=False).search_graph("q", "u1")
        assert out == EMPTY


@pytest.fixture
def swap_services():
    old_main, old_mcp = main._service, mcp_server._service
    yield lambda svc: (setattr(main, "_service", svc), setattr(mcp_server, "_service", svc))
    main._service, mcp_server._service = old_main, old_mcp


class TestRest:
    @pytest.mark.parametrize("path", ["/v1/graph/search", "/graph/search"])
    def test_outage_reported_in_body(self, swap_services, path):
        swap_services(_service(fail=TimeoutError("bridge timeout")))
        resp = TestClient(main.app, raise_server_exceptions=False).post(
            path, json={"query": "q", "user_id": "u1"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["degraded"] is True and body["error"] == "graph_unavailable"

    @pytest.mark.parametrize("path", ["/v1/graph/search", "/graph/search"])
    def test_healthy_and_disabled_stay_ok(self, swap_services, path):
        for svc in (_service(), _service(graphiti=False)):
            swap_services(svc)
            body = TestClient(main.app, raise_server_exceptions=False).post(
                path, json={"query": "q", "user_id": "u1"}
            ).json()
            assert body == {"status": "ok", **EMPTY}


class TestMcp:
    @pytest.mark.asyncio
    async def test_outage_visible_to_agent(self, swap_services):
        swap_services(_service(fail=RuntimeError("graphiti error")))
        result = await mcp_server.call_tool("search_knowledge_graph", {"query": "q", "user_id": "u1"})
        data = json.loads(result[0].text)
        assert data["degraded"] is True and data["error"] == "graph_unavailable"

    @pytest.mark.asyncio
    async def test_healthy_empty_has_no_marker(self, swap_services):
        swap_services(_service())
        result = await mcp_server.call_tool("search_knowledge_graph", {"query": "q", "user_id": "u1"})
        assert json.loads(result[0].text) == EMPTY


class TestInitFailureIsRecorded:
    def _init(self, monkeypatch, graph_store_cfg, factory_exc=None):
        import memory.core as core

        svc = MemoryService()
        monkeypatch.setattr(svc, "_ensure_indexes_at_startup", lambda: None)
        monkeypatch.setattr(svc, "_retry_index_ensure", lambda: None, raising=False)
        monkeypatch.setattr(core, "_memory_from_config", lambda cls, cfg: MagicMock(spec=[]))
        cfg = {"graph_store": graph_store_cfg} if graph_store_cfg else {}
        fake = MagicMock()
        fake.get_mem0_config.return_value = cfg
        monkeypatch.setattr(core, "settings", fake)
        if factory_exc is not None:
            import mem0.utils.factory as factory

            def boom(provider, shim):
                raise factory_exc

            monkeypatch.setattr(factory.GraphStoreFactory, "create", staticmethod(boom))
        svc._filter_indexes_ok = True
        svc._created_at_index_ok = True
        svc._get_memory()
        return svc

    def test_configured_graph_that_fails_to_attach_reads_degraded(self, monkeypatch):
        svc = self._init(monkeypatch, {"provider": "graphiti", "config": {}}, ConnectionError("bolt refused"))
        assert svc._graph_init_error and "bolt refused" in svc._graph_init_error
        out = svc.search_graph("q", "u1")
        assert out["degraded"] is True and out["error"] == "graph_unavailable"

    def test_graph_not_configured_stays_plain(self, monkeypatch):
        svc = self._init(monkeypatch, None)
        assert svc._graph_init_error is None
        assert svc.search_graph("q", "u1") == EMPTY
