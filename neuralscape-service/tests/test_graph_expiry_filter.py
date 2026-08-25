"""Unit tests for the graph LISTING endpoints' default expiry exclusion.

Security fix: `memory/graph_admin.py::get_graph_nodes` / `get_graph_edges`
applied NO expiry filter, while every other read path excludes soft-expired
edges (`memory/groups.py::_live_edges_filter` at the Cypher level in
`memory/search.py`, plus the `_edge_is_invalidated` post-filter). Since
`memory/provenance.py`'s episode cascade SOFT-expires edges (and clears the
`summary` of entity nodes whose surrounding edges are now all gone), an
unfiltered listing endpoint would hand back exactly the facts a visibility
flip/delete just expired — defeating that cascade entirely.

These tests exercise the service methods directly (Neo4j/Graphiti is never
a real dependency — `_run_on_bridge` is mocked, mirroring
`test_episode_cascade.py`'s `_bridge_side_effect` idiom) and the REST
pass-through of the new `include_expired` parameter (v1 routes via a
mocked `_service`; legacy routes via a monkeypatched module-level
`main._run_on_bridge`).
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import main
from main import app


@pytest.fixture(autouse=True)
def mock_memory():
    """Patch main.py's legacy lazy-init globals so no real mem0/Graphiti is
    created (mirrors test_service.py's fixture of the same name — each
    test file that drives the app through TestClient needs its own copy,
    fixtures aren't shared across files without a conftest.py entry)."""
    mock_mem = MagicMock(name="Memory")
    mock_graphiti = MagicMock(name="Graphiti")
    mock_bridge = MagicMock(name="AsyncBridge")

    original_memory = main._memory
    original_graphiti = main._graphiti
    original_bridge = main._bridge

    main._memory = mock_mem
    main._graphiti = mock_graphiti
    main._bridge = mock_bridge

    yield mock_mem

    main._memory = original_memory
    main._graphiti = original_graphiti
    main._bridge = original_bridge


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


@pytest.fixture
def service():
    """A MemoryService with a mocked bridge (no real event loop, no Neo4j)."""
    from memory_service import MemoryService

    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    return svc


def _bridge_side_effect(*results):
    """Build a _run_on_bridge stand-in returning `results` in call order,
    closing each passed coroutine so mocked-bridge unit tests never leak
    'coroutine was never awaited' warnings (matches
    test_episode_cascade.py's established idiom)."""
    remaining = list(results)

    def _bridge(coro, timeout=None):
        coro.close()
        return remaining.pop(0)

    return MagicMock(side_effect=_bridge)


_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_LATER = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _node(uuid, summary="regional summary"):
    return SimpleNamespace(
        uuid=uuid, name=f"node-{uuid}", summary=summary, labels=["Entity"],
        group_id="shared", created_at=_NOW,
    )


def _edge(uuid, source, target, expired_at=None, invalid_at=None):
    return SimpleNamespace(
        uuid=uuid, name="RELATES_TO", fact=f"fact-{uuid}",
        source_node_uuid=source, target_node_uuid=target,
        group_id="shared", created_at=_NOW,
        valid_at=None, invalid_at=invalid_at, expired_at=expired_at,
    )


class TestGetGraphEdgesExpiryFilter:
    def test_excludes_expired_edge_by_default(self, service):
        edges = [_edge("e-live", "n1", "n2"), _edge("e-expired", "n1", "n3", expired_at=_LATER)]
        service._run_on_bridge = _bridge_side_effect(edges)
        result = service.get_graph_edges("ehfaz")
        assert {e["uuid"] for e in result} == {"e-live"}

    def test_excludes_invalidated_edge_by_default(self, service):
        edges = [_edge("e-live", "n1", "n2"), _edge("e-invalid", "n1", "n3", invalid_at=_LATER)]
        service._run_on_bridge = _bridge_side_effect(edges)
        result = service.get_graph_edges("ehfaz")
        assert {e["uuid"] for e in result} == {"e-live"}

    def test_include_expired_returns_everything(self, service):
        edges = [_edge("e-live", "n1", "n2"), _edge("e-expired", "n1", "n3", expired_at=_LATER)]
        service._run_on_bridge = _bridge_side_effect(edges)
        result = service.get_graph_edges("ehfaz", include_expired=True)
        assert {e["uuid"] for e in result} == {"e-live", "e-expired"}

    def test_default_omits_include_expired_kwarg(self, service):
        """The default call signature must exclude expired edges without
        the caller passing anything extra."""
        edges = [_edge("e-expired", "n1", "n3", expired_at=_LATER)]
        service._run_on_bridge = _bridge_side_effect(edges)
        result = service.get_graph_edges("ehfaz", project_id=None, limit=50)
        assert result == []


class TestGetGraphNodesExpiryFilter:
    def test_excludes_node_whose_only_edge_is_expired(self, service):
        nodes = [_node("n1"), _node("n2")]
        edges = [_edge("e1", "n1", "other", expired_at=_LATER)]
        service._run_on_bridge = _bridge_side_effect(nodes, edges)
        result = service.get_graph_nodes("ehfaz")
        # n1's sole edge is expired -> excluded. n2 has no edges at all
        # ("never connected", not "expired") -> kept.
        assert {n["uuid"] for n in result} == {"n2"}

    def test_node_with_a_live_edge_kept(self, service):
        nodes = [_node("n1")]
        edges = [_edge("e1", "n1", "other")]
        service._run_on_bridge = _bridge_side_effect(nodes, edges)
        result = service.get_graph_nodes("ehfaz")
        assert {n["uuid"] for n in result} == {"n1"}

    def test_node_with_no_edges_at_all_is_kept(self, service):
        """A freshly-minted entity with no RELATES_TO edges yet is
        'not yet enriched', not 'expired' — must not be excluded."""
        nodes = [_node("n1")]
        service._run_on_bridge = _bridge_side_effect(nodes, [])
        result = service.get_graph_nodes("ehfaz")
        assert {n["uuid"] for n in result} == {"n1"}

    def test_mixed_parentage_node_kept_when_any_edge_is_live(self, service):
        nodes = [_node("n1")]
        edges = [_edge("e1", "n1", "other1", expired_at=_LATER), _edge("e2", "n1", "other2")]
        service._run_on_bridge = _bridge_side_effect(nodes, edges)
        result = service.get_graph_nodes("ehfaz")
        assert {n["uuid"] for n in result} == {"n1"}

    def test_include_expired_skips_the_edge_lookup_entirely(self, service):
        nodes = [_node("n1"), _node("n2")]
        service._run_on_bridge = _bridge_side_effect(nodes)  # only ONE call expected
        result = service.get_graph_nodes("ehfaz", include_expired=True)
        assert {n["uuid"] for n in result} == {"n1", "n2"}
        assert service._run_on_bridge.call_count == 1

    def test_edge_lookup_failure_fails_open_to_unfiltered(self, service):
        """The live-node check is defense-in-depth on top of the edge
        filter, not the primary mechanism — a failure there must not hide
        nodes that would otherwise be legitimately listed."""
        nodes = [_node("n1")]
        service._get_graphiti = MagicMock(return_value=service._graphiti)

        call_count = {"n": 0}

        def _bridge(coro, timeout=None):
            coro.close()
            call_count["n"] += 1
            if call_count["n"] == 1:
                return nodes
            raise RuntimeError("bridge down")

        service._run_on_bridge = MagicMock(side_effect=_bridge)
        result = service.get_graph_nodes("ehfaz")
        assert {n["uuid"] for n in result} == {"n1"}


class TestV1GraphRoutesPassIncludeExpired:
    """REST pass-through: /v1/graph/{nodes,edges} thread include_expired to
    the service, defaulting to False."""

    def test_v1_graph_nodes_defaults_include_expired_false(self, client, mock_service):
        mock_service.get_graph_nodes.return_value = []
        resp = client.get("/v1/graph/nodes", params={"user_id": "ehfaz"})
        assert resp.status_code == 200
        assert mock_service.get_graph_nodes.call_args.kwargs["include_expired"] is False

    def test_v1_graph_nodes_threads_include_expired_true(self, client, mock_service):
        mock_service.get_graph_nodes.return_value = []
        resp = client.get(
            "/v1/graph/nodes", params={"user_id": "ehfaz", "include_expired": "true"}
        )
        assert resp.status_code == 200
        assert mock_service.get_graph_nodes.call_args.kwargs["include_expired"] is True

    def test_v1_graph_edges_defaults_include_expired_false(self, client, mock_service):
        mock_service.get_graph_edges.return_value = []
        resp = client.get("/v1/graph/edges", params={"user_id": "ehfaz"})
        assert resp.status_code == 200
        assert mock_service.get_graph_edges.call_args.kwargs["include_expired"] is False

    def test_v1_graph_edges_threads_include_expired_true(self, client, mock_service):
        mock_service.get_graph_edges.return_value = []
        resp = client.get(
            "/v1/graph/edges", params={"user_id": "ehfaz", "include_expired": "true"}
        )
        assert resp.status_code == 200
        assert mock_service.get_graph_edges.call_args.kwargs["include_expired"] is True


class TestLegacyGraphRoutesExpiryFilter:
    """The legacy /graph/{nodes,edges} routes duplicate the fetch inline
    (they predate the MemoryService split and use their own module-level
    _graphiti/_bridge globals) — they must get the same default exclusion,
    not just the v1 routes."""

    def test_legacy_graph_edges_excludes_expired_by_default(self, client, monkeypatch):
        import main

        edges = [_edge("e-live", "a", "b"), _edge("e-expired", "a", "c", expired_at=_LATER)]

        def _fake_run_on_bridge(coro):
            coro.close()
            return edges

        monkeypatch.setattr(main, "_run_on_bridge", _fake_run_on_bridge)
        resp = client.get("/graph/edges", params={"user_id": "ehfaz"})
        assert resp.status_code == 200
        assert {e["uuid"] for e in resp.json()["edges"]} == {"e-live"}

    def test_legacy_graph_edges_include_expired_returns_everything(self, client, monkeypatch):
        import main

        edges = [_edge("e-live", "a", "b"), _edge("e-expired", "a", "c", expired_at=_LATER)]

        def _fake_run_on_bridge(coro):
            coro.close()
            return edges

        monkeypatch.setattr(main, "_run_on_bridge", _fake_run_on_bridge)
        resp = client.get("/graph/edges", params={"user_id": "ehfaz", "include_expired": "true"})
        assert resp.status_code == 200
        assert {e["uuid"] for e in resp.json()["edges"]} == {"e-live", "e-expired"}

    def test_legacy_graph_nodes_excludes_node_with_only_expired_edges(self, client, monkeypatch):
        import main

        nodes = [_node("n1"), _node("n2")]
        edges = [_edge("e1", "n1", "other", expired_at=_LATER)]
        calls = {"n": 0}

        def _fake_run_on_bridge(coro):
            coro.close()
            calls["n"] += 1
            return nodes if calls["n"] == 1 else edges

        monkeypatch.setattr(main, "_run_on_bridge", _fake_run_on_bridge)
        resp = client.get("/graph/nodes", params={"user_id": "ehfaz"})
        assert resp.status_code == 200
        assert {n["uuid"] for n in resp.json()["nodes"]} == {"n2"}

    def test_legacy_graph_nodes_include_expired_skips_edge_lookup(self, client, monkeypatch):
        import main

        nodes = [_node("n1"), _node("n2")]
        calls = {"n": 0}

        def _fake_run_on_bridge(coro):
            coro.close()
            calls["n"] += 1
            return nodes

        monkeypatch.setattr(main, "_run_on_bridge", _fake_run_on_bridge)
        resp = client.get("/graph/nodes", params={"user_id": "ehfaz", "include_expired": "true"})
        assert resp.status_code == 200
        assert {n["uuid"] for n in resp.json()["nodes"]} == {"n1", "n2"}
        assert calls["n"] == 1  # no edge-liveness lookup when include_expired=True
