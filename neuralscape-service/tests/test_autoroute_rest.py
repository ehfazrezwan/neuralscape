"""AR2/AR3 REST wiring: knowledge_system="auto" per-op auto-selection.

Verifies the REST code endpoints honor the "auto" sentinel: they resolve+bind
the measured-best healthy engine for the op (mocked here), dispatch through it,
and attribute the SERVED engine (AR3) with routed_by="auto". A no-healthy-engine
resolution degrades to a clean 503 (honest N/A), never a fabricated 200.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from knowledge.base import SystemAnswer


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def _bound_returning(answer):
    """A fake bound CodeKnowledgeSystem whose recall() returns `answer`."""
    bound = MagicMock()
    bound.recall.return_value = answer
    return bound


class TestAutoDispatch:
    def test_neighbors_auto_routes_to_graphify_and_attributes(self, client):
        """auto + neighbors → graphify-lib serves; response attributes the engine."""
        answer = SystemAnswer(
            system_name="code-graphify-lib", content="Neighbors of foo:\n  --> bar"
        )
        bound = _bound_returning(answer)
        with patch(
            "knowledge.code_dispatch.resolve_auto_bound_system",
            return_value=(bound, "code-graphify-lib", "auto: neighbors → code-graphify-lib"),
        ) as auto:
            resp = client.get("/v1/code-graph/neighbors", params={
                "label": "foo", "knowledge_system": "auto",
                "graph_id": "code--ice-bench--smallpy",
            })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["system"] == "code-graphify-lib"  # AR3: engine that served
        assert body["routed_by"] == "auto"
        assert "bar" in body["result"]
        # auto resolver called with the op-class "neighbors"
        assert auto.call_args.args[0] == "neighbors"

    def test_symbol_lookup_auto_routes_to_native(self, client):
        """auto + query (symbol_lookup) → native serves."""
        answer = SystemAnswer(system_name="code-native", content="foo (function) in a.py:1")
        bound = _bound_returning(answer)
        with patch(
            "knowledge.code_dispatch.resolve_auto_bound_system",
            return_value=(bound, "code-native", "auto: query → code-native"),
        ) as auto:
            resp = client.get("/v1/code-graph/query", params={
                "question": "foo", "knowledge_system": "auto",
                "graph_id": "code--ice-bench--smallpy",
            })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["system"] == "code-native"
        assert body["routed_by"] == "auto"
        assert auto.call_args.args[0] == "query"

    def test_locate_auto_carries_hits_and_attribution(self, client):
        answer = SystemAnswer(
            system_name="code-native",
            content="1. pkg.foo (function) — a.py:1",
            hits=[{"fqn": "pkg.foo", "file": "a.py", "line": 1}],
        )
        bound = _bound_returning(answer)
        with patch(
            "knowledge.code_dispatch.resolve_auto_bound_system",
            return_value=(bound, "code-native", "auto: locate → code-native"),
        ):
            resp = client.get("/v1/code-graph/locate", params={
                "query": "the foo thing", "knowledge_system": "auto",
                "graph_id": "code--ice-bench--smallpy",
            })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["system"] == "code-native"
        assert body["routed_by"] == "auto"
        assert body["results"][0]["fqn"] == "pkg.foo"

    def test_auto_no_healthy_engine_is_503(self, client):
        """No capable healthy engine for the op → honest 503, not a fake 200."""
        with patch(
            "knowledge.code_dispatch.resolve_auto_bound_system",
            return_value=(None, None, "auto: no bindable healthy engine for impact"),
        ):
            resp = client.get("/v1/code-graph/impact", params={
                "symbol": "foo", "knowledge_system": "auto",
                "graph_id": "code--ice-bench--smallpy",
            })
        assert resp.status_code == 503, resp.text
        assert "no healthy engine" in resp.json()["detail"]

    def test_auto_requires_graph_id(self, client):
        """auto still needs a code_space (graph_id) to bind an engine → 400."""
        resp = client.get("/v1/code-graph/neighbors", params={
            "label": "foo", "knowledge_system": "auto",
        })
        assert resp.status_code == 400
        assert "code_space" in resp.json()["detail"]
