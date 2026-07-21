"""Phase G REST route tests: knowledge_system dispatch, fallback, index trigger.

The queue + code engines are mocked (no Redis / no live bridge); these verify
request wiring: the additive knowledge_system param dispatches through the bound
system, the generic path is byte-identical when it's absent, and POST
/v1/code-graph/index enqueues on the ingest queue.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import main
from main import app
from knowledge.base import SystemAnswer


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


class TestKnowledgeSystemDispatch:
    def test_neighbors_routes_through_bound_system(self, client):
        """knowledge_system present → dispatch via bound system.recall()."""
        answer = SystemAnswer(system_name="code-cbm", content="Neighbors of foo:\n  --> bar")
        with patch("main._dispatch_code_system", new=AsyncMock(return_value=answer)) as disp:
            resp = client.get("/v1/code-graph/neighbors", params={
                "label": "foo", "knowledge_system": "code-cbm",
                "graph_id": "code--ice-bench--smallpy",
            })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["system"] == "code-cbm"
        assert "bar" in body["result"]
        assert body["graph_id"] == "code--ice-bench--smallpy"
        # dispatched with the right operation
        assert disp.await_args.args[1] == "neighbors"

    def test_query_fallback_byte_identical_when_no_system(self, client):
        """No knowledge_system → the existing native/json path, untouched."""
        with patch("main._dispatch_code_system") as disp, \
             patch("adapters.code_graph.query.query_code_graph", return_value="legacy answer"):
            resp = client.get("/v1/code-graph/query", params={
                "question": "foo", "graph_id": "code--ice-bench--smallpy",
            })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["result"] == "legacy answer"
        assert "system" not in body  # legacy shape unchanged
        disp.assert_not_called()  # dispatch helper never invoked on the fallback path

    def test_dispatch_requires_graph_id(self, client):
        """knowledge_system without graph_id (code_space) is a 400, not a crash."""
        resp = client.get("/v1/code-graph/neighbors", params={
            "label": "foo", "knowledge_system": "code-cbm",
        })
        assert resp.status_code == 400
        assert "code_space" in resp.json()["detail"]


class TestIndexTrigger:
    def test_index_enqueues_on_ingest_queue(self, client):
        enqueue = AsyncMock(return_value="ns-code-idx-1")
        with patch("adapters.code_graph.code_graph_available", return_value=True), \
             patch.object(main._task_manager, "enqueue_code_index", enqueue):
            resp = client.post("/v1/code-graph/index", json={
                "repo_source": "/repos/smallpy",
                "system": "code-cbm",
                "project_id": "proj1",
                "code_space": "code--ice-bench--smallpy",
            })
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["task_id"] == "ns-code-idx-1"
        assert body["code_space"] == "code--ice-bench--smallpy"
        assert body["system"] == "code-cbm"
        payload = enqueue.await_args.args[0]
        assert payload["system"] == "code-cbm"
        assert payload["repo_source"] == "/repos/smallpy"
        assert payload["code_space"] == "code--ice-bench--smallpy"

    def test_index_derives_code_space_from_repo_basename(self, client):
        enqueue = AsyncMock(return_value="ns-code-idx-2")
        with patch("adapters.code_graph.code_graph_available", return_value=True), \
             patch.object(main._task_manager, "enqueue_code_index", enqueue):
            resp = client.post("/v1/code-graph/index", json={
                "repo_source": "/repos/pallets-click",
                "system": "code-graphify-lib",
                "user_id": "alice",
            })
        assert resp.status_code == 202, resp.text
        assert resp.json()["code_space"] == "code--alice--pallets-click"
