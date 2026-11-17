"""DELETE /v1/graph/episodes/{uuid}: an unknown episode is a 404, a backend failure stays a 500."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import main
from main import app


@pytest.fixture
def client(monkeypatch):
    svc = MagicMock(name="MemoryService")
    monkeypatch.setattr(main, "_service", svc)
    return TestClient(app, raise_server_exceptions=False), svc


def test_unknown_episode_is_404(client):
    c, svc = client
    svc.delete_episode.return_value = {"error": "Episode e1 not found", "not_found": True}
    resp = c.delete("/v1/graph/episodes/e1")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_backend_failure_stays_500(client):
    c, svc = client
    svc.delete_episode.return_value = {"error": "Neo4j down"}
    assert c.delete("/v1/graph/episodes/e1").status_code == 500


def test_graph_not_initialised_stays_500(client):
    c, svc = client
    svc.delete_episode.return_value = {"error": "Graphiti not initialized"}
    assert c.delete("/v1/graph/episodes/e1").status_code == 500


def test_foreign_episode_is_still_403(client):
    c, svc = client
    svc.delete_episode.side_effect = PermissionError("not yours")
    assert c.delete("/v1/graph/episodes/e1").status_code == 403


def test_deleted_episode_is_200(client):
    c, svc = client
    svc.delete_episode.return_value = {"message": "Episode e1 deleted"}
    assert c.delete("/v1/graph/episodes/e1").status_code == 200


def _service_with_episodes(records):
    """A real MemoryService whose graph session answers the existence check with ``records``."""
    import asyncio

    from memory_service import MemoryService

    class _Result:
        async def data(self):
            return records

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def run(self, cypher, **params):
            return _Result()

    svc = MemoryService()
    svc._graphiti = MagicMock()
    svc._graphiti.driver.session = lambda: _Session()
    svc._bridge = MagicMock()
    svc._get_graphiti = lambda: svc._graphiti
    svc._run_on_bridge = lambda coro, timeout=None: asyncio.run(coro)
    return svc


def test_service_flags_an_unknown_episode_as_not_found():
    svc = _service_with_episodes([])
    out = svc.delete_episode("nope", user_id="alice")
    assert out["not_found"] is True and "not found" in out["error"]


def test_service_deletes_a_known_episode():
    svc = _service_with_episodes([{"group_id": "user--alice"}])
    out = svc.delete_episode("e1", user_id="alice")
    assert out["message"] == "Episode e1 deleted"
