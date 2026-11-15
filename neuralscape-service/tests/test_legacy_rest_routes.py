"""Legacy (non-/v1) REST routes run through the shared MemoryService.

Regressions (#430, #432, #440):

* ``GET /memories`` / ``POST /search`` called the mem0 fork with top-level
  ``user_id=`` / ``limit=`` kwargs it rejects -> always 500.
* ``/graph/*`` read a second, separately built Graphiti handle that is never
  attached under mem0 2.x -> always 503, and queried the pre-namespace group
  id (the bare user id) instead of ``user--{uid}``.
* ``POST /graph/search`` mutated the shared ``EDGE_HYBRID_SEARCH_RRF`` recipe.
* ``DELETE /memories`` called mem0 ``delete_all`` (vector points only), leaving
  the user's graph facts live and recallable.
* ``POST /memories`` used mem0's own ``add`` (no gates, legacy group ids).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import main
from main import app


@pytest.fixture(autouse=True)
def task_manager():
    tm = MagicMock(name="TaskManager")
    tm.connect = AsyncMock()
    tm.close = AsyncMock()
    tm.enqueue_store = AsyncMock(return_value="task-1")
    original = main._task_manager
    main._task_manager = tm
    yield tm
    main._task_manager = original


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_service():
    svc = MagicMock(name="MemoryService")
    original = main._service
    main._service = svc
    yield svc
    main._service = original


class StrictMem0:
    """The mem0 2.x fork's keyword-only signatures (mem0/memory/main.py)."""

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.get_all_calls = []
        self.add = MagicMock(name="Memory.add")
        self.delete_all = MagicMock(name="Memory.delete_all")

    def get_all(self, *, filters=None, top_k=20, **kwargs):
        if kwargs:
            raise ValueError(f"Top-level entity parameters not supported: {sorted(kwargs)}")
        self.get_all_calls.append({"filters": filters, "top_k": top_k})
        return {"results": self.rows}


@pytest.fixture
def real_service():
    """A real MemoryService whose mem0/graph handles are fakes (no I/O)."""
    from memory_service import MemoryService

    svc = MemoryService()
    svc._memory = StrictMem0(
        [{"id": "m1", "memory": "likes tea", "metadata": {"metadata": {"category": "preference"}}}]
    )
    svc._get_memory = lambda: svc._memory
    svc._ensure_filter_indexes = lambda: None
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._get_graphiti = lambda: svc._graphiti
    original = main._service
    main._service = svc
    yield svc
    main._service = original


def _bridge_returning(*results):
    remaining = list(results)

    def _bridge(coro, timeout=None):
        if hasattr(coro, "close"):
            coro.close()
        return remaining.pop(0)

    return MagicMock(side_effect=_bridge)


# ── GET /memories, POST /search (#430) ───────────────────────────────


def test_list_memories_200_uses_filters_and_top_k(client, real_service):
    resp = client.get("/memories", params={"user_id": "u1", "limit": 7})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert [m["id"] for m in body["memories"]] == ["m1"]
    assert real_service._memory.get_all_calls == [{"filters": {"user_id": "u1"}, "top_k": 7}]


def test_list_memories_agent_and_run_narrow_filters(client, real_service):
    resp = client.get("/memories", params={"user_id": "u1", "agent_id": "a1", "run_id": "r1"})
    assert resp.status_code == 200, resp.text
    assert real_service._memory.get_all_calls[0]["filters"] == {
        "user_id": "u1", "agent_id": "a1", "run_id": "r1",
    }


def test_search_200_calls_service_with_limit(client, mock_service):
    from schemas import MemoryResponse

    mock_service.search.return_value = [MemoryResponse(id="m1", memory="likes tea", score=0.5)]
    resp = client.post("/search", json={"query": "tea", "user_id": "u1", "limit": 3})
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"][0]["id"] == "m1"
    kwargs = mock_service.search.call_args.kwargs
    assert (kwargs["query"], kwargs["user_id"], kwargs["limit"]) == ("tea", "u1", 3)


def test_search_with_agent_or_run_scope_is_refused_not_widened(client, mock_service):
    resp = client.post("/search", json={"query": "tea", "user_id": "u1", "agent_id": "a1"})
    assert resp.status_code == 400
    mock_service.search.assert_not_called()


# ── /graph/* (#432) ──────────────────────────────────────────────────


def _edge(uuid):
    from datetime import datetime, timezone

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SimpleNamespace(
        uuid=uuid, name="RELATES_TO", fact=f"fact-{uuid}", source_node_uuid="a",
        target_node_uuid="b", group_id="user--u1", created_at=now,
        valid_at=None, invalid_at=None, expired_at=None,
    )


def test_graph_edges_uses_service_handle_and_namespaced_groups(client, real_service, monkeypatch):
    from graphiti_core.edges import EntityEdge

    getter = AsyncMock(return_value=[_edge("e1")])
    monkeypatch.setattr(EntityEdge, "get_by_group_ids", getter)
    real_service._run_on_bridge = _bridge_returning([_edge("e1")])

    resp = client.get("/graph/edges", params={"user_id": "u1"})

    assert resp.status_code == 200, resp.text
    assert [e["uuid"] for e in resp.json()["edges"]] == ["e1"]
    # Caller's namespace + shared pool, never the bare pre-namespace user id.
    group_ids = getter.call_args.kwargs["group_ids"]
    assert "user--u1" in group_ids and "shared" in group_ids and "u1" not in group_ids


def test_graph_nodes_episodes_communities_do_not_503(client, real_service, monkeypatch):
    from graphiti_core.nodes import CommunityNode, EntityNode

    monkeypatch.setattr(EntityNode, "get_by_group_ids", AsyncMock(return_value=[]))
    monkeypatch.setattr(CommunityNode, "get_by_group_ids", AsyncMock(return_value=[]))
    real_service._run_on_bridge = _bridge_returning([], [], [])
    real_service._graphiti.retrieve_episodes = MagicMock()
    for path in ("/graph/nodes", "/graph/episodes", "/graph/communities"):
        resp = client.get(path, params={"user_id": "u1"})
        assert resp.status_code == 200, f"{path}: {resp.status_code} {resp.text}"


def test_graph_search_does_not_mutate_shared_recipe(client, real_service):
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

    before = EDGE_HYBRID_SEARCH_RRF.limit
    empty = SimpleNamespace(edges=[], nodes=[], episodes=[], communities=[])
    real_service._run_on_bridge = _bridge_returning(empty)

    resp = client.post("/graph/search", json={"query": "tea", "user_id": "u1", "limit": before + 7})

    assert resp.status_code == 200, resp.text
    assert resp.json()["edges"] == []
    assert real_service._graphiti.search_.call_args.kwargs["config"].limit == before + 7
    assert EDGE_HYBRID_SEARCH_RRF.limit == before
    group_ids = real_service._graphiti.search_.call_args.kwargs["group_ids"]
    assert "user--u1" in group_ids and "u1" not in group_ids


def test_graph_search_invalid_config_is_400(client, mock_service):
    resp = client.post(
        "/graph/search", json={"query": "tea", "user_id": "u1", "search_config": {"limit": "not-a-number"}}
    )
    assert resp.status_code == 400
    mock_service.search_graph.assert_not_called()


# ── DELETE /memories (#440) ──────────────────────────────────────────


def test_delete_all_cascades_through_service_not_mem0_delete_all(client, real_service):
    real_service._delete_private_only = MagicMock(
        return_value={"message": "All private memories deleted", "graph_cascade_unresolved_ids": []}
    )
    resp = client.delete("/memories", params={"user_id": "u1"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "ok"
    real_service._delete_private_only.assert_called_once_with("u1")
    real_service._memory.delete_all.assert_not_called()


def test_delete_all_surfaces_unresolved_graph_cleanup(client, mock_service):
    mock_service.delete_memories.return_value = {
        "message": "x", "graph_cascade_unresolved_ids": ["m9"],
    }
    resp = client.delete("/memories", params={"user_id": "u1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["graph_cascade_unresolved_ids"] == ["m9"]
    assert body["status"] == "partial"


def test_delete_all_scroll_failure_is_5xx_not_ok(client, real_service):
    def boom(user_id):
        raise RuntimeError("qdrant down")

    real_service._scroll_all_user_memories = boom
    resp = client.delete("/memories", params={"user_id": "u1"})
    assert resp.status_code == 503
    assert "nothing was deleted" in resp.json()["detail"]


def test_service_delete_all_cascades_each_private_row_and_keeps_shared(real_service):
    """Service-level proof behind the route: every PRIVATE row goes through the
    provenance cascade and the vector delete; shared rows are untouched."""
    def row(mid, vis):
        return {"id": mid, "payload": {"data": f"fact {mid}", "metadata": {"visibility": vis}}}

    real_service._memory.vector_store = MagicMock()
    real_service._scroll_all_user_memories = lambda uid: [
        row("p1", "private"), row("p2", "private"), row("s1", "shared"),
    ]
    real_service._expire_graph_edges_for_groups = MagicMock()
    real_service._cascade_or_fallback_expire = MagicMock(return_value={"resolved": True})

    result = real_service.delete_memories(user_id="u1")

    cascaded = [c.kwargs["memory_id"] for c in real_service._cascade_or_fallback_expire.call_args_list]
    assert cascaded == ["p1", "p2"]
    deleted = [c.args[0] for c in real_service._memory.vector_store.delete.call_args_list]
    assert deleted == ["p1", "p2"]
    assert real_service._expire_graph_edges_for_groups.call_args.args[0] == ["user--u1"]
    assert result["graph_cascade_unresolved_ids"] == []
    real_service._memory.delete_all.assert_not_called()


def test_delete_with_agent_scope_is_refused_not_widened(client, mock_service):
    resp = client.delete("/memories", params={"user_id": "u1", "run_id": "r1"})
    assert resp.status_code == 400
    mock_service.delete_memories.assert_not_called()


# ── POST /memories ───────────────────────────────────────────────────


def test_post_memories_goes_through_service_write_path(client, real_service):
    from schemas import MemoryResponse

    real_service.extract_and_store = MagicMock(
        return_value=[MemoryResponse(id="m1", memory="likes tea")]
    )
    resp = client.post(
        "/memories",
        json={"messages": [{"role": "user", "content": "I like tea"}], "user_id": "u1", "run_id": "r"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"]["memories"][0]["id"] == "m1"
    kwargs = real_service.extract_and_store.call_args.kwargs
    assert kwargs["user_id"] == "u1" and kwargs["run_id"] == "r"
    real_service._memory.add.assert_not_called()


# ── POST /memories/async + GET /memories/status/{id} ─────────────────


def test_async_add_enqueues_and_returns_poll_url(client, task_manager):
    resp = client.post(
        "/memories/async",
        json={"messages": [{"role": "user", "content": "hi"}], "user_id": "u1"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "status": "accepted", "task_id": "task-1", "poll_url": "/memories/status/task-1",
    }
    assert task_manager.enqueue_store.call_args.kwargs["user_id"] == "u1"


class TestStatusOwnerScope:
    SECRET = "legacy-routes-secret"

    @pytest.fixture(autouse=True)
    def token_auth(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "neuralscape_user_token_secret", self.SECRET)
        monkeypatch.setattr(settings, "neuralscape_api_key", "")
        monkeypatch.setattr(settings, "neuralscape_auth_disabled", False)

    @pytest.fixture(autouse=True)
    def owned_status(self, task_manager):
        async def get_status(task_id, user_id=None, enforce_owner=False, include_graph=False):
            if enforce_owner and user_id != "alice":
                return {"task_id": task_id, "status": "not_found", "result": None, "error": None}
            return {"task_id": task_id, "status": "completed", "result": {}, "error": None}

        task_manager.get_status = AsyncMock(side_effect=get_status)

    def _hdr(self, user):
        from tokens import issue_user_token

        return {"Authorization": f"Bearer {issue_user_token(user, self.SECRET, 3600)}"}

    def test_owner_reads_status(self, client):
        resp = client.get("/memories/status/task-1", headers=self._hdr("alice"))
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_foreign_user_gets_404(self, client):
        resp = client.get("/memories/status/task-1", headers=self._hdr("bob"))
        assert resp.status_code == 404
