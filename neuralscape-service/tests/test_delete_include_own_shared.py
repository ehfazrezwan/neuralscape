"""Opt-in ``include_shared`` on bulk delete removes the caller's OWN shared rows (#479).

Decision: the default (legacy ``DELETE /memories`` keeps the caller's shared rows)
stays; ``include_shared=true`` additionally deletes the caller's own shared rows
through the same per-row provenance cascade as the private path. It never touches
another user's rows, never deletes ``standard`` rows, and never calls mem0
``delete_all`` (which erased standard rows and skipped the per-row cascade).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import main
from main import app
from memory.delete import MemoryListingFailed


def _row(mid, visibility="private", owner=None, project_id=None):
    meta = {"visibility": visibility}
    if owner:
        meta["owner_user_id"] = owner
    if project_id:
        meta["project_id"] = project_id
    return {"id": mid, "payload": {"data": f"fact {mid}", "metadata": meta}}


ROWS = [
    _row("priv-1", "private"),
    _row("priv-2", "private", project_id="alpha"),
    _row("own-shared", "shared", owner="u1"),
    _row("own-shared-noowner", "shared"),
    _row("others-shared", "shared", owner="bob"),
    _row("own-standard", "standard"),
]


@pytest.fixture
def svc():
    from memory_service import MemoryService

    s = MemoryService()
    s._memory = MagicMock(name="Memory")
    s._get_memory = lambda: s._memory
    s._graphiti = MagicMock(name="Graphiti")
    s._bridge = MagicMock(name="AsyncBridge")
    s._scroll_all_user_memories = lambda uid: [dict(r) for r in ROWS]
    s._expire_graph_edges_for_groups = MagicMock()
    s._cascade_or_fallback_expire = MagicMock(return_value={"resolved": True})
    return s


def _deleted(s):
    return [c.args[0] for c in s._memory.vector_store.delete.call_args_list]


def _cascaded(s):
    return [c.kwargs["memory_id"] for c in s._cascade_or_fallback_expire.call_args_list]


# ── service level ──


def test_include_shared_true_deletes_own_private_and_own_shared_only(svc):
    out = svc.delete_memories(user_id="u1", include_shared=True)
    assert _deleted(svc) == ["priv-1", "priv-2", "own-shared", "own-shared-noowner"]
    assert "own-standard" not in _deleted(svc)       # never standard
    assert "others-shared" not in _deleted(svc)      # never someone else's row
    assert "including 2 shared" in out["message"]
    assert "preserved 1 shared" in out["message"] and "preserved 1 standard" in out["message"]
    assert out["failed_ids"] == [] and out["truncated"] is False


def test_include_shared_true_runs_the_per_row_cascade_and_never_mem0_delete_all(svc):
    svc.delete_memories(user_id="u1", include_shared=True)
    svc._memory.delete_all.assert_not_called()
    assert _cascaded(svc) == ["priv-1", "priv-2", "own-shared", "own-shared-noowner"]


def test_include_shared_never_bulk_expires_the_shared_group(svc):
    svc.delete_memories(user_id="u1", include_shared=True)
    groups = svc._expire_graph_edges_for_groups.call_args.args[0]
    assert groups and all(g.startswith("user--u1") for g in groups), groups
    assert not any(g == "shared" or g.startswith("shared") for g in groups)


def test_include_shared_false_is_unchanged(svc):
    out = svc.delete_memories(user_id="u1")
    assert _deleted(svc) == ["priv-1", "priv-2"]
    assert _cascaded(svc) == ["priv-1", "priv-2"]
    assert "Deleted 2 private memories" in out["message"]
    assert "preserved 3 shared" in out["message"] and "preserved 1 standard" in out["message"]
    svc._memory.delete_all.assert_not_called()


def test_include_shared_listing_failure_deletes_nothing_and_raises(svc):
    def boom(uid):
        raise RuntimeError("qdrant down")

    svc._scroll_all_user_memories = boom
    with pytest.raises(MemoryListingFailed):
        svc.delete_memories(user_id="u1", include_shared=True)
    svc._memory.vector_store.delete.assert_not_called()
    svc._memory.delete_all.assert_not_called()


def test_include_shared_partial_failure_is_reported_and_keeps_that_rows_graph(svc):
    def delete(mid):
        if mid == "own-shared":
            raise RuntimeError("row failed")

    svc._memory.vector_store.delete.side_effect = delete
    out = svc.delete_memories(user_id="u1", include_shared=True)
    assert out["failed_ids"] == ["own-shared"] and out["failed"] == 1
    assert "own-shared" not in _cascaded(svc)
    assert "(1 failed)" in out["message"]


# ── REST: legacy and v1 ──


@pytest.fixture(autouse=True)
def task_manager():
    tm = MagicMock(name="TaskManager")
    tm.connect = AsyncMock()
    tm.close = AsyncMock()
    original = main._task_manager
    main._task_manager = tm
    yield tm
    main._task_manager = original


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_service():
    m = MagicMock(name="MemoryService")
    m.delete_memories.return_value = {"message": "ok", "graph_cascade_unresolved_ids": []}
    original = main._service
    main._service = m
    yield m
    main._service = original


@pytest.mark.parametrize("params,expected", [
    ({}, False),
    ({"include_shared": "false"}, False),
    ({"include_shared": "true"}, True),
])
def test_legacy_route_forwards_include_shared(client, mock_service, params, expected):
    resp = client.delete("/memories", params={"user_id": "u1", **params})
    assert resp.status_code == 200, resp.text
    mock_service.delete_memories.assert_called_once_with(user_id="u1", include_shared=expected)


@pytest.mark.parametrize("body,expected", [
    ({"user_id": "u1"}, False),
    ({"user_id": "u1", "include_shared": False}, False),
    ({"user_id": "u1", "include_shared": True}, True),
])
def test_v1_route_forwards_include_shared(client, mock_service, body, expected):
    resp = client.request("DELETE", "/v1/memories", json=body)
    assert resp.status_code == 200, resp.text
    assert mock_service.delete_memories.call_args.kwargs["include_shared"] is expected


@pytest.fixture
def real_main_service(svc):
    original = main._service
    main._service = svc
    yield svc
    main._service = original


@pytest.mark.parametrize("call", ["legacy", "v1"])
def test_routes_delete_own_shared_but_keep_standard_and_others(client, real_main_service, call):
    s = real_main_service
    if call == "legacy":
        resp = client.delete("/memories", params={"user_id": "u1", "include_shared": "true"})
    else:
        resp = client.request("DELETE", "/v1/memories", json={"user_id": "u1", "include_shared": True})
    assert resp.status_code == 200, resp.text
    assert _deleted(s) == ["priv-1", "priv-2", "own-shared", "own-shared-noowner"]
    s._memory.delete_all.assert_not_called()


def test_legacy_route_default_keeps_shared(client, real_main_service):
    resp = client.delete("/memories", params={"user_id": "u1"})
    assert resp.status_code == 200, resp.text
    assert _deleted(real_main_service) == ["priv-1", "priv-2"]


def test_legacy_route_include_shared_listing_failure_is_503(client, real_main_service):
    def boom(uid):
        raise RuntimeError("qdrant down")

    real_main_service._scroll_all_user_memories = boom
    resp = client.delete("/memories", params={"user_id": "u1", "include_shared": "true"})
    assert resp.status_code == 503, resp.text
    real_main_service._memory.vector_store.delete.assert_not_called()
    real_main_service._memory.delete_all.assert_not_called()


def test_legacy_route_include_shared_partial_is_207(client, real_main_service):
    real_main_service._memory.vector_store.delete.side_effect = lambda mid: (
        (_ for _ in ()).throw(RuntimeError("x")) if mid == "own-shared" else None
    )
    resp = client.delete("/memories", params={"user_id": "u1", "include_shared": "true"})
    assert resp.status_code == 207, resp.text
    assert resp.json()["failed_ids"] == ["own-shared"]
