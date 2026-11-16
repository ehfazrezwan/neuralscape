"""Read-side workspace recall (ext.workspaces).

Content ingested with a non-default adapter lands in a ``ref-<adapter>...``
workspace. These tests pin the read side:

- the workspace condition is part of the Qdrant filter on BOTH legs (dense and
  BM25), so a reference row is found even when stronger memory rows would fill
  ``limit`` (proved against qdrant-client's real local filter engine, no server);
- the default (no ``workspaces``) still returns memory rows only;
- the graph leg adds ``--ws--<workspace>`` groups only when asked;
- MCP ``recall_memories`` and REST ``POST /v1/search`` forward ``workspaces``
  and reject malformed names.
"""

import json
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import mcp_server
from memory_service import MemoryService

COLLECTION = "ws_recall_test"
QUERY_VEC = [1.0, 0.0, 0.0, 0.0]


def _point(pid, data, workspace="__absent__", cos=1.0, user="u1"):
    metadata = {"category": "domain_knowledge", "scope": "global"}
    if workspace != "__absent__":
        metadata["workspace"] = workspace
    # Vector at angle controlled by cos: higher cos = nearer the query.
    other = (1.0 - cos * cos) ** 0.5
    return PointStruct(
        id=pid,
        vector=[cos, other, 0.0, 0.0],
        payload={"data": data, "user_id": user, "metadata": metadata,
                 "created_at": "2026-01-01T00:00:00+00:00"},
    )


@pytest.fixture
def pool():
    """A real (in-process) Qdrant: 8 strong memory rows + 1 weaker ref row."""
    client = QdrantClient(":memory:")
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    points = [_point(i, f"memory row {i}", cos=0.99 - i * 0.001) for i in range(1, 7)]
    points.append(_point(7, "memory row null ws", workspace=None, cos=0.985))
    points.append(_point(8, "memory row explicit", workspace="memory", cos=0.984))
    points.append(_point(100, "reference chapter", workspace="ref-x", cos=0.5))
    points.append(_point(101, "other reference", workspace="ref-y", cos=0.6))
    client.upsert(COLLECTION, points)
    return client


@pytest.fixture
def service(pool, monkeypatch):
    import memory.search as search_mod

    monkeypatch.setattr(search_mod.settings, "qdrant_collection", COLLECTION)
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._memory.embedding_model.embed.return_value = QUERY_VEC
    svc._memory.vector_store.client = pool
    svc._memory.vector_store._has_bm25_slot = False
    svc._graphiti = MagicMock(name="Graphiti")
    svc._ensure_filter_indexes = MagicMock()
    svc._search_graph_for_visibility = MagicMock(return_value={"edges": [], "episodes": []})
    return svc


def _texts(results):
    return [r.memory for r in results]


def test_reference_row_found_even_when_stronger_memory_rows_fill_limit(service):
    """Fails on a post-filter: the top 3 by cosine are all memory rows."""
    got = _texts(service.search("q", "u1", limit=3, workspaces=["ref-x"], include_shared=False, vector_only=True))
    assert got == ["reference chapter"]


def test_default_returns_memory_rows_only_including_absent_and_null_workspace(service):
    got = _texts(service.search("q", "u1", limit=20, include_shared=False, vector_only=True))
    assert "reference chapter" not in got and "other reference" not in got
    assert "memory row null ws" in got and "memory row explicit" in got
    assert "memory row 1" in got  # workspace key absent entirely
    assert len(got) == 8


def test_default_is_not_starved_by_reference_rows(service, pool):
    """Reference rows nearer than every memory row must not eat the limit."""
    pool.upsert(COLLECTION, [_point(200 + i, f"ref noise {i}", workspace="ref-x", cos=0.999) for i in range(5)])
    got = _texts(service.search("q", "u1", limit=3, include_shared=False, vector_only=True))
    assert len(got) == 3 and all(t.startswith("memory row") for t in got)


def test_multiple_and_mixed_workspaces(service):
    got = set(_texts(service.search("q", "u1", limit=20, workspaces=["ref-x", "ref-y"],
                                    include_shared=False, vector_only=True)))
    assert got == {"reference chapter", "other reference"}
    mixed = set(_texts(service.search("q", "u1", limit=20, workspaces=["memory", "ref-y"],
                                      include_shared=False, vector_only=True)))
    assert "other reference" in mixed and "memory row 1" in mixed and "reference chapter" not in mixed


def test_empty_list_means_default_not_everything(service):
    got = _texts(service.search("q", "u1", limit=20, workspaces=[], include_shared=False, vector_only=True))
    assert "reference chapter" not in got and len(got) == 8


def test_workspace_condition_is_in_the_filter_of_both_legs(service):
    """The BM25 leg receives the same filter object, workspace condition included."""
    seen = {}
    real_client = service._memory.vector_store.client
    spy = MagicMock(wraps=real_client)
    service._memory.vector_store.client = spy
    service._lexical_pool_hits = MagicMock(side_effect=lambda m, q, f, limit: seen.setdefault("bm25", f) and [])
    service.search("q", "u1", limit=3, workspaces=["ref-x"], include_shared=False, vector_only=True)
    dense_filter = spy.query_points.call_args.kwargs["query_filter"]
    for flt in (dense_filter, seen["bm25"]):
        assert "metadata.workspace" in repr(flt) and "ref-x" in repr(flt)
    assert seen["bm25"] is dense_filter


# ── graph leg ────────────────────────────────────────────────────────────


@pytest.fixture
def graph_service():
    svc = MemoryService()
    svc._do_graph_search = MagicMock(return_value={"edges": []})
    return svc


def _groups(svc, **kw):
    kw.setdefault("visibility", None)
    kw.setdefault("include_shared", True)
    svc._search_graph_for_visibility(query="q", user_id="u1", project_id=None, limit=5, **kw)
    return svc._do_graph_search.call_args.kwargs["group_ids"]


def test_graph_default_groups_unchanged(graph_service):
    base = _groups(graph_service)
    assert not any("--ws--" in g for g in base)
    assert _groups(graph_service, workspaces=None) == base
    assert _groups(graph_service, workspaces=["memory"]) == base


def test_graph_adds_only_requested_workspace_groups(graph_service):
    groups = _groups(graph_service, workspaces=["ref-x"])
    assert groups == ["user--u1--ws--ref-x", "shared--ws--ref-x"]


def test_graph_memory_plus_workspace_and_private_scope(graph_service):
    both = _groups(graph_service, workspaces=["memory", "ref-x"])
    assert "user--u1" in both and "user--u1--ws--ref-x" in both
    private = _groups(graph_service, workspaces=["ref-x"], visibility="private")
    assert private == ["user--u1--ws--ref-x"]  # never another user's / shared groups


# ── MCP + REST plumbing ─────────────────────────────────────────────────


@pytest.fixture
def mcp_service():
    mock_svc = MagicMock(name="MemoryService")
    mock_svc.search.return_value = []
    original = mcp_server._service
    mcp_server._service = mock_svc
    yield mock_svc
    mcp_server._service = original


@pytest.mark.asyncio
async def test_mcp_recall_forwards_workspaces(mcp_service):
    await mcp_server.call_tool("recall_memories", {"query": "q", "user_id": "u1", "workspaces": ["ref-x"]})
    assert mcp_service.search.call_args.kwargs["workspaces"] == ["ref-x"]


@pytest.mark.asyncio
async def test_mcp_recall_default_passes_none(mcp_service):
    await mcp_server.call_tool("recall_memories", {"query": "q", "user_id": "u1"})
    assert mcp_service.search.call_args.kwargs["workspaces"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [["bad name"], ["ok", "x/../y"], ["a" + str(i) for i in range(11)], "ref-x", [5]])
async def test_mcp_recall_rejects_invalid_workspaces(mcp_service, bad):
    out = await mcp_server.call_tool("recall_memories", {"query": "q", "user_id": "u1", "workspaces": bad})
    assert "error" in json.loads(out[0].text)
    mcp_service.search.assert_not_called()


@pytest.mark.asyncio
async def test_mcp_tool_schema_and_description_document_workspaces():
    tool = next(t for t in await mcp_server.list_tools() if t.name == "recall_memories")
    assert tool.inputSchema["properties"]["workspaces"]["type"] == "array"
    assert "ref-" in tool.description and "workspaces" in tool.description


@pytest.fixture
def rest_client():
    from fastapi.testclient import TestClient

    import main

    mock_svc = MagicMock(name="MemoryService")
    mock_svc.search.return_value = []
    original = main._service
    main._service = mock_svc
    client = TestClient(main.app, raise_server_exceptions=False)
    yield client, mock_svc
    main._service = original


def test_rest_search_forwards_workspaces(rest_client):
    client, svc = rest_client
    r = client.post("/v1/search", json={"query": "q", "user_id": "u1", "workspaces": ["ref-x"]})
    assert r.status_code == 200
    assert svc.search.call_args.kwargs["workspaces"] == ["ref-x"]


def test_rest_search_default_passes_none(rest_client):
    client, svc = rest_client
    assert client.post("/v1/search", json={"query": "q", "user_id": "u1"}).status_code == 200
    assert svc.search.call_args.kwargs["workspaces"] is None


@pytest.mark.parametrize("bad", [["bad name"], ["x/y"], ["a" + str(i) for i in range(11)], ["a" * 101]])
def test_rest_search_rejects_invalid_workspaces(rest_client, bad):
    client, svc = rest_client
    r = client.post("/v1/search", json={"query": "q", "user_id": "u1", "workspaces": bad})
    assert r.status_code == 422
    svc.search.assert_not_called()


def test_rest_empty_workspaces_means_default(rest_client):
    client, svc = rest_client
    assert client.post("/v1/search", json={"query": "q", "user_id": "u1", "workspaces": []}).status_code == 200
    assert svc.search.call_args.kwargs["workspaces"] is None


@pytest.mark.asyncio
async def test_mcp_empty_workspaces_means_default(mcp_service):
    await mcp_server.call_tool("recall_memories", {"query": "q", "user_id": "u1", "workspaces": []})
    assert mcp_service.search.call_args.kwargs["workspaces"] is None


def test_empty_string_workspace_row_is_still_a_memory_row(service, pool):
    pool.upsert(COLLECTION, [_point(300, "memory row empty-string ws", workspace="", cos=0.98)])
    got = _texts(service.search("q", "u1", limit=20, include_shared=False, vector_only=True))
    assert "memory row empty-string ws" in got
    ref = _texts(service.search("q", "u1", limit=20, workspaces=["ref-x"], include_shared=False, vector_only=True))
    assert "memory row empty-string ws" not in ref


def _vis_point(pid, data, user, visibility, workspace="ref-x"):
    p = _point(pid, data, user=user, workspace=workspace, cos=0.95)
    p.payload["metadata"]["visibility"] = visibility
    return p


def test_workspace_does_not_cross_users(service, pool):
    """User B asking for ref-x never sees A's private ref-x row (personal pool
    is keyed on user_id, shared pool on visibility=shared); A's SHARED ref-x
    row is visible to B as in any shared recall."""
    pool.upsert(COLLECTION, [
        _vis_point(400, "A private ref row", "userA", "private"),
        _vis_point(401, "A shared ref row", "userA", "shared"),
    ])
    got = _texts(service.search("q", "userB", limit=20, workspaces=["ref-x"], vector_only=True))
    assert "A private ref row" not in got
    assert "A shared ref row" in got
    only_private = _texts(service.search("q", "userB", limit=20, workspaces=["ref-x"],
                                         include_shared=False, vector_only=True))
    assert only_private == []  # B owns nothing; no shared pool requested
    # the owner still reads her own private row
    mine = _texts(service.search("q", "userA", limit=20, workspaces=["ref-x"], include_shared=False, vector_only=True))
    assert "A private ref row" in mine


def test_tombstoned_rows_stay_excluded_under_every_workspace_selection(service, pool):
    """Supersession tombstones (dream_tombstoned) combine with the workspace filter."""
    for pid, ws in ((500, "__absent__"), (501, "ref-x")):
        p = _point(pid, f"tombstoned {pid}", workspace=ws, cos=0.999)
        p.payload["metadata"]["dream_tombstoned"] = True
        pool.upsert(COLLECTION, [p])
    for kw in ({}, {"workspaces": ["ref-x"]}, {"workspaces": ["memory", "ref-x"]}):
        got = _texts(service.search("q", "u1", limit=50, include_shared=False, vector_only=True, **kw))
        assert not any(t.startswith("tombstoned") for t in got)
