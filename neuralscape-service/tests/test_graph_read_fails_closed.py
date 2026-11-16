"""Graph reads return nothing, and make no graph call, when the scope is empty.

A read whose computed group scope is empty returns an empty result and does
not call the graph. The vendored Graphiti entry points follow the same rule,
so a caller that calls them directly behaves the same way.

Covers:
  (a) a read whose computed scope is empty returns no graph rows and never
      reaches the graph search;
  (b) the helper, and every service graph read that takes a group list,
      returns empty for an empty or missing list without a graph call;
  (c) the vendored search / episode entry points return empty for ``[]``;
  (d) the legacy graph routes return their empty value for an empty scope.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memory.groups import _live_node_uuids, _scoped_groups
from memory_service import MemoryService

EMPTY_GRAPH = {"edges": [], "nodes": [], "episodes": [], "communities": []}


@pytest.fixture
def service():
    return MemoryService()


def _results(**kw):
    base = dict(edges=[], nodes=[], episodes=[], communities=[])
    base.update(kw)
    return SimpleNamespace(**base)


def _scoped_graph(other_episode_uuid="11111111-2222-3333-4444-555555555555"):
    """A fake graph client that returns a row only when called without a
    group scope, so any such call shows up as a returned row."""
    g = MagicMock()

    def search_(query, config=None, group_ids=None, **kw):
        if group_ids:
            return _results()
        return _results(
            episodes=[
                SimpleNamespace(
                    uuid=other_episode_uuid,
                    name="ep",
                    content="episode text",
                    created_at=None,
                    valid_at=None,
                )
            ]
        )

    g.search_ = MagicMock(side_effect=search_)
    return g


# --------------------------------------------------------------------------
# the helper
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, [], (), "", "shared", b"x", [None], [""], ["  "], [None, "", 3]])
def test_scoped_groups_returns_empty_for_empty_or_unusable_input(value):
    assert _scoped_groups(value) == []


def test_scoped_groups_keeps_valid_ids_in_order_and_dedupes():
    assert _scoped_groups(["user--a", "shared", "user--a", "", None]) == ["user--a", "shared"]


# --------------------------------------------------------------------------
# (a) a computed scope that is empty
# --------------------------------------------------------------------------


def test_empty_computed_scope_makes_no_graph_call(service):
    graph = _scoped_graph()
    with patch("memory.search.settings.standards_enabled", False), \
         patch.object(service, "_get_graphiti", return_value=graph), \
         patch.object(service, "_run_on_bridge", side_effect=lambda x: x):
        out = service._search_graph_for_visibility(
            query="note", user_id="user-b", project_id=None, limit=10,
            visibility="standard", include_shared=True, include_episodes=True,
        )
    assert out == EMPTY_GRAPH
    graph.search_.assert_not_called()


def test_empty_computed_scope_returns_no_graph_rows_from_search(service):
    """End to end through search(): no graph episode row is returned."""
    graph = _scoped_graph()
    mem = MagicMock()
    mem.embedding_model.embed.return_value = [0.1] * 768
    mem.search.return_value = {"results": []}
    with patch("memory.search.settings.standards_enabled", False), \
         patch("memory.search.settings.graph_episode_recall_enabled", True), \
         patch.object(service, "_get_memory", return_value=mem), \
         patch.object(service, "_get_graphiti", return_value=graph), \
         patch.object(service, "_run_on_bridge", side_effect=lambda x: x):
        rows = service.search(
            query="note", user_id="user-b", project_id=None, visibility="standard",
        )
    assert [r for r in rows if r.source == "episode"] == []
    assert not any("episode text" in (r.memory or "") for r in rows)
    graph.search_.assert_not_called()


def test_non_empty_scope_is_passed_through_to_the_graph(service):
    graph = _scoped_graph()
    with patch("memory.search.settings.standards_enabled", True), \
         patch.object(service, "_get_graphiti", return_value=graph), \
         patch.object(service, "_run_on_bridge", side_effect=lambda x: x):
        service._search_graph_for_visibility(
            query="note", user_id="user-b", project_id="p1", limit=10,
            visibility="standard", include_shared=True,
        )
    groups = graph.search_.call_args.kwargs["group_ids"]
    assert groups and "standard" in groups
    assert not any("user-b" in g for g in groups)


@pytest.mark.parametrize("visibility", [None, "private", "shared", "standard"])
def test_every_visibility_hands_the_graph_a_non_empty_scope(service, visibility):
    graph = _scoped_graph()
    with patch("memory.search.settings.standards_enabled", False), \
         patch.object(service, "_get_graphiti", return_value=graph), \
         patch.object(service, "_run_on_bridge", side_effect=lambda x: x):
        service._search_graph_for_visibility(
            query="note", user_id="user-b", project_id=None, limit=10,
            visibility=visibility, include_shared=True,
        )
    for call in graph.search_.call_args_list:
        assert call.kwargs["group_ids"], f"empty scope sent for visibility={visibility!r}"


# --------------------------------------------------------------------------
# (b) every service graph read returns empty for an empty / None scope
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scope", [[], None, [""]])
def test_do_graph_search_returns_empty_for_an_empty_scope(service, scope):
    graph = _scoped_graph()
    with patch.object(service, "_get_graphiti", return_value=graph), \
         patch.object(service, "_run_on_bridge", side_effect=lambda x: x):
        out = service._do_graph_search(query="q", group_ids=scope, limit=5, include_episodes=True)
    assert out == EMPTY_GRAPH
    graph.search_.assert_not_called()


@pytest.mark.parametrize("scope", [[], None, [""]])
def test_do_graph_search_with_explicit_config_returns_empty_for_an_empty_scope(service, scope):
    graph = _scoped_graph()
    with patch.object(service, "_get_graphiti", return_value=graph), \
         patch.object(service, "_run_on_bridge", side_effect=lambda x: x):
        out = service._do_graph_search(
            query="q", group_ids=scope, limit=5,
            search_config={"limit": 5, "edge_config": {"search_methods": ["bm25"], "reranker": "rrf"}},
        )
    assert out == EMPTY_GRAPH
    graph.search_.assert_not_called()


def test_search_graph_returns_empty_for_an_empty_computed_scope(service):
    graph = _scoped_graph()
    with patch("memory.search._get_group_ids", return_value=[]), \
         patch.object(service, "_get_graphiti", return_value=graph), \
         patch.object(service, "_run_on_bridge", side_effect=lambda x: x):
        out = service.search_graph(query="q", user_id="user-b")
    assert out == EMPTY_GRAPH
    graph.search_.assert_not_called()


def _admin_graph():
    g = MagicMock()
    g.driver = MagicMock()
    g.retrieve_episodes = MagicMock(return_value=[])
    return g


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("get_graph_nodes", {}),
        ("get_graph_edges", {}),
        ("get_graph_episodes", {}),
        ("get_graph_communities", {}),
        ("search_episodes_fulltext", {"query": "episode text here"}),
    ],
)
def test_graph_listing_reads_return_empty_for_an_empty_computed_scope(service, method, kwargs):
    graph = _admin_graph()
    bridge = MagicMock(side_effect=lambda x, **kw: x)
    with patch("memory.graph_admin._get_group_ids", return_value=[]), \
         patch.object(service, "_get_graphiti", return_value=graph), \
         patch.object(service, "_run_on_bridge", bridge):
        out = getattr(service, method)(user_id="user-b", **kwargs)
    assert out == []
    graph.retrieve_episodes.assert_not_called()
    bridge.assert_not_called()  # nothing was scheduled on the graph bridge


def test_liveness_lookup_returns_empty_for_an_empty_scope():
    run = MagicMock()
    driver = MagicMock()
    assert _live_node_uuids(run, driver, ["n1", "n2"], []) == set()
    assert _live_node_uuids(run, driver, ["n1"], None) == set()
    run.assert_not_called()
    driver.session.assert_not_called()


def test_expire_edges_for_memory_makes_no_graph_call_without_a_group(service):
    graph = _scoped_graph()
    service._graphiti = graph
    with patch("memory.delete._scoped_groups", return_value=[]), \
         patch.object(service, "_run_on_bridge", side_effect=lambda x: x) as bridge:
        service._expire_graph_edges_for_memory(
            {"memory": "something", "user_id": "user-b", "metadata": {"visibility": "private"}}
        )
    graph.search_.assert_not_called()
    bridge.assert_not_called()


def test_expire_edges_for_groups_makes_no_graph_call_for_an_empty_list(service):
    service._graphiti = MagicMock()
    with patch.object(service, "_run_on_bridge") as bridge:
        service._expire_graph_edges_for_groups([])
        service._expire_graph_edges_for_groups([""])
    bridge.assert_not_called()
    service._graphiti.driver.session.assert_not_called()


# --------------------------------------------------------------------------
# (c) the vendored Graphiti entry points return empty for an empty list
# --------------------------------------------------------------------------


def _run_vendored_search(group_ids):
    from graphiti_core.search import search as search_mod
    from graphiti_core.search.search_config import SearchConfig
    from graphiti_core.search.search_filters import SearchFilters

    seen: list = []

    def recorder(name):
        async def _fake(*args, **kwargs):
            # edge/node/episode/community search: group_ids is the 5th positional
            seen.append((name, args[4] if len(args) > 4 else None))
            return [], []

        return _fake

    clients = SimpleNamespace(driver=MagicMock(), embedder=MagicMock(), cross_encoder=MagicMock(), tracer=None)
    with patch.object(search_mod, "edge_search", recorder("edge")), \
         patch.object(search_mod, "node_search", recorder("node")), \
         patch.object(search_mod, "episode_search", recorder("episode")), \
         patch.object(search_mod, "community_search", recorder("community")):
        result = asyncio.run(
            search_mod.search(clients, "hello world", group_ids, SearchConfig(), SearchFilters())
        )
    return result, seen


@pytest.mark.parametrize("scope", [[], [""]])
def test_vendored_search_returns_empty_for_an_empty_list(scope):
    result, seen = _run_vendored_search(scope)
    assert seen == [], "an empty group list must not reach any search leg"
    assert result.edges == [] and result.nodes == [] and result.episodes == [] and result.communities == []


def test_vendored_search_still_scopes_a_real_list_and_leaves_none_as_upstream():
    _, seen = _run_vendored_search(["user--a"])
    assert {s[1] if s[1] is None else tuple(s[1]) for s in seen} == {("user--a",)}
    assert len(seen) == 4
    _, seen = _run_vendored_search(None)
    assert len(seen) == 4 and all(s[1] is None for s in seen)


def test_vendored_retrieve_episodes_returns_empty_for_an_empty_list():
    from graphiti_core.utils.maintenance.graph_data_operations import retrieve_episodes

    driver = MagicMock()
    driver.graph_operations_interface = None

    async def _exec(*a, **kw):
        return [], None, None

    driver.execute_query = MagicMock(side_effect=_exec)
    when = SimpleNamespace()
    assert asyncio.run(retrieve_episodes(driver, when, 5, [])) == []
    driver.execute_query.assert_not_called()
    # control: None keeps its documented meaning (no group filter)
    asyncio.run(retrieve_episodes(driver, when, 5, None))
    driver.execute_query.assert_called_once()
    assert "group_ids" not in driver.execute_query.call_args.kwargs


# --------------------------------------------------------------------------
# (d) legacy graph routes
# --------------------------------------------------------------------------


@pytest.fixture
def legacy_client(monkeypatch):
    from fastapi.testclient import TestClient

    import main

    graph = MagicMock(name="Graphiti")
    calls = []

    def _bridge(coro):
        calls.append(coro)
        coro.close()
        return []

    monkeypatch.setattr(main, "_graphiti", graph)
    monkeypatch.setattr(main, "_memory", MagicMock(name="Memory"))
    monkeypatch.setattr(main, "_bridge", MagicMock(name="Bridge"))
    monkeypatch.setattr(main, "_run_on_bridge", _bridge)
    client = TestClient(main.app, raise_server_exceptions=False)
    return main, graph, calls, client


@pytest.mark.parametrize(
    "method,path,key",
    [
        ("get", "/graph/nodes", "nodes"),
        ("get", "/graph/edges", "edges"),
        ("get", "/graph/episodes", "episodes"),
        ("get", "/graph/communities", "communities"),
    ],
)
def test_legacy_listing_routes_return_empty_for_an_empty_scope(legacy_client, monkeypatch, method, path, key):
    main, graph, calls, client = legacy_client
    monkeypatch.setattr(main, "_resolve_user_id", lambda request, uid: "")
    resp = getattr(client, method)(path)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", key: []}
    assert calls == [], "an empty scope must not schedule a graph call"
    graph.retrieve_episodes.assert_not_called()


def test_legacy_graph_search_returns_empty_for_an_empty_scope(legacy_client, monkeypatch):
    main, graph, calls, client = legacy_client
    monkeypatch.setattr(main, "_resolve_user_id", lambda request, uid: "")
    resp = client.post("/graph/search", json={"query": "note"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", **EMPTY_GRAPH}
    graph.search_.assert_not_called()
    assert calls == []


def test_legacy_listing_route_still_queries_for_a_real_scope(legacy_client, monkeypatch):
    main, graph, calls, client = legacy_client
    monkeypatch.setattr(main, "_resolve_user_id", lambda request, uid: "user-a")
    resp = client.get("/graph/episodes")
    assert resp.status_code == 200
    assert len(calls) == 1
    assert graph.retrieve_episodes.call_args.kwargs["group_ids"] == ["user-a"]
