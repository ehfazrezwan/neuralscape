"""Graph reads fail closed on an empty or missing group list.

Graphiti reads an empty ``group_ids`` as "no group filter". The service must
therefore never hand it an empty scope: a computed-empty list (for example the
standard tier while ``STANDARDS_ENABLED`` is off) means no groups, so no rows,
and the graph call is not made. The vendored Graphiti entry points keep the
same rule so a caller that forgets the service helper is still safe.

Covers:
  (a) ``visibility="standard"`` with the standards tier off returns no graph
      rows and never reaches the graph search;
  (b) the helper, and every service graph read that takes a group list,
      refuses an empty or None list without a graph call;
  (c) the vendored search / episode entry points do not widen ``[]``.
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


def _unscoped_graph(foreign_episode_uuid="11111111-2222-3333-4444-555555555555"):
    """A fake Graphiti that behaves like the real one: an empty group list is
    widened to every group, so a foreign episode comes back."""
    g = MagicMock()

    def search_(query, config=None, group_ids=None, **kw):
        if group_ids:
            return _results()
        return _results(
            episodes=[
                SimpleNamespace(
                    uuid=foreign_episode_uuid,
                    name="ep",
                    content="another user's private note",
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
def test_scoped_groups_refuses_empty_or_unusable_input(value):
    assert _scoped_groups(value) == []


def test_scoped_groups_keeps_valid_ids_in_order_and_dedupes():
    assert _scoped_groups(["user--a", "shared", "user--a", "", None]) == ["user--a", "shared"]


# --------------------------------------------------------------------------
# (a) visibility=standard with the tier off
# --------------------------------------------------------------------------


def test_standard_visibility_with_tier_off_makes_no_graph_call(service):
    graph = _unscoped_graph()
    with patch("memory.search.settings.standards_enabled", False), \
         patch.object(service, "_get_graphiti", return_value=graph), \
         patch.object(service, "_run_on_bridge", side_effect=lambda x: x):
        out = service._search_graph_for_visibility(
            query="note", user_id="user-b", project_id=None, limit=10,
            visibility="standard", include_shared=True, include_episodes=True,
        )
    assert out == EMPTY_GRAPH
    graph.search_.assert_not_called()


def test_standard_visibility_with_tier_off_returns_no_graph_rows_from_search(service):
    """End to end through search(): the foreign episode must not surface."""
    graph = _unscoped_graph()
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
    assert not any("private note" in (r.memory or "") for r in rows)
    graph.search_.assert_not_called()


def test_standard_visibility_with_tier_on_still_scopes_to_the_standard_groups(service):
    graph = _unscoped_graph()
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
    graph = _unscoped_graph()
    with patch("memory.search.settings.standards_enabled", False), \
         patch.object(service, "_get_graphiti", return_value=graph), \
         patch.object(service, "_run_on_bridge", side_effect=lambda x: x):
        service._search_graph_for_visibility(
            query="note", user_id="user-b", project_id=None, limit=10,
            visibility=visibility, include_shared=True,
        )
    for call in graph.search_.call_args_list:
        assert call.kwargs["group_ids"], f"empty scope sent for visibility={visibility!r}"


def test_workspace_partition_with_empty_base_scope_makes_no_graph_call(service):
    graph = _unscoped_graph()
    with patch("memory.search.settings.standards_enabled", False), \
         patch.object(service, "_get_graphiti", return_value=graph), \
         patch.object(service, "_run_on_bridge", side_effect=lambda x: x):
        out = service._search_graph_for_visibility(
            query="note", user_id="user-b", project_id=None, limit=10,
            visibility="standard", include_shared=True, workspaces=["memory"],
        )
    assert out == EMPTY_GRAPH
    graph.search_.assert_not_called()


# --------------------------------------------------------------------------
# (b) every service graph read refuses an empty / None scope
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scope", [[], None, [""]])
def test_do_graph_search_refuses_an_empty_scope(service, scope):
    graph = _unscoped_graph()
    with patch.object(service, "_get_graphiti", return_value=graph), \
         patch.object(service, "_run_on_bridge", side_effect=lambda x: x):
        out = service._do_graph_search(query="q", group_ids=scope, limit=5, include_episodes=True)
    assert out == EMPTY_GRAPH
    graph.search_.assert_not_called()


@pytest.mark.parametrize("scope", [[], None, [""]])
def test_do_graph_search_with_explicit_config_refuses_an_empty_scope(service, scope):
    graph = _unscoped_graph()
    with patch.object(service, "_get_graphiti", return_value=graph), \
         patch.object(service, "_run_on_bridge", side_effect=lambda x: x):
        out = service._do_graph_search(
            query="q", group_ids=scope, limit=5,
            search_config={"limit": 5, "edge_config": {"search_methods": ["bm25"], "reranker": "rrf"}},
        )
    assert out == EMPTY_GRAPH
    graph.search_.assert_not_called()


def test_search_graph_refuses_an_empty_computed_scope(service):
    graph = _unscoped_graph()
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
        ("search_episodes_fulltext", {"query": "private note here"}),
    ],
)
def test_graph_listing_reads_refuse_an_empty_computed_scope(service, method, kwargs):
    graph = _admin_graph()
    bridge = MagicMock(side_effect=lambda x, **kw: x)
    with patch("memory.graph_admin._get_group_ids", return_value=[]), \
         patch.object(service, "_get_graphiti", return_value=graph), \
         patch.object(service, "_run_on_bridge", bridge):
        out = getattr(service, method)(user_id="user-b", **kwargs)
    assert out == []
    graph.retrieve_episodes.assert_not_called()
    bridge.assert_not_called()  # nothing was scheduled on the graph bridge


def test_liveness_lookup_refuses_an_empty_scope():
    run = MagicMock()
    driver = MagicMock()
    assert _live_node_uuids(run, driver, ["n1", "n2"], []) == set()
    assert _live_node_uuids(run, driver, ["n1"], None) == set()
    run.assert_not_called()
    driver.session.assert_not_called()


def test_expire_edges_for_memory_makes_no_graph_call_without_a_group(service):
    graph = _unscoped_graph()
    service._graphiti = graph
    with patch("memory.delete._scoped_groups", return_value=[]), \
         patch.object(service, "_run_on_bridge", side_effect=lambda x: x) as bridge:
        service._expire_graph_edges_for_memory(
            {"memory": "something", "user_id": "user-b", "metadata": {"visibility": "private"}}
        )
    graph.search_.assert_not_called()
    bridge.assert_not_called()


def test_expire_edges_for_groups_refuses_an_empty_list(service):
    service._graphiti = MagicMock()
    assert service._expire_graph_edges_for_groups([]) == 0
    service._graphiti.driver.session.assert_not_called()


# --------------------------------------------------------------------------
# (c) the vendored Graphiti entry points do not widen an empty list
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
def test_vendored_search_does_not_widen_an_empty_list(scope):
    result, seen = _run_vendored_search(scope)
    assert seen == [], "an empty group list reached a search leg (widened to no filter)"
    assert result.edges == [] and result.nodes == [] and result.episodes == [] and result.communities == []


def test_vendored_search_still_scopes_a_real_list_and_leaves_none_unscoped():
    _, seen = _run_vendored_search(["user--a"])
    assert {s[1] if s[1] is None else tuple(s[1]) for s in seen} == {("user--a",)}
    assert len(seen) == 4
    _, seen = _run_vendored_search(None)
    assert len(seen) == 4 and all(s[1] is None for s in seen)


def test_vendored_retrieve_episodes_does_not_widen_an_empty_list():
    from graphiti_core.utils.maintenance.graph_data_operations import retrieve_episodes

    driver = MagicMock()
    driver.graph_operations_interface = None

    async def _exec(*a, **kw):
        return [], None, None

    driver.execute_query = MagicMock(side_effect=_exec)
    when = SimpleNamespace()
    assert asyncio.run(retrieve_episodes(driver, when, 5, [])) == []
    driver.execute_query.assert_not_called()
    # control: None is still the explicit unscoped read
    asyncio.run(retrieve_episodes(driver, when, 5, None))
    driver.execute_query.assert_called_once()
    assert "group_ids" not in driver.execute_query.call_args.kwargs
