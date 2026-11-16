"""Vendored-fork capabilities that looked inert, decided one by one (unit side).

* Graph communities (``UPDATE_COMMUNITIES``): the flag must work when ON. Two
  upstream defects made it dead: (1) ``add_episode(update_communities=True)``
  unpacked a list of per-node ``(communities, edges)`` pairs into two names, so
  any episode that resolved other than exactly two entities raised after the
  episode was stored; (2) nothing ever created the FIRST community of a group,
  so ``GET /v1/graph/communities`` stayed empty. The real-store proof is
  ``integration_tests/test_graph_communities_flag_on.py``.
* The legacy ``project:{id}`` / ``global`` / bare-``user_id`` group fallback in
  the Graphiti adapter is gone: no caller can reach it, so it now raises.
* mem0's SQLite history is off by default (``MEM0_HISTORY_ENABLED``).
* The Graphiti cross-encoder is built at start-up and IS reachable, through the
  ``search_config`` override of ``POST /graph/search``.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import third_party_telemetry  # noqa: F401  (must precede any mem0/graphiti import)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class _SyncBridge:
    """Runs the adapter's coroutines to completion, like ``_AsyncBridge.run``."""

    def run(self, coro, timeout=None):
        return asyncio.run(coro)


def _graph(*, update_communities: bool, graphiti=None):
    from mem0.memory.graphiti_memory import MemoryGraph

    mg = object.__new__(MemoryGraph)
    mg._bridge = _SyncBridge()
    mg._update_communities = update_communities
    mg._indices_built = True
    mg.graphiti = graphiti if graphiti is not None else MagicMock()
    return mg


def _episode_result():
    return SimpleNamespace(
        edges=[],
        nodes=[],
        episode=SimpleNamespace(uuid="ep-1", name="ep-name"),
    )


# --------------------------------------------------------------------------
# upstream defect 1: add_episode(update_communities=True) with N != 2 nodes
# --------------------------------------------------------------------------


def _graphiti_with_stubbed_pipeline(monkeypatch, node_count: int, updated):
    """A real ``Graphiti`` whose extraction/resolution steps are replaced by stubs.

    Only the community branch of ``add_episode`` is under test, so the LLM,
    driver and embedder are never touched.
    """

    import graphiti_core.graphiti as gm
    from graphiti_core.nodes import CommunityNode, EntityNode
    from graphiti_core.tracer import create_tracer

    nodes = [EntityNode(name=f"entity-{i}", group_id="g1") for i in range(node_count)]

    async def fake_extract_nodes(*a, **k):
        return list(nodes)

    async def fake_resolve(*a, **k):
        return list(nodes), {}, []

    async def fake_attrs(*a, **k):
        return list(nodes)

    async def fake_update_community(driver, llm, embedder, node):
        updated.append(node.name)
        return (
            [CommunityNode(name=f"community-of-{node.name}", group_id="g1", labels=["Community"])],
            [],
        )

    monkeypatch.setattr(gm, "extract_nodes", fake_extract_nodes)
    monkeypatch.setattr(gm, "resolve_extracted_nodes", fake_resolve)
    monkeypatch.setattr(gm, "extract_attributes_from_nodes", fake_attrs)
    monkeypatch.setattr(gm, "update_community", fake_update_community)

    g = object.__new__(gm.Graphiti)
    g.driver = SimpleNamespace(provider=None, _database="g1", clone=lambda database: g.driver)
    g.clients = SimpleNamespace(driver=g.driver)
    g.llm_client = object()
    g.embedder = object()
    g.max_coroutines = None
    g.tracer = create_tracer(None, "test")
    g.retrieve_episodes = AsyncMock(return_value=[])
    g._extract_and_resolve_edges = AsyncMock(return_value=([], [], []))

    async def fake_process(episode, *a, **k):
        return [], episode

    g._process_episode_data = fake_process
    return g


@pytest.mark.parametrize("node_count", [0, 1, 2, 3, 5])
def test_add_episode_with_update_communities_handles_any_node_count(monkeypatch, node_count):
    """Upstream unpacked the per-node results into two names: only N == 2 worked."""

    from graphiti_core.nodes import EpisodeType
    from graphiti_core.utils.datetime_utils import utc_now

    updated: list[str] = []
    g = _graphiti_with_stubbed_pipeline(monkeypatch, node_count, updated)

    result = asyncio.run(
        g.add_episode(
            name="ep",
            episode_body="some text",
            source_description="test",
            reference_time=utc_now(),
            source=EpisodeType.text,
            group_id="g1",
            update_communities=True,
        )
    )

    assert sorted(updated) == sorted(f"entity-{i}" for i in range(node_count))
    assert [c.name for c in result.communities] == [f"community-of-entity-{i}" for i in range(node_count)]
    assert result.community_edges == []


def test_add_episode_without_the_flag_never_touches_communities(monkeypatch):
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.utils.datetime_utils import utc_now

    updated: list[str] = []
    g = _graphiti_with_stubbed_pipeline(monkeypatch, 3, updated)
    result = asyncio.run(
        g.add_episode(
            name="ep",
            episode_body="x",
            source_description="t",
            reference_time=utc_now(),
            source=EpisodeType.text,
            group_id="g1",
        )
    )
    assert updated == [] and result.communities == [] and result.community_edges == []


# --------------------------------------------------------------------------
# upstream defect 2: nothing created the first community of a group
# --------------------------------------------------------------------------


class _FakeNeo4j:
    """In-memory stand-in for the Community/HAS_MEMBER part of Neo4j (MERGE on uuid, like Graphiti's saves)."""

    def __init__(self):
        self.communities: dict[str, dict] = {}  # uuid -> {group_id, name}
        self.members: dict[str, dict] = {}  # edge uuid -> {community, entity}
        self.queries: list[str] = []

    async def execute_query(self, query, **params):
        self.queries.append(query)
        group = params["g"]
        mine = {u: c for u, c in self.communities.items() if c["group_id"] == group}
        if query.lstrip().startswith("MATCH (c:Community {group_id: $g}) RETURN"):
            populated = {m["community"] for m in self.members.values()}
            return [{"uuid": u, "populated": u in populated} for u in mine], None, None
        if "DETACH DELETE" in query:
            populated = {m["community"] for m in self.members.values()}
            for u in mine:
                if u not in populated:
                    del self.communities[u]
            return [], None, None
        raise AssertionError(f"unexpected query: {query}")


def _community_stubs(monkeypatch, driver, *, clusters):
    """Patch the Graphiti calls ``_ensure_group_communities`` makes; ``clusters`` is a list of member-uuid lists."""

    import graphiti_core.utils.maintenance.community_operations as ops
    from graphiti_core.edges import CommunityEdge
    from graphiti_core.nodes import CommunityNode
    from graphiti_core.utils.datetime_utils import utc_now

    seen: dict = {"builds": [], "embedded": []}

    async def build(drv, llm_client, group_ids):
        seen["builds"].append(list(group_ids))
        await asyncio.sleep(0)  # let a concurrent builder interleave, as two workers would
        nodes, edges = [], []
        for i, members in enumerate(clusters):
            node = CommunityNode(name=f"c{i}", group_id=group_ids[0], labels=["Community"])
            nodes.append(node)
            edges += [
                CommunityEdge(source_node_uuid=node.uuid, target_node_uuid=m, group_id=group_ids[0], created_at=utc_now())
                for m in members
            ]
        return nodes, edges

    async def save_node(self, drv):
        driver.communities[self.uuid] = {"group_id": self.group_id, "name": self.name}

    async def save_edge(self, drv):
        driver.members[self.uuid] = {"community": self.source_node_uuid, "entity": self.target_node_uuid}

    async def embed(self, embedder):
        seen["embedded"].append(self.name)

    monkeypatch.setattr(ops, "build_communities", build)
    monkeypatch.setattr(CommunityNode, "save", save_node)
    monkeypatch.setattr(CommunityEdge, "save", save_edge)
    monkeypatch.setattr(CommunityNode, "generate_name_embedding", embed)
    return seen


def _graphiti_double(driver=None):
    g = MagicMock()
    g.driver = driver or _FakeNeo4j()
    g.add_episode = AsyncMock(return_value=_episode_result())
    # The global wipe in Graphiti.build_communities must never be used.
    g.build_communities = AsyncMock(side_effect=AssertionError("Graphiti.build_communities wipes every group"))
    g.max_coroutines = None
    return g


def test_flag_on_builds_the_first_communities_for_this_group_only(monkeypatch):
    g = _graphiti_double()
    seen = _community_stubs(monkeypatch, g.driver, clusters=[["e1", "e2"], ["e3"]])
    mg = _graph(update_communities=True, graphiti=g)

    out = mg.add(data="Alice works at Acme", filters={"user_id": "u", "group_id": "grp"})

    assert out["episode_uuid"] == "ep-1"
    assert g.add_episode.await_args.kwargs["update_communities"] is True
    assert seen["builds"] == [["grp"]], "the clustering pass must be scoped to the written group"
    assert len(g.driver.communities) == 2 and len(g.driver.members) == 3
    g.build_communities.assert_not_called()


def test_flag_on_does_not_rebuild_when_the_group_already_has_communities(monkeypatch):
    g = _graphiti_double()
    seen = _community_stubs(monkeypatch, g.driver, clusters=[["e1"]])
    mg = _graph(update_communities=True, graphiti=g)

    mg.add(data="one", filters={"user_id": "u", "group_id": "grp"})
    mg.add(data="two", filters={"user_id": "u", "group_id": "grp"})

    assert seen["builds"] == [["grp"]]


def test_flag_off_never_looks_at_communities(monkeypatch):
    g = _graphiti_double()
    seen = _community_stubs(monkeypatch, g.driver, clusters=[["e1"]])
    mg = _graph(update_communities=False, graphiti=g)

    mg.add(data="facts", filters={"user_id": "u", "group_id": "grp"})

    assert g.add_episode.await_args.kwargs["update_communities"] is False
    assert seen["builds"] == [] and g.driver.queries == []


def test_two_concurrent_builders_leave_one_community_per_cluster(monkeypatch):
    """Two graph workers (two adapter instances, no shared lock) writing a group's first episodes."""

    g = _graphiti_double()
    seen = _community_stubs(monkeypatch, g.driver, clusters=[["e1", "e2"], ["e3"]])
    a = _graph(update_communities=True, graphiti=g)
    b = _graph(update_communities=True, graphiti=g)

    async def both():
        await asyncio.gather(a._ensure_group_communities("grp"), b._ensure_group_communities("grp"))

    asyncio.run(both())

    assert len(seen["builds"]) == 2, "both builders must have raced (the check passes for each before either saves)"
    assert len(g.driver.communities) == 2, f"duplicate communities: {g.driver.communities}"
    assert len(g.driver.members) == 3, f"duplicate memberships: {g.driver.members}"


def test_a_memberless_community_is_rebuilt_by_the_next_write(monkeypatch):
    g = _graphiti_double()
    g.driver.communities["orphan"] = {"group_id": "grp", "name": "half-saved"}  # node saved, edges never were
    seen = _community_stubs(monkeypatch, g.driver, clusters=[["e1", "e2"]])
    mg = _graph(update_communities=True, graphiti=g)

    mg.add(data="more", filters={"user_id": "u", "group_id": "grp"})

    assert seen["builds"] == [["grp"]]
    assert "orphan" not in g.driver.communities
    assert len(g.driver.communities) == 1 and len(g.driver.members) == 2


def test_a_community_build_for_one_group_leaves_other_groups_alone(monkeypatch):
    g = _graphiti_double()
    g.driver.communities["other"] = {"group_id": "elsewhere", "name": "keep me"}
    _community_stubs(monkeypatch, g.driver, clusters=[["e1"]])
    _graph(update_communities=True, graphiti=g).add(data="x", filters={"user_id": "u", "group_id": "grp"})
    assert "other" in g.driver.communities


def test_a_failing_community_build_does_not_fail_the_write(monkeypatch, caplog):
    import graphiti_core.utils.maintenance.community_operations as ops

    g = _graphiti_double()
    _community_stubs(monkeypatch, g.driver, clusters=[])

    async def boom(*a, **k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(ops, "build_communities", boom)
    mg = _graph(update_communities=True, graphiti=g)

    with caplog.at_level(logging.WARNING):
        out = mg.add(data="facts", filters={"user_id": "u", "group_id": "grp"})

    assert out["episode_uuid"] == "ep-1", "the episode is already stored; the write must succeed"
    assert "Initial community build failed" in caplog.text


def test_the_flag_reaches_the_adapter_from_settings(monkeypatch):
    import config

    monkeypatch.setattr(config.settings, "update_communities", True)
    assert config.settings.get_mem0_config()["graph_store"]["config"]["update_communities"] is True
    monkeypatch.setattr(config.settings, "update_communities", False)
    assert config.settings.get_mem0_config()["graph_store"]["config"]["update_communities"] is False


# --------------------------------------------------------------------------
# the legacy group-id fallback is gone
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filters",
    [
        {},
        {"user_id": "alice"},
        {"user_id": "alice", "project_id": "p1"},
        {"user_id": "alice", "scope": "global"},
        {"user_id": "alice", "group_id": ""},
    ],
)
def test_a_missing_group_id_raises_instead_of_inventing_a_namespace(filters):
    mg = _graph(update_communities=False)
    with pytest.raises(ValueError, match="group_id"):
        mg._get_group_id(filters)
    with pytest.raises(ValueError, match="group_id"):
        mg._get_group_ids(filters)


@pytest.mark.parametrize("call", ["add", "search", "delete_all", "get_all"])
def test_no_adapter_entry_point_writes_reads_or_deletes_without_a_group(call):
    mg = _graph(update_communities=False)
    filters = {"user_id": "alice", "project_id": "p1"}
    with pytest.raises(ValueError, match="group_id"):
        if call == "add":
            mg.add(data="x", filters=filters)
        elif call == "search":
            mg.search("q", filters)
        elif call == "delete_all":
            mg.delete_all(filters)
        else:
            mg.get_all(filters)
    mg.graphiti.add_episode.assert_not_called() if call == "add" else None


def test_explicit_group_ids_are_honoured():
    mg = _graph(update_communities=False)
    assert mg._get_group_id({"group_id": "private--alice"}) == "private--alice"
    assert mg._get_group_ids({"group_id": "private--alice"}) == ["private--alice"]
    assert mg._get_group_ids({"group_ids": ["a", "b"], "group_id": "c"}) == ["a", "b"]


def test_every_service_graph_call_passes_an_explicit_group_id():
    """The two ``graph.add`` call sites in the write path (the only adapter calls the service makes)."""

    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent / "memory" / "write.py"
    calls = [
        node
        for node in ast.walk(ast.parse(src.read_text()))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).endswith("retry_transient")
        and node.args
        and ast.unparse(node.args[0]) == "self._memory.graph.add"
    ]
    assert len(calls) == 2, "the write path has exactly two graph.add call sites; re-check this pin"
    for call in calls:
        filters = next(k.value for k in call.keywords if k.arg == "filters")
        assert isinstance(filters, ast.Dict)
        assert "group_id" in [ast.literal_eval(k) for k in filters.keys]


# --------------------------------------------------------------------------
# mem0 SQLite history
# --------------------------------------------------------------------------


def test_history_is_off_by_default_and_never_names_the_home_file(monkeypatch):
    import config

    monkeypatch.setattr(config.settings, "mem0_history_enabled", False)
    assert config.settings.get_mem0_config()["history_db_path"] == ":memory:"


def test_history_enabled_leaves_mem0s_own_path_alone(monkeypatch):
    import config

    monkeypatch.setattr(config.settings, "mem0_history_enabled", True)
    assert "history_db_path" not in config.settings.get_mem0_config()


def test_the_no_history_store_covers_every_public_method_mem0_can_call():
    """If a mem0 sync adds a history/messages method, the null store must grow with it."""

    from memory.core import _NoHistory
    from mem0.memory.storage import SQLiteManager

    public = {n for n in dir(SQLiteManager) if not n.startswith("_") and callable(getattr(SQLiteManager, n))}
    missing = {n for n in public if not callable(getattr(_NoHistory, n, None))}
    assert missing == set(), f"_NoHistory lacks {sorted(missing)}"

    null = _NoHistory()
    assert null.add_history("id", None, "text", "ADD") is None
    assert null.get_history("id") == [] and null.get_last_messages("s") == []
    assert null.connection is None  # mem0's reset() checks ``db.connection`` before dropping the table


class _FakeMemory:
    """Stands in for ``Memory``: records the config and carries a closable ``db``."""

    built: list = []

    def __init__(self, config):
        self.config = config
        self.db = MagicMock(name="sqlite_manager")

    @classmethod
    def from_config(cls, config):
        inst = cls(config)
        cls.built.append(inst)
        return inst


def test_memory_from_config_replaces_the_history_store_when_disabled():
    from memory.core import _NoHistory, _memory_from_config

    memory = _memory_from_config(_FakeMemory, {"history_db_path": ":memory:"})
    assert isinstance(memory.db, _NoHistory)


def test_memory_from_config_keeps_the_real_history_store_when_enabled():
    from memory.core import _NoHistory, _memory_from_config

    memory = _memory_from_config(_FakeMemory, {})
    assert not isinstance(memory.db, _NoHistory)


def test_the_service_builds_mem0_with_history_off_by_default(monkeypatch):
    """End to end through ``MemoryService._get_memory``: the Memory it holds has no history store."""

    import config
    import memory.core as core

    monkeypatch.setattr(config.settings, "mem0_history_enabled", False)
    captured = {}

    class _Mem(_FakeMemory):
        @classmethod
        def from_config(cls, cfg):
            captured["history_db_path"] = cfg.get("history_db_path")
            return super().from_config(cfg)

    import mem0

    monkeypatch.setattr(mem0, "Memory", _Mem)
    svc = core.CoreMixin()
    monkeypatch.setattr(svc, "_ensure_indexes_at_startup", lambda: None, raising=False)
    monkeypatch.setattr(svc, "_retry_index_ensure", lambda: None, raising=False)
    svc._filter_indexes_ok = True
    svc._created_at_index_ok = True
    from mem0.utils.factory import GraphStoreFactory

    monkeypatch.setattr(GraphStoreFactory, "create", classmethod(lambda cls, provider, shim: MagicMock()))
    memory = svc._get_memory()

    assert captured["history_db_path"] == ":memory:"
    assert isinstance(memory.db, core._NoHistory)


# --------------------------------------------------------------------------
# cross-encoder: built at start-up, reachable through search_config
# --------------------------------------------------------------------------


def test_create_cross_encoder_builds_the_configured_provider():
    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from mem0.memory.graphiti_memory import _create_cross_encoder

    assert isinstance(_create_cross_encoder("gemini", "k"), GeminiRerankerClient)
    assert isinstance(_create_cross_encoder("openai", "k", "some-model"), OpenAIRerankerClient)
    with pytest.raises(ValueError):
        _create_cross_encoder("nope", "k")


def test_the_settings_providers_for_the_reranker_are_the_ones_the_factory_builds():
    import config

    s = config.Settings(_env_file=None, google_api_key="k")
    assert s._graphiti_ai_studio_models()["graphiti_reranker_provider"] == "gemini"
    assert s._graphiti_gateway_models()["graphiti_reranker_provider"] == "openai"


def test_search_config_override_with_the_cross_encoder_reaches_graphiti_search():
    """``POST /graph/search`` -> ``search_graph(search_config=...)`` -> ``Graphiti.search_(config)``."""

    from graphiti_core.search.search_config import EdgeReranker, EdgeSearchMethod
    from memory_service import MemoryService

    svc = MemoryService()
    graphiti = MagicMock()
    graphiti.search_ = AsyncMock(return_value=SimpleNamespace(edges=[], nodes=[], episodes=[], communities=[]))
    svc._graphiti = graphiti
    svc._bridge = SimpleNamespace(_loop=None)
    svc._get_graphiti = lambda: graphiti
    svc._run_on_bridge = lambda coro, timeout=30.0: asyncio.run(coro)
    svc._enrich_graph_results = lambda *a, **k: None

    svc._do_graph_search(
        "where does alice work",
        ["g1"],
        5,
        search_config={"edge_config": {"search_methods": ["bm25"], "reranker": "cross_encoder"}},
    )

    config = graphiti.search_.call_args.kwargs["config"]
    assert config.edge_config.reranker is EdgeReranker.cross_encoder
    assert config.edge_config.search_methods == [EdgeSearchMethod.bm25]
    assert config.limit == 5


def test_graphiti_search_ranks_edges_with_the_adapters_cross_encoder(monkeypatch):
    """Upstream ``search()`` calls ``cross_encoder.rank`` for an edge config that asks for it."""

    from graphiti_core.edges import EntityEdge
    from graphiti_core.search import search as search_mod
    from graphiti_core.search.search_config import EdgeReranker, EdgeSearchConfig, EdgeSearchMethod, SearchConfig
    from graphiti_core.search.search_filters import SearchFilters
    from graphiti_core.tracer import create_tracer
    from graphiti_core.utils.datetime_utils import utc_now

    def edge(i: int, fact: str):
        return EntityEdge(
            uuid=f"e{i}", group_id="g1", source_node_uuid="a", target_node_uuid="b",
            name="REL", fact=fact, created_at=utc_now(), episodes=[],
        )

    edges = [edge(1, "Bob likes tea"), edge(2, "Alice works at Acme")]

    async def fake_fulltext(*a, **k):
        return edges

    monkeypatch.setattr(search_mod, "edge_fulltext_search", fake_fulltext)

    class _Ranker:
        def __init__(self):
            self.calls = []

        async def rank(self, query, passages):
            self.calls.append((query, list(passages)))
            return sorted(((p, 1.0 if "Alice" in p else 0.1) for p in passages), key=lambda x: -x[1])

    ranker = _Ranker()
    clients = SimpleNamespace(
        driver=SimpleNamespace(provider=None),
        llm_client=None,
        embedder=SimpleNamespace(create=AsyncMock(return_value=[0.0])),
        cross_encoder=ranker,
        tracer=create_tracer(None, "t"),
    )
    config = SearchConfig(
        edge_config=EdgeSearchConfig(search_methods=[EdgeSearchMethod.bm25], reranker=EdgeReranker.cross_encoder),
        limit=5,
    )

    result = asyncio.run(search_mod.search(clients, "who works at Acme", ["g1"], config, SearchFilters()))

    assert ranker.calls and ranker.calls[0][0] == "who works at Acme"
    assert [e.fact for e in result.edges][0] == "Alice works at Acme"


def test_the_gemini_reranker_orders_passages_by_the_model_score():
    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
    from graphiti_core.llm_client import LLMConfig

    scores = {"tea": "5", "acme": "93", "noise": "n/a"}

    async def generate_content(model, contents, config):
        text = contents[0].parts[0].text
        for needle, score in scores.items():
            if needle in text:
                return SimpleNamespace(text=score)
        return SimpleNamespace(text="")

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    reranker = GeminiRerankerClient(config=LLMConfig(api_key="k"), client=client)

    ranked = asyncio.run(reranker.rank("who works at Acme", ["Bob likes tea", "Alice works at acme", "noise here"]))

    assert [p for p, _ in ranked] == ["Alice works at acme", "Bob likes tea", "noise here"]
    assert ranked[0][1] == pytest.approx(0.93) and ranked[-1][1] == 0.0


def test_the_gemini_reranker_works_repeatedly_on_the_bridge_loop_but_not_across_loops():
    """The reranker's HTTP client binds to the loop of its first call.

    Real searches always call it on the adapter's bridge loop, so repeated calls work; a throwaway
    ``asyncio.run`` loop (what the first version of the real-store test did) leaves it unusable for
    the next caller ("Event loop is closed"), which ``search_graph`` reports as a degraded result.
    Runs against a local HTTP stub of the Gemini endpoint.
    """

    import http.server
    import json
    import threading

    from google import genai
    from google.genai import types
    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
    from graphiti_core.llm_client import LLMConfig
    from mem0.memory.graphiti_memory import _AsyncBridge

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            body = json.dumps(
                {"candidates": [{"content": {"role": "model", "parts": [{"text": "90"}]}, "finishReason": "STOP"}]}
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    bridge = _AsyncBridge()
    try:
        client = genai.Client(api_key="k", http_options=types.HttpOptions(base_url=f"http://127.0.0.1:{server.server_port}"))
        reranker = GeminiRerankerClient(config=LLMConfig(api_key="k"), client=client)

        first = bridge.run(reranker.rank("q", ["a", "b"]), timeout=30)
        second = bridge.run(reranker.rank("q", ["a", "b"]), timeout=30)
        assert len(first) == len(second) == 2

        other = GeminiRerankerClient(config=LLMConfig(api_key="k"), client=genai.Client(
            api_key="k", http_options=types.HttpOptions(base_url=f"http://127.0.0.1:{server.server_port}")))
        asyncio.run(other.rank("q", ["a", "b"]))
        with pytest.raises(Exception, match="closed"):
            asyncio.run(other.rank("q", ["a", "b"]))  # a second throwaway loop: the pitfall
    finally:
        server.shutdown()
        bridge._loop.call_soon_threadsafe(bridge._loop.stop)
