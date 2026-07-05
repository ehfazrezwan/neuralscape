"""Real-embedded-Kuzu tests for the NS schema extensions and Tier-1 ports
(solo engine, unit 3 — see docs/neuralscape/29-kuzu-port-inventory.md).

No mocks here: every test constructs the actual KuzuDriver on a tmp path and
runs the ported query text through ``execute_query``, the provider-portable
read path. The MagicMock-based graph tests elsewhere give zero Kuzu coverage
by design — this file is where dialect breaks surface.
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from mem0.memory.graphiti_memory import _build_graph_driver
from memory.kuzu_schema import apply_ns_kuzu_schema, ns_kuzu_schema_statements


def _driver(tmp_path):
    return _build_graph_driver(
        SimpleNamespace(graph_provider="kuzu", kuzu_path=str(tmp_path / "g.kuzu"))
    )


class _PassthroughService(SimpleNamespace):
    """Stand-in for MemoryService: awaits bridge coroutines inline."""

    async def _run_on_bridge_async(self, coro, timeout: float = 30.0):
        return await coro


def _now():
    return datetime.now(timezone.utc)


class TestNsKuzuSchema:
    def test_statement_ordering(self):
        """FTS extension first, then tables/columns, FTS indices last."""
        stmts = ns_kuzu_schema_statements()
        assert stmts[0] == "INSTALL FTS" and stmts[1] == "LOAD EXTENSION FTS"
        assert "Source" in stmts[2] and "DERIVED_FROM" in stmts[3]
        assert all(s.startswith("CALL CREATE_FTS_INDEX") for s in stmts[-4:])
        alters = stmts[4:-4]
        assert alters and all(s.startswith("ALTER TABLE") for s in alters)

    def test_apply_declares_ns_columns(self, tmp_path):
        d = _driver(tmp_path)

        async def run():
            await apply_ns_kuzu_schema(d)
            await d.execute_query(
                "CREATE (n:Entity {uuid: $u, name: $n})", u="e1", n="x"
            )
            await d.execute_query(
                "MATCH (n:Entity {uuid: $u}) SET n.memory_id = $m, n.wiki_path = $w",
                u="e1", m="mem-1", w="wiki/x",
            )
            rows, _, _ = await d.execute_query(
                "MATCH (n:Entity {uuid: $u}) "
                "RETURN n.memory_id AS memory_id, n.wiki_path AS wiki_path",
                u="e1",
            )
            return rows

        assert asyncio.run(run()) == [{"memory_id": "mem-1", "wiki_path": "wiki/x"}]

    def test_dream_columns_on_relates_to_node(self, tmp_path):
        """Edge-level NS props live on the reified RelatesToNode_ table."""
        d = _driver(tmp_path)

        async def run():
            await apply_ns_kuzu_schema(d)
            await d.execute_query(
                "CREATE (r:RelatesToNode_ {uuid: $u, group_id: $g})",
                u="r1", g="user--local",
            )
            await d.execute_query(
                "MATCH (r:RelatesToNode_ {uuid: $u}) SET r.memory_id = $m",
                u="r1", m="mem-9",
            )
            rows, _, _ = await d.execute_query(
                "MATCH (r:RelatesToNode_ {uuid: $u}) RETURN r.memory_id AS m", u="r1"
            )
            return rows

        assert asyncio.run(run()) == [{"m": "mem-9"}]

    def test_apply_is_idempotent(self, tmp_path):
        d = _driver(tmp_path)

        async def run():
            await apply_ns_kuzu_schema(d)
            await apply_ns_kuzu_schema(d)  # re-run must not raise

        asyncio.run(run())

    def test_source_and_derived_from_roundtrip(self, tmp_path):
        d = _driver(tmp_path)

        async def run():
            await apply_ns_kuzu_schema(d)
            await d.execute_query(
                "CREATE (s:Source {key: $k, connector_id: $c, source_key: $sk})",
                k="conn::doc1", c="conn", sk="doc1",
            )
            await d.execute_query("CREATE (n:Episodic {uuid: $u})", u="ep1")
            await d.execute_query(
                "MATCH (n:Episodic {uuid: $u}), (s:Source {key: $k}) "
                "CREATE (n)-[:DERIVED_FROM]->(s)",
                u="ep1", k="conn::doc1",
            )
            rows, _, _ = await d.execute_query(
                "MATCH (n:Episodic)-[:DERIVED_FROM]->(s:Source) "
                "RETURN n.uuid AS uuid, s.key AS key"
            )
            return rows

        assert asyncio.run(run()) == [{"uuid": "ep1", "key": "conn::doc1"}]


class TestFtsBootstrap:
    def test_fts_index_live_after_bootstrap_and_maintained_on_insert(self, tmp_path):
        """Bootstrap creates graphiti's FTS indices; rows inserted AFTER index
        creation are searchable (Kuzu FTS is maintained, not a snapshot)."""
        d = _driver(tmp_path)

        async def run():
            await apply_ns_kuzu_schema(d)
            await d.execute_query(
                "CREATE (e:Episodic {uuid: $u, group_id: $g, content: $c})",
                u="ep-f1", g="user--local", c="the quick brown fox",
            )
            rows, _, _ = await d.execute_query(
                "CALL QUERY_FTS_INDEX('Episodic', 'episode_content', $q, TOP := $limit) "
                "WITH node, score WHERE node.group_id IN $group_ids "
                "RETURN node.uuid AS uuid, node.content AS content, score "
                "ORDER BY score DESC LIMIT $limit",
                q="fox", limit=3, group_ids=["user--local"],
            )
            return rows

        rows = asyncio.run(run())
        assert [r["uuid"] for r in rows] == ["ep-f1"]

    def test_group_filter_excludes_other_pools(self, tmp_path):
        """The exact composed query from search_episodes_fulltext (#3):
        group scoping must hold on Kuzu exactly as on Neo4j."""
        from graphiti_core.driver.driver import GraphProvider
        from graphiti_core.graph_queries import get_nodes_query

        d = _driver(tmp_path)
        cypher = (
            get_nodes_query("episode_content", "$q", limit=3, provider=GraphProvider.KUZU)
            + """
        WITH node, score
        WHERE node.group_id IN $group_ids
        RETURN node.uuid AS uuid, node.content AS content,
               node.created_at AS created_at, score
        ORDER BY score DESC LIMIT $limit
        """
        )

        async def run():
            await apply_ns_kuzu_schema(d)
            await d.execute_query(
                "CREATE (e:Episodic {uuid: $u, group_id: $g, content: $c})",
                u="mine", g="user--local", c="tokyo trip planning",
            )
            await d.execute_query(
                "CREATE (e:Episodic {uuid: $u, group_id: $g, content: $c})",
                u="theirs", g="user--other", c="tokyo restaurant list",
            )
            rows, _, _ = await d.execute_query(
                cypher, q="tokyo", group_ids=["user--local"], limit=3
            )
            return rows

        rows = asyncio.run(run())
        assert [r["uuid"] for r in rows] == ["mine"]


class TestTier1Ports:
    def test_episode_exists_probe_query_on_kuzu(self, tmp_path):
        """Exact Cypher from memory/write.py _graph_episode_exists (#5)."""
        d = _driver(tmp_path)
        cypher = (
            "MATCH (e:Episodic {group_id: $group_id, name: $name}) "
            "RETURN e.uuid AS uuid LIMIT 1"
        )

        async def run():
            await d.execute_query(
                "CREATE (e:Episodic {uuid: $u, group_id: $g, name: $n})",
                u="ep2", g="user--local", n="mem0_episode_abc",
            )
            hit, _, _ = await d.execute_query(
                cypher, group_id="user--local", name="mem0_episode_abc"
            )
            miss, _, _ = await d.execute_query(
                cypher, group_id="user--local", name="nope"
            )
            return hit, miss

        hit, miss = asyncio.run(run())
        assert [r["uuid"] for r in hit] == ["ep2"]
        assert miss == []

    def test_delete_episode_query_on_kuzu(self, tmp_path):
        """Exact Cypher from memory/graph_admin.py delete_episode (#4)."""
        d = _driver(tmp_path)

        async def run():
            await d.execute_query("CREATE (e:Episodic {uuid: $u})", u="ep3")
            await d.execute_query(
                "MATCH (e:Episodic {uuid: $uuid}) DETACH DELETE e", uuid="ep3"
            )
            rows, _, _ = await d.execute_query(
                "MATCH (e:Episodic {uuid: $u}) RETURN e.uuid AS uuid", u="ep3"
            )
            return rows

        assert asyncio.run(run()) == []


class TestGraphPatcherKuzu:
    """Tier-2/3 dreaming patcher ports against the real embedded driver."""

    def _bootstrapped(self, tmp_path):
        d = _driver(tmp_path)
        asyncio.run(apply_ns_kuzu_schema(d))
        return d

    def test_attach_memory_id_stamps_window_and_respects_coalesce(self, tmp_path):
        from extensions.dreaming.graph_patcher import attach_memory_id

        d = self._bootstrapped(tmp_path)
        now = _now()

        async def run():
            await d.execute_query(
                "CREATE (n:Entity {uuid: $u, group_id: $g, created_at: $ts})",
                u="e1", g="user--local", ts=now,
            )
            await d.execute_query(
                "CREATE (e:Episodic {uuid: $u, group_id: $g, created_at: $ts})",
                u="ep1", g="user--local", ts=now,
            )
            await d.execute_query(
                "CREATE (n:Entity {uuid: $u, group_id: $g, created_at: $ts})",
                u="other", g="user--other", ts=now,
            )
            first = await attach_memory_id(
                d, group_id="user--local", memory_id="m-1",
                visibility="private", owner_user_id="local",
                write_started_at=now,
            )
            second = await attach_memory_id(
                d, group_id="user--local", memory_id="m-2",
                visibility=None, owner_user_id=None,
                write_started_at=now,
            )
            rows, _, _ = await d.execute_query(
                "MATCH (n:Entity {uuid: $u}) RETURN n.memory_id AS m, "
                "n.ns_visibility AS v, n.ns_owner AS o", u="e1",
            )
            return first, second, rows

        first, second, rows = asyncio.run(run())
        assert first == 2  # Entity + Episodic in group; other group untouched
        assert second == 2  # matched again, but coalesce kept the first stamp
        assert rows == [{"m": "m-1", "v": "private", "o": "local"}]

    def test_attach_source_ref_merges_source_and_links(self, tmp_path):
        from extensions.dreaming.graph_patcher import attach_source_ref

        d = self._bootstrapped(tmp_path)
        now = _now()
        ref = {
            "connector_id": "gdrive",
            "external_id": "doc-9",
            "connector_type": "file",
            "url": "https://example.com/doc-9",
            "title": None,  # optional prop omitted — must not bind-error
        }

        async def run():
            await d.execute_query(
                "CREATE (e:Episodic {uuid: $u, group_id: $g, created_at: $ts})",
                u="ep-s", g="user--local", ts=now,
            )
            n1 = await attach_source_ref(
                d, group_id="user--local", memory_id="m-s",
                source_ref=ref, write_started_at=now,
            )
            n2 = await attach_source_ref(  # re-run: same Source row, no dupes
                d, group_id="user--local", memory_id="m-s",
                source_ref=ref, write_started_at=now,
            )
            srcs, _, _ = await d.execute_query(
                "MATCH (s:Source) RETURN s.key AS key, s.connector_id AS cid"
            )
            links, _, _ = await d.execute_query(
                "MATCH (n:Episodic)-[:DERIVED_FROM]->(s:Source) "
                "RETURN n.uuid AS uuid, n.ns_connector_id AS cid"
            )
            return n1, n2, srcs, links

        n1, n2, srcs, links = asyncio.run(run())
        assert n1 == 1 and n2 == 1
        assert srcs == [{"key": "gdrive::doc-9", "cid": "gdrive"}]
        assert links == [{"uuid": "ep-s", "cid": "gdrive"}]

    def test_patch_wiki_path_by_memory_ids(self, tmp_path):
        from extensions.dreaming.graph_patcher import patch_wiki_path_by_memory_ids

        d = self._bootstrapped(tmp_path)
        svc = _PassthroughService(_graphiti=SimpleNamespace(driver=d))

        async def run():
            await d.execute_query(
                "CREATE (n:Entity {uuid: 'w1', group_id: 'user--local', memory_id: 'm-w'})"
            )
            count = await patch_wiki_path_by_memory_ids(
                svc, memory_ids=["m-w"], wiki_path="wiki/topic.md",
                group_id="user--local",
            )
            rows, _, _ = await d.execute_query(
                "MATCH (n:Entity {uuid: 'w1'}) RETURN n.wiki_path AS w"
            )
            return count, rows

        count, rows = asyncio.run(run())
        assert count == 1 and rows == [{"w": "wiki/topic.md"}]

    def test_patch_playbook_path_by_memory_ids(self, tmp_path):
        from extensions.strategy_synthesizer.graph_patcher import (
            patch_playbook_path_by_memory_ids,
        )

        d = self._bootstrapped(tmp_path)
        svc = _PassthroughService(_graphiti=SimpleNamespace(driver=d))

        async def run():
            await d.execute_query(
                "CREATE (n:Entity {uuid: 'p1', memory_id: 'm-p'})"
            )
            count = await patch_playbook_path_by_memory_ids(
                svc, memory_ids=["m-p"], playbook_path="playbooks/breakout.md"
            )
            rows, _, _ = await d.execute_query(
                "MATCH (n:Entity {uuid: 'p1'}) RETURN n.strategy_playbook_path AS p"
            )
            return count, rows

        count, rows = asyncio.run(run())
        assert count == 1 and rows == [{"p": "playbooks/breakout.md"}]

    def test_patch_dream_path_by_memory_ids(self, tmp_path):
        from extensions.dreaming.graph_patcher import patch_dream_path_by_memory_ids

        d = self._bootstrapped(tmp_path)

        async def run():
            await d.execute_query(
                "CREATE (n:Entity {uuid: 'd1', group_id: 'user--local', memory_id: 'm-d'})"
            )
            count = await patch_dream_path_by_memory_ids(
                d, memory_ids=["m-d"], dream_path="dreams/2026-07-05.md",
                group_id="user--local",
            )
            rows, _, _ = await d.execute_query(
                "MATCH (n:Entity {uuid: 'd1'}) RETURN n.dream_path AS p"
            )
            return count, rows

        count, rows = asyncio.run(run())
        assert count == 1 and rows == [{"p": "dreams/2026-07-05.md"}]

    def test_invalidate_memory_graph_semantics_matrix(self, tmp_path):
        """The crown-jewel parity test: exclusive edges die, co-asserted and
        empty-provenance edges survive, already-invalid rows are untouched,
        node marking is unconditional."""
        from extensions.dreaming.graph_patcher import invalidate_memory_graph

        d = self._bootstrapped(tmp_path)
        g = "user--local"

        async def run():
            await d.execute_query(
                "CREATE (ep:Episodic {uuid: 'E1', group_id: $g, memory_id: 'm-x'})", g=g
            )
            await d.execute_query(
                "CREATE (n:Entity {uuid: 'N1', group_id: $g, memory_id: 'm-x'})", g=g
            )
            # r-solo: exclusively derived from E1 → must be invalidated
            await d.execute_query(
                "CREATE (r:RelatesToNode_ {uuid: 'r-solo', group_id: $g, episodes: $e})",
                g=g, e=["E1"],
            )
            # r-co: co-asserted by an external episode → must survive
            await d.execute_query(
                "CREATE (r:RelatesToNode_ {uuid: 'r-co', group_id: $g, episodes: $e})",
                g=g, e=["E1", "E-external"],
            )
            # r-empty: no recorded provenance → must survive
            await d.execute_query(
                "CREATE (r:RelatesToNode_ {uuid: 'r-empty', group_id: $g, episodes: $e})",
                g=g, e=[],
            )
            edges = await invalidate_memory_graph(
                d, group_id=g, memory_id="m-x", superseded_by="m-y"
            )
            state, _, _ = await d.execute_query(
                "MATCH (r:RelatesToNode_) WHERE r.group_id = $g "
                "RETURN r.uuid AS uuid, r.invalid_at IS NULL AS live ORDER BY r.uuid",
                g=g,
            )
            nodes, _, _ = await d.execute_query(
                "MATCH (n:Entity {uuid: 'N1'}) RETURN n.dream_superseded_by AS s"
            )
            # fail-safe: a memory with no stamped episodes invalidates nothing
            failsafe = await invalidate_memory_graph(
                d, group_id=g, memory_id="m-unstamped", superseded_by=""
            )
            return edges, state, nodes, failsafe

        edges, state, nodes, failsafe = asyncio.run(run())
        assert edges == 1
        assert state == [
            {"uuid": "r-co", "live": True},
            {"uuid": "r-empty", "live": True},
            {"uuid": "r-solo", "live": False},
        ]
        assert nodes == [{"s": "m-y"}]  # node marking landed
        assert failsafe == 0


class TestTier3Ports:
    """list_projects (#1), search enricher (#2), bridges hub scan (#13)."""

    def _bootstrapped(self, tmp_path):
        d = _driver(tmp_path)
        asyncio.run(apply_ns_kuzu_schema(d))
        return d

    def test_list_projects_on_kuzu(self, tmp_path):
        from memory.reads import ReadsMixin

        d = self._bootstrapped(tmp_path)

        async def seed():
            await d.execute_query(
                "CREATE (n:Entity {uuid: 'a', group_id: 'user--u1--project--alpha'})"
            )
            await d.execute_query(
                "CREATE (e:Episodic {uuid: 'b', group_id: 'shared--project--beta'})"
            )
            await d.execute_query(
                "CREATE (n:Entity {uuid: 'c', group_id: 'user--u1'})"  # global — skipped
            )
            await d.execute_query(
                "CREATE (n:Entity {uuid: 'd', group_id: 'user--other--project--gamma'})"
            )

        asyncio.run(seed())
        fake = SimpleNamespace(
            _get_graphiti=lambda: SimpleNamespace(driver=d),
            _run_on_bridge=lambda coro, timeout=10.0: asyncio.run(coro),
        )
        assert ReadsMixin.list_projects(fake, "u1") == ["alpha", "beta"]

    def test_enrich_graph_results_on_kuzu(self, tmp_path):
        from memory.search import SearchMixin

        d = self._bootstrapped(tmp_path)

        async def seed():
            await d.execute_query(
                "CREATE (n:Entity {uuid: 'n1', memory_id: 'm-n', wiki_path: 'wiki/n.md'})"
            )
            await d.execute_query(
                "CREATE (r:RelatesToNode_ {uuid: 'r1', memory_id: 'm-r', "
                "fact_embedding: $emb})",
                emb=[0.25, 0.5, 0.25],
            )

        asyncio.run(seed())
        fake = SimpleNamespace(
            _graphiti=SimpleNamespace(driver=d),
            _bridge=object(),
            _run_on_bridge=lambda coro, timeout=10.0: asyncio.run(coro),
        )
        nodes = [{"uuid": "n1"}]
        edges = [{"uuid": "r1"}]
        SearchMixin._enrich_graph_results(fake, nodes, edges, [])
        assert nodes[0]["memory_id"] == "m-n"
        assert nodes[0]["wiki_path"] == "wiki/n.md"
        assert edges[0]["memory_id"] == "m-r"
        assert [round(x, 2) for x in edges[0]["fact_embedding"]] == [0.25, 0.5, 0.25]

    def test_bridges_hub_scan_on_kuzu(self, tmp_path):
        from extensions.dreaming.bridges import fetch_graph_rows

        d = self._bootstrapped(tmp_path)

        async def run():
            # "Tokyo" spans two pools with two memories → hub.
            await d.execute_query(
                "CREATE (n:Entity {uuid: 'h1', name: 'Tokyo', memory_id: 'm-1', "
                "group_id: 'user--u1'})"
            )
            await d.execute_query(
                "CREATE (n:Entity {uuid: 'h2', name: ' tokyo ', memory_id: 'm-2', "
                "group_id: 'shared'})"
            )
            # Single-pool entity → filtered out.
            await d.execute_query(
                "CREATE (n:Entity {uuid: 's1', name: 'Osaka', memory_id: 'm-3', "
                "group_id: 'user--u1'})"
            )
            svc = _PassthroughService(
                _graphiti=SimpleNamespace(driver=d), _bridge=object()
            )
            return await fetch_graph_rows(svc, limit=10)

        rows = asyncio.run(run())
        assert len(rows) == 1
        assert rows[0]["name"].strip().lower() == "tokyo"
        assert sorted(rows[0]["memory_ids"]) == ["m-1", "m-2"]
