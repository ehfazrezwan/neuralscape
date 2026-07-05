"""Real-embedded-Kuzu tests for the NS schema extensions and Tier-1 ports
(solo engine, unit 3 — see docs/neuralscape/29-kuzu-port-inventory.md).

No mocks here: every test constructs the actual KuzuDriver on a tmp path and
runs the ported query text through ``execute_query``, the provider-portable
read path. The MagicMock-based graph tests elsewhere give zero Kuzu coverage
by design — this file is where dialect breaks surface.
"""

import asyncio
from types import SimpleNamespace

from mem0.memory.graphiti_memory import _build_graph_driver
from memory.kuzu_schema import apply_ns_kuzu_schema, ns_kuzu_schema_statements


def _driver(tmp_path):
    return _build_graph_driver(
        SimpleNamespace(graph_provider="kuzu", kuzu_path=str(tmp_path / "g.kuzu"))
    )


class TestNsKuzuSchema:
    def test_statements_declare_tables_before_columns(self):
        stmts = ns_kuzu_schema_statements()
        assert "Source" in stmts[0] and "DERIVED_FROM" in stmts[1]
        assert all(s.startswith("ALTER TABLE") for s in stmts[2:])

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
