"""Tests for the graph-provider seam (solo engine, unit 2).

The mem0 fork's ``_build_graph_driver`` selects the Graphiti graph driver from
``graph_config.graph_provider``: ``neo4j`` (team default — a server driver) or
``kuzu`` (solo — the embedded single-process driver). The neo4j branch must be
byte-identical to the old hardcoded construction; the kuzu branch must fail
loud on a missing path or missing package, never fall back silently.

The KuzuSmoke class exercises the REAL embedded driver (kuzu is a dev
dependency): schema setup at construction, then a raw write/read round-trip.
"""

import asyncio
from types import SimpleNamespace

import pytest

from mem0.memory.graphiti_memory import _build_graph_driver


def _graph_config(**overrides) -> SimpleNamespace:
    base = dict(
        graph_provider="neo4j",
        kuzu_path="",
        url="neo4j://127.0.0.1:7687",
        username="neo4j",
        password="pw",
        database="memory",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestDriverSelection:
    def test_neo4j_provider_builds_neo4j_driver(self, monkeypatch):
        captured = {}

        class FakeNeo4jDriver:
            def __init__(self, uri, user, password, database):
                captured.update(uri=uri, user=user, password=password, database=database)

        monkeypatch.setattr(
            "mem0.memory.graphiti_memory.Neo4jDriver", FakeNeo4jDriver
        )
        driver = _build_graph_driver(_graph_config())
        assert isinstance(driver, FakeNeo4jDriver)
        assert captured == {
            "uri": "neo4j://127.0.0.1:7687",
            "user": "neo4j",
            "password": "pw",
            "database": "memory",
        }

    def test_missing_provider_attr_defaults_to_neo4j(self, monkeypatch):
        """Configs predating the seam (no graph_provider key) stay on neo4j."""

        class FakeNeo4jDriver:
            def __init__(self, **kw):
                pass

        monkeypatch.setattr(
            "mem0.memory.graphiti_memory.Neo4jDriver", FakeNeo4jDriver
        )
        cfg = _graph_config()
        del cfg.graph_provider
        assert isinstance(_build_graph_driver(cfg), FakeNeo4jDriver)

    def test_kuzu_without_path_is_loud_error(self):
        with pytest.raises(ValueError, match="kuzu_path"):
            _build_graph_driver(_graph_config(graph_provider="kuzu", kuzu_path=""))

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError, match="graph_provider"):
            _build_graph_driver(_graph_config(graph_provider="falkordb"))

    def test_kuzu_builds_embedded_driver(self, tmp_path):
        from graphiti_core.driver.driver import GraphProvider

        driver = _build_graph_driver(
            _graph_config(graph_provider="kuzu", kuzu_path=str(tmp_path / "g.kuzu"))
        )
        assert driver.provider == GraphProvider.KUZU

    def test_kuzu_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "g.kuzu"
        _build_graph_driver(
            _graph_config(graph_provider="kuzu", kuzu_path=str(nested))
        )
        assert nested.parent.is_dir()


class TestKuzuSmoke:
    """Round-trip against the real embedded Kuzu driver — no server, no mocks."""

    def test_write_read_roundtrip(self, tmp_path):
        driver = _build_graph_driver(
            _graph_config(graph_provider="kuzu", kuzu_path=str(tmp_path / "g.kuzu"))
        )

        async def roundtrip():
            await driver.execute_query(
                "CREATE (n:Entity {uuid: $uuid, name: $name, group_id: $gid})",
                uuid="u-1",
                name="solo smoke",
                gid="user--local",
            )
            rows, _, _ = await driver.execute_query(
                "MATCH (n:Entity {group_id: $gid}) RETURN n.uuid AS uuid, n.name AS name",
                gid="user--local",
            )
            return rows

        rows = asyncio.run(roundtrip())
        assert rows == [{"uuid": "u-1", "name": "solo smoke"}]

    def test_data_persists_across_driver_instances(self, tmp_path):
        """The graph is a durable file, not a per-process cache."""
        db = str(tmp_path / "g.kuzu")

        async def write():
            d = _build_graph_driver(_graph_config(graph_provider="kuzu", kuzu_path=db))
            await d.execute_query(
                "CREATE (n:Entity {uuid: $uuid, name: $name})", uuid="u-2", name="durable"
            )
            await d.close()

        async def read():
            d = _build_graph_driver(_graph_config(graph_provider="kuzu", kuzu_path=db))
            rows, _, _ = await d.execute_query(
                "MATCH (n:Entity {uuid: $uuid}) RETURN n.name AS name", uuid="u-2"
            )
            return rows

        asyncio.run(write())
        assert asyncio.run(read()) == [{"name": "durable"}]
