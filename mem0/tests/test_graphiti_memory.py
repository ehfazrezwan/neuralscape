"""Unit tests for the Graphiti MemoryGraph adapter (mem0.memory.graphiti_memory)."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _AsyncBridge tests (no mocking needed — tests the real implementation)
# ---------------------------------------------------------------------------


class TestAsyncBridge:
    def test_runs_coroutine_and_returns_result(self):
        from mem0.memory.graphiti_memory import _AsyncBridge

        bridge = _AsyncBridge()

        async def coro():
            return 42

        assert bridge.run(coro()) == 42

    def test_propagates_exception(self):
        from mem0.memory.graphiti_memory import _AsyncBridge

        bridge = _AsyncBridge()

        async def boom():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            bridge.run(boom())


# ---------------------------------------------------------------------------
# Shared fixture: patches heavy Graphiti imports so MemoryGraph.__init__
# doesn't attempt real Neo4j connections or LLM client creation.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_graphiti_imports():
    """Patch external dependencies so MemoryGraph can be instantiated in isolation."""
    mock_driver = MagicMock(name="Neo4jDriver")
    mock_graphiti_instance = MagicMock(name="Graphiti")
    mock_graphiti_instance.driver = mock_driver

    mock_graphiti_cls = MagicMock(name="GraphitiClass", return_value=mock_graphiti_instance)
    mock_neo4j_driver_cls = MagicMock(name="Neo4jDriverClass", return_value=mock_driver)

    patches = [
        patch("mem0.memory.graphiti_memory.Neo4jDriver", mock_neo4j_driver_cls),
        patch("mem0.memory.graphiti_memory.Graphiti", mock_graphiti_cls),
        patch("mem0.memory.graphiti_memory._create_llm_client", return_value=MagicMock()),
        patch("mem0.memory.graphiti_memory._create_embedder", return_value=MagicMock()),
        patch("mem0.memory.graphiti_memory._create_cross_encoder", return_value=MagicMock()),
    ]
    for p in patches:
        p.start()

    yield {
        "graphiti_cls": mock_graphiti_cls,
        "graphiti": mock_graphiti_instance,
        "driver_cls": mock_neo4j_driver_cls,
        "driver": mock_driver,
    }

    for p in patches:
        p.stop()


def _make_config():
    """Build a minimal MemoryConfig with graphiti graph_store."""
    from mem0.configs.base import MemoryConfig

    return MemoryConfig(
        graph_store={
            "provider": "graphiti",
            "config": {
                "url": "neo4j://localhost:7687",
                "username": "neo4j",
                "password": "test",
                "database": "testdb",
                "graphiti_llm_provider": "gemini",
                "graphiti_embedder_provider": "gemini",
            },
        },
    )


# ---------------------------------------------------------------------------
# MemoryGraph __init__
# ---------------------------------------------------------------------------


class TestMemoryGraphInit:
    def test_creates_driver_with_config(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)

        mock_graphiti_imports["driver_cls"].assert_called_once_with(
            uri="neo4j://localhost:7687",
            user="neo4j",
            password="test",
            database="testdb",
        )

    def test_creates_graphiti_instance(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)

        mock_graphiti_imports["graphiti_cls"].assert_called_once()

    def test_has_bridge_attr(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph, _AsyncBridge

        config = _make_config()
        mg = MemoryGraph(config)

        assert hasattr(mg, "_bridge")
        assert isinstance(mg._bridge, _AsyncBridge)

    def test_creates_llm_client(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph, _create_llm_client

        config = _make_config()
        mg = MemoryGraph(config)

        _create_llm_client.assert_called_once()


# ---------------------------------------------------------------------------
# MemoryGraph.add
# ---------------------------------------------------------------------------


class TestMemoryGraphAdd:
    def test_calls_add_episode(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)

        # Mock add_episode result
        mock_edge = MagicMock()
        mock_edge.source_node_uuid = "uuid-1"
        mock_edge.target_node_uuid = "uuid-2"
        mock_edge.name = "works_at"

        mock_node_1 = MagicMock()
        mock_node_1.uuid = "uuid-1"
        mock_node_1.name = "Alice"

        mock_node_2 = MagicMock()
        mock_node_2.uuid = "uuid-2"
        mock_node_2.name = "Acme"

        mock_result = MagicMock()
        mock_result.edges = [mock_edge]
        mock_result.nodes = [mock_node_1, mock_node_2]

        mock_graphiti_imports["graphiti"].add_episode = AsyncMock(return_value=mock_result)
        mock_graphiti_imports["graphiti"].build_indices_and_constraints = AsyncMock()

        result = mg.add("Alice works at Acme", {"user_id": "u1"})

        mock_graphiti_imports["graphiti"].add_episode.assert_called_once()
        assert "added_entities" in result
        assert "deleted_entities" in result
        assert len(result["added_entities"]) == 1
        assert result["added_entities"][0]["source"] == "Alice"
        assert result["added_entities"][0]["destination"] == "Acme"
        assert result["added_entities"][0]["relationship"] == "works_at"

    def test_maps_user_id_to_group_id(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)

        mock_result = MagicMock()
        mock_result.edges = []
        mock_result.nodes = []
        mock_graphiti_imports["graphiti"].add_episode = AsyncMock(return_value=mock_result)
        mock_graphiti_imports["graphiti"].build_indices_and_constraints = AsyncMock()

        mg.add("test", {"user_id": "user123"})

        call_kwargs = mock_graphiti_imports["graphiti"].add_episode.call_args
        assert call_kwargs.kwargs.get("group_id") == "user123" or call_kwargs[1].get("group_id") == "user123"

    def test_defaults_group_id_to_default(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)

        mock_result = MagicMock()
        mock_result.edges = []
        mock_result.nodes = []
        mock_graphiti_imports["graphiti"].add_episode = AsyncMock(return_value=mock_result)
        mock_graphiti_imports["graphiti"].build_indices_and_constraints = AsyncMock()

        mg.add("test", {})

        call_kwargs = mock_graphiti_imports["graphiti"].add_episode.call_args
        assert call_kwargs.kwargs.get("group_id") == "default" or call_kwargs[1].get("group_id") == "default"


# ---------------------------------------------------------------------------
# MemoryGraph.search
# ---------------------------------------------------------------------------


class TestMemoryGraphSearch:
    def test_calls_graphiti_search(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)

        mock_edge = MagicMock()
        mock_edge.source_node_uuid = "uuid-1"
        mock_edge.target_node_uuid = "uuid-2"
        mock_edge.name = "works_at"
        mock_edge.fact = "Alice works at Acme"

        mock_node_1 = MagicMock()
        mock_node_1.uuid = "uuid-1"
        mock_node_1.name = "Alice"

        mock_node_2 = MagicMock()
        mock_node_2.uuid = "uuid-2"
        mock_node_2.name = "Acme"

        mock_graphiti_imports["graphiti"].search = AsyncMock(return_value=[mock_edge])
        mock_graphiti_imports["graphiti"].build_indices_and_constraints = AsyncMock()

        with patch("mem0.memory.graphiti_memory.EntityNode") as MockEntityNode:
            MockEntityNode.get_by_uuids = AsyncMock(return_value=[mock_node_1, mock_node_2])

            results = mg.search("Where does Alice work?", {"user_id": "u1"})

        assert len(results) == 1
        assert results[0]["source"] == "Alice"
        assert results[0]["destination"] == "Acme"
        assert results[0]["relationship"] == "works_at"

    def test_returns_empty_on_exception(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)

        mock_graphiti_imports["graphiti"].search = AsyncMock(side_effect=Exception("search failed"))
        mock_graphiti_imports["graphiti"].build_indices_and_constraints = AsyncMock()

        results = mg.search("test", {"user_id": "u1"})
        assert results == []


# ---------------------------------------------------------------------------
# MemoryGraph.get_all
# ---------------------------------------------------------------------------


class TestMemoryGraphGetAll:
    def test_calls_get_by_group_ids(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)

        mock_edge = MagicMock()
        mock_edge.source_node_uuid = "uuid-1"
        mock_edge.target_node_uuid = "uuid-2"
        mock_edge.name = "works_at"
        mock_edge.fact = "Alice works at Acme"

        mock_node_1 = MagicMock()
        mock_node_1.uuid = "uuid-1"
        mock_node_1.name = "Alice"

        mock_node_2 = MagicMock()
        mock_node_2.uuid = "uuid-2"
        mock_node_2.name = "Acme"

        mock_graphiti_imports["graphiti"].build_indices_and_constraints = AsyncMock()

        with patch("mem0.memory.graphiti_memory.EntityEdge") as MockEntityEdge, \
             patch("mem0.memory.graphiti_memory.EntityNode") as MockEntityNode:
            MockEntityEdge.get_by_group_ids = AsyncMock(return_value=[mock_edge])
            MockEntityNode.get_by_uuids = AsyncMock(return_value=[mock_node_1, mock_node_2])

            results = mg.get_all({"user_id": "u1"})

        assert len(results) == 1
        # get_all uses "target" key, not "destination"
        assert results[0]["target"] == "Acme"
        assert results[0]["source"] == "Alice"
        assert results[0]["relationship"] == "works_at"

    def test_returns_empty_on_groups_edges_not_found(self, mock_graphiti_imports):
        from graphiti_core.errors import GroupsEdgesNotFoundError

        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)

        mock_graphiti_imports["graphiti"].build_indices_and_constraints = AsyncMock()

        with patch("mem0.memory.graphiti_memory.EntityEdge") as MockEntityEdge:
            MockEntityEdge.get_by_group_ids = AsyncMock(
                side_effect=GroupsEdgesNotFoundError("no edges")
            )
            results = mg.get_all({"user_id": "u1"})

        assert results == []


# ---------------------------------------------------------------------------
# MemoryGraph.delete_all
# ---------------------------------------------------------------------------


class TestMemoryGraphDeleteAll:
    def test_calls_node_delete_by_group_id(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)

        # Node is imported inside the _delete() coroutine from graphiti_core.nodes
        with patch("graphiti_core.nodes.Node") as MockNode:
            MockNode.delete_by_group_id = AsyncMock()
            mg.delete_all({"user_id": "test_user"})
            MockNode.delete_by_group_id.assert_called_once()
            call_args = MockNode.delete_by_group_id.call_args
            assert call_args.args[1] == "test_user"


# ---------------------------------------------------------------------------
# Group ID mapping
# ---------------------------------------------------------------------------


class TestGroupIdMapping:
    def test_user_id_maps_to_group_id(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)
        assert mg._get_group_id({"user_id": "alice"}) == "alice"

    def test_default_group_id(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)
        assert mg._get_group_id({}) == "default"

    def test_project_id_maps_to_double_hyphen(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)
        assert mg._get_group_id({"project_id": "neuralscape"}) == "project--neuralscape"

    def test_explicit_group_id_takes_precedence(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)
        assert mg._get_group_id({"group_id": "custom-id", "project_id": "ignored"}) == "custom-id"

    def test_scope_without_project_returns_global(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)
        assert mg._get_group_id({"scope": "global"}) == "global"

    def test_get_group_ids_with_project(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)
        result = mg._get_group_ids({"project_id": "neuralscape"})
        assert result == ["global", "project--neuralscape"]

    def test_get_group_ids_explicit_override(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)
        result = mg._get_group_ids({"group_ids": ["a", "b"]})
        assert result == ["a", "b"]

    def test_get_group_ids_no_project(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)
        result = mg._get_group_ids({"user_id": "alice"})
        assert result == ["alice"]


# ---------------------------------------------------------------------------
# Source description builder
# ---------------------------------------------------------------------------


class TestBuildSourceDescription:
    def test_includes_user_agent_run(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)
        desc = mg._build_source_description({
            "user_id": "alice",
            "agent_id": "bot1",
            "run_id": "r42",
        })
        assert "user: alice" in desc
        assert "agent: bot1" in desc
        assert "run: r42" in desc

    def test_empty_filters_returns_mem0(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)
        assert mg._build_source_description({}) == "mem0"

    def test_partial_filters(self, mock_graphiti_imports):
        from mem0.memory.graphiti_memory import MemoryGraph

        config = _make_config()
        mg = MemoryGraph(config)
        desc = mg._build_source_description({"user_id": "bob"})
        assert desc == "user: bob"
        assert "agent" not in desc
