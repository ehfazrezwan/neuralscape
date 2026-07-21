"""Unit tests for E3 locate functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import uuid

import pytest

from adapters.code_graph.engine import EngineCapabilityError, LocateHit
from adapters.code_graph.native_engine import NativeEngine


@pytest.fixture
def mock_bridge():
    """Mock Graphiti bridge with a fake Neo4j driver."""
    import asyncio

    bridge = Mock()
    bridge._loop = asyncio.new_event_loop()

    # Mock driver that returns empty results
    async def mock_run(cypher, **params):
        result = Mock()
        result.data = Mock(return_value=[])
        result.single = Mock(return_value=None)
        return result

    mock_session = Mock()
    mock_session.run = mock_run
    mock_session.__aenter__ = Mock(return_value=mock_session)
    mock_session.__aexit__ = Mock(return_value=None)

    mock_driver = Mock()
    mock_driver.session = Mock(return_value=mock_session)
    bridge.driver = mock_driver

    return bridge


@pytest.fixture
def mock_settings():
    """Mock settings object."""
    settings = Mock()
    settings.code_graph_extracted_confidence = 0.9
    return settings


@pytest.fixture
def mock_memory_service():
    """Mock MemoryService with embedding model and Qdrant client."""
    service = Mock()

    # Mock memory with embedding model
    memory = Mock()
    embedding_model = Mock()
    # Mock embed to return a vector
    embedding_model.embed = Mock(return_value=[0.1] * 768)
    embedding_model.embed_batch = Mock(return_value=[[0.1] * 768])
    memory.embedding_model = embedding_model

    # Mock vector store with Qdrant client
    vector_store = Mock()
    qdrant_client = Mock()

    # Mock query_points to return hits
    mock_hit = Mock()
    mock_hit.id = str(uuid.uuid4())
    mock_hit.score = 0.9
    mock_hit.payload = {
        "fqn": "test.module.TestFunction",
        "kind": "function",
        "file": "test/module.py",
        "line": 10,
        "signature": "def TestFunction():",
        "docstring": "A test function",
        "degree": 5,
        "anchor_id": None,
    }

    qdrant_client.query_points = Mock(return_value=Mock(points=[mock_hit]))
    qdrant_client.get_collection = Mock(side_effect=Exception("Not found"))
    qdrant_client.create_collection = Mock()
    qdrant_client.upsert = Mock()

    vector_store.client = qdrant_client
    vector_store._has_bm25_slot = False  # Disable BM25 for simplicity
    memory.vector_store = vector_store

    service._get_memory = Mock(return_value=memory)
    return service


def test_locate_basic(mock_bridge, mock_settings, mock_memory_service):
    """Test basic locate functionality with mocked embedding and Qdrant."""
    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Patch get_shared_service to return our mock
    with patch("memory_service.get_shared_service", return_value=mock_memory_service):
        hits = engine.locate("test query", k=10)

    # Should return LocateHit objects
    assert isinstance(hits, list)
    assert len(hits) > 0
    assert all(isinstance(hit, LocateHit) for hit in hits)

    # Check first hit structure
    first_hit = hits[0]
    assert first_hit.fqn == "test.module.TestFunction"
    assert first_hit.kind == "function"
    assert first_hit.file == "test/module.py"
    assert first_hit.line == 10
    assert first_hit.signature == "def TestFunction():"
    assert first_hit.docstring == "A test function"
    assert first_hit.score > 0


def test_locate_with_degree_boost(mock_bridge, mock_settings, mock_memory_service):
    """Test that degree boost affects ranking."""
    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Mock two hits with different degrees
    hit1 = Mock()
    hit1.id = str(uuid.uuid4())
    hit1.score = 0.8
    hit1.payload = {
        "fqn": "test.LowDegree",
        "kind": "function",
        "file": "test.py",
        "line": 1,
        "signature": "def LowDegree():",
        "docstring": "",
        "degree": 2,
        "anchor_id": None,
    }

    hit2 = Mock()
    hit2.id = str(uuid.uuid4())
    hit2.score = 0.8
    hit2.payload = {
        "fqn": "test.HighDegree",
        "kind": "function",
        "file": "test.py",
        "line": 10,
        "signature": "def HighDegree():",
        "docstring": "",
        "degree": 20,
        "anchor_id": None,
    }

    # Mock Qdrant to return both hits
    mock_service = mock_memory_service
    mock_service._get_memory().vector_store.client.query_points = Mock(
        return_value=Mock(points=[hit1, hit2])
    )

    with patch("memory_service.get_shared_service", return_value=mock_service):
        hits = engine.locate("test", k=10)

    # High-degree function should rank higher (same base score but higher degree)
    assert len(hits) == 2
    # The high-degree one should be first after degree boost
    assert hits[0].fqn == "test.HighDegree"
    assert hits[1].fqn == "test.LowDegree"


def test_index_symbol_cards(mock_bridge, mock_settings, mock_memory_service):
    """Test that indexing builds and embeds symbol cards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        test_file = repo_path / "test.py"
        test_file.write_text(
            '''
def greet(name):
    """Say hello."""
    return f"Hello, {name}"
'''
        )

        engine = NativeEngine(
            repo_path=str(repo_path),
            code_space="code--user123--testrepo",
            bridge=mock_bridge,
            settings=mock_settings,
        )

        # Mock _run_cypher to return a symbol
        engine._run_cypher = Mock(return_value=[
            {
                "fqn": "test.greet",
                "kind": "function",
                "file": "test.py",
                "span": "2:4",
                "degree": 3,
            }
        ])

        with patch("memory_service.get_shared_service", return_value=mock_memory_service):
            # Call the indexing method
            engine._index_symbol_cards(repo_path)

        # Verify embed_batch was called
        mock_service = mock_memory_service._get_memory()
        mock_service.embedding_model.embed_batch.assert_called_once()

        # Verify Qdrant upsert was called
        mock_service.vector_store.client.upsert.assert_called_once()
        call_args = mock_service.vector_store.client.upsert.call_args
        assert call_args[1]["collection_name"] == "code_index"
        assert len(call_args[1]["points"]) == 1


def test_language_expansion_typescript(mock_bridge, mock_settings):
    """Test TypeScript file detection and parsing."""
    pytest.importorskip("tree_sitter")

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        ts_file = repo_path / "test.ts"
        ts_file.write_text(
            '''
function greet(name: string): string {
    return `Hello, ${name}`;
}

class Greeter {
    greet(name: string): string {
        return greet(name);
    }
}
'''
        )

        engine = NativeEngine(
            repo_path=str(repo_path),
            code_space="code--user123--testrepo",
            bridge=mock_bridge,
            settings=mock_settings,
        )

        # Parse the TypeScript file
        symbols, edges = engine._parse_file(ts_file, repo_path, "typescript")

        # Should extract at least the function and class
        assert len(symbols) >= 2
        fqns = [s.fqn for s in symbols]
        assert any("greet" in fqn for fqn in fqns)
        assert any("Greeter" in fqn for fqn in fqns)


def test_language_expansion_go(mock_bridge, mock_settings):
    """Test Go file detection and parsing."""
    pytest.importorskip("tree_sitter")

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        go_file = repo_path / "test.go"
        go_file.write_text(
            '''
package main

import "fmt"

func Greet(name string) string {
    return fmt.Sprintf("Hello, %s", name)
}

type Greeter struct{}

func (g *Greeter) Greet(name string) string {
    return Greet(name)
}
'''
        )

        engine = NativeEngine(
            repo_path=str(repo_path),
            code_space="code--user123--testrepo",
            bridge=mock_bridge,
            settings=mock_settings,
        )

        # Parse the Go file
        symbols, edges = engine._parse_file(go_file, repo_path, "go")

        # Should extract function and method
        assert len(symbols) >= 1
        fqns = [s.fqn for s in symbols]
        assert any("Greet" in fqn for fqn in fqns)


def test_graphify_json_engine_locate_unsupported():
    """Test that GraphifyJsonEngine.locate raises EngineCapabilityError."""
    pytest.importorskip("graphify")

    from adapters.code_graph.graphify_engine import GraphifyJsonEngine
    import networkx as nx

    # Create empty graph
    G = nx.DiGraph()
    engine = GraphifyJsonEngine(G, "/tmp/test.json")

    with pytest.raises(EngineCapabilityError, match="locate.*native code-intel engine"):
        engine.locate("test query")
