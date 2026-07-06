"""Unit tests for NativeEngine (E2 native code-intel indexer)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from adapters.code_graph.engine import EngineCapabilityError, IndexReport
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
    settings.code_graph_inferred_confidence = 0.7
    settings.code_graph_ambiguous_confidence = 0.5
    settings.code_graph_ambiguous_floor = 0.6
    return settings


@pytest.fixture
def temp_repo():
    """Create a temporary repo with a simple Python file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        test_file = repo_path / "test_module.py"
        test_file.write_text(
            """
def greet(name):
    '''Say hello.'''
    return f"Hello, {name}"

class Greeter:
    def greet(self, name):
        return greet(name)
"""
        )
        yield repo_path


def test_native_engine_init(mock_bridge, mock_settings):
    """Test NativeEngine initialization."""
    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )
    assert engine.code_space == "code--user123--testrepo"
    assert str(engine.repo_path) == "/tmp/test"


def test_native_engine_locate_implemented(mock_bridge, mock_settings):
    """Test that locate() now works in E3 (requires mocked embedding)."""
    from unittest.mock import patch, Mock
    import uuid

    # Mock the memory service
    mock_service = Mock()
    mock_memory = Mock()

    # Mock embedding model
    embedding_model = Mock()
    embedding_model.embed = Mock(return_value=[0.1] * 768)
    mock_memory.embedding_model = embedding_model

    # Mock vector store with Qdrant client
    vector_store = Mock()
    qdrant_client = Mock()

    # Mock empty hits
    qdrant_client.query_points = Mock(return_value=Mock(points=[]))
    qdrant_client.get_collection = Mock(side_effect=Exception("Not found"))
    qdrant_client.create_collection = Mock()

    vector_store.client = qdrant_client
    vector_store._has_bm25_slot = False
    mock_memory.vector_store = vector_store

    mock_service._get_memory = Mock(return_value=mock_memory)

    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Should not raise EngineCapabilityError anymore (E3 implements it)
    with patch("memory_service.get_shared_service", return_value=mock_service):
        hits = engine.locate("test query")

    # Should return empty list (no hits from mocked Qdrant)
    assert isinstance(hits, list)
    assert len(hits) == 0


def test_native_engine_detect_changes_implemented(temp_repo, mock_bridge, mock_settings):
    """Test that detect_changes() works in E5 (persisted vs fresh)."""
    from unittest.mock import patch

    engine = NativeEngine(
        repo_path=str(temp_repo),
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Mock the _fetch_persisted_symbols and _parse_fresh_symbols
    with patch.object(engine, "_fetch_persisted_symbols") as mock_persisted:
        with patch.object(engine, "_parse_fresh_symbols") as mock_fresh:
            mock_persisted.return_value = []
            mock_fresh.return_value = []

            # Should not raise EngineCapabilityError anymore (E5 implements it)
            report = engine.detect_changes()

            # Should return a ChangeReport
            assert hasattr(report, "deleted_symbols")
            assert hasattr(report, "modified_symbols")
            assert hasattr(report, "added_symbols")
            assert hasattr(report, "affected_anchors")
            assert hasattr(report, "summary")


def test_native_engine_semantic_layer_not_implemented(mock_bridge, mock_settings):
    """Test that semantic_layer() raises EngineCapabilityError (Louvain deferred)."""
    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )
    with pytest.raises(EngineCapabilityError, match="semantic_layer.*Louvain"):
        engine.semantic_layer()


def test_native_engine_export_snapshot_not_implemented(mock_bridge, mock_settings):
    """Test that export_snapshot() raises EngineCapabilityError (E6+)."""
    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )
    with pytest.raises(EngineCapabilityError, match="export_snapshot.*E6"):
        engine.export_snapshot()


def test_native_engine_parse_file(temp_repo, mock_bridge, mock_settings):
    """Test parsing a Python file with tree-sitter."""
    pytest.importorskip("tree_sitter_language_pack")

    engine = NativeEngine(
        repo_path=str(temp_repo),
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    test_file = temp_repo / "test_module.py"
    symbols, edges = engine._parse_file(test_file, temp_repo, "python")

    # Should extract at least the function and class
    assert len(symbols) >= 2
    fqns = [s.fqn for s in symbols]
    assert "test_module.greet" in fqns
    assert "test_module.Greeter" in fqns

    # Should have some edges (at least the DEFINES for the method)
    assert len(edges) > 0


def test_native_engine_file_hash(temp_repo, mock_bridge, mock_settings):
    """Test file content hashing."""
    engine = NativeEngine(
        repo_path=str(temp_repo),
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    test_file = temp_repo / "test_module.py"
    hash1 = engine._file_hash(test_file)
    assert len(hash1) == 64  # SHA256 hex digest

    # Same file should have same hash
    hash2 = engine._file_hash(test_file)
    assert hash1 == hash2


def test_extraction_to_epistemic_mapping(mock_bridge, mock_settings):
    """Test confidence → epistemic level mapping."""
    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    assert engine._extraction_to_epistemic("extracted") == "explicit"
    assert engine._extraction_to_epistemic("inferred") == "deductive"
    assert engine._extraction_to_epistemic("ambiguous") == "deductive"
    assert engine._extraction_to_epistemic("unknown") == "deductive"


def test_native_engine_query_no_symbols(mock_bridge, mock_settings):
    """Test query() when no symbols match."""
    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Mock _search_symbols to return empty
    engine._search_symbols = Mock(return_value=[])

    result = engine.query("nonexistent")
    assert "No symbols matching" in result
    assert "code--user123--testrepo" in result


def test_native_engine_neighbors_no_symbol(mock_bridge, mock_settings):
    """Test neighbors() when symbol not found."""
    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Mock _find_symbol to return empty
    engine._find_symbol = Mock(return_value=[])

    result = engine.neighbors("nonexistent")
    assert "No symbol matching" in result


def test_native_engine_path_no_source(mock_bridge, mock_settings):
    """Test path() when source symbol not found."""
    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Mock _find_symbol to return empty for source
    engine._find_symbol = Mock(return_value=[])

    result = engine.path("nonexistent", "target")
    assert "No symbol matching source" in result
