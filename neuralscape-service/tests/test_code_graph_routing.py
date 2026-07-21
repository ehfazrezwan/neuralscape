"""Tests for E2 engine routing (GraphifyJsonEngine vs NativeEngine)."""

from unittest.mock import Mock, patch

import pytest

from adapters.code_graph.engine import EngineCapabilityError
from adapters.code_graph.query import CodeGraphError, _get_native_engine, get_engine


def test_get_engine_json_artifact(tmp_path):
    """Test that .json artifacts route to GraphifyJsonEngine."""
    # Create a minimal graph.json
    graph_json = tmp_path / "graph.json"
    graph_json.write_text('{"nodes": [], "edges": []}')

    mock_settings = Mock()
    mock_settings.code_graph_json_path = str(graph_json)

    with patch("adapters.code_graph.query.load_code_graph") as mock_load:
        import networkx as nx
        mock_load.return_value = nx.Graph()

        engine = get_engine(None, "user123", mock_settings)

        # Should be GraphifyJsonEngine
        from adapters.code_graph.graphify_engine import GraphifyJsonEngine
        assert isinstance(engine, GraphifyJsonEngine)


def test_get_engine_repo_ref_no_config():
    """Test that repo: refs raise when CODE_REPOS not configured."""
    mock_settings = Mock()
    mock_settings.code_repos = {}

    with pytest.raises(CodeGraphError, match="No code_repos configured"):
        get_engine("repo:myrepo", "user123", mock_settings)


def test_get_engine_repo_ref_unknown_repo():
    """Test that unknown repo names raise."""
    mock_settings = Mock()
    mock_settings.code_repos = {"knownrepo": "/path/to/known"}

    with pytest.raises(CodeGraphError, match="No repo configured with name"):
        get_engine("repo:unknownrepo", "user123", mock_settings)


def test_get_engine_repo_ref_success(tmp_path):
    """Test that repo: refs route to NativeEngine."""
    # Create a fake repo directory
    repo_path = tmp_path / "myrepo"
    repo_path.mkdir()

    mock_settings = Mock()
    mock_settings.code_repos = {"myrepo": str(repo_path)}

    # Mock the MemoryService bridge
    with patch("memory_service.get_shared_service") as mock_svc:
        import asyncio
        mock_bridge = Mock()
        mock_bridge._loop = asyncio.new_event_loop()
        mock_bridge.driver = Mock()

        mock_service = Mock()
        mock_service._bridge = mock_bridge
        mock_service._get_memory = Mock()
        mock_svc.return_value = mock_service

        engine = get_engine("repo:myrepo", "user123", mock_settings)

        # Should be NativeEngine
        from adapters.code_graph.native_engine import NativeEngine
        assert isinstance(engine, NativeEngine)
        assert engine.code_space == "code--user123--myrepo"


def test_get_native_engine_caching(tmp_path):
    """Test that NativeEngine instances are cached."""
    repo_path = tmp_path / "myrepo"
    repo_path.mkdir()

    mock_settings = Mock()
    mock_settings.code_repos = {"myrepo": str(repo_path)}

    with patch("memory_service.get_shared_service") as mock_svc:
        import asyncio
        mock_bridge = Mock()
        mock_bridge._loop = asyncio.new_event_loop()
        mock_bridge.driver = Mock()

        mock_service = Mock()
        mock_service._bridge = mock_bridge
        mock_service._get_memory = Mock()
        mock_svc.return_value = mock_service

        # First call creates the engine
        engine1 = _get_native_engine("myrepo", "user123", mock_settings)

        # Second call should return the cached instance
        engine2 = _get_native_engine("myrepo", "user123", mock_settings)

        assert engine1 is engine2  # same instance


def test_get_native_engine_bridge_not_initialized(tmp_path):
    """Test that NativeEngine raises if bridge is None."""
    # Clear the cache to ensure this test runs fresh
    from adapters.code_graph import query
    query._ctx_cache.clear()

    repo_path = tmp_path / "myrepo2"  # different name to avoid cache collision
    repo_path.mkdir()

    mock_settings = Mock()
    mock_settings.code_repos = {"myrepo2": str(repo_path)}

    with patch("memory_service.get_shared_service") as mock_svc:
        mock_service = Mock()
        mock_service._bridge = None  # bridge not initialized
        mock_service._get_memory = Mock()
        mock_svc.return_value = mock_service

        with pytest.raises(CodeGraphError, match="Graphiti bridge not initialized"):
            _get_native_engine("myrepo2", "user123", mock_settings)
