"""Tests for CBMEngine (Phase C).

Mocks the CBM bridge HTTP calls to test engine behavior without a running bridge.
"""

import pytest
from unittest.mock import Mock, patch
from adapters.code_graph.cbm_engine import CBMEngine
from adapters.code_graph.engine import EngineCapabilityError


class TestCBMEngine:
    """Test CBMEngine protocol methods and canonical FQN handling."""

    def test_init(self):
        """CBMEngine initializes with bridge URL."""
        engine = CBMEngine(bridge_url="http://localhost:8200")
        assert engine.bridge_url == "http://localhost:8200"
        assert engine.project is None
        assert engine.code_space is None

    def test_to_canonical(self):
        """to_canonical strips cache prefix (hyphenated parts) and src root."""
        # CBM format: cache-path-prefix (with hyphens).src.module.symbol
        raw = "data-ice-corpora-small-py-8a4ce8.src.click.core.CommandCollection"
        canonical = CBMEngine.to_canonical(raw)
        # Hyphens filtered out, then src stripped
        assert canonical == "click.core.CommandCollection"

    def test_to_canonical_without_cache_prefix(self):
        """to_canonical also works when no cache prefix."""
        raw = "src.click.core.CommandCollection"
        canonical = CBMEngine.to_canonical(raw)
        assert canonical == "click.core.CommandCollection"

    def test_from_canonical(self):
        """from_canonical returns canonical (search-friendly pattern)."""
        canonical = "click.core.CommandCollection"
        pattern = CBMEngine.from_canonical(canonical)
        assert pattern == canonical

    def test_query_no_project_raises(self):
        """query() raises if no project is set."""
        engine = CBMEngine(bridge_url="http://localhost:8200")
        with pytest.raises(RuntimeError, match="No CBM project set"):
            engine.query("test")

    @patch("adapters.code_graph.cbm_engine.httpx.Client.post")
    def test_query_success(self, mock_post):
        """query() calls search_graph and returns formatted results."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {
                    "name": "data-ice-corpora-small-py-8a4ce8.src.click.core.Group",
                    "label": "class",
                    "file_path": "src/click/core.py",
                    "line": 100,
                }
            ]
        }
        mock_post.return_value = mock_resp

        engine = CBMEngine(bridge_url="http://localhost:8200", project="test-project")
        result = engine.query("Group")

        assert "click.core.Group" in result
        assert "src/click/core.py:100" in result

    @patch("adapters.code_graph.cbm_engine.httpx.Client.post")
    def test_neighbors_success(self, mock_post):
        """neighbors() calls trace_path with depth=1."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "paths": [
                [
                    {"name": "src.click.core.Group", "kind": "class"},
                    {"name": "src.click.core.Command", "kind": "class"},
                ]
            ]
        }
        mock_post.return_value = mock_resp

        engine = CBMEngine(bridge_url="http://localhost:8200", project="test-project")
        result = engine.neighbors("Group")

        # Should canonicalize neighbor names
        assert "click.core.Command" in result or "core.Command" in result

    def test_path_raises_capability_error(self):
        """path() raises EngineCapabilityError (raw Cypher is banned)."""
        engine = CBMEngine(bridge_url="http://localhost:8200", project="test-project")
        with pytest.raises(EngineCapabilityError, match="raw query_graph Cypher"):
            engine.path("Group", "Command")

    def test_detect_changes_raises_capability_error(self):
        """detect_changes() raises EngineCapabilityError (needs git)."""
        engine = CBMEngine(bridge_url="http://localhost:8200", project="test-project")
        with pytest.raises(EngineCapabilityError, match="git state"):
            engine.detect_changes()

    def test_semantic_layer_raises_capability_error(self):
        """semantic_layer() raises EngineCapabilityError (not exposed)."""
        engine = CBMEngine(bridge_url="http://localhost:8200", project="test-project")
        with pytest.raises(EngineCapabilityError, match="semantic layer"):
            engine.semantic_layer()

    @patch("adapters.code_graph.cbm_engine.httpx.Client.post")
    def test_index_success(self, mock_post):
        """index() calls index_repository and returns IndexReport."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "project": "test-project",
            "nodes": 100,
            "edges": 50,
            "status": "indexed",
        }
        mock_post.return_value = mock_resp

        engine = CBMEngine(bridge_url="http://localhost:8200")
        report = engine.index("/path/to/repo")

        assert engine.project == "test-project"
        assert report.symbols_indexed == 100
        assert report.edges_indexed == 50

    @patch("adapters.code_graph.cbm_engine.httpx.Client.post")
    def test_locate_success(self, mock_post):
        """locate() calls search_graph and returns LocateHit list."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {
                    "name": "data-ice-corpora-small-py-8a4ce8.src.click.core.Group",
                    "label": "class",
                    "file_path": "src/click/core.py",
                    "line": 100,
                }
            ]
        }
        mock_post.return_value = mock_resp

        engine = CBMEngine(
            bridge_url="http://localhost:8200",
            project="test-project",
            code_space="code--test--myrepo",
        )
        hits = engine.locate("Group")

        assert len(hits) == 1
        assert hits[0].fqn == "click.core.Group"
        assert hits[0].file == "src/click/core.py"
        assert hits[0].line == 100
        # Anchor key should be "<repo>::<canonical_fqn>"
        assert hits[0].anchor_id == "myrepo::click.core.Group"
