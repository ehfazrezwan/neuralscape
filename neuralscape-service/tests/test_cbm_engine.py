"""Tests for CBMEngine (Phase C).

Mocks the CBM bridge HTTP calls (POST tool calls + GET health/status) to test
engine behavior without a running bridge. Ground-truth response shapes verified
against CBM 0.9.0:
  - search_graph → {"total", "results": [{"name", "qualified_name", "label",
                    "file_path", ...}]}    (FQN is `qualified_name`)
  - trace_path   → {"function", "direction",
                    "callees": [{"name","qualified_name","hop"}],
                    "callers": [...]}       (NOT a list of paths)
  - index_status → {"projects", "cache_dir", "cbm_version"}   (GET)
  - health       → {"status": "ok", "cbm_version": ...}       (GET)
"""

import pytest
from unittest.mock import Mock, patch
from adapters.code_graph.cbm_engine import CBMEngine
from adapters.code_graph.engine import EngineCapabilityError


def _resp(payload: dict) -> Mock:
    r = Mock()
    r.status_code = 200
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


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
        raw = "data-ice-corpora-small-py-8a4ce8.src.click.core.CommandCollection"
        canonical = CBMEngine.to_canonical(raw)
        assert canonical == "click.core.CommandCollection"

    def test_to_canonical_without_cache_prefix(self):
        """to_canonical also works when no cache prefix."""
        raw = "src.click.core.CommandCollection"
        assert CBMEngine.to_canonical(raw) == "click.core.CommandCollection"

    def test_from_canonical(self):
        """from_canonical returns canonical (search-friendly pattern)."""
        assert CBMEngine.from_canonical("click.core.Group") == "click.core.Group"

    def test_query_no_project_raises(self):
        """query() raises if no project is set."""
        engine = CBMEngine(bridge_url="http://localhost:8200")
        with pytest.raises(RuntimeError, match="No CBM project set"):
            engine.query("test")

    @patch("adapters.code_graph.cbm_engine.httpx.Client.post")
    def test_query_success_uses_qualified_name(self, mock_post):
        """query() reads FQN from qualified_name (not name) and canonicalizes."""
        mock_post.return_value = _resp({
            "total": 1,
            "results": [
                {
                    "name": "Group",  # short name — must NOT be used for FQN
                    "qualified_name": "data-ice-corpora-small-py-8a4ce8.src.click.core.Group",
                    "label": "class",
                    "file_path": "src/click/core.py",
                    "line": 100,
                }
            ],
        })
        engine = CBMEngine(bridge_url="http://localhost:8200", project="test-project")
        result = engine.query("Group")
        assert "click.core.Group" in result
        assert "src/click/core.py:100" in result

    @patch("adapters.code_graph.cbm_engine.httpx.Client.post")
    def test_neighbors_parses_callees_and_callers(self, mock_post):
        """neighbors() parses the real callees/callers dict shape and canonicalizes."""
        mock_post.return_value = _resp({
            "function": "echo",
            "direction": "both",
            "callees": [
                {"name": "write",
                 "qualified_name": "data-ice-x-8a4.src.click.formatting.HelpFormatter.write",
                 "hop": 1},
            ],
            "callers": [
                {"name": "ship_new",
                 "qualified_name": "data-ice-x-8a4.examples.naval.naval.ship_new",
                 "hop": 1},
            ],
        })
        engine = CBMEngine(bridge_url="http://localhost:8200", project="test-project")
        result = engine.neighbors("echo")
        # Outgoing callee (canonicalized, src stripped)
        assert "--> click.formatting.HelpFormatter.write [CALLS]" in result
        # Incoming caller (canonicalized; examples is NOT a stripped root)
        assert "<-- examples.naval.naval.ship_new [CALLS]" in result

    @patch("adapters.code_graph.cbm_engine.httpx.Client.post")
    def test_neighbors_ignores_relation_filter_never_empty(self, mock_post):
        """neighbors() with a relation_filter must NOT silently drop everything.

        CBM's trace_path carries no edge-type labels, so the filter is ignored
        (all neighbors returned) rather than emptying the answer.
        """
        mock_post.return_value = _resp({
            "function": "echo",
            "direction": "both",
            "callees": [{"name": "write", "qualified_name": "src.click.core.write", "hop": 1}],
            "callers": [],
        })
        engine = CBMEngine(bridge_url="http://localhost:8200", project="test-project")
        result = engine.neighbors("echo", relation_filter="IMPORTS")
        assert "click.core.write" in result  # filter ignored, neighbor still present
        assert "No neighbors" not in result

    @patch("adapters.code_graph.cbm_engine.httpx.Client.post")
    def test_neighbors_no_results(self, mock_post):
        """neighbors() with empty callees/callers reports no neighbors."""
        mock_post.return_value = _resp({
            "function": "lonely", "direction": "both", "callees": [], "callers": [],
        })
        engine = CBMEngine(bridge_url="http://localhost:8200", project="test-project")
        assert "No neighbors found" in engine.neighbors("lonely")

    @patch("adapters.code_graph.cbm_engine.httpx.Client.post")
    def test_neighbors_degrades_to_empty_on_bridge_500(self, mock_post):
        """R-B: a bridge/CBM failure on a symbol degrades to honest N/A, never a raise.

        Even if an older bridge image still 500s on an untraceable symbol
        (`raise_for_status` → HTTPStatusError → RuntimeError in `_call_bridge`),
        neighbors() must return an empty "No neighbors" answer rather than
        propagating the error to the recall caller (PLAN §3.3: the base always
        answers; a per-symbol failure is not a recall error).
        """
        import httpx

        r = Mock()
        r.status_code = 500
        request = httpx.Request("POST", "http://localhost:8200/trace_path")
        response = httpx.Response(500, request=request, text="CBM tool trace_path failed")
        r.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=request, response=response
        )
        mock_post.return_value = r

        engine = CBMEngine(bridge_url="http://localhost:8200", project="test-project")
        # Must NOT raise — degrades to empty (honest N/A).
        result = engine.neighbors("untraceable_symbol")
        assert "No neighbors found for 'untraceable_symbol'" in result

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

    @patch("adapters.code_graph.cbm_engine.httpx.Client.get")
    @patch("adapters.code_graph.cbm_engine.httpx.Client.post")
    def test_index_success_stamps_version_via_get(self, mock_post, mock_get):
        """index() returns IndexReport AND stamps the CBM version via GET /index_status.

        Verifies the version actually comes through (not the cbm@unknown fallback),
        which only works if index_status is fetched with GET (POST → 405).
        """
        mock_post.return_value = _resp({
            "project": "test-project", "nodes": 100, "edges": 50, "status": "indexed",
        })
        mock_get.return_value = _resp({
            "projects": [], "cache_dir": "/data/cbm_cache", "cbm_version": "cbm@0.9.0",
        })

        engine = CBMEngine(bridge_url="http://localhost:8200")
        report = engine.index("/path/to/repo")

        assert engine.project == "test-project"
        assert report.symbols_indexed == 100
        assert report.edges_indexed == 50
        # The version stamp must be the REAL version, proving the GET path works.
        assert report.system_version == "cbm@0.9.0"
        mock_get.assert_called_once()  # index_status fetched via GET
        # And it hit the index_status endpoint, not something else.
        assert "/index_status" in mock_get.call_args[0][0]

    @patch("adapters.code_graph.cbm_engine.httpx.Client.get")
    def test_health_ok(self, mock_get):
        """health() returns True when the bridge reports status ok (GET /health)."""
        mock_get.return_value = _resp({"status": "ok", "cbm_version": "cbm@0.9.0"})
        engine = CBMEngine(bridge_url="http://localhost:8200")
        assert engine.health() is True
        assert "/health" in mock_get.call_args[0][0]

    @patch("adapters.code_graph.cbm_engine.httpx.Client.get")
    def test_health_down_bridge_returns_false(self, mock_get):
        """health() returns False (never raises) when the bridge is unreachable."""
        import httpx
        mock_get.side_effect = httpx.ConnectError("connection refused")
        engine = CBMEngine(bridge_url="http://localhost:8200")
        assert engine.health() is False

    @patch("adapters.code_graph.cbm_engine.httpx.Client.post")
    def test_locate_success_uses_qualified_name(self, mock_post):
        """locate() reads qualified_name, canonicalizes, builds the anchor key."""
        mock_post.return_value = _resp({
            "total": 1,
            "results": [
                {
                    "name": "Group",
                    "qualified_name": "data-ice-corpora-small-py-8a4ce8.src.click.core.Group",
                    "label": "class",
                    "file_path": "src/click/core.py",
                    "line": 100,
                }
            ],
        })
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
        assert hits[0].anchor_id == "myrepo::click.core.Group"


class TestCBMSystemHealthEligibility:
    """A DOWN CBM bridge makes the code-cbm KnowledgeSystem ineligible (PLAN §3.3)."""

    def test_down_bridge_excluded_from_eligible_systems(self):
        """CBM bridge unreachable → system health != ok → excluded by eligible_systems."""
        from knowledge.code_system import CodeKnowledgeSystem
        from knowledge.registry import register_system, eligible_systems, KNOWLEDGE_REGISTRY

        # Engine whose health probe reports down (bridge unreachable).
        down_engine = Mock()
        down_engine.health.return_value = False

        system = CodeKnowledgeSystem(
            name="code-cbm-test-down",
            engine=down_engine,
            capabilities=frozenset({"query", "neighbors", "locate", "index"}),
            transport="http",
        )
        # health() must report unreachable (uses the engine probe).
        assert system.health().status == "unreachable"

        register_system(system)
        try:
            eligible = eligible_systems(operation="query")
            names = {s.info.name for s in eligible}
            assert "code-cbm-test-down" not in names  # down system excluded
        finally:
            KNOWLEDGE_REGISTRY.pop("code-cbm-test-down", None)

    def test_up_bridge_included_in_eligible_systems(self):
        """CBM bridge reachable → system health ok → eligible for its capabilities."""
        from knowledge.code_system import CodeKnowledgeSystem
        from knowledge.registry import register_system, eligible_systems, KNOWLEDGE_REGISTRY

        up_engine = Mock()
        up_engine.health.return_value = True

        system = CodeKnowledgeSystem(
            name="code-cbm-test-up",
            engine=up_engine,
            capabilities=frozenset({"query", "neighbors", "locate", "index"}),
            transport="http",
        )
        assert system.health().status == "ok"

        register_system(system)
        try:
            eligible = eligible_systems(operation="query")
            names = {s.info.name for s in eligible}
            assert "code-cbm-test-up" in names
        finally:
            KNOWLEDGE_REGISTRY.pop("code-cbm-test-up", None)
