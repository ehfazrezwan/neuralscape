"""Unit tests for code_impact delegation and tool/REST surface (E7)."""

from unittest.mock import MagicMock, patch

import pytest


def test_code_impact_delegation():
    """Test code_impact delegation calls engine.blast_radius with correct args."""
    from adapters.code_graph.query import code_impact

    # Mock engine
    mock_engine = MagicMock()
    mock_engine.blast_radius.return_value = "Blast radius from 'foo.bar' (max_hops=4): 3 symbols\n  main.py:10 [function] foo.bar\n  util.py:20 [function] foo.baz"

    # Mock get_engine to return our mock
    with patch("adapters.code_graph.query.get_engine", return_value=mock_engine):
        result = code_impact(
            "foo.bar",
            user_id="test-user",
            settings=MagicMock(),
            graph_id="repo:test-repo",
            max_hops=4,
        )

    assert "Blast radius from 'foo.bar'" in result
    assert "3 symbols" in result
    mock_engine.blast_radius.assert_called_once_with("foo.bar", max_hops=4)


def test_code_impact_clamps_max_hops():
    """Test code_impact clamps max_hops to [1, 16]."""
    from adapters.code_graph.query import code_impact

    mock_engine = MagicMock()
    mock_engine.blast_radius.return_value = "test"

    with patch("adapters.code_graph.query.get_engine", return_value=mock_engine):
        # Test lower bound
        code_impact("foo", user_id="u", settings=MagicMock(), max_hops=0)
        mock_engine.blast_radius.assert_called_with("foo", max_hops=1)

        # Test upper bound
        code_impact("foo", user_id="u", settings=MagicMock(), max_hops=99)
        mock_engine.blast_radius.assert_called_with("foo", max_hops=16)


@pytest.mark.asyncio
async def test_code_impact_mcp_tool():
    """Test code_impact MCP tool calls delegation correctly."""
    from mcp_server import call_tool

    mock_service = MagicMock()
    mock_service.search.return_value = []

    with patch("mcp_server._service", mock_service):
        with patch("adapters.code_graph.code_graph_available", return_value=True):
            with patch("adapters.code_graph.query.code_impact", return_value="test result") as mock_impact:
                with patch("adapters.code_graph.query.get_engine"):
                    result = await call_tool(
                        "code_impact",
                        {"symbol": "foo.bar", "max_hops": 3, "graph_id": "repo:test"},
                    )

    assert len(result) == 1
    assert result[0].text == "test result"
    mock_impact.assert_called_once()


@pytest.mark.asyncio
async def test_code_impact_mcp_tool_unavailable_without_extra():
    """Test code_impact MCP tool returns error when code-graph extra is absent."""
    from mcp_server import call_tool

    with patch("adapters.code_graph.code_graph_available", return_value=False):
        result = await call_tool(
            "code_impact",
            {"symbol": "foo.bar"},
        )

    assert len(result) == 1
    import json
    error_data = json.loads(result[0].text)
    assert "error" in error_data
    assert "code-graph" in error_data["error"].lower()


# ── NativeEngine.blast_radius: real symbol resolution + colon span ──


def _blast_radius_cypher_router(mock_details_span="20:25"):
    """Build a _run_cypher side_effect that routes by cypher content.

    Simulates: _find_symbol (epicenter resolution), _get_blast_neighbors (BFS),
    _get_symbol_details (per-affected details with a colon-formatted span).
    """
    def _router(cypher, **params):
        # _find_symbol: fuzzy FQN substring resolver
        if "CONTAINS toLower($label)" in cypher:
            return [{
                "fqn": "app.service.login",
                "kind": "function",
                "file": "app/service.py",
                "span": "10:15",
            }]
        # _get_blast_neighbors: CALLS/IMPORTS traversal
        if "type(r) IN ['CALLS', 'IMPORTS']" in cypher:
            fqn = params.get("fqn", "")
            if fqn == "app.service.login":
                return [{"neighbor": "app.api.login_route"}]
            return []
        # _get_symbol_details: exact-fqn lookup (RETURN ... s.span AS span)
        if "RETURN s.fqn AS fqn, s.file AS file, s.kind AS kind, s.span AS span" in cypher:
            fqn = params.get("fqn", "")
            details = {
                "app.service.login": {
                    "fqn": "app.service.login", "file": "app/service.py",
                    "kind": "function", "span": "10:15",
                },
                "app.api.login_route": {
                    "fqn": "app.api.login_route", "file": "app/api.py",
                    "kind": "function", "span": mock_details_span,
                },
            }
            return [details[fqn]] if fqn in details else []
        return []
    return _router


def test_blast_radius_resolves_symbol_not_char_by_char():
    """blast_radius resolves the input symbol via _find_symbol (fuzzy FQN match),
    NOT by iterating characters through _search_symbols."""
    from adapters.code_graph.native_engine import NativeEngine

    engine = NativeEngine(
        repo_path="/tmp/repo",
        code_space="code--u--repo",
        bridge=MagicMock(),
        settings=MagicMock(),
    )

    with patch.object(engine, "_run_cypher", side_effect=_blast_radius_cypher_router()):
        # Guard: _search_symbols (the WRONG resolver — expects a keyword list)
        # must NOT be used for epicenter resolution.
        with patch.object(engine, "_search_symbols", side_effect=AssertionError("must use _find_symbol")):
            out = engine.blast_radius("login", max_hops=4)

    # Epicenter resolved to the full FQN
    assert "app.service.login" in out
    # Transitively-affected caller present
    assert "app.api.login_route" in out


def test_blast_radius_parses_colon_span_for_file_line():
    """blast_radius output must show the real line, parsed from the colon-format
    span '<start>:<end>' the native indexer stores (not 0 from a '-' split)."""
    from adapters.code_graph.native_engine import NativeEngine

    engine = NativeEngine(
        repo_path="/tmp/repo",
        code_space="code--u--repo",
        bridge=MagicMock(),
        settings=MagicMock(),
    )

    with patch.object(engine, "_run_cypher", side_effect=_blast_radius_cypher_router(mock_details_span="42:60")):
        out = engine.blast_radius("login", max_hops=4)

    # Colon-parsed start line appears (10 for epicenter, 42 for the caller).
    assert "app/service.py:10" in out
    assert "app/api.py:42" in out
    # The old '-' split bug would have produced ":0" everywhere.
    assert ":0 " not in out


def test_blast_radius_not_found():
    """blast_radius returns a clean not-found message when the symbol is absent."""
    from adapters.code_graph.native_engine import NativeEngine

    engine = NativeEngine(
        repo_path="/tmp/repo",
        code_space="code--u--repo",
        bridge=MagicMock(),
        settings=MagicMock(),
    )

    with patch.object(engine, "_run_cypher", return_value=[]):
        out = engine.blast_radius("nonexistent", max_hops=4)

    assert "No symbol matching 'nonexistent'" in out


# ── code_impact on GraphifyJsonEngine → EngineCapabilityError (→ 501) ──


def test_code_impact_graphify_raises_capability_error():
    """code_impact on a graph.json engine (no blast_radius) raises
    EngineCapabilityError, not a bare AttributeError → HTTP 500."""
    from adapters.code_graph.engine import EngineCapabilityError
    from adapters.code_graph.query import code_impact

    # GraphifyJsonEngine has no blast_radius attribute at all.
    class _FakeGraphifyEngine:
        def query(self, *a, **k):
            return "ok"

    with patch("adapters.code_graph.query.get_engine", return_value=_FakeGraphifyEngine()):
        with pytest.raises(EngineCapabilityError, match="native"):
            code_impact("foo", user_id="u", settings=MagicMock(), graph_id="some.json")


@pytest.mark.asyncio
async def test_code_impact_mcp_graphify_returns_clean_error():
    """code_impact MCP tool on a graph.json engine returns clean error JSON
    (not a raw crash) when blast_radius is unavailable."""
    import json

    from adapters.code_graph.engine import EngineCapabilityError
    from mcp_server import call_tool

    with patch("adapters.code_graph.code_graph_available", return_value=True):
        with patch(
            "adapters.code_graph.query.code_impact",
            side_effect=EngineCapabilityError("code_impact/blast_radius requires the native code-intel engine"),
        ):
            result = await call_tool(
                "code_impact",
                {"symbol": "foo", "graph_id": "some.json"},
            )

    assert len(result) == 1
    error_data = json.loads(result[0].text)
    assert "error" in error_data
    assert "native" in error_data["error"].lower()
