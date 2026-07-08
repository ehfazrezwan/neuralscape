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
