"""Tests for oracle tree-sitter extraction."""

import pytest
from pathlib import Path

from icebench.trackq.oracle import TreeSitterOracle, TREE_SITTER_AVAILABLE


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
def test_oracle_extracts_functions():
    """Test that oracle extracts function definitions."""
    oracle = TreeSitterOracle(str(FIXTURES_DIR), "python")
    oracle.index()

    # Should find helper_function
    assert "helper_function" in oracle.symbols
    sym = oracle.symbols["helper_function"]
    assert sym.file == "sample.py"
    assert sym.kind == "function"
    assert sym.line > 0

    # Should find main
    assert "main" in oracle.symbols


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
def test_oracle_extracts_classes():
    """Test that oracle extracts class definitions."""
    oracle = TreeSitterOracle(str(FIXTURES_DIR), "python")
    oracle.index()

    # Should find Calculator class
    assert "Calculator" in oracle.symbols
    sym = oracle.symbols["Calculator"]
    assert sym.file == "sample.py"
    assert sym.kind == "class"


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
def test_oracle_extracts_methods():
    """Test that oracle extracts method definitions."""
    oracle = TreeSitterOracle(str(FIXTURES_DIR), "python")
    oracle.index()

    # Should find Calculator.add
    assert "Calculator.add" in oracle.symbols
    sym = oracle.symbols["Calculator.add"]
    assert sym.file == "sample.py"
    assert sym.kind == "method"

    # Should find Calculator.multiply
    assert "Calculator.multiply" in oracle.symbols


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
def test_oracle_extracts_calls():
    """Test that oracle extracts function calls."""
    oracle = TreeSitterOracle(str(FIXTURES_DIR), "python")
    oracle.index()

    # Calculator.multiply should call helper_function
    callers = oracle.get_callers("helper_function")
    assert "Calculator.multiply" in callers


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
def test_oracle_symbol_lookup():
    """Test symbol lookup."""
    oracle = TreeSitterOracle(str(FIXTURES_DIR), "python")
    oracle.index()

    # Lookup existing symbol
    sym = oracle.get_symbol_location("helper_function")
    assert sym is not None
    assert sym.name == "helper_function"
    assert sym.file == "sample.py"

    # Lookup non-existent symbol
    sym = oracle.get_symbol_location("nonexistent")
    assert sym is None


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
def test_oracle_get_callers():
    """Test getting callers of a symbol."""
    oracle = TreeSitterOracle(str(FIXTURES_DIR), "python")
    oracle.index()

    # helper_function has callers
    callers = oracle.get_callers("helper_function")
    assert len(callers) > 0
    assert "Calculator.multiply" in callers

    # process_data likely has no callers in fixture
    callers = oracle.get_callers("process_data")
    assert isinstance(callers, list)


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
def test_oracle_find_paths():
    """Test finding paths between symbols."""
    oracle = TreeSitterOracle(str(FIXTURES_DIR), "python")
    oracle.index()

    # Should find path from Calculator.multiply to helper_function (1 hop)
    paths = oracle.find_paths("Calculator.multiply", "helper_function", max_depth=4)

    # At least one path should exist
    assert isinstance(paths, list)
