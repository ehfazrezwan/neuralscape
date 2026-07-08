"""Tests for query generation."""

import pytest
from pathlib import Path

from icebench.adapters.base import Corpus
from icebench.trackq.generate import (
    generate_queries,
    _extract_python_docstrings,
    _clean_docstring,
)
from icebench.trackq.oracle import TREE_SITTER_AVAILABLE


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_corpus():
    """Create a test corpus from fixtures."""
    return Corpus(
        name="test-fixture",
        path=str(FIXTURES_DIR),
        repo_sha="test123",
        language="python",
        loc=100,
        file_count=2,
    )


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
def test_generate_symbol_lookup(fixture_corpus):
    """Test symbol_lookup query generation."""
    queries = generate_queries("symbol_lookup", fixture_corpus, n=10, seed=42)

    assert len(queries) > 0
    for q in queries:
        assert q.op == "symbol_lookup"
        assert "symbol" in q.payload
        assert "corpus" in q.payload
        assert "file" in q.gold
        assert "symbol" in q.gold


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
def test_generate_neighbors_1hop(fixture_corpus):
    """Test neighbors_1hop query generation."""
    queries = generate_queries("neighbors_1hop", fixture_corpus, n=10, seed=42)

    # May have fewer than 10 if not enough symbols have callers
    assert isinstance(queries, list)
    for q in queries:
        assert q.op == "neighbors_1hop"
        assert "symbol" in q.payload
        assert "callers" in q.gold


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
def test_generate_path_le4(fixture_corpus):
    """Test path_le4 query generation."""
    queries = generate_queries("path_le4", fixture_corpus, n=5, seed=42)

    # May have fewer than 5 if not enough valid paths
    assert isinstance(queries, list)
    for q in queries:
        assert q.op == "path_le4"
        assert "from" in q.payload
        assert "to" in q.payload
        assert "paths" in q.gold


def test_generate_nl_locate(fixture_corpus):
    """Test nl_locate query generation."""
    queries = generate_queries("nl_locate", fixture_corpus, n=10, seed=42)

    assert len(queries) > 0
    for q in queries:
        assert q.op == "nl_locate"
        assert "query" in q.payload  # The docstring
        assert "file" in q.gold
        assert "symbol" in q.gold


def test_deterministic_query_generation(fixture_corpus):
    """Test that query generation is deterministic with same seed."""
    queries1 = generate_queries("nl_locate", fixture_corpus, n=5, seed=42)
    queries2 = generate_queries("nl_locate", fixture_corpus, n=5, seed=42)

    assert len(queries1) == len(queries2)

    # Compare payloads (queries should be identical)
    for q1, q2 in zip(queries1, queries2):
        assert q1.payload["query"] == q2.payload["query"]
        assert q1.gold["file"] == q2.gold["file"]
        assert q1.gold["symbol"] == q2.gold["symbol"]


def test_extract_python_docstrings():
    """Test Python docstring extraction."""
    sample_file = FIXTURES_DIR / "sample.py"
    docstrings = _extract_python_docstrings(sample_file)

    assert len(docstrings) > 0

    # Should extract from functions with docstrings
    func_names = {name for name, _ in docstrings}
    assert "helper_function" in func_names
    assert "main" in func_names


def test_clean_docstring():
    """Test docstring cleaning."""
    # Multi-line docstring: should take first line/sentence
    doc = """Process some data.

    This function does important processing.
    """
    cleaned = _clean_docstring(doc)
    assert "Process some data" in cleaned
    assert len(cleaned) < len(doc)

    # Single sentence
    doc = "Add two numbers."
    cleaned = _clean_docstring(doc)
    assert cleaned == "Add two numbers"

    # Multiple sentences: should take first
    doc = "Add two numbers. Returns the sum."
    cleaned = _clean_docstring(doc)
    assert cleaned == "Add two numbers"
