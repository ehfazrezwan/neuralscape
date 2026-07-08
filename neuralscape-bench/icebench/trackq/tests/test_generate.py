"""Tests for query generation."""

import pytest
from pathlib import Path

from icebench.adapters.base import Corpus
from icebench.trackq.generate import (
    generate_queries,
    generate_specs,
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
    """Test symbol_lookup spec generation."""
    specs = generate_specs("symbol_lookup", fixture_corpus, n=10, seed=42)

    assert len(specs) > 0
    for q in specs:
        assert q.op == "symbol_lookup"
        assert "symbol" in q.payload
        assert "corpus" in q.payload
        assert "file" in q.gold
        assert "symbol" in q.gold


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
def test_generate_neighbors_1hop(fixture_corpus):
    """Test neighbors_1hop spec generation."""
    specs = generate_specs("neighbors_1hop", fixture_corpus, n=10, seed=42)

    # May have fewer than 10 if not enough symbols have callers
    assert isinstance(specs, list)
    for q in specs:
        assert q.op == "neighbors_1hop"
        assert "symbol" in q.payload
        assert "callers" in q.gold


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
def test_generate_path_le4(fixture_corpus):
    """Test path_le4 spec generation."""
    specs = generate_specs("path_le4", fixture_corpus, n=5, seed=42)

    # May have fewer than 5 if not enough valid paths
    assert isinstance(specs, list)
    for q in specs:
        assert q.op == "path_le4"
        assert "from" in q.payload
        assert "to" in q.payload
        assert "paths" in q.gold


def test_generate_nl_locate(fixture_corpus):
    """Test nl_locate spec generation."""
    specs = generate_specs("nl_locate", fixture_corpus, n=10, seed=42)

    assert len(specs) > 0
    for q in specs:
        assert q.op == "nl_locate"
        assert "query" in q.payload  # The docstring
        assert "file" in q.gold
        assert "symbol" in q.gold


# ---- Runner-contract tests (issue #1): generate_queries returns dict payloads ----


class _FakeAdapter:
    """Faithful stand-in for a SystemAdapter that records what it receives.

    Mirrors the real adapters' contract: query(op, payload) where payload is a
    plain dict and specific keys are read per op. Asserts the runner-forwarded
    element is a dict and has the keys the real adapters index into.
    """

    _REQUIRED_KEYS = {
        "symbol_lookup": ["symbol"],
        "neighbors_1hop": ["symbol"],
        "path_le4": ["from", "to"],
        "nl_locate": ["query"],
    }

    def __init__(self):
        self.received = []

    def query(self, op: str, payload: dict):
        # This is exactly how the real adapters treat the argument.
        assert isinstance(payload, dict), f"payload must be a dict, got {type(payload)}"
        for key in self._REQUIRED_KEYS.get(op, []):
            assert key in payload, f"payload for {op} missing required key {key!r}"
        # Adapters normalize corpus via payload.get("corpus"); it must be present.
        assert "corpus" in payload
        self.received.append((op, payload))


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
@pytest.mark.parametrize("op", ["symbol_lookup", "neighbors_1hop", "path_le4", "nl_locate"])
def test_generate_queries_returns_dict_payloads(fixture_corpus, op):
    """generate_queries must return plain dicts (runner forwards them verbatim)."""
    queries = generate_queries(op, fixture_corpus, n=5, seed=42)
    assert isinstance(queries, list)
    for payload in queries:
        assert isinstance(payload, dict)


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
def test_runner_call_path_feeds_dict_to_adapter(fixture_corpus):
    """Replicate run.py cmd_query loop and prove adapters receive dict payloads.

    Mirrors:
        queries = generate_queries(op, corpus, n=..., seed=...)
        for i, query_payload in enumerate(queries):
            result = system.query(op, query_payload)
    """
    adapter = _FakeAdapter()
    for op in ("symbol_lookup", "neighbors_1hop", "path_le4", "nl_locate"):
        queries = generate_queries(op, fixture_corpus, n=3, seed=42)
        for i, query_payload in enumerate(queries):
            adapter.query(op, query_payload)  # asserts dict + required keys

    # At least one op should have produced queries on the fixture.
    assert adapter.received


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
def test_generate_queries_payload_matches_specs(fixture_corpus):
    """generate_queries payloads must be exactly the specs' payloads (same order)."""
    specs = generate_specs("symbol_lookup", fixture_corpus, n=5, seed=42)
    payloads = generate_queries("symbol_lookup", fixture_corpus, n=5, seed=42)
    assert [s.payload for s in specs] == payloads


def test_deterministic_query_generation(fixture_corpus):
    """Test that query generation is deterministic with same seed."""
    specs1 = generate_specs("nl_locate", fixture_corpus, n=5, seed=42)
    specs2 = generate_specs("nl_locate", fixture_corpus, n=5, seed=42)

    assert len(specs1) == len(specs2)

    # Compare payloads (queries should be identical)
    for q1, q2 in zip(specs1, specs2):
        assert q1.payload["query"] == q2.payload["query"]
        assert q1.gold["file"] == q2.gold["file"]
        assert q1.gold["symbol"] == q2.gold["symbol"]


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
@pytest.mark.parametrize("op", ["symbol_lookup", "nl_locate"])
def test_prefix_stable_generation(fixture_corpus, op):
    """Generating with a larger n yields a superset (prefix-stable).

    The scorer relies on this to recover gold by rep index without knowing the
    exact n used at run time.
    """
    small = generate_specs(op, fixture_corpus, n=2, seed=42)
    large = generate_specs(op, fixture_corpus, n=10, seed=42)

    # small must be a prefix of large
    assert len(small) <= len(large)
    for a, b in zip(small, large):
        assert a.payload == b.payload
        assert a.gold == b.gold


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
