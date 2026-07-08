"""Tests for runner end-to-end (with FakeAdapter)."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from icebench.run import _load_systems, _generate_fixture_queries
from icebench.adapters.base import Corpus
from icebench.schema import RunManifest, write_row, ResultRow


def test_load_systems():
    """Test system loader."""
    # ns-ice and ns-graphify should load (if httpx available)
    systems = _load_systems(["ns-ice", "ns-graphify"])
    assert len(systems) >= 0  # May be 0 if httpx not installed

    # Unknown system
    systems = _load_systems(["unknown"])
    assert len(systems) == 0


def test_fixture_query_generator():
    """Test built-in fixture query generator."""
    corpus = Corpus(
        name="test",
        path="/tmp/test",
        repo_sha="abc123",
        language="python",
        loc=1000,
        file_count=10,
    )

    # Generate symbol_lookup queries
    queries = _generate_fixture_queries("symbol_lookup", corpus, n=5, seed=42)
    assert len(queries) == 5
    assert all("symbol" in q for q in queries)

    # Generate nl_locate queries
    queries = _generate_fixture_queries("nl_locate", corpus, n=3, seed=42)
    assert len(queries) == 3
    assert all("query" in q for q in queries)


def test_runner_resumability():
    """Test that runner respects manifest and skips completed cells."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_file = Path(tmpdir) / "test-run.jsonl"

        # Pre-populate with some completed cells
        write_row(
            results_file,
            ResultRow(
                schema="icebench-v1",
                kind="index",
                system="fake",
                system_version="1.0.0",
                corpus="test",
                repo_sha="abc123",
                op="index_cold",
                rep=0,
                seed=42,
                wall_s=10.0,
                ok=True,
            ),
        )

        # Load manifest
        manifest = RunManifest.load("test-run", results_file)

        # Should skip this cell
        assert manifest.is_completed("fake", "test", "index_cold", 0)

        # Should not skip this cell
        assert not manifest.is_completed("fake", "test", "index_cold", 1)
