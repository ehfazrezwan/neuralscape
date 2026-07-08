"""Tests for Graphify standalone adapter."""

import pytest
from pathlib import Path
from icebench.adapters.base import (
    Corpus,
    UnsupportedOp,
    OP_CLASSES,
)
from icebench.systems.graphify_standalone import GraphifyStandaloneAdapter


@pytest.fixture
def adapter():
    """Create a graphify adapter instance."""
    return GraphifyStandaloneAdapter(
        graphify_bin="/data/ice/tools/graphify",
        python_bin="python3",
    )


@pytest.fixture
def corpus(tmp_path):
    """Create a test corpus."""
    return Corpus(
        name="test",
        path=str(tmp_path),
        repo_sha="abc123",
        language="python",
        loc=100,
        file_count=5,
    )


def test_adapter_attributes(adapter):
    """Test adapter has required attributes."""
    assert adapter.name == "graphify"
    assert "graphify-cli" in adapter.version
    assert hasattr(adapter, "capabilities")


def test_capabilities(adapter):
    """Test graphify capabilities are correct subset of OP_CLASSES."""
    caps = adapter.capabilities()
    assert isinstance(caps, set)
    assert caps <= OP_CLASSES  # Subset of valid ops
    assert caps == {"symbol_lookup", "neighbors_1hop", "path_le4"}


def test_unsupported_ops(adapter, corpus):
    """Test that unsupported operations raise UnsupportedOp."""
    # nl_locate is not supported
    with pytest.raises(UnsupportedOp):
        adapter.query("nl_locate", {"corpus": corpus, "query": "test"})

    # blast_radius is not supported
    with pytest.raises(UnsupportedOp):
        adapter.query("blast_radius", {"corpus": corpus, "symbol": "test"})


def test_index_cold_missing_binary(tmp_path):
    """Test index_cold handles missing graphify binary."""
    adapter = GraphifyStandaloneAdapter(
        graphify_bin="/nonexistent/path",
        python_bin="python3",
    )
    corpus = Corpus(
        name="test",
        path=str(tmp_path),
        repo_sha="abc123",
        language="python",
        loc=100,
        file_count=5,
    )

    result = adapter.index_cold(corpus)

    assert not result.ok
    assert result.dnf
    assert "not found" in result.dnf_reason


def test_index_incremental_na(adapter, corpus):
    """Test incremental index returns N/A."""
    result = adapter.index_incremental(corpus, ["file.py"])

    assert not result.ok
    assert result.dnf
    assert result.dnf_reason == "incremental_na"


def test_snapshot_operations_no_graph(adapter, corpus):
    """Test snapshot operations when no graph exists."""
    # Export without graph returns None
    result = adapter.export_snapshot(corpus)
    assert result is None


def test_import_snapshot(adapter, corpus, tmp_path):
    """Test snapshot import creates graphify-out directory."""
    # Create a fake snapshot file
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text('{"nodes": [], "edges": []}')

    result = adapter.import_snapshot(corpus, str(snapshot))

    assert result is not None
    assert result.ok
    # Check graphify-out was created
    graphify_out = Path(corpus.path) / "graphify-out"
    assert graphify_out.exists()


def test_store_size_bytes_no_index(adapter, corpus):
    """Test store_size_bytes returns 0 when not indexed."""
    size = adapter.store_size_bytes(corpus)
    assert size == 0


def test_query_without_corpus(adapter):
    """Test query without corpus in payload."""
    result = adapter.query("symbol_lookup", {"symbol": "test"})
    assert not result.ok
    assert "No corpus specified" in result.answer.get("error", "")


def test_conformance_to_protocol(adapter):
    """Test adapter conforms to SystemAdapter protocol."""
    # Check all required methods exist
    assert callable(getattr(adapter, "capabilities", None))
    assert callable(getattr(adapter, "index_cold", None))
    assert callable(getattr(adapter, "index_incremental", None))
    assert callable(getattr(adapter, "index_second", None))
    assert callable(getattr(adapter, "store_size_bytes", None))
    assert callable(getattr(adapter, "export_snapshot", None))
    assert callable(getattr(adapter, "import_snapshot", None))
    assert callable(getattr(adapter, "query", None))
    assert callable(getattr(adapter, "teardown", None))

    # Check attributes
    assert hasattr(adapter, "name")
    assert hasattr(adapter, "version")


@pytest.mark.skipif(
    not Path("/data/ice/tools/graphify").exists(),
    reason="Graphify not installed"
)
def test_version_detection():
    """Test version detection from installed graphify."""
    adapter = GraphifyStandaloneAdapter(
        graphify_bin="/data/ice/tools/graphify",
        python_bin="python3",
    )
    # Version should start with graphify-cli@
    # May be "unknown" if not properly installed or importable
    assert adapter.version.startswith("graphify-cli@")
