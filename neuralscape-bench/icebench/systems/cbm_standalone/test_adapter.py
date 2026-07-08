"""Tests for CBM standalone adapter."""

import pytest
from pathlib import Path
from icebench.adapters.base import (
    Corpus,
    UnsupportedOp,
    OP_CLASSES,
)
from icebench.systems.cbm_standalone import CBMStandaloneAdapter


@pytest.fixture
def adapter():
    """Create a CBM adapter instance."""
    return CBMStandaloneAdapter(
        cbm_bin="/data/ice/tools/cbm/codebase-memory-mcp",
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
    assert adapter.name == "cbm"
    assert "cbm@" in adapter.version
    assert hasattr(adapter, "capabilities")


def test_capabilities(adapter):
    """Test CBM capabilities are correct subset of OP_CLASSES."""
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
    """Test index_cold handles missing CBM binary."""
    adapter = CBMStandaloneAdapter(cbm_bin="/nonexistent/path")
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


def test_sigabrt_detection(tmp_path):
    """Test SIGABRT detection in index results."""
    # Create a fake binary
    fake_bin = tmp_path / "cbm"
    fake_bin.write_text("#!/bin/sh\nexit 134")
    fake_bin.chmod(0o755)

    adapter = CBMStandaloneAdapter(cbm_bin=str(fake_bin))

    # Simulate SIGABRT returncode
    from icebench.rail import RailResult
    from unittest.mock import patch

    fake_result = RailResult(
        returncode=134,  # SIGABRT
        stdout="",
        stderr="",
        wall_s=1.0,
        peak_rss_mb=100,
        cpu_s=0.8,
        timed_out=False,
        oom_killed=False,
        memory_cap_mb=12288,
        timeout_s=3600,
        mechanism="systemd",
    )

    corpus = Corpus(
        name="test",
        path="/tmp/test",
        repo_sha="abc123",
        language="python",
        loc=100,
        file_count=5,
    )

    # Check that SIGABRT is detected and reported
    with patch("icebench.systems.cbm_standalone.adapter.run_with_rail", return_value=fake_result):
        result = adapter.index_cold(corpus)

        assert not result.ok
        assert result.dnf
        assert "SIGABRT" in result.dnf_reason


def test_snapshot_operations_no_artifact(adapter, corpus):
    """Test snapshot operations when no artifact exists."""
    # Export without artifact returns None
    result = adapter.export_snapshot(corpus)
    assert result is None


def test_import_snapshot(adapter, corpus, tmp_path):
    """Test snapshot import creates .codebase-memory directory."""
    # Create a fake snapshot file
    snapshot = tmp_path / "snapshot.db.zst"
    snapshot.write_bytes(b"fake zstd data")

    result = adapter.import_snapshot(corpus, str(snapshot))

    assert result is not None
    assert result.ok
    # Check .codebase-memory was created
    cbm_dir = Path(corpus.path) / ".codebase-memory"
    assert cbm_dir.exists()


def test_store_size_bytes_no_index(adapter, corpus):
    """Test store_size_bytes when no index exists."""
    # Should return 0 or sum of cache dir (may have other projects)
    size = adapter.store_size_bytes(corpus)
    assert isinstance(size, int)
    assert size >= 0


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
    not Path("/data/ice/tools/cbm/codebase-memory-mcp").exists(),
    reason="CBM binary not installed"
)
def test_version_detection():
    """Test version detection from installed CBM."""
    adapter = CBMStandaloneAdapter(cbm_bin="/data/ice/tools/cbm/codebase-memory-mcp")
    # Version should be detected if binary exists
    # (may still be unknown if binary is not executable or --version fails)
    assert "cbm@" in adapter.version
