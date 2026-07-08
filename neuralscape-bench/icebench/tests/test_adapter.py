"""Tests for SystemAdapter protocol and N/A handling."""

import pytest

from icebench.adapters.base import (
    SystemAdapter,
    Corpus,
    IndexResult,
    QueryResult,
    SnapshotResult,
    UnsupportedOp,
    OP_CLASSES,
)


class FakeAdapter:
    """Fake adapter for testing the protocol."""

    def __init__(self, capabilities: set[str]):
        self.name = "fake"
        self.version = "1.0.0"
        self._caps = capabilities

    def capabilities(self) -> set[str]:
        return self._caps

    def index_cold(self, corpus: Corpus) -> IndexResult:
        return IndexResult(
            wall_s=1.0,
            peak_rss_mb=100,
            cpu_s=0.8,
            symbols=100,
            edges=50,
            files=10,
            ok=True,
        )

    def index_incremental(self, corpus: Corpus, touched: list[str]) -> IndexResult:
        return IndexResult(
            wall_s=0.5,
            peak_rss_mb=50,
            cpu_s=0.4,
            symbols=100,
            edges=50,
            files=10,
            ok=True,
        )

    def index_second(self, corpus: Corpus) -> IndexResult:
        return self.index_cold(corpus)

    def store_size_bytes(self, corpus: Corpus) -> int:
        return 1024 * 1024

    def export_snapshot(self, corpus: Corpus) -> SnapshotResult | None:
        # N/A for this fake adapter
        return None

    def import_snapshot(self, corpus: Corpus, blob_path: str) -> SnapshotResult | None:
        return None

    def query(self, op: str, payload: dict) -> QueryResult:
        if op not in self.capabilities():
            raise UnsupportedOp(f"Operation {op} not supported")

        return QueryResult(
            latency_ms=10.0,
            answer={"result": "dummy"},
            ok=True,
        )

    def teardown(self, corpus: Corpus) -> None:
        pass


def test_adapter_protocol():
    """Test that FakeAdapter satisfies the protocol."""
    adapter = FakeAdapter({"symbol_lookup", "neighbors_1hop"})

    # Check attributes
    assert adapter.name == "fake"
    assert adapter.version == "1.0.0"
    assert adapter.capabilities() == {"symbol_lookup", "neighbors_1hop"}

    # Test operations
    corpus = Corpus(
        name="test",
        path="/tmp/test",
        repo_sha="abc123",
        language="python",
        loc=1000,
        file_count=10,
    )

    result = adapter.index_cold(corpus)
    assert result.ok
    assert result.symbols == 100

    result = adapter.index_incremental(corpus, ["file.py"])
    assert result.ok

    size = adapter.store_size_bytes(corpus)
    assert size > 0


def test_unsupported_op():
    """Test UnsupportedOp exception for N/A operations."""
    adapter = FakeAdapter({"symbol_lookup"})

    corpus = Corpus(
        name="test",
        path="/tmp/test",
        repo_sha="abc123",
        language="python",
        loc=1000,
        file_count=10,
    )

    # Supported op works
    result = adapter.query("symbol_lookup", {"symbol": "foo"})
    assert result.ok

    # Unsupported op raises
    with pytest.raises(UnsupportedOp):
        adapter.query("blast_radius", {"symbol": "foo"})


def test_snapshot_na():
    """Test N/A handling for snapshots."""
    adapter = FakeAdapter({"symbol_lookup"})

    corpus = Corpus(
        name="test",
        path="/tmp/test",
        repo_sha="abc123",
        language="python",
        loc=1000,
        file_count=10,
    )

    # Snapshot returns None (N/A)
    result = adapter.export_snapshot(corpus)
    assert result is None

    result = adapter.import_snapshot(corpus, "/tmp/snapshot.bin")
    assert result is None


def test_op_classes():
    """Test OP_CLASSES constant."""
    assert "symbol_lookup" in OP_CLASSES
    assert "neighbors_1hop" in OP_CLASSES
    assert "path_le4" in OP_CLASSES
    assert "nl_locate" in OP_CLASSES
    assert "blast_radius" in OP_CLASSES
    assert len(OP_CLASSES) == 5
