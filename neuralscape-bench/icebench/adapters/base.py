"""
SystemAdapter protocol — LOCKED CONTRACT for H2/H3.

This protocol defines the interface all benchmark systems must implement.
DO NOT change this after the H1 PR without notifying the executor.
"""

from typing import Protocol
from dataclasses import dataclass


# Operation classes supported by code-intel systems
OP_CLASSES = {
    "symbol_lookup",
    "neighbors_1hop",
    "path_le4",
    "nl_locate",
    "blast_radius",
}


class UnsupportedOp(Exception):
    """Raised when a system does not support a given operation."""

    pass


@dataclass
class IndexResult:
    """Result from an indexing operation."""

    wall_s: float
    peak_rss_mb: float
    cpu_s: float
    symbols: int
    edges: int
    files: int
    ok: bool
    dnf: bool = False
    dnf_reason: str | None = None


@dataclass
class QueryResult:
    """Result from a query operation."""

    latency_ms: float
    answer: dict  # System-specific answer format
    ok: bool


@dataclass
class SnapshotResult:
    """Result from snapshot export/import."""

    wall_s: float
    bytes: int
    ok: bool
    dnf: bool = False
    dnf_reason: str | None = None


@dataclass
class Corpus:
    """A code corpus to benchmark against."""

    name: str
    path: str
    repo_sha: str
    language: str
    loc: int
    file_count: int


class SystemAdapter(Protocol):
    """
    Protocol defining the interface for all benchmark systems.

    Adapters MUST NOT add intelligence: no query rewriting, no fallback logic
    beyond parsing the system's own answer. An unsupported op => the adapter
    omits it from capabilities() and query() raises UnsupportedOp (the runner
    records N/A, never 0, never emulated).
    """

    name: str  # "ns-graphify" | "ns-ice" | "graphify" | "cbm"
    version: str  # pinned tool/system version string

    def capabilities(self) -> set[str]:
        """
        Return the subset of OP_CLASSES this system supports.

        Returns:
            Set of operation names from OP_CLASSES.
        """
        ...

    # ---- Track P: Indexing operations ----

    def index_cold(self, corpus: Corpus) -> IndexResult:
        """
        Perform a cold (from scratch) index of the corpus.

        Args:
            corpus: The corpus to index.

        Returns:
            IndexResult with timing and resource metrics.
        """
        ...

    def index_incremental(self, corpus: Corpus, touched: list[str]) -> IndexResult:
        """
        Perform an incremental re-index after files are touched.

        Args:
            corpus: The corpus to re-index.
            touched: List of file paths that were modified.

        Returns:
            IndexResult with timing and resource metrics.
        """
        ...

    def index_second(self, corpus: Corpus) -> IndexResult:
        """
        Perform a second full index run (stability probe).

        Args:
            corpus: The corpus to re-index.

        Returns:
            IndexResult with timing and resource metrics.
        """
        ...

    def store_size_bytes(self, corpus: Corpus) -> int:
        """
        Measure the on-disk storage size for the indexed corpus.

        Args:
            corpus: The indexed corpus.

        Returns:
            Size in bytes.
        """
        ...

    def export_snapshot(self, corpus: Corpus) -> SnapshotResult | None:
        """
        Export a snapshot of the indexed corpus.

        Args:
            corpus: The corpus to snapshot.

        Returns:
            SnapshotResult if supported, None if N/A.
        """
        ...

    def import_snapshot(self, corpus: Corpus, blob_path: str) -> SnapshotResult | None:
        """
        Import a snapshot for the corpus.

        Args:
            corpus: The corpus the snapshot belongs to.
            blob_path: Path to the snapshot file.

        Returns:
            SnapshotResult if supported, None if N/A.
        """
        ...

    # ---- Track P/Q: Query operations ----

    def query(self, op: str, payload: dict) -> QueryResult:
        """
        Execute a query operation.

        Args:
            op: Operation name (must be in capabilities()).
            payload: Operation-specific payload.

        Returns:
            QueryResult with latency and answer.

        Raises:
            UnsupportedOp: If op not in capabilities().
        """
        ...

    # ---- Cleanup ----

    def teardown(self, corpus: Corpus) -> None:
        """
        Clean up any resources/state for the corpus.

        Args:
            corpus: The corpus to tear down.
        """
        ...
