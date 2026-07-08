"""
Results schema — LOCKED CONTRACT.

Every measurement is one JSONL row appended to /data/ice/results/raw/<run_id>.jsonl.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal, Iterator


# Schema version
SCHEMA_VERSION = "icebench-v1"

# Machine identifier
MACHINE_ID = "ns-bench 8vCPU/31GB"


@dataclass
class ResultRow:
    """A single benchmark result row."""

    schema: str  # Always "icebench-v1"
    kind: Literal["index", "query", "snapshot", "store"]
    system: str  # "ns-ice" | "ns-graphify" | "graphify" | "cbm"
    system_version: str
    corpus: str
    repo_sha: str
    op: str  # "symbol_lookup" | "index_cold" | etc.
    rep: int  # Repetition number (0-based)
    seed: int  # Random seed for reproducibility

    # Metrics (depend on kind)
    wall_s: float | None = None
    peak_rss_mb: float | None = None
    cpu_s: float | None = None
    bytes: int | None = None
    latency_ms: float | None = None

    # Status
    ok: bool = True
    dnf: bool = False  # Did Not Finish (timeout or OOM)
    dnf_reason: str | None = None

    # Answer (for query kind)
    answer: dict | None = None

    # Metadata
    machine: str = MACHINE_ID
    ts: str | None = None  # UTC ISO timestamp

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.ts is None:
            self.ts = datetime.now(timezone.utc).isoformat()


def write_row(results_file: Path, row: ResultRow) -> None:
    """
    Atomically append a result row to the JSONL file.

    Args:
        results_file: Path to results file.
        row: Result row to write.
    """
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "a") as f:
        json.dump(asdict(row), f)
        f.write("\n")


def read_rows(results_file: Path) -> Iterator[ResultRow]:
    """
    Read all result rows from a JSONL file.

    Args:
        results_file: Path to results file.

    Yields:
        ResultRow objects.
    """
    if not results_file.exists():
        return

    with open(results_file) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                yield ResultRow(**data)


@dataclass
class RunManifest:
    """
    Manifest for resumable runs.

    Tracks which (system, corpus, op, rep) cells have been completed.
    """

    run_id: str
    results_file: Path
    completed: set[tuple[str, str, str, int]]  # (system, corpus, op, rep)

    @classmethod
    def load(cls, run_id: str, results_file: Path) -> "RunManifest":
        """
        Load or create a manifest by scanning the results file.

        Args:
            run_id: Unique run identifier.
            results_file: Path to results file.

        Returns:
            RunManifest with completed cells loaded.
        """
        completed = set()
        for row in read_rows(results_file):
            completed.add((row.system, row.corpus, row.op, row.rep))

        return cls(
            run_id=run_id,
            results_file=results_file,
            completed=completed,
        )

    def is_completed(self, system: str, corpus: str, op: str, rep: int) -> bool:
        """
        Check if a cell has been completed.

        Args:
            system: System name.
            corpus: Corpus name.
            op: Operation name.
            rep: Repetition number.

        Returns:
            True if completed.
        """
        return (system, corpus, op, rep) in self.completed

    def mark_completed(self, system: str, corpus: str, op: str, rep: int) -> None:
        """
        Mark a cell as completed.

        Args:
            system: System name.
            corpus: Corpus name.
            op: Operation name.
            rep: Repetition number.
        """
        self.completed.add((system, corpus, op, rep))
