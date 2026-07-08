"""
NS-ICE adapter: native index CLI -> REST code-graph tools.

Uses E7's native index CLI and I3's snapshot_cli for all operations.
Supports all 5 op classes.

Capabilities: symbol_lookup, neighbors_1hop, path_le4, nl_locate, blast_radius
"""

import json
import subprocess
import time
from pathlib import Path

import httpx

from icebench.adapters.base import (
    SystemAdapter,
    Corpus,
    IndexResult,
    QueryResult,
    SnapshotResult,
    UnsupportedOp,
)


class NSIceAdapter:
    """Adapter for NS using native code-intel engine."""

    def __init__(
        self,
        api_url: str = "http://localhost:8499",
        python_bin: str = "python",
    ):
        """
        Initialize the adapter.

        Args:
            api_url: NS API base URL.
            python_bin: Python binary to use for CLI calls.
        """
        self.name = "ns-ice"
        self.version = "ns-api@TODO"  # TODO: Read from API /health
        self.api_url = api_url
        self.python_bin = python_bin
        self.client = httpx.Client(timeout=120.0)

    def capabilities(self) -> set[str]:
        """NS-ICE supports all 5 op classes."""
        return {
            "symbol_lookup",
            "neighbors_1hop",
            "path_le4",
            "nl_locate",
            "blast_radius",
        }

    def _make_code_space(self, corpus: Corpus) -> str:
        """Generate a code_space identifier for the corpus."""
        return f"code--ice-bench--{corpus.name}"

    def index_cold(self, corpus: Corpus) -> IndexResult:
        """Run native index CLI (cold/full)."""
        # ICE-INTEGRATE: Guard CLI module existence
        code_space = self._make_code_space(corpus)

        start = time.time()
        try:
            # Run the native index CLI
            result = subprocess.run(
                [
                    self.python_bin,
                    "-m",
                    "adapters.code_graph.native_index_cli",
                    "--repo-path",
                    corpus.path,
                    "--code-space",
                    code_space,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse JSON output
            output = json.loads(result.stdout)
            wall_s = output.get("wall_s", time.time() - start)

            return IndexResult(
                wall_s=wall_s,
                peak_rss_mb=0,  # TODO: Track via psutil wrapper
                cpu_s=0,
                symbols=output.get("symbols", 0),
                edges=output.get("edges", 0),
                files=output.get("files", 0),
                ok=True,
            )

        except FileNotFoundError:
            # CLI module not found (E7 not merged yet)
            return IndexResult(
                wall_s=time.time() - start,
                peak_rss_mb=0,
                cpu_s=0,
                symbols=0,
                edges=0,
                files=0,
                ok=False,
                dnf=True,
                dnf_reason="native_index_cli module not found (E7 not merged)",
            )
        except subprocess.CalledProcessError as e:
            return IndexResult(
                wall_s=time.time() - start,
                peak_rss_mb=0,
                cpu_s=0,
                symbols=0,
                edges=0,
                files=0,
                ok=False,
                dnf=False,
            )
        except json.JSONDecodeError:
            return IndexResult(
                wall_s=time.time() - start,
                peak_rss_mb=0,
                cpu_s=0,
                symbols=0,
                edges=0,
                files=0,
                ok=False,
            )

    def index_incremental(self, corpus: Corpus, touched: list[str]) -> IndexResult:
        """Run native index CLI with --incremental flag."""
        code_space = self._make_code_space(corpus)

        start = time.time()
        try:
            result = subprocess.run(
                [
                    self.python_bin,
                    "-m",
                    "adapters.code_graph.native_index_cli",
                    "--repo-path",
                    corpus.path,
                    "--code-space",
                    code_space,
                    "--incremental",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            output = json.loads(result.stdout)
            wall_s = output.get("wall_s", time.time() - start)

            return IndexResult(
                wall_s=wall_s,
                peak_rss_mb=0,
                cpu_s=0,
                symbols=output.get("symbols", 0),
                edges=output.get("edges", 0),
                files=output.get("files", 0),
                ok=True,
            )

        except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
            return IndexResult(
                wall_s=time.time() - start,
                peak_rss_mb=0,
                cpu_s=0,
                symbols=0,
                edges=0,
                files=0,
                ok=False,
            )

    def index_second(self, corpus: Corpus) -> IndexResult:
        """Second full index (stability probe)."""
        return self.index_cold(corpus)

    def store_size_bytes(self, corpus: Corpus) -> int:
        """
        Measure Neo4j store size for this code_space.

        Note: This requires querying Neo4j or the API for storage metrics.
        For now, return 0 as a placeholder.
        """
        # TODO: Implement via Neo4j query or API endpoint
        return 0

    def export_snapshot(self, corpus: Corpus) -> SnapshotResult | None:
        """Export snapshot via snapshot_cli."""
        # ICE-INTEGRATE: Guard snapshot_cli module
        code_space = self._make_code_space(corpus)
        snapshot_path = Path(corpus.path) / f"snapshot-{corpus.name}.bin"

        start = time.time()
        try:
            result = subprocess.run(
                [
                    self.python_bin,
                    "-m",
                    "adapters.code_graph.snapshot_cli",
                    "export",
                    "--code-space",
                    code_space,
                    "--output",
                    str(snapshot_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            wall_s = time.time() - start
            return SnapshotResult(
                wall_s=wall_s,
                bytes=snapshot_path.stat().st_size if snapshot_path.exists() else 0,
                ok=True,
            )

        except FileNotFoundError:
            # snapshot_cli not found (I3 not merged)
            return SnapshotResult(
                wall_s=time.time() - start,
                bytes=0,
                ok=False,
                dnf=True,
                dnf_reason="snapshot_cli module not found (I3 not merged)",
            )
        except subprocess.CalledProcessError:
            return SnapshotResult(
                wall_s=time.time() - start,
                bytes=0,
                ok=False,
            )

    def import_snapshot(self, corpus: Corpus, blob_path: str) -> SnapshotResult | None:
        """Import snapshot via snapshot_cli."""
        code_space = self._make_code_space(corpus)

        start = time.time()
        try:
            result = subprocess.run(
                [
                    self.python_bin,
                    "-m",
                    "adapters.code_graph.snapshot_cli",
                    "import",
                    "--code-space",
                    code_space,
                    "--input",
                    blob_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            wall_s = time.time() - start
            return SnapshotResult(
                wall_s=wall_s,
                bytes=Path(blob_path).stat().st_size,
                ok=True,
            )

        except (FileNotFoundError, subprocess.CalledProcessError):
            return SnapshotResult(
                wall_s=time.time() - start,
                bytes=0,
                ok=False,
            )

    def query(self, op: str, payload: dict) -> QueryResult:
        """Execute a query via NS REST code-graph tools."""
        if op not in self.capabilities():
            raise UnsupportedOp(f"Operation {op} not supported by ns-ice")

        code_space = self._make_code_space(payload.get("corpus", Corpus("", "", "", "", 0, 0)))

        start = time.perf_counter()

        try:
            if op == "symbol_lookup":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/query",
                    params={
                        "query": payload["symbol"],
                        "graph_id": code_space,
                    },
                )
            elif op == "neighbors_1hop":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/neighbors",
                    params={
                        "symbol": payload["symbol"],
                        "graph_id": code_space,
                    },
                )
            elif op == "path_le4":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/path",
                    params={
                        "from_symbol": payload["from"],
                        "to_symbol": payload["to"],
                        "max_depth": 4,
                        "graph_id": code_space,
                    },
                )
            elif op == "nl_locate":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/locate",
                    params={
                        "query": payload["query"],
                        "graph_id": code_space,
                    },
                )
            elif op == "blast_radius":
                # ICE-INTEGRATE: E7 adds this endpoint
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/impact",
                    params={
                        "symbol": payload["symbol"],
                        "max_hops": payload.get("max_hops", 4),
                        "graph_id": code_space,
                    },
                )
            else:
                raise UnsupportedOp(f"Unexpected op: {op}")

            latency_ms = (time.perf_counter() - start) * 1000

            # Parse answer
            answer = {}
            if resp.status_code == 200:
                answer = {"text": resp.text, "status": "ok"}
            else:
                answer = {"error": resp.text, "status": "error"}

            return QueryResult(
                latency_ms=latency_ms,
                answer=answer,
                ok=resp.status_code == 200,
            )

        except httpx.RequestError as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return QueryResult(
                latency_ms=latency_ms,
                answer={"error": str(e), "status": "error"},
                ok=False,
            )

    def teardown(self, corpus: Corpus) -> None:
        """
        Clean up Neo4j nodes for this code_space.

        Note: This should delete all nodes/edges with the code_space label.
        For now, this is a no-op.
        """
        # TODO: Implement via Cypher DELETE or API endpoint
        pass
