"""
Graphify standalone adapter: graphify CLI -> graph.json -> local query engine.

Drives graphify as its own standalone product (NOT through Neuralscape).
Uses graphify CLI to produce graph.json, then answers queries by loading and
traversing that graph directly using its own built-in query commands.

Capabilities: symbol_lookup, neighbors_1hop, path_le4
N/A: nl_locate, blast_radius (graphify has no NL query or impact analysis)

All indexing operations run under the safety rail (icebench.rail.run_with_rail)
so peak RSS / CPU-seconds are measured and memory/timeout breaches become DNF
rows rather than VM crashes.
"""

import json
import time
from pathlib import Path

from icebench.adapters.base import (
    Corpus,
    IndexResult,
    QueryResult,
    SnapshotResult,
    UnsupportedOp,
)
from icebench.rail import RailConfig, run_with_rail


class GraphifyStandaloneAdapter:
    """Adapter for graphify as a standalone system."""

    def __init__(
        self,
        graphify_bin: str = "/data/ice/tools/graphify",
        python_bin: str = "python3",
        rail: RailConfig | None = None,
    ):
        """
        Initialize the adapter.

        Args:
            graphify_bin: Path to graphify installation directory.
            python_bin: Python binary to run graphify module.
            rail: Safety-rail config (cap + timeout).
        """
        self.name = "graphify"
        # Read version from installed graphify if available
        self.version = self._get_version(graphify_bin, python_bin)
        self.graphify_bin = graphify_bin
        self.python_bin = python_bin
        self.rail = rail or RailConfig()

    def _get_version(self, graphify_bin: str, python_bin: str) -> str:
        """Get graphify version from installation."""
        try:
            import subprocess
            result = subprocess.run(
                [python_bin, "-m", "graphify", "--version"],
                cwd=graphify_bin,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Output format: "graphify 0.9.10"
                version = result.stdout.strip().split()[-1]
                return f"graphify-cli@{version}"
        except Exception:
            pass
        return "graphify-cli@unknown"

    def capabilities(self) -> set[str]:
        """
        Graphify natively supports 3 structural operations.

        - symbol_lookup: explain <symbol>
        - neighbors_1hop: explain <symbol> (returns neighbors)
        - path_le4: path <from> <to> (shortest path query)

        nl_locate and blast_radius are N/A (graphify has no NL query or
        impact analysis tools).
        """
        return {"symbol_lookup", "neighbors_1hop", "path_le4"}

    def _graphify_out_path(self, corpus: Corpus) -> Path:
        """Path to graphify-out directory for a corpus."""
        return Path(corpus.path) / "graphify-out"

    def _graph_json_path(self, corpus: Corpus) -> Path:
        """Path where graphify writes its graph.json for a corpus."""
        return self._graphify_out_path(corpus) / "graph.json"

    def index_cold(self, corpus: Corpus) -> IndexResult:
        """
        Run graphify extraction on a corpus (cold index).

        Invokes: python3 -m graphify extract <corpus.path>
        This produces graphify-out/graph.json as the entire index.
        """
        # Guard: graphify binary/module must exist
        if not Path(self.graphify_bin).exists():
            return IndexResult(
                wall_s=0,
                peak_rss_mb=0,
                cpu_s=0,
                symbols=0,
                edges=0,
                files=0,
                ok=False,
                dnf=True,
                dnf_reason=f"graphify not found: {self.graphify_bin}",
            )

        # Run graphify extract under the safety rail
        cmd = [
            self.python_bin,
            "-m",
            "graphify",
            "extract",
            corpus.path,
            "--no-cluster",  # Skip LLM clustering to stay local/deterministic
            "--max-workers",
            "1",  # Single-threaded to reduce memory footprint
        ]

        res = run_with_rail(cmd, self.rail, cwd=Path(self.graphify_bin))

        # DNF (timeout or OOM)
        if res.dnf:
            return IndexResult(
                wall_s=res.wall_s,
                peak_rss_mb=res.peak_rss_mb,
                cpu_s=res.cpu_s,
                symbols=0,
                edges=0,
                files=0,
                ok=False,
                dnf=True,
                dnf_reason=res.dnf_reason,
            )

        # Failed (non-zero exit)
        if res.returncode != 0:
            return IndexResult(
                wall_s=res.wall_s,
                peak_rss_mb=res.peak_rss_mb,
                cpu_s=res.cpu_s,
                symbols=0,
                edges=0,
                files=0,
                ok=False,
            )

        # Parse graph.json for counts
        graph_json_path = self._graph_json_path(corpus)
        if not graph_json_path.exists():
            return IndexResult(
                wall_s=res.wall_s,
                peak_rss_mb=res.peak_rss_mb,
                cpu_s=res.cpu_s,
                symbols=0,
                edges=0,
                files=0,
                ok=False,
            )

        try:
            with open(graph_json_path) as f:
                graph_data = json.load(f)
            symbols = len(graph_data.get("nodes", []))
            edges = len(graph_data.get("edges", []))
            files = len(graph_data.get("files", []))
        except (OSError, json.JSONDecodeError):
            symbols = edges = files = 0

        return IndexResult(
            wall_s=res.wall_s,
            peak_rss_mb=res.peak_rss_mb,
            cpu_s=res.cpu_s,
            symbols=symbols,
            edges=edges,
            files=files,
            ok=True,
        )

    def index_incremental(self, corpus: Corpus, touched: list[str]) -> IndexResult:
        """
        Graphify has no incremental mode => N/A.

        Returns DNF with reason to distinguish from a failed index.
        """
        return IndexResult(
            wall_s=0,
            peak_rss_mb=0,
            cpu_s=0,
            symbols=0,
            edges=0,
            files=0,
            ok=False,
            dnf=True,
            dnf_reason="incremental_na",
        )

    def index_second(self, corpus: Corpus) -> IndexResult:
        """Second full index (stability probe)."""
        return self.index_cold(corpus)

    def store_size_bytes(self, corpus: Corpus) -> int:
        """
        Measure graphify-out/ directory size.

        graphify's entire on-disk store is the graphify-out directory
        (graph.json + GRAPH_REPORT.md + graph.html if generated).
        """
        graphify_out = self._graphify_out_path(corpus)
        if not graphify_out.exists():
            return 0

        total = 0
        for item in graphify_out.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    pass
        return total

    def export_snapshot(self, corpus: Corpus) -> SnapshotResult | None:
        """
        Snapshot = copy graphify-out/graph.json.

        graphify has no native snapshot command; the graph.json file IS
        the snapshot.
        """
        graph_json_path = self._graph_json_path(corpus)
        if not graph_json_path.exists():
            return None

        start = time.monotonic()
        snapshot_path = Path(corpus.path) / f"snapshot-{corpus.name}.json"
        try:
            snapshot_path.write_bytes(graph_json_path.read_bytes())
            wall_s = time.monotonic() - start
            return SnapshotResult(
                wall_s=wall_s,
                bytes=snapshot_path.stat().st_size,
                ok=True,
            )
        except OSError:
            return SnapshotResult(
                wall_s=time.monotonic() - start,
                bytes=0,
                ok=False,
            )

    def import_snapshot(self, corpus: Corpus, blob_path: str) -> SnapshotResult | None:
        """Import = copy snapshot to graphify-out/graph.json."""
        start = time.monotonic()
        graph_json_path = self._graph_json_path(corpus)

        # Ensure graphify-out directory exists
        graph_json_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            graph_json_path.write_bytes(Path(blob_path).read_bytes())
            wall_s = time.monotonic() - start
            return SnapshotResult(
                wall_s=wall_s,
                bytes=graph_json_path.stat().st_size,
                ok=True,
            )
        except OSError:
            return SnapshotResult(
                wall_s=time.monotonic() - start,
                bytes=0,
                ok=False,
            )

    def query(self, op: str, payload: dict) -> QueryResult:
        """
        Execute a query using graphify's built-in CLI commands.

        - symbol_lookup: graphify explain "<symbol>"
        - neighbors_1hop: graphify explain "<symbol>" (same as lookup)
        - path_le4: graphify path "<from>" "<to>"
        """
        if op not in self.capabilities():
            raise UnsupportedOp(f"Operation {op} not supported by graphify")

        # Resolve corpus to get graph.json path
        corpus = payload.get("corpus")
        if corpus is None:
            return QueryResult(
                latency_ms=0,
                answer={"error": "No corpus specified", "status": "error"},
                ok=False,
            )

        graph_json_path = self._graph_json_path(corpus)
        if not graph_json_path.exists():
            return QueryResult(
                latency_ms=0,
                answer={"error": "Graph not indexed", "status": "error"},
                ok=False,
            )

        start = time.perf_counter()

        try:
            import subprocess

            if op in ("symbol_lookup", "neighbors_1hop"):
                symbol = payload.get("symbol", "")
                cmd = [
                    self.python_bin,
                    "-m",
                    "graphify",
                    "explain",
                    symbol,
                    "--graph",
                    str(graph_json_path),
                ]
            elif op == "path_le4":
                from_sym = payload.get("from", "")
                to_sym = payload.get("to", "")
                cmd = [
                    self.python_bin,
                    "-m",
                    "graphify",
                    "path",
                    from_sym,
                    to_sym,
                    "--graph",
                    str(graph_json_path),
                ]
            else:
                raise UnsupportedOp(f"Unexpected op: {op}")

            result = subprocess.run(
                cmd,
                cwd=self.graphify_bin,
                capture_output=True,
                text=True,
                timeout=30,
            )

            latency_ms = (time.perf_counter() - start) * 1000

            if result.returncode == 0:
                answer = {"text": result.stdout, "status": "ok"}
            else:
                answer = {"error": result.stderr or result.stdout, "status": "error"}

            return QueryResult(
                latency_ms=latency_ms,
                answer=answer,
                ok=result.returncode == 0,
            )

        except subprocess.TimeoutExpired:
            latency_ms = (time.perf_counter() - start) * 1000
            return QueryResult(
                latency_ms=latency_ms,
                answer={"error": "Query timeout", "status": "error"},
                ok=False,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return QueryResult(
                latency_ms=latency_ms,
                answer={"error": str(e), "status": "error"},
                ok=False,
            )

    def teardown(self, corpus: Corpus) -> None:
        """Clean up graphify-out directory and any snapshots."""
        graphify_out = self._graphify_out_path(corpus)
        if graphify_out.exists():
            try:
                import shutil
                shutil.rmtree(graphify_out)
            except OSError:
                pass

        # Remove snapshot if it exists
        snapshot_path = Path(corpus.path) / f"snapshot-{corpus.name}.json"
        if snapshot_path.exists():
            try:
                snapshot_path.unlink()
            except OSError:
                pass
