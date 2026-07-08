"""
Graphify standalone adapter: graphify CLI -> graph.json -> graphify CLI queries.

Drives graphify as its own standalone product (NOT through Neuralscape). Uses
the graphify CLI to produce graph.json (that IS its index), then answers queries
by invoking graphify's own `explain` / `path` commands against that graph.json.
No added intelligence: every answer is graphify's own CLI output, parsed as-is.

Verified against graphify 0.9.10 (safishamsi/graphify, MIT) on this VM:
  index: `graphify extract <path> --code-only --no-cluster`
         -> <path>/graphify-out/graph.json  (keys: nodes, edges, hyperedges,
            input_tokens, output_tokens — NO "files" key)
  query: `graphify explain "<label-or-id>" --graph <graph.json>`   (symbol/neighbors)
         `graphify path "<a>" "<b>" --graph <graph.json>`          (path)

Capabilities: symbol_lookup, neighbors_1hop, path_le4
N/A: nl_locate (graphify has no NL->symbol resolution; `query` is graph BFS,
     not semantic locate), blast_radius (no impact/change-propagation tool).

EVERY graphify subprocess — index AND query — runs through the safety rail
(icebench.rail.run_with_rail) so the memory cap + wall-timeout + peak-RSS/CPU
capture apply and a blowup is recorded as DNF instead of hitting the shared VM.
"""

import json
import subprocess
import time
from pathlib import Path

from icebench.adapters.base import (
    Corpus,
    IndexResult,
    QueryResult,
    SnapshotResult,
    UnsupportedOp,
)
from icebench.rail import RailConfig, RailResult, run_with_rail


# Default location of the graphify CLI executable installed by install.sh
# (a venv console script; see systems/graphify_standalone/install.sh).
DEFAULT_GRAPHIFY_BIN = "/data/ice/tools/graphify/.venv/bin/graphify"


class GraphifyStandaloneAdapter:
    """Adapter for graphify as a standalone system."""

    def __init__(
        self,
        graphify_bin: str = DEFAULT_GRAPHIFY_BIN,
        rail: RailConfig | None = None,
    ):
        """
        Initialize the adapter.

        Args:
            graphify_bin: Path to the graphify CLI executable.
            rail: Safety-rail config (cap + timeout). Runner injects the
                CLI-configured one; defaults to RailConfig() otherwise.
        """
        self.name = "graphify"
        self.graphify_bin = graphify_bin
        self.rail = rail or RailConfig()
        self.version = self._get_version()

    def _get_version(self) -> str:
        """
        Read the graphify version via `graphify --version` (bounded metadata
        probe, not an index/query workload).
        """
        if not Path(self.graphify_bin).exists():
            return "graphify-cli@unknown"
        try:
            result = subprocess.run(
                [self.graphify_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Output format: "graphify 0.9.10"
                return f"graphify-cli@{result.stdout.strip().split()[-1]}"
        except Exception:
            pass
        return "graphify-cli@unknown"

    def capabilities(self) -> set[str]:
        """
        Graphify natively supports 3 structural operations.

        - symbol_lookup: `graphify explain <symbol>` returns node details
        - neighbors_1hop: `graphify explain <symbol>` returns direct connections
        - path_le4: `graphify path <from> <to>` returns the shortest path

        nl_locate and blast_radius are N/A (see module docstring).
        """
        return {"symbol_lookup", "neighbors_1hop", "path_le4"}

    def _graphify_out_path(self, corpus: Corpus) -> Path:
        """Directory where graphify writes its output for a corpus."""
        return Path(corpus.path) / "graphify-out"

    def _graph_json_path(self, corpus: Corpus) -> Path:
        """Path to graphify's graph.json for a corpus."""
        return self._graphify_out_path(corpus) / "graph.json"

    def _index(self, corpus: Corpus) -> IndexResult:
        """Run `graphify extract` under the rail and parse graph.json."""
        if not Path(self.graphify_bin).exists():
            return IndexResult(
                wall_s=0, peak_rss_mb=0, cpu_s=0, symbols=0, edges=0, files=0,
                ok=False, dnf=True,
                dnf_reason=f"graphify not found: {self.graphify_bin}",
            )

        # `--code-only` = pure tree-sitter AST extraction (no LLM key, fully
        # deterministic); `--no-cluster` skips the LLM clustering pass. graphify
        # writes to <corpus.path>/graphify-out/ regardless of cwd.
        cmd = [
            self.graphify_bin,
            "extract",
            corpus.path,
            "--code-only",
            "--no-cluster",
        ]
        res = run_with_rail(cmd, self.rail)

        if res.dnf:
            return IndexResult(
                wall_s=res.wall_s, peak_rss_mb=res.peak_rss_mb, cpu_s=res.cpu_s,
                symbols=0, edges=0, files=0, ok=False, dnf=True,
                dnf_reason=res.dnf_reason,
            )
        if res.returncode != 0:
            return IndexResult(
                wall_s=res.wall_s, peak_rss_mb=res.peak_rss_mb, cpu_s=res.cpu_s,
                symbols=0, edges=0, files=0, ok=False,
            )

        graph_json_path = self._graph_json_path(corpus)
        if not graph_json_path.exists():
            return IndexResult(
                wall_s=res.wall_s, peak_rss_mb=res.peak_rss_mb, cpu_s=res.cpu_s,
                symbols=0, edges=0, files=0, ok=False,
            )

        try:
            with open(graph_json_path) as f:
                graph_data = json.load(f)
            nodes = graph_data.get("nodes", [])
            symbols = len(nodes)
            edges = len(graph_data.get("edges", []))
            # graph.json has no "files" key; derive the file count from the
            # distinct source_file values across nodes.
            files = len({
                n.get("source_file")
                for n in nodes
                if n.get("source_file")
            })
        except (OSError, json.JSONDecodeError):
            symbols = edges = files = 0

        return IndexResult(
            wall_s=res.wall_s, peak_rss_mb=res.peak_rss_mb, cpu_s=res.cpu_s,
            symbols=symbols, edges=edges, files=files, ok=True,
        )

    def index_cold(self, corpus: Corpus) -> IndexResult:
        """Cold (from-scratch) index of the corpus."""
        return self._index(corpus)

    def index_incremental(self, corpus: Corpus, touched: list[str]) -> IndexResult:
        """
        Graphify has no incremental index mode => N/A.

        Returns DNF with dnf_reason='incremental_na' so the runner records N/A
        rather than emulating an incremental run.
        """
        return IndexResult(
            wall_s=0, peak_rss_mb=0, cpu_s=0, symbols=0, edges=0, files=0,
            ok=False, dnf=True, dnf_reason="incremental_na",
        )

    def index_second(self, corpus: Corpus) -> IndexResult:
        """Second full index (stability probe)."""
        return self._index(corpus)

    def store_size_bytes(self, corpus: Corpus) -> int:
        """
        Measure graphify's on-disk store: the whole graphify-out/ directory
        (graph.json + GRAPH_REPORT.md + graph.html), scoped to this corpus.
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

        graphify has no dedicated snapshot command; graph.json IS the portable
        store, so a byte-copy is the faithful snapshot.
        """
        graph_json_path = self._graph_json_path(corpus)
        if not graph_json_path.exists():
            return None

        start = time.monotonic()
        snapshot_path = Path(corpus.path) / f"snapshot-{corpus.name}.json"
        try:
            snapshot_path.write_bytes(graph_json_path.read_bytes())
            return SnapshotResult(
                wall_s=time.monotonic() - start,
                bytes=snapshot_path.stat().st_size,
                ok=True,
            )
        except OSError:
            return SnapshotResult(wall_s=time.monotonic() - start, bytes=0, ok=False)

    def import_snapshot(self, corpus: Corpus, blob_path: str) -> SnapshotResult | None:
        """Import = copy a snapshot back to graphify-out/graph.json."""
        start = time.monotonic()
        graph_json_path = self._graph_json_path(corpus)
        graph_json_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            graph_json_path.write_bytes(Path(blob_path).read_bytes())
            return SnapshotResult(
                wall_s=time.monotonic() - start,
                bytes=graph_json_path.stat().st_size,
                ok=True,
            )
        except OSError:
            return SnapshotResult(wall_s=time.monotonic() - start, bytes=0, ok=False)

    def _query_result(self, res: RailResult, start: float) -> QueryResult:
        """Map a rail-run graphify CLI query into a QueryResult."""
        latency_ms = (time.perf_counter() - start) * 1000
        if res.dnf:
            return QueryResult(
                latency_ms=latency_ms,
                answer={"error": res.dnf_reason, "status": "dnf"},
                ok=False,
            )
        if res.returncode == 0:
            return QueryResult(
                latency_ms=latency_ms,
                answer={"text": res.stdout, "status": "ok"},
                ok=True,
            )
        return QueryResult(
            latency_ms=latency_ms,
            answer={"error": res.stderr or res.stdout, "status": "error"},
            ok=False,
        )

    def query(self, op: str, payload: dict) -> QueryResult:
        """
        Execute a query via graphify's own CLI (routed through the rail).

        payload must carry the target Corpus under "corpus" so we can locate
        its graph.json.
        """
        if op not in self.capabilities():
            raise UnsupportedOp(f"Operation {op} not supported by graphify")

        corpus = payload.get("corpus")
        if not isinstance(corpus, Corpus):
            return QueryResult(
                latency_ms=0,
                answer={"error": "payload['corpus'] must be a Corpus", "status": "error"},
                ok=False,
            )

        graph_json_path = self._graph_json_path(corpus)
        if not graph_json_path.exists():
            return QueryResult(
                latency_ms=0,
                answer={"error": "graph not indexed", "status": "error"},
                ok=False,
            )

        if op in ("symbol_lookup", "neighbors_1hop"):
            symbol = payload.get("symbol", "")
            cmd = [
                self.graphify_bin, "explain", symbol,
                "--graph", str(graph_json_path),
            ]
        elif op == "path_le4":
            cmd = [
                self.graphify_bin, "path",
                payload.get("from", ""), payload.get("to", ""),
                "--graph", str(graph_json_path),
            ]
        else:  # pragma: no cover - guarded by capabilities() check above
            raise UnsupportedOp(f"Unexpected op: {op}")

        start = time.perf_counter()
        res = run_with_rail(cmd, self.rail)
        return self._query_result(res, start)

    def teardown(self, corpus: Corpus) -> None:
        """Remove the graphify-out directory and any snapshot artifact."""
        graphify_out = self._graphify_out_path(corpus)
        if graphify_out.exists():
            try:
                import shutil
                shutil.rmtree(graphify_out)
            except OSError:
                pass
        snapshot_path = Path(corpus.path) / f"snapshot-{corpus.name}.json"
        if snapshot_path.exists():
            try:
                snapshot_path.unlink()
            except OSError:
                pass
