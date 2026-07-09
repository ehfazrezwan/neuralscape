"""
NS-Graphify adapter: graphify CLI -> graph.json -> NS ingest -> REST tools.

Uses the graphify CLI (H2 pins it) to produce graph.json, ingests into NS,
then answers queries via REST code-graph tools.

Capabilities: symbol_lookup, neighbors_1hop, path_le4
N/A: nl_locate, blast_radius (GraphifyJsonEngine raises NotSupported)

The graphify CLI (the heavy indexing step) is routed through the safety rail
(icebench.rail.run_with_rail) so peak RSS / CPU-seconds are MEASURED and a
memory/timeout breach becomes a DNF row rather than a crash. The subsequent
NS ingest is a local HTTP call whose wall time is added on top.
"""

import json
import os
import time
from pathlib import Path

import httpx

from icebench.adapters.base import (
    Corpus,
    IndexResult,
    QueryResult,
    SnapshotResult,
    UnsupportedOp,
)
from icebench.rail import RailConfig, run_with_rail


def _corpus_name(payload: dict) -> str | None:
    """Normalize a query payload's corpus reference to a name string."""
    c = payload.get("corpus")
    if isinstance(c, Corpus):
        return c.name
    return c  # str or None


class NSGraphifyAdapter:
    """Adapter for NS using graphify CLI + GraphifyJsonEngine."""

    def __init__(
        self,
        api_url: str | None = None,
        graphify_bin: str = "/data/ice/tools/graphify/.venv/bin/graphify",
        rail: RailConfig | None = None,
    ):
        """
        Initialize the adapter.

        Args:
            api_url: NS API base URL. If None, uses ICE_API_URL env var or
                defaults to http://localhost:8599.
            graphify_bin: Path to graphify binary (H2 pins it).
            rail: Safety-rail config (cap + timeout). Runner injects the
                CLI-configured one; defaults to RailConfig() otherwise.
        """
        self.name = "ns-graphify"
        self.version = "graphify-cli@TODO+ns-api@TODO"  # H2 pins versions
        self.api_url = api_url or os.environ.get("ICE_API_URL", "http://localhost:8599")
        self.graphify_bin = graphify_bin
        self.rail = rail or RailConfig()
        self.client = httpx.Client(timeout=120.0)
        # Keyed by corpus NAME (string) — Corpus is an unhashable dataclass.
        self._graph_ids: dict[str, str] = {}

    def capabilities(self) -> set[str]:
        """Graphify JSON engine supports 3 structural ops."""
        return {"symbol_lookup", "neighbors_1hop", "path_le4"}

    def _graph_json_path(self, corpus: Corpus) -> Path:
        """Path where graphify writes its graph.json for a corpus."""
        return Path(corpus.path) / "graphify-out" / "graph.json"

    def index_cold(self, corpus: Corpus) -> IndexResult:
        """Run graphify CLI (under the rail) + ingest graph.json into NS."""
        # ICE-INTEGRATE: guard graphify binary existence (H2 installs it)
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
                dnf_reason=f"graphify binary not found: {self.graphify_bin}",
            )

        overall_start = time.monotonic()

        # Heavy step: run graphify under the safety rail (measures RSS/CPU).
        res = run_with_rail(
            [self.graphify_bin, "extract", corpus.path, "--code-only", "--no-cluster"],
            self.rail,
            cwd=Path(corpus.path),
        )

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

        with open(graph_json_path) as f:
            graph_data = json.load(f)

        symbols = len(graph_data.get("nodes", []))
        edges = len(graph_data.get("edges", []))

        # Ingest graph.json into NS (local HTTP; wall added on top of CLI time).
        try:
            with open(graph_json_path, "rb") as f:
                resp = self.client.post(
                    f"{self.api_url}/v1/ingest/files",
                    files={"files": ("graph.json", f, "application/json")},
                    data={
                        "category": "domain_knowledge",
                        "project_id": f"ice-bench-{corpus.name}",
                    },
                )
        except httpx.RequestError as e:
            return IndexResult(
                wall_s=time.monotonic() - overall_start,
                peak_rss_mb=res.peak_rss_mb,
                cpu_s=res.cpu_s,
                symbols=symbols,
                edges=edges,
                files=len(graph_data.get("files", [])),
                ok=False,
            )

        if resp.status_code != 202:
            return IndexResult(
                wall_s=time.monotonic() - overall_start,
                peak_rss_mb=res.peak_rss_mb,
                cpu_s=res.cpu_s,
                symbols=symbols,
                edges=edges,
                files=len(graph_data.get("files", [])),
                ok=False,
            )

        # Store the graph_id (keyed by name) for queries.
        # TODO(smoke): poll the ingest task and read the real graph_id.
        self._graph_ids[corpus.name] = f"ice-bench-{corpus.name}"

        return IndexResult(
            wall_s=time.monotonic() - overall_start,
            peak_rss_mb=res.peak_rss_mb,
            cpu_s=res.cpu_s,
            symbols=symbols,
            edges=edges,
            files=len(graph_data.get("files", [])),
            ok=True,
        )

    def index_incremental(self, corpus: Corpus, touched: list[str]) -> IndexResult:
        """Graphify doesn't support incremental => re-run cold."""
        return self.index_cold(corpus)

    def index_second(self, corpus: Corpus) -> IndexResult:
        """Second full index (stability probe)."""
        return self.index_cold(corpus)

    def store_size_bytes(self, corpus: Corpus) -> int:
        """Measure graph.json size (graphify's entire on-disk store)."""
        graph_json_path = self._graph_json_path(corpus)
        if graph_json_path.exists():
            return graph_json_path.stat().st_size
        return 0

    def export_snapshot(self, corpus: Corpus) -> SnapshotResult | None:
        """Snapshot = copy graph.json."""
        graph_json_path = self._graph_json_path(corpus)
        if not graph_json_path.exists():
            return None

        start = time.monotonic()
        snapshot_path = Path(corpus.path) / f"snapshot-{corpus.name}.json"
        snapshot_path.write_bytes(graph_json_path.read_bytes())
        wall_s = time.monotonic() - start

        return SnapshotResult(
            wall_s=wall_s,
            bytes=snapshot_path.stat().st_size,
            ok=True,
        )

    def import_snapshot(self, corpus: Corpus, blob_path: str) -> SnapshotResult | None:
        """Import = copy snapshot to graph.json."""
        start = time.monotonic()
        graph_json_path = self._graph_json_path(corpus)
        graph_json_path.write_bytes(Path(blob_path).read_bytes())
        wall_s = time.monotonic() - start

        return SnapshotResult(
            wall_s=wall_s,
            bytes=graph_json_path.stat().st_size,
            ok=True,
        )

    def query(self, op: str, payload: dict) -> QueryResult:
        """Execute a query via NS REST code-graph tools."""
        if op not in self.capabilities():
            raise UnsupportedOp(f"Operation {op} not supported by ns-graphify")

        # Accept an explicit graph_id, else resolve by corpus name (string).
        name = _corpus_name(payload)
        graph_id = payload.get("graph_id") or (self._graph_ids.get(name) if name else None)

        start = time.perf_counter()

        try:
            if op == "symbol_lookup":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/query",
                    params={"query": payload["symbol"], "graph_id": graph_id},
                )
            elif op == "neighbors_1hop":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/neighbors",
                    params={"symbol": payload["symbol"], "graph_id": graph_id},
                )
            elif op == "path_le4":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/path",
                    params={
                        "from_symbol": payload["from"],
                        "to_symbol": payload["to"],
                        "max_depth": 4,
                        "graph_id": graph_id,
                    },
                )
            else:
                raise UnsupportedOp(f"Unexpected op: {op}")

            latency_ms = (time.perf_counter() - start) * 1000

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
        """Remove the local graph.json + snapshot and forget the graph_id."""
        graph_json_path = self._graph_json_path(corpus)
        if graph_json_path.exists():
            try:
                graph_json_path.unlink()
            except OSError:
                pass
        snapshot_path = Path(corpus.path) / f"snapshot-{corpus.name}.json"
        if snapshot_path.exists():
            try:
                snapshot_path.unlink()
            except OSError:
                pass
        self._graph_ids.pop(corpus.name, None)
