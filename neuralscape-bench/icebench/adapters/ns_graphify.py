"""
NS-Graphify adapter: graphify CLI -> graph.json -> NS ingest -> REST tools.

Uses the graphify CLI (H2 pins it) to produce graph.json, ingests into NS,
then answers queries via REST code-graph tools.

Capabilities: symbol_lookup, neighbors_1hop, path_le4
N/A: nl_locate, blast_radius (GraphifyJsonEngine raises NotSupported)
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from icebench.adapters.base import (
    SystemAdapter,
    Corpus,
    IndexResult,
    QueryResult,
    SnapshotResult,
    UnsupportedOp,
)


class NSGraphifyAdapter:
    """Adapter for NS using graphify CLI + GraphifyJsonEngine."""

    def __init__(
        self,
        api_url: str = "http://localhost:8499",
        graphify_bin: str = "/data/ice/tools/graphify",
    ):
        """
        Initialize the adapter.

        Args:
            api_url: NS API base URL.
            graphify_bin: Path to graphify binary.
        """
        self.name = "ns-graphify"
        self.version = "graphify-cli@TODO+ns-api@TODO"  # H2 pins versions
        self.api_url = api_url
        self.graphify_bin = graphify_bin
        self.client = httpx.Client(timeout=120.0)
        self._graph_ids: dict[str, str] = {}  # corpus.name -> graph_id

    def capabilities(self) -> set[str]:
        """Graphify JSON engine supports 3 structural ops."""
        return {"symbol_lookup", "neighbors_1hop", "path_le4"}

    def index_cold(self, corpus: Corpus) -> IndexResult:
        """Run graphify CLI + ingest graph.json."""
        # ICE-INTEGRATE: Guard graphify binary existence
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

        start = time.time()

        # Run graphify CLI on the corpus
        graph_json_path = Path(corpus.path) / "graph.json"
        try:
            result = subprocess.run(
                [self.graphify_bin, "analyze", corpus.path],
                cwd=corpus.path,
                capture_output=True,
                text=True,
                check=True,
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

        # Parse graph.json to count symbols/edges
        if not graph_json_path.exists():
            return IndexResult(
                wall_s=time.time() - start,
                peak_rss_mb=0,
                cpu_s=0,
                symbols=0,
                edges=0,
                files=0,
                ok=False,
            )

        with open(graph_json_path) as f:
            graph_data = json.load(f)

        symbols = len(graph_data.get("nodes", []))
        edges = len(graph_data.get("edges", []))

        # Ingest graph.json into NS
        with open(graph_json_path, "rb") as f:
            resp = self.client.post(
                f"{self.api_url}/v1/ingest/files",
                files={"files": ("graph.json", f, "application/json")},
                data={
                    "category": "domain_knowledge",
                    "project_id": f"ice-bench-{corpus.name}",
                },
            )

        if resp.status_code != 202:
            return IndexResult(
                wall_s=time.time() - start,
                peak_rss_mb=0,
                cpu_s=0,
                symbols=symbols,
                edges=edges,
                files=0,
                ok=False,
            )

        # Store the graph_id for queries
        task_data = resp.json()
        # TODO: Poll for task completion and extract graph_id
        # For now, assume graph_id is based on project_id
        self._graph_ids[corpus.name] = f"ice-bench-{corpus.name}"

        wall_s = time.time() - start
        return IndexResult(
            wall_s=wall_s,
            peak_rss_mb=0,  # TODO: Track via psutil
            cpu_s=0,
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
        """Measure graph.json size."""
        graph_json_path = Path(corpus.path) / "graph.json"
        if graph_json_path.exists():
            return graph_json_path.stat().st_size
        return 0

    def export_snapshot(self, corpus: Corpus) -> SnapshotResult | None:
        """Snapshot = copy graph.json."""
        graph_json_path = Path(corpus.path) / "graph.json"
        if not graph_json_path.exists():
            return None

        start = time.time()
        snapshot_path = Path(corpus.path) / f"snapshot-{corpus.name}.json"
        snapshot_path.write_bytes(graph_json_path.read_bytes())
        wall_s = time.time() - start

        return SnapshotResult(
            wall_s=wall_s,
            bytes=snapshot_path.stat().st_size,
            ok=True,
        )

    def import_snapshot(self, corpus: Corpus, blob_path: str) -> SnapshotResult | None:
        """Import = copy snapshot to graph.json."""
        start = time.time()
        graph_json_path = Path(corpus.path) / "graph.json"
        graph_json_path.write_bytes(Path(blob_path).read_bytes())
        wall_s = time.time() - start

        return SnapshotResult(
            wall_s=wall_s,
            bytes=graph_json_path.stat().st_size,
            ok=True,
        )

    def query(self, op: str, payload: dict) -> QueryResult:
        """Execute a query via NS REST code-graph tools."""
        if op not in self.capabilities():
            raise UnsupportedOp(f"Operation {op} not supported by ns-graphify")

        graph_id = payload.get("graph_id") or self._graph_ids.get(payload.get("corpus"))

        start = time.perf_counter()

        if op == "symbol_lookup":
            resp = self.client.get(
                f"{self.api_url}/v1/code-graph/query",
                params={
                    "query": payload["symbol"],
                    "graph_id": graph_id,
                },
            )
        elif op == "neighbors_1hop":
            resp = self.client.get(
                f"{self.api_url}/v1/code-graph/neighbors",
                params={
                    "symbol": payload["symbol"],
                    "graph_id": graph_id,
                },
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

    def teardown(self, corpus: Corpus) -> None:
        """Clean up graph.json."""
        graph_json_path = Path(corpus.path) / "graph.json"
        if graph_json_path.exists():
            graph_json_path.unlink()
