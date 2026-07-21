"""
NS-ICE adapter: native index CLI -> REST code-graph tools.

Uses E7's native index CLI and I3's snapshot_cli for all operations.
Supports all 5 op classes.

Capabilities: symbol_lookup, neighbors_1hop, path_le4, nl_locate, blast_radius

Every index/snapshot subprocess is routed through the safety rail
(icebench.rail.run_with_rail) so peak RSS / CPU-seconds are MEASURED and a
memory/timeout breach becomes a DNF row rather than a crash.
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
from icebench.rail import RailConfig, RailResult, run_with_rail
from icebench.util import dir_size_bytes


# Substrings that indicate the E7/I3 CLI module is not yet merged.
_MISSING_MODULE_MARKERS = ("No module named", "cannot find module")

# Default bind-mounted stack root (see docker-compose.ice.yml).
DEFAULT_STACK_DIR = "/data/ice/stack"


def _corpus_name(payload: dict) -> str | None:
    """Normalize a query payload's corpus reference to a name string."""
    c = payload.get("corpus")
    if isinstance(c, Corpus):
        return c.name
    return c  # str or None


class NSIceAdapter:
    """Adapter for NS using native code-intel engine."""

    def __init__(
        self,
        api_url: str = "http://localhost:8599",
        rail: RailConfig | None = None,
        stack_dir: str | None = None,
        service_dir: str | None = None,
    ):
        """
        Initialize the adapter.

        The index/snapshot CLIs run on the HOST (via `uv run`) against the ice
        stack's EXPOSED ports, so the safety rail's `/usr/bin/time -v` measures
        the indexer's own peak RSS / CPU — directly comparable to the host-run
        competitors (graphify/CBM). Queries go through the REST API (the native
        surface). Both hit the same ice Neo4j/Qdrant.

        Args:
            api_url: NS API base URL (host→container mapped, default 8599).
            rail: Safety-rail config (cap + timeout).
            stack_dir: Bind-mounted stack root for store-size measurement.
            service_dir: neuralscape-service dir the index CLI runs from
                (`uv run` resolves its venv there).
        """
        self.name = "ns-ice"
        self.api_url = api_url
        self.rail = rail or RailConfig()
        self.stack_dir = stack_dir or os.environ.get("ICE_STACK_DIR", DEFAULT_STACK_DIR)
        self.service_dir = Path(
            service_dir
            or os.environ.get("ICE_SERVICE_DIR", "/data/ice/neuralscape/neuralscape-service")
        )
        self.client = httpx.Client(timeout=120.0)
        # Env for host-run index/snapshot CLIs → the ice stack's exposed ports.
        # GOOGLE_API_KEY + NEO4J_PASSWORD inherit from the runner's environment
        # (sourced from /data/ice/.env at launch).
        self._cli_env = {
            **os.environ,
            "NEO4J_URI": os.environ.get("ICE_NEO4J_URI", "neo4j://localhost:7787"),
            "NEO4J_DATABASE": os.environ.get("ICE_NEO4J_DATABASE", "memory"),
            "QDRANT_URL": os.environ.get("ICE_QDRANT_URL", "http://localhost:6433"),
            "REDIS_URL": os.environ.get("ICE_REDIS_URL", "redis://localhost:6479"),
        }
        self.version = self._read_version()

    def _read_version(self) -> str:
        """Best-effort NS API version from /health (falls back to a marker)."""
        try:
            r = self.client.get(f"{self.api_url}/health", timeout=5.0)
            if r.status_code == 200:
                return f"ns-ice@{r.json().get('service', 'neuralscape')}"
        except Exception:
            pass
        return "ns-ice@unknown"

    def _index_cmd(self, corpus: Corpus, incremental: bool) -> list[str]:
        """Host `uv run` invocation of the native index CLI (rail-measured)."""
        cmd = [
            "uv", "run", "python", "-m", "adapters.code_graph.native_index_cli",
            "--repo-path", corpus.path,
            "--code-space", self._make_code_space(corpus.name),
        ]
        if incremental:
            cmd.append("--incremental")
        return cmd

    def capabilities(self) -> set[str]:
        """NS-ICE supports all 5 op classes."""
        return {
            "symbol_lookup",
            "neighbors_1hop",
            "path_le4",
            "nl_locate",
            "blast_radius",
        }

    def _make_code_space(self, corpus_name: str) -> str:
        """Generate a code_space identifier for a corpus name."""
        return f"code--ice-bench--{corpus_name}"

    def _index_result_from_rail(self, res: RailResult, integrate_hint: str) -> IndexResult:
        """
        Map a RailResult from the native index CLI into an IndexResult.

        Parses the CLI's JSON summary line for symbol/edge/file counts; carries
        rail-measured peak_rss_mb / cpu_s; classifies DNF and missing-module.
        """
        # Rail breach (OOM / timeout) => DNF.
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

        # Missing E7/I3 module => clear DNF, not a crash.
        if res.returncode != 0 and any(m in res.stderr for m in _MISSING_MODULE_MARKERS):
            return IndexResult(
                wall_s=res.wall_s,
                peak_rss_mb=res.peak_rss_mb,
                cpu_s=res.cpu_s,
                symbols=0,
                edges=0,
                files=0,
                ok=False,
                dnf=True,
                dnf_reason=f"module-missing ({integrate_hint})",
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

        # Parse the CLI's JSON summary line (last non-empty stdout line).
        symbols = edges = files = 0
        wall_s = res.wall_s
        for line in reversed(res.stdout.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                symbols = int(data.get("symbols", 0))
                edges = int(data.get("edges", 0))
                files = int(data.get("files", 0))
                wall_s = float(data.get("wall_s", res.wall_s))
            except (ValueError, TypeError):
                pass
            break

        return IndexResult(
            wall_s=wall_s,
            peak_rss_mb=res.peak_rss_mb,
            cpu_s=res.cpu_s,
            symbols=symbols,
            edges=edges,
            files=files,
            ok=True,
        )

    def index_cold(self, corpus: Corpus) -> IndexResult:
        """Run native index CLI (cold/full) on the host under the safety rail."""
        res = run_with_rail(
            self._index_cmd(corpus, incremental=False),
            self.rail, cwd=self.service_dir, env=self._cli_env,
        )
        return self._index_result_from_rail(res, "E7 native_index_cli")

    def index_incremental(self, corpus: Corpus, touched: list[str]) -> IndexResult:
        """Run native index CLI with --incremental on the host under the rail."""
        res = run_with_rail(
            self._index_cmd(corpus, incremental=True),
            self.rail, cwd=self.service_dir, env=self._cli_env,
        )
        return self._index_result_from_rail(res, "E7 native_index_cli --incremental")

    def index_second(self, corpus: Corpus) -> IndexResult:
        """Second full index (stability probe)."""
        return self.index_cold(corpus)

    def store_size_bytes(self, corpus: Corpus) -> int:
        """
        Measure the NS ICE on-disk store footprint.

        NS uses SHARED services (Neo4j + Qdrant), so this measures the whole
        bind-mounted stack under ``stack_dir`` (neo4j/data + qdrant). The runner
        should diff this around teardown for a per-corpus delta. Documented as a
        whole-store measurement, not a per-code_space isolate.
        """
        total = 0
        for sub in ("neo4j/data", "qdrant"):
            total += dir_size_bytes(Path(self.stack_dir) / sub)
        return total

    def export_snapshot(self, corpus: Corpus) -> SnapshotResult | None:
        """Export snapshot via snapshot_cli under the safety rail."""
        # ICE-INTEGRATE: I3-fixed adapters.code_graph.snapshot_cli
        code_space = self._make_code_space(corpus.name)
        snapshot_path = Path(corpus.path) / f"snapshot-{corpus.name}.bin"
        cmd = [
            "uv", "run", "python", "-m", "adapters.code_graph.snapshot_cli",
            "export", "--code-space", code_space, "--output", str(snapshot_path),
        ]
        res = run_with_rail(cmd, self.rail, cwd=self.service_dir, env=self._cli_env)

        if res.dnf:
            return SnapshotResult(
                wall_s=res.wall_s, bytes=0, ok=False, dnf=True, dnf_reason=res.dnf_reason
            )
        if res.returncode != 0 and any(m in res.stderr for m in _MISSING_MODULE_MARKERS):
            return SnapshotResult(
                wall_s=res.wall_s,
                bytes=0,
                ok=False,
                dnf=True,
                dnf_reason="module-missing (I3 snapshot_cli)",
            )
        if res.returncode != 0:
            return SnapshotResult(wall_s=res.wall_s, bytes=0, ok=False)

        size = snapshot_path.stat().st_size if snapshot_path.exists() else 0
        return SnapshotResult(wall_s=res.wall_s, bytes=size, ok=True)

    def import_snapshot(self, corpus: Corpus, blob_path: str) -> SnapshotResult | None:
        """Import snapshot via snapshot_cli under the safety rail."""
        # ICE-INTEGRATE: I3-fixed adapters.code_graph.snapshot_cli
        code_space = self._make_code_space(corpus.name)
        cmd = [
            "uv", "run", "python", "-m", "adapters.code_graph.snapshot_cli",
            "import", "--code-space", code_space, "--input", blob_path,
        ]
        res = run_with_rail(cmd, self.rail, cwd=self.service_dir, env=self._cli_env)

        if res.dnf:
            return SnapshotResult(
                wall_s=res.wall_s, bytes=0, ok=False, dnf=True, dnf_reason=res.dnf_reason
            )
        if res.returncode != 0 and any(m in res.stderr for m in _MISSING_MODULE_MARKERS):
            return SnapshotResult(
                wall_s=res.wall_s,
                bytes=0,
                ok=False,
                dnf=True,
                dnf_reason="module-missing (I3 snapshot_cli)",
            )
        if res.returncode != 0:
            return SnapshotResult(wall_s=res.wall_s, bytes=0, ok=False)

        size = Path(blob_path).stat().st_size if Path(blob_path).exists() else 0
        return SnapshotResult(wall_s=res.wall_s, bytes=size, ok=True)

    def query(self, op: str, payload: dict) -> QueryResult:
        """Execute a query via NS REST code-graph tools."""
        if op not in self.capabilities():
            raise UnsupportedOp(f"Operation {op} not supported by ns-ice")

        name = _corpus_name(payload)
        code_space = self._make_code_space(name) if name else None

        start = time.perf_counter()

        try:
            if op == "symbol_lookup":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/query",
                    params={"question": payload["symbol"], "graph_id": code_space},
                )
            elif op == "neighbors_1hop":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/neighbors",
                    params={"label": payload["symbol"], "graph_id": code_space},
                )
            elif op == "path_le4":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/path",
                    params={
                        "source": payload["from"],
                        "target": payload["to"],
                        "max_hops": 4,
                        "graph_id": code_space,
                    },
                )
            elif op == "nl_locate":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/locate",
                    params={"query": payload["query"], "graph_id": code_space},
                )
            elif op == "blast_radius":
                # ICE-INTEGRATE: E7 adds GET /v1/code-graph/impact
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
        Delete the code_space's graph state so shared services don't accumulate.

        ICE-INTEGRATE: uses the code-graph delete route if present; tolerates its
        absence (best-effort — never raises).
        """
        code_space = self._make_code_space(corpus.name)
        try:
            self.client.delete(
                f"{self.api_url}/v1/code-graph/graph",
                params={"graph_id": code_space},
            )
        except httpx.RequestError:
            pass
        # Remove any local snapshot artifact.
        snapshot_path = Path(corpus.path) / f"snapshot-{corpus.name}.bin"
        if snapshot_path.exists():
            try:
                snapshot_path.unlink()
            except OSError:
                pass
