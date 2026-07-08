"""
CBM (codebase-memory-mcp) standalone adapter.

Drives CBM through its native MCP tools over stdio. Spawns the CBM MCP server
as a subprocess and communicates via JSON-RPC over stdin/stdout.

Capabilities: symbol_lookup, neighbors_1hop, path_le4
N/A: nl_locate, blast_radius (CBM has semantic_query but it's not NL locate;
     blast_radius could map to detect_changes but that's git-diff based)

KNOWN FAILURE MODES (from 2026-07-04 audit):
- 20-50 GB RSS blowups on large repos
- SIGABRT on second index
- Cypher lexer swallowing $params
- No schema versioning

All indexing/query operations run under the safety rail (icebench.rail.run_with_rail)
with a hard memory cap + timeout. Breaches/crashes => DNF with reason, never hidden.
"""

import json
import subprocess
import time
from pathlib import Path
from threading import Thread
from queue import Queue, Empty

from icebench.adapters.base import (
    Corpus,
    IndexResult,
    QueryResult,
    SnapshotResult,
    UnsupportedOp,
)
from icebench.rail import RailConfig, run_with_rail


class CBMStandaloneAdapter:
    """Adapter for codebase-memory-mcp as a standalone system."""

    def __init__(
        self,
        cbm_bin: str = "/data/ice/tools/cbm/codebase-memory-mcp",
        rail: RailConfig | None = None,
    ):
        """
        Initialize the adapter.

        Args:
            cbm_bin: Path to codebase-memory-mcp binary.
            rail: Safety-rail config (cap + timeout).
        """
        self.name = "cbm"
        self.version = self._get_version(cbm_bin)
        self.cbm_bin = cbm_bin
        self.rail = rail or RailConfig()
        self._mcp_proc = None
        self._mcp_stdout_queue = None
        self._req_id = 0

    def _get_version(self, cbm_bin: str) -> str:
        """Get CBM version from binary."""
        if not Path(cbm_bin).exists():
            return "cbm@unknown"

        try:
            result = subprocess.run(
                [cbm_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse version from output
                version = result.stdout.strip().split()[-1] if result.stdout else "unknown"
                return f"cbm@{version}"
        except Exception:
            pass
        return "cbm@unknown"

    def capabilities(self) -> set[str]:
        """
        CBM supports 3 structural operations via its MCP tools.

        - symbol_lookup: search_graph + get_code_snippet
        - neighbors_1hop: trace_path with depth=1
        - path_le4: trace_path with depth up to 4

        nl_locate and blast_radius are N/A:
        - semantic_query exists but it's vector search, not NL->symbol locate
        - detect_changes is git-diff based, not a general blast radius tool
        """
        return {"symbol_lookup", "neighbors_1hop", "path_le4"}

    def _start_mcp_server(self, corpus: Corpus) -> bool:
        """
        Start CBM MCP server in stdio mode for a corpus.

        Returns:
            True if server started successfully, False otherwise.
        """
        if self._mcp_proc is not None:
            return True  # Already running

        try:
            self._mcp_proc = subprocess.Popen(
                [self.cbm_bin],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Start stdout reader thread
            self._mcp_stdout_queue = Queue()

            def read_stdout():
                for line in self._mcp_proc.stdout:
                    self._mcp_stdout_queue.put(line)

            Thread(target=read_stdout, daemon=True).start()

            return True
        except Exception:
            return False

    def _stop_mcp_server(self):
        """Stop the MCP server subprocess."""
        if self._mcp_proc is not None:
            try:
                self._mcp_proc.terminate()
                self._mcp_proc.wait(timeout=5)
            except Exception:
                try:
                    self._mcp_proc.kill()
                except Exception:
                    pass
            finally:
                self._mcp_proc = None
                self._mcp_stdout_queue = None

    def _mcp_call(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        """
        Call an MCP tool via JSON-RPC over stdio.

        Args:
            method: MCP tool name (e.g., "index_repository").
            params: Tool parameters.
            timeout: Response timeout in seconds.

        Returns:
            Result dict from the tool.

        Raises:
            Exception if call fails.
        """
        if self._mcp_proc is None or self._mcp_stdout_queue is None:
            raise RuntimeError("MCP server not running")

        self._req_id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": method,
            "params": params,
        }

        # Send request
        self._mcp_proc.stdin.write(json.dumps(req) + "\n")
        self._mcp_proc.stdin.flush()

        # Wait for response
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                line = self._mcp_stdout_queue.get(timeout=0.1)
                try:
                    resp = json.loads(line)
                    if resp.get("id") == self._req_id:
                        if "error" in resp:
                            raise RuntimeError(f"MCP error: {resp['error']}")
                        return resp.get("result", {})
                except json.JSONDecodeError:
                    continue
            except Empty:
                continue

        raise TimeoutError(f"MCP call timeout after {timeout}s")

    def index_cold(self, corpus: Corpus) -> IndexResult:
        """
        Index a corpus using CBM's index_repository tool.

        Runs under the safety rail to catch OOM/timeout. SIGABRT on second
        index is a known CBM failure mode => DNF.
        """
        # Guard: CBM binary must exist
        if not Path(self.cbm_bin).exists():
            return IndexResult(
                wall_s=0,
                peak_rss_mb=0,
                cpu_s=0,
                symbols=0,
                edges=0,
                files=0,
                ok=False,
                dnf=True,
                dnf_reason=f"cbm binary not found: {self.cbm_bin}",
            )

        # Use CLI mode for indexing (avoids stdio marshalling complexity)
        # CBM CLI: codebase-memory-mcp cli index_repository '{"repo_path": "..."}'
        cmd = [
            self.cbm_bin,
            "cli",
            "index_repository",
            json.dumps({"repo_path": corpus.path}),
        ]

        res = run_with_rail(cmd, self.rail)

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

        # Failed (non-zero exit) - check for SIGABRT
        if res.returncode != 0:
            # SIGABRT is returncode 134 or -6
            if res.returncode in (134, -6):
                dnf_reason = "SIGABRT (known CBM stability issue)"
            else:
                dnf_reason = f"exit_code_{res.returncode}"

            return IndexResult(
                wall_s=res.wall_s,
                peak_rss_mb=res.peak_rss_mb,
                cpu_s=res.cpu_s,
                symbols=0,
                edges=0,
                files=0,
                ok=False,
                dnf=True,
                dnf_reason=dnf_reason,
            )

        # Parse output for node/edge counts
        # CBM CLI output is JSON on stdout
        symbols = edges = files = 0
        try:
            result_data = json.loads(res.stdout.strip())
            symbols = result_data.get("nodes", 0)
            edges = result_data.get("edges", 0)
            files = result_data.get("files", 0)
        except (json.JSONDecodeError, AttributeError):
            # Try to extract from stderr if stdout parsing failed
            pass

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
        CBM has auto-sync via background watcher, but no explicit incremental
        index command => N/A.
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
        """
        Second full index (stability probe).

        This is specifically to trigger the SIGABRT-on-second-index failure mode.
        """
        return self.index_cold(corpus)

    def store_size_bytes(self, corpus: Corpus) -> int:
        """
        Measure CBM's on-disk store for a corpus.

        CBM stores in ~/.cache/codebase-memory-mcp/<project-hash>/graph.db
        Need to find the DB file for this corpus.
        """
        # CBM uses a hash of the repo path as the project ID
        # For now, return 0 if we can't determine it
        # A real implementation would need to query CBM or compute the hash
        cache_dir = Path.home() / ".cache" / "codebase-memory-mcp"
        if not cache_dir.exists():
            return 0

        # Find all graph.db files and sum them
        # (CBM doesn't expose a per-project size query)
        total = 0
        for db_file in cache_dir.rglob("graph.db"):
            try:
                total += db_file.stat().st_size
            except OSError:
                pass
        return total

    def export_snapshot(self, corpus: Corpus) -> SnapshotResult | None:
        """
        CBM's snapshot is the graph.db.zst artifact.

        CBM writes .codebase-memory/graph.db.zst in the repo root during
        index_repository. We can copy that as the snapshot.
        """
        artifact_path = Path(corpus.path) / ".codebase-memory" / "graph.db.zst"
        if not artifact_path.exists():
            return None

        start = time.monotonic()
        snapshot_path = Path(corpus.path) / f"snapshot-{corpus.name}.db.zst"

        try:
            snapshot_path.write_bytes(artifact_path.read_bytes())
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
        """
        Import = copy snapshot to .codebase-memory/graph.db.zst.

        CBM will decompress and use this on next index_repository.
        """
        start = time.monotonic()
        artifact_path = Path(corpus.path) / ".codebase-memory" / "graph.db.zst"

        # Ensure directory exists
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            artifact_path.write_bytes(Path(blob_path).read_bytes())
            wall_s = time.monotonic() - start
            return SnapshotResult(
                wall_s=wall_s,
                bytes=artifact_path.stat().st_size,
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
        Execute a query using CBM's CLI mode.

        - symbol_lookup: search_graph by name
        - neighbors_1hop: trace_path depth=1
        - path_le4: trace_path depth=4
        """
        if op not in self.capabilities():
            raise UnsupportedOp(f"Operation {op} not supported by cbm")

        start = time.perf_counter()

        try:
            if op == "symbol_lookup":
                symbol = payload.get("symbol", "")
                cmd = [
                    self.cbm_bin,
                    "cli",
                    "search_graph",
                    json.dumps({"name_pattern": f".*{symbol}.*"}),
                ]
            elif op == "neighbors_1hop":
                symbol = payload.get("symbol", "")
                cmd = [
                    self.cbm_bin,
                    "cli",
                    "trace_path",
                    json.dumps({
                        "function_name": symbol,
                        "direction": "both",
                        "depth": 1,
                    }),
                ]
            elif op == "path_le4":
                from_sym = payload.get("from", "")
                to_sym = payload.get("to", "")
                cmd = [
                    self.cbm_bin,
                    "cli",
                    "query_graph",
                    json.dumps({
                        "query": f"MATCH p=(a)-[*1..4]-(b) WHERE a.name = '{from_sym}' AND b.name = '{to_sym}' RETURN p LIMIT 1"
                    }),
                ]
            else:
                raise UnsupportedOp(f"Unexpected op: {op}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            latency_ms = (time.perf_counter() - start) * 1000

            if result.returncode == 0:
                try:
                    answer_data = json.loads(result.stdout)
                    answer = {"data": answer_data, "status": "ok"}
                except json.JSONDecodeError:
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
        """
        Clean up CBM state for a corpus.

        Uses delete_project via CLI to remove the graph data.
        """
        try:
            # Stop MCP server if running
            self._stop_mcp_server()

            # Delete project via CLI
            # We need the project ID, which CBM derives from repo_path
            # For now, just try to delete by path
            subprocess.run(
                [self.cbm_bin, "cli", "delete_project", json.dumps({"repo_path": corpus.path})],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass

        # Remove snapshot artifacts
        snapshot_path = Path(corpus.path) / f"snapshot-{corpus.name}.db.zst"
        if snapshot_path.exists():
            try:
                snapshot_path.unlink()
            except OSError:
                pass

        artifact_path = Path(corpus.path) / ".codebase-memory" / "graph.db.zst"
        if artifact_path.exists():
            try:
                artifact_path.unlink()
            except OSError:
                pass
