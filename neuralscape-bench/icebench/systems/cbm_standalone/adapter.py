"""
CBM (codebase-memory-mcp) standalone adapter.

Drives CBM through its native `cli <tool> <json>` surface (its single static
binary). Each ICEBench op maps to one CBM tool call; the tool's JSON response
is parsed as-is. No added intelligence (no rewriting, no fallback beyond parsing
CBM's own answer).

Verified against codebase-memory-mcp 0.9.0 (DeusData/codebase-memory-mcp, MIT)
on this VM. Real CLI surface (stdout is clean JSON; logs go to stderr):
  index:  `cbm cli index_repository '{"repo_path": "<path>"}'`
          -> {"project": "<slug>", "nodes": N, "edges": M, "status": "indexed"}
  list:   `cbm cli list_projects '{}'`
          -> {"projects": [{"name","root_path","nodes","edges","size_bytes"}]}
  query:  `cbm cli search_graph '{"project": P, "name_pattern": "..."}'`   (symbol)
          `cbm cli trace_path  '{"project": P, "function_name": F,
                                 "direction": "both", "depth": 1}'`         (neighbors)
          `cbm cli query_graph '{"project": P, "query": "<cypher>"}'`       (path)
  delete: `cbm cli delete_project '{"project": P}'`
Every query tool REQUIRES a "project" argument (the slug from list_projects);
we resolve it by matching a project's root_path to the corpus path.

Capabilities: symbol_lookup, neighbors_1hop, path_le4
N/A: nl_locate (CBM's semantic_query is vector search over code chunks, not
     NL->symbol resolution), blast_radius (CBM's detect_changes is git-diff
     based, not general change-propagation impact analysis).

KNOWN FAILURE MODES (2026-07-04 audit — treated as DATA, never engineered
around): 20-50 GB RSS blowups on large repos, SIGABRT on a second index, Cypher
lexer fragility (it rejects `$params` and `p=(...)` path assignment). Path
queries therefore avoid `$params` (unsupported) and single-quote-escape
interpolated names as the best available injection defense.

EVERY CBM subprocess — index, list, query, delete — runs through the safety rail
(icebench.rail.run_with_rail) with the hard memory cap + wall-timeout so an
OOM/timeout breach or SIGABRT crash is recorded as DNF instead of taking down
the shared VM. index_second is specifically the SIGABRT stability probe.
"""

import json
import os
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


DEFAULT_CBM_BIN = "/data/ice/tools/cbm/codebase-memory-mcp"
DEFAULT_CBM_CACHE_DIR = "/data/ice/tools/cbm_cache"


def _cypher_quote(value: str) -> str:
    """
    Escape a string for safe interpolation into a single-quoted Cypher literal.

    CBM's Cypher lexer rejects `$params` (a known limitation), so parameterized
    queries aren't available. Doubling single quotes is the standard string-
    literal escape and the best available defense against injection here.
    """
    return value.replace("'", "''")


class CBMStandaloneAdapter:
    """Adapter for codebase-memory-mcp as a standalone system."""

    def __init__(
        self,
        cbm_bin: str = DEFAULT_CBM_BIN,
        cache_dir: str | None = None,
        rail: RailConfig | None = None,
    ):
        """
        Initialize the adapter.

        Args:
            cbm_bin: Path to the codebase-memory-mcp binary.
            cache_dir: CBM cache dir (its SQLite stores live here). Kept under
                /data/ice by default so nothing lands on the tight root fs.
            rail: Safety-rail config (cap + timeout).
        """
        self.name = "cbm"
        self.cbm_bin = cbm_bin
        self.cache_dir = cache_dir or os.environ.get(
            "CBM_CACHE_DIR", DEFAULT_CBM_CACHE_DIR
        )
        self.rail = rail or RailConfig()
        # Cache resolved CBM project slugs keyed by corpus name.
        self._project_names: dict[str, str] = {}
        self.version = self._get_version()

    # ---- helpers ----

    def _env(self) -> dict:
        """Env for CBM subprocesses: pin the cache dir + quiet the logs."""
        env = dict(os.environ)
        env["CBM_CACHE_DIR"] = self.cache_dir
        env["CBM_LOG_LEVEL"] = "none"  # keep stderr clean; stdout stays JSON
        return env

    def _get_version(self) -> str:
        """
        Read the CBM version via `cbm --version` (bounded metadata probe, not an
        index/query workload).
        """
        if not Path(self.cbm_bin).exists():
            return "cbm@unknown"
        try:
            result = subprocess.run(
                [self.cbm_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                env=self._env(),
            )
            if result.returncode == 0 and result.stdout.strip():
                # Output: "codebase-memory-mcp 0.9.0"
                return f"cbm@{result.stdout.strip().split()[-1]}"
        except Exception:
            pass
        return "cbm@unknown"

    def _cli(self, tool: str, args: dict) -> RailResult:
        """Run one `cbm cli <tool> <json>` under the safety rail."""
        cmd = [self.cbm_bin, "cli", tool, json.dumps(args)]
        return run_with_rail(cmd, self.rail, env=self._env())

    @staticmethod
    def _parse_json(res: RailResult) -> dict | None:
        """Parse a CBM CLI stdout payload as JSON (stdout is clean JSON)."""
        try:
            return json.loads(res.stdout.strip())
        except (json.JSONDecodeError, AttributeError):
            return None

    def _list_projects(self) -> list[dict]:
        """Return CBM's indexed-project records (via list_projects, rail-run)."""
        res = self._cli("list_projects", {})
        if res.dnf or res.returncode != 0:
            return []
        data = self._parse_json(res)
        if not data:
            return []
        return data.get("projects", [])

    def _resolve_project(self, corpus: Corpus) -> str | None:
        """
        Resolve CBM's project slug for a corpus by matching root_path.

        Prefers the cached slug (learned at index time); otherwise queries
        list_projects and matches on the real (symlink-resolved) corpus path.
        This avoids guessing CBM's path->slug transformation.
        """
        cached = self._project_names.get(corpus.name)
        if cached:
            return cached

        target = os.path.realpath(corpus.path)
        for proj in self._list_projects():
            root = proj.get("root_path")
            if root and os.path.realpath(root) == target:
                name = proj.get("name")
                if name:
                    self._project_names[corpus.name] = name
                    return name
        return None

    # ---- capabilities ----

    def capabilities(self) -> set[str]:
        """
        CBM supports 4 operations via its MCP/CLI tools (was wrongly 3 in ice-final).

        - symbol_lookup: search_graph (name_pattern)
        - neighbors_1hop: trace_path (depth=1, direction=both)
        - path_le4: query_graph (Cypher variable-length path [*1..4])
        - nl_locate: search_code (semantic vector search over code chunks)

        blast_radius is N/A (detect_changes is git-diff based, not general impact).
        """
        return {"symbol_lookup", "neighbors_1hop", "path_le4", "nl_locate"}

    # ---- indexing ----

    def _index(self, corpus: Corpus) -> IndexResult:
        """Run index_repository under the rail; classify DNF/SIGABRT."""
        if not Path(self.cbm_bin).exists():
            return IndexResult(
                wall_s=0, peak_rss_mb=0, cpu_s=0, symbols=0, edges=0, files=0,
                ok=False, dnf=True,
                dnf_reason=f"cbm binary not found: {self.cbm_bin}",
            )

        res = self._cli("index_repository", {"repo_path": corpus.path})

        # Rail breach (OOM / timeout) => DNF. CBM is the reason the rail exists.
        if res.dnf:
            return IndexResult(
                wall_s=res.wall_s, peak_rss_mb=res.peak_rss_mb, cpu_s=res.cpu_s,
                symbols=0, edges=0, files=0, ok=False, dnf=True,
                dnf_reason=res.dnf_reason,
            )

        if res.returncode != 0:
            # SIGABRT (134 / -6) is a known CBM stability failure => DNF.
            if res.returncode in (134, -6):
                dnf_reason = "SIGABRT (known CBM stability issue)"
            else:
                dnf_reason = f"exit_code_{res.returncode}"
            return IndexResult(
                wall_s=res.wall_s, peak_rss_mb=res.peak_rss_mb, cpu_s=res.cpu_s,
                symbols=0, edges=0, files=0, ok=False, dnf=True,
                dnf_reason=dnf_reason,
            )

        data = self._parse_json(res) or {}
        symbols = int(data.get("nodes", 0))
        edges = int(data.get("edges", 0))
        # CBM's index_repository summary reports nodes+edges but no file count,
        # so files is left at 0 (honest: the tool does not surface it here).
        files = 0

        # Cache the project slug CBM assigned, for later query/size/delete.
        proj = data.get("project")
        if proj:
            self._project_names[corpus.name] = proj

        return IndexResult(
            wall_s=res.wall_s, peak_rss_mb=res.peak_rss_mb, cpu_s=res.cpu_s,
            symbols=symbols, edges=edges, files=files, ok=True,
        )

    def index_cold(self, corpus: Corpus) -> IndexResult:
        """
        Cold (from-scratch) index of the corpus.

        Deletes any existing project first to ensure a true cold index (not a
        cache hit). This is critical for accurate cold-index benchmarking —
        CBM's caching made rep0 the only true cold measurement, with later reps
        being ~0.13s no-ops.
        """
        # Delete any existing project to force a true cold index.
        project = self._resolve_project(corpus)
        if project:
            try:
                self._cli("delete_project", {"project": project})
            except Exception:
                pass
            self._project_names.pop(corpus.name, None)

        return self._index(corpus)

    def index_incremental(self, corpus: Corpus, touched: list[str]) -> IndexResult:
        """
        CBM has a background auto-sync watcher but no explicit incremental
        index command in CLI mode => N/A (DNF with dnf_reason='incremental_na').
        """
        return IndexResult(
            wall_s=0, peak_rss_mb=0, cpu_s=0, symbols=0, edges=0, files=0,
            ok=False, dnf=True, dnf_reason="incremental_na",
        )

    def index_second(self, corpus: Corpus) -> IndexResult:
        """
        Second full index (stability probe).

        Specifically probes CBM's known SIGABRT-on-second-index failure mode;
        a crash is recorded as DNF by _index().
        """
        return self._index(corpus)

    def store_size_bytes(self, corpus: Corpus) -> int:
        """
        On-disk store size for THIS corpus only.

        CBM's list_projects reports a per-project size_bytes; we match the
        corpus by root_path and return that project's size. This scopes to the
        corpus's own store rather than summing every DB under the cache dir.
        Falls back to this corpus's <cache>/<slug>.db* files if the field is
        absent.
        """
        target = os.path.realpath(corpus.path)
        for proj in self._list_projects():
            root = proj.get("root_path")
            if root and os.path.realpath(root) == target:
                size = proj.get("size_bytes")
                if isinstance(size, int) and size > 0:
                    return size
                # Fall back to this project's own DB files (not a global sum).
                name = proj.get("name")
                if name:
                    return self._db_files_size(name)
        return 0

    def _db_files_size(self, project_name: str) -> int:
        """Sum this project's own SQLite store files (<slug>.db + wal/shm)."""
        cache = Path(self.cache_dir)
        total = 0
        for suffix in (".db", ".db-wal", ".db-shm"):
            f = cache / f"{project_name}{suffix}"
            if f.exists():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total

    # ---- snapshots ----

    def export_snapshot(self, corpus: Corpus) -> SnapshotResult | None:
        """
        Snapshot = copy CBM's own SQLite store for this corpus.

        CBM's documented portable artifact (.codebase-memory/graph.db.zst) is
        only emitted for git repos; its persistent store is <cache>/<slug>.db,
        which is what actually relocates an index. We copy that store file as
        the faithful snapshot. Returns None (N/A) if no store exists.
        """
        project_name = self._resolve_project(corpus)
        if not project_name:
            return None
        db_path = Path(self.cache_dir) / f"{project_name}.db"
        if not db_path.exists():
            return None

        start = time.monotonic()
        snapshot_path = Path(corpus.path) / f"snapshot-{corpus.name}.db"
        try:
            snapshot_path.write_bytes(db_path.read_bytes())
            return SnapshotResult(
                wall_s=time.monotonic() - start,
                bytes=snapshot_path.stat().st_size,
                ok=True,
            )
        except OSError:
            return SnapshotResult(wall_s=time.monotonic() - start, bytes=0, ok=False)

    def import_snapshot(self, corpus: Corpus, blob_path: str) -> SnapshotResult | None:
        """
        Import = restore a CBM store file into the cache under this corpus's slug.

        Requires the corpus to have been indexed (so a slug exists); otherwise
        returns None (N/A) rather than guessing CBM's path->slug transform.
        """
        project_name = self._resolve_project(corpus)
        if not project_name:
            return None

        start = time.monotonic()
        dest = Path(self.cache_dir) / f"{project_name}.db"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(Path(blob_path).read_bytes())
            return SnapshotResult(
                wall_s=time.monotonic() - start,
                bytes=dest.stat().st_size,
                ok=True,
            )
        except OSError:
            return SnapshotResult(wall_s=time.monotonic() - start, bytes=0, ok=False)

    # ---- queries ----

    def _query_result(self, res: RailResult, start: float) -> QueryResult:
        """Map a rail-run CBM CLI query into a QueryResult."""
        latency_ms = (time.perf_counter() - start) * 1000
        if res.dnf:
            return QueryResult(
                latency_ms=latency_ms,
                answer={"error": res.dnf_reason, "status": "dnf"},
                ok=False,
            )
        if res.returncode != 0:
            return QueryResult(
                latency_ms=latency_ms,
                answer={"error": res.stderr or res.stdout, "status": "error"},
                ok=False,
            )
        data = self._parse_json(res)
        if data is None:
            return QueryResult(
                latency_ms=latency_ms,
                answer={"text": res.stdout, "status": "ok"},
                ok=True,
            )
        # CBM signals a tool-level failure with an "error" key in its JSON.
        if isinstance(data, dict) and "error" in data:
            return QueryResult(
                latency_ms=latency_ms,
                answer={"data": data, "status": "error"},
                ok=False,
            )
        return QueryResult(
            latency_ms=latency_ms,
            answer={"data": data, "status": "ok"},
            ok=True,
        )

    def query(self, op: str, payload: dict) -> QueryResult:
        """
        Execute a query via CBM's own CLI tools (routed through the rail).

        payload must carry the target Corpus under "corpus" so we can resolve
        CBM's project slug (every CBM query tool requires a "project" argument).
        """
        if op not in self.capabilities():
            raise UnsupportedOp(f"Operation {op} not supported by cbm")

        corpus = payload.get("corpus")
        if not isinstance(corpus, Corpus):
            return QueryResult(
                latency_ms=0,
                answer={"error": "payload['corpus'] must be a Corpus", "status": "error"},
                ok=False,
            )

        project = self._resolve_project(corpus)
        if not project:
            return QueryResult(
                latency_ms=0,
                answer={"error": "corpus not indexed in CBM", "status": "error"},
                ok=False,
            )

        if op == "symbol_lookup":
            symbol = payload.get("symbol", "")
            args = {"project": project, "name_pattern": f".*{symbol}.*"}
            tool = "search_graph"
        elif op == "neighbors_1hop":
            args = {
                "project": project,
                "function_name": payload.get("symbol", ""),
                "direction": "both",
                "depth": 1,
            }
            tool = "trace_path"
        elif op == "path_le4":
            frm = _cypher_quote(payload.get("from", ""))
            to = _cypher_quote(payload.get("to", ""))
            # Verified working syntax on CBM 0.9.0: a bare (no `p=`) undirected
            # variable-length match with escaped literals. Returns a row iff a
            # path of length <=4 exists between the two named symbols.
            cypher = (
                f"MATCH (a)-[*1..4]-(b) WHERE a.name = '{frm}' "
                f"AND b.name = '{to}' RETURN b.name LIMIT 1"
            )
            args = {"project": project, "query": cypher}
            tool = "query_graph"
        elif op == "nl_locate":
            # `search_code` does semantic vector search over code chunks
            query = payload.get("query", "")
            args = {"project": project, "pattern": query}
            tool = "search_code"
        else:  # pragma: no cover - guarded by capabilities() check above
            raise UnsupportedOp(f"Unexpected op: {op}")

        start = time.perf_counter()
        res = self._cli(tool, args)
        return self._query_result(res, start)

    # ---- cleanup ----

    def teardown(self, corpus: Corpus) -> None:
        """Delete CBM's project (via delete_project, rail-run) + snapshots."""
        project = self._resolve_project(corpus)
        if project:
            try:
                self._cli("delete_project", {"project": project})
            except Exception:
                pass
            self._project_names.pop(corpus.name, None)

        snapshot_path = Path(corpus.path) / f"snapshot-{corpus.name}.db"
        if snapshot_path.exists():
            try:
                snapshot_path.unlink()
            except OSError:
                pass
