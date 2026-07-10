"""
NS-CBM adapter: CBM engine via NS REST code-graph routed endpoints.

Indexes and queries a corpus through the NS REST API using `knowledge_system=code-cbm`.
The indexer runs server-side on the ingest worker, so peak RSS / CPU measurements
are N/A (set to 0 with an explanatory docstring).

Capabilities: symbol_lookup, neighbors_1hop, nl_locate
N/A: path_le4 (CBM's Cypher path requires raw query, banned in routing),
     blast_radius (CBM has no general impact analysis)

Format-only adapter: NO added intelligence, NO query rewriting, NO fallback
beyond parsing the REST response.
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


DEFAULT_API_URL = "http://localhost:8699"
DEFAULT_INDEX_TIMEOUT_S = 900  # 15 minutes
DEFAULT_POLL_SLEEP_S = 1.0


def _corpus_name(payload: dict) -> str | None:
    """Normalize a query payload's corpus reference to a name string."""
    c = payload.get("corpus")
    if isinstance(c, Corpus):
        return c.name
    return c  # str or None


class NSCbmAdapter:
    """Adapter for NS using CBM engine via routed REST endpoints."""

    def __init__(
        self,
        api_url: str | None = None,
        index_timeout_s: int = DEFAULT_INDEX_TIMEOUT_S,
        poll_sleep_s: float = DEFAULT_POLL_SLEEP_S,
    ):
        """
        Initialize the adapter.

        Indexes via async `POST /v1/code-graph/index` (enqueued on ingest queue)
        with `system=code-cbm`, polls for completion, then queries through the
        routed code-graph endpoints with `knowledge_system=code-cbm`.

        Args:
            api_url: NS API base URL. If None, uses ICE_API_URL env var or
                defaults to http://localhost:8699.
            index_timeout_s: Max seconds to poll for index completion.
            poll_sleep_s: Sleep duration between poll attempts.
        """
        self.name = "ns-cbm"
        self.api_url = api_url or os.environ.get("ICE_API_URL", DEFAULT_API_URL)
        self.index_timeout_s = index_timeout_s
        self.poll_sleep_s = poll_sleep_s
        self.client = httpx.Client(timeout=120.0)
        self.version = self._read_version()

    def _read_version(self) -> str:
        """Best-effort NS API version from /health."""
        try:
            r = self.client.get(f"{self.api_url}/health", timeout=5.0)
            if r.status_code == 200:
                return f"ns-cbm@{r.json().get('service', 'neuralscape')}"
        except Exception:
            pass
        return "ns-cbm@unknown"

    def capabilities(self) -> set[str]:
        """
        NS-CBM supports 3 ops via routed REST endpoints.

        - symbol_lookup: routed search_graph
        - neighbors_1hop: routed trace_path (depth=1)
        - nl_locate: routed search_code (semantic)

        NOT supported:
        - path_le4: CBM path requires raw Cypher, banned → engine raises
        - blast_radius: CBM has no general impact analysis
        """
        return {"symbol_lookup", "neighbors_1hop", "nl_locate"}

    def _make_code_space(self, corpus_name: str) -> str:
        """Generate a code_space identifier for a corpus name."""
        return f"code--ice-bench--{corpus_name}"

    def _poll_task(self, task_id: str) -> dict | None:
        """
        Poll task status until completed/failed or timeout.

        Returns:
            Task result dict on success, None on timeout/failure.
        """
        poll_url = f"{self.api_url}/v1/memories/status/{task_id}"
        deadline = time.monotonic() + self.index_timeout_s

        while time.monotonic() < deadline:
            try:
                r = self.client.get(poll_url)
                if r.status_code != 200:
                    return None

                data = r.json()
                status = data.get("status")

                if status == "completed":
                    return data.get("result")
                elif status == "failed":
                    return None

                # Still queued/processing
                time.sleep(self.poll_sleep_s)

            except httpx.RequestError:
                return None

        # Timeout
        return None

    def index_cold(self, corpus: Corpus) -> IndexResult:
        """
        Index via async POST /v1/code-graph/index with system=code-cbm.

        The index runs server-side on the ingest worker, so peak_rss_mb and
        cpu_s are set to 0 (honest N/A, not measurable from the host).
        """
        code_space = self._make_code_space(corpus.name)
        project_id = code_space  # Keep it simple and self-consistent

        start = time.monotonic()

        try:
            resp = self.client.post(
                f"{self.api_url}/v1/code-graph/index",
                json={
                    "repo_source": corpus.path,
                    "project_id": project_id,
                    "system": "code-cbm",
                    "code_space": code_space,
                },
            )

            if resp.status_code != 202:
                return IndexResult(
                    wall_s=time.monotonic() - start,
                    peak_rss_mb=0,
                    cpu_s=0,
                    symbols=0,
                    edges=0,
                    files=0,
                    ok=False,
                )

            resp_data = resp.json()
            task_id = resp_data.get("task_id")
            if not task_id:
                return IndexResult(
                    wall_s=time.monotonic() - start,
                    peak_rss_mb=0,
                    cpu_s=0,
                    symbols=0,
                    edges=0,
                    files=0,
                    ok=False,
                )

            # Poll for completion
            result = self._poll_task(task_id)
            wall_s = time.monotonic() - start

            if result is None:
                # Timeout or failed
                return IndexResult(
                    wall_s=wall_s,
                    peak_rss_mb=0,
                    cpu_s=0,
                    symbols=0,
                    edges=0,
                    files=0,
                    ok=False,
                    dnf=True,
                    dnf_reason=f"index_poll_timeout>{self.index_timeout_s}s",
                )

            # Extract metrics from result
            symbols = int(result.get("symbols_indexed", 0))
            edges = int(result.get("edges_indexed", 0))
            files = int(result.get("files_indexed", 0))

            return IndexResult(
                wall_s=wall_s,
                peak_rss_mb=0,  # Server-side index, RSS not host-measurable
                cpu_s=0,  # Server-side index, CPU not host-measurable
                symbols=symbols,
                edges=edges,
                files=files,
                ok=True,
            )

        except httpx.RequestError as e:
            return IndexResult(
                wall_s=time.monotonic() - start,
                peak_rss_mb=0,
                cpu_s=0,
                symbols=0,
                edges=0,
                files=0,
                ok=False,
            )

    def index_incremental(self, corpus: Corpus, touched: list[str]) -> IndexResult:
        """CBM through NS doesn't support incremental => re-run cold."""
        return self.index_cold(corpus)

    def index_second(self, corpus: Corpus) -> IndexResult:
        """Second full index (stability probe)."""
        return self.index_cold(corpus)

    def store_size_bytes(self, corpus: Corpus) -> int:
        """
        Store size through NS REST.

        NS uses shared services (Neo4j/Qdrant) and the CBM bridge has its own
        cache. Through-REST we have no direct access to isolate per-code_space
        storage. Return 0 with this docstring note (honest N/A).
        """
        return 0

    def export_snapshot(self, corpus: Corpus) -> SnapshotResult | None:
        """Snapshot export not yet implemented for ns-cbm (honest N/A)."""
        return None

    def import_snapshot(self, corpus: Corpus, blob_path: str) -> SnapshotResult | None:
        """Snapshot import not yet implemented for ns-cbm (honest N/A)."""
        return None

    def query(self, op: str, payload: dict) -> QueryResult:
        """Execute a query via NS REST code-graph tools with knowledge_system=code-cbm."""
        if op not in self.capabilities():
            raise UnsupportedOp(f"Operation {op} not supported by ns-cbm")

        name = _corpus_name(payload)
        code_space = self._make_code_space(name) if name else None

        start = time.perf_counter()

        try:
            if op == "symbol_lookup":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/query",
                    params={
                        "question": payload["symbol"],
                        "knowledge_system": "code-cbm",
                        "graph_id": code_space,
                    },
                )
            elif op == "neighbors_1hop":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/neighbors",
                    params={
                        "label": payload["symbol"],
                        "knowledge_system": "code-cbm",
                        "graph_id": code_space,
                    },
                )
            elif op == "nl_locate":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/locate",
                    params={
                        "query": payload["query"],
                        "knowledge_system": "code-cbm",
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
        Delete the code_space's graph state (best-effort, never raises).
        """
        code_space = self._make_code_space(corpus.name)
        try:
            self.client.delete(
                f"{self.api_url}/v1/code-graph/graph",
                params={"graph_id": code_space},
            )
        except httpx.RequestError:
            pass
