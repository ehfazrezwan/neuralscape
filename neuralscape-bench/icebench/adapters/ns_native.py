"""
NS-Native adapter: the frozen NativeEngine via NS REST code-graph routed endpoints.

Phase G-final (GF3): drives the native engine through the SAME through-NS surface
as ns-cbm / ns-graphify-lib — index via `POST /v1/code-graph/index
{system:"code-native"}`, query via `GET /v1/code-graph/*?knowledge_system=code-native`
— so native, graphify-lib, and cbm are compared apples-to-apples across ONE REST
surface, ONE corpus, ONE scorer.

This differs from the `ns-ice` / `ns-ice-det` adapters, which drive the DIRECT
native/graphify-json fallback path (no `knowledge_system`, CLI index). ns-native
exercises the routed KnowledgeSystem dispatch instead, requiring the server to run
with CODE_NATIVE_ENABLED=true (the bench stack opts in; prod stays default-off).

Capabilities: symbol_lookup, neighbors_1hop, path_le4, blast_radius, nl_locate
(NativeEngine supports every op — reported honestly: post-Phase-A symbol_lookup is
strong but neighbors are ~0 by design, the unfunded (b) resolution gap).

Format-only adapter: NO added intelligence, NO query rewriting, NO fallback beyond
parsing the REST response. The indexer runs server-side on the ingest worker, so
peak RSS / CPU are N/A (0, documented).
"""

import os
import time

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

_SYSTEM = "code-native"


def _corpus_name(payload: dict) -> str | None:
    """Normalize a query payload's corpus reference to a name string."""
    c = payload.get("corpus")
    if isinstance(c, Corpus):
        return c.name
    return c  # str or None


class NSNativeAdapter:
    """Adapter for NS using the native engine via routed REST endpoints."""

    def __init__(
        self,
        api_url: str | None = None,
        index_timeout_s: int = DEFAULT_INDEX_TIMEOUT_S,
        poll_sleep_s: float = DEFAULT_POLL_SLEEP_S,
    ):
        """
        Initialize the adapter.

        Indexes via async `POST /v1/code-graph/index` (enqueued on ingest queue)
        with `system=code-native`, polls for completion, then queries through the
        routed code-graph endpoints with `knowledge_system=code-native`.

        Args:
            api_url: NS API base URL. If None, uses ICE_API_URL env var or
                defaults to http://localhost:8699.
            index_timeout_s: Max seconds to poll for index completion.
            poll_sleep_s: Sleep duration between poll attempts.
        """
        self.name = "ns-native"
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
                return f"ns-native@{r.json().get('service', 'neuralscape')}"
        except Exception:
            pass
        return "ns-native@unknown"

    def capabilities(self) -> set[str]:
        """
        NS-Native supports all 5 ops via routed REST endpoints.

        - symbol_lookup: routed query (post-Phase-A lookup fix)
        - neighbors_1hop: routed neighbors (~0 by design — phantom/module edges
          intentionally dropped; the (b) resolution work is unfunded)
        - path_le4: routed path
        - blast_radius: routed impact analysis
        - nl_locate: routed locate (dense-embedding search over code_index)
        """
        return {"symbol_lookup", "neighbors_1hop", "path_le4", "blast_radius", "nl_locate"}

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
        Index via async POST /v1/code-graph/index with system=code-native.

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
                    "system": _SYSTEM,
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

        except httpx.RequestError:
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
        """The through-NS index always runs a full build => re-run cold."""
        return self.index_cold(corpus)

    def index_second(self, corpus: Corpus) -> IndexResult:
        """Second full index (stability probe)."""
        return self.index_cold(corpus)

    def store_size_bytes(self, corpus: Corpus) -> int:
        """
        Store size through NS REST.

        NS uses shared services (Neo4j/Qdrant); through-REST we have no direct
        access to isolate per-code_space storage. Return 0 (honest N/A).
        """
        return 0

    def export_snapshot(self, corpus: Corpus) -> SnapshotResult | None:
        """Snapshot export not implemented for ns-native (honest N/A)."""
        return None

    def import_snapshot(self, corpus: Corpus, blob_path: str) -> SnapshotResult | None:
        """Snapshot import not implemented for ns-native (honest N/A)."""
        return None

    def query(self, op: str, payload: dict) -> QueryResult:
        """Execute a query via NS REST code-graph tools with knowledge_system=code-native."""
        if op not in self.capabilities():
            raise UnsupportedOp(f"Operation {op} not supported by ns-native")

        name = _corpus_name(payload)
        code_space = self._make_code_space(name) if name else None

        start = time.perf_counter()

        try:
            if op == "symbol_lookup":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/query",
                    params={
                        "question": payload["symbol"],
                        "knowledge_system": _SYSTEM,
                        "graph_id": code_space,
                    },
                )
            elif op == "neighbors_1hop":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/neighbors",
                    params={
                        "label": payload["symbol"],
                        "knowledge_system": _SYSTEM,
                        "graph_id": code_space,
                    },
                )
            elif op == "path_le4":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/path",
                    params={
                        "source": payload["from"],
                        "target": payload["to"],
                        "max_hops": 4,
                        "knowledge_system": _SYSTEM,
                        "graph_id": code_space,
                    },
                )
            elif op == "blast_radius":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/impact",
                    params={
                        "symbol": payload["symbol"],
                        "max_hops": payload.get("max_hops", 4),
                        "knowledge_system": _SYSTEM,
                        "graph_id": code_space,
                    },
                )
            elif op == "nl_locate":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/locate",
                    params={
                        "query": payload["query"],
                        "knowledge_system": _SYSTEM,
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

        NOTE (honest): the `DELETE /v1/code-graph/graph` route is not implemented
        server-side, so this is currently a silent no-op — the through-NS index is
        not force-cold per rep. The trustworthy cold number is rep0 / the index
        payload's true engine duration. Documented in the comparison report.
        """
        code_space = self._make_code_space(corpus.name)
        try:
            self.client.delete(
                f"{self.api_url}/v1/code-graph/graph",
                params={"graph_id": code_space},
            )
        except httpx.RequestError:
            pass
