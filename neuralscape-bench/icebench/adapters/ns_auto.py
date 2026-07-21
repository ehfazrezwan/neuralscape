"""
NS-Auto adapter: NS auto-selects the best engine PER OP (no explicit engine pin).

AR4 (auto-routing): drives the SAME through-NS REST code-graph surface as
ns-native / ns-graphify-lib / ns-cbm, but passes ``knowledge_system=auto`` instead
of a specific engine. The server's per-op auto-router (knowledge/autoroute.py)
resolves each op to its measured-best HEALTHY capable engine — symbol_lookup →
native, neighbors/path/blast → graphify-lib, nl_locate → native — with graceful
fallback to the next-best when the top engine is down.

The payoff this adapter proves: **one system that wins every column**, which no
single real engine does (native owns symbol_lookup/nl_locate but is ~0 on
structure; graphify owns structure but has no symbol_lookup/nl_locate).

Every response attributes which engine actually served (AR3: the ``system``
field); this adapter captures it as ``served_by`` in the answer so the Track-Q
scorer parses each op with the correct engine's format parser — and so a
health-fallback run is visible (the served engine differs from the winner).

Format-only adapter: NO added intelligence, NO query rewriting, NO fallback in
the adapter itself — the fallback is the SERVER's per-op auto-router. Indexing
is server-side on the ingest worker (peak RSS / CPU N/A, documented as 0).

Note: indexing needs a concrete engine (you can't "auto"-index — you index INTO
a store). This adapter indexes via ``code-native`` by default (``NS_AUTO_INDEX_SYSTEM``
overrides) so the auto reads have a corpus to route over; the QUERY path is what
this system measures.
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

# The auto sentinel — NOT an engine, the per-op auto-router.
_SYSTEM = "auto"


def _corpus_name(payload: dict) -> str | None:
    """Normalize a query payload's corpus reference to a name string."""
    c = payload.get("corpus")
    if isinstance(c, Corpus):
        return c.name
    return c  # str or None


class NSAutoAdapter:
    """Adapter for NS auto-routing (knowledge_system=auto) via routed REST endpoints."""

    def __init__(
        self,
        api_url: str | None = None,
        index_timeout_s: int = DEFAULT_INDEX_TIMEOUT_S,
        poll_sleep_s: float = DEFAULT_POLL_SLEEP_S,
    ):
        self.name = "ns-auto"
        self.api_url = api_url or os.environ.get("ICE_API_URL", DEFAULT_API_URL)
        self.index_timeout_s = index_timeout_s
        self.poll_sleep_s = poll_sleep_s
        # The engine to index INTO (auto is a query-time router, not an indexer).
        # For the moat/native anchor host + all-op coverage, default native.
        self.index_system = os.environ.get("NS_AUTO_INDEX_SYSTEM", "code-native")
        self.client = httpx.Client(timeout=120.0)
        self.version = self._read_version()

    def _read_version(self) -> str:
        """Best-effort NS API version from /health."""
        try:
            r = self.client.get(f"{self.api_url}/health", timeout=5.0)
            if r.status_code == 200:
                return f"ns-auto@{r.json().get('service', 'neuralscape')}"
        except Exception:
            pass
        return "ns-auto@unknown"

    def capabilities(self) -> set[str]:
        """Auto covers every op (each routed to its best engine)."""
        return {"symbol_lookup", "neighbors_1hop", "path_le4", "blast_radius", "nl_locate"}

    def _make_code_space(self, corpus_name: str) -> str:
        return f"code--ice-bench--{corpus_name}"

    def _poll_task(self, task_id: str) -> dict | None:
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
                time.sleep(self.poll_sleep_s)
            except httpx.RequestError:
                return None
        return None

    def index_cold(self, corpus: Corpus) -> IndexResult:
        """Index via POST /v1/code-graph/index into `index_system` (native default).

        Auto is a query-time router; the corpus must be indexed into a concrete
        store first. Server-side index → RSS/CPU N/A (0, documented).
        """
        code_space = self._make_code_space(corpus.name)
        project_id = code_space
        start = time.monotonic()
        try:
            resp = self.client.post(
                f"{self.api_url}/v1/code-graph/index",
                json={
                    "repo_source": corpus.path,
                    "project_id": project_id,
                    "system": self.index_system,
                    "code_space": code_space,
                },
            )
            if resp.status_code != 202:
                return IndexResult(wall_s=time.monotonic() - start, peak_rss_mb=0,
                                   cpu_s=0, symbols=0, edges=0, files=0, ok=False)
            task_id = resp.json().get("task_id")
            if not task_id:
                return IndexResult(wall_s=time.monotonic() - start, peak_rss_mb=0,
                                   cpu_s=0, symbols=0, edges=0, files=0, ok=False)
            result = self._poll_task(task_id)
            wall_s = time.monotonic() - start
            if result is None:
                return IndexResult(wall_s=wall_s, peak_rss_mb=0, cpu_s=0, symbols=0,
                                   edges=0, files=0, ok=False, dnf=True,
                                   dnf_reason=f"index_poll_timeout>{self.index_timeout_s}s")
            return IndexResult(
                wall_s=wall_s, peak_rss_mb=0, cpu_s=0,
                symbols=int(result.get("symbols_indexed", 0)),
                edges=int(result.get("edges_indexed", 0)),
                files=int(result.get("files_indexed", 0)),
                ok=True,
            )
        except httpx.RequestError:
            return IndexResult(wall_s=time.monotonic() - start, peak_rss_mb=0,
                               cpu_s=0, symbols=0, edges=0, files=0, ok=False)

    def index_incremental(self, corpus: Corpus, touched: list[str]) -> IndexResult:
        return self.index_cold(corpus)

    def index_second(self, corpus: Corpus) -> IndexResult:
        return self.index_cold(corpus)

    def store_size_bytes(self, corpus: Corpus) -> int:
        return 0

    def export_snapshot(self, corpus: Corpus) -> SnapshotResult | None:
        return None

    def import_snapshot(self, corpus: Corpus, blob_path: str) -> SnapshotResult | None:
        return None

    def query(self, op: str, payload: dict) -> QueryResult:
        """Query the routed endpoint with knowledge_system=auto (per-op auto-select).

        Captures the SERVED engine (AR3 attribution: response `system` field) as
        `served_by` so the scorer parses with the right engine's format.
        """
        if op not in self.capabilities():
            raise UnsupportedOp(f"Operation {op} not supported by ns-auto")

        name = _corpus_name(payload)
        code_space = self._make_code_space(name) if name else None
        start = time.perf_counter()

        try:
            if op == "symbol_lookup":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/query",
                    params={"question": payload["symbol"],
                            "knowledge_system": _SYSTEM, "graph_id": code_space},
                )
            elif op == "neighbors_1hop":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/neighbors",
                    params={"label": payload["symbol"],
                            "knowledge_system": _SYSTEM, "graph_id": code_space},
                )
            elif op == "path_le4":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/path",
                    params={"source": payload["from"], "target": payload["to"],
                            "max_hops": 4, "knowledge_system": _SYSTEM,
                            "graph_id": code_space},
                )
            elif op == "blast_radius":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/impact",
                    params={"symbol": payload["symbol"],
                            "max_hops": payload.get("max_hops", 4),
                            "knowledge_system": _SYSTEM, "graph_id": code_space},
                )
            elif op == "nl_locate":
                resp = self.client.get(
                    f"{self.api_url}/v1/code-graph/locate",
                    params={"query": payload["query"],
                            "knowledge_system": _SYSTEM, "graph_id": code_space},
                )
            else:
                raise UnsupportedOp(f"Unexpected op: {op}")

            latency_ms = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                # AR3: capture which engine the auto-router chose for this op.
                served_by = None
                try:
                    served_by = resp.json().get("system")
                except Exception:
                    served_by = None
                answer = {"text": resp.text, "status": "ok", "served_by": served_by}
            else:
                answer = {"error": resp.text, "status": "error"}

            return QueryResult(
                latency_ms=latency_ms, answer=answer, ok=resp.status_code == 200
            )
        except httpx.RequestError as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return QueryResult(
                latency_ms=latency_ms,
                answer={"error": str(e), "status": "error"},
                ok=False,
            )

    def teardown(self, corpus: Corpus) -> None:
        """Reset the code_space so the next index rep is cold (via the index engine)."""
        code_space = self._make_code_space(corpus.name)
        try:
            self.client.delete(
                f"{self.api_url}/v1/code-graph/graph",
                params={"graph_id": code_space, "system": self.index_system},
            )
        except httpx.RequestError:
            pass
