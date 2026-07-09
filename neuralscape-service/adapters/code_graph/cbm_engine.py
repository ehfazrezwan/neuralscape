"""CBMEngine — CodeIntelEngine implementation over the CBM bridge (Phase C).

Talks to the CBM bridge service over HTTP REST (not directly to CBM). Maps
CodeIntelEngine protocol methods to the bridge's structured JSON tools. Methods
CBM can't support (path via raw Cypher is BANNED; detect_changes needs git) raise
EngineCapabilityError (honest N/A).

Canonical FQN normalization (to_canonical/from_canonical) is Phase C's core
deliverable — see PLAN §2. CBM's native FQN format is cache-path-prefixed,
dot-joined (e.g. `data-ice-corpora-...-8a4ce8….src.click.core.CommandCollection`).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any
from urllib.parse import urljoin

import httpx

from adapters.code_graph.engine import (
    ChangeReport,
    EngineCapabilityError,
    IndexReport,
    LocateHit,
    SemanticFact,
)

logger = logging.getLogger(__name__)

# M4 FIX: Health probe cache TTL (seconds). Prevents hot-path spam.
_HEALTH_CACHE_TTL = 10.0

# M4 FIX: Health probe timeout (seconds). Bounded to prevent event-loop stall.
_HEALTH_PROBE_TIMEOUT = 2.0


class CBMEngine:
    """CodeIntelEngine implementation talking to the CBM bridge over HTTP.

    The bridge exposes structured JSON tools only (no raw query_graph Cypher).
    This engine maps the CodeIntelEngine protocol to those tools and handles
    canonical FQN normalization.

    Attributes:
        bridge_url: Base URL of the CBM bridge service (e.g. http://cbm-bridge:8200).
        project: CBM project slug (from index_repository or resolved from code_space).
        code_space: NS code_space ref (code--owner--repo).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        bridge_url: str,
        project: str | None = None,
        code_space: str | None = None,
        timeout: int = 60,
    ):
        """Initialize CBM engine.

        Args:
            bridge_url: Base URL of the CBM bridge (e.g. http://cbm-bridge:8200).
            project: CBM project slug (from index_repository).
            code_space: NS code_space ref (code--owner--repo).
            timeout: HTTP timeout for bridge calls (operational, not health probe).
        """
        self.bridge_url = bridge_url.rstrip("/")
        self.project = project
        self.code_space = code_space
        self.timeout = timeout
        self._http_client = httpx.Client(timeout=timeout)
        self._version: str | None = None
        # M4 FIX: Health probe cache
        self._health_cache: tuple[bool, float] | None = None  # (result, timestamp)

    def _call_bridge(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """POST a CBM bridge endpoint (tool calls: search_graph, trace_path, ...).

        Args:
            endpoint: Bridge endpoint path (e.g. /search_graph).
            data: JSON request body.

        Returns:
            Parsed JSON response.

        Raises:
            RuntimeError: On HTTP error or timeout.
        """
        url = urljoin(self.bridge_url, endpoint)
        try:
            resp = self._http_client.post(url, json=data)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.error(f"CBM bridge call failed: {endpoint} - {e}")
            raise RuntimeError(f"CBM bridge call failed: {e}") from e

    def _get_bridge(self, endpoint: str) -> dict[str, Any]:
        """GET a CBM bridge endpoint (the bridge exposes /index_status and
        /health as GET, not POST — a POST there 405s).

        Args:
            endpoint: Bridge endpoint path (e.g. /index_status, /health).

        Returns:
            Parsed JSON response.

        Raises:
            RuntimeError: On HTTP error or timeout.
        """
        url = urljoin(self.bridge_url, endpoint)
        try:
            resp = self._http_client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.error(f"CBM bridge GET failed: {endpoint} - {e}")
            raise RuntimeError(f"CBM bridge GET failed: {e}") from e

    def health(self) -> bool:
        """Bridge reachability probe: GET /health (a REAL CBM tool call runs
        server-side). Returns True iff the bridge reports status ok.

        Used by CodeKnowledgeSystem.health() so a DOWN bridge makes the code-cbm
        system ineligible for routing (PLAN §3.3). Never raises — a network
        failure means "not healthy", not a crash.

        M4 FIX: Short timeout (2s, not 60s) + TTL cache (~10s) to bound hot-path cost.
        resolve_systems calls health() synchronously on the MCP event loop (mcp_server.py:
        1365); a black-holed bridge must not stall the loop for 60s per routed recall.
        """
        # Check cache first
        if self._health_cache is not None:
            result, timestamp = self._health_cache
            if (time.time() - timestamp) < _HEALTH_CACHE_TTL:
                logger.debug("CBM health: using cached result (age %.1fs)", time.time() - timestamp)
                return result

        # Probe with bounded timeout (not the operational self.timeout)
        try:
            # Create a short-timeout client just for the health probe
            with httpx.Client(timeout=_HEALTH_PROBE_TIMEOUT) as probe_client:
                url = urljoin(self.bridge_url, "/health")
                resp = probe_client.get(url)
                resp.raise_for_status()
                data = resp.json()
                result = data.get("status") == "ok"
        except Exception:  # noqa: BLE001 — unreachable bridge ⇒ not healthy
            logger.debug("CBM bridge health probe failed", exc_info=True)
            result = False

        # Cache the result with timestamp
        self._health_cache = (result, time.time())
        return result

    def _fetch_version(self) -> str:
        """Fetch and cache the CBM version via GET /index_status."""
        if self._version is None:
            try:
                status = self._get_bridge("/index_status")
                self._version = status.get("cbm_version") or "cbm@unknown"
            except Exception:  # noqa: BLE001
                self._version = "cbm@unknown"
        return self._version

    def _ensure_project(self) -> str:
        """Ensure we have a project slug; raise if not."""
        if not self.project:
            raise RuntimeError("No CBM project set (index() must be called first)")
        return self.project

    # ── Canonical FQN normalization (Phase C core deliverable) ──────────

    @staticmethod
    def to_canonical(raw_fqn: str) -> str:
        """Normalize CBM's native FQN to canonical form.

        CBM format: Sometimes includes cache-path prefix with hyphens, then dots.
        Examples:
          - `data-ice-corpora-small-py-8a4ce8.src.click.core.CommandCollection`
          - `src.click.core.CommandCollection` (if no cache prefix)

        Canonical format: `<module>.<qualname>` (src/lib roots stripped, '/' → '.')
        Example: `click.core.CommandCollection`

        Per PLAN §2: canonical_fqn := <repo-relative module path, src/lib roots
        stripped, '/' → '.'> + '.' + <qualname dotted>.

        Args:
            raw_fqn: CBM's native FQN (may have cache-path prefix).

        Returns:
            Canonical FQN (no cache prefix, src/lib stripped).
        """
        # Strategy: Split on dots. If any part contains hyphens (cache-path segments),
        # skip those. Then strip src/lib roots.

        parts = raw_fqn.split(".")

        # Remove cache-path segments (contain hyphens, not valid Python identifiers)
        clean_parts = [p for p in parts if "-" not in p]

        # Strip genuine source-root directories from the start ONLY. Narrow set
        # (src/lib) so real module names like `core`/`app`/`main` survive — see
        # NativeEngine.to_canonical for the rationale.
        root_markers = {"src", "lib"}
        while clean_parts and clean_parts[0] in root_markers:
            clean_parts.pop(0)

        canonical = ".".join(clean_parts)
        logger.debug(f"CBM to_canonical: {raw_fqn} → {canonical}")
        return canonical

    @staticmethod
    def from_canonical(canonical_fqn: str) -> str:
        """Convert canonical FQN back to CBM's native format (best-effort).

        Since CBM's cache-path prefix is dynamic (depends on repo path), we can't
        reconstruct it exactly. Instead, we return a search-friendly pattern:
        just the canonical FQN, which CBM's name_pattern search tolerates.

        For queries, CBM's search_graph accepts partial name patterns, so the
        canonical form works directly.

        Args:
            canonical_fqn: Canonical FQN (e.g. click.core.CommandCollection).

        Returns:
            Search pattern for CBM (same as canonical for search purposes).
        """
        # For search, canonical works as-is (CBM matches substrings).
        return canonical_fqn

    # ── CodeIntelEngine protocol methods ────────────────────────────────

    def query(
        self,
        question: str,
        *,
        mode: str = "bfs",
        depth: int = 3,
        token_budget: int = 2000,
    ) -> str:
        """Search the code graph (maps to CBM search_graph).

        Args:
            question: Natural-language question or keyword search.
            mode: Ignored (CBM's search_graph doesn't support mode).
            depth: Ignored (CBM's search_graph is flat).
            token_budget: Ignored (CBM returns raw results).

        Returns:
            Text rendering of search results.
        """
        project = self._ensure_project()
        result = self._call_bridge(
            "/search_graph",
            {"project": project, "name_pattern": question},
        )

        results = result.get("results", [])
        if not results:
            return f"No symbols matching '{question}' found."

        lines = [f"Symbols matching '{question}':"]
        for r in results[:20]:  # Limit to 20 results for readability
            # CBM's FQN is `qualified_name` (dotted, cache-prefixed); `name` is
            # the short symbol name (or a file path for polluted Variable nodes).
            raw_fqn = r.get("qualified_name") or r.get("name", "")
            kind = r.get("label", "")
            file = r.get("file_path", "")
            line = r.get("line", "")
            canonical = self.to_canonical(raw_fqn) if raw_fqn else ""
            lines.append(f"  - {canonical} ({kind}) — {file}:{line}")

        return "\n".join(lines)

    def neighbors(
        self,
        label: str,
        *,
        relation_filter: str = "",
    ) -> str:
        """Direct in/out neighbors (maps to CBM trace_path with depth=1).

        CBM ``trace_path`` returns ``{callees:[...], callers:[...]}`` where each
        step is ``{name, qualified_name, hop}``. Callees are outgoing (CALLS-out),
        callers are incoming (CALLS-in) — the only relation CBM exposes here is the
        call edge, so ``relation_filter`` is IGNORED rather than used to drop
        results (CBM's trace_path carries no edge-type labels to substring-match
        against; silently emptying the answer would violate neighbors() semantics).

        Args:
            label: Symbol label/FQN to look up.
            relation_filter: Ignored — CBM exposes no edge-type labels here.

        Returns:
            Text list of neighbors (--> callees, <-- callers).
        """
        project = self._ensure_project()

        # CBM trace_path keys on the short function name, not the full FQN.
        function_name = label.split(".")[-1]

        result = self._call_bridge(
            "/trace_path",
            {
                "project": project,
                "function_name": function_name,
                "direction": "both",
                "depth": 1,
            },
        )

        callees = result.get("callees", [])
        callers = result.get("callers", [])

        if not callees and not callers:
            return f"No neighbors found for '{label}'."

        lines = [f"Neighbors of '{label}':"]

        # Outgoing (this symbol calls these)
        for step in callees[:50]:
            raw_fqn = step.get("qualified_name") or step.get("name", "")
            canonical = self.to_canonical(raw_fqn) if raw_fqn else ""
            lines.append(f"  --> {canonical} [CALLS]")

        # Incoming (these call this symbol)
        for step in callers[:50]:
            raw_fqn = step.get("qualified_name") or step.get("name", "")
            canonical = self.to_canonical(raw_fqn) if raw_fqn else ""
            lines.append(f"  <-- {canonical} [CALLS]")

        return "\n".join(lines)

    def path(
        self,
        source: str,
        target: str,
        *,
        max_hops: int = 8,
    ) -> str:
        """Shortest path between two symbols.

        CBM can do this via query_graph Cypher, but raw Cypher is BANNED
        (lexer quote-escape bug → silent WHERE drop). We raise EngineCapabilityError.

        Raises:
            EngineCapabilityError: CBM can't do path without raw Cypher.
        """
        raise EngineCapabilityError(
            "CBM path operation requires raw query_graph Cypher, which is banned "
            "(lexer quote-escape bug causes silent WHERE-drop). Use graphify for path."
        )

    def locate(
        self,
        query: str,
        *,
        k: int = 10,
    ) -> list[LocateHit]:
        """Hybrid code retrieval (maps to CBM search_graph).

        Args:
            query: Natural-language description or symbol name pattern.
            k: Max hits to return.

        Returns:
            List of LocateHit objects.
        """
        project = self._ensure_project()
        result = self._call_bridge(
            "/search_graph",
            {"project": project, "name_pattern": query},
        )

        results = result.get("results", [])
        hits: list[LocateHit] = []

        for r in results[:k]:
            # CBM's FQN is `qualified_name`; `name` is short/file-path.
            raw_fqn = r.get("qualified_name") or r.get("name", "")
            kind = r.get("label", "")
            file = r.get("file_path", "")
            line = r.get("line", 0)

            canonical = self.to_canonical(raw_fqn) if raw_fqn else ""

            # Build anchor key: "<repo>::<canonical_fqn>"
            # Extract repo from code_space (code--owner--repo)
            repo = self.code_space.split("--")[-1] if self.code_space else "unknown"
            anchor_key = f"{repo}::{canonical}"

            hits.append(
                LocateHit(
                    fqn=canonical,
                    kind=kind,
                    file=file,
                    line=line,
                    signature="",  # CBM doesn't provide signatures
                    docstring="",  # CBM doesn't provide docstrings
                    score=1.0,  # CBM doesn't provide scores
                    anchor_id=anchor_key,
                    memories=None,  # Anchor join happens in CodeKnowledgeSystem
                )
            )

        return hits

    def get_symbol_inventory(self) -> set[str]:
        """Get current symbol inventory (canonical FQNs) for liveness tracking.

        Phase E: honest N/A (SCOPED FOLLOW-UP). The CBM bridge shim exposes only
        search_graph / trace_path / get_code_snippet / get_architecture /
        index_status / index_repository / delete_project (PLAN §3.1) — none of
        which is a complete, bounded symbol enumeration. ``index_status`` returns
        only counts ({project, nodes, edges}); ``search_graph`` is a name-pattern
        query, not an exhaustive listing. A faithful inventory needs a new bridge
        ``/list_symbols`` endpoint on the CBM bridge service (a separate
        pinned-source image, out of scope for this NS-service PR).

        Until that endpoint exists, this raises EngineCapabilityError, and the
        inventory-diff liveness consumer degrades gracefully ("inventory method
        unavailable") rather than treating it as a failure. Native and graphify
        engines DO support inventory diff today.

        Raises:
            EngineCapabilityError: CBM has no exhaustive symbol-enumeration tool.
        """
        raise EngineCapabilityError(
            "CBM has no exhaustive symbol-enumeration tool (index_status returns "
            "only counts; search_graph is a name-pattern query). Inventory-diff "
            "liveness for CBM is a scoped follow-up needing a bridge /list_symbols "
            "endpoint. Native/graphify support inventory diff today."
        )

    def detect_changes(
        self,
        since: str | bytes | None = None,
    ) -> ChangeReport:
        """Detect changes since a ref.

        CBM's detect_changes needs git state, which we may not have in a
        service context. Raise EngineCapabilityError.

        Raises:
            EngineCapabilityError: CBM detect_changes needs git.
        """
        raise EngineCapabilityError(
            "CBM detect_changes requires git state (git-diff based). "
            "Use graphify's affected() for stateless impact analysis."
        )

    def semantic_layer(self) -> list[SemanticFact]:
        """Distill semantic layer.

        CBM doesn't expose semantic facts (communities, hotspots) via its tools.
        Raise EngineCapabilityError.

        Raises:
            EngineCapabilityError: CBM doesn't expose semantic layer.
        """
        raise EngineCapabilityError(
            "CBM doesn't expose semantic layer (communities/hotspots) via its tools."
        )

    def index(
        self,
        source: str,
        *,
        incremental: bool = True,
    ) -> IndexReport:
        """Index a repository via CBM.

        Args:
            source: Repo path to index.
            incremental: Ignored (CBM handles incrementality internally).

        Returns:
            IndexReport with files/symbols/edges counts.
        """
        import time

        start = time.time()

        result = self._call_bridge(
            "/index_repository",
            {"repo_path": source},
        )

        # Update our project slug
        self.project = result.get("project", "")
        nodes = result.get("nodes", 0)
        edges = result.get("edges", 0)

        duration = time.time() - start

        # Stamp the CBM version (GET /index_status — POST would 405).
        version = self._fetch_version()

        return IndexReport(
            files_indexed=0,  # CBM doesn't report files count
            symbols_indexed=nodes,
            edges_indexed=edges,
            incremental=False,  # CBM always does full index
            duration_s=duration,
            system_version=version,
        )

    def export_snapshot(self) -> bytes:
        """Export a snapshot.

        CBM's graph.db artifact could be exported, but it's not exposed via
        the bridge's tools. Raise EngineCapabilityError for now.

        Raises:
            EngineCapabilityError: Snapshot export not exposed.
        """
        raise EngineCapabilityError(
            "CBM snapshot export (graph.db artifact) not exposed via bridge tools."
        )
