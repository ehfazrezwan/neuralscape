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
            timeout: HTTP timeout for bridge calls.
        """
        self.bridge_url = bridge_url.rstrip("/")
        self.project = project
        self.code_space = code_space
        self.timeout = timeout
        self._http_client = httpx.Client(timeout=timeout)
        self._version: str | None = None

    def _call_bridge(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Call a CBM bridge endpoint.

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

        # Strip common root directories (src, lib, etc.) from the start.
        root_markers = {"src", "lib", "pkg", "internal", "app", "core", "main"}
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
            name = r.get("name", "")
            kind = r.get("label", "")
            file = r.get("file_path", "")
            line = r.get("line", "")
            canonical = self.to_canonical(name) if name else ""
            lines.append(f"  - {canonical} ({kind}) — {file}:{line}")

        return "\n".join(lines)

    def neighbors(
        self,
        label: str,
        *,
        relation_filter: str = "",
    ) -> str:
        """Direct in/out neighbors (maps to CBM trace_path with depth=1).

        Args:
            label: Symbol label to look up.
            relation_filter: Only edges containing this substring (applied client-side).

        Returns:
            Text list of neighbors.
        """
        project = self._ensure_project()

        # CBM trace_path needs the function name, not the full FQN.
        # Extract the last part (the actual symbol name).
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

        paths = result.get("paths", [])
        if not paths:
            return f"No neighbors found for '{label}'."

        lines = [f"Neighbors of '{label}':"]
        rel_filter = (relation_filter or "").lower()

        for path in paths[:50]:  # Limit for readability
            # CBM path format: [{"name": "...", "kind": "..."}, ...]
            if len(path) < 2:
                continue
            # path[0] is the source, path[1] is the neighbor
            neighbor = path[1]
            name = neighbor.get("name", "")
            kind = neighbor.get("kind", "")
            canonical = self.to_canonical(name) if name else ""

            # Apply relation filter (CBM doesn't give us edge labels, so skip if filter is set)
            if rel_filter:
                continue  # CBM doesn't expose edge relation types in trace_path

            # Determine direction (CBM doesn't tell us, so we use "related")
            lines.append(f"  -- {canonical} ({kind})")

        if len(lines) == 1:
            lines.append("  (no neighbors matching the filter)")

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
            name = r.get("name", "")
            kind = r.get("label", "")
            file = r.get("file_path", "")
            line = r.get("line", 0)

            canonical = self.to_canonical(name) if name else ""

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

        # Fetch version if not cached
        if self._version is None:
            try:
                status = self._call_bridge("/index_status", {})
                self._version = status.get("cbm_version", "cbm@unknown")
            except Exception:
                self._version = "cbm@unknown"

        return IndexReport(
            files_indexed=0,  # CBM doesn't report files count
            symbols_indexed=nodes,
            edges_indexed=edges,
            incremental=False,  # CBM always does full index
            duration_s=duration,
            system_version=self._version,
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
