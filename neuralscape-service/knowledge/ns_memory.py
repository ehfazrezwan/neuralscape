"""NSMemorySystem — wraps the existing MemoryService search facade as a KnowledgeSystem.

The base system (kind="base", transport="in-process"). Its ``recall()`` delegates
to the SAME code path the MCP tools call today (``memory/search.py`` pool fan-out
+ RRF fusion); this wrapper only adapts signatures. **ZERO behavior change** —
the wrapper is a pure E1-style refactor so tests can verify byte-identical
responses.

Default and always-eligible (health always "ok" when the service initialized).
"""

from __future__ import annotations

import logging

from knowledge.base import (
    HealthStatus,
    IndexReport,
    IndexRequest,
    KnowledgeSystemInfo,
    RecallRequest,
    SystemAnswer,
    TaskRef,
)
from memory_service import get_shared_service
from schemas import SearchMemoryRequest

logger = logging.getLogger(__name__)


class NSMemorySystem:
    """Wrap MemoryService as a KnowledgeSystem (the base system).

    This is the default, always-eligible knowledge backend. It wraps the
    existing ``memory/search.py`` search facade verbatim — no new logic, only
    signature adaptation.
    """

    info = KnowledgeSystemInfo(
        name="ns-memory",
        kind="base",
        capabilities=frozenset(
            {
                "recall",  # Semantic search (the core read path)
                "timeline",  # Chronological window (timeline tool)
                "cards",  # Identity cards (get_card tool)
                "ask",  # Synthesized answers (ask_memory tool)
                "graph_search",  # Graphiti knowledge graph search
            }
        ),
        transport="in-process",
    )

    def __init__(self):
        """Initialize the base system wrapper.

        Lazy-loads the shared MemoryService on first use (matching the existing
        lazy-init pattern in main.py + mcp_server.py).
        """
        self._service = None

    def _get_service(self):
        """Lazy-load the shared MemoryService."""
        if self._service is None:
            self._service = get_shared_service()
        return self._service

    def health(self) -> HealthStatus:
        """Health check: is the MemoryService initialized and reachable?

        Returns "ok" when mem0 + Graphiti are initialized (vector_store and
        graphiti attributes exist). Returns "degraded" if partially initialized,
        "unreachable" on failure.
        """
        try:
            svc = self._get_service()
            # Check that the core lazy-init has run (mem0 + Graphiti available).
            vector_ok = svc._memory is not None
            graph_ok = svc._graphiti is not None

            if vector_ok and graph_ok:
                return HealthStatus(
                    status="ok",
                    details={"vector_store": "ok", "graph_store": "ok"},
                )
            elif vector_ok:
                return HealthStatus(
                    status="degraded",
                    details={"vector_store": "ok", "graph_store": "not_initialized"},
                )
            else:
                return HealthStatus(
                    status="degraded",
                    details={
                        "vector_store": "not_initialized",
                        "graph_store": "not_initialized" if not graph_ok else "ok",
                    },
                )
        except Exception as e:
            logger.exception("NS memory health check failed")
            return HealthStatus(
                status="unreachable",
                details={"error": str(e)},
            )

    def recall(self, req: RecallRequest) -> SystemAnswer:
        """Delegate to the existing MemoryService.search() code path.

        This is the SAME search that ``recall_memories`` MCP tool calls today.
        The wrapper only adapts signatures: RecallRequest → SearchMemoryRequest,
        and the search result list → SystemAnswer.
        """
        svc = self._get_service()

        # Map RecallRequest to SearchMemoryRequest (the existing search contract).
        # RecallRequest has extra code-system fields (operation, label, source,
        # target, mode, depth) that base ignores.
        search_req = SearchMemoryRequest(
            query=req.query,
            user_id=req.user_id,
            project_id=req.project_id,
            limit=req.limit,
            # SearchMemoryRequest has many more optional filters (categories,
            # scope, domain, observation_type, concepts, visibility,
            # include_shared, index_only). RecallRequest doesn't expose them
            # yet (Phase B is pure refactor; additive params come in Phase D).
            # For now, use defaults (all categories, both scopes, shared+private).
        )

        # Call the existing search facade (SAME code path as today).
        results = svc.search(
            query=search_req.query,
            user_id=search_req.user_id or "default-user",
            project_id=search_req.project_id,
            categories=None,  # All categories
            scope=None,  # Both scopes
            limit=search_req.limit,
            # The search method has more params (domain, observation_type,
            # concepts, visibility, include_shared, index_only, workspaces);
            # all defaulted here so behavior is unchanged.
        )

        # Convert search results to SystemAnswer.
        # results is list[MemoryResponse]; render as content + structured hits.
        if not results:
            return SystemAnswer(
                system_name=self.info.name,
                system_version=None,  # Base doesn't stamp versions yet (Phase C adds this)
                content="No memories found.",
                hits=[],
                metadata={"result_count": 0},
            )

        # Format content as a plain-text list (mirrors how MCP tools render today).
        lines = []
        for i, mem in enumerate(results, 1):
            lines.append(f"{i}. {mem.memory_text} (category={mem.category})")
        content = "\n".join(lines)

        # Structured hits: convert MemoryResponse objects to dicts.
        hits = [mem.model_dump() for mem in results]

        return SystemAnswer(
            system_name=self.info.name,
            system_version=None,
            content=content,
            hits=hits,
            metadata={"result_count": len(results)},
        )

    def index(self, req: IndexRequest) -> TaskRef | IndexReport:
        """Base system has no separate indexing operation (writes happen via remember).

        Returns a no-op IndexReport so the protocol is satisfied. Code systems
        override this with real index triggering.
        """
        return IndexReport(
            files_indexed=0,
            symbols_indexed=0,
            edges_indexed=0,
            incremental=False,
            duration_s=0.0,
            system_version=None,
        )
