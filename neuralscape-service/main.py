"""Neuralscape Service — FastAPI + MCP service for mem0 with Graphiti backend.

Provides both legacy endpoints (root) and new v1 endpoints with scoping,
categories, and a shared MemoryService business logic layer.
"""

import asyncio
import base64
import hashlib
import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import adapters  # noqa: F401 — registers knowledge-adapter taxonomies at import (deterministic MEMORY_CATEGORIES)
from config import settings
from extensions import ExtensionRegistry
from extensions.events import EventType, EVENT_PAYLOAD_MODELS
from logging_config import configure_logging
from memory_service import get_shared_service

# Configure structured logging before anything else
configure_logging()
from context_formatter import format_context_for_injection
from index_format import index_row
from schemas import (
    MEMORY_CATEGORIES,
    AskMemoryRequest,
    AskMemoryResponse,
    AssembleContextRequest,
    AssembleContextResponse,
    BulkDeleteRequest,
    CategoryListResponse,
    CheckpointRequest,
    CheckpointResponse,
    CheckpointVerdict,
    ConnectorConfigRequest,
    ContextResponse,
    ExtractionInstructionsRequest,
    ExtractionInstructionsResponse,
    GetMemoriesRequest,
    GetMemoriesResponse,
    GraphSearchRequest,
    IndexRow,
    IngestDocumentRequest,
    IngestTextRequest,
    MemoryResponse,
    MemoryVisibility,
    QueueStatusResponse,
    RawMemoryBatchRequest,
    PatchMemoryRequest,
    RawMemoryRequest,
    RetagRequest,
    SavingsDetail,
    SearchIndexResponse,
    SearchMemoryRequest,
    SearchMemoryResponse,
    StoreMemoryRequest,
    StoreMemoryResponse,
    TaskAcceptedResponse,
    TaskStatusResponse,
    TimelineRequest,
    TimelineResponse,
    normalize_visibility,
)
from task_manager import TaskManager

logger = logging.getLogger(__name__)

# Shared service instance — the SAME object the mounted MCP server uses
# (audit 27 #35: one MemoryService per process, not one per surface).
_service = get_shared_service()

# Redis-backed task manager (initialized in lifespan)
_task_manager = TaskManager()

# Extension registry (discovered + started in lifespan)
_extension_registry = ExtensionRegistry()

# Connector vault (initialized in lifespan when connectors are enabled)
_vault = None

# Legacy lazy-init globals (kept for backward compat with old endpoints + tests)
_memory = None
_graphiti = None
_bridge = None
_async_memory = None
_legacy_init_lock = threading.Lock()


def _get_memory():
    """Lazy-initialize mem0 Memory with Graphiti backend (legacy).

    Thread-safe: uses double-checked locking to prevent concurrent initialization.
    """
    global _memory, _graphiti, _bridge
    if _memory is None:
        with _legacy_init_lock:
            if _memory is None:
                from mem0 import Memory

                config = settings.get_mem0_config()
                _memory = Memory.from_config(config)

                if hasattr(_memory, "graph") and hasattr(_memory.graph, "graphiti"):
                    _graphiti = _memory.graph.graphiti
                    _bridge = _memory.graph._bridge

    return _memory


def _get_graphiti():
    """Get the underlying Graphiti instance (legacy)."""
    _get_memory()
    return _graphiti


def _run_on_bridge(coro):
    """Run an async coroutine on the Graphiti adapter's event loop (legacy)."""
    if _bridge is None:
        raise HTTPException(status_code=503, detail="Graphiti bridge not initialized")
    return _bridge.run(coro)


def _get_async_memory():
    """Lazy-initialize an AsyncMemory instance for background processing."""
    global _async_memory
    if _async_memory is None:
        from mem0 import AsyncMemory

        config = settings.get_mem0_config()
        _async_memory = AsyncMemory.from_config(config)
    return _async_memory


# MCP HTTP session manager (initialized when MCP_TRANSPORT=http)
_mcp_session_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Neuralscape Service...")

    # Validate required configuration before anything else
    settings.validate_required()

    # Initialize the service (this also initializes mem0 + Graphiti)
    _service._get_memory()
    # Connect task manager to Redis
    await _task_manager.connect()

    # Discover and start extensions
    await _extension_registry.discover()
    await _extension_registry.startup_all()
    _extension_registry.mount_routes(app)

    # Initialize the connector vault when enabled.
    if settings.connectors_enabled:
        global _vault
        try:
            from connectors.vault import ConnectorVault

            _vault = ConnectorVault.from_settings(settings)
            logger.info("Connector vault initialized")
        except Exception as e:
            logger.warning(f"Connector vault init failed (connector API disabled): {e}")

    # Start MCP HTTP session manager if enabled and connect its task manager
    if _mcp_session_manager is not None:
        from mcp_server import _task_manager as mcp_task_manager
        await mcp_task_manager.connect()
        async with _mcp_session_manager.run():
            yield
        await mcp_task_manager.close()
    else:
        yield

    # Shutdown extensions with timeout to prevent hanging on misbehaving extensions
    try:
        await asyncio.wait_for(
            _extension_registry.shutdown_all(),
            timeout=10,
        )
    except asyncio.TimeoutError:
        logger.warning("Extension shutdown timed out after 10s, continuing shutdown")
    except Exception as e:
        logger.warning(f"Error during extension shutdown: {e}")

    # Shutdown with timeout to prevent hanging on unresponsive backends
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_shutdown_sync),
            timeout=10,
        )
    except asyncio.TimeoutError:
        logger.warning("Service shutdown timed out after 10s, forcing exit")
    except Exception as e:
        logger.warning(f"Error during shutdown: {e}")

    await _task_manager.close()
    logger.info("Neuralscape Service stopped.")


def _shutdown_sync():
    """Synchronous shutdown logic for _service and _async_memory."""
    _service.close()
    if _async_memory and hasattr(_async_memory, "graph") and hasattr(_async_memory.graph, "graphiti"):
        _async_memory.graph._bridge.run(_async_memory.graph.graphiti.close())


app = FastAPI(
    title="Neuralscape Memory Service",
    description="Production-grade memory layer with scoped memories, categories, REST and MCP interfaces",
    version="0.2.0",
    lifespan=lifespan,
)

# Register Bearer token auth middleware (no-op when NEURALSCAPE_API_KEY is unset)
from auth import BearerAuthMiddleware

app.add_middleware(BearerAuthMiddleware)


# ── Global exception handler ─────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return generic error to clients."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


# ══════════════════════════════════════════════
# Legacy endpoints (backward compat, root path)
# ══════════════════════════════════════════════


class LegacyAddMemoryRequest(BaseModel):
    messages: list[dict] = Field(description="Messages to add (list of {role, content} dicts)")
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    metadata: dict | None = None


class LegacySearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    limit: int = 10


class LegacyGraphSearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    limit: int = 10
    search_config: dict | None = Field(
        default=None,
        description="Optional SearchConfig dict to override default hybrid search",
    )


@app.get("/health/live")
async def health_live():
    """Pure process liveness — answers "should this process be killed?".

    Deliberately performs NO dependency checks (no Redis/Qdrant/Neo4j calls):
    under heavy write/enrichment load the readiness probes in /health can
    exceed the container healthcheck timeout, which marked a merely-busy API
    unhealthy and let the autoheal sidecar restart it mid-flight. The
    container healthcheck (and autoheal) must watch this endpoint; external
    monitors that care about backend reachability should keep using /health.
    """
    return {"status": "alive"}


@app.get("/health")
async def health():
    """Health check that verifies backend connectivity.

    Returns 200 with per-backend status when core services are reachable.
    Returns 503 if critical backends (vector store) are unreachable.
    """
    checks: dict[str, str] = {}

    # Check Redis (task queue)
    try:
        if _task_manager.pool:
            await _task_manager.pool.ping()
            checks["redis"] = "ok"
        else:
            checks["redis"] = "not_connected"
    except Exception:
        checks["redis"] = "unreachable"

    # Check vector store (Qdrant via mem0) — lightweight: just verify mem0 is initialized
    try:
        if _service._memory is not None:
            checks["vector_store"] = "ok"
        else:
            checks["vector_store"] = "not_initialized"
    except Exception:
        checks["vector_store"] = "unreachable"

    # Check graph store (Neo4j/Graphiti)
    try:
        if _service._graphiti is not None:
            checks["graph_store"] = "ok"
        else:
            checks["graph_store"] = "not_initialized"
    except Exception:
        checks["graph_store"] = "unreachable"

    # Overall status: degraded if Redis is down, unhealthy if vector store is down
    if checks.get("vector_store") == "unreachable":
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "service": "neuralscape-memory", "checks": checks},
        )

    overall = "ok"
    if any(v != "ok" for v in checks.values()):
        overall = "degraded"

    return {"status": overall, "service": "neuralscape-memory", "checks": checks}


@app.post("/memories")
async def add_memory(req: LegacyAddMemoryRequest):
    """Add a memory through mem0 (vector + graph). Legacy endpoint."""
    m = _get_memory()
    try:
        result = await asyncio.to_thread(
            m.add,
            messages=req.messages,
            user_id=req.user_id or settings.default_user_id,
            agent_id=req.agent_id,
            run_id=req.run_id,
            metadata=req.metadata,
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.exception("add_memory failed")
        raise HTTPException(status_code=500, detail="Memory storage failed")


@app.post("/search")
async def search_memories(req: LegacySearchRequest):
    """Search memories through mem0. Legacy endpoint."""
    m = _get_memory()
    try:
        result = await asyncio.to_thread(
            m.search,
            query=req.query,
            user_id=req.user_id or settings.default_user_id,
            agent_id=req.agent_id,
            run_id=req.run_id,
            limit=req.limit,
        )
        return {"status": "ok", "results": result}
    except Exception as e:
        logger.exception("search_memories failed")
        raise HTTPException(status_code=500, detail="Memory search failed")


@app.get("/memories")
async def list_memories(
    user_id: str = Query(default=None),
    agent_id: str = Query(default=None),
    run_id: str = Query(default=None),
    limit: int = Query(default=100),
):
    """List all memories for a user. Legacy endpoint."""
    m = _get_memory()
    try:
        result = await asyncio.to_thread(
            m.get_all,
            user_id=user_id or settings.default_user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit,
        )
        return {"status": "ok", "memories": result}
    except Exception as e:
        logger.exception("list_memories failed")
        raise HTTPException(status_code=500, detail="Failed to list memories")


@app.delete("/memories")
async def delete_memories(
    user_id: str = Query(default=None),
    agent_id: str = Query(default=None),
    run_id: str = Query(default=None),
):
    """Delete all memories for a user. Legacy endpoint."""
    m = _get_memory()
    try:
        await asyncio.to_thread(
            m.delete_all,
            user_id=user_id or settings.default_user_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        return {"status": "ok", "message": "All memories deleted"}
    except Exception as e:
        logger.exception("delete_memories failed")
        raise HTTPException(status_code=500, detail="Failed to delete memories")


@app.post("/memories/async")
async def add_memory_async(req: LegacyAddMemoryRequest):
    """Add a memory in the background (non-blocking). Returns a task_id to poll."""
    task_id = await _task_manager.enqueue_store(
        messages=req.messages,
        user_id=req.user_id or settings.default_user_id,
        agent_id=req.agent_id,
        run_id=req.run_id,
    )
    return TaskAcceptedResponse(
        task_id=task_id,
        poll_url=f"/memories/status/{task_id}",
    )


@app.get("/memories/status/{task_id}")
async def get_task_status(task_id: str):
    """Check the status of an async memory addition task."""
    result = await _task_manager.get_status(task_id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return result


# Legacy graph endpoints
@app.get("/graph/nodes")
async def list_graph_nodes_legacy(
    user_id: str = Query(default=None),
    limit: int = Query(default=50),
):
    """List entity nodes from Graphiti. Legacy endpoint."""
    g = _get_graphiti()
    if g is None:
        raise HTTPException(status_code=503, detail="Graphiti not initialized")

    from graphiti_core.nodes import EntityNode

    group_id = user_id or settings.default_user_id
    try:
        nodes = await asyncio.to_thread(
            _run_on_bridge,
            EntityNode.get_by_group_ids(g.driver, group_ids=[group_id], limit=limit),
        )
        return {
            "status": "ok",
            "nodes": [
                {
                    "uuid": n.uuid,
                    "name": n.name,
                    "summary": n.summary,
                    "labels": n.labels,
                    "group_id": n.group_id,
                    "created_at": n.created_at.isoformat(),
                }
                for n in nodes
            ],
        }
    except Exception as e:
        logger.warning("list_graph_nodes failed: %s", e)
        return {"status": "ok", "nodes": []}


@app.get("/graph/edges")
async def list_graph_edges_legacy(
    user_id: str = Query(default=None),
    limit: int = Query(default=50),
):
    """List entity edges (facts) from Graphiti. Legacy endpoint."""
    g = _get_graphiti()
    if g is None:
        raise HTTPException(status_code=503, detail="Graphiti not initialized")

    from graphiti_core.edges import EntityEdge
    from graphiti_core.errors import GroupsEdgesNotFoundError

    group_id = user_id or settings.default_user_id
    try:
        edges = await asyncio.to_thread(
            _run_on_bridge,
            EntityEdge.get_by_group_ids(g.driver, group_ids=[group_id], limit=limit),
        )
        return {
            "status": "ok",
            "edges": [
                {
                    "uuid": e.uuid,
                    "name": e.name,
                    "fact": e.fact,
                    "source_node_uuid": e.source_node_uuid,
                    "target_node_uuid": e.target_node_uuid,
                    "group_id": e.group_id,
                    "created_at": e.created_at.isoformat(),
                    "valid_at": e.valid_at.isoformat() if e.valid_at else None,
                    "invalid_at": e.invalid_at.isoformat() if e.invalid_at else None,
                    "expired_at": e.expired_at.isoformat() if e.expired_at else None,
                }
                for e in edges
            ],
        }
    except GroupsEdgesNotFoundError:
        return {"status": "ok", "edges": []}
    except Exception as e:
        logger.warning("list_graph_edges failed: %s", e)
        return {"status": "ok", "edges": []}


@app.get("/graph/episodes")
async def list_graph_episodes_legacy(
    user_id: str = Query(default=None),
    limit: int = Query(default=20),
):
    """List episodic nodes from Graphiti. Legacy endpoint."""
    g = _get_graphiti()
    if g is None:
        raise HTTPException(status_code=503, detail="Graphiti not initialized")

    group_id = user_id or settings.default_user_id
    now = datetime.now(timezone.utc)
    try:
        episodes = await asyncio.to_thread(
            _run_on_bridge,
            g.retrieve_episodes(
                reference_time=now,
                last_n=limit,
                group_ids=[group_id],
            ),
        )
        return {
            "status": "ok",
            "episodes": [
                {
                    "uuid": ep.uuid,
                    "name": ep.name,
                    "content": ep.content,
                    "source_description": ep.source_description,
                    "group_id": ep.group_id,
                    "created_at": ep.created_at.isoformat(),
                    "valid_at": ep.valid_at.isoformat() if ep.valid_at else None,
                }
                for ep in episodes
            ],
        }
    except Exception as e:
        logger.warning("list_graph_episodes failed: %s", e)
        return {"status": "ok", "episodes": []}


@app.get("/graph/communities")
async def list_graph_communities_legacy(
    user_id: str = Query(default=None),
    limit: int = Query(default=20),
):
    """List community nodes from Graphiti. Legacy endpoint."""
    g = _get_graphiti()
    if g is None:
        raise HTTPException(status_code=503, detail="Graphiti not initialized")

    from graphiti_core.nodes import CommunityNode

    group_id = user_id or settings.default_user_id
    try:
        communities = await asyncio.to_thread(
            _run_on_bridge,
            CommunityNode.get_by_group_ids(g.driver, group_ids=[group_id], limit=limit),
        )
        return {
            "status": "ok",
            "communities": [
                {
                    "uuid": c.uuid,
                    "name": c.name,
                    "summary": c.summary if hasattr(c, "summary") else "",
                    "group_id": c.group_id,
                    "created_at": c.created_at.isoformat(),
                }
                for c in communities
            ],
        }
    except Exception as e:
        logger.warning("list_graph_communities failed: %s", e)
        return {"status": "ok", "communities": []}


@app.post("/graph/search")
async def advanced_graph_search_legacy(req: LegacyGraphSearchRequest):
    """Advanced Graphiti search with configurable SearchConfig. Legacy endpoint."""
    g = _get_graphiti()
    if g is None:
        raise HTTPException(status_code=503, detail="Graphiti not initialized")

    from graphiti_core.search.search_config import SearchConfig
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

    group_id = req.user_id or settings.default_user_id

    if req.search_config:
        try:
            config = SearchConfig(**req.search_config)
        except (TypeError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid search_config: {e}",
            )
    else:
        config = EDGE_HYBRID_SEARCH_RRF

    config.limit = req.limit

    try:
        results = await asyncio.to_thread(
            _run_on_bridge,
            g.search_(
                query=req.query,
                config=config,
                group_ids=[group_id],
            ),
        )

        return {
            "status": "ok",
            "edges": [
                {"uuid": e.uuid, "name": e.name, "fact": e.fact}
                for e in results.edges
            ],
            "nodes": [
                {"uuid": n.uuid, "name": n.name, "summary": n.summary}
                for n in results.nodes
            ],
            "episodes": [
                {"uuid": ep.uuid, "name": ep.name, "content": ep.content}
                for ep in results.episodes
            ],
            "communities": [
                {"uuid": c.uuid, "name": c.name}
                for c in results.communities
            ],
        }
    except Exception as e:
        logger.exception("graph search failed")
        raise HTTPException(status_code=500, detail="Graph search failed")


# ══════════════════════════════════════════════
# V1 API (new endpoints with scoping + categories)
# ══════════════════════════════════════════════

v1_router = APIRouter(prefix="/v1", tags=["v1"])


def _resolve_user_id(request: Request, body_user_id: str | None) -> str:
    """Resolve the authoritative user_id for a v1 request.

    - When the auth middleware verified a per-user token, ``request.state.user_id``
      is set; that wins. If the body also supplied ``user_id`` and they disagree,
      reject with 400 to prevent token-vs-body identity confusion.
    - Legacy shared-key callers don't have ``request.state.user_id``; we trust
      the body's ``user_id`` as before.
    - When neither path supplies an id, fall back to ``settings.default_user_id``.

    Raises HTTPException(400) on token/body mismatch.
    """
    token_user_id = getattr(request.state, "user_id", None)
    if token_user_id:
        if body_user_id and body_user_id != token_user_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Request body user_id ({body_user_id!r}) does not match the "
                    f"user_id encoded in the auth token ({token_user_id!r})."
                ),
            )
        return token_user_id
    return body_user_id or settings.default_user_id


def _authorize_standard_write(user_id: str, visibility) -> None:
    """Reject writes to the authoritative ``standard`` tier by non-dictators.

    No-op unless the request targets ``visibility="standard"``. Called
    synchronously before enqueue so the caller gets a 403 immediately instead
    of a 202 followed by a silent worker failure. ``store_raw`` re-checks as a
    backstop for the sync-fallback / worker paths.
    """
    if normalize_visibility(visibility) != MemoryVisibility.STANDARD.value:
        return
    if not settings.standards_enabled:
        raise HTTPException(status_code=403, detail="The 'standard' visibility tier is disabled.")
    if not settings.is_dictator(user_id):
        raise HTTPException(
            status_code=403,
            detail=f"User {user_id!r} is not authorized to write 'standard'-tier memories.",
        )


# ── Remember ──────────────────────────────────


@v1_router.post("/memories", status_code=202)
async def v1_store_memories(req: StoreMemoryRequest, request: Request):
    """Store memories from conversation via LLM extraction (async).

    Enqueues the extraction task to a background worker and returns immediately
    with a task_id that can be polled via GET /v1/memories/status/{task_id}.
    Falls back to synchronous storage if Redis is unavailable.
    """
    user_id = _resolve_user_id(request, req.user_id)
    try:
        task_id = await _task_manager.enqueue_store(
            messages=req.messages,
            user_id=user_id,
            project_id=req.project_id,
            agent_id=req.agent_id,
            run_id=req.run_id,
            occurred_at=req.occurred_at,
        )
    except (ConnectionError, OSError) as e:
        logger.warning(f"Redis unavailable, falling back to sync store: {e}")
        memories = await asyncio.to_thread(
            _service.extract_and_store,
            messages=req.messages,
            user_id=user_id,
            project_id=req.project_id,
            agent_id=req.agent_id,
            run_id=req.run_id,
            occurred_at=req.occurred_at,
        )
        return JSONResponse(
            status_code=200,
            content=StoreMemoryResponse(memories=memories).model_dump(exclude_none=True),
        )

    return TaskAcceptedResponse(
        task_id=task_id,
        poll_url=f"/v1/memories/status/{task_id}",
    )


@v1_router.post("/memories/raw", status_code=202)
async def v1_store_raw_memory(req: RawMemoryRequest, request: Request):
    """Store a single pre-categorized fact (async, no LLM extraction).

    Enqueues the storage task to a background worker and returns immediately.
    Falls back to synchronous storage if Redis is unavailable.
    Memory-model v2 fields (domain, observation_type, concepts, source_type,
    related_memory_ids, confidence, expires_at) are all optional.

    Multi-user: when authenticated with a per-user token, the user_id is
    taken from the token (authoritative). When sending user_id in the body
    as well, it must match the token's user_id.
    """
    if req.category not in MEMORY_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category: {req.category}. Must be one of: {list(MEMORY_CATEGORIES.keys())}",
        )
    if req.scope == "project" and not req.project_id:
        raise HTTPException(status_code=400, detail="project_id is required when scope='project'")

    # Resolve identity from token (preferred) or body (legacy). Raises 400 on mismatch.
    resolved_user_id = _resolve_user_id(request, req.user_id)
    # Authoritative-tier write-gate (synchronous 403 before enqueue).
    _authorize_standard_write(resolved_user_id, req.visibility)

    # Build the kwargs once; passed through to both the queue path and the sync fallback.
    raw_kwargs = req.model_dump(exclude_none=True)
    raw_kwargs["user_id"] = resolved_user_id
    # serialize datetime so it survives JSON enqueue
    if "expires_at" in raw_kwargs and hasattr(raw_kwargs["expires_at"], "isoformat"):
        raw_kwargs["expires_at"] = raw_kwargs["expires_at"].isoformat()

    try:
        task_id = await _task_manager.enqueue_raw(**raw_kwargs)
    except (ConnectionError, OSError) as e:
        logger.warning(f"Redis unavailable, falling back to sync store: {e}")
        # Sync path expects datetime, not ISO string
        sync_kwargs = dict(raw_kwargs)
        if "expires_at" in sync_kwargs and isinstance(sync_kwargs["expires_at"], str):
            try:
                sync_kwargs["expires_at"] = datetime.fromisoformat(
                    sync_kwargs["expires_at"].replace("Z", "+00:00")
                )
            except ValueError:
                sync_kwargs.pop("expires_at", None)
        try:
            memories = await asyncio.to_thread(_service.store_raw, **sync_kwargs)
        except PermissionError as pe:
            raise HTTPException(status_code=403, detail=str(pe)) from pe
        return JSONResponse(
            status_code=200,
            content=StoreMemoryResponse(memories=memories).model_dump(exclude_none=True),
        )

    return TaskAcceptedResponse(
        task_id=task_id,
        poll_url=f"/v1/memories/status/{task_id}",
    )


# ── Ingest (data-layer documents) ────────────────


@v1_router.post("/ingest", status_code=202)
async def v1_ingest_document(req: IngestDocumentRequest, request: Request):
    """Ingest a document from a data layer: chunk → passages + distilled facts.

    Each produced memory carries the ``source`` descriptor (passages get a
    per-chunk ``chunk_index``/``span``; facts get the parent-level descriptor)
    so a consuming agent can trace the memory back and re-fetch via the
    retrieval handle. Async (202 + poll) with a sync fallback if Redis is down.
    Re-ingesting unchanged content is idempotent (content-hash dedup).
    """
    if req.scope == "project" and not req.project_id:
        raise HTTPException(status_code=400, detail="project_id is required when scope='project'")

    resolved_user_id = _resolve_user_id(request, req.user_id)
    _authorize_standard_write(resolved_user_id, req.visibility)
    doc = {
        "content": req.content,
        "source": req.source.model_dump(exclude_none=True),
        "user_id": resolved_user_id,
        "category": req.category,
        "scope": req.scope,
        "project_id": req.project_id,
        "visibility": req.visibility.value if req.visibility else None,
        "tags": req.tags,
        "agent_id": req.agent_id,
        "run_id": req.run_id,
        "extract_facts": req.extract_facts,
        "index_passages": req.index_passages,
        "adapter": req.adapter,
    }
    doc = {k: v for k, v in doc.items() if v is not None}

    try:
        task_id = await _task_manager.enqueue_ingest_document(doc)
    except (ConnectionError, OSError) as e:
        logger.warning(f"Redis unavailable, falling back to sync ingest: {e}")
        from ingest.pipeline import IngestDoc, ingest_document

        result = await asyncio.to_thread(ingest_document, _service, IngestDoc(**doc))
        # Redis is down (that's why we're in the sync fallback), so the deferred
        # graph jobs can't be enqueued — report them honestly as skipped instead
        # of returning full job payloads. These facts stay vector-only until a
        # graph backfill/re-ingest.
        _skipped = len(result.pop("graph_jobs", []) or [])
        if _skipped:
            logger.warning(f"Sync-fallback ingest: {_skipped} fact graph enrichment(s) skipped (Redis down)")
            result["graph_jobs_skipped"] = _skipped
        return JSONResponse(status_code=200, content={"status": "ok", **result})

    return TaskAcceptedResponse(
        task_id=task_id,
        poll_url=f"/v1/memories/status/{task_id}",
    )


def _fallback_source_ref(content: bytes | str, title: str | None, connector_type: str) -> dict:
    """Synthetic provenance used only when artifact storage is disabled.

    Every ingested memory still needs a ``source_ref`` to be traceable; when we
    aren't persisting an artifact, the best we can do is a content-hash backlink.
    """
    raw = content.encode() if isinstance(content, str) else content
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return {
        "connector_id": connector_type,
        "connector_type": connector_type,
        "external_id": digest,
        "parent_id": digest,
        "title": title,
        "last_synced_at": datetime.now(timezone.utc).isoformat(),
    }


@v1_router.post("/ingest/text", status_code=202)
async def v1_ingest_text(req: IngestTextRequest, request: Request):
    """Manually provide a block of context — a first-class ingestion path.

    The context is persisted as a Markdown artifact on the storage volume
    (organized by user/project/category) and the produced memories reference it,
    so manual context is just as traceable as an uploaded file. Chunks into
    passages + distils LLM facts; async (202 + poll) on the dedicated ingest
    queue; idempotent via content-hash dedup.
    """
    if req.scope == "project" and not req.project_id:
        raise HTTPException(status_code=400, detail="project_id is required when scope='project'")

    resolved_user_id = _resolve_user_id(request, req.user_id)
    # Reject non-dictator standard-tier ingests synchronously (else every
    # produced job would fail later in the worker — a lost-job trap).
    _authorize_standard_write(resolved_user_id, req.visibility)

    # Persist the pasted context as an artifact so its memories reference a real,
    # re-fetchable source (falls back to a hash-only ref when storage is off).
    if settings.ingest_storage_enabled:
        from ingest.storage import artifact_source_ref, store_artifact

        fname = (req.title or "context") + ".md"
        art = await asyncio.to_thread(
            store_artifact, req.content.encode(), fname, resolved_user_id,
            req.project_id, req.category, settings,
        )
        source_ref = artifact_source_ref(art, connector_type="manual")
    else:
        source_ref = _fallback_source_ref(req.content, req.title, "manual")

    doc = {
        "content": req.content,
        "source": source_ref,
        "user_id": resolved_user_id,
        "category": req.category,
        "scope": req.scope,
        "project_id": req.project_id,
        "visibility": req.visibility.value if req.visibility else None,
        "tags": req.tags,
        "agent_id": req.agent_id,
        "run_id": req.run_id,
        "extract_facts": req.extract_facts,
        "index_passages": req.index_passages,
        "adapter": req.adapter,
        "occurred_at": req.occurred_at,
    }
    doc = {k: v for k, v in doc.items() if v is not None}

    try:
        task_id = await _task_manager.enqueue_ingest_document(doc)
    except (ConnectionError, OSError) as e:
        logger.warning(f"Redis unavailable, falling back to sync ingest: {e}")
        from ingest.pipeline import IngestDoc, ingest_document

        result = await asyncio.to_thread(ingest_document, _service, IngestDoc(**doc))
        # Redis is down (that's why we're in the sync fallback), so the deferred
        # graph jobs can't be enqueued — report them honestly as skipped instead
        # of returning full job payloads. These facts stay vector-only until a
        # graph backfill/re-ingest.
        _skipped = len(result.pop("graph_jobs", []) or [])
        if _skipped:
            logger.warning(f"Sync-fallback ingest: {_skipped} fact graph enrichment(s) skipped (Redis down)")
            result["graph_jobs_skipped"] = _skipped
        return JSONResponse(status_code=200, content={"status": "ok", **result})

    return TaskAcceptedResponse(task_id=task_id, poll_url=f"/v1/memories/status/{task_id}")


@v1_router.post("/ingest/files", status_code=202)
async def v1_ingest_files(
    request: Request,
    files: list[UploadFile] = File(..., description="One or more files, or a .zip to expand"),
    category: str = Form("domain_knowledge"),
    scope: str = Form("global"),
    project_id: str | None = Form(None),
    user_id: str | None = Form(None),
    visibility: str | None = Form(None),
    tags: str | None = Form(None, description="Comma-separated tags"),
    extract_facts: bool = Form(True),
    index_passages: bool = Form(True),
    adapter: str = Form("default", description="Knowledge adapter (e.g. 'default', 'trading_strategy')"),
    page_offset: int = Form(
        0,
        description=(
            "Added to figure page numbers in provenance refs. Use when uploading "
            "a slice of a larger document (e.g. pages 61-80 of a book → 60) so "
            "exemplar page refs stay relative to the original."
        ),
    ),
):
    """Upload one or more files (or a zip / zipped folder) to ingest into memory.

    Zips are expanded server-side; each resulting file is parsed (Docling →
    Markdown, MarkItDown fallback), chunked into passages, and distilled into
    graph facts. Parsing runs on the dedicated ingest worker (not this request),
    so a large batch never blocks the API. Returns one task_id per file to poll.
    """
    # ── Validate form fields up-front (parity with the Pydantic endpoints) ──
    if scope not in ("global", "project"):
        raise HTTPException(status_code=400, detail="scope must be 'global' or 'project'")
    if scope == "project" and not project_id:
        raise HTTPException(status_code=400, detail="project_id is required when scope='project'")
    if category not in MEMORY_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category '{category}'")
    if visibility is not None and visibility not in ("private", "shared", "standard"):
        raise HTTPException(status_code=400, detail="visibility must be 'private', 'shared', or 'standard'")
    parsed_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    if parsed_tags and len(parsed_tags) > 20:
        raise HTTPException(status_code=400, detail="at most 20 tags are allowed")
    if not 0 <= page_offset <= 100_000:
        raise HTTPException(status_code=400, detail="page_offset must be between 0 and 100000")
    try:
        from schemas import validate_adapter_name

        validate_adapter_name(adapter)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve

    resolved_user_id = _resolve_user_id(request, user_id)
    # Standard-tier ingestion (a dictator uploading org standards) is allowed, but
    # only for authorized dictators — reject others synchronously so we don't
    # enqueue a batch of jobs that each fail later in the worker. store_raw forces
    # standards to global scope regardless of the scope form field.
    _authorize_standard_write(resolved_user_id, visibility)
    max_file_bytes = settings.ingest_max_file_mb * 1024 * 1024
    max_request_bytes = settings.ingest_max_request_mb * 1024 * 1024

    options = {
        "category": category,
        "scope": scope,
        "project_id": project_id,
        "visibility": visibility,
        "tags": parsed_tags,
        "extract_facts": extract_facts,
        "index_passages": index_passages,
        "adapter": adapter,
        "page_offset": page_offset or None,  # omitted when 0 (the default)
        "agent_id": None,
        "run_id": None,
    }
    options = {k: v for k, v in options.items() if v is not None}

    from ingest.archive import ArchiveError, ArchiveTooLarge, is_zip, iter_archive
    from ingest.storage import artifact_source_ref, store_artifact

    enqueued: list[dict] = []
    totals = {"files": 0, "bytes": 0}

    async def _store_and_enqueue(name: str, data: bytes, *, okf_bundle: bool = False) -> None:
        """Persist one file + enqueue its ingest job, enforcing request caps.

        Called incrementally per upload / per zip member so we never hold the
        whole request in memory at once. ``okf_bundle=True`` stores the zip as
        ONE artifact and enqueues the whole-bundle OKF walker instead of the
        per-file parser.
        """
        totals["files"] += 1
        if totals["files"] > settings.ingest_max_files:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds the {settings.ingest_max_files}-file limit",
            )
        totals["bytes"] += len(data)
        if totals["bytes"] > max_request_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds the {settings.ingest_max_request_mb}MB total-request limit",
            )
        # Persist to the shared volume; hand the worker a path (not bytes) so
        # large files don't travel through Redis, and stamp a source_ref that
        # points back to the stored artifact.
        connector_type = "okf_bundle" if okf_bundle else "file_upload"
        payload = {"filename": name, "options": options, "user_id": resolved_user_id}
        if settings.ingest_storage_enabled:
            art = await asyncio.to_thread(
                store_artifact, data, name, resolved_user_id, project_id, category, settings,
            )
            payload["stored_path"] = art.rel_path
            payload["source_ref"] = artifact_source_ref(art, connector_type=connector_type)
        else:
            payload["data_b64"] = base64.b64encode(data).decode()
            payload["source_ref"] = _fallback_source_ref(data, name, connector_type)
        try:
            if okf_bundle:
                task_id = await _task_manager.enqueue_ingest_okf_bundle(payload)
            else:
                task_id = await _task_manager.enqueue_ingest_file(payload)
        except (ConnectionError, OSError) as e:
            raise HTTPException(status_code=503, detail=f"Ingest queue unavailable: {e}")
        enqueued.append({
            "filename": name,
            "task_id": task_id,
            "file_id": payload["source_ref"]["external_id"],
        })

    # Process each upload as it arrives — a zip is expanded and its members
    # handled one at a time; anything else is handled as a single file.
    for upload in files:
        data = await upload.read()
        name = upload.filename or "upload"
        if is_zip(data) and name.lower().endswith(".zip"):
            # OKF detection first: a zipped OKF bundle (root okf_version
            # marker, or typed frontmatter across its markdown members) is
            # ingested as ONE knowledge bundle — concepts keep their types,
            # cross-links become graph hints — instead of as loose files.
            from ingest.okf_bundle import is_okf_bundle, load_bundle_zip

            try:
                okf_files = await asyncio.to_thread(
                    load_bundle_zip,
                    data,
                    max_file_bytes=max_file_bytes,
                    max_files=settings.ingest_max_files,
                    max_total_uncompressed_bytes=settings.ingest_max_archive_uncompressed_mb
                    * 1024 * 1024,
                )
            except ArchiveTooLarge as e:
                raise HTTPException(status_code=413, detail=str(e))
            except ArchiveError as e:
                raise HTTPException(status_code=400, detail=str(e))
            if okf_files and is_okf_bundle(okf_files):
                await _store_and_enqueue(name, data, okf_bundle=True)
                continue
            try:
                for member_name, member_data in iter_archive(
                    data,
                    max_file_bytes=max_file_bytes,
                    max_files=settings.ingest_max_files,
                    max_total_uncompressed_bytes=settings.ingest_max_archive_uncompressed_mb
                    * 1024 * 1024,
                ):
                    await _store_and_enqueue(member_name, member_data)
            except ArchiveTooLarge as e:
                raise HTTPException(status_code=413, detail=str(e))
            except ArchiveError as e:
                raise HTTPException(status_code=400, detail=str(e))
        else:
            if len(data) > max_file_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"'{name}' exceeds the {settings.ingest_max_file_mb}MB per-file limit",
                )
            await _store_and_enqueue(name, data)

    if not enqueued:
        raise HTTPException(status_code=400, detail="No ingestible files found in the upload")

    return JSONResponse(status_code=202, content={"files": enqueued, "count": len(enqueued)})


@v1_router.get("/ingest/artifacts/{file_id}")
async def v1_get_artifact(
    file_id: str,
    request: Request,
    user_id: str | None = Query(None),
):
    """Download a previously ingested artifact by id (owner-scoped).

    This is the ``url`` / retrieval handle stamped onto every file-ingested
    memory's source_ref, so an agent or user can fetch the original file back.
    """
    from fastapi.responses import FileResponse

    from ingest.storage import find_artifact

    caller = _resolve_user_id(request, user_id)
    found = await asyncio.to_thread(find_artifact, file_id, caller, settings)
    if not found:
        raise HTTPException(status_code=404, detail=f"Artifact '{file_id}' not found")
    abs_path, filename = found
    return FileResponse(abs_path, filename=filename)


@v1_router.get("/ingest/exemplars/{image_id}")
async def v1_get_exemplar_image(
    image_id: str,
    request: Request,
    user_id: str | None = Query(None),
):
    """Download a visual-exemplar image by its image id (owner-scoped).

    ``image_id`` is the ``source_ref.external_id`` stamped on the exemplar
    memory (the image's content hash). Resolution goes THROUGH the memory —
    the caller can only fetch images referenced by an exemplar they own — so
    a foreign/unknown id 404s rather than probing the store. This is the
    retrieval path a consuming agent (e.g. a chart-vision few-shot) uses to
    pull exemplar image bytes over HTTP; requires the API container to mount
    the exemplar volume (see docker-compose / EXEMPLAR_STORE_DIR).
    """
    from fastapi.responses import Response

    from adapters.trading.exemplars import find_exemplar_uri, read_exemplar_image

    if not settings.exemplar_store_enabled:
        raise HTTPException(status_code=404, detail="Exemplar store is disabled")
    caller = _resolve_user_id(request, user_id)
    uri = await asyncio.to_thread(find_exemplar_uri, _service, image_id=image_id, user_id=caller)
    if not uri:
        raise HTTPException(status_code=404, detail=f"Exemplar '{image_id}' not found")
    try:
        data = await asyncio.to_thread(read_exemplar_image, uri, settings)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=f"Exemplar '{image_id}' not found") from e
    ext = uri.rsplit(".", 1)[-1].lower()
    media = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
             "gif": "image/gif", "webp": "image/webp"}.get(ext, "application/octet-stream")
    return Response(content=data, media_type=media)


# ── Code-graph delegation (Graphify behind the NS surface) ──────────
# Roadmap F2 hard rule: the interaction interface is ALWAYS Neuralscape.
# These routes are thin delegations to the graphifyy library over a
# graph.json — the REST twins of the query_code_graph / get_code_neighbors /
# code_path MCP tools, and the resolution target of every code-graph
# memory's source_ref url. Graphs resolve by owner-scoped artifact graph_id
# (an ingested bundle) or the deployment-configured default path only —
# never an arbitrary caller-supplied filesystem path.


def _code_graph_or_501():
    """Return the code-graph query module or raise 501 when the extra is absent."""
    from adapters.code_graph import _MISSING_EXTRA_MSG, code_graph_available

    if not code_graph_available():
        raise HTTPException(status_code=501, detail=_MISSING_EXTRA_MSG)
    from adapters.code_graph import query as cg_query

    return cg_query


def _map_code_graph_error(e: Exception) -> HTTPException:
    from adapters.code_graph.query import CodeGraphNotConfigured

    if isinstance(e, CodeGraphNotConfigured):
        return HTTPException(status_code=400, detail=str(e))
    return HTTPException(status_code=404, detail=str(e))


@v1_router.get("/code-graph/query")
async def v1_code_graph_query(
    request: Request,
    question: str = Query(..., min_length=1, max_length=2000),
    mode: str = Query("bfs", pattern="^(bfs|dfs)$"),
    depth: int = Query(3, ge=1, le=6),
    token_budget: int = Query(2000, ge=100, le=20000),
    graph_id: str | None = Query(None, description="Artifact id of an ingested graph.json bundle (owner-scoped); omit for the configured default graph"),
    user_id: str | None = Query(None),
):
    """Search the code graph (BFS/DFS from scored seeds) — Graphify's query, NS's surface."""
    cg = _code_graph_or_501()
    caller = _resolve_user_id(request, user_id)
    try:
        text = await asyncio.to_thread(
            cg.query_code_graph,
            question,
            user_id=caller,
            settings=settings,
            graph_id=graph_id,
            mode=mode,
            depth=depth,
            token_budget=token_budget,
        )
    except cg.CodeGraphError as e:
        raise _map_code_graph_error(e) from e
    return {"result": text, "graph_id": graph_id}


@v1_router.get("/code-graph/neighbors")
async def v1_code_graph_neighbors(
    request: Request,
    label: str = Query(..., min_length=1, max_length=500),
    relation_filter: str = Query("", max_length=100),
    graph_id: str | None = Query(None),
    user_id: str | None = Query(None),
):
    """Direct in/out neighbors of one code-graph node, with relation + confidence tags."""
    cg = _code_graph_or_501()
    caller = _resolve_user_id(request, user_id)
    try:
        text = await asyncio.to_thread(
            cg.get_code_neighbors,
            label,
            user_id=caller,
            settings=settings,
            graph_id=graph_id,
            relation_filter=relation_filter,
        )
    except cg.CodeGraphError as e:
        raise _map_code_graph_error(e) from e
    return {"result": text, "graph_id": graph_id}


@v1_router.get("/code-graph/path")
async def v1_code_graph_path(
    request: Request,
    source: str = Query(..., min_length=1, max_length=500),
    target: str = Query(..., min_length=1, max_length=500),
    max_hops: int = Query(8, ge=1, le=32),
    graph_id: str | None = Query(None),
    user_id: str | None = Query(None),
):
    """Shortest connection path between two code-graph symbols (how does A reach B?)."""
    cg = _code_graph_or_501()
    caller = _resolve_user_id(request, user_id)
    try:
        text = await asyncio.to_thread(
            cg.code_path,
            source,
            target,
            user_id=caller,
            settings=settings,
            graph_id=graph_id,
            max_hops=max_hops,
        )
    except cg.CodeGraphError as e:
        raise _map_code_graph_error(e) from e
    return {"result": text, "graph_id": graph_id}


# ── Connectors (data-layer sources) ──────────────


def _require_vault():
    """Return the connector vault or raise 503 when connectors are disabled."""
    if _vault is None:
        raise HTTPException(
            status_code=503,
            detail="Connectors are disabled. Set CONNECTORS_ENABLED=true and NEURALSCAPE_VAULT_KEY.",
        )
    return _vault


async def _require_owned_connector(vault, connector_id: str, caller: str) -> dict:
    """Return the connector record only if ``caller`` owns it, else 404.

    Connectors are per-owner (``owner_user_id`` set at registration). A non-owner
    gets 404 — not 403 — so the endpoint can't be used as an oracle to probe
    which connector_ids exist in another user's namespace.
    """
    rec = await vault.get_redacted(connector_id)
    if rec is None or rec.get("owner_user_id") != caller:
        raise HTTPException(status_code=404, detail=f"Connector '{connector_id}' not found")
    return rec


@v1_router.post("/connectors", status_code=201)
async def v1_register_connector(req: ConnectorConfigRequest, request: Request):
    """Register (or replace) a data-layer connector instance. Secrets are encrypted at rest."""
    vault = _require_vault()
    resolved_user_id = _resolve_user_id(request, None)
    record = req.model_dump(mode="json", exclude_none=True)
    record["owner_user_id"] = resolved_user_id
    redacted = await vault.put(record)
    return JSONResponse(status_code=201, content=redacted)


@v1_router.get("/connectors")
async def v1_list_connectors(request: Request):
    """List the caller's configured connectors (credentials redacted)."""
    vault = _require_vault()
    caller = _resolve_user_id(request, None)
    connectors = [c for c in await vault.list() if c.get("owner_user_id") == caller]
    return {"connectors": connectors}


@v1_router.get("/connectors/{connector_id}")
async def v1_get_connector(connector_id: str, request: Request):
    """Get one of the caller's connector configs (credentials redacted)."""
    vault = _require_vault()
    caller = _resolve_user_id(request, None)
    return await _require_owned_connector(vault, connector_id, caller)


@v1_router.delete("/connectors/{connector_id}")
async def v1_delete_connector(connector_id: str, request: Request):
    """Delete one of the caller's connectors (does not remove ingested memories)."""
    vault = _require_vault()
    caller = _resolve_user_id(request, None)
    await _require_owned_connector(vault, connector_id, caller)
    removed = await vault.delete(connector_id)
    return {"status": "ok", "deleted": removed}


@v1_router.post("/connectors/{connector_id}/sync", status_code=202)
async def v1_sync_connector(connector_id: str, request: Request):
    """Trigger an immediate sync for one of the caller's connectors."""
    vault = _require_vault()
    caller = _resolve_user_id(request, None)
    await _require_owned_connector(vault, connector_id, caller)
    task_id = await _task_manager.enqueue_connector_sync(connector_id)
    return TaskAcceptedResponse(task_id=task_id, poll_url=f"/v1/memories/status/{task_id}")


@v1_router.post("/memories/raw/batch", status_code=202)
async def v1_store_raw_batch(req: RawMemoryBatchRequest, request: Request):
    """Store multiple pre-categorized facts in one request (memory-model v2).

    All items dispatched as a single ARQ task. Each item is independent —
    a single bad item does not block the others. Falls back to synchronous
    storage if Redis is unavailable.

    Multi-user: when authenticated with a per-user token, all items inherit
    the token's user_id (per-item user_id in the body must match if set).
    """
    # Per-item validation up front so we fail fast on bad input.
    for idx, item in enumerate(req.memories):
        if item.category not in MEMORY_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"Item {idx}: invalid category '{item.category}'.",
            )
        if item.scope == "project" and not item.project_id:
            raise HTTPException(
                status_code=400,
                detail=f"Item {idx}: project_id is required when scope='project'.",
            )

    # Resolve identity once for the whole batch. When a token is present,
    # it's authoritative: every item must agree (or be absent / empty), and
    # we overwrite each item's user_id with the token's. Without a token
    # (legacy shared-key callers), per-item body user_id is trusted.
    token_user_id = getattr(request.state, "user_id", None)
    if token_user_id:
        for idx, item in enumerate(req.memories):
            # Reject any item that explicitly tries to write under a
            # different user_id. A blank or absent value is acceptable —
            # we'll fill it from the token below.
            if item.user_id and item.user_id != token_user_id:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Item {idx}: body user_id ({item.user_id!r}) does not match "
                        f"the auth token's user_id ({token_user_id!r})."
                    ),
                )

    # Serialize datetimes so the batch survives JSON enqueue.
    items_payload: list[dict] = []
    for item in req.memories:
        d = item.model_dump(exclude_none=True)
        if "expires_at" in d and hasattr(d["expires_at"], "isoformat"):
            d["expires_at"] = d["expires_at"].isoformat()
        if token_user_id:
            # Token wins. Overwrite any per-item value (including empty
            # strings, which `setdefault` would have preserved) so a
            # caller can't submit `item.user_id=""` to sidestep the
            # token's namespace.
            d["user_id"] = token_user_id
        else:
            d.setdefault("user_id", settings.default_user_id)
        items_payload.append(d)

    # Authoritative-tier write-gate per item (synchronous 403 before enqueue).
    for idx, d in enumerate(items_payload):
        try:
            _authorize_standard_write(d["user_id"], d.get("visibility"))
        except HTTPException as he:
            raise HTTPException(status_code=he.status_code, detail=f"Item {idx}: {he.detail}") from he

    try:
        task_id = await _task_manager.enqueue_raw_batch(items=items_payload)
    except (ConnectionError, OSError) as e:
        logger.warning(f"Redis unavailable, falling back to sync batch store: {e}")
        memories = await asyncio.to_thread(_service.store_raw_batch, items_payload)
        return JSONResponse(
            status_code=200,
            content=StoreMemoryResponse(memories=memories).model_dump(exclude_none=True),
        )

    return TaskAcceptedResponse(
        task_id=task_id,
        poll_url=f"/v1/memories/status/{task_id}",
    )


@v1_router.get("/memories/status/{task_id}", response_model=TaskStatusResponse)
async def v1_get_task_status(task_id: str):
    """Poll async task status."""
    result = await _task_manager.get_status(task_id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return result


# ── Recall ────────────────────────────────────


def _meter_index_recall_bg(op: str, user_id: str, hits, response) -> None:
    """E2: measure + ledger one index-serving recall — entirely off the hot
    path (audit 27 #11).

    ``hits`` are the full MemoryResponse objects behind the index rows (their
    stored write-time token counts are the measured baseline); ``response``
    is the EXACT response model being served — the whole rendered body is
    NS-injected overhead, measured verbatim rather than approximated by the
    rows alone (never overclaim).

    The body serialization + tokenization AND the Redis ledger append run on
    the shared telemetry executor: a slow Redis or tokenizer can no longer
    delay the response, and a meter exception can never fail a successful
    recall. Trade-off (deliberate): the response no longer carries the
    per-recall ``savings`` line/detail — the honest measurement still lands
    in the ledger and surfaces via GET /v1/metrics.
    """

    def _measure_and_record() -> None:
        import savings_meter as sm

        payload = json.dumps(
            response.model_dump(exclude_none=True), default=str, ensure_ascii=False
        )
        event = sm.measure_recall(op, hits, index_payload=payload)
        if event is not None:
            sm.record_event(user_id, event)

    try:
        import telemetry

        telemetry.submit(_measure_and_record)
    except Exception:
        logger.debug("savings metering dispatch failed (non-fatal)", exc_info=True)


def _meter_full_recall_bg(op: str, user_id: str, hits) -> None:
    """E2: ledger a full-payload recall (served == baseline) off the hot path."""

    def _measure_and_record() -> None:
        import savings_meter as sm

        event = sm.measure_recall(op, hits, served_full=True)
        if event is not None:
            sm.record_event(user_id, event)

    try:
        import telemetry

        telemetry.submit(_measure_and_record)
    except Exception:
        logger.debug("savings metering dispatch failed (non-fatal)", exc_info=True)


# Union response model: Pydantic v2 smart-union validates the returned model
# instance against its exact member type (a SearchMemoryResponse can never be
# coerced into SearchIndexResponse or vice versa — IndexRow requires `title`,
# full results carry `memory`), so both modes stay first-class in OpenAPI.
@v1_router.post("/search", response_model=SearchMemoryResponse | SearchIndexResponse)
async def v1_search_memories(req: SearchMemoryRequest, request: Request):
    """Semantic search with scope/category filters.

    When project_id is provided, searches both global and project memories.
    Memory-model v2 filters (domain, observation_type, concepts) are optional.

    Multi-user: results combine the caller's personal memories with the
    shared pool. Pass `visibility="private"` to restrict to personal only,
    `visibility="shared"` to restrict to the team pool, or
    `include_shared=False` to skip the shared pool entirely.

    Retrieval economics (C1): pass `index_only=true` to get compact index
    rows ({id, title, category, glyph, age, tokens, score}) instead of full
    payloads — filter the index, then batch-fetch via
    POST /v1/memories/batch-get.
    """
    user_id = _resolve_user_id(request, req.user_id)
    try:
        results = await asyncio.to_thread(
            _service.search,
            query=req.query,
            user_id=user_id,
            project_id=req.project_id,
            categories=req.categories,
            scope=req.scope,
            limit=req.limit,
            domain=req.domain,
            observation_type=req.observation_type,
            concepts=req.concepts,
            memory_kind=req.memory_kind,
            visibility=req.visibility.value if req.visibility else None,
            include_shared=req.include_shared,
        )
        if req.index_only:
            response = SearchIndexResponse(
                results=[IndexRow(**index_row(r)) for r in results]
            )
            _meter_index_recall_bg("search_index", user_id, results, response)
            return response
        _meter_full_recall_bg("search", user_id, results)
        return SearchMemoryResponse(results=results)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 search failed")
        raise HTTPException(status_code=500, detail="Memory search failed")


@v1_router.post("/memories/batch-get", response_model=GetMemoriesResponse)
async def v1_batch_get_memories(req: GetMemoriesRequest, request: Request):
    """Batch-fetch full memory payloads by id (C1, layer 3 of the contract).

    The intended workflow: search with `index_only=true`, filter the index
    rows, then fetch only the chosen ids here (max 50 per call). Visibility
    is enforced per id — ids the caller may not read come back in `missing`,
    indistinguishable from nonexistent ones.
    """
    caller = _resolve_user_id(request, req.user_id)
    try:
        out = await asyncio.to_thread(_service.get_memories_by_ids, req.ids, caller)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 batch-get failed")
        raise HTTPException(status_code=500, detail="Failed to fetch memories") from e
    _meter_full_recall_bg("get_memories", caller, out["results"])
    return GetMemoriesResponse(results=out["results"], missing=out["missing"])


@v1_router.post("/timeline", response_model=TimelineResponse)
async def v1_timeline(req: TimelineRequest, request: Request):
    """Chronological window around an anchor memory (C2).

    `anchor` is a memory id (UUID) or a search query (resolved to its best
    vector hit). Returns ±`depth` caller-visible memories around the anchor
    in created_at order as compact index rows, the anchor row marked with
    `anchor: true`. Answers "what was happening around X?" — dream insights
    and session-context memories interleave naturally. 404 when the anchor
    can't be resolved (unknown id, unreadable id, or no search hit).
    """
    caller = _resolve_user_id(request, req.user_id)
    try:
        out = await asyncio.to_thread(
            _service.timeline,
            anchor=req.anchor,
            user_id=caller,
            depth=req.depth,
            project_id=req.project_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 timeline failed")
        raise HTTPException(status_code=500, detail="Timeline failed") from e
    if out is None:
        raise HTTPException(
            status_code=404, detail=f"Timeline anchor {req.anchor!r} could not be resolved"
        )
    anchor_id = out["anchor_id"]
    response = TimelineResponse(
        anchor_id=anchor_id,
        results=[
            IndexRow(**index_row(m, anchor=(m.id == anchor_id)))
            for m in out["memories"]
        ],
    )
    _meter_index_recall_bg("timeline", caller, out["memories"], response)
    return response


# ── Ask (C3: reasoning-tiered question answering) ──


@v1_router.post("/ask", response_model=AskMemoryResponse)
async def v1_ask_memory(req: AskMemoryRequest, request: Request):
    """Answer a question from the caller's memories (C3).

    ``reasoning_level`` (minimal | low | medium | high) jointly selects
    retrieval breadth, the answering LLM's follow-up-search budget,
    thinking depth, and the output cap. Dialectic disciplines: grep-first
    for enumeration questions, forced update-language passes, surface-both
    on contradictions (preferring the newer fact), and strict abstention —
    "I don't know" comes back as ``abstained: true``, never a fabrication.
    Citations only ever contain retrieved memory ids. Sync per NS
    convention (reads return 200 directly); each LLM call is capped by the
    tier's ``ASK_TIMEOUT_*_S``.
    """
    from ask import AskUnavailable, ask_memory

    user_id = _resolve_user_id(request, req.user_id)
    try:
        out = await ask_memory(
            _service,
            question=req.question,
            user_id=user_id,
            reasoning_level=req.reasoning_level,
            project_id=req.project_id,
        )
    except AskUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 ask failed")
        raise HTTPException(status_code=500, detail="Ask failed") from e

    # E2: ledger the ask — baseline = full content of everything retrieved
    # as evidence (precomputed from stored counts inside ask_memory), served
    # = the synthesized answer actually returned. Values are captured before
    # dispatch so the fire-and-forget task never races the pop below.
    evidence_tokens = int(out.pop("_evidence_tokens", 0) or 0)
    answer_text = out.get("answer") or ""

    def _meter_ask() -> None:
        import savings_meter as sm

        event = sm.measure_ask(evidence_tokens, answer_text)
        if event is not None:
            sm.record_event(user_id, event)

    try:
        import telemetry

        telemetry.submit(_meter_ask)
    except Exception:
        logger.debug("ask metering dispatch failed (non-fatal)", exc_info=True)
    return AskMemoryResponse(**out)


# ── Checkpoint + queue visibility (C4) ──


@v1_router.post("/checkpoint", status_code=202, response_model=CheckpointResponse)
async def v1_checkpoint(req: CheckpointRequest, request: Request):
    """Batch-save up to 25 memories + an optional session note in one call (C4).

    Per-item content-hash dedup runs synchronously BEFORE enqueue (the
    verdicts come back immediately); non-duplicates are dispatched as ONE
    background batch job — 202 + a single task id, never blocking on
    extraction. When every item is a duplicate and there's no session
    note, returns 200 with ``task_id: null`` (nothing to enqueue). Falls
    back to synchronous storage if Redis is unavailable.
    """
    import checkpoint as checkpoint_mod

    user_id = _resolve_user_id(request, req.user_id)
    try:
        prepared = await asyncio.to_thread(
            checkpoint_mod.prepare_checkpoint, _service, req, user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    verdicts = [CheckpointVerdict(**v) for v in prepared["verdicts"]]
    common = {
        "verdicts": verdicts,
        "duplicates": prepared["duplicates"],
        "session_note_included": prepared["session_note_included"],
        "enqueued": len(prepared["to_enqueue"]),
    }
    if not prepared["to_enqueue"]:
        return JSONResponse(
            status_code=200,
            content=CheckpointResponse(
                status="ok", task_id=None, poll_url=None, **common
            ).model_dump(exclude_none=True),
        )

    try:
        task_id = await _task_manager.enqueue_raw_batch(items=prepared["to_enqueue"])
    except (ConnectionError, OSError) as e:
        logger.warning(f"Redis unavailable, falling back to sync checkpoint store: {e}")
        await asyncio.to_thread(_service.store_raw_batch, prepared["to_enqueue"])
        return JSONResponse(
            status_code=200,
            content=CheckpointResponse(
                status="completed", task_id=None, poll_url=None, **common
            ).model_dump(exclude_none=True),
        )

    # E1: stream the checkpoint batch (fire-and-forget, private to the
    # caller — the per-item memory_stored events follow from the worker).
    # Audit 27 #11: publish via the telemetry executor — publish_event does
    # sync Redis I/O and must not run on the API event loop.
    from event_stream import publish_event_bg

    publish_event_bg("checkpoint_saved", {
        "user_id": user_id,
        "visibility": "private",
        "enqueued": len(prepared["to_enqueue"]),
        "duplicates": prepared["duplicates"],
        "task_id": task_id,
    })

    return CheckpointResponse(
        task_id=task_id,
        poll_url=f"/v1/memories/status/{task_id}",
        **common,
    )


# ── Live event stream (E1) + savings metrics (E2) ──


@v1_router.get("/stream")
async def v1_event_stream(request: Request, user_id: str | None = Query(default=None)):
    """SSE live feed of the caller's memory events (E1).

    ``text/event-stream`` of memory_stored, dream_actions_applied,
    insights_stored and checkpoint_saved events. Visibility is enforced at
    publish time (private events are only ever published to their owner's
    channel — see event_stream.channel_for) and re-checked per message on
    the subscribe side (event_stream.visible_to). The caller sees exactly
    two channels: their own and the shared pool. A ``: keep-alive`` comment
    goes out roughly every 20s; client disconnect tears the subscription
    down within ~1s.
    """
    import redis.asyncio as aioredis

    import event_stream as es

    caller = _resolve_user_id(request, user_id)
    if not settings.event_stream_enabled:
        raise HTTPException(status_code=503, detail="Event stream is disabled")

    client = aioredis.from_url(settings.redis_url)
    pubsub = client.pubsub()
    try:
        await pubsub.subscribe(es.SHARED_CHANNEL, f"{es.CHANNEL_PREFIX}{caller}")
    except Exception as e:
        try:
            await client.aclose()
        except Exception:
            pass
        raise HTTPException(status_code=503, detail="Event stream unavailable") from e

    async def _gen():
        try:
            async for frame in es.sse_event_stream(
                pubsub, caller, request.is_disconnected
            ):
                yield frame
        finally:
            try:
                await pubsub.unsubscribe()
                await pubsub.aclose()
            except Exception:
                pass
            try:
                await client.aclose()
            except Exception:
                pass

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@v1_router.get("/metrics")
async def v1_metrics(request: Request, user_id: str | None = Query(default=None)):
    """Cumulative token-savings totals (E2): per-caller + instance-wide.

    Sums the append-only savings ledger (Redis streams ``ns:savings:{user}``)
    via O(1) running totals. ``net_tokens_saved`` is measured and SIGNED —
    it includes every overhead charge (index rows, savings lines, and the
    per-release tool-schema constant charged once per user per UTC day) and
    may be negative. ``rederivation_savings_estimate`` is the separate,
    clearly-labeled heuristic — never blended into the measured net.
    """
    import savings_meter as sm

    caller = _resolve_user_id(request, user_id)
    snapshot = await asyncio.to_thread(sm.metrics_snapshot, caller)
    return {"status": "ok", "savings_meter": snapshot}


@v1_router.get("/queue/status", response_model=QueueStatusResponse)
async def v1_queue_status(request: Request, user_id: str | None = Query(default=None)):
    """Aggregate queue view for the caller (C4 queue visibility).

    Counts the caller's recently-enqueued tasks by live ARQ status
    (queued/processing/completed/failed, plus expired-out-of-Redis),
    reports instance-wide pending depth per queue, and a ``caught_up``
    boolean — one poll for "is my work done?" instead of one per task.
    """
    caller = _resolve_user_id(request, user_id)
    try:
        out = await _task_manager.get_queue_status(caller)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 queue status failed")
        raise HTTPException(status_code=500, detail="Queue status failed") from e
    return QueueStatusResponse(**out)


@v1_router.post("/graph/search")
async def v1_graph_search(req: GraphSearchRequest, request: Request):
    """Knowledge graph search (entities, facts, relationships)."""
    user_id = _resolve_user_id(request, req.user_id)
    try:
        results = await asyncio.to_thread(
            _service.search_graph,
            query=req.query,
            user_id=user_id,
            project_id=req.project_id,
            limit=req.limit,
            search_config=req.search_config,
        )
        return {"status": "ok", **results}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 graph search failed")
        raise HTTPException(status_code=500, detail="Graph search failed")


# ── Context ───────────────────────────────────


@v1_router.get("/context/global", response_model=ContextResponse)
async def v1_get_global_context(request: Request, user_id: str | None = Query(default=None)):
    """Get only global user context (preferences, skills, etc.)."""
    resolved_user_id = _resolve_user_id(request, user_id)
    try:
        return await asyncio.to_thread(
            _service.get_global_context,
            user_id=resolved_user_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 get_global_context failed")
        raise HTTPException(status_code=500, detail="Failed to load global context")


@v1_router.get("/context/inject")
async def v1_inject_context(
    request: Request,
    user_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    max_chars: int = Query(default=8000, ge=500, le=32000),
):
    """Return formatted markdown context for lifecycle hook injection.

    Optimized for Claude Code SessionStart hooks — returns concise markdown
    organized by category, suitable for additionalContext injection.
    """
    resolved_user_id = _resolve_user_id(request, user_id)
    try:
        if project_id:
            context = await asyncio.to_thread(
                _service.get_project_context,
                user_id=resolved_user_id,
                project_id=project_id,
            )
        else:
            context = await asyncio.to_thread(
                _service.get_global_context,
                user_id=resolved_user_id,
            )

        formatted = format_context_for_injection(
            context.categories,
            max_chars=max_chars,
            standards=context.standards,
        )
        return {"additionalContext": formatted}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 inject_context failed")
        raise HTTPException(status_code=500, detail="Failed to generate injection context")


@v1_router.post("/context/assemble", response_model=AssembleContextResponse)
async def v1_assemble_context(req: AssembleContextRequest, request: Request):
    """Token-budgeted prompt-ready context bundle (E3). Additive — the
    existing GET /v1/context/* surface is unchanged.

    Sections under the hard ``budget_tokens`` cap: recent session messages
    (~60% of the working budget), the session's rolling summary slot
    (~40%; the summarizer maintains `short`/`long` slots asynchronously as
    conversation writes cross message-count thresholds), the identity
    card, and — when ``query`` is given — compact relevant-memory index
    rows. ``format`` selects the provider shape (plain | anthropic |
    openai). Every response is ledgered through the savings meter
    (baseline = full transcript + full query-hit content vs. the bundle
    actually served). Sync read per NS convention; REST only (no MCP tool
    — the plugin consumes this endpoint).
    """
    from context_assembler import assemble_context

    user_id = _resolve_user_id(request, req.user_id)
    try:
        out = await asyncio.to_thread(
            assemble_context,
            _service,
            user_id=user_id,
            budget_tokens=req.budget_tokens,
            session_id=req.session_id,
            project_id=req.project_id,
            query=req.query,
            fmt=req.format,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 assemble_context failed")
        raise HTTPException(status_code=500, detail="Context assembly failed") from e
    detail = out.pop("savings_detail", None)
    return AssembleContextResponse(
        **out, savings_detail=SavingsDetail(**detail) if detail else None
    )


@v1_router.get("/context/{project_id}", response_model=ContextResponse)
async def v1_get_project_context(
    project_id: str,
    request: Request,
    user_id: str | None = Query(default=None),
    limit: int | None = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
):
    """Get project + global context organized by category (paginated).

    ``limit`` defaults to ``None`` (return everything) for backward
    compatibility; pass ``limit``/``offset`` to page newest-first.
    """
    resolved_user_id = _resolve_user_id(request, user_id)
    try:
        return await asyncio.to_thread(
            _service.get_project_context,
            user_id=resolved_user_id,
            project_id=project_id,
            limit=limit,
            offset=offset,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 get_project_context failed")
        raise HTTPException(status_code=500, detail="Failed to load project context")


@v1_router.get("/projects")
async def v1_list_projects(request: Request, user_id: str | None = Query(default=None)):
    """List the caller's distinct project_ids.

    Projects are implicit (a ``project_id`` is just a scoping label), so this
    derives the list from the caller's stored memories. Powers the plugin's
    `project` selection skill — especially in Claude Cowork, which has no
    working directory to infer a project from.
    """
    resolved_user_id = _resolve_user_id(request, user_id)
    try:
        projects = await asyncio.to_thread(
            _service.list_projects,
            user_id=resolved_user_id,
        )
        return {"projects": projects}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 list_projects failed")
        raise HTTPException(status_code=500, detail="Failed to list projects")


# ── Settings: custom extraction instructions (E4) ─────────────────────


def _instructions_scope(user_id: str, project_id: str | None) -> tuple[str, str]:
    """(scope, target_id) for an extraction-instructions request."""
    if project_id:
        return "project", project_id
    return "user", user_id


@v1_router.put(
    "/settings/extraction-instructions",
    response_model=ExtractionInstructionsResponse,
)
async def v1_put_extraction_instructions(
    req: ExtractionInstructionsRequest, request: Request
):
    """Set (or clear) custom extraction instructions (E4).

    Per-user guidance (no ``project_id``) is self-service — the caller sets
    their own. Project-wide guidance (``project_id`` given) shapes what
    gets extracted for EVERY member's writes to that project, so it is
    dictator-only — mirroring the standards write gate
    (``_authorize_standard_write``); unlike the `standard` tier it does not
    require STANDARDS_ENABLED, because guidance is useful without the
    authoritative memory pool. Empty/whitespace ``instructions`` clears.
    The token budget (≤ EXTRACTION_INSTRUCTIONS_MAX_TOKENS) is enforced
    here, at save time — 400 with the measured count when exceeded.
    """
    import extraction_settings as es

    if not settings.extraction_instructions_enabled:
        raise HTTPException(
            status_code=403, detail="Custom extraction instructions are disabled."
        )
    user_id = _resolve_user_id(request, req.user_id)
    if req.project_id and not settings.is_dictator(user_id):
        raise HTTPException(
            status_code=403,
            detail=(
                f"User {user_id!r} is not authorized to set project-wide "
                "extraction instructions (dictator-only, mirroring the "
                "standards write gate)."
            ),
        )
    scope, target_id = _instructions_scope(user_id, req.project_id)
    try:
        record = await asyncio.to_thread(
            es.set_instructions,
            user_id=None if req.project_id else user_id,
            project_id=req.project_id,
            instructions=req.instructions,
            updated_by=user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 put extraction-instructions failed")
        raise HTTPException(status_code=500, detail="Failed to store instructions") from e
    return ExtractionInstructionsResponse(
        scope=scope,
        target_id=target_id,
        instructions=record.get("instructions"),
        tokens=int(record.get("tokens") or 0),
        updated_at=record.get("updated_at"),
        updated_by=record.get("updated_by"),
    )


@v1_router.get(
    "/settings/extraction-instructions",
    response_model=ExtractionInstructionsResponse,
)
async def v1_get_extraction_instructions(
    request: Request,
    user_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
):
    """Read one scope's extraction instructions (E4).

    Without ``project_id``: the caller's own per-user guidance. With
    ``project_id``: the project-wide guidance (readable by any
    authenticated caller — it shapes shared extraction, so members may
    inspect what a dictator set).
    """
    import extraction_settings as es

    resolved_user_id = _resolve_user_id(request, user_id)
    scope, target_id = _instructions_scope(resolved_user_id, project_id)
    try:
        record = await asyncio.to_thread(
            es.get_instructions,
            user_id=None if project_id else resolved_user_id,
            project_id=project_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 get extraction-instructions failed")
        raise HTTPException(status_code=500, detail="Failed to read instructions") from e
    record = record or {}
    return ExtractionInstructionsResponse(
        scope=scope,
        target_id=target_id,
        instructions=record.get("instructions"),
        tokens=int(record.get("tokens") or 0),
        updated_at=record.get("updated_at"),
        updated_by=record.get("updated_by"),
    )


# ── Processes (dictator-authored authoritative playbooks) ─────────────


@v1_router.get("/processes")
async def v1_list_processes(
    request: Request, project_id: str | None = Query(default=None)
):
    """List available dictator-authored processes ({slug, title}).

    Empty unless PROCESSES_ENABLED. Powers the plugin's `process` picker.
    """
    _resolve_user_id(request, None)  # identity check (403/400 on bad token)
    try:
        processes = await asyncio.to_thread(_service.list_processes, project_id=project_id)
        return {"processes": processes}
    except Exception:
        logger.exception("v1 list_processes failed")
        raise HTTPException(status_code=500, detail="Failed to list processes")


@v1_router.get("/processes/{slug}")
async def v1_get_process(
    slug: str, request: Request, project_id: str | None = Query(default=None)
):
    """Return a full process bundle by slug (definition + ordered steps).

    404 when unknown or when PROCESSES_ENABLED is off.
    """
    _resolve_user_id(request, None)
    try:
        process = await asyncio.to_thread(_service.get_process, slug, project_id)
    except Exception:
        logger.exception("v1 get_process failed")
        raise HTTPException(status_code=500, detail="Failed to load process")
    if process is None:
        raise HTTPException(status_code=404, detail=f"Process {slug!r} not found")
    return process


# ── Manage ────────────────────────────────────


# ── OKF bundle export (G1) ───────────────────────


@v1_router.get("/export/okf")
async def v1_export_okf(
    request: Request,
    project_id: str | None = Query(default=None, max_length=100),
    scope: str | None = Query(default=None),
    visibility: str | None = Query(default=None),
    user_id: str | None = Query(default=None, max_length=100),
):
    """Download the caller's readable memories as an OKF v0.1 bundle zip.

    One concept document per live memory (frontmatter type from the
    category mapping, the full NS envelope as extension keys), per-folder
    ``index.md`` progressive disclosure, the bundle-root version marker,
    and a ``log.md`` history. Visibility is enforced by construction:
    the caller's effective identity (token > body > default — the same
    precedence every read path uses) selects the personal pool, and
    ``visibility=shared`` builds a team bundle from the shared pool
    alone, so private memories can never appear in a shared bundle.
    """
    if scope is not None and scope not in ("global", "project"):
        raise HTTPException(status_code=400, detail="scope must be 'global' or 'project'")
    if visibility is not None and visibility not in ("private", "shared"):
        raise HTTPException(status_code=400, detail="visibility must be 'private' or 'shared'")
    resolved_user_id = _resolve_user_id(request, user_id)

    from okf.export import export_bundle

    try:
        data, stats = await asyncio.to_thread(
            export_bundle,
            _service,
            user_id=resolved_user_id,
            project_id=project_id,
            scope=scope,
            visibility=visibility,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("OKF export failed")
        raise HTTPException(status_code=500, detail="Failed to build OKF bundle")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"okf-bundle-{stamp}.zip"
    from fastapi.responses import Response as FastAPIResponse

    return FastAPIResponse(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-OKF-Concepts": str(stats["concepts"]),
            "X-OKF-Files": str(stats["files"]),
        },
    )


@v1_router.get("/memories", response_model=list[MemoryResponse])
async def v1_list_memories(
    request: Request,
    user_id: str | None = Query(default=None),
    scope: str | None = Query(default=None),
    category: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    include_tombstoned: bool = Query(
        default=False,
        description=(
            "Audit escape hatch: include rows the dreaming sweep tombstoned "
            "(hidden from listings and recall by default)."
        ),
    ),
    fields: str = Query(
        default="full",
        pattern="^(full|index)$",
        description=(
            "'full' (default) returns complete rows; 'index' strips content "
            "payloads down to index-level fields (id, title, category, tags, "
            "timestamps, observation_type, token_estimate) — a fraction of "
            "the bytes for clients that only need to scan/match, e.g. the "
            "plugin file read gate (audit 27 #31)."
        ),
    ),
):
    """List memories with filters (scope, category, project_id)."""
    resolved_user_id = _resolve_user_id(request, user_id)
    try:
        rows = await asyncio.to_thread(
            _service.list_memories,
            user_id=resolved_user_id,
            scope=scope,
            category=category,
            project_id=project_id,
            limit=limit,
            include_tombstoned=include_tombstoned,
        )
        if fields == "index":
            from index_format import distill_title

            rows = [
                row.model_copy(
                    update={
                        "memory": "",
                        # Legacy rows without a write-time title still need a
                        # scannable label — distill one server-side.
                        "title": row.title or distill_title(row.memory),
                    }
                )
                for row in rows
            ]
        return rows
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 list_memories failed")
        raise HTTPException(status_code=500, detail="Failed to list memories")


@v1_router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def v1_get_memory(memory_id: str):
    """Get a single memory by ID."""
    result = await asyncio.to_thread(_service.get_memory, memory_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return result


@v1_router.get("/memories/{memory_id}/reasoning_chain")
async def v1_get_reasoning_chain(
    memory_id: str,
    max_depth: int = Query(default=3, ge=1, le=10),
):
    """Walk a memory's ``derived_from`` provenance into a reasoning tree.

    Returns ``{status, chain}`` where ``chain`` is a recursive
    ``{memory_id, content, epistemic_level, children}`` tree resolving the
    premises a derived memory (dream MERGE survivor, REM insight, or any
    write that supplied ``derived_from``) was built from. Cycle-protected
    and capped (~50 nodes); leaves may carry ``missing`` / ``cycle`` /
    ``truncated`` markers. 404 when the root memory doesn't exist.
    """
    chain = await asyncio.to_thread(_service.get_reasoning_chain, memory_id, max_depth)
    if chain is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return {"status": "ok", "chain": chain}


@v1_router.patch("/memories/{memory_id}")
@v1_router.put("/memories/{memory_id}")  # transitional alias for the old update route
async def v1_patch_memory(memory_id: str, req: PatchMemoryRequest, request: Request):
    """Partially update a memory (metadata and/or content).

    True PATCH semantics: only fields present in the request body are applied;
    an explicit ``null`` clears the field where legal. The vector-store write
    is synchronous (200 + the updated memory); any knowledge-graph work
    (content re-ingest, project/visibility partition migration) is enqueued on
    the graph queue and reported via ``graph`` / ``graph_task_id``.

    Permission model: shared memories take metadata edits from any
    authenticated user; content and visibility edits are owner-or-dictator;
    private is owner-only; standard tier is dictator-only. Dictators override
    every tier (the same admin escape hatch the delete path has).
    """
    caller = _resolve_user_id(request, req.user_id)
    changes = {k: getattr(req, k) for k in req.model_fields_set if k != "user_id"}
    if not changes:
        raise HTTPException(status_code=400, detail="No fields to update — provide at least one field")
    try:
        result = await asyncio.to_thread(_service.patch_memory, memory_id, caller, changes)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe)) from pe
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 patch_memory failed")
        raise HTTPException(status_code=500, detail="Failed to update memory") from e

    graph = result["graph"]
    graph_task_id = None
    if result.get("graph_job"):
        try:
            graph_task_id = await _task_manager.enqueue_graph_enrichment(**result["graph_job"])
            graph = graph.replace("_pending", "_queued")
        except Exception as e:
            # Never run Graphiti inline on a request thread — report honestly instead.
            logger.warning(f"Graph enqueue failed for edited memory {memory_id}: {e}")
            graph = "enqueue_failed"

    mem = result.get("memory")
    return {
        "status": "ok",
        "memory": mem.model_dump(exclude_none=True) if mem is not None else None,
        "graph": graph,
        "graph_task_id": graph_task_id,
    }


@v1_router.post("/memories/retag", status_code=202)
async def v1_retag_memories(req: RetagRequest, request: Request):
    """Bulk retag memories matching a filter set (async, 202 + poll).

    Filters AND together; ops are add_tags / remove_tags / set_category /
    set_project_id (explicit null clears). Content and visibility are not
    bulk-editable. ``dry_run=true`` returns matched/would-update counts
    synchronously without writing.
    """
    caller = _resolve_user_id(request, req.user_id)
    filters = req.filters.model_dump(exclude_none=True, mode="json")
    ops = req.ops.model_dump(exclude_none=True, mode="json")
    if "set_project_id" in req.ops.model_fields_set:
        ops["set_project_id"] = req.ops.set_project_id  # preserve explicit-null (clear)

    if req.dry_run:
        result = await asyncio.to_thread(_service.retag_memories, caller, filters, ops, True)
        result.pop("graph_jobs", None)
        return JSONResponse(status_code=200, content=result)

    try:
        task_id = await _task_manager.enqueue_retag(caller, filters, ops)
    except (ConnectionError, OSError) as e:
        if "set_project_id" in ops:
            # A project change migrates graph partitions, which needs the graph
            # queue — without Redis there is no safe sync fallback.
            raise HTTPException(
                status_code=503,
                detail="Retag with set_project_id requires the task queue (Redis unavailable)",
            ) from e
        logger.warning(f"Redis unavailable, running retag synchronously: {e}")
        result = await asyncio.to_thread(_service.retag_memories, caller, filters, ops, False)
        result.pop("graph_jobs", None)
        return JSONResponse(status_code=200, content=result)

    return TaskAcceptedResponse(
        task_id=task_id,
        poll_url=f"/v1/memories/status/{task_id}",
    )


@v1_router.delete("/memories/{memory_id}")
async def v1_delete_memory(memory_id: str, request: Request):
    """Delete a single memory by ID.

    Passes the caller identity so deletion of authoritative ``standard``-tier
    memories can be restricted to dictators.
    """
    caller = _resolve_user_id(request, None)
    try:
        return await asyncio.to_thread(_service.delete_memory, memory_id, caller)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe)) from pe
    except Exception:
        logger.exception("v1 delete_memory failed")
        raise HTTPException(status_code=500, detail="Failed to delete memory")


@v1_router.delete("/memories")
async def v1_bulk_delete_memories(req: BulkDeleteRequest, request: Request):
    """Bulk delete memories with filters."""
    user_id = _resolve_user_id(request, req.user_id)
    try:
        return await asyncio.to_thread(
            _service.delete_memories,
            user_id=user_id,
            scope=req.scope,
            category=req.category,
            project_id=req.project_id,
            filter_null_category=req.filter_null_category,
            include_shared=req.include_shared,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 bulk_delete failed")
        raise HTTPException(status_code=500, detail="Failed to delete memories")


@v1_router.get("/categories", response_model=CategoryListResponse)
async def v1_list_categories():
    """List available memory categories and their descriptions."""
    return CategoryListResponse(categories=MEMORY_CATEGORIES)


# ── Graph introspection (v1, with project_id filter) ──


@v1_router.get("/graph/nodes")
async def v1_graph_nodes(
    request: Request,
    user_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    limit: int = Query(default=50),
):
    """List entity nodes from Graphiti with project_id filter."""
    resolved_user_id = _resolve_user_id(request, user_id)
    nodes = await asyncio.to_thread(
        _service.get_graph_nodes,
        user_id=resolved_user_id,
        project_id=project_id,
        limit=limit,
    )
    return {"status": "ok", "nodes": nodes}


@v1_router.get("/graph/edges")
async def v1_graph_edges(
    request: Request,
    user_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    limit: int = Query(default=50),
):
    """List entity edges (facts) from Graphiti with project_id filter."""
    resolved_user_id = _resolve_user_id(request, user_id)
    edges = await asyncio.to_thread(
        _service.get_graph_edges,
        user_id=resolved_user_id,
        project_id=project_id,
        limit=limit,
    )
    return {"status": "ok", "edges": edges}


@v1_router.get("/graph/episodes")
async def v1_graph_episodes(
    request: Request,
    user_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    limit: int = Query(default=20),
):
    """List episodic nodes from Graphiti with project_id filter."""
    resolved_user_id = _resolve_user_id(request, user_id)
    episodes = await asyncio.to_thread(
        _service.get_graph_episodes,
        user_id=resolved_user_id,
        project_id=project_id,
        limit=limit,
    )
    return {"status": "ok", "episodes": episodes}


@v1_router.delete("/graph/episodes/junk")
async def v1_delete_junk_episodes(
    request: Request,
    user_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    dry_run: bool = Query(default=False),
):
    """Delete junk episodic nodes (raw event logs, assistant message dumps).

    Use dry_run=true to preview what would be deleted.
    """
    resolved_user_id = _resolve_user_id(request, user_id)
    try:
        result = await asyncio.to_thread(
            _service.delete_junk_episodes,
            user_id=resolved_user_id,
            project_id=project_id,
            dry_run=dry_run,
        )
        return {"status": "ok", **result}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 delete_junk_episodes failed")
        raise HTTPException(status_code=500, detail="Failed to delete junk episodes")


@v1_router.delete("/graph/episodes/{episode_uuid}")
async def v1_delete_episode(episode_uuid: str):
    """Delete a single episodic node by UUID."""
    try:
        result = await asyncio.to_thread(
            _service.delete_episode,
            episode_uuid=episode_uuid,
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"status": "ok", **result}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 delete_episode failed")
        raise HTTPException(status_code=500, detail="Failed to delete episode")


@v1_router.get("/graph/communities")
async def v1_graph_communities(
    request: Request,
    user_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    limit: int = Query(default=20),
):
    """List community nodes from Graphiti with project_id filter."""
    resolved_user_id = _resolve_user_id(request, user_id)
    communities = await asyncio.to_thread(
        _service.get_graph_communities,
        user_id=resolved_user_id,
        project_id=project_id,
        limit=limit,
    )
    return {"status": "ok", "communities": communities}


# ── Extensions ─────────────────────────────────


class EmitEventRequest(BaseModel):
    """Request body for posting events to the extension registry."""

    event_type: str = Field(description="Event type (e.g. 'memory_stored', 'session_start')")
    payload: dict = Field(default_factory=dict, description="Event payload data")


@v1_router.get("/extensions")
async def v1_list_extensions():
    """List all registered extensions with their status."""
    return {"status": "ok", "extensions": _extension_registry.list_extensions()}


@v1_router.post("/extensions/events")
async def v1_emit_extension_event(req: EmitEventRequest):
    """Post an event to all registered extensions.

    Broadcasts the event to extensions whose manifest.hooks includes
    the given event_type. Useful for external callers (e.g. OpenClaw hooks).
    For known event types, validates the payload against the expected schema.
    """
    # Validate payload for known event types
    try:
        event_enum = EventType(req.event_type)
        payload_model = EVENT_PAYLOAD_MODELS.get(event_enum)
        if payload_model is not None:
            try:
                payload_model.model_validate(req.payload)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid payload for event type '{req.event_type}': {e}",
                )
    except ValueError:
        pass  # Unknown/custom event type — allow pass-through

    result = await _extension_registry.emit_event(req.event_type, req.payload)
    return {
        "status": "ok",
        "event_type": req.event_type,
        "extensions_notified": result.notified_count,
        "responses": result.responses,
    }


# The wiki_synthesizer's /admin/synthesize endpoints were retired with the
# extension itself — dreaming (its successor) exposes its admin surface at
# /v1/extensions/dreaming/run and /v1/extensions/dreaming/status via the
# extension-route mount. See docs/DREAMING_MODE_SPEC.md.


# Mount v1 router
app.include_router(v1_router)

# Mount the built-in OAuth 2.1 Authorization Server (discovery metadata, DCR,
# consent, token). Endpoints self-disable (404) unless NEURALSCAPE_PUBLIC_URL
# and NEURALSCAPE_USER_TOKEN_SECRET are both set, so this is inert for local
# dev / Claude Code CLI. It's what lets Claude Cowork add Neuralscape as a
# custom MCP connector with a "Connect → log in" flow.
from oauth import router as oauth_router

app.include_router(oauth_router)


class _McpTrailingSlash:
    """Serve /mcp and /mcp/ identically — never redirect between them.

    The protected-resource metadata advertises the resource as
    ``{public_url}/mcp``, so Anthropic's connector POSTs to /mcp exactly.
    Starlette's mount answers that with a 307 to /mcp/, which the
    connector does not follow (and behind the tunnel the Location is
    generated as http://), so the initialize handshake dies as
    "Authorization with the MCP server failed". Rewriting the path
    before routing removes the redirect entirely.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and scope.get("path") == "/mcp":
            scope = {**scope, "path": "/mcp/", "raw_path": b"/mcp/"}
        await self.app(scope, receive, send)


# Mount MCP HTTP transport at /mcp/ for remote agent access
if settings.mcp_transport == "http":
    from mcp_server import create_mcp_http_app
    mcp_app, _mcp_session_manager = create_mcp_http_app()
    app.mount("/mcp", mcp_app)
    app.add_middleware(_McpTrailingSlash)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
