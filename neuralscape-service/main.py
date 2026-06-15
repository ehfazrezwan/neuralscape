"""Neuralscape Service — FastAPI + MCP service for mem0 with Graphiti backend.

Provides both legacy endpoints (root) and new v1 endpoints with scoping,
categories, and a shared MemoryService business logic layer.
"""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings
from extensions import ExtensionRegistry
from extensions.events import EventType, EVENT_PAYLOAD_MODELS
from logging_config import configure_logging
from memory_service import MemoryService

# Configure structured logging before anything else
configure_logging()
from context_formatter import format_context_for_injection
from schemas import (
    MEMORY_CATEGORIES,
    BulkDeleteRequest,
    CategoryListResponse,
    ContextResponse,
    GraphSearchRequest,
    MemoryResponse,
    RawMemoryBatchRequest,
    RawMemoryRequest,
    SearchMemoryRequest,
    SearchMemoryResponse,
    StoreMemoryRequest,
    StoreMemoryResponse,
    TaskAcceptedResponse,
    TaskStatusResponse,
    UpdateMemoryRequest,
)
from task_manager import TaskManager

logger = logging.getLogger(__name__)

# Shared service instance
_service = MemoryService()

# Redis-backed task manager (initialized in lifespan)
_task_manager = TaskManager()

# Extension registry (discovered + started in lifespan)
_extension_registry = ExtensionRegistry()

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
        memories = await asyncio.to_thread(_service.store_raw, **sync_kwargs)
        return JSONResponse(
            status_code=200,
            content=StoreMemoryResponse(memories=memories).model_dump(exclude_none=True),
        )

    return TaskAcceptedResponse(
        task_id=task_id,
        poll_url=f"/v1/memories/status/{task_id}",
    )


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


@v1_router.post("/search", response_model=SearchMemoryResponse)
async def v1_search_memories(req: SearchMemoryRequest, request: Request):
    """Semantic search with scope/category filters.

    When project_id is provided, searches both global and project memories.
    Memory-model v2 filters (domain, observation_type, concepts) are optional.

    Multi-user: results combine the caller's personal memories with the
    shared pool. Pass `visibility="private"` to restrict to personal only,
    `visibility="shared"` to restrict to the team pool, or
    `include_shared=False` to skip the shared pool entirely.
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
            visibility=req.visibility.value if req.visibility else None,
            include_shared=req.include_shared,
        )
        return SearchMemoryResponse(results=results)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 search failed")
        raise HTTPException(status_code=500, detail="Memory search failed")


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
        )
        return {"additionalContext": formatted}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v1 inject_context failed")
        raise HTTPException(status_code=500, detail="Failed to generate injection context")


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


# ── Manage ────────────────────────────────────


@v1_router.get("/memories", response_model=list[MemoryResponse])
async def v1_list_memories(
    request: Request,
    user_id: str | None = Query(default=None),
    scope: str | None = Query(default=None),
    category: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    """List memories with filters (scope, category, project_id)."""
    resolved_user_id = _resolve_user_id(request, user_id)
    try:
        return await asyncio.to_thread(
            _service.list_memories,
            user_id=resolved_user_id,
            scope=scope,
            category=category,
            project_id=project_id,
            limit=limit,
        )
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


@v1_router.put("/memories/{memory_id}")
async def v1_update_memory(memory_id: str, req: UpdateMemoryRequest):
    """Update a memory's content or category."""
    try:
        return await asyncio.to_thread(
            _service.update_memory,
            memory_id=memory_id,
            content=req.content,
            category=req.category,
            tags=req.tags,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("v1 update_memory failed")
        raise HTTPException(status_code=500, detail="Failed to update memory")


@v1_router.delete("/memories/{memory_id}")
async def v1_delete_memory(memory_id: str):
    """Delete a single memory by ID."""
    try:
        return await asyncio.to_thread(_service.delete_memory, memory_id)
    except Exception as e:
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


class SynthesizeRequest(BaseModel):
    """Body for POST /v1/admin/synthesize."""

    category: str | None = Field(
        default=None,
        description="Restrict synthesis to a single NeuralScape category. None = all categories.",
    )
    dry_run: bool = Field(
        default=False,
        description=(
            "When true, do everything except write to the vault and patch Neo4j. "
            "The response still reports what would have happened."
        ),
    )


@v1_router.post("/admin/synthesize")
async def v1_admin_synthesize(req: SynthesizeRequest):
    """Manually trigger one wiki-synthesis pass.

    Gated by ``WIKI_SYNTHESIZER_ENABLED`` — when disabled, the response
    contains zero pages and an explanatory error. Useful for development
    and for one-shot synthesis after a content backfill.
    """
    from extensions.wiki_synthesizer.config import synthesizer_settings
    from extensions.wiki_synthesizer.synthesizer import synthesize_all

    if not synthesizer_settings.enabled:
        return {
            "pages_created": 0,
            "pages_updated": 0,
            "memories_processed": 0,
            "pages_skipped_empty": 0,
            "errors": ["WIKI_SYNTHESIZER_ENABLED=false — set the env var to true to run"],
            "pages": [],
        }

    result = await synthesize_all(
        service=_service,
        settings=synthesizer_settings,
        only_category=req.category,
        dry_run=req.dry_run,
    )
    return {
        "pages_created": result.pages_created,
        "pages_updated": result.pages_updated,
        "memories_processed": result.memories_processed,
        "pages_skipped_empty": result.pages_skipped_empty,
        "errors": result.errors,
        "pages": [
            {
                "category": p.category,
                "group_id": p.group_id,
                "wiki_path": p.wiki_path,
                "created": p.created,
                "source_memory_count": p.source_memory_count,
            }
            for p in result.pages
        ],
    }


@v1_router.get("/admin/synthesize/status")
async def v1_admin_synthesize_status():
    """Return the most recent synthesis-run state plus current config.

    Process-local: when the API and worker are separate processes, each
    has its own ``last_run`` snapshot. The API process reports runs
    triggered through ``POST /v1/admin/synthesize``; the worker process
    reports its cron runs. Cross-process unification is a follow-up.
    """
    from extensions.wiki_synthesizer.config import synthesizer_settings
    from extensions.wiki_synthesizer.synthesizer import get_last_run_snapshot

    return {
        "enabled": synthesizer_settings.enabled,
        "cron_hours": synthesizer_settings.cron_hours,
        "max_memories_per_page": synthesizer_settings.max_memories_per_page,
        "gemini_timeout_seconds": synthesizer_settings.gemini_timeout_seconds,
        "gemini_max_retries": synthesizer_settings.gemini_max_retries,
        "attach_window_seconds": synthesizer_settings.attach_window_seconds,
        "wiki_dir": str(synthesizer_settings.wiki_dir),
        "last_run": get_last_run_snapshot(),
    }


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
