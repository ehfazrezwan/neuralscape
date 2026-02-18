"""Neuralscape Service — FastAPI + MCP service for mem0 with Graphiti backend.

Provides both legacy endpoints (root) and new v1 endpoints with scoping,
categories, and a shared MemoryService business logic layer.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import APIRouter, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from config import settings
from memory_service import MemoryService
from schemas import (
    MEMORY_CATEGORIES,
    BulkDeleteRequest,
    CategoryListResponse,
    ContextResponse,
    GraphSearchRequest,
    MemoryResponse,
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

# Legacy lazy-init globals (kept for backward compat with old endpoints + tests)
_memory = None
_graphiti = None
_bridge = None
_async_memory = None


def _get_memory():
    """Lazy-initialize mem0 Memory with Graphiti backend (legacy)."""
    global _memory, _graphiti, _bridge
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
    # Initialize the service (this also initializes mem0 + Graphiti)
    _service._get_memory()
    # Connect task manager to Redis
    await _task_manager.connect()

    # Start MCP HTTP session manager if enabled and connect its task manager
    if _mcp_session_manager is not None:
        from mcp_server import _task_manager as mcp_task_manager
        await mcp_task_manager.connect()
        async with _mcp_session_manager.run():
            yield
        await mcp_task_manager.close()
    else:
        yield

    await _task_manager.close()
    _service.close()
    if _async_memory and hasattr(_async_memory, "graph") and hasattr(_async_memory.graph, "graphiti"):
        _async_memory.graph._bridge.run(_async_memory.graph.graphiti.close())
    logger.info("Neuralscape Service stopped.")


app = FastAPI(
    title="Neuralscape Memory Service",
    description="Production-grade memory layer with scoped memories, categories, REST and MCP interfaces",
    version="0.2.0",
    lifespan=lifespan,
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
    return {"status": "ok", "service": "neuralscape-memory"}


@app.post("/memories")
async def add_memory(req: LegacyAddMemoryRequest):
    """Add a memory through mem0 (vector + graph). Legacy endpoint."""
    m = _get_memory()
    try:
        result = m.add(
            messages=req.messages,
            user_id=req.user_id or settings.default_user_id,
            agent_id=req.agent_id,
            run_id=req.run_id,
            metadata=req.metadata,
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.exception("add_memory failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_memories(req: LegacySearchRequest):
    """Search memories through mem0. Legacy endpoint."""
    m = _get_memory()
    try:
        result = m.search(
            query=req.query,
            user_id=req.user_id or settings.default_user_id,
            agent_id=req.agent_id,
            run_id=req.run_id,
            limit=req.limit,
        )
        return {"status": "ok", "results": result}
    except Exception as e:
        logger.exception("search_memories failed")
        raise HTTPException(status_code=500, detail=str(e))


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
        result = m.get_all(
            user_id=user_id or settings.default_user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit,
        )
        return {"status": "ok", "memories": result}
    except Exception as e:
        logger.exception("list_memories failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories")
async def delete_memories(
    user_id: str = Query(default=None),
    agent_id: str = Query(default=None),
    run_id: str = Query(default=None),
):
    """Delete all memories for a user. Legacy endpoint."""
    m = _get_memory()
    try:
        m.delete_all(
            user_id=user_id or settings.default_user_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        return {"status": "ok", "message": "All memories deleted"}
    except Exception as e:
        logger.exception("delete_memories failed")
        raise HTTPException(status_code=500, detail=str(e))


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
def list_graph_nodes_legacy(
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
        nodes = _run_on_bridge(
            EntityNode.get_by_group_ids(g.driver, group_ids=[group_id], limit=limit)
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
    except Exception:
        return {"status": "ok", "nodes": []}


@app.get("/graph/edges")
def list_graph_edges_legacy(
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
        edges = _run_on_bridge(
            EntityEdge.get_by_group_ids(g.driver, group_ids=[group_id], limit=limit)
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
    except Exception:
        return {"status": "ok", "edges": []}


@app.get("/graph/episodes")
def list_graph_episodes_legacy(
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
        episodes = _run_on_bridge(
            g.retrieve_episodes(
                reference_time=now,
                last_n=limit,
                group_ids=[group_id],
            )
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
    except Exception:
        return {"status": "ok", "episodes": []}


@app.get("/graph/communities")
def list_graph_communities_legacy(
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
        communities = _run_on_bridge(
            CommunityNode.get_by_group_ids(g.driver, group_ids=[group_id], limit=limit)
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
    except Exception:
        return {"status": "ok", "communities": []}


@app.post("/graph/search")
def advanced_graph_search_legacy(req: LegacyGraphSearchRequest):
    """Advanced Graphiti search with configurable SearchConfig. Legacy endpoint."""
    g = _get_graphiti()
    if g is None:
        raise HTTPException(status_code=503, detail="Graphiti not initialized")

    from graphiti_core.search.search_config import SearchConfig
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

    group_id = req.user_id or settings.default_user_id

    if req.search_config:
        config = SearchConfig(**req.search_config)
    else:
        config = EDGE_HYBRID_SEARCH_RRF

    config.limit = req.limit

    try:
        results = _run_on_bridge(
            g.search_(
                query=req.query,
                config=config,
                group_ids=[group_id],
            )
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
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════
# V1 API (new endpoints with scoping + categories)
# ══════════════════════════════════════════════

v1_router = APIRouter(prefix="/v1", tags=["v1"])


# ── Remember ──────────────────────────────────


@v1_router.post("/memories", response_model=TaskAcceptedResponse, status_code=202)
async def v1_store_memories(req: StoreMemoryRequest):
    """Store memories from conversation via LLM extraction (async).

    Enqueues the extraction task to a background worker and returns immediately
    with a task_id that can be polled via GET /v1/memories/status/{task_id}.
    """
    task_id = await _task_manager.enqueue_store(
        messages=req.messages,
        user_id=req.user_id,
        project_id=req.project_id,
        agent_id=req.agent_id,
        run_id=req.run_id,
    )
    return TaskAcceptedResponse(
        task_id=task_id,
        poll_url=f"/v1/memories/status/{task_id}",
    )


@v1_router.post("/memories/raw", response_model=TaskAcceptedResponse, status_code=202)
async def v1_store_raw_memory(req: RawMemoryRequest):
    """Store a single pre-categorized fact (async, no LLM extraction).

    Enqueues the storage task to a background worker and returns immediately.
    """
    if req.category not in MEMORY_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category: {req.category}. Must be one of: {list(MEMORY_CATEGORIES.keys())}",
        )
    if req.scope == "project" and not req.project_id:
        raise HTTPException(status_code=400, detail="project_id is required when scope='project'")

    task_id = await _task_manager.enqueue_raw(
        content=req.content,
        user_id=req.user_id,
        category=req.category,
        scope=req.scope,
        project_id=req.project_id,
        tags=req.tags,
        agent_id=req.agent_id,
        run_id=req.run_id,
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
async def v1_search_memories(req: SearchMemoryRequest):
    """Semantic search with scope/category filters.

    When project_id is provided, searches both global and project memories.
    """
    try:
        results = _service.search(
            query=req.query,
            user_id=req.user_id,
            project_id=req.project_id,
            categories=req.categories,
            scope=req.scope,
            limit=req.limit,
        )
        return SearchMemoryResponse(results=results)
    except Exception as e:
        logger.exception("v1 search failed")
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.post("/graph/search")
async def v1_graph_search(req: GraphSearchRequest):
    """Knowledge graph search (entities, facts, relationships)."""
    try:
        results = _service.search_graph(
            query=req.query,
            user_id=req.user_id,
            project_id=req.project_id,
            limit=req.limit,
            search_config=req.search_config,
        )
        return {"status": "ok", **results}
    except Exception as e:
        logger.exception("v1 graph search failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Context ───────────────────────────────────


@v1_router.get("/context/global", response_model=ContextResponse)
async def v1_get_global_context(user_id: str = Query(...)):
    """Get only global user context (preferences, skills, etc.)."""
    try:
        return _service.get_global_context(user_id=user_id)
    except Exception as e:
        logger.exception("v1 get_global_context failed")
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.get("/context/{project_id}", response_model=ContextResponse)
async def v1_get_project_context(
    project_id: str,
    user_id: str = Query(...),
):
    """Get full project + global context organized by category."""
    try:
        return _service.get_project_context(user_id=user_id, project_id=project_id)
    except Exception as e:
        logger.exception("v1 get_project_context failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Manage ────────────────────────────────────


@v1_router.get("/memories", response_model=list[MemoryResponse])
async def v1_list_memories(
    user_id: str = Query(...),
    scope: str | None = Query(default=None),
    category: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    """List memories with filters (scope, category, project_id)."""
    try:
        return _service.list_memories(
            user_id=user_id,
            scope=scope,
            category=category,
            project_id=project_id,
            limit=limit,
        )
    except Exception as e:
        logger.exception("v1 list_memories failed")
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def v1_get_memory(memory_id: str):
    """Get a single memory by ID."""
    result = _service.get_memory(memory_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return result


@v1_router.put("/memories/{memory_id}")
async def v1_update_memory(memory_id: str, req: UpdateMemoryRequest):
    """Update a memory's content or category."""
    try:
        return _service.update_memory(
            memory_id=memory_id,
            content=req.content,
            category=req.category,
            tags=req.tags,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("v1 update_memory failed")
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.delete("/memories/{memory_id}")
async def v1_delete_memory(memory_id: str):
    """Delete a single memory by ID."""
    try:
        return _service.delete_memory(memory_id)
    except Exception as e:
        logger.exception("v1 delete_memory failed")
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.delete("/memories")
async def v1_bulk_delete_memories(req: BulkDeleteRequest):
    """Bulk delete memories with filters."""
    try:
        return _service.delete_memories(
            user_id=req.user_id,
            scope=req.scope,
            category=req.category,
            project_id=req.project_id,
        )
    except Exception as e:
        logger.exception("v1 bulk_delete failed")
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.get("/categories", response_model=CategoryListResponse)
async def v1_list_categories():
    """List available memory categories and their descriptions."""
    return CategoryListResponse(categories=MEMORY_CATEGORIES)


# ── Graph introspection (v1, with project_id filter) ──


@v1_router.get("/graph/nodes")
async def v1_graph_nodes(
    user_id: str = Query(...),
    project_id: str | None = Query(default=None),
    limit: int = Query(default=50),
):
    """List entity nodes from Graphiti with project_id filter."""
    nodes = _service.get_graph_nodes(user_id=user_id, project_id=project_id, limit=limit)
    return {"status": "ok", "nodes": nodes}


@v1_router.get("/graph/edges")
async def v1_graph_edges(
    user_id: str = Query(...),
    project_id: str | None = Query(default=None),
    limit: int = Query(default=50),
):
    """List entity edges (facts) from Graphiti with project_id filter."""
    edges = _service.get_graph_edges(user_id=user_id, project_id=project_id, limit=limit)
    return {"status": "ok", "edges": edges}


@v1_router.get("/graph/episodes")
async def v1_graph_episodes(
    user_id: str = Query(...),
    project_id: str | None = Query(default=None),
    limit: int = Query(default=20),
):
    """List episodic nodes from Graphiti with project_id filter."""
    episodes = _service.get_graph_episodes(user_id=user_id, project_id=project_id, limit=limit)
    return {"status": "ok", "episodes": episodes}


@v1_router.get("/graph/communities")
async def v1_graph_communities(
    user_id: str = Query(...),
    project_id: str | None = Query(default=None),
    limit: int = Query(default=20),
):
    """List community nodes from Graphiti with project_id filter."""
    communities = _service.get_graph_communities(user_id=user_id, project_id=project_id, limit=limit)
    return {"status": "ok", "communities": communities}


# Mount v1 router
app.include_router(v1_router)

# Mount MCP HTTP transport at /mcp/ for remote agent access
if settings.mcp_transport == "http":
    from mcp_server import create_mcp_http_app
    mcp_app, _mcp_session_manager = create_mcp_http_app()
    app.mount("/mcp", mcp_app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
