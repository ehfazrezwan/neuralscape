"""Neuralscape Service — lightweight FastAPI + MCP service for mem0 with Graphiti backend."""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

# Lazy-init globals
_memory = None
_graphiti = None
_bridge = None


def _get_memory():
    """Lazy-initialize mem0 Memory with Graphiti backend."""
    global _memory, _graphiti, _bridge
    if _memory is None:
        from mem0 import Memory

        config = settings.get_mem0_config()
        _memory = Memory.from_config(config)

        # Extract the Graphiti instance and async bridge from the graph store
        if hasattr(_memory, "graph") and hasattr(_memory.graph, "graphiti"):
            _graphiti = _memory.graph.graphiti
            _bridge = _memory.graph._bridge

    return _memory


def _get_graphiti():
    """Get the underlying Graphiti instance (requires _get_memory() first)."""
    _get_memory()
    return _graphiti


def _run_on_bridge(coro):
    """Run an async coroutine on the Graphiti adapter's event loop."""
    if _bridge is None:
        raise HTTPException(status_code=503, detail="Graphiti bridge not initialized")
    return _bridge.run(coro)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Neuralscape Service...")
    _get_memory()
    yield
    if _graphiti and _bridge:
        _bridge.run(_graphiti.close())
    logger.info("Neuralscape Service stopped.")


app = FastAPI(
    title="Neuralscape Memory Service",
    description="Lightweight mem0 + Graphiti memory service with REST and MCP interfaces",
    version="0.1.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────


class AddMemoryRequest(BaseModel):
    messages: list[dict] = Field(description="Messages to add (list of {role, content} dicts)")
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    metadata: dict | None = None


class SearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    limit: int = 10


class GraphSearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    limit: int = 10
    search_config: dict | None = Field(
        default=None,
        description="Optional SearchConfig dict to override default hybrid search",
    )


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "service": "neuralscape-memory"}


# ──────────────────────────────────────────────
# Core Memory Endpoints
# ──────────────────────────────────────────────


@app.post("/memories")
async def add_memory(req: AddMemoryRequest):
    """Add a memory through mem0 (vector + graph)."""
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
async def search_memories(req: SearchRequest):
    """Search memories through mem0."""
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
    """List all memories for a user."""
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
    """Delete all memories for a user."""
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


# ──────────────────────────────────────────────
# Advanced Graph Endpoints (via Graphiti directly)
# ──────────────────────────────────────────────


@app.get("/graph/nodes")
def list_graph_nodes(
    user_id: str = Query(default=None),
    limit: int = Query(default=50),
):
    """List entity nodes from Graphiti."""
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
def list_graph_edges(
    user_id: str = Query(default=None),
    limit: int = Query(default=50),
):
    """List entity edges (facts) from Graphiti."""
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
def list_graph_episodes(
    user_id: str = Query(default=None),
    limit: int = Query(default=20),
):
    """List episodic nodes from Graphiti."""
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
def list_graph_communities(
    user_id: str = Query(default=None),
    limit: int = Query(default=20),
):
    """List community nodes from Graphiti."""
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
def advanced_graph_search(req: GraphSearchRequest):
    """Advanced Graphiti search with configurable SearchConfig."""
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
