---
title: Service Architecture & API Surface
date: 2026-05-06
tags: [reference, neuralscape, api, fastapi]
source: handwritten
---

# Service Architecture & API Surface

## Overview

Neuralscape's HTTP entry point is a single FastAPI app (`v0.2.0`) defined in `neuralscape-service/main.py:172`. It exposes two route families: a legacy mem0-compatible surface mounted at the root (`/health`, `/memories`, `/search`, `/graph/*`) and a v1 surface mounted under `/v1/*` that adds scopes, categories, and async task semantics. The same process also conditionally mounts the MCP server's Streamable HTTP transport at `/mcp/`. Background extraction is offloaded to ARQ workers (see [async-pipeline](./07-async-pipeline.md)); routes themselves are thin wrappers around [memory-service-core](./04-memory-service-core.md).

## App lifecycle

The app is wired through an `asynccontextmanager` lifespan at `neuralscape-service/main.py:112-162`.

**Startup order:**

1. `settings.validate_required()` — fail fast on missing env vars (`main.py:117`).
2. `_service._get_memory()` — eagerly initialize mem0 + Graphiti (`main.py:120`).
3. `_task_manager.connect()` — open ARQ Redis pool (`main.py:122`).
4. `_extension_registry.discover()` then `.startup_all()` then `.mount_routes(app)` (`main.py:125-127`). See [plugin-system](./09-plugin-system.md).
5. If `settings.mcp_transport == "http"`, enter the MCP `StreamableHTTPSessionManager.run()` context (`main.py:130-135`).

**Shutdown:** every step is wrapped in `asyncio.wait_for(..., timeout=10)` so a hung backend cannot block exit. Extension shutdown runs first (`main.py:140-148`), then sync mem0/async-mem0 cleanup via `asyncio.to_thread(_shutdown_sync)` (`main.py:151-159`), then `_task_manager.close()`.

**Singletons** live at module scope: `_service` (`main.py:47`), `_task_manager` (`main.py:50`), `_extension_registry` (`main.py:53`). Lazy mem0/Graphiti construction happens inside `MemoryService._get_memory()` with double-checked locking. Every blocking backend call is dispatched through `asyncio.to_thread(...)` to keep the event loop responsive.

## Middleware & auth

Only one middleware is registered:

```python
app.add_middleware(BearerAuthMiddleware)
```

`BearerAuthMiddleware` (`neuralscape-service/auth.py:23`) is a no-op when `NEURALSCAPE_API_KEY` is unset (local dev). When set, every request that is not in `PUBLIC_PATHS = {"/health", "/api/v1/health"}` must carry `Authorization: Bearer <token>`; the token is matched with `hmac.compare_digest` (`auth.py:42`). Failures return `{"detail": "..."}` with HTTP 401.

There is **no CORS, OpenTelemetry, or rate-limiting middleware**. There is also no request-ID middleware, despite structlog being configured — see [Gotchas](#gotchas).

## Route surface

### Legacy routes (root, mem0-compatible)

| Method | Path | Line | Notes |
|---|---|---|---|
| GET | `/health` | `main.py:227` | Public; pings Redis + Qdrant + Neo4j |
| POST | `/memories` | `main.py:278` | Sync `mem0.add` |
| POST | `/search` | `main.py:297` | Sync `mem0.search` |
| GET | `/memories` | `main.py:316` | List by user/agent/run |
| DELETE | `/memories` | `main.py:339` | Bulk delete by ids |
| POST | `/memories/async` | `main.py:360` | Returns 202 |
| GET | `/memories/status/{task_id}` | `main.py:375` | Poll status |
| GET | `/graph/nodes` | `main.py:385` | Graphiti nodes |
| GET | `/graph/edges` | `main.py:422` | Graphiti edges |
| GET | `/graph/episodes` | `main.py:466` | Episode log |
| GET | `/graph/communities` | `main.py:507` | Community detection |
| POST | `/graph/search` | `main.py:543` | Hybrid graph search |

### V1 routes (`/v1/*`, scoped + categorized)

| Method | Path | Line |
|---|---|---|
| POST | `/v1/memories` | `main.py:612` (202 / sync 200 fallback) |
| POST | `/v1/memories/raw` | `main.py:649` (202 / sync 200 fallback) |
| GET | `/v1/memories/status/{task_id}` | `main.py:699` |
| POST | `/v1/search` | `main.py:711` |
| POST | `/v1/graph/search` | `main.py:733` |
| GET | `/v1/context/global` | `main.py:754` |
| GET | `/v1/context/inject` | `main.py:767` |
| GET | `/v1/context/{project_id}` | `main.py:801` |
| GET | `/v1/memories` | `main.py:821` |
| GET | `/v1/memories/{memory_id}` | `main.py:844` |
| PUT | `/v1/memories/{memory_id}` | `main.py:853` |
| DELETE | `/v1/memories/{memory_id}` | `main.py:871` |
| DELETE | `/v1/memories` | `main.py:881` (bulk) |
| GET | `/v1/categories` | `main.py:898` |
| GET | `/v1/graph/nodes` | `main.py:907` |
| GET | `/v1/graph/edges` | `main.py:923` |
| GET | `/v1/graph/episodes` | `main.py:939` |
| DELETE | `/v1/graph/episodes/junk` | `main.py:955` |
| DELETE | `/v1/graph/episodes/{episode_uuid}` | `main.py:978` |
| GET | `/v1/graph/communities` | `main.py:996` |
| GET | `/v1/extensions` | `main.py:1022` |
| POST | `/v1/extensions/events` | `main.py:1028` |

Memory model details (scopes, categories) live in [memory-model](./03-memory-model.md).

## MCP HTTP transport

The MCP server is mounted as a sub-app when `MCP_TRANSPORT=http`:

```python
if settings.mcp_transport == "http":
    from mcp_server import create_mcp_http_app
    mcp_app, _mcp_session_manager = create_mcp_http_app()
    app.mount("/mcp", mcp_app)
```
(`neuralscape-service/main.py:1064-1067`)

The session manager is a `StreamableHTTPSessionManager` configured with `stateless=True` (`mcp_server.py:437-460`). Crucially, `_mcp_session_manager` is created at module scope (`main.py:109`) so the **parent** lifespan can drive `async with _mcp_session_manager.run()` — Starlette sub-apps do not inherit a parent's lifespan. See [mcp-server](./08-mcp-server.md).

## Health checks

`/health` (`main.py:227`, public) returns:

- 200 `{"status": "ok"}` when Redis, Qdrant (mem0 initialized), and Neo4j (Graphiti initialized) all check out.
- 200 `{"status": "degraded"}` if any non-critical check is non-`ok`.
- 503 `{"status": "unhealthy"}` if the vector store is `unreachable`.

`/api/v1/health` is listed as public in `auth.py:20` but **the route is never registered** — minor doc gap.

## Error handling & logging

A global exception handler converts uncaught errors to a generic 500:

```python
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
```
(`main.py:186-193`)

Per-endpoint, the pattern is `try ... except: logger.exception(...); raise HTTPException(500, "...")`. Validation failures surface as 400 from explicit checks (e.g. `/v1/memories/raw` rejects unknown categories at `main.py:656-662`).

Logging is structlog, configured in `logging_config.py`: JSON in prod (`LOG_FORMAT=json`, the default) and console in dev. Processors include `merge_contextvars`.

## Gotchas

1. **202-with-sync-fallback.** `POST /v1/memories` and `POST /v1/memories/raw` are documented as returning 202, but on `ConnectionError`/`OSError` from Redis they fall back to `extract_and_store` / `store_raw` synchronously and return **200 with the materialized memories** (`main.py:628-641`, `675-691`). Clients that branch on status code must handle both.
2. **Lazy backend init still bites.** Despite the lifespan calling `_get_memory()` at startup, mem0's vector-store and Graphiti's first request can pay extra latency on cold starts, especially the LLM client warmup. See [storage-backends](./06-storage-backends.md).
3. **`/api/v1/health` is allowlisted but undefined.** `auth.py:20` carves out the path, but no `@app.get("/api/v1/health")` exists.
4. **No request-ID correlation.** structlog ships with `merge_contextvars`, but nothing populates a per-request correlation ID — trace stitching across worker hops is manual.
5. **Module-scope MCP manager.** `_mcp_session_manager` must be created before `app = FastAPI(...)` so the lifespan can `async with` it; this is the only reason `mcp_server.create_mcp_http_app()` is split into a factory + manager.
6. **No CORS/rate limiting.** Anything beyond the bearer-token gate (e.g. browser clients on a different origin, abuse protection) must be added by a reverse proxy. See [deployment](./11-deployment.md).

## Related

- [mcp-server](./08-mcp-server.md)
- [async-pipeline](./07-async-pipeline.md)
- [memory-model](./03-memory-model.md)
- [memory-service-core](./04-memory-service-core.md)
- [deployment](./11-deployment.md)
- [00-overview](./00-overview.md)
