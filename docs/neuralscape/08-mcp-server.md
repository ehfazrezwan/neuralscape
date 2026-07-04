---
title: MCP Server
date: 2026-05-06
tags: [reference, neuralscape, mcp, tools]
source: handwritten
---

# MCP Server

## Overview

Neuralscape ships an [MCP](https://modelcontextprotocol.io) server built on the official **MCP Python SDK** (`mcp>=1.0.0`, not FastMCP). It exposes the same `MemoryService` business logic that backs the REST API, but as a tool-only surface for AI agents. Fourteen tools cover read, write, edit, ingest, and delete operations across the global/project memory model. The server runs over two transports: **stdio** (for local Claude Code attachment) and **Streamable HTTP** (mounted at `/mcp/` on the FastAPI app for remote agents).

The server is defined in `neuralscape-service/mcp_server.py:1-476`. Module-level singletons share state with the rest of the service:

```python
server = Server("neuralscape-memory")          # mcp_server.py:23
_service = MemoryService()                      # mcp_server.py:26
_task_manager = TaskManager()                   # mcp_server.py:29
```

`_service` is the same instance referenced by the REST endpoints, so MCP and HTTP clients see the same Qdrant + Neo4j state.

## Tool surface

| # | Tool | Required inputs | Optional inputs | MemoryService entry | Mode |
|---|------|----------------|-----------------|--------------------|------|
| 1 | `recall_memories` | `query`, `user_id` | `project_id`, `categories[]`, `limit` (10) | `.search()` | sync |
| 2 | `remember` | `content`, `user_id`, `category` (enum) | `project_id`, `tags[]`, `wait` (false) | `_task_manager.enqueue_raw()` -> `.store_raw()` fallback | async + fallback |
| 3 | `remember_conversation` | `messages[{role, content}]`, `user_id` | `project_id`, `wait` (false) | `_task_manager.enqueue_store()` -> `.extract_and_store()` fallback | async + fallback |
| 4 | `get_project_context` | `user_id`, `project_id` | — | `.get_project_context()` | sync |
| 5 | `search_knowledge_graph` | `query`, `user_id` | `project_id`, `limit` (10) | `.search_graph()` | sync |
| 6 | `list_memories` | `user_id` | `scope`, `category`, `project_id`, `limit` (100) | `.list_memories()` | sync |
| 7 | `delete_memories` | `user_id` | `memory_id`, `scope`, `category`, `project_id`, `filter_null_category` | `.delete_memory()` / `.delete_memories()` | sync |
| 8 | `edit_memory` | `memory_id` | `content`, `category`, `project_id`, `tags[]`, `visibility`, v2 fields (`domain`, `observation_type`, `concepts`, `confidence`, `expires_at`) | `.patch_memory()` | sync + graph job |
| 9 | `retag_memories` | ≥1 filter, ≥1 op | filters: `scope`, `category`, `project_id`, `visibility`, `tags_contains[]`; ops: `add_tags[]`, `remove_tags[]`, `set_category`, `set_project_id`; `dry_run` | `.retag_memories()` / `enqueue_retag()` | async (dry_run sync) |

(The table lists the founding tool set plus the edit tools; the live server has 19 tools — later additions `ingest_document`, `ingest_text`, `list_projects`, `list_processes`, `get_process`, `get_reasoning_chain`, `schedule_dream`, `get_card`, `get_memories`, and `timeline` are covered in [21-document-ingestion](./21-document-ingestion.md) and the processes docs.)

All tool handlers return `list[TextContent]` containing a single `TextContent(type="text", text=json.dumps(...))`. Tool definitions are at `neuralscape-service/mcp_server.py:32-294`; dispatch lives in the `call_tool` handler at `neuralscape-service/mcp_server.py:297-434`.

### Read tools

`recall_memories`, `search_knowledge_graph`, `get_project_context`, and `list_memories` are pure reads. Their description fields tell the agent how to use them; for example, `recall_memories` instructs callers to "ALWAYS call this tool before starting work on a task to load relevant context" and explains the `source` field on results: `graph` rows reflect the latest contradiction-resolved state and should win over `vector` rows when they conflict (see [04-memory-service-core](./04-memory-service-core.md) and [03-memory-model](./03-memory-model.md)).

### Write tools

`remember` and `remember_conversation` are async by default. They enqueue an ARQ task via `TaskManager` and return `{"status": "accepted", "task_id": ...}` immediately. When the caller passes `wait: true`, the handler awaits `_task_manager.wait_for_result(task_id)` and returns the materialized result. See [07-async-pipeline](./07-async-pipeline.md) for the full enqueue and worker flow.

`remember` derives `scope` from the category before enqueueing (`mcp_server.py:315-322`): semantic categories default to `global`, project categories default to `project`, and a supplied `project_id` will upgrade a non-global category to `project` scope.

### Delete tool

`delete_memories` accepts either a single `memory_id` (which short-circuits all other filters) or a filter combination of `scope`, `category`, `project_id`, and `filter_null_category` (`mcp_server.py:415-427`). Its description warns: *"Use with caution — deleted memories cannot be recovered."*

### Edit tools

`edit_memory` patches a single memory **in place** — the memory keeps its ID, author (`owner_user_id`), and `created_at`, which is exactly what the old delete+recreate workaround destroyed. Changes are presence-keyed (an explicit `null` clears a clearable field; an omitted field is untouched), `scope` is always re-derived from the effective category + project_id, and any knowledge-graph work (content re-ingest, project/visibility partition migration) is enqueued on the graph queue and reported back as `graph` / `graph_task_id` — never run inline.

`retag_memories` is the bulk-housekeeping counterpart: AND-semantics filters select the rows, ops (`add_tags` / `remove_tags` / `set_category` / `set_project_id`) mutate them, and `dry_run: true` previews `matched`/`updated` counts synchronously. The real run returns `{"status": "accepted", "task_id"}` like every other bulk write. At least one filter is required — unfiltered whole-store sweeps are refused at the boundary.

**Permission split (both tools, enforced in `MemoryService`):** shared memories accept *metadata* edits (tags/category/project_id/v2 fields) from any authenticated teammate, but *content* and *visibility* changes are owner-or-dictator; private memories are owner-only; `standard`-tier is dictator-only. Other users' private memories never enter a retag's candidate set (the scroll filter admits only the shared/standard pools plus the caller's own rows), so even the skip counters can't leak their existence. Content edits are blocked on `passage` memories — they mirror an ingested artifact; re-ingest the corrected source instead.

## Transports

### stdio

Local agents (Claude Code, Codex, etc.) launch the server as a subprocess and exchange JSON-RPC over stdin/stdout:

```python
async def run_stdio():
    await _task_manager.connect()
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        await _task_manager.close()
```

Entry point: `uv run python mcp_server.py` (`mcp_server.py:463-475`). The task manager is connected and torn down inside `run_stdio` so async writes work for the lifetime of the process.

### Streamable HTTP

For remote agents the server is mounted on the FastAPI app at `/mcp/`. `create_mcp_http_app()` wraps the same `Server` instance in a `StreamableHTTPSessionManager` and a Starlette router:

```python
session_manager = StreamableHTTPSessionManager(
    app=server, json_response=True, stateless=True,
)
mcp_app = Starlette(routes=[Mount("/", app=session_manager.handle_request)])
```

Defined at `neuralscape-service/mcp_server.py:437-460`, mounted at `neuralscape-service/main.py:1064-1067`, and run inside the FastAPI lifespan at `neuralscape-service/main.py:130-135` (FastAPI does not propagate lifespan to mounted sub-apps, so the session manager is started by the parent). With `stateless=True` there is no server-side session persistence between requests; each call carries its own context.

## Identity & sessions

There is no MCP-level session, cookie, or login. Every tool takes a `user_id` argument; if the caller omits it, the dispatcher falls back to `settings.default_user_id` (default `"default_user"`) at `neuralscape-service/mcp_server.py:299`. Each tool call is independent.

When the optional `NEURALSCAPE_API_KEY` is set, bearer auth is enforced by FastAPI middleware in front of the mounted `/mcp/` route — auth is an HTTP-layer concern and not handled inside the MCP server itself. stdio transport runs with the trust of whoever launched the process. See [02-service-architecture](./02-service-architecture.md) for the middleware stack.

## Error handling & Redis fallback

Every tool dispatch is wrapped in a single `try`/`except` that logs the exception and returns it to the caller as JSON (`mcp_server.py:432-434`):

```python
except Exception as e:
    logger.exception(f"MCP tool '{name}' failed")
    return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
```

The two write tools have an additional layer: they catch `ConnectionError` and `OSError` from the Redis client, log a warning, and fall back to a synchronous `MemoryService` call. The response then flips to `{"status": "completed", "result": {...}, "fallback": "sync"}` so the agent knows the write happened inline (`mcp_server.py:333-345` for `remember`, `mcp_server.py:362-371` for `remember_conversation`). This keeps memory capture working even when the ARQ queue is unreachable, at the cost of higher latency for that single call.

## Notes

- **Resources and Prompts are not implemented.** The server registers only `list_tools` and `call_tool` handlers; there are no `resources/list` or `prompts/list` endpoints. This matches the project description in `CLAUDE.md` ("MCP server with 7 tools").
- **Read-only / destructive split is informal.** The MCP SDK's `Tool` type does not carry a `read_only` annotation here; classification lives in the human-readable description text. Reads: `recall_memories`, `search_knowledge_graph`, `get_project_context`, `list_memories`. Destructive: `remember`, `remember_conversation`, `delete_memories`.
- **Category enums** for `remember` and `list_memories` are drawn from `MEMORY_CATEGORIES.keys()` in `schemas.py`, so the schema stays in sync with [03-memory-model](./03-memory-model.md) additions automatically.

## Related

- [02-service-architecture](./02-service-architecture.md)
- [04-memory-service-core](./04-memory-service-core.md)
- [03-memory-model](./03-memory-model.md)
- [07-async-pipeline](./07-async-pipeline.md)
