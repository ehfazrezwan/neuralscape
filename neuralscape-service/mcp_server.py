"""MCP server exposing neuralscape memory operations as 7 tools.

Supports both stdio transport (local Claude Code) and Streamable HTTP
transport (remote agent access via /mcp/ endpoint on port 8199).
"""

import asyncio
import contextlib
import json
import logging
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from config import settings
from memory_service import MemoryService
from schemas import MEMORY_CATEGORIES
from task_manager import TaskManager

logger = logging.getLogger(__name__)

server = Server("neuralscape-memory")

# Shared service instance
_service = MemoryService()

# Task manager for async memory operations (initialized at startup)
_task_manager = TaskManager()


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="recall_memories",
            description=(
                "Search across the user's global and project-specific memories using semantic search. "
                "ALWAYS call this tool before starting work on a task to load relevant context about "
                "user preferences, project conventions, tech stack, and past decisions. "
                "When project_id is provided, searches both global user memories and project-specific memories, "
                "returning the most relevant results sorted by relevance score. "
                "Results include a 'source' field: 'graph' results come from the knowledge graph and reflect "
                "the latest contradiction-resolved state; 'vector' results come from the vector store. "
                "When vector and graph results conflict, prefer graph-sourced results as authoritative."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query describing what you want to recall",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User ID to scope the search. OPTIONAL when connected over an authenticated connector — the server uses your token identity and ignores this. Set it only for local/stdio use.",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Project ID to include project-specific memories in search (also searches global)",
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional category filter. Categories: preference, personal_fact, "
                            "technical_skill, domain_knowledge, tech_stack, convention, architecture, "
                            "dependency, decision, interaction, workflow, procedure, task_context"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default: 10)",
                    },
                    "visibility": {
                        "type": "string",
                        "description": (
                            "Multi-user: restrict to 'private' (caller's own memories only) "
                            "or 'shared' (team-wide pool only). Default: both pools merged."
                        ),
                        "enum": ["private", "shared"],
                    },
                    "include_shared": {
                        "type": "boolean",
                        "description": (
                            "When false, exclude the shared team pool entirely "
                            "(search caller's private memories only). Default: true."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="remember",
            description=(
                "Store a single categorized fact about the user or project. Use this when you learn "
                "something important during a conversation that should be remembered for future sessions. "
                "Examples: user preferences, coding style choices, project decisions, tech stack info. "
                "Each memory must be assigned a category for proper organization. "
                "To correct or update an existing fact, simply store the new version — the knowledge graph "
                "will automatically detect the contradiction and expire the old fact. "
                "No need to delete the old memory first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact to remember — a clear, standalone sentence",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User ID this memory belongs to",
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Memory category. One of: preference, personal_fact, technical_skill, "
                            "domain_knowledge, tech_stack, convention, architecture, dependency, "
                            "decision, interaction, workflow, procedure, task_context"
                        ),
                        "enum": list(MEMORY_CATEGORIES.keys()),
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Project ID for project-specific memories (required for tech_stack, convention, architecture, dependency)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional free-form tags for additional organization",
                    },
                    "domain": {
                        "type": "string",
                        "description": "Memory-model v2 — high-level life-context",
                        "enum": ["coding", "research", "meeting", "writing", "ops", "personal", "general"],
                    },
                    "observation_type": {
                        "type": "string",
                        "description": "Memory-model v2 — shape of the observation, orthogonal to category",
                        "enum": [
                            "bugfix", "feature", "refactor", "decision", "discovery",
                            "gotcha", "pattern", "trade_off", "research_note",
                            "meeting_outcome", "task_plan", "fact",
                        ],
                    },
                    "concepts": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "how-it-works", "why-it-exists", "what-changed",
                                "problem-solution", "gotcha", "pattern", "trade-off",
                                "open-question", "next-step", "blocker",
                            ],
                        },
                        "maxItems": 5,
                        "description": "Memory-model v2 — controlled-vocab cross-cutting tags (1-5 items)",
                    },
                    "source_type": {
                        "type": "string",
                        "description": "Memory-model v2 — provenance",
                        "enum": ["conversation", "tool_extraction", "explicit", "imported", "compiler"],
                    },
                    "related_memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 10,
                        "description": "Memory-model v2 — UUIDs of related memories (graph linkage)",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Memory-model v2 — extractor's self-rated 0.0-1.0",
                    },
                    "expires_at": {
                        "type": "string",
                        "description": "Memory-model v2 — ISO 8601 timestamp; memory is purged after this",
                    },
                    "visibility": {
                        "type": "string",
                        "description": (
                            "Multi-user: 'private' (only the writer reads) or 'shared' "
                            "(any authenticated user reads). Defaults per-category — "
                            "preference/personal_fact/etc. default private; tech_stack/"
                            "convention/architecture/decision/etc. default shared."
                        ),
                        "enum": ["private", "shared"],
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "If true, wait for memory to be fully stored before returning. Default: false (fire-and-forget).",
                    },
                },
                "required": ["content", "category"],
            },
        ),
        Tool(
            name="remember_conversation",
            description=(
                "Extract and store multiple memories from a conversation. Pass the conversation "
                "messages and the service will use AI to identify important facts, categorize them, "
                "and store each one. Use this at the end of a productive conversation to capture "
                "all learned information at once."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                        "description": "Conversation messages to extract memories from",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User ID these memories belong to",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Project ID if conversation is project-specific",
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "If true, wait for memories to be fully stored before returning. Default: false (fire-and-forget).",
                    },
                },
                "required": ["messages"],
            },
        ),
        Tool(
            name="get_project_context",
            description=(
                "Load the full context for a project: all user preferences (global) plus "
                "project-specific memories (tech stack, conventions, architecture, decisions), "
                "organized by category. Use this at the start of a session to bootstrap your "
                "understanding of the user and their project."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User ID to load context for",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Project ID to load project-specific context for",
                    },
                },
                "required": ["project_id"],
            },
        ),
        Tool(
            name="search_knowledge_graph",
            description=(
                "Search the knowledge graph for entities, relationships, and facts. "
                "Use this for structured queries about relationships between concepts, "
                "people, technologies, or decisions. Returns graph edges (facts), "
                "entity nodes, episodes, and communities."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for the knowledge graph",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User ID to scope the search",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Optional project ID to include project graph data",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 10)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_memories",
            description=(
                "List stored memories with optional filters. Use this to inspect what "
                "has been remembered, verify stored information, or audit memory contents. "
                "Filter by scope (global/project), category, or project_id."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User ID to list memories for",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Filter by scope: 'global' or 'project'",
                        "enum": ["global", "project"],
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category",
                        "enum": list(MEMORY_CATEGORIES.keys()),
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Filter by project ID",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 100)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="delete_memories",
            description=(
                "Delete memories by ID or by filters. Use with caution — deleted memories "
                "cannot be recovered. Can delete a single memory by ID, or bulk delete by "
                "scope, category, or project_id. Bulk deletes only remove the caller's "
                "PRIVATE memories by default; shared (team) memories the caller authored "
                "are preserved. Set include_shared=true to also remove shared writes "
                "(rarely correct — confirm with the user first)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User ID whose memories to delete",
                    },
                    "memory_id": {
                        "type": "string",
                        "description": "Specific memory ID to delete (if set, ignores other filters)",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Delete memories with this scope",
                        "enum": ["global", "project"],
                    },
                    "category": {
                        "type": "string",
                        "description": "Delete memories with this category",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Delete memories for this project",
                    },
                    "filter_null_category": {
                        "type": "boolean",
                        "description": "When True, delete only memories with null/missing category instead of all",
                    },
                    "include_shared": {
                        "type": "boolean",
                        "description": (
                            "When True, also remove the caller's shared (team) "
                            "writes. Default False — shared memories are team "
                            "artifacts and one user's bulk delete shouldn't wipe "
                            "them. Confirm with the user before enabling."
                        ),
                    },
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # Identity precedence:
    #   1. The OAuth/per-user token the request authenticated with (authoritative
    #      — set by BearerAuthMiddleware for this request). Over an authenticated
    #      HTTP connector the model needn't pass user_id at all, and a mismatched
    #      arguments["user_id"] is ignored rather than trusted.
    #   2. An explicit user_id argument (stdio / local Claude Code, legacy).
    #   3. The configured default_user_id.
    from auth import current_user_id

    user_id = current_user_id.get() or arguments.get("user_id") or settings.default_user_id

    try:
        if name == "recall_memories":
            results = _service.search(
                query=arguments["query"],
                user_id=user_id,
                project_id=arguments.get("project_id"),
                categories=arguments.get("categories"),
                limit=arguments.get("limit", 10),
                visibility=arguments.get("visibility"),
                include_shared=arguments.get("include_shared", True),
            )
            output = [r.model_dump(exclude_none=True) for r in results]
            return [TextContent(type="text", text=json.dumps(output, default=str))]

        elif name == "remember":
            # Determine scope from category
            from schemas import default_scope_for_category, GLOBAL_CATEGORIES
            category = arguments["category"]
            project_id = arguments.get("project_id")
            wait = arguments.get("wait", False)

            scope = default_scope_for_category(category).value
            if project_id and category not in GLOBAL_CATEGORIES:
                scope = "project"

            # Memory-model v2 + multi-user fields (all optional)
            v2_fields = {
                "domain": arguments.get("domain"),
                "observation_type": arguments.get("observation_type"),
                "concepts": arguments.get("concepts"),
                "source_type": arguments.get("source_type"),
                "related_memory_ids": arguments.get("related_memory_ids"),
                "confidence": arguments.get("confidence"),
                "expires_at": arguments.get("expires_at"),
                "visibility": arguments.get("visibility"),
            }

            try:
                task_id = await _task_manager.enqueue_raw(
                    content=arguments["content"],
                    user_id=user_id,
                    category=category,
                    scope=scope,
                    project_id=project_id,
                    tags=arguments.get("tags"),
                    **v2_fields,
                )
            except (ConnectionError, OSError) as e:
                # Redis unavailable — fall back to synchronous storage
                logger.warning(f"Redis unavailable, falling back to sync store: {e}")
                # store_raw expects expires_at as datetime
                sync_v2 = dict(v2_fields)
                if sync_v2.get("expires_at") and isinstance(sync_v2["expires_at"], str):
                    try:
                        from datetime import datetime as _dt
                        sync_v2["expires_at"] = _dt.fromisoformat(
                            sync_v2["expires_at"].replace("Z", "+00:00")
                        )
                    except ValueError:
                        sync_v2["expires_at"] = None
                memories = _service.store_raw(
                    content=arguments["content"],
                    user_id=user_id,
                    category=category,
                    scope=scope,
                    project_id=project_id,
                    tags=arguments.get("tags"),
                    **{k: v for k, v in sync_v2.items() if v is not None},
                )
                output = [m.model_dump(exclude_none=True) for m in memories]
                return [TextContent(type="text", text=json.dumps({"status": "completed", "result": {"memories": output}, "fallback": "sync"}, default=str))]

            if wait:
                result = await _task_manager.wait_for_result(task_id)
                return [TextContent(type="text", text=json.dumps({"status": result["status"], "task_id": task_id, "result": result.get("result")}, default=str))]

            return [TextContent(type="text", text=json.dumps({"status": "accepted", "task_id": task_id}, default=str))]

        elif name == "remember_conversation":
            wait = arguments.get("wait", False)

            try:
                task_id = await _task_manager.enqueue_store(
                    messages=arguments["messages"],
                    user_id=user_id,
                    project_id=arguments.get("project_id"),
                )
            except (ConnectionError, OSError) as e:
                # Redis unavailable — fall back to synchronous storage
                logger.warning(f"Redis unavailable, falling back to sync store: {e}")
                memories = _service.extract_and_store(
                    messages=arguments["messages"],
                    user_id=user_id,
                    project_id=arguments.get("project_id"),
                )
                output = [m.model_dump(exclude_none=True) for m in memories]
                return [TextContent(type="text", text=json.dumps({"status": "completed", "result": {"memories": output}, "fallback": "sync"}, default=str))]

            if wait:
                result = await _task_manager.wait_for_result(task_id)
                return [TextContent(type="text", text=json.dumps({"status": result["status"], "task_id": task_id, "result": result.get("result")}, default=str))]

            return [TextContent(type="text", text=json.dumps({"status": "accepted", "task_id": task_id}, default=str))]

        elif name == "get_project_context":
            context = _service.get_project_context(
                user_id=user_id,
                project_id=arguments["project_id"],
            )
            # Convert to serializable format
            output = {
                "user_id": context.user_id,
                "project_id": context.project_id,
                "categories": {
                    cat: [m.model_dump(exclude_none=True) for m in memories]
                    for cat, memories in context.categories.items()
                },
            }
            return [TextContent(type="text", text=json.dumps(output, default=str))]

        elif name == "search_knowledge_graph":
            results = _service.search_graph(
                query=arguments["query"],
                user_id=user_id,
                project_id=arguments.get("project_id"),
                limit=arguments.get("limit", 10),
            )
            return [TextContent(type="text", text=json.dumps(results, default=str))]

        elif name == "list_memories":
            results = _service.list_memories(
                user_id=user_id,
                scope=arguments.get("scope"),
                category=arguments.get("category"),
                project_id=arguments.get("project_id"),
                limit=arguments.get("limit", 100),
            )
            output = [r.model_dump(exclude_none=True) for r in results]
            return [TextContent(type="text", text=json.dumps(output, default=str))]

        elif name == "delete_memories":
            memory_id = arguments.get("memory_id")
            if memory_id:
                result = _service.delete_memory(memory_id)
            else:
                result = _service.delete_memories(
                    user_id=user_id,
                    scope=arguments.get("scope"),
                    category=arguments.get("category"),
                    project_id=arguments.get("project_id"),
                    filter_null_category=arguments.get("filter_null_category", False),
                    include_shared=arguments.get("include_shared", False),
                )
            return [TextContent(type="text", text=json.dumps(result, default=str))]

        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    except Exception as e:
        logger.exception(f"MCP tool '{name}' failed")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


def _ensure_accept(accept: bytes) -> bytes:
    """Return an Accept header value containing both JSON and SSE media types.

    MCP's StreamableHTTPSessionManager 406-rejects a POST whose Accept header
    lacks ``application/json`` (in json-response mode) or lacks either type (in
    SSE mode). Claude Cowork's connector sends ``Accept: text/event-stream``
    only, so its ``initialize`` handshake 406s ("McpServerError") even though
    OAuth succeeded. Adding the missing media type(s) makes the SDK's check
    pass. Idempotent — a header that already advertises both (e.g. Claude Code
    CLI) is returned unchanged.
    """
    low = accept.lower()
    has_json = b"application/json" in low
    has_sse = b"text/event-stream" in low
    if has_json and has_sse:
        return accept
    parts = [accept.decode("latin-1").strip()] if accept.strip() else []
    if not has_json:
        parts.append("application/json")
    if not has_sse:
        parts.append("text/event-stream")
    return ", ".join(parts).encode("latin-1")


def _accepts_json(accept: bytes) -> bool:
    """Whether the client's original Accept header allows a JSON response.

    Empty/absent and ``*/*`` count as yes — that matches every client we serve
    today (Claude Code CLI sends both media types; curl defaults to ``*/*``).
    Only a client that explicitly narrows Accept to exclude JSON — Cowork's
    connector sends ``text/event-stream`` alone — answers no.
    """
    if not accept.strip():
        return True
    low = accept.lower()
    return b"application/json" in low or b"*/*" in low


class _AcceptRouter:
    """ASGI dispatcher that content-negotiates between JSON and SSE transports.

    Cowork's connector sends ``Accept: text/event-stream`` only and rejects an
    ``application/json`` response body, while the Claude Code plugin/CLI path
    has always received JSON. One global ``json_response`` mode can't serve
    both, so /mcp runs two session managers over the same MCP server: requests
    whose original Accept allows JSON keep today's JSON responses unchanged;
    SSE-only clients get an SSE response. After routing, the Accept header is
    normalized (see ``_ensure_accept``) so neither manager's 406 gate trips.
    ASGI guarantees header names are lowercased bytes, so we match
    ``b"accept"``.
    """

    def __init__(self, json_app, sse_app):
        self.json_app = json_app
        self.sse_app = sse_app

    async def __call__(self, scope, receive, send):
        app = self.json_app
        if scope.get("type") == "http":
            headers = list(scope.get("headers", []))
            current = next((v for k, v in headers if k == b"accept"), b"")
            if not _accepts_json(current):
                app = self.sse_app
            normalized = _ensure_accept(current)
            if normalized != current:
                headers = [(k, v) for k, v in headers if k != b"accept"]
                headers.append((b"accept", normalized))
                scope = {**scope, "headers": headers}
        await app(scope, receive, send)


class _SessionManagerGroup:
    """Run several StreamableHTTPSessionManagers under one lifespan context.

    main.py holds a single ``async with manager.run():`` — this keeps that
    contract while /mcp is backed by two managers (JSON + SSE modes).
    """

    def __init__(self, *managers):
        self.managers = managers

    @contextlib.asynccontextmanager
    async def run(self):
        async with contextlib.AsyncExitStack() as stack:
            for manager in self.managers:
                await stack.enter_async_context(manager.run())
            yield


def create_mcp_http_app():
    """Create a Starlette ASGI app for Streamable HTTP MCP transport.

    This is mounted on the FastAPI app at /mcp/ for remote agent access.
    The session manager's run() context must be managed by the parent app's
    lifespan (FastAPI doesn't trigger lifespan for mounted sub-apps).

    Returns (mcp_app, session_manager_group) so main.py can start the session
    managers in its own lifespan. Two managers back the mount — JSON-response
    mode for clients that accept application/json (Claude Code plugin/CLI) and
    SSE mode for Cowork's SSE-only connector — dispatched by _AcceptRouter on
    the client's original Accept header.
    """
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    json_manager = StreamableHTTPSessionManager(
        app=server,
        json_response=True,
        stateless=True,
    )
    sse_manager = StreamableHTTPSessionManager(
        app=server,
        json_response=False,
        stateless=True,
    )

    router = _AcceptRouter(json_manager.handle_request, sse_manager.handle_request)
    mcp_app = Starlette(
        routes=[Mount("/", app=router)],
    )
    return mcp_app, _SessionManagerGroup(json_manager, sse_manager)


async def run_stdio():
    """Run MCP server over stdio transport."""
    # Initialize task manager for stdio mode
    await _task_manager.connect()
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        await _task_manager.close()


if __name__ == "__main__":
    asyncio.run(run_stdio())
