"""MCP server exposing neuralscape memory operations as 7 tools.

Supports both stdio transport (local Claude Code) and Streamable HTTP
transport (remote agent access via /mcp/ endpoint on port 8199).
"""

import asyncio
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
                        "description": "User ID to scope the search",
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
                },
                "required": ["query", "user_id"],
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
                    "wait": {
                        "type": "boolean",
                        "description": "If true, wait for memory to be fully stored before returning. Default: false (fire-and-forget).",
                    },
                },
                "required": ["content", "user_id", "category"],
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
                "required": ["messages", "user_id"],
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
                "required": ["user_id", "project_id"],
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
                "required": ["query", "user_id"],
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
                "required": ["user_id"],
            },
        ),
        Tool(
            name="delete_memories",
            description=(
                "Delete memories by ID or by filters. Use with caution — deleted memories "
                "cannot be recovered. Can delete a single memory by ID, or bulk delete by "
                "scope, category, or project_id."
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
                },
                "required": ["user_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    user_id = arguments.get("user_id", settings.default_user_id)

    try:
        if name == "recall_memories":
            results = _service.search(
                query=arguments["query"],
                user_id=user_id,
                project_id=arguments.get("project_id"),
                categories=arguments.get("categories"),
                limit=arguments.get("limit", 10),
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

            try:
                task_id = await _task_manager.enqueue_raw(
                    content=arguments["content"],
                    user_id=user_id,
                    category=category,
                    scope=scope,
                    project_id=project_id,
                    tags=arguments.get("tags"),
                )
            except (ConnectionError, OSError) as e:
                # Redis unavailable — fall back to synchronous storage
                logger.warning(f"Redis unavailable, falling back to sync store: {e}")
                memories = _service.store_raw(
                    content=arguments["content"],
                    user_id=user_id,
                    category=category,
                    scope=scope,
                    project_id=project_id,
                    tags=arguments.get("tags"),
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
                )
            return [TextContent(type="text", text=json.dumps(result, default=str))]

        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    except Exception as e:
        logger.exception(f"MCP tool '{name}' failed")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


def create_mcp_http_app():
    """Create a Starlette ASGI app for Streamable HTTP MCP transport.

    This is mounted on the FastAPI app at /mcp/ for remote agent access.
    The session manager's run() context must be managed by the parent app's
    lifespan (FastAPI doesn't trigger lifespan for mounted sub-apps).

    Returns (mcp_app, session_manager) so main.py can start the session
    manager in its own lifespan.
    """
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    session_manager = StreamableHTTPSessionManager(
        app=server,
        json_response=True,
        stateless=True,
    )

    mcp_app = Starlette(
        routes=[Mount("/", app=session_manager.handle_request)],
    )
    return mcp_app, session_manager


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
