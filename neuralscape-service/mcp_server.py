"""MCP server exposing mem0+Graphiti memory operations as tools."""

import asyncio
import json
import logging
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from config import settings

logger = logging.getLogger(__name__)

server = Server("neuralscape-memory")

# Lazy-init
_memory = None


def _get_memory():
    global _memory
    if _memory is None:
        from mem0 import Memory

        config = settings.get_mem0_config()
        _memory = Memory.from_config(config)
    return _memory


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="add_memories",
            description="Store a new memory. Provide the text content and optional user_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content to store",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User ID to scope the memory (default: default_user)",
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="search_memory",
            description="Search for relevant memories using a query string.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User ID to scope the search",
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
            description="List all stored memories for a user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User ID to list memories for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 100)",
                    },
                },
            },
        ),
        Tool(
            name="delete_memories",
            description="Delete all memories for a user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User ID whose memories to delete",
                    },
                },
            },
        ),
        Tool(
            name="search_graph",
            description="Advanced Graphiti graph search. Returns edges, nodes, episodes, and communities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User ID (group_id) to scope the search",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 10)",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    m = _get_memory()
    user_id = arguments.get("user_id", settings.default_user_id)

    if name == "add_memories":
        content = arguments["content"]
        result = m.add(
            messages=[{"role": "user", "content": content}],
            user_id=user_id,
        )
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    elif name == "search_memory":
        query = arguments["query"]
        limit = arguments.get("limit", 10)
        result = m.search(query=query, user_id=user_id, limit=limit)
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    elif name == "list_memories":
        limit = arguments.get("limit", 100)
        result = m.get_all(user_id=user_id, limit=limit)
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    elif name == "delete_memories":
        m.delete_all(user_id=user_id)
        return [TextContent(type="text", text=json.dumps({"status": "deleted"}))]

    elif name == "search_graph":
        # Use the underlying Graphiti instance for advanced search
        graphiti = None
        if hasattr(m, "graph") and hasattr(m.graph, "graphiti"):
            graphiti = m.graph.graphiti

        if graphiti is None:
            return [TextContent(type="text", text=json.dumps({"error": "Graphiti not available"}))]

        query = arguments["query"]
        limit = arguments.get("limit", 10)

        from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

        config = EDGE_HYBRID_SEARCH_RRF
        config.limit = limit

        results = await graphiti.search_(
            query=query,
            config=config,
            group_ids=[user_id],
        )

        output = {
            "edges": [{"name": e.name, "fact": e.fact} for e in results.edges],
            "nodes": [{"name": n.name, "summary": n.summary} for n in results.nodes],
        }
        return [TextContent(type="text", text=json.dumps(output, default=str))]

    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
