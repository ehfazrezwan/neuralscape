"""MCP server exposing neuralscape memory operations as 22 core tools.

Core tools (22): recall_memories, get_memories, timeline, remember,
remember_conversation, ingest_document, ingest_text, get_project_context,
search_knowledge_graph, list_memories, list_projects, delete_memories,
list_processes, get_process, edit_memory, retag_memories,
get_reasoning_chain, schedule_dream, get_card, ask_memory, checkpoint,
queue_status.

Plus three code-graph delegation tools over an ingested/configured Graphify
graph.json — query_code_graph, get_code_neighbors, code_path (NS's own
surface — clients never talk to Graphify's MCP server directly) — registered
when the optional ``code-graph`` extra is installed (dev installs have it).

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

import adapters  # noqa: F401 — registers knowledge-adapter taxonomies at import (deterministic MEMORY_CATEGORIES for validation; tool enums stay core-13)
from config import settings
from memory_service import get_shared_service
from schemas import CORE_MEMORY_CATEGORIES
from task_manager import TaskManager

logger = logging.getLogger(__name__)

server = Server("neuralscape-memory")

# Shared service instance — the SAME object main.py's REST routes use when
# this server is mounted at /mcp/ (audit 27 #35: one MemoryService per
# process; previously the MCP surface cold-initialized a second full
# mem0/Graphiti stack on its first tool call).
_service = get_shared_service()

# Task manager for async memory operations (initialized at startup)
_task_manager = TaskManager()

# Process-lifetime ARQ pool for schedule_dream enqueues (audit 27 #35:
# previously a fresh Redis connection was created and torn down per call).
_arq_pool = None
_arq_pool_lock = asyncio.Lock()


async def _get_arq_pool():
    """Lazily create (once) and reuse the ARQ Redis pool for job enqueues."""
    global _arq_pool
    if _arq_pool is None:
        async with _arq_pool_lock:
            if _arq_pool is None:
                import arq

                from config import parse_redis_settings

                _arq_pool = await arq.create_pool(parse_redis_settings())
    return _arq_pool


def _meter_mcp_index_bg(op: str, user_id: str, hits, body: dict) -> None:
    """E2: measure + ledger one index-serving MCP recall — entirely off the
    hot path (audit 27 #11).

    ``hits`` are full MemoryResponse objects (stored write-time token counts
    = measured baseline); ``body`` is the EXACT response body being served
    (rows + hint + wrapper keys) — the whole thing is NS-injected overhead,
    so it is measured verbatim rather than approximated by the rows alone
    (never overclaim).

    Serialization + tokenization + the Redis ledger append run on the shared
    telemetry executor: a slow Redis/tokenizer can no longer delay the tool
    response, and a meter exception can never fail a successful recall.
    Trade-off (deliberate): the response body no longer carries the
    per-recall ``savings`` line/detail — the measurement still lands in the
    ledger and surfaces via GET /v1/metrics.
    """

    def _measure_and_record() -> None:
        import savings_meter as sm

        payload = json.dumps(body, default=str, ensure_ascii=False)
        event = sm.measure_recall(op, hits, index_payload=payload)
        if event is not None:
            sm.record_event(user_id, event)

    try:
        import telemetry

        telemetry.submit(_measure_and_record)
    except Exception:
        logger.debug("savings metering dispatch failed (non-fatal)", exc_info=True)


def _meter_mcp_full_bg(op: str, user_id: str, hits) -> None:
    """E2: ledger a full-payload MCP recall (served == baseline) off the hot path."""

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


def _standard_write_error(visibility, user_id: str) -> list[TextContent] | None:
    """Return an MCP error payload if a ``standard``-tier write isn't allowed.

    Returns ``None`` when the write is permitted (not a standard, or the caller
    is an authorized dictator). Gating happens BEFORE enqueue on every MCP write
    path so a rejected authoritative write fails fast rather than as a silent
    background job failure.
    """
    from schemas import MemoryVisibility, normalize_visibility

    if normalize_visibility(visibility) != MemoryVisibility.STANDARD.value:
        return None
    if not settings.standards_enabled:
        return [TextContent(type="text", text=json.dumps(
            {"error": "The 'standard' visibility tier is disabled (STANDARDS_ENABLED=false)."}))]
    if not settings.is_dictator(user_id):
        return [TextContent(type="text", text=json.dumps(
            {"error": f"User {user_id!r} is not authorized to write 'standard'-tier memories."}))]
    return None


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="recall_memories",
            description=(
                "Search across the user's global and project-specific memories using semantic search. "
                "ALWAYS call this tool before starting work on a task to load relevant context about "
                "user preferences, project conventions, tech stack, and past decisions. "
                "For broad scans, PREFER the token-efficient 3-layer workflow: (1) search with "
                "index_only=true to get a compact index — {id, title, category, glyph, age, tokens, "
                "score} per hit, ~50-100 tokens each instead of full payloads; (2) filter the index by "
                "title/category/age/token cost; (3) call get_memories(ids=[...]) for full payloads of "
                "the hits you need. This is guidance, not a gate: titles are lossy ~10-word summaries, "
                "so when a title looks even plausibly relevant, fetch its full content rather than "
                "ruling it out from the title alone — a full payload costs ~5-20x an index row, cheap "
                "next to missing the memory you needed. "
                "When project_id is provided, searches both global user memories and project-specific memories, "
                "returning the most relevant results sorted by relevance score. "
                "Full (non-index) results include a 'source' field: 'graph' results come from the knowledge "
                "graph and reflect the latest contradiction-resolved state; 'vector' results come from the "
                "vector store. When vector and graph results conflict, prefer graph-sourced results as "
                "authoritative. Memories ingested from a data layer carry a 'source_ref' (origin "
                "url/connector + a 'retrieval' handle {mcp_server, tool, args}); use that handle to fetch "
                "the original source or more context when a result references external content. "
                "OUTPUT FORMAT (M5 hardening): Normal recall returns a JSON array of memory objects. "
                "When code fusion is enabled (project with coding query), returns a stable JSON envelope: "
                "{fused: true, sections: {structure: {...}, semantics: {...}, memories: [...]}, ...}. "
                "This stable envelope ensures clients never receive a surprise type flip."
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
                            "Multi-user: restrict to 'private' (caller's own memories only), "
                            "'shared' (team-wide pool only), or 'standard' (authoritative "
                            "dictator-set pool). Default: private + shared + standard merged."
                        ),
                        "enum": ["private", "shared", "standard"],
                    },
                    "include_shared": {
                        "type": "boolean",
                        "description": (
                            "When false, exclude the shared team pool entirely "
                            "(search caller's private memories only). Default: true."
                        ),
                    },
                    "index_only": {
                        "type": "boolean",
                        "description": (
                            "When true, return compact index rows ({id, title, category, "
                            "glyph, age, tokens, score}) instead of full payloads — "
                            "~50-100 tokens per hit. Filter these, then call "
                            "get_memories(ids=[...]) for the few you need. Default: false."
                        ),
                    },
                    "knowledge_system": {
                        "type": "string",
                        "description": (
                            "Optional knowledge-system hint (additive). NOTE: on recall_memories "
                            "the code leg is governed by the deterministic router's fusion gate "
                            "(project config + coding signal) — passing this param does NOT force "
                            "an explicit code system here (that behavior is on the code tools: "
                            "query_code_graph / get_code_neighbors / code_path / locate / "
                            "code_impact and their REST twins). Generic recall stays base-only "
                            "unless a coding signal + an indexed code system warrant a fusion leg."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_memories",
            description=(
                "Batch-fetch FULL memory payloads by id (layer 3 of the index-first workflow: "
                "recall_memories(index_only=true) → filter the index → get_memories for the chosen ids). "
                "Returns every stored field — content, category, tags, memory-model v2 fields, provenance "
                "(derived_from, epistemic_level, source_ref), visibility, owner. Max 50 ids per call. "
                "Ids that don't exist or that you may not read (another user's private memory) are "
                "returned in 'missing'. Prefer filtering the index first on broad scans (each full "
                "payload costs ~5-20x an index row in tokens), but don't hesitate to expand any id "
                "whose title looks relevant — titles are lossy ~10-word summaries and the index "
                "can't show what a memory actually says."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 50,
                        "description": "Memory IDs to fetch (from index rows, timeline rows, or related_memory_ids)",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Caller user ID (optional under token auth)",
                    },
                },
                "required": ["ids"],
            },
        ),
        Tool(
            name="timeline",
            description=(
                "Chronological window around an anchor memory: what was happening before and after it. "
                "The anchor is a memory id (UUID) or a natural-language query (resolved to the best "
                "search hit). Returns up to ±depth caller-visible memories around the anchor in "
                "created_at order as compact index rows ({id, title, category, glyph, age, tokens}), "
                "the anchor marked with anchor:true. Dream insights and session-context memories "
                "interleave naturally. Use it for \"what was happening around X?\", standups, and "
                "weekly digests; follow up with get_memories(ids=[...]) for the rows worth reading "
                "in full."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "anchor": {
                        "type": "string",
                        "description": "Memory ID (UUID) or a search query to anchor the window on",
                    },
                    "depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                        "description": "Memories to include on each side of the anchor (default 10)",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Optional project scope (includes that project's + global memories)",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Caller user ID (optional under token auth)",
                    },
                },
                "required": ["anchor"],
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
                            "decision, interaction, workflow, procedure, task_context. Categories "
                            "registered by an installed knowledge adapter (e.g. the trading "
                            "taxonomy) are also accepted server-side even though only the core 13 "
                            "are advertised here."
                        ),
                        "enum": list(CORE_MEMORY_CATEGORIES),
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
                    "derived_from": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 10,
                        "description": (
                            "Provenance: memory IDs this fact was derived from "
                            "(premises). Walkable later via get_reasoning_chain."
                        ),
                    },
                    "epistemic_level": {
                        "type": "string",
                        "description": (
                            "How this fact is known: 'explicit' (directly stated), "
                            "'deductive' (entailed by the derived_from premises), "
                            "'inductive' (pattern across premises), or 'reflection' "
                            "(higher-order insight). Leave unset when unknown."
                        ),
                        "enum": ["explicit", "deductive", "inductive", "reflection"],
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
                    "occurred_at": {
                        "type": "string",
                        "description": (
                            "Event time (ISO 8601): when the remembered fact/event "
                            "actually happened, for historical ingestion (imported "
                            "journals, old chat exports). Omit when unknown — readers "
                            "fall back to created_at (storage time). Future dates are "
                            "rejected beyond a small clock-skew allowance."
                        ),
                    },
                    "visibility": {
                        "type": "string",
                        "description": (
                            "Multi-user: 'private' (only the writer reads), 'shared' "
                            "(any authenticated user reads), or 'standard' (an authoritative "
                            "org rule — writable ONLY by a dictator; requires STANDARDS_ENABLED). "
                            "Defaults per-category — preference/personal_fact/etc. default "
                            "private; tech_stack/convention/architecture/decision/etc. default "
                            "shared. 'standard' is never a default; set it explicitly."
                        ),
                        "enum": ["private", "shared", "standard"],
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
            name="ingest_document",
            description=(
                "Ingest a document from a data layer (a Google Drive file, a Notion page, "
                "an API response, anything you can fetch) into memory. The content is chunked "
                "into verbatim passages AND distilled into atomic facts; every resulting memory "
                "is stamped with a 'source_ref' provenance descriptor — where it came from plus a "
                "structured retrieval handle {mcp_server, tool, args} so a future agent can re-fetch "
                "the original. Use this to index external content the user wants remembered. "
                "Pass the source descriptor so the memory can point back to its origin."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The full document text to ingest",
                    },
                    "source": {
                        "type": "object",
                        "description": (
                            "Provenance descriptor. connector_id + connector_type are required; "
                            "include url/title/external_id and a 'retrieval' handle so agents can re-fetch."
                        ),
                        "properties": {
                            "connector_id": {"type": "string", "description": "Connector instance id, e.g. 'notion-personal'"},
                            "connector_type": {"type": "string", "enum": ["google_drive", "notion", "generic_rest", "mcp"]},
                            "external_id": {"type": "string", "description": "Stable id in the source system"},
                            "url": {"type": "string", "description": "Human/clickable link"},
                            "title": {"type": "string"},
                            "retrieval": {
                                "type": "object",
                                "description": "How to re-fetch: which MCP server + tool + args",
                                "properties": {
                                    "mcp_server": {"type": "string"},
                                    "tool": {"type": "string"},
                                    "args": {"type": "object"},
                                },
                            },
                        },
                        "required": ["connector_id", "connector_type"],
                    },
                    "category": {
                        "type": "string",
                        "description": "Category for produced memories (default: domain_knowledge)",
                        "enum": list(CORE_MEMORY_CATEGORIES),
                    },
                    "project_id": {"type": "string", "description": "Project id (sets project scope)"},
                    "scope": {"type": "string", "enum": ["global", "project"]},
                    "visibility": {"type": "string", "enum": ["private", "shared", "standard"]},
                    "extract_facts": {"type": "boolean", "description": "Also run LLM extraction for distilled facts (default true)"},
                    "index_passages": {"type": "boolean", "description": "Chunk + store verbatim passages (default true)"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "adapter": {"type": "string", "description": "Knowledge adapter selecting taxonomy/chunker/extractor/graph-ontology (e.g. 'default', 'trading_strategy'). Default 'default'."},
                    "wait": {"type": "boolean", "description": "Wait for ingest to finish before returning. Default false."},
                },
                "required": ["content", "source"],
            },
        ),
        Tool(
            name="ingest_text",
            description=(
                "Manually provide a block of context to remember — a first-class ingestion path. "
                "Use this when the user pastes notes, documentation, a transcript, or any longer "
                "passage they want indexed. The text is persisted as a Markdown artifact on the "
                "server (organized by user/project/category) and the produced memories reference "
                "it, so it's fully traceable. It's chunked into verbatim passages AND distilled "
                "into atomic facts (same pipeline as ingest_document). "
                "For a single short fact prefer 'remember'; for pasted longer context use this."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The context text to ingest"},
                    "title": {"type": "string", "description": "Optional human label for this context"},
                    "user_id": {
                        "type": "string",
                        "description": "User ID. OPTIONAL over an authenticated connector — the token identity is used and this is ignored.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category for produced memories (default: domain_knowledge)",
                        "enum": list(CORE_MEMORY_CATEGORIES),
                    },
                    "project_id": {"type": "string", "description": "Project id (sets project scope)"},
                    "scope": {"type": "string", "enum": ["global", "project"]},
                    "visibility": {"type": "string", "enum": ["private", "shared", "standard"]},
                    "extract_facts": {"type": "boolean", "description": "Also run LLM extraction for distilled facts (default true)"},
                    "index_passages": {"type": "boolean", "description": "Chunk + store verbatim passages (default true)"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "adapter": {"type": "string", "description": "Knowledge adapter selecting taxonomy/chunker/extractor/graph-ontology (e.g. 'default', 'trading_strategy'). Default 'default'."},
                    "wait": {"type": "boolean", "description": "Wait for ingest to finish before returning. Default false."},
                },
                "required": ["content"],
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
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 25,
                        "description": "Max memories to return this page (default 25, newest first). Use with offset to page through a large project, or raise it to pull more at once.",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "Number of (newest-first) memories to skip (default 0).",
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
                        "enum": list(CORE_MEMORY_CATEGORIES),
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Filter by project ID",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 100)",
                    },
                    "include_tombstoned": {
                        "type": "boolean",
                        "description": (
                            "Audit escape hatch: include rows the dreaming sweep "
                            "tombstoned (hidden by default). Default false."
                        ),
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="list_projects",
            description=(
                "List the distinct projects the user can scope memory to — their own "
                "private projects plus all team-shared projects. Use this to let the user "
                "pick which project to scope memory to, especially when there is no "
                "working directory to infer one from (e.g. in Claude Cowork). Projects "
                "are implicit: a project exists once any memory is stored under its "
                "project_id, so a brand-new project name is valid and will be created on "
                "the first remember() call. Returns a sorted list of project_id strings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User ID to list projects for",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="list_processes",
            description=(
                "List the team processes available to run — named, authoritative "
                "playbooks defined by a Neuralscape dictator (e.g. a standard "
                "consulting workflow). Use this when the user wants to start or pick a "
                "defined process, or describes a task that might match one. Returns "
                "[{slug, title, description}] — match the user's request against title "
                "+ description to pick the best process, then call get_process(slug). "
                "Empty when the process feature is disabled."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Caller user ID (optional under token auth)"},
                    "project_id": {"type": "string", "description": "Optional project scope"},
                },
                "required": [],
            },
        ),
        Tool(
            name="get_process",
            description=(
                "Load a dictator-authored process playbook by slug, pulling in ALL the "
                "standards recorded for it: its definition, ordered steps, and guidelines "
                "(rules/gates/tone constraints). These are AUTHORITATIVE — when running a "
                "process, follow its steps in order, honor its guidelines, and let them "
                "override personal preferences on conflict. Returns {slug, title, "
                "definition, steps[], guidelines[]}."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "slug": {"type": "string", "description": "Process slug from list_processes"},
                    "user_id": {"type": "string", "description": "Caller user ID (optional under token auth)"},
                    "project_id": {"type": "string", "description": "Optional project scope"},
                },
                "required": ["slug"],
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
        Tool(
            name="edit_memory",
            description=(
                "Partially update an existing memory in place — the memory keeps its ID, "
                "author, and creation time (no lossy delete+recreate). Only the fields you "
                "pass are changed; pass null to clear a clearable field (project_id, tags, "
                "expires_at, domain, observation_type, concepts, confidence). Scope is "
                "re-derived from category + project_id automatically. Permission model: "
                "SHARED memories accept metadata edits (tags/category/project_id/v2 fields) "
                "from any authenticated teammate — housekeeping is collaborative — but "
                "content and visibility changes are owner-or-dictator only. PRIVATE "
                "memories are owner-only; 'standard' tier is dictator-only. Content edits "
                "are blocked on 'passage' memories (verbatim chunks of an ingested "
                "artifact — re-ingest the corrected source instead). Graph impact is "
                "handled automatically: content or project/visibility changes enqueue a "
                "background knowledge-graph re-ingest (reported as graph_task_id)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "ID of the memory to edit"},
                    "content": {"type": "string", "description": "New memory text (re-embeds; owner/dictator only)"},
                    "category": {"type": "string", "description": "New category (must be a registered category)"},
                    "project_id": {"type": ["string", "null"], "description": "New project ID; null clears it"},
                    "tags": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Full replacement tags list; null clears"},
                    "visibility": {"type": "string", "enum": ["private", "shared", "standard"], "description": "New visibility tier (owner/dictator only; 'standard' is dictator-only)"},
                    "domain": {"type": ["string", "null"], "description": "Memory-model v2 life-context domain"},
                    "observation_type": {"type": ["string", "null"], "description": "Memory-model v2 observation shape"},
                    "concepts": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Memory-model v2 concept tags (max 5)"},
                    "confidence": {"type": ["number", "null"], "description": "Memory-model v2 confidence 0.0-1.0"},
                    "expires_at": {"type": ["string", "null"], "description": "ISO 8601 expiry; null clears"},
                    "user_id": {"type": "string", "description": "Caller user ID (optional under token auth)"},
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="get_reasoning_chain",
            description=(
                "Walk the provenance chain of a derived memory: resolve the "
                "'derived_from' premise memories recursively into a tree of "
                "{memory_id, content, epistemic_level, children}. Use this to "
                "audit WHY the system believes a derived fact — e.g. a dream-"
                "consolidation survivor or a reflection insight — before "
                "acting on it. epistemic_level tells you how each node is "
                "known: 'explicit' (directly stated), 'deductive' (entailed "
                "by its premises), 'inductive' (pattern across premises), "
                "'reflection' (higher-order insight). Cycle-protected and "
                "capped (~50 nodes); leaf nodes may carry missing/cycle/"
                "truncated markers instead of children."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory whose reasoning chain to walk",
                    },
                    "max_depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3,
                        "description": "How many levels of premises to resolve (default 3)",
                    },
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="retag_memories",
            description=(
                "Bulk-edit organizational metadata on every memory matching a filter set — "
                "the housekeeping tool for stamping a project code or tag across many "
                "memories at once. Filters AND together (scope, category, project_id, "
                "visibility, tags_contains); at least one is required (no whole-store "
                "sweeps). Ops: add_tags, remove_tags, set_category, set_project_id "
                "(null clears the project). Content and visibility are NOT bulk-editable. "
                "Memories keep their ID, author, and creation time. Other users' private "
                "memories are never touched (or counted); per-row permission and validity "
                "violations are skipped and reported as skipped_forbidden/skipped_invalid. "
                "Start with dry_run=true to preview matched/updated counts, then run for "
                "real (returns a task_id to poll — writes are async)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["global", "project"], "description": "Filter: memories with this scope"},
                    "category": {"type": "string", "description": "Filter: memories with this category"},
                    "project_id": {"type": "string", "description": "Filter: memories in this project"},
                    "visibility": {"type": "string", "enum": ["private", "shared", "standard"], "description": "Filter: memories at this visibility tier"},
                    "tags_contains": {"type": "array", "items": {"type": "string"}, "description": "Filter: memories carrying ALL of these tags"},
                    "add_tags": {"type": "array", "items": {"type": "string"}, "description": "Op: tags to add to each matched memory"},
                    "remove_tags": {"type": "array", "items": {"type": "string"}, "description": "Op: tags to remove from each matched memory"},
                    "set_category": {"type": "string", "description": "Op: replace the category on each matched memory"},
                    "set_project_id": {"type": ["string", "null"], "description": "Op: replace the project ID (null clears it)"},
                    "dry_run": {"type": "boolean", "description": "Preview counts without writing (default false)"},
                    "user_id": {"type": "string", "description": "Caller user ID (optional under token auth)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="schedule_dream",
            description=(
                "Schedule one dreaming consolidation sweep (A3-lite manual trigger). "
                "Enqueues the sweep onto the background graph worker — the same path "
                "as POST /v1/extensions/dreaming/run — and returns immediately with a "
                "job id; poll /v1/extensions/dreaming/status for the DreamRun result. "
                "The sweep merges duplicates, invalidates contradictions, prunes noise "
                "(reversible tombstones), reframes stale tenses and stores reflection "
                "insights, per pool. Per-pool gates still apply: time since last dream, "
                "a settling guard (pools written to in the last ~30 minutes defer with "
                "status 'settling'), and a minimum-new-memories volume gate — pass "
                "force=true to bypass the gates (never the pool lock). Requires "
                "DREAMING_ENABLED=true unless force=true."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pool": {
                        "type": "string",
                        "description": (
                            "Restrict the sweep to one pool key (e.g. 'shared', "
                            "'shared--project--<pid>', 'user--<uid>', "
                            "'user--<uid>--project--<pid>'). Omit to sweep all pools."
                        ),
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Plan and report every action without writing anything (default false)",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Bypass the time/settling/volume gates — never the pool lock (default false)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_card",
            description=(
                "Fetch the pinned identity card for a user or project — a grammar-"
                "constrained grounding artifact (max 40 lines of 'IDENTITY: / "
                "ATTRIBUTE: / RELATIONSHIP: / INSTRUCTION:') maintained by the "
                "dreaming sweep. Inject it at SESSION START for always-available "
                "grounding: who the user/project is, stable traits, and standing "
                "instructions — without spending a search. With no arguments it "
                "returns the calling user's card; pass project_id for a project's "
                "card. Cards are pinned artifacts, never searchable memories; "
                "returns an error when no card exists yet (the nightly sweep "
                "builds them)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "Project whose card to fetch (mutually exclusive with pool)",
                    },
                    "pool": {
                        "type": "string",
                        "description": (
                            "Advanced: explicit pool key ('user--<uid>' or "
                            "'shared--project--<pid>'). Overrides project_id."
                        ),
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Caller user ID (optional under token auth)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="ask_memory",
            description=(
                "Ask a question and get a direct ANSWER synthesized from the caller's stored "
                "memories (instead of raw search hits — use recall_memories when you want the "
                "memories themselves). reasoning_level jointly selects retrieval breadth, the "
                "answering model's follow-up-search budget, thinking depth, and answer length: "
                "'minimal' = one search + direct answer (fastest/cheapest); 'low' = adds a forced "
                "update-language pass + 1 follow-up search; 'medium' = adds a grep-style exact-"
                "keyword pass + 2 follow-ups; 'high' = full iterative search loop (4 follow-ups). "
                "Disciplines baked in: enumeration/counting questions get exact-match passes and a "
                "dedup table before any count; newer facts supersede older ones; contradictions "
                "are surfaced BOTH-with-timestamps preferring the newer; \"I don't know\" is "
                "returned honestly (abstained: true) rather than fabricating. The answer cites "
                "memory ids inline and 'citations' only ever contains ids that were actually "
                "retrieved."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer from stored memories",
                    },
                    "reasoning_level": {
                        "type": "string",
                        "enum": ["minimal", "low", "medium", "high"],
                        "default": "low",
                        "description": (
                            "Effort knob: minimal (single search, direct answer), low, "
                            "medium, high (iterative search loop + exact-match passes)"
                        ),
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Optional project scope (searches that project's + global memories)",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Caller user ID (optional under token auth)",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="checkpoint",
            description=(
                "Batch-save up to 25 memories PLUS an optional structured session note in ONE "
                "call (one tool card, one background task id) — use this at the end of a session "
                "instead of many individual remember calls. Each item is {content, category, plus "
                "any memory-model v2 fields (tags, domain, observation_type, concepts, "
                "confidence, visibility, project_id, ...)}. Per-item content-hash dedup runs "
                "BEFORE enqueue and the verdicts come back immediately (verdict: 'new' | "
                "'duplicate' with the existing_id), then the non-duplicates are stored as one "
                "async batch job (poll the task_id, or just continue — writes never block on "
                "extraction). session_note is {request, investigated, learned, completed, "
                "next_steps} and is "
                "stored as a single task_context memory (observation_type=meeting_outcome) so the "
                "next session picks it up."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "maxItems": 25,
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "The fact to remember"},
                                "category": {
                                    "type": "string",
                                    "enum": list(CORE_MEMORY_CATEGORIES),
                                },
                                "project_id": {"type": "string"},
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "domain": {"type": "string"},
                                "observation_type": {"type": "string"},
                                "concepts": {"type": "array", "items": {"type": "string"}},
                                "source_type": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "expires_at": {"type": "string"},
                                "visibility": {"type": "string", "enum": ["private", "shared", "standard"]},
                            },
                            "required": ["content", "category"],
                        },
                        "description": "Memories to save (≤25). Dedup verdicts are returned per item.",
                    },
                    "session_note": {
                        "type": "object",
                        "properties": {
                            "request": {"type": "string", "description": "What the user asked for this session"},
                            "investigated": {"type": "string", "description": "What was explored/read/probed along the way"},
                            "learned": {"type": "string", "description": "What was learned/discovered"},
                            "completed": {"type": "string", "description": "What got done"},
                            "next_steps": {"type": "string", "description": "What should happen next"},
                        },
                        "description": "Optional structured end-of-session note (stored as one task_context memory)",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Default project for the session note (items carry their own project_id)",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Caller user ID (optional under token auth)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="queue_status",
            description=(
                "Aggregate view of the caller's recent background tasks across the memory "
                "queues — ONE call for \"is my work done?\" instead of polling per task id. "
                "Returns counts of the caller's recently-enqueued tasks by live status "
                "(queued/processing/completed/failed + expired-out-of-Redis), instance-wide "
                "pending depth per queue (main/graph/ingest), and a caught_up boolean that is "
                "true when nothing of the caller's is queued or in flight. Use it after a "
                "checkpoint or bulk ingest before querying what was stored."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Caller user ID (optional under token auth)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_project_knowledge_config",
            description=(
                "Get a project's knowledge routing config (Phase D): which code systems "
                "are indexed, whether code fusion is enabled for generic recall, and the "
                "default engine. Use this to check project settings before querying."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "Project ID to get config for",
                    },
                },
                "required": ["project_id"],
            },
        ),
        Tool(
            name="set_project_knowledge_config",
            description=(
                "Set/update a project's knowledge routing config (Phase D): which code "
                "systems are available, whether to fuse code into generic recall (default "
                "TRUE per decision #3), and which engine to prefer. Typically set at index "
                "time but can be edited later."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "Project ID to configure",
                    },
                    "code_systems": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of code system names (e.g. ['code-cbm'])",
                    },
                    "fuse_code_into_recall": {
                        "type": "boolean",
                        "description": "Enable code fusion for generic recall (default TRUE)",
                    },
                    "default_engine": {
                        "type": "string",
                        "description": "Default code engine name (e.g. 'code-cbm')",
                    },
                },
                "required": ["project_id"],
            },
        ),
    ] + _code_graph_tools()


# Shared schema fragments for the code-graph delegation tools. These tools are
# NS's own surface over a Graphify code graph (roadmap F2 hard rule: agents
# never talk to Graphify's MCP server directly) and register only when the
# optional graphifyy library is installed.
_CODE_GRAPH_COMMON_PROPS = {
    "graph_id": {
        "type": "string",
        "description": (
            "graph_id of an ingested code-graph bundle (the artifact id stamped "
            "into code-graph memories' source_ref; owner-scoped). Omit to use "
            "the server's configured default graph."
        ),
    },
    "user_id": {
        "type": "string",
        "description": "Caller user ID (optional under token auth); scopes graph_id resolution.",
    },
    "knowledge_system": {
        "type": "string",
        "description": (
            "Optional explicit code system override (additive). Registered code systems: "
            "'code-cbm', 'code-graphify-lib', 'code-native'. When given (together with "
            "graph_id as the code_space), the tool dispatches to that engine via the "
            "knowledge-system seam; omit to use the legacy graph_id ref-shape path "
            "(native/graphify-json). Mirrors the REST /v1/code-graph tools."
        ),
    },
}


def _code_graph_tools() -> list[Tool]:
    """The three code-graph delegation tools, when the code-graph extra is installed."""
    from adapters.code_graph import code_graph_available

    if not code_graph_available():
        return []
    return [
        Tool(
            name="query_code_graph",
            description=(
                "Search a project's code knowledge graph (built by Graphify, served "
                "by Neuralscape) with BFS/DFS traversal from the best-matching nodes. "
                "Use this — not recalled memories — for CODE STRUCTURE questions "
                "(what calls X, what's in module Y): the graph reflects the code as "
                "analyzed, while structural memories would rot. Memories whose "
                "source_ref retrieval handle names this tool re-fetch live structure "
                "through it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Natural-language question or keyword search over the code graph"},
                    "mode": {"type": "string", "enum": ["bfs", "dfs"], "default": "bfs", "description": "bfs=broad context, dfs=trace a specific path"},
                    "depth": {"type": "integer", "default": 3, "description": "Traversal depth (1-6)"},
                    "token_budget": {"type": "integer", "default": 2000, "description": "Max output tokens"},
                    **_CODE_GRAPH_COMMON_PROPS,
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="get_code_neighbors",
            description=(
                "List the direct incoming/outgoing neighbors of one code-graph node "
                "(function/class/file/module) with each edge's relation and "
                "confidence tag (EXTRACTED/INFERRED/AMBIGUOUS). The precise tool for "
                "'what directly touches X?'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Node label or id to look up (e.g. 'MemoryEngine')"},
                    "relation_filter": {"type": "string", "description": "Only edges whose relation contains this substring (e.g. 'call')"},
                    **_CODE_GRAPH_COMMON_PROPS,
                },
                "required": ["label"],
            },
        ),
        Tool(
            name="code_path",
            description=(
                "Shortest connection path between two code-graph symbols — how does "
                "A reach B? Each hop shows the relation and its confidence tag. "
                "Useful for tracing an unexpected coupling flagged in a 'boundary' "
                "memory back through the actual code structure."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Label of the starting symbol"},
                    "target": {"type": "string", "description": "Label of the target symbol"},
                    "max_hops": {"type": "integer", "default": 8, "description": "Give up beyond this many hops (1-32)"},
                    **_CODE_GRAPH_COMMON_PROPS,
                },
                "required": ["source", "target"],
            },
        ),
        Tool(
            name="locate",
            description=(
                "Hybrid code retrieval: find symbols (functions, classes, methods) by "
                "natural-language description or name pattern. Uses dense embeddings + "
                "BM25 lexical search + graph degree signal for ranking. Returns file:line "
                "locations with signatures and docstrings. Use this to find relevant code "
                "before diving into structure queries (query_code_graph, get_code_neighbors, "
                "code_path). E3: requires NativeEngine (repo:<name> refs); raises "
                "EngineCapabilityError on GraphifyJsonEngine (.json artifacts)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language description or symbol name pattern (e.g., 'authentication logic', 'UserManager')"},
                    "k": {"type": "integer", "default": 10, "description": "Max hits to return (1-50)"},
                    **_CODE_GRAPH_COMMON_PROPS,
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="code_impact",
            description=(
                "Compute blast radius from a given symbol: BFS over CALLS/IMPORTS edges "
                "to find all symbols affected by changes to the given symbol. Returns a "
                "text summary (file:line format) of affected symbols. Use this to understand "
                "the impact of modifying or deleting a symbol. E7: requires NativeEngine "
                "(repo:<name> refs); raises EngineCapabilityError on GraphifyJsonEngine "
                "(.json artifacts)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "FQN or partial match of the epicenter symbol (e.g., 'UserManager.login', 'fetch_data')"},
                    "max_hops": {"type": "integer", "default": 4, "description": "Maximum BFS depth (1-16)"},
                    **_CODE_GRAPH_COMMON_PROPS,
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="code_graph_index",
            description=(
                "Phase G: index a repository INTO a code knowledge system through "
                "Neuralscape (async). Enqueues on the ingest queue and returns a "
                "task_id to poll via the memory status endpoint. Once complete, the "
                "corpus is queryable via the code tools with knowledge_system set. "
                "Records repo_sha/indexed_at/engine_version per code_space, sets the "
                "project's routing config, and runs the external-engine liveness diff."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_source": {"type": "string", "description": "Absolute path (or ref) of the repo to index"},
                    "system": {"type": "string", "description": "Target code system: code-cbm | code-graphify-lib | code-native"},
                    "project_id": {"type": "string", "description": "Project scope; sets routing config so later recalls route to this engine"},
                    "code_space": {"type": "string", "description": "Explicit code_space (code--owner--repo); default derived from user_id + repo basename"},
                    "user_id": {"type": "string", "description": "Caller user ID (optional under token auth)"},
                },
                "required": ["repo_source", "system"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    # Identity precedence:
    #   1. The OAuth/per-user token the request authenticated with (authoritative
    #      — set by BearerAuthMiddleware for this request). Over an authenticated
    #      HTTP connector the model needn't pass user_id at all, and a mismatched
    #      arguments["user_id"] is ignored rather than trusted.
    #   2. An explicit user_id argument (stdio / local Claude Code, legacy).
    #   3. The configured default_user_id.
    from auth import current_user_id

    # Tools like list_memories / delete_memories declare no required fields, so a
    # client may omit `arguments` entirely (None). Normalize before any access.
    arguments = arguments or {}
    user_id = current_user_id.get() or arguments.get("user_id") or settings.default_user_id

    try:
        if name == "recall_memories":
            # Phase E: fusion — when the router says code leg is warranted, execute
            # it concurrently with base legs and compose sections.
            from knowledge.router import resolve_systems

            route_decision = resolve_systems(
                query=arguments["query"],
                project_id=arguments.get("project_id"),
                knowledge_system=arguments.get("knowledge_system"),
                is_code_tool=False,  # recall_memories is generic recall, not a code tool
            )
            logger.debug(
                "recall_memories route decision: %s (layer %d, fusion=%s, systems=%s)",
                route_decision.rationale,
                route_decision.layer,
                route_decision.wants_code_fusion,
                route_decision.code_system_names,
            )

            def _base_search():
                # Run the synchronous, graph-backed search in a worker thread so a
                # slow read can't freeze the MCP server's event loop (which would
                # stall concurrent fire-and-forget writes and time them out).
                return _service.search(
                    query=arguments["query"],
                    user_id=user_id,
                    project_id=arguments.get("project_id"),
                    categories=arguments.get("categories"),
                    limit=arguments.get("limit", 10),
                    visibility=arguments.get("visibility"),
                    include_shared=arguments.get("include_shared", True),
                )

            # Phase E/F: resolve ALL healthy code system(s) the ROUTER decided
            # (structured signal — never infer fusion from rationale text). Respect
            # project config (default_engine/code_systems); do NOT re-pick a backend
            # here. Phase F: when >1 code system answers the same op, cross-engine
            # dedup (canonical-FQN + per-op preference) merges them into one answer.
            #
            # HARDENING FIX (M1): skip code systems that are capability placeholders
            # (no real indexed graph for this project). Never compose a [structure]
            # section from "No graph loaded" or a placeholder — that would flip
            # recall_memories output from JSON to fused text for every project_id +
            # coding query, regressing the production API.
            # Phase G: resolve REAL per-code_space engines (not the registry
            # capability placeholders). The project's routing config carries the
            # code_space the code system indexed; bind an engine to it via the
            # code_dispatch seam. Without a code_space we cannot serve a real code
            # answer — skip fusion so recall output stays byte-identical (the M1
            # hardening invariant: never compose from a placeholder / empty engine).
            code_systems_list = []
            if route_decision.wants_code_fusion and arguments.get("project_id"):
                from knowledge.code_dispatch import resolve_bound_code_system
                from knowledge.registry import get_system
                from knowledge.router import get_project_config

                _proj_cfg = get_project_config(arguments.get("project_id"))
                _cfg_code_space = getattr(_proj_cfg, "code_space", None) if _proj_cfg else None
                for _name in route_decision.code_system_names:
                    # Prefer the registered system's own code_space when it's a
                    # real (pre-bound) entry; otherwise use the project config's
                    # code_space to bind a per-space engine to the placeholder.
                    _reg = get_system(_name)
                    _reg_eng = getattr(_reg, "_engine", None) if _reg else None
                    _reg_cs = getattr(_reg_eng, "code_space", None) if _reg_eng else None
                    _bind_cs = (
                        _reg_cs if _reg_cs and _reg_cs != "__registry_capability__"
                        else (_cfg_code_space or "")
                    )
                    # Bind OFF the loop (may lazy-build graphify / probe CBM bridge).
                    _cand = await asyncio.to_thread(
                        resolve_bound_code_system, _name, _bind_cs, user_id, settings
                    )
                    if _cand is None:
                        logger.debug("Skipping %s: could not bind engine (code_space=%r)",
                                     _name, _bind_cs)
                        continue
                    if _cand.health().status != "ok":
                        logger.debug("Skipping %s: unhealthy", _name)
                        continue
                    # Guard (M1 invariant): an engine with no loaded graph can't serve —
                    # never compose a [structure] section from an empty/placeholder engine.
                    _eng = getattr(_cand, "_engine", None)
                    if _eng is not None and hasattr(_eng, "G") and _eng.G is None:
                        logger.debug("Skipping %s: no graph loaded", _name)
                        continue
                    code_systems_list.append(_cand)

            if code_systems_list:
                # Fusion path: run the code leg CONCURRENTLY with the base legs
                # (overlap the two slowest calls; same intent as search.py's
                # ThreadPoolExecutor graph-leg overlap).
                from knowledge.base import RecallRequest

                async def _one_code_recall(_sys):
                    try:
                        code_req = RecallRequest(
                            query=arguments["query"],
                            user_id=user_id,
                            project_id=arguments.get("project_id"),
                            limit=arguments.get("limit", 10),
                            operation="query",
                        )
                        return await asyncio.to_thread(_sys.recall, code_req)
                    except Exception as e:
                        logger.warning(
                            "Code leg (%s) failed (base still answers): %s",
                            _sys.info.name, e, exc_info=True,
                        )
                        return None

                async def _code_leg():
                    # Query every resolved code system concurrently.
                    answers = await asyncio.gather(
                        *[_one_code_recall(s) for s in code_systems_list]
                    )
                    answers = [a for a in answers if a is not None]
                    if not answers:
                        return None
                    if len(answers) == 1:
                        return answers[0]
                    # Phase F cross-engine dedup: >1 code system answered the same
                    # op → dedup on canonical FQN, per-op precision preference,
                    # attribute BOTH (PLAN §6). This is the REAL production path.
                    from knowledge.fusion import dedup_code_answers

                    return dedup_code_answers(answers, operation="query")

                results, code_answer = await asyncio.gather(
                    asyncio.to_thread(_base_search),
                    _code_leg(),
                )

                # Representative code system for anchor-join engine resolution
                # (to_canonical / code_space). All code engines canonicalize
                # compatibly, so the first resolved system is a safe reference.
                code_sys = code_systems_list[0]

                if code_answer is not None:
                    try:
                        from knowledge.fusion import (
                            batched_anchor_lookup,
                            compose_fusion_answer,
                            extract_fqns_from_code_answer,
                        )
                        from knowledge.base import SystemAnswer

                        fqns = extract_fqns_from_code_answer(code_answer)

                        # Batched anchor join. CRITICAL: derive repo from the code
                        # system's code_space (the SAME way _get_anchor_memories
                        # does) — NOT from project_id, which may differ from the
                        # repo name and would silently miss every anchor.
                        anchor_memories = {}
                        engine = getattr(code_sys, "_engine", None)
                        if fqns and engine is not None and hasattr(engine, "to_canonical"):
                            code_space = getattr(engine, "code_space", "") or ""
                            _parts = code_space.split("--")
                            repo = _parts[-1] if len(_parts) >= 3 else "unknown"
                            anchor_memories = await asyncio.to_thread(
                                batched_anchor_lookup,
                                fqns=fqns,
                                repo=repo,
                                to_canonical_fn=engine.to_canonical,
                                user_id=user_id,
                                limit_per_anchor=3,
                            )

                        # Base recall section — keep FULL memory text (do NOT clip;
                        # clipping destroys IDs/citations vs normal recall).
                        # MemoryResponse's text field is `.memory` (not `.content`).
                        base_content = "\n".join(
                            f"{i+1}. [{r.category}] {r.memory}"
                            for i, r in enumerate(results)
                        )
                        base_answer = SystemAnswer(
                            system_name="ns-memory",
                            content=base_content,
                            hits=[r.model_dump(exclude_none=True) for r in results],
                        )

                        # M5 FIX: Return stable JSON envelope for fused output, not type flip.
                        # Before fix: compose_fusion_answer returns plain text sections,
                        # flipping output from JSON array to prose text.
                        # After fix: structured JSON with fused=true flag and sections dict.
                        fused_sections = {}
                        if code_answer and code_answer.content:
                            fused_sections["structure"] = {
                                "system": code_answer.system_name,
                                "content": code_answer.content,
                                "hits": code_answer.hits or [],
                            }
                        if anchor_memories:
                            fused_sections["semantics"] = anchor_memories
                        if results:
                            fused_sections["memories"] = [
                                r.model_dump(exclude_none=True) for r in results
                            ]

                        fused_output = {
                            "fused": True,
                            "sections": fused_sections,
                            "query": arguments["query"],
                            "project_id": arguments.get("project_id"),
                        }

                        logger.info(
                            "Phase E: fused answer (%d code FQNs, %d anchored, %d base results)",
                            len(fqns), len(anchor_memories), len(results),
                        )
                        return [TextContent(
                            type="text",
                            text=json.dumps(fused_output, default=str, ensure_ascii=False)
                        )]
                    except Exception as e:
                        logger.warning("Fusion compose failed (fallback to base-only): %s", e, exc_info=True)
                        # Fall through to base-only output with `results` already set.
            else:
                # No code fusion — plain base recall (byte-identical to today).
                results = await asyncio.to_thread(_base_search)

            # Base-only output (Phase D behavior, or fusion fallback)
            if arguments.get("index_only"):
                from index_format import index_row

                rows = [index_row(r) for r in results]
                body = {
                    "index_only": True, "results": rows,
                    "hint": "Filter these rows, then call get_memories(ids=[...]) for full payloads.",
                }
                # E2: honest ledger entry for this recall — measured and
                # recorded off the hot path (audit 27 #11).
                _meter_mcp_index_bg("search_index", user_id, results, body)
                return [TextContent(type="text", text=json.dumps(
                    body, default=str, ensure_ascii=False))]
            _meter_mcp_full_bg("search", user_id, results)
            output = [r.model_dump(exclude_none=True) for r in results]
            return [TextContent(type="text", text=json.dumps(output, default=str))]

        elif name == "get_memories":
            ids = arguments.get("ids")
            if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
                return [TextContent(type="text", text=json.dumps(
                    {"error": "ids must be a non-empty list of memory-id strings"}))]
            try:
                out = await asyncio.to_thread(_service.get_memories_by_ids, ids, user_id)
            except ValueError as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
            _meter_mcp_full_bg("get_memories", user_id, out["results"])
            return [TextContent(type="text", text=json.dumps(
                {
                    "results": [r.model_dump(exclude_none=True) for r in out["results"]],
                    "missing": out["missing"],
                },
                default=str))]

        elif name == "timeline":
            from index_format import index_row

            anchor = arguments.get("anchor")
            if not anchor or not isinstance(anchor, str):
                return [TextContent(type="text", text=json.dumps(
                    {"error": "anchor must be a memory id (UUID) or a search query"}))]
            try:
                depth = int(arguments.get("depth", 10))
            except (TypeError, ValueError):
                depth = 10
            try:
                out = await asyncio.to_thread(
                    _service.timeline,
                    anchor=anchor,
                    user_id=user_id,
                    depth=depth,
                    project_id=arguments.get("project_id"),
                )
            except ValueError as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
            if out is None:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Timeline anchor {anchor!r} could not be resolved "
                              "(unknown id, unreadable id, or no search hit)"}))]
            anchor_id = out["anchor_id"]
            rows = [index_row(mem, anchor=(mem.id == anchor_id)) for mem in out["memories"]]
            body = {
                "anchor_id": anchor_id, "results": rows,
                "hint": "Rows are oldest→newest. Call get_memories(ids=[...]) for full payloads.",
            }
            _meter_mcp_index_bg("timeline", user_id, out["memories"], body)
            return [TextContent(type="text", text=json.dumps(
                body, default=str, ensure_ascii=False))]

        elif name == "remember":
            # Determine scope from category
            from schemas import default_scope_for_category, GLOBAL_CATEGORIES
            category = arguments["category"]
            project_id = arguments.get("project_id")
            wait = arguments.get("wait", False)

            scope = default_scope_for_category(category).value
            if project_id and category not in GLOBAL_CATEGORIES:
                scope = "project"
            # Standards are always org-wide/global — the server (store_raw)
            # enforces scope="global" for visibility="standard" regardless of
            # category, so a project-category standard (e.g. a global convention)
            # doesn't get a spurious scope="project".

            # Server-side validation of the A1 provenance fields: MCP tool
            # JSON-schema constraints are client hints, not enforced here, so
            # mirror RawMemoryRequest's limits before enqueue (a bad value
            # would otherwise only fail later as a silent background job).
            from schemas import EPISTEMIC_LEVEL_VOCAB
            _derived_from = arguments.get("derived_from")
            if _derived_from is not None and (
                not isinstance(_derived_from, list)
                or len(_derived_from) > 10
                or not all(isinstance(i, str) for i in _derived_from)
            ):
                return [TextContent(type="text", text=json.dumps(
                    {"error": "derived_from must be a list of at most 10 memory-id strings"}))]
            _epistemic_level = arguments.get("epistemic_level")
            if _epistemic_level is not None and _epistemic_level not in EPISTEMIC_LEVEL_VOCAB:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Invalid epistemic_level {_epistemic_level!r}. "
                              f"Must be one of: {sorted(EPISTEMIC_LEVEL_VOCAB)}"}))]
            # Event time: validate + normalize at the tool boundary (same
            # rationale as above — a bad value must fail HERE, actionably,
            # not later as a silent background-job failure).
            _occurred_at = arguments.get("occurred_at")
            if _occurred_at is not None:
                from schemas import validate_occurred_at
                try:
                    _occurred_at = validate_occurred_at(_occurred_at)
                except ValueError as ve:
                    return [TextContent(type="text", text=json.dumps({"error": str(ve)}))]

            # Memory-model v2 + multi-user fields (all optional)
            v2_fields = {
                "domain": arguments.get("domain"),
                "observation_type": arguments.get("observation_type"),
                "concepts": arguments.get("concepts"),
                "source_type": arguments.get("source_type"),
                "related_memory_ids": arguments.get("related_memory_ids"),
                "confidence": arguments.get("confidence"),
                "expires_at": arguments.get("expires_at"),
                "occurred_at": _occurred_at,
                "derived_from": arguments.get("derived_from"),
                "epistemic_level": arguments.get("epistemic_level"),
                "visibility": arguments.get("visibility"),
            }

            # Authoritative-tier write-gate: only dictators may write standards.
            # Return an actionable error to the model *before* enqueue so a
            # rejected write isn't lost as a silent background failure.
            from schemas import MemoryVisibility, normalize_visibility
            if normalize_visibility(arguments.get("visibility")) == MemoryVisibility.STANDARD.value:
                if not settings.standards_enabled:
                    return [TextContent(type="text", text=json.dumps(
                        {"error": "The 'standard' visibility tier is disabled (STANDARDS_ENABLED=false)."}))]
                if not settings.is_dictator(user_id):
                    return [TextContent(type="text", text=json.dumps(
                        {"error": f"User {user_id!r} is not authorized to write 'standard'-tier memories."}))]

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
                memories = await asyncio.to_thread(
                    _service.store_raw,
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
                memories = await asyncio.to_thread(
                    _service.extract_and_store,
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

        elif name == "ingest_document":
            wait = arguments.get("wait", False)
            # Gate standard-tier ingests before enqueue (else the job fails later).
            _err = _standard_write_error(arguments.get("visibility"), user_id)
            if _err:
                return _err
            # Reject unknown adapters loudly — a silent degrade-to-default would
            # ingest without the taxonomy/ontology the caller asked for.
            try:
                from schemas import validate_adapter_name

                validate_adapter_name(arguments.get("adapter", "default"))
            except ValueError as ve:
                return [TextContent(type="text", text=json.dumps({"status": "error", "error": str(ve)}))]
            scope = arguments.get("scope")
            project_id = arguments.get("project_id")
            if not scope:
                scope = "project" if project_id else "global"
            doc = {
                "content": arguments["content"],
                "source": arguments["source"],
                "user_id": user_id,
                "category": arguments.get("category", "domain_knowledge"),
                "scope": scope,
                "project_id": project_id,
                "visibility": arguments.get("visibility"),
                "tags": arguments.get("tags"),
                "extract_facts": arguments.get("extract_facts", True),
                "index_passages": arguments.get("index_passages", True),
                "adapter": arguments.get("adapter", "default"),
            }
            doc = {k: v for k, v in doc.items() if v is not None}

            try:
                task_id = await _task_manager.enqueue_ingest_document(doc)
            except (ConnectionError, OSError) as e:
                logger.warning(f"Redis unavailable, falling back to sync ingest: {e}")
                from ingest.pipeline import IngestDoc, ingest_document
                # Offload the blocking ingest so it doesn't stall the async MCP loop.
                result = await asyncio.to_thread(ingest_document, _service, IngestDoc(**doc))
                _skipped = len(result.pop("graph_jobs", []) or [])
                if _skipped:
                    logger.warning(f"Sync-fallback ingest: {_skipped} fact graph enrichment(s) skipped (Redis down)")
                    result["graph_jobs_skipped"] = _skipped
                return [TextContent(type="text", text=json.dumps({"status": "completed", "result": result, "fallback": "sync"}, default=str))]

            if wait:
                result = await _task_manager.wait_for_result(task_id)
                return [TextContent(type="text", text=json.dumps({"status": result["status"], "task_id": task_id, "result": result.get("result")}, default=str))]

            return [TextContent(type="text", text=json.dumps({"status": "accepted", "task_id": task_id}, default=str))]

        elif name == "ingest_text":
            from pydantic import ValidationError
            from schemas import IngestTextRequest

            wait = arguments.get("wait", False)
            # Infer scope (project when a project_id is present) before validating.
            raw = dict(arguments)
            if not raw.get("scope"):
                raw["scope"] = "project" if raw.get("project_id") else "global"
            # Validate with the shared schema so MCP callers can't bypass the REST
            # limits/enums (invalid scope/category/visibility, oversized content).
            try:
                req = IngestTextRequest(
                    **{k: v for k, v in raw.items() if k in IngestTextRequest.model_fields}
                )
            except ValidationError as e:
                return [TextContent(type="text", text=json.dumps({"status": "error", "error": str(e)}, default=str))]
            if req.scope == "project" and not req.project_id:
                return [TextContent(type="text", text=json.dumps({"status": "error", "error": "project_id is required when scope='project'"}, default=str))]
            # Gate standard-tier ingests before enqueue (else the job fails later).
            _err = _standard_write_error(req.visibility, user_id)
            if _err:
                return _err

            content = req.content
            category = req.category
            project_id = req.project_id

            # Persist the context as an artifact so its memories reference a real,
            # re-fetchable source (falls back to a hash-only ref when storage off).
            if settings.ingest_storage_enabled:
                from ingest.storage import artifact_source_ref, store_artifact
                fname = (req.title or "context") + ".md"
                art = await asyncio.to_thread(
                    store_artifact, content.encode(), fname, user_id, project_id, category, settings,
                )
                source_ref = artifact_source_ref(art, connector_type="manual")
            else:
                import hashlib as _hashlib
                from datetime import datetime as _dt, timezone as _tz
                digest = _hashlib.sha256(content.encode()).hexdigest()[:16]
                source_ref = {
                    "connector_id": "manual",
                    "connector_type": "manual",
                    "external_id": digest,
                    "parent_id": digest,
                    "title": req.title,
                    "last_synced_at": _dt.now(_tz.utc).isoformat(),
                }
            doc = {
                "content": content,
                "source": source_ref,
                "user_id": user_id,
                "category": category,
                "scope": req.scope,
                "project_id": project_id,
                "visibility": req.visibility.value if req.visibility else None,
                "tags": req.tags,
                "extract_facts": req.extract_facts,
                "index_passages": req.index_passages,
                "adapter": req.adapter,
            }
            # source is a dict with None title possibly — keep it; drop top-level Nones only.
            doc = {k: v for k, v in doc.items() if v is not None}

            try:
                task_id = await _task_manager.enqueue_ingest_document(doc)
            except (ConnectionError, OSError) as e:
                logger.warning(f"Redis unavailable, falling back to sync ingest: {e}")
                from ingest.pipeline import IngestDoc, ingest_document
                # Offload the blocking ingest so it doesn't stall the async MCP loop.
                result = await asyncio.to_thread(ingest_document, _service, IngestDoc(**doc))
                _skipped = len(result.pop("graph_jobs", []) or [])
                if _skipped:
                    logger.warning(f"Sync-fallback ingest: {_skipped} fact graph enrichment(s) skipped (Redis down)")
                    result["graph_jobs_skipped"] = _skipped
                return [TextContent(type="text", text=json.dumps({"status": "completed", "result": result, "fallback": "sync"}, default=str))]

            if wait:
                result = await _task_manager.wait_for_result(task_id)
                return [TextContent(type="text", text=json.dumps({"status": result["status"], "task_id": task_id, "result": result.get("result")}, default=str))]

            return [TextContent(type="text", text=json.dumps({"status": "accepted", "task_id": task_id}, default=str))]

        elif name == "get_project_context":
            # Default to a small bounded page so a large project doesn't overflow
            # the agent tool-result token limit (a 218-memory project at limit=100
            # was ~112K chars); callers can raise limit / page with offset.
            context = await asyncio.to_thread(
                _service.get_project_context,
                user_id=user_id,
                project_id=arguments["project_id"],
                limit=arguments.get("limit", 25),
                offset=arguments.get("offset", 0),
            )
            # Convert to serializable format
            output = {
                "user_id": context.user_id,
                "project_id": context.project_id,
                "total": context.total,
                "returned": context.returned,
                "offset": context.offset,
                "limit": context.limit,
                "has_more": context.has_more,
                "categories": {
                    cat: [m.model_dump(exclude_none=True) for m in memories]
                    for cat, memories in context.categories.items()
                },
            }
            return [TextContent(type="text", text=json.dumps(output, default=str))]

        elif name == "search_knowledge_graph":
            results = await asyncio.to_thread(
                _service.search_graph,
                query=arguments["query"],
                user_id=user_id,
                project_id=arguments.get("project_id"),
                limit=arguments.get("limit", 10),
            )
            return [TextContent(type="text", text=json.dumps(results, default=str))]

        elif name == "list_memories":
            results = await asyncio.to_thread(
                _service.list_memories,
                user_id=user_id,
                scope=arguments.get("scope"),
                category=arguments.get("category"),
                project_id=arguments.get("project_id"),
                limit=arguments.get("limit", 100),
                include_tombstoned=bool(arguments.get("include_tombstoned", False)),
            )
            output = [r.model_dump(exclude_none=True) for r in results]
            return [TextContent(type="text", text=json.dumps(output, default=str))]

        elif name == "list_projects":
            projects = await asyncio.to_thread(_service.list_projects, user_id=user_id)
            return [TextContent(type="text", text=json.dumps({"projects": projects}, default=str))]

        elif name == "list_processes":
            processes = await asyncio.to_thread(
                _service.list_processes, project_id=arguments.get("project_id")
            )
            return [TextContent(type="text", text=json.dumps({"processes": processes}, default=str))]

        elif name == "get_process":
            process = await asyncio.to_thread(
                _service.get_process, arguments["slug"], arguments.get("project_id")
            )
            if process is None:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Process {arguments['slug']!r} not found or processes are disabled."}))]
            return [TextContent(type="text", text=json.dumps(process, default=str))]

        elif name == "delete_memories":
            memory_id = arguments.get("memory_id")
            if memory_id:
                # Pass caller identity so standard-tier deletes are gated to dictators.
                result = await asyncio.to_thread(_service.delete_memory, memory_id, user_id)
            else:
                result = await asyncio.to_thread(
                    _service.delete_memories,
                    user_id=user_id,
                    scope=arguments.get("scope"),
                    category=arguments.get("category"),
                    project_id=arguments.get("project_id"),
                    filter_null_category=arguments.get("filter_null_category", False),
                    include_shared=arguments.get("include_shared", False),
                )
            return [TextContent(type="text", text=json.dumps(result, default=str))]

        elif name == "edit_memory":
            # Presence-keyed changes: the raw arguments dict distinguishes an
            # explicit null (clear the field) from an omitted field.
            _EDITABLE = (
                "content", "category", "project_id", "tags", "visibility",
                "domain", "observation_type", "concepts", "confidence", "expires_at",
            )
            changes = {k: arguments[k] for k in _EDITABLE if k in arguments}
            if not changes:
                return [TextContent(type="text", text=json.dumps(
                    {"error": "No fields to update — provide at least one editable field"}))]
            try:
                result = await asyncio.to_thread(
                    _service.patch_memory, arguments["memory_id"], user_id, changes
                )
            except (LookupError, PermissionError, ValueError) as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

            graph = result["graph"]
            graph_task_id = None
            if result.get("graph_job"):
                try:
                    graph_task_id = await _task_manager.enqueue_graph_enrichment(
                        **result["graph_job"]
                    )
                    graph = graph.replace("_pending", "_queued")
                except Exception as e:
                    logger.warning(f"Graph enqueue failed for edited memory: {e}")
                    graph = "enqueue_failed"
            mem = result.get("memory")
            return [TextContent(type="text", text=json.dumps({
                "status": "ok",
                "memory": mem.model_dump(exclude_none=True) if mem is not None else None,
                "graph": graph,
                "graph_task_id": graph_task_id,
            }, default=str))]

        elif name == "get_reasoning_chain":
            try:
                max_depth = int(arguments.get("max_depth", 3))
            except (TypeError, ValueError):
                max_depth = 3
            max_depth = min(max(max_depth, 1), 10)
            chain = await asyncio.to_thread(
                _service.get_reasoning_chain, arguments["memory_id"], max_depth
            )
            if chain is None:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Memory {arguments['memory_id']!r} not found"}))]
            return [TextContent(type="text", text=json.dumps(
                {"status": "ok", "chain": chain}, default=str))]

        elif name == "retag_memories":
            # Truthiness, not `is not None`: an empty string / empty list would
            # produce no Qdrant condition in the service and turn a "filtered"
            # retag into an unfiltered sweep.
            filters = {
                k: arguments[k]
                for k in ("scope", "category", "project_id", "visibility", "tags_contains")
                if arguments.get(k)
            }
            ops = {
                k: arguments[k]
                for k in ("add_tags", "remove_tags", "set_category")
                if arguments.get(k)
            }
            if "set_project_id" in arguments:
                ops["set_project_id"] = arguments["set_project_id"]
            if not filters:
                return [TextContent(type="text", text=json.dumps(
                    {"error": "At least one filter is required — refusing an unfiltered retag sweep"}))]
            if not ops:
                return [TextContent(type="text", text=json.dumps(
                    {"error": "At least one operation is required (add_tags/remove_tags/set_category/set_project_id)"}))]
            overlap = set(ops.get("add_tags") or []) & set(ops.get("remove_tags") or [])
            if overlap:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Tags cannot be both added and removed: {sorted(overlap)}"}))]

            if arguments.get("dry_run"):
                result = await asyncio.to_thread(
                    _service.retag_memories, user_id, filters, ops, True
                )
                result.pop("graph_jobs", None)
                return [TextContent(type="text", text=json.dumps(result, default=str))]

            task_id = await _task_manager.enqueue_retag(user_id, filters, ops)
            return [TextContent(type="text", text=json.dumps(
                {"status": "accepted", "task_id": task_id}))]

        elif name == "code_graph_index":
            # Phase G: through-NS index trigger (MCP twin of POST /v1/code-graph/index).
            from adapters.code_graph import _MISSING_EXTRA_MSG, code_graph_available

            if not code_graph_available():
                return [TextContent(type="text", text=json.dumps({"error": _MISSING_EXTRA_MSG}))]
            repo_source = arguments.get("repo_source")
            system = arguments.get("system")
            if not repo_source or not system:
                return [TextContent(type="text", text=json.dumps(
                    {"error": "repo_source and system are required"}))]
            code_space = arguments.get("code_space")
            if not code_space:
                from pathlib import Path as _P

                repo_name = _P(str(repo_source).rstrip("/")).name or "repo"
                code_space = f"code--{user_id}--{repo_name}"
            task_id = await _task_manager.enqueue_code_index({
                "system": system,
                "repo_source": repo_source,
                "code_space": code_space,
                "project_id": arguments.get("project_id"),
                "user_id": user_id,
            })
            return [TextContent(type="text", text=json.dumps({
                "status": "accepted",
                "task_id": task_id,
                "poll_url": f"/v1/memories/status/{task_id}",
                "code_space": code_space,
                "system": system,
            }))]

        elif name in ("query_code_graph", "get_code_neighbors", "code_path", "locate", "code_impact"):
            # NS-surface delegations to the code-graph adapter (roadmap F2:
            # the interaction interface is ALWAYS Neuralscape). Availability-
            # gated: without the optional extra these tools aren't listed, but
            # a stale client may still call them — answer with the remedy.
            from adapters.code_graph import _MISSING_EXTRA_MSG, code_graph_available

            if not code_graph_available():
                return [TextContent(type="text", text=json.dumps({"error": _MISSING_EXTRA_MSG}))]
            from adapters.code_graph.engine import EngineCapabilityError
            from adapters.code_graph.query import (
                CodeGraphError,
                code_impact,
                code_path,
                get_code_neighbors,
                locate_symbols,
                query_code_graph,
            )

            # Phase D: wire the router for code tools (always route to a code system).
            # The 5 code tools are ALREADY explicit code-routing (layer 1); the only
            # choice is WHICH backend. Log the decision but keep calling the existing
            # query.py functions (backward compat; Phase E will wire to system.recall()).
            from knowledge.router import resolve_systems

            # Map tool name to operation hint for router
            operation_map = {
                "query_code_graph": "query",
                "get_code_neighbors": "neighbors",
                "code_path": "path",
                "locate": "locate",
                "code_impact": "impact",
            }
            route_decision = resolve_systems(
                query=arguments.get("question") or arguments.get("query") or arguments.get("label") or "",
                project_id=None,  # Code tools don't expose project_id yet (Phase E)
                knowledge_system=arguments.get("knowledge_system"),
                graph_id=arguments.get("graph_id"),
                operation=operation_map.get(name),
                is_code_tool=True,
            )
            logger.debug(
                "Code tool %s route decision: %s (layer %d)",
                name,
                route_decision.rationale,
                route_decision.layer,
            )

            # Phase G: when the caller gives an explicit knowledge_system, dispatch
            # through the bound engine's system.recall() (mirrors the REST twins).
            # graph_id carries the code_space. Omit → legacy query.py path below.
            explicit_system = arguments.get("knowledge_system")
            if explicit_system:
                code_space = arguments.get("graph_id")
                operation = operation_map.get(name)
                if not code_space:
                    return [TextContent(type="text", text=json.dumps(
                        {"error": "knowledge_system requires graph_id (the code_space)"}))]
                resolved = route_decision.systems[0] if route_decision.systems else None
                if resolved is None or resolved.info.name != explicit_system:
                    return [TextContent(type="text", text=json.dumps({"error": (
                        f"knowledge_system '{explicit_system}' unavailable or does not "
                        f"support '{operation}' ({route_decision.rationale})")}))]
                from knowledge.base import RecallRequest
                from knowledge.code_dispatch import resolve_bound_code_system

                bound = await asyncio.to_thread(
                    resolve_bound_code_system, explicit_system, code_space, user_id, settings
                )
                if bound is None:
                    return [TextContent(type="text", text=json.dumps({"error": (
                        f"could not bind engine for '{explicit_system}' at code_space "
                        f"'{code_space}'")}))]
                creq = RecallRequest(
                    query=(arguments.get("question") or arguments.get("query")
                           or arguments.get("label") or arguments.get("source")
                           or arguments.get("symbol") or "code"),
                    user_id=user_id,
                    operation=operation,
                    label=arguments.get("label") or arguments.get("symbol"),
                    source=arguments.get("source"),
                    target=arguments.get("target"),
                    mode=arguments.get("mode", "bfs"),
                    depth=arguments.get("depth", 3),
                    limit=arguments.get("k", 10),
                    max_hops=arguments.get("max_hops"),
                    relation_filter=arguments.get("relation_filter"),
                    token_budget=arguments.get("token_budget"),
                )
                try:
                    answer = await asyncio.to_thread(bound.recall, creq)
                except EngineCapabilityError as e:
                    return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
                except Exception as e:  # noqa: BLE001
                    return [TextContent(type="text", text=json.dumps({"error": f"{name} failed: {e}"}))]
                if name == "locate":
                    return [TextContent(type="text", text=json.dumps(
                        answer.hits or [], default=str, ensure_ascii=False))]
                return [TextContent(type="text", text=answer.content)]

            try:
                if name == "query_code_graph":
                    text = await asyncio.to_thread(
                        query_code_graph,
                        arguments["question"],
                        user_id=user_id,
                        settings=settings,
                        graph_id=arguments.get("graph_id"),
                        mode=arguments.get("mode", "bfs"),
                        depth=arguments.get("depth", 3),
                        token_budget=arguments.get("token_budget", 2000),
                    )
                elif name == "get_code_neighbors":
                    text = await asyncio.to_thread(
                        get_code_neighbors,
                        arguments["label"],
                        user_id=user_id,
                        settings=settings,
                        graph_id=arguments.get("graph_id"),
                        relation_filter=arguments.get("relation_filter", ""),
                    )
                elif name == "code_path":
                    text = await asyncio.to_thread(
                        code_path,
                        arguments["source"],
                        arguments["target"],
                        user_id=user_id,
                        settings=settings,
                        graph_id=arguments.get("graph_id"),
                        max_hops=arguments.get("max_hops", 8),
                    )
                elif name == "locate":
                    hits = await asyncio.to_thread(
                        locate_symbols,
                        arguments["query"],
                        user_id=user_id,
                        settings=settings,
                        graph_id=arguments.get("graph_id"),
                        k=arguments.get("k", 10),
                    )
                    # Format locate results as JSON (list of LocateHit dicts)
                    from dataclasses import asdict
                    output = [asdict(hit) for hit in hits]
                    return [TextContent(type="text", text=json.dumps(output, default=str, ensure_ascii=False))]
                else:  # code_impact
                    text = await asyncio.to_thread(
                        code_impact,
                        arguments["symbol"],
                        user_id=user_id,
                        settings=settings,
                        graph_id=arguments.get("graph_id"),
                        max_hops=arguments.get("max_hops", 4),
                    )
            except EngineCapabilityError as e:
                # e.g. code_impact/locate on a graph.json engine — clean error JSON,
                # never a raw crash.
                return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
            except CodeGraphError as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps({"error": f"locate failed: {e}"}))]
            return [TextContent(type="text", text=text)]

        elif name == "get_project_knowledge_config":
            from knowledge.router import get_project_config

            project_id = arguments.get("project_id")
            if not project_id:
                return [TextContent(type="text", text=json.dumps({"error": "project_id required"}))]

            config = get_project_config(project_id)
            if config is None:
                # Return default config
                result = {
                    "project_id": project_id,
                    "code_systems": [],
                    "fuse_code_into_recall": True,
                    "default_engine": None,
                }
            else:
                result = {
                    "project_id": config.project_id,
                    "code_systems": config.code_systems,
                    "fuse_code_into_recall": config.fuse_code_into_recall,
                    "default_engine": config.default_engine,
                }
            return [TextContent(type="text", text=json.dumps(result, default=str))]

        elif name == "set_project_knowledge_config":
            from knowledge.router import ProjectKnowledgeConfig, set_project_config

            project_id = arguments.get("project_id")
            if not project_id:
                return [TextContent(type="text", text=json.dumps({"error": "project_id required"}))]

            config = ProjectKnowledgeConfig(
                project_id=project_id,
                code_systems=arguments.get("code_systems") or [],
                fuse_code_into_recall=arguments.get("fuse_code_into_recall", True),
                default_engine=arguments.get("default_engine"),
            )
            set_project_config(config)
            result = {
                "project_id": config.project_id,
                "code_systems": config.code_systems,
                "fuse_code_into_recall": config.fuse_code_into_recall,
                "default_engine": config.default_engine,
            }
            return [TextContent(type="text", text=json.dumps(result, default=str))]

        elif name == "schedule_dream":
            # Mirrors POST /v1/extensions/dreaming/run: never sweep in-process
            # (a sweep is minutes of LLM + store work) — enqueue onto the graph
            # worker's queue and let the caller poll the status endpoint.
            from extensions.dreaming.config import dreaming_settings

            force = bool(arguments.get("force", False))
            if not dreaming_settings.enabled and not force:
                return [TextContent(type="text", text=json.dumps({
                    "error": "DREAMING_ENABLED=false — set the env var (or force=true) to run"
                }))]
            # Reuse the process-lifetime pool (audit 27 #35) — no per-call
            # connect/teardown on this path.
            arq_pool = await _get_arq_pool()
            job = await arq_pool.enqueue_job(
                "run_dream_sweep",
                arguments.get("pool"),
                bool(arguments.get("dry_run", False)),
                force,
                _queue_name=settings.graph_queue_name,
            )
            return [TextContent(type="text", text=json.dumps({
                "status": "enqueued",
                "job_id": job.job_id if job else None,
                "poll": "/v1/extensions/dreaming/status",
            }))]

        elif name == "get_card":
            from functools import partial

            from extensions.dreaming.card import build_card_view, resolve_card_pool
            from extensions.dreaming.sweep import _get_redis

            pool = resolve_card_pool(
                user_id=user_id,
                project_id=arguments.get("project_id"),
                pool=arguments.get("pool"),
            )
            if not pool:
                return [TextContent(type="text", text=json.dumps(
                    {"error": "Cannot resolve a pool — pass project_id or pool"}))]
            # Shared read contract with the REST route (build_card_view):
            # private cards gate on the caller's EFFECTIVE identity
            # (`user_id` above) — a verified token identity wins and
            # cannot be overridden by arguments; legacy shared-key /
            # stdio callers are scoped to the user they claimed, the
            # same trust model as every other read tool on this surface.
            view = await asyncio.to_thread(
                partial(
                    build_card_view,
                    pool,
                    user_id,
                    is_dictator=settings.is_dictator(user_id),
                    redis=_get_redis(),
                )
            )
            if view["status"] == "forbidden":
                return [TextContent(type="text", text=json.dumps(
                    {"error": "Another user's private card is not readable"}))]
            if view["status"] == "not_found":
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"No identity card for pool {pool!r} yet — "
                              "the dreaming sweep builds cards"}))]
            return [TextContent(type="text", text=json.dumps(view))]

        elif name == "ask_memory":
            from ask import REASONING_TIERS, AskUnavailable, ask_memory

            question = arguments.get("question")
            if not question or not isinstance(question, str):
                return [TextContent(type="text", text=json.dumps(
                    {"error": "question must be a non-empty string"}))]
            level = arguments.get("reasoning_level", "low")
            if level not in REASONING_TIERS:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Invalid reasoning_level {level!r}. "
                              f"Must be one of: {list(REASONING_TIERS)}"}))]
            try:
                out = await ask_memory(
                    _service,
                    question=question,
                    user_id=user_id,
                    reasoning_level=level,
                    project_id=arguments.get("project_id"),
                )
            except AskUnavailable as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

            # E2: ledger the ask (baseline = retrieved evidence, served =
            # answer). `_evidence_tokens` is internal — popped, never served.
            # Values captured before dispatch; measured/recorded off the hot
            # path on the telemetry executor (audit 27 #11).
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
            return [TextContent(type="text", text=json.dumps(out, default=str, ensure_ascii=False))]

        elif name == "checkpoint":
            import checkpoint as checkpoint_mod

            from pydantic import ValidationError
            from schemas import CheckpointRequest

            # Validate with the shared schema so MCP callers get the same
            # ≤25 bound / per-item field validation as the REST route.
            try:
                req = CheckpointRequest(
                    **{k: v for k, v in arguments.items() if k in CheckpointRequest.model_fields}
                )
            except ValidationError as e:
                return [TextContent(type="text", text=json.dumps(
                    {"status": "error", "error": str(e)}, default=str))]
            try:
                prepared = await asyncio.to_thread(
                    checkpoint_mod.prepare_checkpoint, _service, req, user_id
                )
            except (ValueError, PermissionError) as e:
                return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

            common = {
                "verdicts": prepared["verdicts"],
                "duplicates": prepared["duplicates"],
                "session_note_included": prepared["session_note_included"],
                "enqueued": len(prepared["to_enqueue"]),
            }
            if not prepared["to_enqueue"]:
                return [TextContent(type="text", text=json.dumps(
                    {"status": "ok", "task_id": None, **common,
                     "hint": "Every item was already stored — nothing enqueued."},
                    default=str))]
            try:
                task_id = await _task_manager.enqueue_raw_batch(items=prepared["to_enqueue"])
            except (ConnectionError, OSError) as e:
                logger.warning(f"Redis unavailable, falling back to sync checkpoint store: {e}")
                memories = await asyncio.to_thread(
                    _service.store_raw_batch, prepared["to_enqueue"]
                )
                return [TextContent(type="text", text=json.dumps(
                    {"status": "completed", "task_id": None, **common,
                     "stored": len(memories), "fallback": "sync"},
                    default=str))]
            return [TextContent(type="text", text=json.dumps(
                {"status": "accepted", "task_id": task_id, **common}, default=str))]

        elif name == "queue_status":
            out = await _task_manager.get_queue_status(user_id)
            return [TextContent(type="text", text=json.dumps(
                {"status": "ok", **out}, default=str))]

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
