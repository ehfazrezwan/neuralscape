"""REST route inventory used by the read.rest-v1-surface sweep.

Every entry is (method, path template, request kwargs, expected statuses), built
so the request is harmless: invalid bodies (validation 4xx), random ids,
dry-run flags, or a request the stack must refuse (a foreign user's id).
The sweep fails when a route answers 405, FastAPI's own "Not Found", or 5xx.
"""

from __future__ import annotations

# (method, path, kwargs for the client call). "{rand}" -> a random UUID,
# "{project}" -> the run's project id, "{foreign_memory}" -> a memory id owned
# by the OTHER user (so the call is refused, never executed).
PROBES: list[tuple[str, str, dict]] = [
    ("GET", "/health/live", {}),
    ("GET", "/health", {}),
    ("POST", "/v1/memories", {"json": {"__probe": True}}),
    ("POST", "/v1/memories/raw", {"json": {"__probe": True}}),
    ("POST", "/v1/memories/raw/batch", {"json": {"__probe": True}}),
    ("POST", "/v1/ingest", {"json": {"__probe": True}}),
    ("POST", "/v1/ingest/text", {"json": {"__probe": True}}),
    ("POST", "/v1/ingest/files", {}),
    ("GET", "/v1/ingest/artifacts/{rand}", {}),
    ("GET", "/v1/ingest/exemplars/{rand}", {}),
    ("GET", "/v1/code-graph/query", {}),
    ("GET", "/v1/code-graph/neighbors", {}),
    ("GET", "/v1/code-graph/path", {}),
    ("GET", "/v1/code-graph/locate", {}),
    ("GET", "/v1/code-graph/impact", {}),
    ("POST", "/v1/code-graph/index", {"json": {"__probe": True}}),
    ("DELETE", "/v1/code-graph/graph", {}),
    ("POST", "/v1/connectors", {"json": {"__probe": True}}),
    ("GET", "/v1/connectors", {}),
    ("GET", "/v1/connectors/{rand}", {}),
    ("DELETE", "/v1/connectors/{rand}", {}),
    ("POST", "/v1/connectors/{rand}/sync", {}),
    ("GET", "/v1/memories/status/{rand}", {}),
    ("POST", "/v1/search", {"json": {"__probe": True}}),
    ("POST", "/v1/memories/batch-get", {"json": {"__probe": True}}),
    ("POST", "/v1/timeline", {"json": {"__probe": True}}),
    ("POST", "/v1/ask", {"json": {"__probe": True}}),
    ("POST", "/v1/checkpoint", {"json": {"__probe": True}}),
    ("GET", "/v1/metrics", {}),
    ("GET", "/v1/queue/status", {}),
    ("POST", "/v1/graph/search", {"json": {"__probe": True}}),
    ("GET", "/v1/context/global", {}),
    ("GET", "/v1/context/inject", {}),
    ("POST", "/v1/context/assemble", {"json": {"__probe": True}}),
    ("GET", "/v1/context/{project}", {}),
    ("GET", "/v1/projects", {}),
    ("PUT", "/v1/settings/extraction-instructions", {"json": {"__probe": True}}),
    ("GET", "/v1/settings/extraction-instructions", {}),
    ("GET", "/v1/processes", {}),
    ("GET", "/v1/processes/{rand}", {}),
    ("GET", "/v1/export/okf", {"params": {"project_id": "zq-route-probe-none"}}),
    ("GET", "/v1/memories", {"params": {"limit": 1}}),
    ("GET", "/v1/memories/{rand}", {}),
    ("GET", "/v1/memories/{rand}/reasoning_chain", {}),
    ("PATCH", "/v1/memories/{rand}", {"json": {}}),
    ("PUT", "/v1/memories/{rand}", {"json": {}}),
    ("POST", "/v1/memories/retag", {"json": {"__probe": True}}),
    ("DELETE", "/v1/memories/{foreign_memory}", {}),
    ("DELETE", "/v1/memories", {"json": {"category": "zq-route-probe-none"}}),
    ("GET", "/v1/categories", {}),
    ("GET", "/v1/graph/nodes", {"params": {"limit": 1}}),
    ("GET", "/v1/graph/edges", {"params": {"limit": 1}}),
    ("GET", "/v1/graph/episodes", {"params": {"limit": 1}}),
    ("DELETE", "/v1/graph/episodes/junk", {"params": {"dry_run": "true"}}),
    ("GET", "/v1/graph/communities", {"params": {"limit": 1}}),
    ("GET", "/v1/extensions", {}),
    ("POST", "/v1/extensions/events", {"json": {"__probe": True}}),
    ("GET", "/v1/projects/{project}/knowledge-config", {}),
    ("PUT", "/v1/projects/{project}/knowledge-config", {"json": {"__probe": True}}),
    ("POST", "/v1/admin/rescope-private-derivatives", {"json": {}}),
    ("POST", "/v1/admin/audit-private-leakage", {"json": {}}),
    ("POST", "/v1/extensions/dreaming/run", {"json": {}}),
    ("GET", "/v1/extensions/dreaming/status", {}),
    ("GET", "/v1/extensions/dreaming/card", {}),
    ("POST", "/v1/extensions/conversation-compiler/flush", {"json": {"__probe": True}}),
    ("POST", "/v1/extensions/conversation-compiler/compile", {"json": {"__probe": True}}),
    ("POST", "/v1/extensions/conversation-compiler/query", {"json": {"__probe": True}}),
    ("POST", "/v1/extensions/conversation-compiler/lint", {"json": {"structural_only": "not-a-bool"}}),
    ("GET", "/v1/extensions/conversation-compiler/status", {}),
]

# Routes the sweep deliberately does not call, and where they ARE exercised.
COVERED_ELSEWHERE: dict[tuple[str, str], str] = {
    ("GET", "/v1/stream"): "test_extensions.py (SSE reader)",
    ("DELETE", "/v1/graph/episodes/{episode_uuid}"): "test_graph.py (graph admin)",
}

# Public OAuth front door: disabled unless a public URL is configured; not in
# this PR's families. Excluded from the OpenAPI drift check by prefix.
EXCLUDED_PREFIXES = ("/oauth", "/.well-known", "/docs", "/redoc", "/openapi.json")

# MCP tools registered unconditionally (the 26 core tools) vs. the optional
# code-graph extra's tools.
MCP_CORE_TOOLS = {
    "recall_memories", "get_memories", "timeline", "remember", "remember_conversation",
    "ingest_document", "ingest_text", "get_project_context", "search_knowledge_graph",
    "list_memories", "list_projects", "list_processes", "get_process", "delete_memories",
    "edit_memory", "get_reasoning_chain", "retag_memories", "schedule_dream", "get_card",
    "ask_memory", "checkpoint", "queue_status", "get_project_knowledge_config",
    "set_project_knowledge_config", "rescope_private_derivatives", "audit_private_leakage",
}
MCP_CODE_TOOLS = {
    "query_code_graph", "get_code_neighbors", "code_path", "locate", "code_impact",
    "code_graph_index", "code_graph_delete",
}
