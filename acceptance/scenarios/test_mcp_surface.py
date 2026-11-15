"""MCP Streamable HTTP at /mcp/: tool surface, identity, unauthenticated refusal."""

from __future__ import annotations

import json

from conftest import Fact
from nsaccept import eventually, expect_check_fails, negative_control, proves
from nsaccept.mcp import MCPClient

CORE_TOOLS = {
    "recall_memories", "remember", "get_memories", "timeline", "checkpoint",
    "queue_status", "list_memories", "edit_memory", "delete_memories", "ask_memory",
    "remember_conversation", "get_project_context", "ingest_text",
}


def check_tool_surface(tools: list[dict]) -> None:
    names = {t.get("name") for t in tools}
    missing = sorted(CORE_TOOLS - names)
    assert not missing, f"missing MCP tools: {missing}"
    bare = [t["name"] for t in tools if not t.get("inputSchema")]
    assert not bare, f"tools without inputSchema: {bare}"


def check_list_memories_has_marker(mcp: MCPClient, fact: Fact) -> None:
    res = mcp.call_tool("list_memories", {"project_id": fact.project_id, "limit": 100})
    assert not res.get("isError"), f"list_memories isError: {MCPClient.result_text(res)[:200]}"
    text = MCPClient.result_text(res)
    json.loads(text)  # the tool must return JSON
    assert fact.marker in text, f"marker {fact.marker} not in list_memories output"


@proves("read.mcp-tool-surface")
def test_initialize_and_tool_list(mcp_a):
    info = mcp_a.initialize()
    assert info.get("serverInfo", {}).get("name"), f"initialize result: {str(info)[:200]}"
    check_tool_surface(mcp_a.list_tools())


@proves("read.mcp-tool-surface", "auth.bearer-middleware")
def test_unauthenticated_mcp_is_401(ctx):
    anon = MCPClient(ctx.base_url, token=None)
    try:
        r = anon.post_raw({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                       "clientInfo": {"name": "ns-acceptance", "version": "1"}},
        })
    finally:
        anon.close()
    assert r.status_code == 401, f"unauthenticated /mcp/ initialize -> {r.status_code}"


@proves("auth.mcp-identity-contextvar")
def test_mcp_tool_call_returns_callers_data(mcp_a, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became visible"
    mcp_a.initialize()
    eventually(lambda: check_list_memories_has_marker(mcp_a, a_fact), timeout=60, what="MCP list_memories shows A's fact")


@proves("auth.mcp-identity-contextvar")
def test_mcp_tool_call_hides_other_users_data(mcp_b, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became visible"
    mcp_b.initialize()
    res = mcp_b.call_tool("list_memories", {"project_id": a_fact.project_id, "limit": 100})
    assert a_fact.marker not in MCPClient.result_text(res), "B's MCP list_memories leaked A's marker"


@proves("auth.mcp-identity-contextvar", "read.mcp-tool-surface")
@negative_control
def test_mcp_marker_check_fails_for_user_b(mcp_b, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became visible"
    mcp_b.initialize()
    expect_check_fails(check_list_memories_has_marker, mcp_b, a_fact)


@proves("read.mcp-tool-surface")
@negative_control
def test_tool_surface_check_fails_on_truncated_list(mcp_a):
    mcp_a.initialize()
    tools = mcp_a.list_tools()
    expect_check_fails(check_tool_surface, [t for t in tools if t["name"] != "recall_memories"])
