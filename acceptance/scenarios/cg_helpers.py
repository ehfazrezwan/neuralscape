"""Shared by the code-graph scenario modules (native and CBM): not a test module."""

from __future__ import annotations

from nsaccept.client import NSClient
from nsaccept.mcp import MCPClient

#: a function in the indexed ``knowledge/router.py`` that calls CALLEE
SYMBOL = "resolve_systems"
CALLEE = "_has_coding_signal"


def check_mentions(text: str, *names: str) -> None:
    missing = [n for n in names if n not in (text or "")]
    assert not missing, f"{missing} not in: {(text or '')[:240]!r}"


def cg_get(client: NSClient, path: str, **params) -> dict:
    r = client.get(f"/v1/code-graph/{path}", params=params)
    assert r.status_code == 200, f"GET /v1/code-graph/{path} {params} -> {r.status_code}: {r.text[:200]}"
    return r.json()


def mcp_text(mcp: MCPClient, tool: str, arguments: dict) -> str:
    res = mcp.call_tool(tool, arguments)
    text = MCPClient.result_text(res)
    assert not res.get("isError"), f"MCP {tool} isError: {text[:200]}"
    return text
