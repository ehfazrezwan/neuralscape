"""Minimal MCP client over Streamable HTTP (JSON-RPC 2.0).

Handles both ``application/json`` and ``text/event-stream`` responses and the
``mcp-session-id`` header (the stack's /mcp endpoint is stateless today, but a
session id is honoured if the server hands one out).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

PROTOCOL_VERSION = "2025-03-26"


class McpError(RuntimeError):
    def __init__(self, message: str, code: int | None = None, status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def parse_rpc_body(content_type: str, text: str, want_id: int | None) -> dict | None:
    """Extract the JSON-RPC message answering ``want_id`` from a response body."""
    if not text.strip():
        return None
    if "text/event-stream" in content_type:
        messages: list[dict] = []
        for block in text.replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(
                line[5:].lstrip() if line.startswith("data:") else ""
                for line in block.split("\n")
                if line.startswith("data:")
            )
            if data.strip():
                try:
                    messages.append(json.loads(data))
                except json.JSONDecodeError:
                    continue
    else:
        parsed = json.loads(text)
        messages = parsed if isinstance(parsed, list) else [parsed]
    for msg in messages:
        if isinstance(msg, dict) and msg.get("id") == want_id and ("result" in msg or "error" in msg):
            return msg
    return None


class MCPClient:
    def __init__(self, base_url: str, token: str | None = None, path: str = "/mcp/", timeout: float = 60.0) -> None:
        self.url = base_url.rstrip("/") + path
        self.token = token
        self.session_id: str | None = None
        self._next_id = 0
        self.http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.http.close()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
            headers["mcp-protocol-version"] = PROTOCOL_VERSION
        return headers

    def post_raw(self, payload: dict) -> httpx.Response:
        """POST one JSON-RPC message; no interpretation (use for auth probes)."""
        return self.http.post(self.url, headers=self._headers(), content=json.dumps(payload))

    def rpc(self, method: str, params: dict | None = None) -> Any:
        self._next_id += 1
        rid = self._next_id
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            payload["params"] = params
        resp = self.post_raw(payload)
        if resp.status_code >= 400:
            raise McpError(f"{method} -> HTTP {resp.status_code}: {resp.text[:200]}", status=resp.status_code)
        sid = resp.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid
        msg = parse_rpc_body(resp.headers.get("content-type", ""), resp.text, rid)
        if msg is None:
            raise McpError(f"{method}: no JSON-RPC response with id {rid} in body")
        if "error" in msg:
            err = msg["error"] or {}
            raise McpError(f"{method}: {err.get('message')}", code=err.get("code"))
        return msg["result"]

    def notify(self, method: str, params: dict | None = None) -> int:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        resp = self.post_raw(payload)
        if resp.status_code >= 400:
            raise McpError(f"{method} -> HTTP {resp.status_code}: {resp.text[:200]}", status=resp.status_code)
        return resp.status_code

    def initialize(self) -> dict:
        result = self.rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "ns-acceptance", "version": "1"},
            },
        )
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict]:
        return self.rpc("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        """Returns the raw tools/call result ({content: [...], isError})."""
        return self.rpc("tools/call", {"name": name, "arguments": arguments or {}})

    @staticmethod
    def result_text(result: dict) -> str:
        return "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
