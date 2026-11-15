"""Small building blocks shared by the scenario families.

Everything here talks to the stack through its public surface only (REST via
``NSClient``, MCP via ``MCPClient``); nothing imports service code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .client import NSClient
from .mcp import MCPClient


@dataclass
class Written:
    """One async write: what was sent, how the task ended, which memory it made.

    Never raises: fixtures record, scenarios assert (a fixture error would be
    reported as ``error``, not ``fail``).
    """

    marker: str
    content: str
    task_id: str = ""
    status: dict = field(default_factory=dict)
    memory_id: str = ""
    error: str = ""

    @property
    def completed(self) -> bool:
        return self.status.get("status") == "completed"

    def require(self) -> "Written":
        """Assert the write finished and produced a memory (call from a scenario)."""
        assert not self.error, f"write for {self.marker} did not start: {self.error}"
        assert self.completed, f"write for {self.marker} ended as {self.status}"
        assert self.memory_id, f"write for {self.marker} completed without a memory id: {self.status}"
        return self


def memory_ids_in(status: dict) -> list[str]:
    """Memory ids from a completed task's ``result.memories`` list."""
    result = status.get("result") or {}
    memories = result.get("memories") if isinstance(result, dict) else None
    return [m["id"] for m in memories or [] if isinstance(m, dict) and m.get("id")]


def write_raw(
    client: NSClient,
    content: str,
    category: str,
    *,
    marker: str = "",
    timeout: float = 120.0,
    **kw: Any,
) -> Written:
    """POST /v1/memories/raw and poll the task to a terminal state."""
    w = Written(marker=marker, content=content)
    try:
        w.task_id = client.store_raw(content, category, **kw)
        w.status = client.poll_task(w.task_id, timeout=timeout)
        ids = memory_ids_in(w.status)
        w.memory_id = ids[0] if ids else ""
    except AssertionError as exc:
        w.error = str(exc)
    return w


def list_rows(client: NSClient, **params: Any) -> list[dict]:
    """GET /v1/memories (full rows); asserts 200."""
    params.setdefault("limit", 500)
    resp = client.get("/v1/memories", params=params)
    assert resp.status_code == 200, f"GET /v1/memories -> {resp.status_code}: {resp.text[:200]}"
    body = resp.json()
    assert isinstance(body, list), f"GET /v1/memories returned {type(body).__name__}, not a list"
    return body


def rows_with(rows: list[dict], marker: str) -> list[dict]:
    return [r for r in rows if marker in (r.get("memory") or "")]


def search_texts(client: NSClient, query: str, **kw: Any) -> list[dict]:
    return client.search(query, **kw)


def mcp_json(mcp: MCPClient, tool: str, arguments: dict | None = None) -> Any:
    """Call an MCP tool and return its JSON payload (asserts the call itself succeeded)."""
    res = mcp.call_tool(tool, arguments or {})
    text = MCPClient.result_text(res)
    assert not res.get("isError"), f"MCP {tool} isError: {text[:200]}"
    try:
        return json.loads(text)
    except ValueError:
        raise AssertionError(f"MCP {tool} returned non-JSON text: {text[:200]!r}") from None


def graph_episodes(client: NSClient, project_id: str | None = None, limit: int = 200) -> list[dict]:
    params: dict[str, Any] = {"limit": limit}
    if project_id:
        params["project_id"] = project_id
    resp = client.get("/v1/graph/episodes", params=params)
    assert resp.status_code == 200, f"GET /v1/graph/episodes -> {resp.status_code}: {resp.text[:200]}"
    return resp.json().get("episodes", [])


def graph_listing(client: NSClient, kind: str, project_id: str | None = None, limit: int = 200) -> list[dict]:
    """GET /v1/graph/{nodes,edges,episodes,communities} -> the list under the same key."""
    params: dict[str, Any] = {"limit": limit}
    if project_id:
        params["project_id"] = project_id
    resp = client.get(f"/v1/graph/{kind}", params=params)
    assert resp.status_code == 200, f"GET /v1/graph/{kind} -> {resp.status_code}: {resp.text[:200]}"
    body = resp.json()
    assert body.get("status") == "ok", f"graph {kind} status: {body.get('status')!r}"
    return body.get(kind, [])


def blob_has(items: Any, needle: str) -> bool:
    """Case-insensitive substring over the JSON form of ``items``."""
    return needle.lower() in json.dumps(items, default=str).lower()
