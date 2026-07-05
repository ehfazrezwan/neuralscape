"""Async Neuralscape REST client for benchmarking.

Targets only core endpoints present on both `dev` and the perf branch:
writes (202 → poll), search/graph-search/list/context reads, health, and a
namespaced bulk-delete for cleanup. Auth is optional (local dev needs none;
pass a bearer token otherwise).
"""

from __future__ import annotations

import asyncio
import time

import httpx


class TaskTimeout(Exception):
    """An async write did not reach a terminal status within the poll timeout."""


class NeuralscapeClient:
    def __init__(self, base_url: str, token: str | None = None, request_timeout: float = 120.0,
                 max_connections: int = 100, http: httpx.AsyncClient | None = None):
        # Tests inject a MockTransport-backed client here instead of overwriting
        # the private ._http afterward (which would leak the client created below).
        if http is not None:
            self._http = http
            return
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=request_timeout,
            # The stress test fans out to users x per-user-concurrency clients;
            # raise the pool ceiling so the harness doesn't self-throttle.
            limits=httpx.Limits(max_connections=max_connections,
                                max_keepalive_connections=max_connections),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    # ── health ──
    async def health(self) -> dict:
        r = await self._http.get("/health")
        r.raise_for_status()
        return r.json()

    # ── writes ──
    async def raw_write(
        self,
        content: str,
        *,
        user_id: str,
        category: str = "domain_knowledge",
        scope: str = "global",
        project_id: str | None = None,
        tags: list[str] | None = None,
        visibility: str | None = None,
    ) -> dict:
        """POST /v1/memories/raw. Returns the response body ({task_id,...} on 202)."""
        body: dict = {"content": content, "user_id": user_id, "category": category, "scope": scope}
        if project_id:
            body["project_id"] = project_id
        if tags:
            body["tags"] = tags
        if visibility:
            body["visibility"] = visibility
        r = await self._http.post("/v1/memories/raw", json=body)
        r.raise_for_status()
        return r.json()

    async def extract_write(self, messages: list[dict], *, user_id: str, project_id: str | None = None,
                            run_id: str | None = None) -> dict:
        body: dict = {"messages": messages, "user_id": user_id}
        if project_id:
            body["project_id"] = project_id
        if run_id:
            body["run_id"] = run_id
        r = await self._http.post("/v1/memories", json=body)
        r.raise_for_status()
        return r.json()

    async def wait_for_task(self, task_id: str, *, timeout_s: float, interval_s: float) -> dict:
        """Poll GET /v1/memories/status/{task_id} until terminal. Raises TaskTimeout."""
        deadline = time.perf_counter() + timeout_s
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TaskTimeout(f"task {task_id} not terminal after {timeout_s}s")
            # Bound each poll by the time left so a single stalled GET can't run
            # far past the declared poll timeout (up to the client-wide timeout).
            r = await self._http.get(f"/v1/memories/status/{task_id}",
                                     timeout=max(0.001, remaining))
            # 404 is a terminal "task unknown/expired" state, not a hard error
            # that should abort the run — surface it like the other terminals.
            if r.status_code == 404:
                return {"task_id": task_id, "status": "not_found"}
            r.raise_for_status()
            data = r.json()
            status = data.get("status")
            if status in ("completed", "failed", "not_found"):
                return data
            await asyncio.sleep(interval_s)

    # ── reads (synchronous 200) ──
    async def search(self, query: str, *, user_id: str, project_id: str | None = None, limit: int = 10) -> dict:
        body: dict = {"query": query, "user_id": user_id, "limit": limit}
        if project_id:
            body["project_id"] = project_id
        r = await self._http.post("/v1/search", json=body)
        r.raise_for_status()
        return r.json()

    async def graph_search(self, query: str, *, user_id: str, project_id: str | None = None, limit: int = 10) -> dict:
        body: dict = {"query": query, "user_id": user_id, "limit": limit}
        if project_id:
            body["project_id"] = project_id
        r = await self._http.post("/v1/graph/search", json=body)
        r.raise_for_status()
        return r.json()

    async def ask(self, question: str, *, user_id: str, project_id: str | None = None,
                  reasoning_level: str = "high") -> dict:
        """POST /v1/ask — reasoning-tiered question answering (C3). Sync 200."""
        body: dict = {"question": question, "user_id": user_id,
                      "reasoning_level": reasoning_level}
        if project_id:
            body["project_id"] = project_id
        r = await self._http.post("/v1/ask", json=body)
        r.raise_for_status()
        return r.json()

    async def list_memories(self, *, user_id: str, limit: int = 100) -> dict:
        r = await self._http.get("/v1/memories", params={"user_id": user_id, "limit": limit})
        r.raise_for_status()
        return r.json()

    async def context_global(self, *, user_id: str) -> dict:
        r = await self._http.get("/v1/context/global", params={"user_id": user_id})
        r.raise_for_status()
        return r.json()

    # ── cleanup ──
    async def delete_bench_data(self, *, user_id: str) -> dict:
        """Bulk-delete everything owned by a bench user (DELETE /v1/memories + body).

        Best-effort: bench data is namespaced, so a failure here is non-fatal.
        """
        try:
            r = await self._http.request(
                "DELETE", "/v1/memories",
                json={"user_id": user_id, "include_shared": True},
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 — cleanup must never crash a run
            return {"status": "cleanup_failed", "error": str(e)}
