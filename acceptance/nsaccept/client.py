"""REST client for the stack's /v1 surface (bearer auth)."""

from __future__ import annotations

import time
from typing import Any, Callable

import httpx

TERMINAL = {"completed", "failed"}


class NSClient:
    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        user_id: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.token = token
        self.http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def close(self) -> None:
        self.http.close()

    def request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        return self.http.request(method, path, **kw)

    def get(self, path: str, **kw: Any) -> httpx.Response:
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw: Any) -> httpx.Response:
        return self.request("POST", path, **kw)

    def delete(self, path: str, **kw: Any) -> httpx.Response:
        return self.request("DELETE", path, **kw)

    # ── async writes ────────────────────────────────────────────
    def store_raw(
        self,
        content: str,
        category: str,
        *,
        scope: str = "global",
        project_id: str | None = None,
        visibility: str | None = "private",
        **extra: Any,
    ) -> str:
        """POST /v1/memories/raw; returns the task id (expects 202)."""
        body: dict[str, Any] = {"content": content, "category": category, "scope": scope}
        if project_id:
            body["project_id"] = project_id
        if visibility:
            body["visibility"] = visibility
        body.update(extra)
        resp = self.post("/v1/memories/raw", json=body)
        assert resp.status_code == 202, f"POST /v1/memories/raw -> {resp.status_code}: {resp.text[:200]}"
        task_id = resp.json().get("task_id")
        assert task_id, f"202 without task_id: {resp.text[:200]}"
        return task_id

    def task_status(self, task_id: str) -> dict:
        resp = self.get(f"/v1/memories/status/{task_id}")
        assert resp.status_code == 200, f"status poll -> {resp.status_code}: {resp.text[:200]}"
        return resp.json()

    def poll_task(
        self,
        task_id: str,
        timeout: float = 120.0,
        interval: float = 1.0,
        not_found_grace: float = 10.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> dict:
        """Poll GET /v1/memories/status/{id} until completed or failed.

        Returns the final status dict. On timeout returns a synthetic
        ``{"status": "timeout", ...}`` instead of raising, so a fixture can
        record it and the scenario can assert on it. ``not_found`` is tolerated
        for ``not_found_grace`` seconds, then reported as final.
        """
        start = clock()
        last: dict = {"task_id": task_id, "status": "timeout"}
        while clock() - start < timeout:
            last = self.task_status(task_id)
            status = last.get("status")
            if status in TERMINAL:
                return last
            if status == "not_found" and clock() - start >= not_found_grace:
                return last
            sleep(interval)
        return {**last, "status": "timeout", "last_status": last.get("status")}

    # ── reads ───────────────────────────────────────────────────
    def search(self, query: str, *, project_id: str | None = None, limit: int = 10, **extra: Any) -> list[dict]:
        body: dict[str, Any] = {"query": query, "limit": limit}
        if project_id:
            body["project_id"] = project_id
        body.update(extra)
        resp = self.post("/v1/search", json=body)
        assert resp.status_code == 200, f"POST /v1/search -> {resp.status_code}: {resp.text[:200]}"
        return resp.json().get("results", [])
