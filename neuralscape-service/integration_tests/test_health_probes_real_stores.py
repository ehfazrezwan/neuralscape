"""Real-store proof that ``/health`` probes the vector store (and ``/health/live`` does not).

Qdrant, Redis and Neo4j are real (the ``stores`` fixture probes all three and
fails on the gate when one is down). The app, its auth middleware and the
``/health`` handler are the real ones; only the service handle that carries the
vector-store client is a thin holder, because building the full MemoryService
would also need the LLM and embedder.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

UNREACHABLE_QDRANT = "http://127.0.0.1:1"


class _RealRedisPool:
    """``ping`` through a fresh real connection (the TestClient serves on its own loop)."""

    def __init__(self, url: str) -> None:
        self.url = url

    async def ping(self) -> bool:
        from redis import asyncio as aioredis

        client = aioredis.from_url(self.url)
        try:
            return bool(await client.ping())
        finally:
            await client.aclose()


def _wire(monkeypatch, stores, qdrant_url: str) -> TestClient:
    from qdrant_client import QdrantClient

    import knowledge
    import main

    client = QdrantClient(url=qdrant_url, timeout=2)
    monkeypatch.setattr(knowledge, "list_systems", lambda: [])
    monkeypatch.setattr(
        main,
        "_service",
        SimpleNamespace(_memory=SimpleNamespace(vector_store=SimpleNamespace(client=client)), _graphiti=object()),
    )
    monkeypatch.setattr(main, "_task_manager", SimpleNamespace(pool=_RealRedisPool(stores.redis_url)))
    return TestClient(main.app, raise_server_exceptions=False)


def test_health_is_ok_with_real_stores(stores, monkeypatch):
    http = _wire(monkeypatch, stores, stores.qdrant_url)

    resp = http.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"]["redis"] == "ok"
    assert body["checks"]["vector_store"] == "ok"
    assert resp.headers["X-Request-ID"]


def test_health_reports_the_vector_store_down_while_liveness_stays_up(stores, monkeypatch):
    http = _wire(monkeypatch, stores, UNREACHABLE_QDRANT)

    ready = http.get("/health")
    live = http.get("/health/live")

    assert ready.status_code == 503
    assert ready.json()["status"] == "unhealthy"
    assert ready.json()["checks"]["vector_store"] == "unreachable"
    assert ready.json()["checks"]["redis"] == "ok"  # the other backends are still reported
    assert live.status_code == 200
    assert live.json() == {"status": "alive"}
