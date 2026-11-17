"""Real-store proof that ``/health`` probes the vector store (and ``/health/live`` does not).

Qdrant, Redis and Neo4j are real (the ``stores`` fixture probes all three and
fails on the gate when one is down). The app, its auth middleware and the
``/health`` handler are the real ones; only the service handle that carries the
vector-store client is a thin holder, because building the full MemoryService
would also need the LLM and embedder.
"""

from __future__ import annotations

import asyncio
import threading
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


class _BridgeLoop:
    """A real event loop on its own thread, like the Graphiti bridge: the Neo4j driver lives on it."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def build_driver(self, uri: str, user: str, password: str):
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        async def make():
            return Neo4jDriver(uri, user, password)

        return asyncio.run_coroutine_threadsafe(make(), self.loop).result(10)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()


def _graph_service(qdrant_client, bridge: _BridgeLoop | None, driver):
    """The service handle /health reads: a vector client, plus a graph that is (or is not yet) initialised."""

    from memory_service import MemoryService

    fake = SimpleNamespace(_bridge=SimpleNamespace(_loop=bridge.loop if bridge else None))
    return SimpleNamespace(
        _memory=SimpleNamespace(vector_store=SimpleNamespace(client=qdrant_client)),
        _graphiti=SimpleNamespace(driver=driver) if driver is not None else None,
        _run_on_bridge=lambda coro, timeout=30.0: MemoryService._run_on_bridge(fake, coro, timeout),
    )


def _wire(monkeypatch, stores, qdrant_url: str, *, neo4j_uri: str | None = None, graph_initialised: bool = True):
    """Returns ``(TestClient, bridge)``; the caller closes the bridge."""

    from qdrant_client import QdrantClient

    import knowledge
    import main

    client = QdrantClient(url=qdrant_url, timeout=2)
    bridge = _BridgeLoop()
    driver = (
        bridge.build_driver(neo4j_uri or stores.neo4j_uri, stores.neo4j_user, stores.neo4j_password)
        if graph_initialised
        else None
    )
    monkeypatch.setattr(knowledge, "list_systems", lambda: [])
    monkeypatch.setattr(main, "_service", _graph_service(client, bridge, driver))
    monkeypatch.setattr(main, "_task_manager", SimpleNamespace(pool=_RealRedisPool(stores.redis_url)))
    return TestClient(main.app, raise_server_exceptions=False), bridge


def test_health_is_ok_with_real_stores(stores, monkeypatch):
    http, bridge = _wire(monkeypatch, stores, stores.qdrant_url)
    try:
        resp = http.get("/health")
    finally:
        bridge.close()

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok", body
    assert body["checks"]["redis"] == "ok", body
    assert body["checks"]["vector_store"] == "ok", body
    assert body["checks"]["graph_store"] == "ok", body  # a real Neo4j round trip on the first call
    assert resp.headers["X-Request-ID"]


def test_health_reports_a_not_yet_initialised_graph_without_probing_it(stores, monkeypatch):
    """Cold service (graph not built yet): the state is reported as before, never probed or built."""
    http, bridge = _wire(monkeypatch, stores, stores.qdrant_url, graph_initialised=False)
    try:
        resp = http.get("/health")
    finally:
        bridge.close()

    body = resp.json()
    assert resp.status_code == 200, body
    assert body["checks"]["graph_store"] == "not_initialized", body
    assert body["checks"]["vector_store"] == "ok", body
    assert body["status"] == "degraded", body


def test_health_reports_an_unreachable_graph_as_degraded_not_unhealthy(stores, monkeypatch):
    http, bridge = _wire(monkeypatch, stores, stores.qdrant_url, neo4j_uri="neo4j://127.0.0.1:1")
    try:
        resp = http.get("/health")
    finally:
        bridge.close()

    body = resp.json()
    assert resp.status_code == 200, body
    assert body["checks"]["graph_store"] == "unreachable", body
    assert body["status"] == "degraded", body


def test_health_reports_the_vector_store_down_while_liveness_stays_up(stores, monkeypatch):
    http, bridge = _wire(monkeypatch, stores, UNREACHABLE_QDRANT)
    try:
        ready = http.get("/health")
        live = http.get("/health/live")
    finally:
        bridge.close()

    assert ready.status_code == 503, ready.text
    assert ready.json()["status"] == "unhealthy", ready.json()
    assert ready.json()["checks"]["vector_store"] == "unreachable", ready.json()
    assert ready.json()["checks"]["redis"] == "ok", ready.json()  # the other backends are still reported
    assert live.status_code == 200
    assert live.json() == {"status": "alive"}
