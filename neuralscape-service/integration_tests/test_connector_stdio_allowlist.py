"""Stdio MCP connector with CONNECTORS_ENABLED on, real Redis vault, real subprocess.

Registers connectors through the REST app, reads them back from the Redis-backed
vault and runs the adapter against a spawned MCP stdio server. Requires Redis at
REDIS_URL (the gate box provides it); an unreachable Redis is a failure, never a skip.
"""

from __future__ import annotations

import asyncio
import shlex
import sys
import uuid
from pathlib import Path

import pytest
import redis.asyncio as aioredis
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

import main
from config import settings
from connectors.registry import build_adapter
from connectors.stdio_policy import ConnectorConfigError
from connectors.vault import ConnectorVault

pytestmark = pytest.mark.integration

FIXTURE = str(Path(__file__).with_name("mcp_stdio_fixture_server.py"))
COMMAND = [sys.executable, FIXTURE]


class _PerCallRedis:
    """Opens a Redis client per operation so each event loop (TestClient request,
    asyncio.run) gets its own connection."""

    def __init__(self, url: str):
        self._url = url

    async def _run(self, op: str, *a):
        r = aioredis.Redis.from_url(self._url, decode_responses=True)
        try:
            return await getattr(r, op)(*a)
        finally:
            await r.aclose()

    async def get(self, *a):
        return await self._run("get", *a)

    async def delete(self, *a):
        return await self._run("delete", *a)

    async def keys(self, *a):
        return await self._run("keys", *a)

    async def eval(self, *a):
        return await self._run("eval", *a)


@pytest.fixture
def stack(monkeypatch):
    # No auth secrets + explicit opt-out: the same state as a trusted local machine.
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", "")
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    monkeypatch.setattr(settings, "neuralscape_auth_disabled", True)
    monkeypatch.setattr(settings, "connectors_enabled", True)
    monkeypatch.setattr(settings, "vault_key", Fernet.generate_key().decode())
    monkeypatch.setattr(settings, "connector_stdio_commands", shlex.join(COMMAND))
    vault = ConnectorVault(_PerCallRedis(settings.redis_url), Fernet(settings.vault_key.encode()))
    # Fail loudly (not skip) if Redis is not there.
    assert asyncio.run(vault._store.eval("return 1", 0)) == 1
    monkeypatch.setattr(main, "_vault", vault)
    client = TestClient(main.app, raise_server_exceptions=False)
    created: list[str] = []
    yield client, vault, created
    for cid in created:
        asyncio.run(vault.delete(cid))


def _id() -> str:
    return f"stdio-it-{uuid.uuid4().hex[:10]}"


def _body(cid: str, config: dict) -> dict:
    return {"connector_id": cid, "connector_type": "mcp", "credentials": {}, "config": config}


def test_allowlisted_stdio_connector_registers_and_reads_real_resources(stack):
    client, vault, created = stack
    cid = _id()
    created.append(cid)
    cfg = {"transport": "stdio", "command": COMMAND[0], "args": COMMAND[1:], "server_name": "fixture"}
    r = client.post("/v1/connectors", json=_body(cid, cfg))
    assert r.status_code == 201, r.text

    record = asyncio.run(vault.get(cid))
    assert record["config"]["command"] == COMMAND[0]
    adapter = build_adapter(record)

    async def run():
        resources, cursor = await adapter.list_resources()
        assert cursor is None
        assert sorted(r.external_id for r in resources) == ["fixture://alpha", "fixture://beta"]
        alpha = next(r for r in resources if r.external_id == "fixture://alpha")
        return await adapter.fetch(alpha)

    assert asyncio.run(run()) == "alpha resource body: stdio connector works"


def test_non_allowlisted_command_refused_at_registration_and_never_spawns(stack, tmp_path):
    client, vault, created = stack
    marker = tmp_path / "ran"
    cid = _id()
    created.append(cid)
    cfg = {"transport": "stdio", "command": sys.executable, "args": ["-c", f"open({str(marker)!r}, 'w')"]}
    r = client.post("/v1/connectors", json=_body(cid, cfg))
    assert r.status_code == 422
    assert "allowlist" in r.json()["detail"]
    assert asyncio.run(vault.get(cid)) is None
    assert not marker.exists()


def test_legacy_stored_stdio_config_refused_at_session_time_and_never_spawns(stack, tmp_path):
    client, vault, created = stack
    marker = tmp_path / "ran"
    cid = _id()
    created.append(cid)
    # A record stored before the allowlist existed: written straight to the vault.
    asyncio.run(vault.put({
        "connector_id": cid, "connector_type": "mcp", "credentials": {},
        "owner_user_id": settings.default_user_id,
        "config": {"transport": "stdio", "command": sys.executable,
                   "args": ["-c", f"open({str(marker)!r}, 'w')"]},
    }))
    adapter = build_adapter(asyncio.run(vault.get(cid)))
    with pytest.raises(ConnectorConfigError, match="allowlist"):
        asyncio.run(adapter.list_resources())
    assert not marker.exists()


def test_stdio_disabled_when_allowlist_empty(stack, monkeypatch):
    client, vault, created = stack
    monkeypatch.setattr(settings, "connector_stdio_commands", "")
    cid = _id()
    created.append(cid)
    cfg = {"transport": "stdio", "command": COMMAND[0], "args": COMMAND[1:]}
    r = client.post("/v1/connectors", json=_body(cid, cfg))
    assert r.status_code == 422 and "not enabled" in r.json()["detail"]


# ── worker sync path (process_connector_sync), real Redis vault + real subprocess ──


def _worker_ctx(vault):
    return {"vault": vault, "service": object()}


@pytest.fixture
def ingested(monkeypatch):
    """Capture what the sync would ingest; the stores behind ingest are out of scope here."""
    import ingest.pipeline as pipeline
    import worker

    docs: list = []

    def fake_ingest(service, doc):
        docs.append(doc)
        return {"adapter": None, "graph_jobs": []}

    async def no_graph(*a, **k):
        return None

    monkeypatch.setattr(pipeline, "ingest_document", fake_ingest)
    monkeypatch.setattr(worker, "_enqueue_graph_jobs", no_graph)
    return docs


def test_worker_sync_ingests_allowlisted_stdio_resources(stack, ingested):
    import worker

    client, vault, created = stack
    cid = _id()
    created.append(cid)
    cfg = {"transport": "stdio", "command": COMMAND[0], "args": COMMAND[1:]}
    assert client.post("/v1/connectors", json=_body(cid, cfg)).status_code == 201
    generation = asyncio.run(vault.get(cid))["generation"]

    result = asyncio.run(worker.process_connector_sync(_worker_ctx(vault), cid, generation))

    assert result["synced"] == 2, result
    texts = sorted(d.content for d in ingested)
    assert texts == ["alpha resource body: stdio connector works", "beta resource body"]
    assert asyncio.run(vault.get(cid))["last_synced_at"]


def test_worker_sync_refuses_legacy_stdio_and_spawns_nothing(stack, ingested, tmp_path):
    import worker

    client, vault, created = stack
    marker = tmp_path / "ran"
    cid = _id()
    created.append(cid)
    asyncio.run(vault.put({
        "connector_id": cid, "connector_type": "mcp", "credentials": {},
        "owner_user_id": settings.default_user_id,
        "config": {"transport": "stdio", "command": sys.executable,
                   "args": ["-c", f"open({str(marker)!r}, 'w')"]},
    }))
    generation = asyncio.run(vault.get(cid))["generation"]

    with pytest.raises(ConnectorConfigError):
        asyncio.run(worker.process_connector_sync(_worker_ctx(vault), cid, generation))

    assert not marker.exists()
    assert ingested == []
    assert asyncio.run(vault.get(cid))["last_synced_at"] is None
