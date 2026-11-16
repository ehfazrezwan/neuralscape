"""Connector destination policy with CONNECTORS_ENABLED on, real Redis vault.

* Registering connectors that point at loopback, the gate's own Redis host or the
  cloud metadata address is refused (422) and nothing reaches the vault.
* A connector stored before the guard existed (written straight to the vault) is
  refused at sync time and the target never sees a request.
* A connector to an allowlisted internal fixture host (an in-test HTTP server on
  127.0.0.1, and a real MCP Streamable HTTP server) works end to end through
  ``worker.process_connector_sync``, over the real guarded socket path.

Needs Redis (the ``stores`` fixture); an unreachable Redis fails on the gate.
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest
import redis.asyncio as aioredis
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

import main
from config import settings
from connectors.destination import DestinationRefused
from connectors.vault import ConnectorVault

pytestmark = pytest.mark.integration

MCP_FIXTURE = str(Path(__file__).with_name("mcp_http_fixture_server.py"))


class _PerCallRedis:
    """Opens a Redis client per operation so each event loop gets its own connection."""

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
def stack(monkeypatch, stores):
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", "")
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    monkeypatch.setattr(settings, "neuralscape_auth_disabled", True)
    monkeypatch.setattr(settings, "connectors_enabled", True)
    monkeypatch.setattr(settings, "vault_key", Fernet.generate_key().decode())
    monkeypatch.setattr(settings, "connector_allowed_hosts", "")
    vault = ConnectorVault(_PerCallRedis(stores.redis_url), Fernet(settings.vault_key.encode()))
    assert asyncio.run(vault._store.eval("return 1", 0)) == 1
    monkeypatch.setattr(main, "_vault", vault)
    client = TestClient(main.app, raise_server_exceptions=False)
    created: list[str] = []
    yield client, vault, created
    for cid in created:
        asyncio.run(vault.delete(cid))


def _id() -> str:
    return f"ssrf-it-{uuid.uuid4().hex[:10]}"


BEARER = f"it-{uuid.uuid4().hex}"  # generated per run: no literal credential in the tree


def _body(cid, ctype, config):
    return {"connector_id": cid, "connector_type": ctype, "credentials": {"bearer": BEARER}, "config": config}


@pytest.fixture
def ingested(monkeypatch):
    """Capture what a sync would ingest; the stores behind ingest are out of scope here."""
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


def _worker_ctx(vault):
    return {"vault": vault, "service": object()}


# ── an in-test REST target on loopback ──────────────────────────────────────


class _Target:
    def __init__(self):
        self.hits: list[tuple[str, str | None]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep test output quiet
                pass

            def do_GET(self):  # noqa: N802
                outer.hits.append((self.path, self.headers.get("Authorization")))
                if self.path == "/items":
                    body = json.dumps([{"id": "a"}, {"id": "b"}])
                else:
                    body = json.dumps({"text": f"body of {self.path.rsplit('/', 1)[-1]}"})
                data = body.encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def target():
    with _Target() as t:
        yield t


def _rest_config(port):
    base = f"http://127.0.0.1:{port}"
    return {"list_url": f"{base}/items", "item_url_template": f"{base}/items/{{id}}", "detail_content_field": "text"}


# ── refusals ────────────────────────────────────────────────────────────────


def test_connectors_aimed_at_internal_targets_are_refused_and_not_stored(stack, stores, target):
    client, vault, created = stack
    redis_host = urlparse(stores.redis_url).hostname or "localhost"
    internal = [
        f"http://127.0.0.1:{target.port}/items",  # loopback, plain http
        f"https://127.0.0.1:{target.port}/items",  # loopback, https
        "https://localhost/items",
        f"https://{redis_host}:6379/",  # the gate's own Redis host
        "https://169.254.169.254/latest/meta-data/",
        "https://[fd00:ec2::254]/latest/meta-data/",
        "https://[::ffff:127.0.0.1]/items",
        "https://2130706433/items",
    ]
    for url in internal:
        for ctype, config in (("generic_rest", {"list_url": url}), ("mcp", {"url": url})):
            cid = _id()
            created.append(cid)
            r = client.post("/v1/connectors", json=_body(cid, ctype, config))
            assert r.status_code == 422, (ctype, url, r.text)
            assert asyncio.run(vault.get(cid)) is None, (ctype, url)
    assert target.hits == []  # nothing was ever fetched


def test_stored_internal_connector_is_refused_at_sync_and_target_sees_nothing(stack, ingested, target):
    import worker

    client, vault, created = stack
    cid = _id()
    created.append(cid)
    # A record stored before the guard existed: written straight to the vault.
    asyncio.run(vault.put({
        "connector_id": cid, "connector_type": "generic_rest", "credentials": {"bearer": BEARER},
        "owner_user_id": settings.default_user_id, "config": _rest_config(target.port),
    }))
    generation = asyncio.run(vault.get(cid))["generation"]

    with pytest.raises(DestinationRefused):
        asyncio.run(worker.process_connector_sync(_worker_ctx(vault), cid, generation))

    assert target.hits == []
    assert ingested == []
    assert asyncio.run(vault.get(cid))["last_synced_at"] is None


# ── allowlisted internal host works end to end ──────────────────────────────


def test_allowlisted_internal_rest_connector_syncs_end_to_end(stack, ingested, target, monkeypatch):
    import worker

    client, vault, created = stack
    monkeypatch.setattr(settings, "connector_allowed_hosts", f"127.0.0.1:{target.port}")
    cid = _id()
    created.append(cid)
    r = client.post("/v1/connectors", json=_body(cid, "generic_rest", _rest_config(target.port)))
    assert r.status_code == 201, r.text
    generation = asyncio.run(vault.get(cid))["generation"]

    result = asyncio.run(worker.process_connector_sync(_worker_ctx(vault), cid, generation))

    assert result["synced"] == 2, result
    assert sorted(d.content for d in ingested) == ["body of a", "body of b"]
    assert {path for path, _ in target.hits} == {"/items", "/items/a", "/items/b"}
    assert all(auth == f"Bearer {BEARER}" for _, auth in target.hits)  # credentials only reach the allowlisted host
    assert asyncio.run(vault.get(cid))["last_synced_at"]


def test_allowlist_is_exact_to_the_port(stack, target, monkeypatch):
    client, vault, created = stack
    monkeypatch.setattr(settings, "connector_allowed_hosts", f"127.0.0.1:{target.port + 1}")
    cid = _id()
    created.append(cid)
    r = client.post("/v1/connectors", json=_body(cid, "generic_rest", _rest_config(target.port)))
    assert r.status_code == 422


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_allowlisted_internal_mcp_http_connector_syncs_end_to_end(stack, ingested, monkeypatch):
    import worker

    client, vault, created = stack
    port = _free_port()
    proc = subprocess.Popen([sys.executable, MCP_FIXTURE, str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                assert proc.poll() is None, "MCP fixture server exited"
                time.sleep(0.3)
        else:
            pytest.fail("MCP fixture server did not start")

        url = f"http://127.0.0.1:{port}/mcp"
        cid = _id()
        created.append(cid)
        # Flag-on contrast: refused until the exact host:port is allowlisted.
        assert client.post("/v1/connectors", json=_body(cid, "mcp", {"url": url})).status_code == 422
        monkeypatch.setattr(settings, "connector_allowed_hosts", f"127.0.0.1:{port}")
        assert client.post("/v1/connectors", json=_body(cid, "mcp", {"url": url})).status_code == 201
        generation = asyncio.run(vault.get(cid))["generation"]

        result = asyncio.run(worker.process_connector_sync(_worker_ctx(vault), cid, generation))

        assert result["synced"] == 1, result
        assert [d.content for d in ingested] == ["gamma resource body: http connector works"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
