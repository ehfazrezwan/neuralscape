"""Connectors with ``CONNECTORS_ENABLED`` on: vault encryption at rest, owner isolation, REST management, real sync.

Real Redis holds the vault (``ns:connector:<id>``) and the ARQ queue; the REST
routes are the service's own (``/v1/connectors``) behind real per-user tokens; the
connector is the service's generic REST adapter pointed at a loopback HTTP source;
the sync runs the service's own ``process_connector_sync`` against a real Qdrant
collection with real embeddings (``llm_key``). #556's integration test covers the
revision bookkeeping (it replaces the ingest); this one covers what that leaves:

* the credentials blob is Fernet ciphertext in the raw Redis value (the secret never
  appears in plaintext there, in any API response, or in a listing) and decrypts
  back with the vault key; the non-secret fields stay readable;
* owner isolation: another user cannot list, get, delete or sync the connector, and
  cannot register over its id; each of those answers the same 404 as a missing id;
  the owner's record is byte-identical afterwards;
* list / get / delete work for the owner, and a sync request is accepted (202) and
  queued for the owner only;
* a sync run decrypts the credentials, sends them to the source, ingests the items
  into the owner's memory with ``source_ref`` pointing at the connector, and leaves
  the stored credentials encrypted;
* with the flag off the app's real startup builds no vault and every route is a 503 (and with it on, it does).

Flag used: ``connectors_enabled`` (+ ``vault_key``). Needs Redis, Qdrant, Neo4j (service
init) and GOOGLE_API_KEY.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

SECRET = "connector-it-token-secret"
BEARER = "nsit-conn-bearer-DO-NOT-LEAK-7f3a9c"


class _PerCallStore:
    """Redis commands on a fresh client per call (TestClient requests run on different event loops)."""

    def __init__(self, url):
        self._url = url

    def __getattr__(self, name):
        import redis.asyncio as aioredis

        async def call(*args, **kwargs):
            client = aioredis.Redis.from_url(self._url, decode_responses=True)
            try:
                return await getattr(client, name)(*args, **kwargs)
            finally:
                await client.aclose()

        return call


class _Source:
    """Loopback JSON API that records the Authorization header it receives."""

    def __init__(self):
        self.auth_seen: list[str | None] = []
        self.items = [{"id": f"doc-{n}", "rev": "1", "body": f"# Harbour note {n}\n\nThe ferry horn {n} is pitched in B flat."}
                      for n in (1, 2)]
        source = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                source.auth_seen.append(self.headers.get("Authorization"))
                body = json.dumps({"items": source.items}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/items"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class _FakeRedis:
    async def enqueue_job(self, *args, **kwargs):
        return SimpleNamespace(job_id="job")


def _bearer(uid):
    from tokens import issue_user_token

    return {"Authorization": f"Bearer {issue_user_token(uid, SECRET, 3600)}"}


@pytest.fixture
def env(flags, stores, llm_key, unique_ns, monkeypatch):
    from qdrant_client import QdrantClient

    import main
    from connectors.vault import ConnectorVault

    key = Fernet.generate_key().decode()
    queue = f"nsit-conn-q-{unique_ns.user_id[-10:]}"
    flags(
        connectors_enabled=True, vault_key=key, redis_url=stores.redis_url, ingest_queue_name=queue,
        neuralscape_user_token_secret=SECRET, neuralscape_api_key="",
        qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url,
    )
    store = _PerCallStore(stores.redis_url)
    from config import settings

    vault = ConnectorVault.from_settings(settings, store=store)
    monkeypatch.setattr(main, "_vault", vault)

    # enqueue through a pool made on the request's own loop (the pool is loop-bound)
    from arq import create_pool

    from config import parse_redis_settings
    from task_manager import TaskManager

    class PerCallPool:
        """Any ArqRedis command on a pool made on the calling loop."""

        def __getattr__(self, name):
            async def call(*args, **kwargs):
                p = await create_pool(parse_redis_settings())
                try:
                    return await getattr(p, name)(*args, **kwargs)
                finally:
                    await p.aclose()

            return call

    tm = TaskManager()
    tm.pool = PerCallPool()
    monkeypatch.setattr(main, "_task_manager", tm)

    alice, bob = unique_ns.user_id, unique_ns.user_id + "-bob"
    cid = f"nsit-conn-{unique_ns.user_id[-10:]}"
    source = _Source()

    def cleanup():
        source.close()
        asyncio.run(store.delete(f"ns:connector:{cid}", f"ns:connector:{cid}-b", queue))
        QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)

    unique_ns.on_cleanup(cleanup)
    from memory_service import MemoryService

    svc = MemoryService()
    svc._get_memory()
    return SimpleNamespace(
        client=TestClient(main.app), vault=vault, store=store, key=key, queue=queue, alice=alice, bob=bob,
        cid=cid, source=source, svc=svc, stores=stores,
    )


def _register(env, user, cid=None, **over):
    body = {
        "connector_id": cid or env.cid, "connector_type": "generic_rest", "name": "harbour notes",
        "credentials": {"bearer": BEARER},
        "config": {"list_url": env.source.url, "items_path": "items", "id_field": "id",
                   "revision_field": "rev", "content_field": "body"},
        "default_category": "domain_knowledge", "default_scope": "global",
    }
    body.update(over)
    return env.client.post("/v1/connectors", json=body, headers=_bearer(user))


def _raw(env, cid=None) -> str | None:
    return asyncio.run(env.store.get(f"ns:connector:{cid or env.cid}"))


def test_credentials_are_ciphertext_at_rest_and_never_returned(env):
    resp = _register(env, env.alice)
    assert resp.status_code == 201, resp.text
    assert BEARER not in resp.text and resp.json()["has_credentials"] is True

    raw = _raw(env)
    assert raw and BEARER not in raw, "the secret is in the raw Redis value in plaintext"
    stored = json.loads(raw)
    assert "credentials" not in stored and stored["credentials_enc"]
    assert stored["owner_user_id"] == env.alice and stored["connector_type"] == "generic_rest"   # non-secret fields readable
    assert json.loads(Fernet(env.key.encode()).decrypt(stored["credentials_enc"].encode())) == {"bearer": BEARER}
    with pytest.raises(Exception):
        Fernet(Fernet.generate_key()).decrypt(stored["credentials_enc"].encode())          # a different key cannot read it

    for r in (env.client.get("/v1/connectors", headers=_bearer(env.alice)),
              env.client.get(f"/v1/connectors/{env.cid}", headers=_bearer(env.alice))):
        assert r.status_code == 200 and BEARER not in r.text and "credentials_enc" not in r.text


def test_owner_can_list_get_and_delete_and_others_see_nothing(env):
    assert _register(env, env.alice).status_code == 201
    mine = env.client.get("/v1/connectors", headers=_bearer(env.alice)).json()["connectors"]
    assert [c["connector_id"] for c in mine] == [env.cid]
    assert env.client.get("/v1/connectors", headers=_bearer(env.bob)).json()["connectors"] == []
    got = env.client.get(f"/v1/connectors/{env.cid}", headers=_bearer(env.alice))
    assert got.status_code == 200 and got.json()["connector_id"] == env.cid and got.json()["owner_user_id"] == env.alice

    deleted = env.client.delete(f"/v1/connectors/{env.cid}", headers=_bearer(env.alice))
    assert deleted.status_code == 200 and deleted.json()["deleted"] is True
    assert _raw(env) is None
    assert env.client.get(f"/v1/connectors/{env.cid}", headers=_bearer(env.alice)).status_code == 404


def test_another_user_cannot_get_delete_sync_or_take_over_and_the_record_is_untouched(env):
    assert _register(env, env.alice).status_code == 201
    before = _raw(env)
    missing = env.client.get("/v1/connectors/nsit-no-such-connector", headers=_bearer(env.bob))

    get = env.client.get(f"/v1/connectors/{env.cid}", headers=_bearer(env.bob))
    delete = env.client.delete(f"/v1/connectors/{env.cid}", headers=_bearer(env.bob))
    sync = env.client.post(f"/v1/connectors/{env.cid}/sync", headers=_bearer(env.bob))
    takeover = _register(env, env.bob, credentials={"bearer": "attacker-secret"})
    for r in (get, delete, sync, takeover):
        assert r.status_code == 404, (r.request.method, r.request.url, r.status_code, r.text)
    assert missing.status_code == 404
    # no oracle: the answer for a foreign id is the answer for a missing id, apart from the id echoed back
    assert get.json()["detail"].replace(env.cid, "<id>") == missing.json()["detail"].replace("nsit-no-such-connector", "<id>")
    assert delete.json()["detail"] == get.json()["detail"] and sync.json()["detail"] == get.json()["detail"]
    assert _raw(env) == before, "a non-owner changed the stored record"
    assert env.client.get(f"/v1/connectors/{env.cid}", headers=_bearer(env.alice)).status_code == 200
    assert "attacker-secret" not in (_raw(env) or "")

    # nothing was queued by the refused sync
    queued = asyncio.run(env.store.zcard(env.queue))
    assert queued == 0


def test_owner_sync_request_is_accepted_and_queued(env):
    assert _register(env, env.alice).status_code == 201
    resp = env.client.post(f"/v1/connectors/{env.cid}/sync", headers=_bearer(env.alice))
    assert resp.status_code == 202, resp.text
    assert resp.json()["task_id"]
    assert asyncio.run(env.store.zcard(env.queue)) == 1


def test_a_sync_decrypts_ingests_into_the_owners_memory_and_keeps_credentials_encrypted(env):
    import worker

    assert _register(env, env.alice).status_code == 201
    record = asyncio.run(env.vault.get(env.cid))
    ctx = {"vault": env.vault, "service": env.svc, "redis": _FakeRedis()}

    out = asyncio.run(worker.process_connector_sync(ctx, env.cid, record["generation"]))

    assert out["synced"] == 2 and out["skipped"] == 0, out
    assert f"Bearer {BEARER}" in env.source.auth_seen, "the decrypted credential was not sent to the source"
    rows = env.svc._scroll_all_user_memories(env.alice)
    refs = [(r["payload"].get("metadata") or {}).get("source_ref") or {} for r in rows]
    assert refs and all(r.get("connector_id") == env.cid for r in refs if r), refs
    assert {r.get("external_id") for r in refs if r.get("external_id")} >= {"doc-1", "doc-2"}
    assert env.svc._scroll_all_user_memories(env.bob) == [], "another user received the connector's memories"

    after = _raw(env)
    assert BEARER not in after and json.loads(after)["last_revision_by_id"] == {"doc-1": "1", "doc-2": "1"}


def test_flag_off_the_real_startup_builds_no_vault_and_every_route_is_503(env, flags, monkeypatch):
    """Runs the app's real lifespan: with the flag off it leaves no vault, so every connector route is a 503."""
    import main

    flags(connectors_enabled=False)
    monkeypatch.setattr(main, "_vault", None)
    h = _bearer(env.alice)
    with TestClient(main.app) as c:
        assert main._vault is None
        assert c.get("/v1/connectors", headers=h).status_code == 503
        assert c.get(f"/v1/connectors/{env.cid}", headers=h).status_code == 503
        assert c.delete(f"/v1/connectors/{env.cid}", headers=h).status_code == 503
        assert c.post(f"/v1/connectors/{env.cid}/sync", headers=h).status_code == 503
        reg = c.post("/v1/connectors", headers=h, json={
            "connector_id": env.cid, "connector_type": "generic_rest", "credentials": {"bearer": BEARER}})
        assert reg.status_code == 503
    assert _raw(env) is None


def test_flag_on_the_real_startup_builds_the_vault_and_the_routes_answer(env, flags, monkeypatch):
    import main

    flags(connectors_enabled=True)
    monkeypatch.setattr(main, "_vault", None)
    with TestClient(main.app) as c:
        assert main._vault is not None, "startup did not build the vault with CONNECTORS_ENABLED and a valid key"
        r = c.get("/v1/connectors", headers=_bearer(env.alice))
        assert r.status_code == 200 and r.json() == {"connectors": []}
