"""``ENGINE_ROUTING_ENABLED=true`` end to end through the privileged admin routes, on real stores.

Real: a Qdrant server (generation 1, the routed projection collection), PostgreSQL with pgvector (the
shadow, then generation 2), a SQLite ledger (the canonical source: object references and tombstones),
the switch the service builds at start-up (``main._start_engine_switch`` -> ``engine_adapters_switch.start``
-> ``build_default_switch``) and the FastAPI app with its real dictator-only routes.

Stood in for: the memory service's rows (a dict behind ``get_memory``; the real service needs the
extraction LLM) and the embedder (a deterministic token hash; no model is called), and the ledger's vector
stage, whose caller lands in a later pull request: ``switch.project_apply`` / ``project_remove`` are the exact
entry points that stage calls, so the writes below are made through them.

Cycle: backfill -> start-dual-write -> a write reaches BOTH engines -> catch-up -> flip -> a read is served by
the flipped engine (a post-flip write exists only there) -> rollback (replayed onto the old engine, generation
never goes backwards) -> a second window -> abandon.  Erasure fences are checked on both engines throughout.

Needs a disposable Qdrant and PostgreSQL with the ``vector`` extension (see ``engine_env``); fails instead of
skipping when ``NS_INTEGRATION_REQUIRED`` is set.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from engine_env import PG_DSN, QDRANT_URL, require_env

pytestmark = [pytest.mark.integration]

DIMS = 64
ADMIN = "engine-admin"
TENANT = "tenant-rest"
BASE = "/v1/admin/engine-switch"


@pytest.fixture(autouse=True, scope="module")
def _stores_configured():
    require_env(PostgreSQL=PG_DSN, Qdrant=QDRANT_URL)


class _Rows:
    def __init__(self) -> None:
        self.rows: dict[str, SimpleNamespace] = {}

    def put(self, memory_id: str, text: str) -> None:
        self.rows[memory_id] = SimpleNamespace(memory=text, owner_user_id=ADMIN, visibility="private", project_id=None)

    def get_memory(self, memory_id: str):
        return self.rows.get(memory_id)


def _pg_count(schema: str, record_id: str) -> int:
    import psycopg

    with psycopg.connect(PG_DSN) as conn:
        return conn.execute(
            f"SELECT count(*) FROM {schema}.records WHERE record_id = %s AND NOT removed", (record_id,)
        ).fetchone()[0]


def _qdrant_has(stack, record_id: str) -> bool:
    """Whether the generation-1 Qdrant collection holds the record (read through its own adapter)."""

    from engine_ports import OperationContext

    ctx = OperationContext(tenant_id=TENANT, correlation_id="rest-test", deadline_ms=30_000)
    return stack.qdrant_adapter.get(ctx, record_id) is not None


@pytest.fixture()
def stack(monkeypatch, tmp_path):
    import psycopg
    from qdrant_client import QdrantClient

    import engine_adapters_switch as switch_module
    import main
    from config import settings
    from engine_adapters_current import hashed_bag_embedder
    from ledger.sqlite import SqliteLedger

    collection = "ns_rest_" + uuid.uuid4().hex[:10]
    schema = "ns_rest_" + uuid.uuid4().hex[:10]
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    for name, value in {
        "engine_routing_enabled": True,
        "engine_routing_recall_enabled": True,
        "engine_routing_qdrant_collection": collection,
        "engine_routing_dimensions": DIMS,
        "engine_pgvector_enabled": True,
        "engine_pgvector_dsn": PG_DSN,
        "engine_pgvector_schema": schema,
        "engine_pgvector_dimensions": DIMS,
        "qdrant_url": QDRANT_URL,
        "dictator_user_ids": ADMIN,
        "default_user_id": ADMIN,
        "neuralscape_auth_disabled": True,
        "neuralscape_user_token_secret": "",
        "neuralscape_api_key": "",
    }.items():
        monkeypatch.setattr(settings, name, value)

    rows = _Rows()
    embed = hashed_bag_embedder(DIMS)
    memory = SimpleNamespace(vector_store=None, embedding_model=SimpleNamespace(embed=embed))
    monkeypatch.setattr(main, "_service", SimpleNamespace(_get_memory=lambda: memory, get_memory=rows.get_memory))
    ledger = SqliteLedger(str(tmp_path / "ledger.db"))
    monkeypatch.setattr(main._task_manager, "ledger", ledger, raising=False)
    monkeypatch.setattr(
        main._task_manager, "producer", SimpleNamespace(tenant_for=lambda user: TENANT), raising=False
    )

    asyncio.run(main._start_engine_switch())  # the service's own start-up path
    switch = switch_module.get_switch()
    assert switch is not None, switch_module.unavailable_reason()

    client = QdrantClient(url=QDRANT_URL)
    yield SimpleNamespace(
        switch=switch, ledger=ledger, rows=rows, collection=collection, schema=schema, qdrant=client, main=main,
        qdrant_adapter=switch.router._active.adapter,  # generation 1, kept to read the old engine after a flip
    )

    asyncio.run(main._stop_engine_switch())
    try:
        client.delete_collection(collection)
    except Exception:
        pass
    client.close()
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    ledger.close()


def _register(stack, memory_id: str, text: str, revision: int = 1) -> None:
    from ledger.port import ObjectRef

    stack.rows.put(memory_id, text)
    stack.ledger.put_object_ref(ObjectRef(TENANT, memory_id, revision, 1))


def test_flag_on_cycle_through_the_admin_routes_on_real_stores(stack):
    from fastapi.testclient import TestClient

    client = TestClient(stack.main.app)
    switch, schema = stack.switch, stack.schema

    status = client.get(f"{BASE}/status")
    assert status.status_code == 200, status.text
    assert (status.json()["generation"], status.json()["phase"], status.json()["active"]) == (1, "steady", "qdrant")
    assert status.json()["recall_ready"] is False  # nothing projected yet: routed recall is refused with a reason
    assert status.json()["recall_blocked_reason"] == "projection_not_synced"

    # A memory that predates routing: the backfill (catch-up with no window open) brings generation 1 to canonical state.
    _register(stack, "m-old", "an older memory about wind turbines")
    backfill = client.post(f"{BASE}/catch-up")
    assert backfill.status_code == 200, backfill.text
    assert backfill.json()["backfilled"] is True and backfill.json()["report"]["copied"] == 1
    assert backfill.json()["recall_ready"] is True
    assert _qdrant_has(stack, "m-old")

    started = client.post(f"{BASE}/start-dual-write")
    assert started.status_code == 200, started.text
    assert started.json()["phase"] == "dual_write"

    # A real write during the window reaches BOTH real engines.
    _register(stack, "m-new", "a newer memory about harbour cranes")
    switch.project_apply(TENANT, "m-new", text="a newer memory about harbour cranes", user_id=ADMIN)
    assert _qdrant_has(stack, "m-new")
    assert _pg_count(schema, "m-new") == 1

    # An erased record is fenced on both engines and never resurrected by a late projection.
    _register(stack, "m-gone", "a memory that will be erased")
    switch.project_apply(TENANT, "m-gone", text="a memory that will be erased", user_id=ADMIN)
    assert _pg_count(schema, "m-gone") == 1
    stack.ledger.tombstone(TENANT, "m-gone", 1, "test_erasure")
    switch.project_remove(TENANT, "m-gone")
    switch.project_apply(TENANT, "m-gone", text="a memory that will be erased", user_id=ADMIN, revision=1)
    assert _pg_count(schema, "m-gone") == 0 and not _qdrant_has(stack, "m-gone")

    # Catch-up copies the pre-window memory to the shadow; then flip.
    assert _pg_count(schema, "m-old") == 0
    caught = client.post(f"{BASE}/catch-up")
    assert caught.status_code == 200, caught.text
    assert caught.json()["report"]["copied"] >= 1 and caught.json()["report"]["conflicts"] == 0
    assert caught.json()["lag"] == 0 and _pg_count(schema, "m-old") == 1

    flipped = client.post(f"{BASE}/flip")
    assert flipped.status_code == 200, flipped.text
    body = flipped.json()
    assert (body["generation"], body["phase"], body["active"], body["previous"]) == (2, "steady", "pgvector", "qdrant")
    assert body["recall_ready"] is True

    # A write after the flip lands on pgvector only (the old engine is kept for rollback); the read is served by it.
    _register(stack, "m-after", "a memory written after the flip about glacier surveys")
    switch.project_apply(TENANT, "m-after", text="a memory written after the flip about glacier surveys", user_id=ADMIN)
    assert _pg_count(schema, "m-after") == 1 and not _qdrant_has(stack, "m-after")
    hits = switch.recall(ADMIN, "glacier surveys", 5)
    assert hits and hits[0].record_id == "m-after"
    assert {h.record_id for h in switch.recall(ADMIN, "wind turbines harbour cranes", 10)} >= {"m-old", "m-new"}

    # Rollback replays what was accepted since the flip onto the old engine; the generation never goes backwards.
    rolled = client.post(f"{BASE}/rollback")
    assert rolled.status_code == 200, rolled.text
    assert (rolled.json()["generation"], rolled.json()["active"]) == (3, "qdrant")
    assert _qdrant_has(stack, "m-after")
    assert [h.record_id for h in switch.recall(ADMIN, "glacier surveys", 5)][:1] == ["m-after"]

    # A second window opens and is abandoned; nothing moves.
    assert client.post(f"{BASE}/start-dual-write").json()["phase"] == "dual_write"
    abandoned = client.post(f"{BASE}/abandon")
    assert abandoned.status_code == 200, abandoned.text
    assert (abandoned.json()["phase"], abandoned.json()["generation"], abandoned.json()["active"]) == (
        "steady", 3, "qdrant",
    )


def test_flag_off_contrast_and_dictator_gate_on_the_same_stack(stack, monkeypatch):
    from fastapi.testclient import TestClient

    from config import settings

    client = TestClient(stack.main.app)
    assert client.get(f"{BASE}/status").status_code == 200
    monkeypatch.setattr(settings, "dictator_user_ids", "someone-else")
    assert client.get(f"{BASE}/status").status_code == 403  # only a dictator may drive the switch
    monkeypatch.setattr(settings, "dictator_user_ids", ADMIN)
    monkeypatch.setattr(settings, "engine_routing_enabled", False)
    assert client.get(f"{BASE}/status").status_code == 404  # the flag off hides every route
    assert client.post(f"{BASE}/flip").status_code == 404
