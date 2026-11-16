"""``LEDGER_ENABLED=true``: the write paths on the durable ledger, on the REAL memory stack.

Real: the FastAPI app (in process, through its ASGI interface), the application's own ``TaskManager`` (it opens the
ledger and the write producer itself), Redis (the ARQ pool and the ledger worker's wake-ups), the application's own
``worker.LedgerWorkerSettings`` run as a real ARQ worker (its start-up builds the real ``MemoryService``, installs
the engine switch and opens the ledger), Qdrant and Neo4j/Graphiti (the memory service's stores) and Gemini (the
extraction model and the embedder).  The ledger is PostgreSQL in a schema of its own, and SQLite in a file of its
own (the two parametrized runs of the same scenarios).  Nothing is mocked.

* **remember**: ``POST /v1/memories/raw`` answers 202 with an intent id, a retry with the same ``Idempotency-Key``
  returns the original intent, the same key with another body is a 409; a worker applies it; the receipt lists the
  stages and the memory; the memory is recalled by ``POST /v1/search``.
* **edit** and **delete** go through the ledger too (202 + receipt).  A delete erases the row from Qdrant and the
  episode from Neo4j and leaves a tombstone in the ledger.
* **cutover**: the privileged ``/v1/admin/ledger-cutover`` routes move writes ``legacy`` -> ``dual_accept`` ->
  ``ledger_only`` -> ``complete`` and refuse a rollback after ``complete``.
* **engine switch**: with ``ENGINE_ROUTING_ENABLED`` the worker's vector stage projects each accepted write into the
  routed engine, and the API's admin routes answer from a real ledger (never 503 ``ledger_not_open``).
* **flag off**: the same call is a legacy queue job (no intent), exactly as on ``dev``.

A missing store or key is a failure under ``NS_INTEGRATION_REQUIRED`` and a skip elsewhere.  Needs Redis, Qdrant,
Neo4j, PostgreSQL (``NS_TEST_PG_DSN``) and ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import copy
import uuid

import httpx
import pytest
import pytest_asyncio

psycopg = pytest.importorskip("psycopg")  # a declared condition: the optional PostgreSQL driver
from psycopg.conninfo import make_conninfo  # noqa: E402

BACKENDS = ["sqlite", "postgres"]
FACT = "{token}: the user keeps a pet axolotl called Quillon in a tank in the study."
QUESTION = "What pet does the user keep in the study?"


@pytest.fixture(params=BACKENDS)
def ledger_flags(request, flags, stores, unique_ns, tmp_path, llm_key):
    """The flag on over a ledger of its own, a private ARQ queue, an isolated Qdrant collection, auth off."""

    from config import settings

    flags(
        ledger_enabled=True,
        redis_url=stores.redis_url,
        arq_queue_name=f"nsit-{uuid.uuid4().hex[:10]}",
        qdrant_collection=unique_ns.qdrant_collection,
        neuralscape_auth_disabled=True,
        neuralscape_user_token_secret="",
        neuralscape_api_key="",
        default_user_id=unique_ns.user_id,
        dictator_user_ids=unique_ns.user_id,
    )
    if request.param == "sqlite":
        flags(ledger_backend="sqlite", ledger_sqlite_path=str(tmp_path / "ledger.db"))
    else:
        pg_dsn = request.getfixturevalue("pg_dsn")
        schema = f"ledger_nsit_{uuid.uuid4().hex[:12]}"
        with psycopg.connect(pg_dsn, autocommit=True) as admin:
            admin.execute(f'CREATE SCHEMA "{schema}"')

        def drop() -> None:
            with psycopg.connect(pg_dsn, autocommit=True) as admin:
                admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')

        unique_ns.on_cleanup(drop)
        flags(ledger_backend="postgres", ledger_postgres_dsn=make_conninfo(pg_dsn, options=f"-c search_path={schema}"))
    return settings


class Stack:
    """The app, its task manager and the real service, built the way the processes build them."""

    def __init__(self, service, manager, stores, unique_ns) -> None:
        import main

        self.service, self.manager, self.stores, self.ns = service, manager, stores, unique_ns
        self.user = unique_ns.user_id
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test")
        self.tasks: list[str] = []

    async def post(self, path, **kw):
        return await self.client.post(path, **kw)

    async def run_worker(self) -> None:
        """One burst of the application's own ledger worker (start-up, wake-ups, poll cron, shutdown)."""

        import worker as worker_module
        from arq.worker import create_worker
        from config import parse_redis_settings
        from write_producer import ledger_queue_name

        runner = create_worker(
            worker_module.LedgerWorkerSettings,
            redis_settings=parse_redis_settings(),
            queue_name=ledger_queue_name(),
            burst=True,
            poll_delay=0.2,
            handle_signals=False,
            cron_jobs=[copy.copy(job) for job in worker_module.LedgerWorkerSettings.cron_jobs],
        )
        try:
            await runner.async_run()
        finally:
            await runner.close()

    async def settle(self, task_id: str, timeout: float = 240.0) -> dict:
        """Run the worker until the task's receipt is terminal; returns the polled status."""

        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            await self.run_worker()
            status = (await self.client.get(f"/v1/memories/status/{task_id}")).json()
            if status["status"] in ("completed", "failed"):
                return status
            assert asyncio.get_running_loop().time() < deadline, f"task {task_id} did not settle: {status}"

    async def close(self) -> None:
        await self.client.aclose()
        await self.manager.close()


@pytest_asyncio.fixture()
async def stack(ledger_flags, stores, unique_ns, monkeypatch):
    import main
    from memory.groups import _build_group_id
    from memory_service import MemoryService
    from qdrant_client import QdrantClient
    from task_manager import TaskManager

    service = MemoryService()
    service._get_memory()  # real mem0 + Graphiti; fails loudly if a store is unreachable
    manager = TaskManager()
    await manager.connect()  # opens the ledger, the payload store and the write producer
    assert manager.producer is not None and manager.ledger is not None
    monkeypatch.setattr(main, "_service", service)
    monkeypatch.setattr(main, "_task_manager", manager)
    group = _build_group_id("private", unique_ns.user_id, None, None)
    built = Stack(service, manager, stores, unique_ns)

    def cleanup() -> None:
        async def purge():
            async with service._graphiti.driver.session() as session:
                await session.run("MATCH (n {group_id:$g}) DETACH DELETE n", g=group)

        try:
            service._run_on_bridge(purge(), timeout=30.0)
        except Exception:  # noqa: BLE001 - best effort; the nsit prefix finds leftovers
            pass
        client = QdrantClient(url=stores.qdrant_url)
        client.delete_collection(unique_ns.qdrant_collection)
        client.delete_collection(f"{unique_ns.qdrant_collection}_engine")
        import redis

        redis_client = redis.Redis.from_url(stores.redis_url)
        try:
            for key in redis_client.scan_iter(match="arq:*ledger-wake:*"):
                redis_client.delete(key)
            for task_id in built.tasks:
                for key in redis_client.scan_iter(match=f"*{task_id}*"):
                    redis_client.delete(key)
        finally:
            redis_client.close()

    unique_ns.on_cleanup(cleanup)
    try:
        yield built
    finally:
        await built.close()


async def _neo4j_episodes(stack, needle: str) -> int:
    from memory.groups import _build_group_id

    group = _build_group_id("private", stack.user, None, None)

    async def count():
        async with stack.service._graphiti.driver.session() as session:
            result = await session.run(
                "MATCH (e:Episodic) WHERE e.group_id = $g AND e.content CONTAINS $needle RETURN count(e) AS n",
                g=group, needle=needle,
            )
            return (await result.single())["n"]

    return await asyncio.to_thread(stack.service._run_on_bridge, count(), 60.0)


def _qdrant_has(stack, memory_id: str) -> bool:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=stack.stores.qdrant_url)
    return bool(client.retrieve(stack.ns.qdrant_collection, ids=[memory_id]))


async def _remember(stack, token: str, key: str):
    body = {"content": FACT.format(token=token), "category": "preference", "user_id": stack.user}
    return await stack.post("/v1/memories/raw", json=body, headers={"Idempotency-Key": key}), body


@pytest.mark.asyncio
async def test_remember_edit_and_delete_go_through_the_ledger(stack):
    token = f"ledgerproof-{uuid.uuid4().hex[:8]}"

    # remember: 202 + an intent id; a retry with the same key is the original intent; another body is a 409
    accepted, body = await _remember(stack, token, f"remember-{token}")
    assert accepted.status_code == 202, accepted.text
    task_id = accepted.json()["task_id"]
    stack.tasks.append(task_id)
    assert task_id.startswith("intent-") and accepted.json()["poll_url"].endswith(task_id)
    retry = await stack.post("/v1/memories/raw", json=body, headers={"Idempotency-Key": f"remember-{token}"})
    assert retry.status_code == 202 and retry.json()["task_id"] == task_id
    conflict = await stack.post(
        "/v1/memories/raw", json={**body, "content": body["content"] + " Different."},
        headers={"Idempotency-Key": f"remember-{token}"},
    )
    assert conflict.status_code == 409

    # before a worker runs, the receipt says so (a real ledger read, never a queue result)
    waiting = (await stack.client.get(f"/v1/memories/status/{task_id}")).json()
    assert waiting["status"] in ("queued", "processing"), waiting

    # a worker applies it: receipts for both stages and the memory it created
    done = await stack.settle(task_id)
    assert done["status"] == "completed", done
    receipt = done["result"]["receipt"]
    assert receipt["status"] == "applied" and {s["stage"] for s in receipt["stages"]} >= {"vector", "graph"}
    assert all(s["state"] in ("applied", "skipped") for s in receipt["stages"]), receipt["stages"]
    memory_id = receipt["outputs"][0]["id"]
    again = await stack.post("/v1/memories/raw", json=body, headers={"Idempotency-Key": f"remember-{token}"})
    assert again.json()["task_id"] == task_id  # still the original intent after it ran

    # recallable through search (real embeddings), and present in Qdrant and in the graph
    searched = await stack.post("/v1/search", json={"query": QUESTION, "user_id": stack.user, "limit": 5})
    assert searched.status_code == 200, searched.text
    assert memory_id in [hit["id"] for hit in searched.json()["results"]]
    assert _qdrant_has(stack, memory_id)
    assert await _neo4j_episodes(stack, token) >= 1

    # edit through the ledger: If-Match names the registered revision; the new text is the stored row
    new_text = FACT.format(token=token) + " The axolotl is blue."
    edit = await stack.client.patch(
        f"/v1/memories/{memory_id}", json={"content": new_text, "user_id": stack.user}, headers={"If-Match": "1"},
    )
    assert edit.status_code == 202, edit.text
    assert edit.json()["revision"] == 2
    stack.tasks.append(edit.json()["task_id"])
    stale = await stack.client.patch(
        f"/v1/memories/{memory_id}", json={"content": "stale", "user_id": stack.user}, headers={"If-Match": "1"},
    )
    assert stale.status_code == 409 and stale.json()["detail"]["error"]["code"] == "stale_revision"
    edited = await stack.settle(edit.json()["task_id"])
    assert edited["status"] == "completed", edited
    row = (await stack.client.get(f"/v1/memories/{memory_id}", params={"user_id": stack.user})).json()
    assert "The axolotl is blue." in (row.get("memory") or row.get("memory", ""))

    # delete through the ledger: 202, then the row, the episode and the vector are gone and a tombstone remains
    deleted = await stack.client.request("DELETE", f"/v1/memories/{memory_id}", params={"user_id": stack.user})
    assert deleted.status_code == 202, deleted.text
    stack.tasks.append(deleted.json()["task_id"])
    erased = await stack.settle(deleted.json()["task_id"])
    assert erased["status"] == "completed", erased
    assert not _qdrant_has(stack, memory_id)
    assert await _neo4j_episodes(stack, token) == 0
    assert (await stack.client.get(f"/v1/memories/{memory_id}", params={"user_id": stack.user})).status_code == 404
    tenant = stack.manager.producer.tenant_for(stack.user)
    assert stack.manager.ledger.get_tombstone(tenant, memory_id) is not None
    after = await stack.post("/v1/search", json={"query": QUESTION, "user_id": stack.user, "limit": 5})
    assert memory_id not in [hit["id"] for hit in after.json()["results"]]


@pytest.mark.asyncio
async def test_the_cutover_routes_move_writes_through_every_phase(stack, flags):
    import main
    from task_manager import TaskManager

    # The cutover controller is opened by the task manager only when the feature is on: reconnect with it on.
    flags(ledger_cutover_enabled=True)
    await stack.manager.close()
    manager = TaskManager()
    await manager.connect()
    stack.manager = manager
    main._task_manager = manager
    assert manager.cutover is not None

    status = (await stack.client.get("/v1/admin/ledger-cutover")).json()
    assert status["phase"] == "legacy"
    # in `legacy` a write is a legacy queue job, not an intent
    legacy = await _remember(stack, f"phase-legacy-{uuid.uuid4().hex[:6]}", "k-legacy")
    assert legacy[0].status_code == 202 and not legacy[0].json()["task_id"].startswith("intent-")
    stack.tasks.append(legacy[0].json()["task_id"])

    started = await stack.client.post("/v1/admin/ledger-cutover/start")
    assert started.status_code == 200 and started.json()["phase"] == "dual_accept", started.text
    during = await _remember(stack, f"phase-dual-{uuid.uuid4().hex[:6]}", "k-dual")
    assert during[0].json()["task_id"].startswith("intent-")  # new writes are accepted by the ledger now
    stack.tasks.append(during[0].json()["task_id"])
    assert (await stack.settle(during[0].json()["task_id"]))["status"] == "completed"

    backfill = await stack.client.post("/v1/admin/ledger-cutover/backfill")
    assert backfill.status_code == 200, backfill.text
    # The legacy job queued before the start is what the fence waits for.  There is no legacy worker in this
    # test, so take it out of the queue as a worker finishing it would, then close the fence.
    import redis

    from config import settings

    client = redis.Redis.from_url(stack.stores.redis_url)
    try:
        legacy_id = legacy[0].json()["task_id"]
        assert client.zscore(f"arq:queue:{settings.arq_queue_name}", legacy_id) is not None  # it was queued
        blocked = await stack.client.post("/v1/admin/ledger-cutover/advance")
        assert blocked.status_code == 409, blocked.text  # in-flight legacy work: the fence stays open
        client.zrem(f"arq:queue:{settings.arq_queue_name}", legacy_id)
        client.delete(f"arq:job:{legacy_id}")
    finally:
        client.close()
    advanced = await stack.client.post("/v1/admin/ledger-cutover/advance")
    assert advanced.status_code == 200 and advanced.json()["phase"] == "ledger_only", advanced.text
    completed = await stack.client.post("/v1/admin/ledger-cutover/complete")
    assert completed.status_code == 200 and completed.json()["phase"] == "complete", completed.text
    rollback = await stack.client.post("/v1/admin/ledger-cutover/rollback")
    assert rollback.status_code == 409  # refused after complete


@pytest.mark.asyncio
async def test_the_engine_switch_routes_answer_from_a_real_ledger_and_the_worker_projects(stack, flags, stores):
    import main
    import engine_adapters_switch as sw
    from config import VECTOR_EMBEDDING_DIMS
    from qdrant_client import QdrantClient

    engine_collection = f"{stack.ns.qdrant_collection}_engine"
    flags(
        engine_routing_enabled=True,
        engine_routing_dimensions=VECTOR_EMBEDDING_DIMS,
        engine_routing_qdrant_collection=engine_collection,
        qdrant_url=stores.qdrant_url,
    )
    token = f"engineproof-{uuid.uuid4().hex[:8]}"
    accepted, _ = await _remember(stack, token, f"engine-{token}")
    task_id = accepted.json()["task_id"]
    stack.tasks.append(task_id)
    done = await stack.settle(task_id)  # the worker starts its own switch and its vector stage projects
    assert done["status"] == "completed", done
    client = QdrantClient(url=stores.qdrant_url)
    assert client.count(engine_collection).count == 1  # the projection exists in the routed engine

    # the API process builds its switch from the real ledger the task manager opened: no 503
    await main._start_engine_switch()
    try:
        assert sw.get_switch() is not None and sw.unavailable_reason() is None
        status = await stack.client.get("/v1/admin/engine-switch/status")
        assert status.status_code == 200, status.text
        assert status.json()["generation"] == 1
        caught_up = await stack.client.post("/v1/admin/engine-switch/catch-up")
        assert caught_up.status_code == 200, caught_up.text
    finally:
        await main._stop_engine_switch()


@pytest.mark.asyncio
async def test_with_the_flag_off_the_same_write_is_a_legacy_queue_job(stack, flags):
    import main
    from task_manager import TaskManager

    flags(ledger_enabled=False)
    await stack.manager.close()
    manager = TaskManager()
    await manager.connect()
    main._task_manager = manager
    stack.manager = manager
    assert manager.producer is None and manager.ledger is None
    response, _ = await _remember(stack, f"off-{uuid.uuid4().hex[:6]}", "k-off")
    assert response.status_code == 202, response.text
    task_id = response.json()["task_id"]
    stack.tasks.append(task_id)
    assert not task_id.startswith("intent-")  # a queue job id, no intent, no receipt
