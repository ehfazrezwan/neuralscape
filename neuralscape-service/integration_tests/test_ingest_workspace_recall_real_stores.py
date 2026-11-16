"""Ingest with ``workspace`` through the real API + queue + worker, then recall by workspace.

``POST /v1/ingest/text`` (the real FastAPI route) -> the service's own ``TaskManager`` ->
a real ``arq`` worker running ``process_ingest_document`` over real Qdrant and the real
embedder and LLM (distilled facts) -> ``POST /v1/search``: with ``workspaces=["ref-..."]``
the passages and facts are found, with the default (memory workspace) they are not.

Needs Neo4j (the service initialises its graph store), Qdrant, Redis and GOOGLE_API_KEY.
Graph enrichment jobs go to a throwaway queue nobody consumes.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio

import pytest

pytestmark = pytest.mark.integration

SECRET = "nsit-" + "s" * 32


def test_ingest_text_workspace_is_recalled_only_when_that_workspace_is_asked_for(flags, stores, llm_key, unique_ns, monkeypatch):
    from arq.worker import Worker
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    import main
    import worker
    from config import parse_redis_settings
    from memory_service import MemoryService
    from task_manager import TaskManager
    from tokens import issue_user_token

    queue = f"nsit-ingest-ws-{unique_ns.user_id[-10:]}"
    graph_queue = f"{queue}-graph"
    ingest_queue = f"{queue}-ingest"
    flags(
        qdrant_collection=unique_ns.qdrant_collection,
        qdrant_url=stores.qdrant_url,
        redis_url=stores.redis_url,
        arq_queue_name=queue,
        graph_queue_name=graph_queue,
        ingest_queue_name=ingest_queue,
        ingest_storage_enabled=False,
        neuralscape_user_token_secret=SECRET,
        neuralscape_api_key="",
    )
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    service = MemoryService()
    service._get_memory()
    uid = unique_ns.user_id
    tag = uid[-6:]
    text = (
        f"# Harbour notes ({tag})\n\nThe harbour pilot hut posts the tide table every morning at dawn. "
        f"The north quay closes to traffic during spring tides. Reference code {tag}."
    )
    ws = "ref-nsit-harbour"
    headers = {"Authorization": f"Bearer {issue_user_token(uid, SECRET, 3600)}"}

    class PerCallPool:
        """TestClient serves each request on its own anyio portal loop, and an arq/redis pool is
        bound to the loop that created it. So the app under test gets a pool that opens a
        short-lived connection on whichever loop calls it (same wiring as
        test_extension_route_identity_redis.py); the worker and status poll below use a real
        TaskManager made on their own loop."""

        def pipeline(self, *args, **kwargs):
            import contextlib

            from arq.connections import create_pool

            @contextlib.asynccontextmanager
            async def scope():
                pool = await create_pool(parse_redis_settings(), default_queue_name=queue)
                try:
                    async with pool.pipeline(*args, **kwargs) as pipe:
                        yield pipe
                finally:
                    await pool.aclose()

            return scope()

        def __getattr__(self, name):
            async def call(*args, **kwargs):
                from arq.connections import create_pool

                pool = await create_pool(parse_redis_settings(), default_queue_name=queue)
                try:
                    return await getattr(pool, name)(*args, **kwargs)
                finally:
                    await pool.aclose()

            return call

    app_tm = TaskManager()
    app_tm.pool = PerCallPool()
    monkeypatch.setattr(main, "_task_manager", app_tm)
    monkeypatch.setattr(main, "_service", service)
    client = TestClient(main.app)
    resp = client.post("/v1/ingest/text", json={"content": text, "workspace": ws}, headers=headers)
    assert resp.status_code == 202, resp.text
    task_id = resp.json()["task_id"]

    async def drain():
        tm = TaskManager()
        await tm.connect()
        try:
            w = Worker(
                functions=[worker.process_ingest_document],
                queue_name=ingest_queue,
                redis_settings=parse_redis_settings(),
                ctx={"service": service},
                burst=True,
                poll_delay=0.1,
                handle_signals=False,
            )
            try:
                await w.main()
            finally:
                await w.close()
            return await tm.get_status(task_id)
        finally:
            leftovers = [f"arq:result:{task_id}", queue, ingest_queue, graph_queue]
            for jid in await tm.pool.zrange(graph_queue, 0, -1):
                leftovers.append("arq:job:" + (jid.decode() if isinstance(jid, bytes) else jid))
            await tm.pool.delete(*leftovers)
            await tm.pool.aclose()

    status = asyncio.run(drain())
    assert status["status"] == "completed", status
    assert status["result"]["passages"] >= 1

    def search(**extra):
        resp = client.post("/v1/search", json={"query": f"tide table pilot hut {tag}", "limit": 20, **extra}, headers=headers)
        assert resp.status_code == 200, resp.text
        return resp.json()["results"]

    in_ws = search(workspaces=[ws])
    assert in_ws and all((m.get("workspace") == ws) for m in in_ws), in_ws
    assert {m.get("memory_kind") for m in in_ws} >= {"passage"}
    assert not [m for m in search() if tag in m["memory"]], "the memory workspace must not see reference-workspace rows"
