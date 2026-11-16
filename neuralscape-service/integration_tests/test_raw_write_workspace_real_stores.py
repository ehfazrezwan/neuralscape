"""A raw write with ``workspace`` lands in that workspace, through the real queue and worker.

Real Redis (the ARQ queue and result cache), real Qdrant (a throwaway collection) and the
real embedder. The service's own ``TaskManager.enqueue_raw`` enqueues the write, a real
``arq`` worker runs the service's own ``process_memory_raw`` job, and the stored row's
workspace metadata is read back from Qdrant. Before the fix ``enqueue_raw`` did not accept
``workspace`` at all (the REST route answered 500).

Needs Neo4j (the service initialises its graph store), Qdrant, Redis and GOOGLE_API_KEY.
Graph enrichment jobs are routed to a throwaway queue nobody consumes.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio

import pytest

pytestmark = pytest.mark.integration


def test_workspace_on_a_raw_write_reaches_the_stored_row(flags, stores, llm_key, unique_ns):
    from arq.worker import Worker
    from qdrant_client import QdrantClient

    import worker
    from config import parse_redis_settings
    from memory_service import MemoryService
    from task_manager import TaskManager

    queue = f"nsit-raw-ws-{unique_ns.user_id[-10:]}"
    graph_queue = f"{queue}-graph"
    flags(
        qdrant_collection=unique_ns.qdrant_collection,
        qdrant_url=stores.qdrant_url,
        redis_url=stores.redis_url,
        arq_queue_name=queue,
        graph_queue_name=graph_queue,
    )
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    service = MemoryService()
    service._get_memory()
    uid = unique_ns.user_id
    text = f"The harbour pilot hut posts the tide table at dawn ({uid[-6:]})."

    async def scenario():
        tm = TaskManager()
        await tm.connect()
        ids = []
        try:
            ids.append(await tm.enqueue_raw(content=text, user_id=uid, category="preference", workspace="ref-nsit-books"))
            ids.append(await tm.enqueue_raw(content=text, user_id=uid, category="preference"))
            assert ids[0] != ids[1], "the workspace must be part of the job identity"
            w = Worker(
                functions=[worker.process_memory_raw],
                queue_name=queue,
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
            return [await tm.get_status(i) for i in ids]
        finally:
            redis = tm.pool
            leftovers = [f"arq:result:{i}" for i in ids] + [queue, graph_queue]
            for jid in await redis.zrange(graph_queue, 0, -1):
                leftovers.append("arq:job:" + (jid.decode() if isinstance(jid, bytes) else jid))
            await redis.delete(*leftovers)
            await redis.aclose()

    statuses = asyncio.run(scenario())
    assert [s["status"] for s in statuses] == ["completed", "completed"], statuses

    rows = [r["payload"] for r in service._scroll_all_user_memories(uid)]
    mine = [r for r in rows if (r.get("data") or r.get("memory") or "").startswith("The harbour pilot hut")]
    workspaces = sorted((r.get("metadata") or {}).get("workspace") or "memory" for r in mine)
    assert workspaces == ["memory", "ref-nsit-books"], rows
