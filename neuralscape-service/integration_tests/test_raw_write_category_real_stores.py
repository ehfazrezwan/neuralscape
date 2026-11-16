"""The same text under another category is a distinct memory, through the real queue and worker.

Real Redis (the ARQ queue), real Qdrant (a throwaway collection) and the real embedder.
The service's own ``TaskManager.enqueue_raw`` enqueues three raw writes (text under
"tech_stack", the same text again, the same text under "convention"), a real ``arq``
worker runs the service's own ``process_memory_raw`` job, and the stored rows are read
back from Qdrant. Before the fix the worker's near-duplicate probe compared text and
visibility only, so the third write returned the first row's id and stored nothing.

Needs Neo4j (the service initialises its graph store), Qdrant, Redis and GOOGLE_API_KEY.
Graph enrichment jobs are routed to a throwaway queue nobody consumes.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio

import pytest

pytestmark = pytest.mark.integration


def test_recategorised_text_is_stored_beside_the_original(flags, stores, llm_key, unique_ns):
    from arq.worker import Worker
    from qdrant_client import QdrantClient

    import worker
    from config import parse_redis_settings
    from memory_service import MemoryService
    from task_manager import TaskManager

    queue = f"nsit-raw-cat-{unique_ns.user_id[-10:]}"
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
    text = f"The lighthouse keeper logs the tide table every dawn ({uid[-6:]})."
    kw = dict(content=text, user_id=uid, scope="project", project_id="p-nsit", visibility="private")

    async def run_one(tm, category):
        task_id = await tm.enqueue_raw(category=category, **kw)
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
        return task_id, await tm.get_status(task_id)

    async def scenario():
        tm = TaskManager()
        await tm.connect()
        task_ids = []
        try:
            out = []
            for category in ("tech_stack", "tech_stack", "convention"):
                task_id, status = await run_one(tm, category)
                task_ids.append(task_id)
                assert status["status"] == "completed", status
                out.append(status["result"]["memories"][0]["id"])
            return out
        finally:
            redis = tm.pool
            leftovers = [f"arq:result:{i}" for i in task_ids] + [queue, graph_queue]
            for jid in await redis.zrange(graph_queue, 0, -1):
                leftovers.append("arq:job:" + (jid.decode() if isinstance(jid, bytes) else jid))
            await redis.delete(*leftovers)
            await redis.aclose()

    first, again, other = asyncio.run(scenario())
    assert again == first, "an identical re-write must still collapse onto the first row"
    assert other != first, "re-categorised duplicate was silently absorbed into the old row"

    rows = [r["payload"] for r in service._scroll_all_user_memories(uid)]
    categories = sorted((r.get("metadata") or {}).get("category") for r in rows if "lighthouse keeper" in (r.get("data") or ""))
    assert categories == ["convention", "tech_stack"], rows
