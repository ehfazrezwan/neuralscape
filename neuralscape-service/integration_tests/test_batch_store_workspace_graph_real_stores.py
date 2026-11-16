"""A batch store keeps each item's workspace and defers graph work to the graph queue.

Real Redis (ARQ queues), real Qdrant (throwaway collection) and the real embedder. The
service's own ``TaskManager.enqueue_raw_batch`` enqueues the batch, a real ``arq`` worker
runs ``process_memory_raw_batch``, and we read back (a) the stored rows' workspace from
Qdrant and (b) the ``process_graph_enrichment`` jobs that landed on the graph queue (a
throwaway one nobody consumes). Before the fix the rows landed in the memory workspace and
the batch job ran Graphiti inline on the fast worker.

Needs Neo4j (the service initialises its graph store), Qdrant, Redis and GOOGLE_API_KEY.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio

import pytest

pytestmark = pytest.mark.integration


def test_batch_items_keep_their_workspace_and_graph_jobs_go_to_the_graph_queue(flags, stores, llm_key, unique_ns):
    from arq.jobs import Job
    from arq.worker import Worker
    from qdrant_client import QdrantClient

    import worker
    from config import parse_redis_settings
    from memory_service import MemoryService
    from task_manager import TaskManager

    queue = f"nsit-batch-ws-{unique_ns.user_id[-10:]}"
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
    tag = uid[-6:]
    items = [
        {"content": f"Reference batch fact {n} on tide tables ({tag})", "user_id": uid,
         "category": "preference", "workspace": "ref-nsit-books"}
        for n in (1, 2)
    ] + [{"content": f"Plain batch fact on pilot huts ({tag})", "user_id": uid, "category": "preference"}]

    async def scenario():
        tm = TaskManager()
        await tm.connect()
        task_id = await tm.enqueue_raw_batch(items)
        cleanup = [f"arq:result:{task_id}", queue, graph_queue]
        try:
            w = Worker(
                functions=[worker.process_memory_raw_batch],
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
            status = await tm.get_status(task_id)
            graph_ids = [j.decode() if isinstance(j, bytes) else j for j in await tm.pool.zrange(graph_queue, 0, -1)]
            graph_jobs = [await Job(j, tm.pool, _queue_name=graph_queue).info() for j in graph_ids]
            cleanup += [f"arq:job:{j}" for j in graph_ids]
            return status, graph_jobs
        finally:
            await tm.pool.delete(*cleanup)
            await tm.pool.aclose()

    status, graph_jobs = asyncio.run(scenario())
    assert status["status"] == "completed", status
    # Rows in a reference workspace are vector-only: stored, but no graph job. Only the plain
    # (memory-workspace) item gets one.
    assert status["result"]["graph_jobs_enqueued"] == 1

    rows = [r["payload"] for r in service._scroll_all_user_memories(uid)]
    by_ws = sorted((r.get("metadata") or {}).get("workspace") or "memory" for r in rows)
    assert by_ws == ["memory", "ref-nsit-books", "ref-nsit-books"], rows

    assert len(graph_jobs) == 1 and all(j.function == "process_graph_enrichment" for j in graph_jobs)
    # ...and that one job is the plain row's, not a reference row's.
    job_text = " ".join(map(str, graph_jobs[0].args))
    assert "pilot huts" in job_text and "tide tables" not in job_text
