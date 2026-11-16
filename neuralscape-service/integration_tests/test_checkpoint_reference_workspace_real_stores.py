"""A checkpoint item bound for a reference workspace is stored there, via the batch path.

End to end through the product's checkpoint chain: ``prepare_checkpoint`` (the dedup
pre-check ``dedup_verdicts``) -> ``TaskManager.enqueue_raw_batch`` -> a real ``arq`` worker
running ``process_memory_raw_batch`` -> ``store_raw_batch`` with ``graph_jobs``. Real Redis,
real Qdrant (throwaway collection) and the real embedder.

The same text is first seeded in the default workspace and in a different reference
workspace. The checkpoint item (same text, workspace ``ref-nsit-books``) must NOT be treated
as a duplicate of either: dedup is keyed on workspace. It must be stored in its own
workspace, be recallable only from that workspace, and (reference rows being vector-only)
raise no graph job, while a memory-workspace item in the same checkpoint still defers its
graph enrichment to the graph queue rather than running it inline.

Needs Neo4j (the service initialises its graph store), Qdrant, Redis and GOOGLE_API_KEY.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio

import pytest

pytestmark = pytest.mark.integration

REF = "ref-nsit-books"
OTHER_REF = "ref-nsit-other"


def test_checkpoint_item_in_a_reference_workspace_is_stored_there_not_dropped_as_a_duplicate(
    flags, stores, llm_key, unique_ns
):
    from arq.jobs import Job
    from arq.worker import Worker
    from qdrant_client import QdrantClient

    import worker
    from checkpoint import prepare_checkpoint
    from config import parse_redis_settings
    from memory_service import MemoryService
    from schemas import CheckpointRequest
    from task_manager import TaskManager

    queue = f"nsit-ckpt-ws-{unique_ns.user_id[-10:]}"
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
    ref_text = f"Reference checkpoint passage on estuary tide tables ({tag})"
    mem_text = f"Plain checkpoint fact on alpine pilot huts ({tag})"

    # Preconditions: identical text already lives in the default workspace and in another
    # reference workspace. Neither is in REF, so neither may shadow the checkpoint item.
    seed_default = service.store_raw(
        content=ref_text, user_id=uid, category="preference", add_to_graph=False
    )[0].id
    seed_other = service.store_raw(
        content=ref_text, user_id=uid, category="preference", workspace=OTHER_REF, add_to_graph=False
    )[0].id
    assert seed_default != seed_other

    req = CheckpointRequest(
        memories=[
            {"content": ref_text, "category": "preference", "workspace": REF},
            {"content": mem_text, "category": "preference"},
        ]
    )
    prepared = prepare_checkpoint(service, req, uid)
    assert [v["verdict"] for v in prepared["verdicts"]] == ["new", "new"], prepared["verdicts"]
    assert len(prepared["to_enqueue"]) == 2

    async def scenario():
        tm = TaskManager()
        await tm.connect()
        task_id = await tm.enqueue_raw_batch(prepared["to_enqueue"])
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
    stored_ids = {m["id"] for m in status["result"]["memories"]}
    assert len(stored_ids) == 2, status["result"]

    # 1 + 2: stored in the reference workspace, beside (not instead of) the two seed rows.
    rows = {r["id"]: r["payload"] for r in service._scroll_all_user_memories(uid)}
    ws_of = {i: ((p.get("metadata") or {}).get("workspace") or "memory") for i, p in rows.items()}
    assert len(rows) == 4, ws_of
    ref_rows = [i for i, w in ws_of.items() if w == REF]
    assert len(ref_rows) == 1 and ref_rows[0] in stored_ids, ws_of
    assert ref_rows[0] not in (seed_default, seed_other)
    assert rows[ref_rows[0]]["data"] == ref_text
    assert ws_of[seed_default] == "memory" and ws_of[seed_other] == OTHER_REF  # seeds untouched

    # 3: recall is workspace-scoped.
    in_ref = {m.id for m in service.search(query=ref_text, user_id=uid, workspaces=[REF], vector_only=True)}
    in_other = {m.id for m in service.search(query=ref_text, user_id=uid, workspaces=[OTHER_REF], vector_only=True)}
    in_default = {m.id for m in service.search(query=ref_text, user_id=uid, vector_only=True)}
    assert ref_rows[0] in in_ref
    assert ref_rows[0] not in in_other and ref_rows[0] not in in_default

    # 4: graph work is deferred to the graph queue. The reference row is vector-only (no job);
    # the memory-workspace item gets exactly one job, none ran inline.
    assert status["result"]["graph_jobs_enqueued"] == 1
    assert len(graph_jobs) == 1 and graph_jobs[0].function == "process_graph_enrichment"
    assert ref_rows[0] not in str(graph_jobs[0].args) + str(graph_jobs[0].kwargs)
