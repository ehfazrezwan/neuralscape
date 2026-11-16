"""A legacy job the cutover fence skipped reaches a waiting or polling client as ``fenced`` (real ARQ + Redis).

Needs a disposable Redis (``ledger_env``); a failure under ``NS_INTEGRATION_REQUIRED``, a skip elsewhere.
"""

from __future__ import annotations

import asyncio

import pytest

from ledger.cutover import FENCED_RESULT
from ledger_env import REDIS_URL, require_env


@pytest.fixture(autouse=True)
def _fail_when_required_store_is_missing():
    require_env(**{"Redis (V1_TEST_REDIS_URL or REDIS_URL)": REDIS_URL})


async def fenced_job(ctx):  # module level: ARQ names a function by its qualified name
    return dict(FENCED_RESULT)


async def normal_job(ctx):
    return {"stored": 1}


def test_wait_for_result_reports_a_fenced_job_as_fenced_on_real_arq(monkeypatch) -> None:
    import uuid

    from arq import create_pool
    from arq.worker import create_worker

    import task_manager
    from config import parse_redis_settings, settings

    queue = f"fenced-test-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(settings, "redis_url", REDIS_URL)
    monkeypatch.setattr(settings, "arq_queue_name", queue)

    async def go():
        pool = await create_pool(parse_redis_settings(), default_queue_name=queue)
        manager = task_manager.TaskManager()
        manager.pool = pool
        fenced = await pool.enqueue_job("fenced_job", _queue_name=queue)
        normal = await pool.enqueue_job("normal_job", _queue_name=queue)
        worker = create_worker(
            type("S", (), {"functions": [fenced_job, normal_job], "redis_settings": parse_redis_settings(),
                           "queue_name": queue, "burst": True, "poll_delay": 0.05}),
        )
        await worker.async_run()
        await worker.close()
        waited = await manager.wait_for_result(fenced.job_id, timeout=10)
        polled = await manager.get_status(fenced.job_id)
        ordinary = await manager.wait_for_result(normal.job_id, timeout=10)
        await pool.aclose()
        return waited, polled, ordinary

    waited, polled, ordinary = asyncio.run(go())
    assert polled["status"] == "fenced", polled
    assert waited["status"] == "fenced" and waited["error"] == "legacy_write_after_cutover"  # used to be "completed"
    assert ordinary["status"] == "completed" and ordinary["result"] == {"stored": 1}
