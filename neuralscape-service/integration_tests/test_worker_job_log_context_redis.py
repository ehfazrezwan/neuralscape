"""Real ARQ worker proof that job context is bound per job and never leaks across jobs.

A real ``arq.worker.Worker`` (burst mode) pulls two jobs off a private queue on
the real Redis and runs them concurrently with the same ``on_job_start`` /
``after_job_end`` hooks the three worker classes install. Each job logs after
both have started, so a leak between concurrently running jobs would show.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import pytest
import structlog
from arq import create_pool
from arq.worker import Worker

pytestmark = pytest.mark.integration


_STARTED: list[str] = []


async def job_log_probe(ctx, label):
    """Module level: ARQ names a function by its qualified name."""
    _STARTED.append(label)
    for _ in range(200):  # wait until both jobs are running at once
        if len(_STARTED) == 2:
            break
        await asyncio.sleep(0.05)
    assert len(_STARTED) == 2, "the two jobs never overlapped"
    logging.getLogger("nsit.job").info("body %s", label)
    await asyncio.sleep(0.1)
    logging.getLogger("nsit.job").info("end %s", label)
    return label


@pytest.fixture()
def restore_logging():
    root = logging.getLogger()
    saved = (list(root.handlers), root.level, structlog.get_config())
    yield
    structlog.contextvars.clear_contextvars()
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])
    structlog.configure(**saved[2])


def test_each_job_logs_only_its_own_job_id_and_function(stores, flags, capsys, restore_logging):
    import worker
    from config import parse_redis_settings
    from logging_config import configure_logging

    flags(redis_url=stores.redis_url)
    queue = f"nsit-joblog-{uuid.uuid4().hex[:8]}"
    ids = {"alpha": f"nsit-job-alpha-{uuid.uuid4().hex[:6]}", "beta": f"nsit-job-beta-{uuid.uuid4().hex[:6]}"}
    _STARTED.clear()

    async def scenario():
        redis_settings = parse_redis_settings()
        pool = await create_pool(redis_settings, default_queue_name=queue)
        for label, job_id in ids.items():
            await pool.enqueue_job("job_log_probe", label, _job_id=job_id, _queue_name=queue)
        w = Worker(
            functions=[job_log_probe],
            queue_name=queue,
            redis_settings=redis_settings,
            burst=True,
            max_jobs=2,
            poll_delay=0.1,
            handle_signals=False,
            on_job_start=worker._make_on_job_start(queue),
            after_job_end=worker._make_after_job_end(queue),
        )
        try:
            await w.main()
        finally:
            await w.close()
            await pool.aclose()

    configure_logging(worker=True)
    asyncio.run(scenario())

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip().startswith("{")]
    for label, job_id in ids.items():
        mine = [r for r in records if r.get("event") in (f"body {label}", f"end {label}")]
        assert len(mine) == 2, records
        for record in mine:
            assert record["job_id"] == job_id
            assert record["function"] == "job_log_probe"
            assert record["queue"] == queue
    # nothing from the jobs leaked into the (test) process context
    assert "job_id" not in structlog.contextvars.get_contextvars()
