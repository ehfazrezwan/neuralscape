"""queue.empty webhook destination guard, through the REAL ``after_job_end`` hook on real Redis.

A real ``arq.worker.Worker`` (burst mode) runs one job off a private, otherwise empty queue
with the hook the three worker classes install. ``WEBHOOK_QUEUE_EMPTY_URL`` is set:

* to a loopback sink: the event is delivered (private and loopback targets stay allowed);
* to a literal link-local address: nothing is delivered and the refusal is logged.

The second test fails if the guard is removed (the hook would then dispatch to the address).
"""

from __future__ import annotations

import asyncio
import http.server
import json
import logging
import threading
import time
import uuid

import pytest
from arq import create_pool
from arq.worker import Worker

pytestmark = pytest.mark.integration


async def webhook_guard_probe(ctx):
    """Module level: ARQ names a function by its qualified name."""
    return "done"


class _Sink:
    def __init__(self):
        self.bodies: list[dict] = []
        sink = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                sink.bodies.append(json.loads(self.rfile.read(length)))
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _run_one_job(queue: str, job_id: str) -> None:
    import worker
    from config import parse_redis_settings

    async def scenario():
        redis_settings = parse_redis_settings()
        pool = await create_pool(redis_settings, default_queue_name=queue)
        await pool.enqueue_job("webhook_guard_probe", _job_id=job_id, _queue_name=queue)
        w = Worker(
            functions=[webhook_guard_probe],
            queue_name=queue,
            redis_settings=redis_settings,
            burst=True,
            max_jobs=1,
            poll_delay=0.1,
            handle_signals=False,
            after_job_end=worker._make_after_job_end(queue),
        )
        try:
            await w.main()
        finally:
            await w.close()
            await pool.aclose()

    asyncio.run(scenario())


def _wait(predicate, seconds=10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_empty_queue_delivers_to_a_loopback_sink(stores, flags, unique_ns):
    sink = _Sink()
    unique_ns.on_cleanup(sink.close)
    queue = f"nsit-whguard-{uuid.uuid4().hex[:8]}"
    job_id = f"nsit-job-{uuid.uuid4().hex[:8]}"
    flags(redis_url=stores.redis_url, webhook_queue_empty_url=f"http://127.0.0.1:{sink.port}/hook")

    _run_one_job(queue, job_id)

    assert _wait(lambda: sink.bodies), "the queue.empty event never reached the loopback sink"
    assert sink.bodies[0]["event"] == "queue.empty"
    assert sink.bodies[0]["queue"] == queue
    assert sink.bodies[0]["job_id"] == job_id


def test_empty_queue_refuses_a_link_local_target_and_logs_it(stores, flags, caplog):
    queue = f"nsit-whguard-{uuid.uuid4().hex[:8]}"
    flags(redis_url=stores.redis_url, webhook_queue_empty_url="http://169.254.169.254/latest/meta-data?token=zzz")
    started: list[str] = []
    real_start = threading.Thread.start

    def watching_start(self, *args, **kwargs):
        started.append(self.name)
        return real_start(self, *args, **kwargs)

    threading.Thread.start = watching_start
    try:
        with caplog.at_level(logging.WARNING, logger="webhooks"):
            _run_one_job(queue, f"nsit-job-{uuid.uuid4().hex[:8]}")
    finally:
        threading.Thread.start = real_start

    assert "ns-queue-webhook" not in started, "a delivery thread was started for a link-local target"
    refused = [r for r in caplog.records if r.name == "webhooks" and "169.254.169.254" in r.getMessage()]
    assert refused, "the refusal was not logged"
    assert "token=zzz" not in caplog.text
