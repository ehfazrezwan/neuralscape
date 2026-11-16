"""Flag-on proof: the queue.empty webhook with a real Redis queue and a real ARQ worker.

``WEBHOOK_QUEUE_EMPTY_URL`` points at a local HTTP sink (loopback, ephemeral
port). A job is enqueued on a private queue in real Redis (``stores``) and run by
a real ``arq`` worker whose ``after_job_end`` is the service's own hook
(``worker._make_after_job_end``). When the queue drains the sink must receive
exactly the documented payload ({event, queue, job_id, user_id, timestamp}), with
``user_id`` attributed through the ``ns:task-user:`` map. Also: with the URL unset
nothing is sent; a URL that is not absolute http(s) is refused; a redirecting
endpoint is not followed.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import json
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class _Sink:
    def __init__(self, redirect_to: str | None = None):
        self.bodies: list[dict] = []  # JSON bodies of the POSTs it accepted
        self.methods: list[str] = []  # EVERY request it saw, of any method (a followed redirect would show as GET)
        sink = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b""
                sink.methods.append(self.command)
                if redirect_to:
                    self.send_response(302)
                    self.send_header("Location", redirect_to)
                    self.end_headers()
                    return
                if self.command == "POST" and raw:
                    sink.bodies.append(json.loads(raw))
                self.send_response(200)
                self.end_headers()

            do_POST = do_GET = do_PUT = do_HEAD = _handle  # noqa: N815

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/hook"

    def wait(self, n=1, timeout=15.0):
        end = time.time() + timeout
        while time.time() < end and len(self.bodies) < n:
            time.sleep(0.1)
        return list(self.bodies)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def sink():
    s = _Sink()
    yield s
    s.close()


async def _run_one_job(stores, queue, user_id):
    """Enqueue one job on ``queue`` and drain it with a real burst arq worker using the service's hook."""
    from arq import create_pool
    from arq.connections import RedisSettings
    from arq.worker import Worker

    import worker as ns_worker

    async def noop(ctx):
        return "done"

    redis_settings = RedisSettings.from_dsn(stores.redis_url)
    pool = await create_pool(redis_settings)
    job_id = f"it-wh-{uuid.uuid4().hex[:8]}"
    leftovers = [
        queue, f"{queue}:health-check", f"ns:task-user:{job_id}", f"arq:job:{job_id}",
        f"arq:result:{job_id}", f"arq:retry:{job_id}", f"arq:in-progress:{job_id}",
    ]
    try:
        await pool.set(f"ns:task-user:{job_id}", user_id, ex=300)
        await pool.enqueue_job("noop", _job_id=job_id, _queue_name=queue)
        w = Worker(
            functions=[__import__("arq").func(noop, name="noop")],
            redis_settings=redis_settings,
            queue_name=queue,
            burst=True,
            keep_result=0,  # leave no result key behind
            handle_signals=False,
            after_job_end=ns_worker._make_after_job_end(queue),
        )
        await w.main()
        await w.close()
    finally:
        await pool.delete(*leftovers)  # every key this run could have created
        await pool.aclose()
    return job_id


def test_a_drained_real_queue_posts_the_documented_event(flags, stores, unique_ns, sink):
    queue = f"nsit:{unique_ns.group_prefix}:queue"
    flags(webhook_queue_empty_url=sink.url)

    job_id = asyncio.run(_run_one_job(stores, queue, unique_ns.user_id))

    (event,) = sink.wait(1)
    assert event["event"] == "queue.empty"
    assert event["queue"] == queue
    assert event["job_id"] == job_id
    assert event["user_id"] == unique_ns.user_id
    datetime.fromisoformat(event["timestamp"])


def test_url_unset_sends_nothing(flags, stores, unique_ns, sink):
    flags(webhook_queue_empty_url="")
    asyncio.run(_run_one_job(stores, f"nsit:{unique_ns.group_prefix}:queue", unique_ns.user_id))
    assert sink.wait(1, timeout=3.0) == []


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://127.0.0.1/x", "http:///nohost", "127.0.0.1:9/x", ""])
def test_a_url_that_is_not_absolute_http_is_refused(url):
    import webhooks

    assert webhooks.fire_queue_empty(url, {"event": "queue.empty"}) is False


def test_redirects_are_not_followed():
    import webhooks

    target = _Sink()
    redirector = _Sink(redirect_to=target.url)
    try:
        assert webhooks.fire_queue_empty(redirector.url, {"event": "queue.empty"}) is True
        end = time.time() + 10
        while time.time() < end and not redirector.methods:
            time.sleep(0.1)
        time.sleep(1.0)  # room for a (wrongly) followed redirect to arrive
        assert redirector.methods == ["POST"], "the webhook was not delivered to the redirecting endpoint"
        assert target.methods == [], f"the webhook followed a redirect: the target saw {target.methods}"
    finally:
        target.close()
        redirector.close()
