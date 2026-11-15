"""Real-Redis check that /flush binds identity to the per-user token (sec-02).

Lives in ``integration_tests/`` (outside the host/container unit suites, like
the other live-store qualifications) and is run explicitly with Redis up; an
unreachable Redis is a test failure. Uses a private ARQ queue name so no live
worker can pick the job up.
"""

import asyncio
import uuid
from unittest.mock import MagicMock

import pytest
from arq import create_pool
from arq.jobs import Job
from fastapi.testclient import TestClient

from config import parse_redis_settings, settings
from main import app
from task_manager import TaskManager
from tokens import issue_user_token

SECRET = "route-identity-integration-secret"
PREFIX = "/v1/extensions/conversation-compiler"
FLUSH = {"user_message": "hi", "assistant_response": "hello", "session_id": "s-int"}


pytestmark = pytest.mark.integration


def _bearer(user_id):
    return {"Authorization": f"Bearer {issue_user_token(user_id, SECRET, 3600)}"}


@pytest.fixture()
def real_queue(monkeypatch):
    from extensions.conversation_compiler.routes import create_router

    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    queue = f"ns-test-sec02-{uuid.uuid4().hex[:8]}"
    loop = asyncio.new_event_loop()
    pool = loop.run_until_complete(create_pool(parse_redis_settings(), default_queue_name=queue))

    class PerCallPool:
        """TestClient serves requests on its own loop; a redis pool is bound to
        the loop that made it, so enqueue through a short-lived pool made there."""

        async def enqueue_job(self, *args, **kwargs):
            p = await create_pool(parse_redis_settings(), default_queue_name=queue)
            try:
                return await p.enqueue_job(*args, **kwargs)
            finally:
                await p.close()

    tm = TaskManager()
    tm.pool = PerCallPool()
    before = len(app.router.routes)
    app.include_router(create_router(MagicMock(), MagicMock(), task_manager=tm), prefix=PREFIX)
    try:
        yield loop, pool, queue
    finally:
        del app.router.routes[before:]
        loop.run_until_complete(pool.delete(queue))
        loop.run_until_complete(pool.close())
        loop.close()


def test_flush_naming_other_user_is_refused_and_queue_stays_empty(real_queue):
    loop, pool, queue = real_queue
    resp = TestClient(app).post(PREFIX + "/flush", json={**FLUSH, "user_id": "bob"}, headers=_bearer("alice"))
    assert resp.status_code == 400
    assert loop.run_until_complete(pool.zcard(queue)) == 0


def test_flush_naming_token_user_enqueues_job_carrying_that_user(real_queue):
    loop, pool, queue = real_queue
    resp = TestClient(app).post(PREFIX + "/flush", json={**FLUSH, "user_id": "alice"}, headers=_bearer("alice"))
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]
    assert loop.run_until_complete(pool.zcard(queue)) == 1
    info = loop.run_until_complete(Job(task_id, pool, _queue_name=queue).info())
    assert info.function == "process_conversation_flush"
    assert info.args[-1] == "alice"
