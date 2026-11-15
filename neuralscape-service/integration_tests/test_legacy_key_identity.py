"""Real-Redis proof of the legacy-key identity binding and token revocation (sec-05).

Lives in ``integration_tests/`` and is run explicitly with Redis up; an
unreachable Redis is a test failure (no skips). The real app and middleware
serve real routes; the write route enqueues onto a private ARQ queue so no
live worker can pick the job up.

With ``NEURALSCAPE_API_KEY`` (+ ``NEURALSCAPE_API_KEY_USER_ID``) and
``NEURALSCAPE_USER_TOKEN_SECRET`` configured:
  - a write as the bound identity is enqueued under that identity, and the
    status owner record names it;
  - a write naming another user is a 400 and nothing is enqueued;
  - a revoked per-user token is 401 while an unrevoked one works.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager

import pytest
from arq import create_pool
from arq.jobs import Job
from fastapi.testclient import TestClient

import main
from config import parse_redis_settings, settings
from main import app
from task_manager import TaskManager, _task_owner_key
from tokens import issue_user_token, token_fingerprint

pytestmark = pytest.mark.integration

KEY = "legacy-key-integration-secret"
SECRET = "legacy-key-integration-token-secret"
BOUND = "alice-bound"
ROUTE = "/v1/memories"


def _body(text, user_id=None):
    body = {"messages": [{"role": "user", "content": text}]}
    if user_id:
        body["user_id"] = user_id
    return body


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def real_queue(monkeypatch):
    """Point the app's real TaskManager at a private queue on the real Redis."""
    queue = f"ns-test-sec05-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(settings, "neuralscape_api_key", KEY)
    monkeypatch.setattr(settings, "neuralscape_api_key_user_id", BOUND)
    monkeypatch.setattr(settings, "neuralscape_legacy_key_trust_body_user_id", False)
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_revoked_tokens", "")

    loop = asyncio.new_event_loop()
    pool = loop.run_until_complete(create_pool(parse_redis_settings(), default_queue_name=queue))

    class PerCallPool:
        """TestClient serves requests on its own loop and a redis pool is bound
        to the loop that made it, so every call goes through a short-lived pool
        made on the serving loop."""

        def __getattr__(self, name):
            async def call(*args, **kwargs):
                p = await create_pool(parse_redis_settings(), default_queue_name=queue)
                try:
                    return await getattr(p, name)(*args, **kwargs)
                finally:
                    await p.close()

            return call

        @asynccontextmanager
        async def pipeline(self, transaction=True):
            p = await create_pool(parse_redis_settings(), default_queue_name=queue)
            try:
                async with p.pipeline(transaction=transaction) as pipe:
                    yield pipe
            finally:
                await p.close()

    tm = TaskManager()
    tm.pool = PerCallPool()
    saved = main._task_manager
    main._task_manager = tm
    created = []
    try:
        yield loop, pool, queue, created
    finally:
        main._task_manager = saved
        for task_id in created:
            loop.run_until_complete(pool.delete(_task_owner_key(task_id)))
        loop.run_until_complete(pool.delete(queue))
        loop.run_until_complete(pool.close())
        loop.close()


def _queued(loop, pool, queue):
    return loop.run_until_complete(pool.zcard(queue))


def _owners(loop, pool, task_id):
    return {m.decode() if isinstance(m, bytes) else m
            for m in loop.run_until_complete(pool.smembers(_task_owner_key(task_id)))}


def test_write_as_bound_identity_is_enqueued_under_that_identity(real_queue):
    loop, pool, queue, created = real_queue
    resp = TestClient(app).post(ROUTE, json=_body("bound identity write"), headers=_bearer(KEY))
    assert resp.status_code == 202
    assert "deprecated" in resp.headers["X-Neuralscape-Deprecation"]
    task_id = resp.json()["task_id"]
    created.append(task_id)
    assert _queued(loop, pool, queue) == 1
    info = loop.run_until_complete(Job(task_id, pool, _queue_name=queue).info())
    assert info.function == "process_memory_store"
    assert info.args[1] == BOUND  # (messages, user_id, ...)
    assert _owners(loop, pool, task_id) == {BOUND}


def test_write_naming_the_bound_user_is_accepted(real_queue):
    loop, pool, queue, created = real_queue
    resp = TestClient(app).post(ROUTE, json=_body("names bound user", BOUND), headers=_bearer(KEY))
    assert resp.status_code == 202
    created.append(resp.json()["task_id"])
    assert _queued(loop, pool, queue) == 1


def test_write_naming_another_user_is_400_and_nothing_is_enqueued(real_queue):
    loop, pool, queue, _ = real_queue
    resp = TestClient(app).post(ROUTE, json=_body("impersonation", "bob"), headers=_bearer(KEY))
    assert resp.status_code == 400
    assert "does not match" in resp.json()["detail"]
    assert _queued(loop, pool, queue) == 0


def test_opt_out_restores_trust_of_the_body_user(real_queue, monkeypatch):
    loop, pool, queue, created = real_queue
    monkeypatch.setattr(settings, "neuralscape_legacy_key_trust_body_user_id", True)
    resp = TestClient(app).post(ROUTE, json=_body("opt-out write", "bob"), headers=_bearer(KEY))
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]
    created.append(task_id)
    info = loop.run_until_complete(Job(task_id, pool, _queue_name=queue).info())
    assert info.args[1] == "bob"


def test_revoked_token_is_401_and_unrevoked_token_works(real_queue, monkeypatch):
    loop, pool, queue, created = real_queue
    leaked = issue_user_token("carol", SECRET, 3600)
    fine = issue_user_token("carol", SECRET, 7200)
    client = TestClient(app)

    resp = client.post(ROUTE, json=_body("before revocation"), headers=_bearer(leaked))
    assert resp.status_code == 202
    created.append(resp.json()["task_id"])
    assert _queued(loop, pool, queue) == 1

    monkeypatch.setattr(settings, "neuralscape_revoked_tokens", token_fingerprint(leaked))
    resp = client.post(ROUTE, json=_body("after revocation"), headers=_bearer(leaked))
    assert resp.status_code == 401
    assert _queued(loop, pool, queue) == 1  # nothing new enqueued

    resp = client.post(ROUTE, json=_body("unrevoked token"), headers=_bearer(fine))
    assert resp.status_code == 202
    created.append(resp.json()["task_id"])
    assert _queued(loop, pool, queue) == 2
