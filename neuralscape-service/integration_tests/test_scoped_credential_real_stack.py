"""The scoped-credential round trip (C11) on the REAL memory stack.

Real Qdrant + Neo4j (Graphiti) + the embedding model through ``MemoryService``, and a real Redis queue
through ``TaskManager`` (a private ARQ queue name, so no live worker takes the job).  The FastAPI app runs
with ``NS_POLICY_MODE=enforce`` and ``CREDENTIALS_ENABLED=1`` over a real authority store (SQLite).  The
queue's worker is this test: it takes the enqueued job back out of Redis and runs ``process_memory_raw``
with the job's own arguments, exactly what the worker would do.

A user writes with the ordinary token, then a scoped credential issued to the same user must write, list,
search and get rows in the same namespace.  Before the fix the credential reached the memory layer under
its opaque subject id, so its list and search were empty and its get was a 404.

A missing or unreachable service is a test FAILURE, never a skip.  Required env: REDIS_URL, QDRANT_URL,
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GOOGLE_API_KEY.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import os
import uuid

import pytest
from arq import create_pool
from arq.jobs import Job
from fastapi.testclient import TestClient

import credentials
import main
import policy_enforcement
import policy_migration as pm
import worker
from authority_store_sql import SqliteAuthorityStore
from config import parse_redis_settings, settings
from contracts_policy import PolicyStatement
from credentials import CredentialService
from policy_enforcement import PolicyEnforcer, _handle, _mem0_payload_fetcher, memory_row_lookup
from task_manager import TaskManager
from tokens import issue_user_token

pytestmark = pytest.mark.integration

REQUIRED_ENV = ("REDIS_URL", "QDRANT_URL", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "GOOGLE_API_KEY")
SECRET = "scoped-real-stack-secret"
TENANT = "tenant-real"


@pytest.fixture(scope="module")
def svc():
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    assert not missing, f"integration environment incomplete, missing: {missing}"
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()  # real mem0 + Graphiti; raises if a store is unreachable
    yield service


class Stack:
    def __init__(self, svc, monkeypatch, tmp_path) -> None:
        self.svc = svc
        self.user = f"it-scoped-{uuid.uuid4().hex[:10]}"
        self.queue = f"ns-test-scoped-{uuid.uuid4().hex[:8]}"
        self.loop = asyncio.new_event_loop()
        self.pool = self.loop.run_until_complete(
            create_pool(parse_redis_settings(), default_queue_name=self.queue)
        )
        queue = self.queue

        class PerCallPool:
            """TestClient serves requests on its own loop and a redis pool is bound to the loop that made
            it, so every call goes through a short-lived pool made there."""

            def __getattr__(self, name):
                async def call(*args, **kwargs):
                    pool = await create_pool(parse_redis_settings(), default_queue_name=queue)
                    try:
                        return await getattr(pool, name)(*args, **kwargs)
                    finally:
                        await pool.close()

                return call

        tasks = TaskManager()
        tasks.pool = PerCallPool()
        monkeypatch.setattr(main, "_task_manager", tasks)
        monkeypatch.setattr(main, "_service", svc)

        self.store = SqliteAuthorityStore(str(tmp_path / "authority.db"))
        report = pm.apply_plan(
            self.store,
            pm.collect_inventory(
                aliases=({}, {}),
                pools=[pm.PoolObservation(self.user, "private", None, None)],
                extra_user_ids=[self.user],
            ),
            TENANT,
            include_legacy_key=False,
        )
        self.subject = report.subject_ids[self.user]
        self.store.grant_statements(
            TENANT,
            [
                PolicyStatement(
                    id="manage-credentials", effect="allow", subject_id=self.subject, credential_id=None,
                    action="manage_credentials", resource=_handle("credential", self.subject, TENANT),
                )
            ],
        )
        monkeypatch.setenv("CREDENTIALS_ENABLED", "1")
        monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
        monkeypatch.setattr(settings, "ns_auth_profile", "legacy_compat")
        monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
        monkeypatch.setattr(settings, "neuralscape_api_key", "")
        credentials.configure(CredentialService(self.store, refresh_grace=0))
        policy_enforcement.configure(
            PolicyEnforcer(self.store, self.store, row_lookup=memory_row_lookup(_mem0_payload_fetcher(svc), self.store))
        )
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.owner = issue_user_token(self.user, SECRET, 3600)

    def call(self, method, path, bearer, **kw):
        return self.client.request(method, path, headers={"Authorization": f"Bearer {bearer}"}, **kw)

    def write_and_run_worker(self, bearer, text) -> str:
        """POST the write (real Redis enqueue), then run the queued job as the worker would."""

        response = self.call(
            "POST", "/v1/memories/raw", bearer,
            json={"content": text, "category": "preference", "visibility": "private"},
        )
        assert response.status_code == 202, response.text
        task_id = response.json()["task_id"]
        info = self.loop.run_until_complete(Job(task_id, self.pool, _queue_name=self.queue).info())
        assert info.function == "process_memory_raw"
        assert info.args[1] == self.user, "the job must carry the user's own id, not the opaque subject"
        self.loop.run_until_complete(worker.process_memory_raw({"service": self.svc}, *info.args, **info.kwargs))
        return task_id

    def close(self) -> None:
        try:
            self.svc.delete_memories(user_id=self.user, include_shared=True)
        finally:
            credentials.configure(None)
            policy_enforcement.configure(None)
            self.store.close()
            self.loop.run_until_complete(self.pool.delete(self.queue))
            self.loop.run_until_complete(self.pool.close())
            self.loop.close()


@pytest.fixture
def stack(svc, monkeypatch, tmp_path):
    s = Stack(svc, monkeypatch, tmp_path)
    yield s
    s.close()


def test_a_scoped_credential_round_trips_on_the_real_stack(stack) -> None:
    stack.write_and_run_worker(stack.owner, "The user prefers dark roast coffee in the morning")
    owner_rows = stack.call("GET", "/v1/memories", stack.owner).json()
    assert owner_rows, "the owner token must see its own row"

    issued = stack.call("POST", "/v1/credentials", stack.owner, json={"scope": ["create", "read", "list", "search"]})
    assert issued.status_code == 200, issued.text
    token = issued.json()["access_token"]

    listed = stack.call("GET", "/v1/memories", token)
    assert listed.status_code == 200
    assert {r["id"] for r in listed.json()} == {r["id"] for r in owner_rows}  # C11: not empty
    found = stack.call("POST", "/v1/search", token, json={"query": "dark roast coffee"})
    assert found.status_code == 200 and found.json()["results"], "search must find the owner's row"
    fetched = stack.call("GET", f"/v1/memories/{owner_rows[0]['id']}", token)
    assert fetched.status_code == 200, fetched.text  # C11: was a 404

    # a write through the credential lands in the same namespace the owner token reads
    stack.write_and_run_worker(token, "The user also keeps a standing desk at work")
    again = stack.call("GET", "/v1/memories", stack.owner).json()
    assert len(again) >= len(owner_rows) + 1

    # a scope without delete cannot delete, and the row survives
    denied = stack.call("DELETE", f"/v1/memories/{owner_rows[0]['id']}", token)
    assert denied.status_code == 404
    assert stack.call("GET", f"/v1/memories/{owner_rows[0]['id']}", stack.owner).status_code == 200
