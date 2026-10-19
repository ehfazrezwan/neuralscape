"""Task status is readable only by the user who enqueued the task.

With per-user tokens, ``GET /v1/memories/status/{id}`` and the legacy
``GET /memories/status/{id}`` must answer another user's task exactly like an
unknown id (404, no state, no result). With the shared key there is no caller
identity on the request, so the status is returned as before.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from arq.jobs import JobStatus
from fastapi.testclient import TestClient

import main
from config import settings
from main import app
from task_manager import TaskManager
from tokens import issue_user_token

SECRET = "task-owner-test-secret"
SHARED_KEY = "task-owner-shared-key"
# The owner record's Redis key (task_manager._task_owner_key).
def _task_owner_key(task_id):
    return f"ns:task-owner:{task_id}"


ROUTES = ["/v1/memories/status/{}", "/memories/status/{}"]


class FakeRedis:
    """Async Redis stub with the commands task tracking uses."""

    def __init__(self):
        self.sets: dict[str, set[str]] = {}
        self.ttls: dict[str, int] = {}
        self.strings: dict[str, str] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    async def sadd(self, key, *members):
        self.sets.setdefault(key, set()).update(members)

    async def sismember(self, key, member):
        return member in self.sets.get(key, set())

    async def expire(self, key, ttl):
        self.ttls[key] = ttl
        return True

    async def set(self, key, value, ex=None):
        self.strings[key] = value

    async def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    async def zremrangebyscore(self, key, lo, hi):
        return 0


class FakeJob:
    def __init__(self, result):
        self._result = result

    async def result_info(self):
        return MagicMock(success=True, result=self._result)


PRIVATE_RESULT = {"memories": [{"id": "m-1", "memory": "alice private text", "user_id": "alice"}]}


@pytest.fixture()
def tm(monkeypatch):
    manager = TaskManager()
    manager.pool = PipelineRedis()
    manager.pool.enqueue_job = AsyncMock(side_effect=lambda *a, **k: MagicMock(job_id=k.get("_job_id") or "job-x"))
    known: dict[str, JobStatus] = {}

    async def find_job(task_id):
        status = known.get(task_id, JobStatus.not_found)
        return FakeJob(PRIVATE_RESULT), status

    monkeypatch.setattr(manager, "_find_job", find_job)
    manager.known = known
    original = main._task_manager
    main._task_manager = manager
    yield manager
    main._task_manager = original


@pytest.fixture()
def per_user_auth(monkeypatch):
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")


def _bearer(user_id):
    return {"Authorization": f"Bearer {issue_user_token(user_id, SECRET, 3600)}"}


async def _alice_task(tm, status=JobStatus.complete):
    task_id = await tm.enqueue_raw(content="alice private text", user_id="alice", category="preference")
    tm.known[task_id] = status
    return task_id


class TestOwnerOnlyStatus:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", ROUTES)
    async def test_owner_reads_own_result(self, tm, per_user_auth, route):
        task_id = await _alice_task(tm)
        resp = TestClient(app).get(route.format(task_id), headers=_bearer("alice"))
        assert resp.status_code == 200
        assert resp.json()["result"] == PRIVATE_RESULT

    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", ROUTES)
    @pytest.mark.parametrize("status", [JobStatus.queued, JobStatus.in_progress, JobStatus.complete])
    async def test_other_user_gets_unknown_task_404(self, tm, per_user_auth, route, status):
        task_id = await _alice_task(tm, status)
        client = TestClient(app)
        other = client.get(route.format(task_id), headers=_bearer("bob"))
        unknown = client.get(route.format("ns-does-not-exist"), headers=_bearer("bob"))
        assert other.status_code == 404
        assert "alice" not in other.text
        # Same body shape as an id that never existed (only the echoed id differs).
        assert other.json()["detail"].replace(task_id, "X") == unknown.json()["detail"].replace(
            "ns-does-not-exist", "X"
        )

    @pytest.mark.asyncio
    async def test_owner_check_runs_before_job_lookup(self, tm, per_user_auth, monkeypatch):
        task_id = await _alice_task(tm)
        lookups = []

        async def spy(tid):
            lookups.append(tid)
            return FakeJob(PRIVATE_RESULT), JobStatus.complete

        monkeypatch.setattr(tm, "_find_job", spy)
        assert TestClient(app).get(f"/v1/memories/status/{task_id}", headers=_bearer("bob")).status_code == 404
        assert lookups == []

    @pytest.mark.asyncio
    async def test_unrecorded_owner_fails_closed(self, tm, per_user_auth):
        tm.known["ns-orphan"] = JobStatus.complete
        resp = TestClient(app).get("/v1/memories/status/ns-orphan", headers=_bearer("alice"))
        assert resp.status_code == 404


class TestSharedKeyStatus:
    """The shared key carries no caller identity: status is not owner-checked."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", ROUTES)
    async def test_shared_key_reads_any_task(self, tm, monkeypatch, route):
        monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
        monkeypatch.setattr(settings, "neuralscape_api_key", SHARED_KEY)
        task_id = await _alice_task(tm)
        resp = TestClient(app).get(route.format(task_id), headers={"Authorization": f"Bearer {SHARED_KEY}"})
        assert resp.status_code == 200
        assert resp.json()["result"] == PRIVATE_RESULT

    @pytest.mark.asyncio
    async def test_auth_disabled_reads_any_task(self, tm, monkeypatch):
        monkeypatch.setattr(settings, "neuralscape_user_token_secret", "")
        monkeypatch.setattr(settings, "neuralscape_api_key", "")
        task_id = await _alice_task(tm)
        assert TestClient(app).get(f"/v1/memories/status/{task_id}").status_code == 200


class TestOwnerRecording:
    @pytest.mark.asyncio
    async def test_owner_record_outlives_queue_expiry_and_result(self, tm):
        task_id = await _alice_task(tm)
        # ARQ: one day before an unstarted job expires, one hour of kept result.
        assert tm.pool.ttls[_task_owner_key(task_id)] >= 24 * 3600 + 3600

    @pytest.mark.asyncio
    async def test_batch_grants_only_a_single_writer(self, tm):
        single = await tm.enqueue_raw_batch([
            {"content": "a", "user_id": "alice", "category": "decision"},
            {"content": "b", "user_id": "alice", "category": "decision"},
        ])
        mixed = await tm.enqueue_raw_batch([
            {"content": "a", "user_id": "alice", "category": "decision"},
            {"content": "b", "user_id": "bob", "category": "decision"},
        ])
        assert tm.pool.sets[_task_owner_key(single)] == {"alice"}
        assert _task_owner_key(mixed) not in tm.pool.sets

    @pytest.mark.asyncio
    async def test_connector_sync_records_requesting_owner(self, tm):
        task_id = await tm.enqueue_connector_sync("notion-1", user_id="alice", generation="g1")
        assert await tm.is_task_owner(task_id, "alice")
        assert not await tm.is_task_owner(task_id, "bob")

    @pytest.mark.asyncio
    async def test_okf_bundle_records_owner(self, tm):
        task_id = await tm.enqueue_ingest_okf_bundle({"user_id": "alice", "filename": "b.zip", "data_b64": "x"})
        assert await tm.is_task_owner(task_id, "alice")

    @pytest.mark.asyncio
    async def test_owner_record_failure_never_fails_enqueue(self, tm):
        tm.pool.fail_exec = True
        task_id = await tm.enqueue_raw(content="x", user_id="alice", category="preference")
        assert task_id

    @pytest.mark.asyncio
    async def test_internal_status_read_is_unchecked(self, tm):
        task_id = await _alice_task(tm)
        assert (await tm.get_status(task_id))["status"] == "completed"


# ── Follow-up: every enqueue that hands out a pollable task id ──────────────


class FakePipeline:
    """MULTI/EXEC stub: commands are buffered and applied only on execute."""

    def __init__(self, redis, transaction):
        self.redis = redis
        self.transaction = transaction
        self.ops = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def sadd(self, key, *members):
        self.ops.append(("sadd", key, members))
        return self

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))
        return self

    async def execute(self):
        if self.redis.fail_exec:
            raise ConnectionError("redis down")
        self.redis.transactions.append((self.transaction, [(op, key) for op, key, _ in self.ops]))
        for op, key, arg in self.ops:
            if op == "sadd":
                self.redis.sets.setdefault(key, set()).update(arg)
            else:
                self.redis.ttls[key] = arg
        return [True] * len(self.ops)


class PipelineRedis(FakeRedis):
    """FakeRedis that also supports pipelines and logs direct owner-key writes."""

    def __init__(self):
        super().__init__()
        self.fail_exec = False
        self.transactions = []
        self.direct = []
        self.closed = False

    def pipeline(self, transaction=True):
        return FakePipeline(self, transaction)

    async def sadd(self, key, *members):
        self.direct.append(("sadd", key))
        await super().sadd(key, *members)

    async def expire(self, key, ttl):
        self.direct.append(("expire", key))
        return await super().expire(key, ttl)

    async def close(self):
        self.closed = True


@pytest.fixture()
def ptm(tm):
    """``tm`` (already backed by the pipeline-capable Redis stub)."""
    return tm


@pytest.fixture()
def mounted(monkeypatch):
    """Mount extra routers on the real app (real auth middleware) for one test."""
    before = len(app.router.routes)

    def mount(router, prefix):
        app.include_router(router, prefix=prefix)

    yield mount
    del app.router.routes[before:]


def _poll(task_id, user):
    return TestClient(app).get(f"/v1/memories/status/{task_id}", headers=_bearer(user))


class TestOwnerRecordIsAtomic:
    @pytest.mark.asyncio
    async def test_sadd_and_expire_in_one_transaction(self, ptm):
        task_id = await ptm.enqueue_raw(content="x", user_id="alice", category="preference")
        key = _task_owner_key(task_id)
        assert (True, [("sadd", key), ("expire", key)]) in ptm.pool.transactions
        assert [c for c in ptm.pool.direct if c[1] == key] == []

    @pytest.mark.asyncio
    async def test_failed_transaction_leaves_no_unexpiring_owner_key(self, ptm):
        ptm.pool.fail_exec = True
        task_id = await ptm.enqueue_raw(content="x", user_id="alice", category="preference")
        assert task_id
        key = _task_owner_key(task_id)
        assert key not in ptm.pool.sets and key not in ptm.pool.ttls


class TestOwnerCheckRedisError:
    """A Redis failure during the owner check answers 503 for every id alike."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", ROUTES)
    async def test_owner_check_error_is_503_without_job_lookup(self, ptm, per_user_auth, monkeypatch, route):
        task_id = await _alice_task(ptm)
        lookups = []

        async def spy(tid):
            lookups.append(tid)
            return FakeJob(PRIVATE_RESULT), JobStatus.complete

        monkeypatch.setattr(ptm, "_find_job", spy)
        ptm.pool.sismember = AsyncMock(side_effect=ConnectionError("redis down"))
        client = TestClient(app)
        owner = client.get(route.format(task_id), headers=_bearer("alice"))
        other = client.get(route.format(task_id), headers=_bearer("bob"))
        unknown = client.get(route.format("ns-does-not-exist"), headers=_bearer("bob"))
        for resp in (owner, other, unknown):
            assert resp.status_code == 503
            assert "alice" not in resp.text and task_id not in resp.text
        assert owner.json() == other.json() == unknown.json()
        assert lookups == []


class TestConnectorSyncRoute:
    @pytest.mark.asyncio
    async def test_sync_started_by_alice_is_hers_alone(self, ptm, per_user_auth, monkeypatch):
        vault = MagicMock()
        vault.get_redacted = AsyncMock(return_value={"connector_id": "notion-1", "owner_user_id": "alice"})
        monkeypatch.setattr(main, "_vault", vault)
        client = TestClient(app)
        resp = client.post("/v1/connectors/notion-1/sync", headers=_bearer("alice"))
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]
        ptm.known[task_id] = JobStatus.complete
        assert _poll(task_id, "alice").status_code == 200
        assert _poll(task_id, "bob").status_code == 404
        # bob cannot start (or learn about) alice's connector sync either.
        assert client.post("/v1/connectors/notion-1/sync", headers=_bearer("bob")).status_code == 404


class TestConversationCompilerTasks:
    PREFIX = "/v1/extensions/conversation-compiler"

    @pytest.fixture()
    def compiler(self, ptm, mounted):
        from extensions.conversation_compiler.routes import create_router

        mounted(create_router(MagicMock(), MagicMock(), task_manager=ptm), self.PREFIX)
        return ptm

    FLUSH = {"user_message": "hi", "assistant_response": "hello", "session_id": "s-1"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path,body", [("/flush", FLUSH), ("/compile", {"date": "2026-10-18"})])
    async def test_enqueuer_can_poll_and_others_cannot(self, compiler, per_user_auth, path, body):
        resp = TestClient(app).post(self.PREFIX + path, json={**body, "user_id": "alice"}, headers=_bearer("alice"))
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]
        compiler.known[task_id] = JobStatus.complete
        assert _poll(task_id, "alice").status_code == 200
        assert _poll(task_id, "bob").status_code == 404

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path,body", [("/flush", FLUSH), ("/compile", {})])
    async def test_task_for_another_user_has_no_owner(self, compiler, per_user_auth, path, body):
        """A token holder naming another user in the body owns nothing: fail closed."""
        resp = TestClient(app).post(self.PREFIX + path, json={**body, "user_id": "bob"}, headers=_bearer("alice"))
        task_id = resp.json()["task_id"]
        assert not await compiler.is_task_owner(task_id, "alice")
        assert not await compiler.is_task_owner(task_id, "bob")

    @pytest.mark.asyncio
    async def test_shared_key_records_body_user(self, compiler, monkeypatch):
        monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
        monkeypatch.setattr(settings, "neuralscape_api_key", SHARED_KEY)
        resp = TestClient(app).post(
            self.PREFIX + "/flush",
            json={**self.FLUSH, "user_id": "alice"},
            headers={"Authorization": f"Bearer {SHARED_KEY}"},
        )
        assert await compiler.is_task_owner(resp.json()["task_id"], "alice")

    @pytest.mark.asyncio
    async def test_auto_compile_records_session_user(self, ptm, monkeypatch):
        from extensions.conversation_compiler import ConversationCompilerExtension
        from extensions.conversation_compiler.config import compiler_settings

        monkeypatch.setattr(compiler_settings, "auto_compile", True)
        monkeypatch.setattr(compiler_settings, "compile_after_hour", 0)
        ext = ConversationCompilerExtension()
        ext._task_manager = ptm
        ext._writer = MagicMock(is_daily_log_compiled=MagicMock(return_value=False))
        out = await ext._handle_session_end({"user_id": "alice"})
        assert out["status"] == "accepted"
        assert await ptm.is_task_owner(out["task_id"], "alice")
        assert not await ptm.is_task_owner(out["task_id"], "bob")


class TestDreamSweepTasks:
    @pytest.fixture()
    def dreaming(self, ptm, mounted, monkeypatch):
        import arq

        from extensions.dreaming import DreamingExtension
        from extensions.dreaming.config import dreaming_settings

        monkeypatch.setattr(dreaming_settings, "enabled", True)
        monkeypatch.setattr(arq, "create_pool", AsyncMock(return_value=ptm.pool))
        mounted(DreamingExtension().get_routes(), "/v1/extensions/dreaming")
        return ptm

    @pytest.mark.asyncio
    async def test_rest_sweep_pollable_by_requester_only(self, dreaming, per_user_auth):
        resp = TestClient(app).post("/v1/extensions/dreaming/run", json={"dry_run": True}, headers=_bearer("alice"))
        assert resp.status_code == 202
        task_id = resp.json()["job_id"]
        dreaming.known[task_id] = JobStatus.complete
        assert _poll(task_id, "alice").status_code == 200
        assert _poll(task_id, "bob").status_code == 404

    @pytest.mark.asyncio
    async def test_mcp_sweep_records_effective_user(self, ptm, monkeypatch):
        import json

        import mcp_server
        from auth import current_user_id
        from extensions.dreaming.config import dreaming_settings

        monkeypatch.setattr(dreaming_settings, "enabled", True)
        monkeypatch.setattr(mcp_server, "_arq_pool", ptm.pool)
        token = current_user_id.set("alice")
        try:
            result = await mcp_server.call_tool("schedule_dream", {"dry_run": True})
        finally:
            current_user_id.reset(token)
        task_id = json.loads(result[0].text)["job_id"]
        assert await ptm.is_task_owner(task_id, "alice")
        assert not await ptm.is_task_owner(task_id, "bob")


# ── Read grants are separate from queue bookkeeping ──────────────────────────


def _shared_key_auth(monkeypatch):
    """Shared key and per-user tokens both accepted (a mixed deployment)."""
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", SHARED_KEY)
    return {"Authorization": f"Bearer {SHARED_KEY}"}


class TestMixedUserBatch:
    """A batch spanning several users grants no per-user read of its result."""

    MIXED_RESULT = {
        "memories": [
            {"id": "m-a", "memory": "alice private text", "user_id": "alice"},
            {"id": "m-b", "memory": "bob private text", "user_id": "bob"},
        ]
    }

    @pytest.fixture()
    def batch_tm(self, ptm, monkeypatch):
        async def find_job(task_id):
            return FakeJob(self.MIXED_RESULT), ptm.known.get(task_id, JobStatus.not_found)

        monkeypatch.setattr(ptm, "_find_job", find_job)
        return ptm

    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", ROUTES)
    async def test_shared_key_mixed_batch_is_not_readable_per_user(self, batch_tm, monkeypatch, route):
        shared = _shared_key_auth(monkeypatch)
        client = TestClient(app)
        resp = client.post(
            "/v1/memories/raw/batch",
            json={"memories": [
                {"content": "alice private text", "user_id": "alice", "category": "decision"},
                {"content": "bob private text", "user_id": "bob", "category": "decision"},
            ]},
            headers=shared,
        )
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]
        batch_tm.known[task_id] = JobStatus.complete
        unknown = client.get(route.format("ns-does-not-exist"), headers=_bearer("alice"))
        for user in ("alice", "bob", "carol"):
            poll = client.get(route.format(task_id), headers=_bearer(user))
            assert poll.status_code == 404
            assert "private text" not in poll.text
            assert poll.json()["detail"].replace(task_id, "X") == unknown.json()["detail"].replace(
                "ns-does-not-exist", "X"
            )
        # The shared key, which may write and read as any user, still reads it.
        assert client.get(route.format(task_id), headers=shared).json()["result"] == self.MIXED_RESULT

    @pytest.mark.asyncio
    async def test_mixed_batch_still_tracked_for_every_writer(self, batch_tm):
        task_id = await batch_tm.enqueue_raw_batch([
            {"content": "a", "user_id": "alice", "category": "decision"},
            {"content": "b", "user_id": "bob", "category": "decision"},
        ])
        assert _task_owner_key(task_id) not in batch_tm.pool.sets
        for user in ("alice", "bob"):
            assert task_id in batch_tm.pool.zsets[f"ns:user-tasks:{user}"]

    @pytest.mark.asyncio
    async def test_mixed_batch_job_id_is_deterministic(self, batch_tm):
        items = [
            {"content": "a", "user_id": "alice", "category": "decision"},
            {"content": "b", "user_id": "bob", "category": "decision"},
        ]
        assert await batch_tm.enqueue_raw_batch(items) == await batch_tm.enqueue_raw_batch(list(items))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", ROUTES)
    async def test_single_user_batch_pollable_by_its_user(self, batch_tm, per_user_auth, route):
        client = TestClient(app)
        resp = client.post(
            "/v1/memories/raw/batch",
            json={"memories": [
                {"content": "one", "category": "decision"},
                {"content": "two", "category": "decision"},
            ]},
            headers=_bearer("alice"),
        )
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]
        batch_tm.known[task_id] = JobStatus.complete
        assert client.get(route.format(task_id), headers=_bearer("alice")).status_code == 200
        assert client.get(route.format(task_id), headers=_bearer("bob")).status_code == 404

    @pytest.mark.asyncio
    async def test_shared_key_single_user_batch_granted_to_that_user(self, batch_tm, monkeypatch):
        shared = _shared_key_auth(monkeypatch)
        resp = TestClient(app).post(
            "/v1/memories/raw/batch",
            json={"memories": [{"content": "one", "user_id": "alice", "category": "decision"}]},
            headers=shared,
        )
        task_id = resp.json()["task_id"]
        assert await batch_tm.is_task_owner(task_id, "alice")
        assert not await batch_tm.is_task_owner(task_id, "bob")


# ── Connector sync identity follows the connector instance and its owner ────


class _VaultStore:
    """In-memory async KV with the redis methods ConnectorVault uses."""

    def __init__(self):
        self.d: dict[str, str] = {}

    async def set(self, k, v):
        self.d[k] = v

    async def get(self, k):
        return self.d.get(k)

    async def delete(self, k):
        return 1 if self.d.pop(k, None) is not None else 0

    async def keys(self, pattern):
        import fnmatch

        return [k for k in self.d if fnmatch.fnmatch(k, pattern)]


class TestConnectorSyncIdentity:
    CID = "notion-shared-name"
    BODY = {"connector_id": CID, "connector_type": "notion", "credentials": {"token": "t"}}

    @pytest.fixture()
    def vault(self, monkeypatch):
        from cryptography.fernet import Fernet

        from connectors.vault import ConnectorVault

        v = ConnectorVault(_VaultStore(), Fernet(Fernet.generate_key()))
        monkeypatch.setattr(main, "_vault", v)
        return v

    @pytest.fixture()
    def arq(self, ptm, monkeypatch):
        """ARQ enqueue semantics: an id with a queued or retained job returns None."""
        seen: dict[str, dict] = {}
        results: dict[str, dict] = {}

        async def enqueue_job(fn, *args, _job_id=None, **kw):
            if _job_id in seen:
                return None
            seen[_job_id] = {"fn": fn, "args": args, **kw}
            return MagicMock(job_id=_job_id)

        async def find_job(task_id):
            if task_id not in seen:
                return FakeJob(None), JobStatus.not_found
            return FakeJob(results.get(task_id)), JobStatus.complete

        ptm.pool.enqueue_job = enqueue_job
        monkeypatch.setattr(ptm, "_find_job", find_job)
        ptm.seen, ptm.results = seen, results
        return ptm

    def _register(self, client, user):
        assert client.post("/v1/connectors", json=self.BODY, headers=_bearer(user)).status_code == 201

    def _sync(self, client, user):
        resp = client.post(f"/v1/connectors/{self.CID}/sync", headers=_bearer(user))
        assert resp.status_code == 202
        return resp.json()["task_id"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("change", ["delete_recreate", "reassign"])
    @pytest.mark.parametrize("old_job", ["cached", "expired"])
    async def test_owner_change_gives_disjoint_jobs(self, vault, arq, per_user_auth, change, old_job):
        client = TestClient(app)
        self._register(client, "alice")
        old = self._sync(client, "alice")
        arq.results[old] = {"owner": "alice"}
        if old_job == "expired":
            # The old result aged out of ARQ; the id is free again.
            del arq.seen[old]
        if change == "delete_recreate":
            assert client.delete(f"/v1/connectors/{self.CID}", headers=_bearer("alice")).status_code == 200
        self._register(client, "bob")
        new = self._sync(client, "bob")
        arq.results[new] = {"owner": "bob"}

        assert new != old
        if old_job == "cached":
            assert _poll(old, "alice").json()["result"] == {"owner": "alice"}
        assert _poll(new, "bob").json()["result"] == {"owner": "bob"}
        # Neither owner can read the other's sync.
        assert _poll(old, "bob").status_code == 404
        assert _poll(new, "alice").status_code == 404

    @pytest.mark.asyncio
    async def test_same_owner_recreate_gets_a_fresh_job(self, vault, arq, per_user_auth):
        client = TestClient(app)
        self._register(client, "alice")
        old = self._sync(client, "alice")
        assert client.delete(f"/v1/connectors/{self.CID}", headers=_bearer("alice")).status_code == 200
        self._register(client, "alice")
        assert self._sync(client, "alice") != old

    @pytest.mark.asyncio
    async def test_api_and_cron_coalesce_on_one_job(self, vault, arq, per_user_auth, monkeypatch):
        import worker

        client = TestClient(app)
        self._register(client, "alice")
        # Re-registering the same instance (same owner) keeps its identity.
        self._register(client, "alice")
        monkeypatch.setattr(settings, "connectors_enabled", True)
        out = await worker.connector_sync_cron({"vault": vault, "redis": arq.pool})
        assert out["enqueued"] == 1
        (cron_id,) = arq.seen
        api_id = self._sync(client, "alice")
        assert api_id == cron_id
        assert len(arq.seen) == 1
        # The owner who asked can poll the coalesced job; nobody else can.
        assert _poll(api_id, "alice").status_code == 200
        assert _poll(api_id, "bob").status_code == 404

    @pytest.mark.asyncio
    async def test_stale_job_does_not_sync_a_replaced_connector(self, vault, monkeypatch):
        """A job enqueued for a previous instance skips, and never moves the new cursor."""
        import worker

        await vault.put({**self.BODY, "owner_user_id": "alice"})
        old_generation = (await vault.get(self.CID))["generation"]
        await vault.delete(self.CID)
        await vault.put({**self.BODY, "owner_user_id": "bob"})
        out = await worker.process_connector_sync({"vault": vault}, self.CID, old_generation)
        assert out["skipped"] is True
        await vault.update_sync_state(self.CID, cursor="stale", generation=old_generation)
        assert (await vault.get(self.CID))["cursor"] is None

    @pytest.mark.asyncio
    async def test_reregister_keeps_cursor_and_generation(self, vault):
        await vault.put({**self.BODY, "owner_user_id": "alice"})
        first = await vault.get(self.CID)
        await vault.update_sync_state(self.CID, cursor="c1", generation=first["generation"])
        await vault.put({**self.BODY, "owner_user_id": "alice", "generation": "forged"})
        again = await vault.get(self.CID)
        assert again["cursor"] == "c1"
        assert again["generation"] == first["generation"]


# ── Graph tasks from an edit belong to the user who made the edit ───────────


class TestSharedEditGraphTask:
    """A non-owner's allowed edit of a shared memory: the editor polls its graph task."""

    @pytest.fixture()
    def edit_service(self, monkeypatch):
        from types import SimpleNamespace

        from memory_service import MemoryService

        svc = MemoryService()
        svc._memory = MagicMock(name="Memory")
        svc._graphiti = None
        svc._bridge = None
        svc._memory.vector_store.get.return_value = SimpleNamespace(
            id="m1",
            payload={
                "data": "Team decision text",
                "hash": "h",
                "created_at": "2026-01-01T00:00:00+00:00",
                "user_id": "javi",
                "metadata": {
                    "scope": "project", "category": "decision", "project_id": "neuralscape",
                    "owner_user_id": "javi", "visibility": "shared",
                },
            },
        )
        svc.get_memory = MagicMock(return_value=None)
        svc._cascade_or_fallback_expire = MagicMock(return_value={"resolved": True})
        monkeypatch.setattr(main, "_service", svc)
        return svc

    @pytest.fixture()
    def graph_jobs(self, ptm):
        calls = []

        async def enqueue_job(fn, *args, _job_id=None, **kw):
            calls.append((fn, args))
            return MagicMock(job_id=_job_id)

        ptm.pool.enqueue_job = enqueue_job
        return calls

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["patch", "put"])
    async def test_editor_polls_graph_task_others_cannot(self, ptm, edit_service, graph_jobs, per_user_auth, method):
        client = TestClient(app)
        resp = getattr(client, method)(
            "/v1/memories/m1", json={"project_id": "bon002"}, headers=_bearer("robb")
        )
        assert resp.status_code == 200, resp.text
        task_id = resp.json()["graph_task_id"]
        assert task_id
        # The graph job still targets the storage owner's partition.
        ((fn, args),) = graph_jobs
        assert fn == "process_graph_enrichment"
        assert args[2] == "javi" and args[3] == "bon002" and args[4] == "shared"
        ptm.known[task_id] = JobStatus.complete
        assert _poll(task_id, "robb").status_code == 200
        assert _poll(task_id, "mallory").status_code == 404

    @pytest.mark.asyncio
    async def test_shared_key_edit_grants_the_body_user(self, ptm, edit_service, graph_jobs, monkeypatch):
        shared = _shared_key_auth(monkeypatch)
        client = TestClient(app)
        resp = client.patch(
            "/v1/memories/m1", json={"project_id": "bon002", "user_id": "robb"}, headers=shared
        )
        task_id = resp.json()["graph_task_id"]
        ptm.known[task_id] = JobStatus.complete
        assert client.get(f"/v1/memories/status/{task_id}", headers=shared).status_code == 200
        assert _poll(task_id, "robb").status_code == 200
        assert _poll(task_id, "mallory").status_code == 404

    @pytest.mark.asyncio
    async def test_mcp_editor_owns_graph_task(self, ptm, graph_jobs, monkeypatch):
        import json

        import mcp_server
        from auth import current_user_id

        svc = MagicMock()
        svc.patch_memory.return_value = {
            "memory": None,
            "graph": "migration_pending",
            "graph_job": {
                "memory_id": "m1", "content": "Team decision text", "user_id": "javi",
                "project_id": "bon002", "visibility": "shared", "source_ref": None,
            },
        }
        monkeypatch.setattr(mcp_server, "_service", svc)
        monkeypatch.setattr(mcp_server, "_task_manager", ptm)
        token = current_user_id.set("robb")
        try:
            result = await mcp_server.call_tool("edit_memory", {"memory_id": "m1", "project_id": "bon002"})
        finally:
            current_user_id.reset(token)
        task_id = json.loads(result[0].text)["graph_task_id"]
        assert await ptm.is_task_owner(task_id, "robb")
        assert not await ptm.is_task_owner(task_id, "mallory")
        ((fn, args),) = graph_jobs
        assert args[2] == "javi"
