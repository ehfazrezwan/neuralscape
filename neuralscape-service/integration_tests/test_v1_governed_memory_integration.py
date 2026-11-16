"""V1 spine: governed durable memory, end to end.

Real: Redis and an ARQ worker, a SQLite ledger, a SQLite authority store, the
policy enforcement layer in ``enforce`` mode, the real FastAPI app with real
HMAC user tokens, the publication fence, and a real ``SIGKILL`` of a worker
process.

MOCKED (labelled): the memory service.  ``GovService`` replaces the mem0 /
Qdrant / Neo4j / Gemini stack: no LLM, no embedder, no vector or graph store.
It is file-backed so a killed worker's effects survive.  It models the legacy
service's rules (owner-only private rows, owner-only edits) and counts calls.
Because the legacy service has no tenant concept, shared-pool tenant isolation
is not tested here; the tests use private pools, where isolation is enforced by
the API-boundary policy and the fence.

Needs a disposable Redis (``V1_TEST_REDIS_URL``, else database 14 of ``REDIS_URL``; the tests FLUSH it).
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import multiprocessing
import os
import signal
import time
from pathlib import Path

import httpx
import pytest

import main
import policy_enforcement
from authority_store_sql import SqliteAuthorityStore
from config import settings
from contracts_lifecycle import IntentStatus, StageStatus
from contracts_policy import PolicyStatement
from identity_binding import bind_identity
from ledger.sqlite import SqliteLedger
from policy_enforcement import PolicyEnforcer, pool_handle
from publication_fence import PublicationFence
from schemas import MemoryResponse
from tokens import issue_user_token
from write_producer import (
    LedgerWriteWorker,
    MemoryRowLookup,
    MemorySourceResolver,
    SqlitePayloadStore,
    ledger_queue_name,
)

from ledger_env import REDIS_URL, require_env  # noqa: E402

@pytest.fixture(autouse=True)
def _fail_when_required_store_is_missing():
    """Missing Redis is a failure on the gate (NS_INTEGRATION_REQUIRED) and a skip elsewhere."""
    require_env(**{"Redis (V1_TEST_REDIS_URL or REDIS_URL)": REDIS_URL})

SECRET = "v1-integration-secret"
CONTENT_ACTIONS = ("read", "search", "list", "create", "update", "delete", "background_derive")


class GovService:
    """MOCK memory service (file-backed; see the module docstring)."""

    def __init__(self, state: str, block: str | None = None) -> None:
        self.state, self.block = state, block
        if not Path(state).exists():
            Path(state).write_text(json.dumps({"memories": {}, "store": 0, "graph": [], "patch": 0, "deleted": [], "narrow": None}))

    def _locked(self, fn):
        with open(self.state + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(Path(self.state).read_text())
            out = fn(data)
            Path(self.state).write_text(json.dumps(data))
            return out

    def read(self) -> dict:
        return json.loads(Path(self.state).read_text())

    def narrow_when_storing(self, content: str) -> None:
        self._locked(lambda d: d.update(narrow=content))

    @staticmethod
    def _response(mid: str, row: dict) -> MemoryResponse:
        return MemoryResponse(
            id=mid, memory=row["content"], visibility=row["visibility"],
            owner_user_id=row["owner"], project_id=row.get("project_id"),
        )

    def store_raw(self, **kw):
        mid = "mem-" + hashlib.sha1(f"{kw['user_id']}|{kw['content']}".encode()).hexdigest()[:10]

        def op(d):
            d["store"] += 1
            created = mid not in d["memories"]
            d["memories"][mid] = {
                "content": kw["content"], "owner": kw["user_id"],
                "visibility": kw.get("visibility") or "private", "project_id": kw.get("project_id"),
            }
            if d.get("narrow") == kw["content"]:  # an external privacy change lands right after the write
                d["memories"][mid]["visibility"] = "private"
                d["narrow"] = None
            return created

        created = self._locked(op)
        if self.block:
            Path(self.block).write_text("blocked")
            time.sleep(120)
        row = self.read()["memories"][mid]
        return [self._response(mid, row)], created

    def get_memory(self, memory_id, caller_user_id=None):
        row = self.read()["memories"].get(memory_id)
        if row is None:
            return None
        if caller_user_id is not None and row["visibility"] == "private" and row["owner"] != caller_user_id:
            return None
        return self._response(memory_id, row)

    def search(self, *, query, user_id, **kw):
        out = []
        for mid, row in self.read()["memories"].items():
            if query.lower() in row["content"].lower() and row["owner"] == user_id:
                out.append(self._response(mid, row))
        return out

    def patch_memory(self, memory_id, caller_user_id, changes):
        def op(d):
            row = d["memories"].get(memory_id)
            if row is None:
                raise LookupError(memory_id)
            if row["owner"] != caller_user_id:
                raise PermissionError("owner only")
            d["patch"] += 1
            if "content" in changes:
                row["content"] = changes["content"]
            if "visibility" in changes:
                row["visibility"] = changes["visibility"]
            return {"memory": None, "graph_job": {"memory_id": memory_id} if {"content", "visibility"} & set(changes) else None, "graph": "reingest_pending"}

        return self._locked(op)

    def enrich_graph(self, **kw):
        self._locked(lambda d: d["graph"].append(kw["memory_id"]))
        return True

    def delete_memory(self, memory_id, caller_user_id=None):
        self._locked(lambda d: (d["memories"].pop(memory_id, None), d["deleted"].append(memory_id)))
        return {}


def _stmt(identifier, subject, action, handle):
    return PolicyStatement(id=identifier, effect="allow", subject_id=subject, credential_id=None, action=action, resource=handle)


def _child_worker(ledger_path: str, authority_path: str, state: str, block: str) -> None:
    ledger = SqliteLedger(ledger_path)
    payloads = SqlitePayloadStore(ledger_path + ".payloads")
    authority = SqliteAuthorityStore(authority_path)
    service = GovService(state, block=block)
    fence = PublicationFence(authority, MemorySourceResolver(service, ledger, payloads, authority))
    worker = LedgerWriteWorker(ledger, payloads, service, owner="child", lease_seconds=1.0, fence=fence)
    asyncio.run(worker.drain())


class World:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.ledger_path = str(root / "ledger.db")
        self.authority_path = str(root / "authority.db")
        self.state = str(root / "state.json")
        self.service = GovService(self.state)
        self.authority = SqliteAuthorityStore(self.authority_path)
        self.ids: dict[str, str] = {}
        for user, tenant in (("alice", "tenant-a"), ("carol", "tenant-a"), ("bob", "tenant-b")):
            subject = bind_identity(self.authority, "local", user).subject_id
            self.authority.add_subject(subject, tenant, "human", f"cred-{user}")
            self.ids[user] = subject
        self.publish_policy()
        self.tokens = {u: issue_user_token(u, SECRET, 3600) for u in self.ids}

    def publish_policy(self) -> None:
        n = iter(range(1000))
        for tenant, members in (("tenant-a", ("alice", "carol")), ("tenant-b", ("bob",))):
            statements = []
            for user in members:
                for action in CONTENT_ACTIONS:
                    statements.append(_stmt(f"s{next(n)}", self.ids[user], action, pool_handle(tenant, "private", self.ids[user])))
            if tenant == "tenant-a":
                shared = pool_handle(tenant, "shared", None)
                for action in ("create", "share", "background_derive"):
                    statements.append(_stmt(f"s{next(n)}", self.ids["alice"], action, shared))
            self.authority.publish_policy(tenant, statements)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "redis_url", REDIS_URL)
    monkeypatch.setattr(settings, "ledger_enabled", True)
    monkeypatch.setattr(settings, "ledger_backend", "sqlite")
    monkeypatch.setattr(settings, "ledger_sqlite_path", str(tmp_path / "ledger.db"))
    monkeypatch.setattr(settings, "authority_enabled", True)
    monkeypatch.setattr(settings, "authority_backend", "sqlite")
    monkeypatch.setattr(settings, "authority_sqlite_path", str(tmp_path / "authority.db"))
    monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
    monkeypatch.setattr(settings, "ns_auth_profile", "enforced")
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    w = World(tmp_path)
    monkeypatch.setattr(main, "_service", w.service)
    monkeypatch.setattr(main, "_meter_full_recall_bg", lambda *a, **k: None)
    policy_enforcement.configure(
        PolicyEnforcer(w.authority, w.authority, row_lookup=MemoryRowLookup(w.service, w.authority))
    )
    yield w
    policy_enforcement.configure(None)
    w.authority.close()


async def connected_manager():
    from task_manager import TaskManager

    manager = TaskManager()
    await manager.connect()
    await manager.pool.flushdb()
    return manager


async def run_arq_worker(world: World, manager) -> None:
    """A real ARQ worker on the ledger queue (burst), with the fence enabled."""
    from arq.worker import Worker

    import worker as worker_module
    from config import parse_redis_settings

    ledger = SqliteLedger(world.ledger_path)
    payloads = SqlitePayloadStore(world.ledger_path + ".payloads")
    authority = SqliteAuthorityStore(world.authority_path)
    runner = Worker(
        functions=[worker_module.process_ledger_work],
        redis_settings=parse_redis_settings(),
        queue_name=ledger_queue_name(),
        burst=True, max_jobs=2, poll_delay=0.1, handle_signals=False,
        ctx={"ledger": ledger, "ledger_payloads": payloads, "authority": authority, "service": world.service},
    )
    try:
        await runner.async_run()
    finally:
        await runner.close()
        payloads.close()
        ledger.close()
        authority.close()


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test")


def auth(world: World, user: str, **headers) -> dict:
    return {"Authorization": f"Bearer {world.tokens[user]}", **headers}


def policy_denied(response: httpx.Response) -> bool:
    detail = response.json().get("detail") if response.headers.get("content-type", "").startswith("application/json") else None
    return response.status_code in (403, 404, 503) and isinstance(detail, dict) and "error" in detail


async def status(manager, task_id, user):
    return await manager.get_status(task_id, user_id=user)


@pytest.mark.asyncio
async def test_v1_governed_memory_lifecycle(world, monkeypatch) -> None:
    manager = await connected_manager()
    monkeypatch.setattr(main, "_task_manager", manager)
    try:
        async with client() as http:
            # 1. write -> accepted (not applied) -> applied by the worker
            write = await http.post(
                "/v1/memories/raw", headers=auth(world, "alice"),
                json={"content": "alice prefers dark mode", "category": "preference", "visibility": "private"},
            )
            assert write.status_code == 202, write.text
            task = write.json()["task_id"]
            assert (await status(manager, task, "alice"))["status"] == "queued"
            await run_arq_worker(world, manager)
            applied = await status(manager, task, "alice")
            assert applied["status"] == "completed"
            receipt = applied["result"]["receipt"]
            assert receipt["status"] == "applied" and [s["state"] for s in receipt["stages"]] == ["applied", "applied"]
            memory_id = receipt["outputs"][0]["id"]
            assert receipt["outputs"][0]["revision"] == 1

            # receipts are scoped to the writer: another tenant and another
            # member of the same tenant both get not_found
            assert (await status(manager, task, "bob"))["status"] == "not_found"
            assert (await status(manager, task, "carol"))["status"] == "not_found"

            # 2. recall: owner sees it; other tenant and same-tenant non-owner do not
            hit = await http.post("/v1/search", headers=auth(world, "alice"), json={"query": "dark mode"})
            assert [r["memory"] for r in hit.json()["results"]] == ["alice prefers dark mode"]
            for other in ("bob", "carol"):
                miss = await http.post("/v1/search", headers=auth(world, other), json={"query": "dark mode"})
                assert miss.status_code == 200 and miss.json()["results"] == []
                direct = await http.get(f"/v1/memories/{memory_id}", headers=auth(world, other))
                assert policy_denied(direct) and direct.json()["detail"]["error"]["code"] == "not_found"
            assert (await http.get(f"/v1/memories/{memory_id}", headers=auth(world, "alice"))).status_code == 200

            # 3. governed correction at an expected revision
            denied = await http.patch(f"/v1/memories/{memory_id}", headers=auth(world, "carol", **{"If-Match": '"1"'}), json={"content": "carol edit"})
            assert policy_denied(denied)
            missing = await http.patch(f"/v1/memories/{memory_id}", headers=auth(world, "alice"), json={"content": "x"})
            assert missing.status_code == 428  # an expected revision is required
            edit = await http.patch(
                f"/v1/memories/{memory_id}", headers=auth(world, "alice", **{"If-Match": '"1"'}),
                json={"content": "alice prefers light mode"},
            )
            assert edit.status_code == 202 and edit.json()["revision"] == 2
            before = await http.post("/v1/search", headers=auth(world, "alice"), json={"query": "mode"})
            assert [r["memory"] for r in before.json()["results"]] == ["alice prefers dark mode"]  # not applied yet
            await run_arq_worker(world, manager)
            edit_status = await status(manager, edit.json()["task_id"], "alice")
            assert edit_status["result"]["receipt"]["status"] == "applied"
            assert edit_status["result"]["receipt"]["outputs"][0]["revision"] == 2

            # recall returns only the current revision
            after = await http.post("/v1/search", headers=auth(world, "alice"), json={"query": "mode"})
            assert [r["memory"] for r in after.json()["results"]] == ["alice prefers light mode"]
            assert (await http.post("/v1/search", headers=auth(world, "alice"), json={"query": "dark"})).json()["results"] == []

            # 4. stale revision replay is a 409 and writes nothing
            patches_before = world.service.read()["patch"]
            stale = await http.patch(
                f"/v1/memories/{memory_id}", headers=auth(world, "alice", **{"If-Match": '"1"'}),
                json={"content": "stale overwrite"},
            )
            assert stale.status_code == 409
            assert stale.json()["detail"]["error"] == {
                "code": "stale_revision", "message": "The memory changed since that revision; re-read it and retry.",
                "retryable": False, "current_revision": 2,
            }
            assert manager.ledger.scan_unqueued() == ()
            await run_arq_worker(world, manager)
            assert world.service.read()["patch"] == patches_before
            assert world.service.read()["memories"][memory_id]["content"] == "alice prefers light mode"

            # 5. privacy narrowing between stages: the queued graph derivative is denied
            world.service.narrow_when_storing("team notes")
            shared = await http.post(
                "/v1/memories/raw", headers=auth(world, "alice"),
                json={"content": "team notes", "category": "preference", "visibility": "shared"},
            )
            assert shared.status_code == 202, shared.text
            await run_arq_worker(world, manager)
            narrowed = (await status(manager, shared.json()["task_id"], "alice"))["result"]["receipt"]
            assert narrowed["status"] == "partial"
            states = {s["stage"]: (s["state"], s["error_code"]) for s in narrowed["stages"]}
            assert states["vector"] == ("applied", None)
            assert states["graph"] == ("failed", "policy_namespace_changed")
            assert not [m for m in world.service.read()["graph"] if m.endswith(narrowed["outputs"][0]["id"][-10:])]

            # 6. credential retired while a job is queued: the job is denied, nothing published
            queued = await http.post(
                "/v1/memories/raw", headers=auth(world, "carol"),
                json={"content": "carol private note", "category": "preference", "visibility": "private"},
            )
            assert queued.status_code == 202, queued.text
            store_calls = world.service.read()["store"]
            world.authority.retire_credential("cred-carol")
            await run_arq_worker(world, manager)
            intent = manager.ledger.get_intent("tenant-a", queued.json()["task_id"])
            assert intent.status is IntentStatus.FAILED
            receipts = manager.ledger.receipts("tenant-a", queued.json()["task_id"])
            vector = next(r for r in receipts if r.stage.value == "vector")
            assert vector.status is StageStatus.FAILED and vector.error.code == "policy_credential_retired"
            assert world.service.read()["store"] == store_calls  # the provider was never called
            assert all("carol private note" != m["content"] for m in world.service.read()["memories"].values())
            after_retire = await http.post("/v1/search", headers=auth(world, "carol"), json={"query": "x"})
            assert policy_denied(after_retire)  # and her next request is denied too
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_v1_governed_worker_killed_mid_stage_completes_exactly_once(world, monkeypatch) -> None:
    manager = await connected_manager()
    monkeypatch.setattr(main, "_task_manager", manager)
    block = str(world.root / "blocked")
    try:
        async with client() as http:
            write = await http.post(
                "/v1/memories/raw", headers=auth(world, "bob"),
                json={"content": "bob prefers tea", "category": "preference", "visibility": "private"},
            )
            assert write.status_code == 202
            task = write.json()["task_id"]
        child = multiprocessing.get_context("spawn").Process(
            target=_child_worker, args=(world.ledger_path, world.authority_path, world.state, block)
        )
        child.start()
        deadline = time.monotonic() + 30
        while not Path(block).exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        assert Path(block).exists(), "child never reached the vector stage"
        os.kill(child.pid, signal.SIGKILL)
        child.join(10)
        assert child.exitcode == -signal.SIGKILL
        mid = await status(manager, task, "bob")
        assert mid["status"] == "processing"  # the effect exists; no applied receipt yet

        await asyncio.sleep(1.4)  # the dead worker's 1s lease expires
        survivor_ledger = SqliteLedger(world.ledger_path)
        payloads = SqlitePayloadStore(world.ledger_path + ".payloads")
        authority = SqliteAuthorityStore(world.authority_path)
        try:
            service = GovService(world.state)
            fence = PublicationFence(authority, MemorySourceResolver(service, survivor_ledger, payloads, authority))
            worker = LedgerWriteWorker(survivor_ledger, payloads, service, owner="survivor", lease_seconds=30.0, fence=fence)
            assert await worker.drain() == 1
        finally:
            payloads.close()
            survivor_ledger.close()
            authority.close()

        done = await status(manager, task, "bob")
        receipt = done["result"]["receipt"]
        assert done["status"] == "completed" and receipt["status"] == "applied"
        assert [s["state"] for s in receipt["stages"]] == ["applied", "applied"]
        data = world.service.read()
        assert len([m for m in data["memories"].values() if m["content"] == "bob prefers tea"]) == 1
        applied = [r for r in manager.ledger.receipts("tenant-b", task) if r.status is StageStatus.APPLIED]
        assert sorted(r.stage.value for r in applied) == ["graph", "vector"]
        assert manager.ledger.effect_recorded("tenant-b", f"vector:{task}")
        assert manager.ledger.claim("anyone", 5.0, tenant_id="tenant-b") is None
    finally:
        await manager.close()
