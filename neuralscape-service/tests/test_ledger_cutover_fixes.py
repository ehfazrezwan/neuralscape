"""Corrections to the backfill and cutover found by review: each case fails on the code before the fix."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ledger.backfill import Backfill, LegacyRow, baseline_marker, content_checksum, is_backfill_baseline
from ledger.cutover import FENCED_RESULT, CutoverController, CutoverError
from ledger.port import ObjectRef, ResurrectionDenied
from ledger.sqlite import SqliteLedger
from tests.test_ledger_backfill import FakeScanner, row
from write_producer import MemoryRevisions, SqlitePayloadStore

TENANT = "tenant-default"
OWNERS = {"alice": "tenant-default", "bob": "tenant-b"}


@pytest.fixture
def env(tmp_path):
    ledger = SqliteLedger(str(tmp_path / "ledger.db"))
    kv = SqlitePayloadStore(str(tmp_path / "kv.db"))
    yield ledger, kv
    kv.close()
    ledger.close()


def backfill(env, rows, resolver=None, **kw):
    ledger, kv = env
    scanner = FakeScanner(rows)
    return Backfill(ledger, scanner, resolver or (lambda owner: OWNERS.get(owner)), kv, page_size=4, **kw), scanner


# ------------------------------------------------------------------ an erasure racing the registration
class ErasingLedger:
    """Tombstones the row between the backfill's tombstone check and its registration."""

    def __init__(self, inner, victim):
        self._inner, self._victim = inner, victim

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def put_object_ref(self, ref):
        if ref.record_id == self._victim:
            self._inner.tombstone(ref.tenant_id, ref.record_id, 1, "user_delete")
        return self._inner.put_object_ref(ref)


def test_an_erasure_racing_the_registration_counts_the_row_as_erased_instead_of_aborting_the_pass(env) -> None:
    ledger, kv = env
    scanner = FakeScanner([row(i) for i in range(6)])
    racing = Backfill(ErasingLedger(ledger, "mem-002"), scanner, lambda owner: TENANT, kv, page_size=4)
    ledger.tombstone(TENANT, "mem-probe", 1, "user_delete")
    with pytest.raises(ResurrectionDenied):  # the ledger itself refuses a reference for an erased record
        ledger.put_object_ref(ObjectRef(TENANT, "mem-probe", 1, 0))
    report = racing.run()  # used to raise ResurrectionDenied
    assert report.done and report.counts.erased_skipped == 1 and report.counts.registered == 5
    assert not ledger.has_object_refs(TENANT, "mem-002")
    assert racing.reconcile().ok


# ------------------------------------------------------------------ reconcile compares content
def test_reconcile_reports_a_row_edited_behind_the_ledger_and_the_next_pass_rebaselines_it(env) -> None:
    ledger, _ = env
    bf, scanner = backfill(env, [row(1), row(2)])
    bf.run()
    scanner.rows[0] = LegacyRow(id="mem-001", owner_user_id="alice", content="edited after the baseline")
    report = bf.reconcile()
    assert not report.ok and report.drifted == ["mem-001"] and report.summary()["drifted"] == 1
    again = bf.run()
    assert again.counts.rebaselined == 1 and again.counts.registered == 0
    revision, latest = MemoryRevisions(ledger).latest(TENANT, "mem-001"), ledger.get_object_ref(TENANT, "mem-001", 2)
    assert revision == 2 and is_backfill_baseline(latest) and latest.checksum == content_checksum("edited after the baseline")
    assert bf.reconcile().ok
    assert bf.run().counts.rebaselined == 0  # idempotent


def test_a_stale_write_path_baseline_is_rebaselined_so_it_no_longer_blocks_the_cutover(env) -> None:
    ledger, kv = env
    MemoryRevisions(ledger).register_baseline(TENANT, "mem-001", content_checksum("what the ledger saw"), "intent-x")
    bf, _ = backfill(env, [row(1, content="changed by a legacy writer")])
    assert bf.run().counts.rebaselined == 1  # a revision-1 baseline holds only a checksum: it is re-baselined
    assert bf.reconcile().ok


def test_an_unedited_row_is_not_reported_as_drifted_after_ledger_edits_advance_its_revision(env) -> None:
    ledger, _ = env
    bf, _ = backfill(env, [row(1)])
    bf.run()
    ledger.put_object_ref(ObjectRef(TENANT, "mem-001", 2, 0, (), "sha256:an-edit-the-ledger-owns"))
    assert bf.reconcile().ok  # the ledger owns revision 2 now; only revision 1 and backfill baselines are compared


# ------------------------------------------------------------------ stray references need no tenant list
def test_a_stray_reference_in_another_tenant_is_found_without_naming_tenants(env) -> None:
    ledger, _ = env
    bf, _ = backfill(env, [row(1, owner="alice"), row(2, owner="bob")])  # built like the admin route: no extra_tenants
    bf.run()
    assert bf.reconcile().ok
    ledger.put_object_ref(ObjectRef("tenant-b", "mem-001", 1, 0, (baseline_marker("tenant-b"),), "sha256:stray"))
    report = bf.reconcile()
    assert report.wrong_tenant == ["mem-001"] and not report.ok


def test_a_reference_for_a_quarantined_owner_is_found_too(env) -> None:
    ledger, _ = env
    bf, _ = backfill(env, [row(1, owner="alice"), row(2, owner="bob"), row(3, owner="ghost")])
    bf.run()
    ledger.put_object_ref(ObjectRef(TENANT, "mem-003", 1, 0, (), "sha256:invented"))
    assert bf.reconcile().wrong_tenant == ["mem-003"]


# ------------------------------------------------------------------ the fence is visible to a poller
def test_a_fenced_job_reaches_the_polling_client_as_fenced_not_completed() -> None:
    from arq.jobs import JobStatus

    import task_manager

    manager = task_manager.TaskManager()

    class Job:
        async def result_info(self):
            return SimpleNamespace(success=True, result=dict(FENCED_RESULT))

    async def find(task_id):
        return Job(), JobStatus.complete

    manager._find_job = find  # type: ignore[method-assign]
    got = asyncio.run(manager.get_status("legacy-task"))
    assert got["status"] == "fenced" and got["error"] == "legacy_write_after_cutover"
    assert got["result"] == FENCED_RESULT

    class Done(Job):
        async def result_info(self):
            return SimpleNamespace(success=True, result={"status": "fenced", "other": "a job's own field"})

    async def find_done(task_id):
        return Done(), JobStatus.complete

    manager._find_job = find_done  # type: ignore[method-assign]
    assert asyncio.run(manager.get_status("legacy-task"))["status"] == "completed"  # only the fence's own marker counts


# ------------------------------------------------------------------ every legacy writer carries the fence
@pytest.mark.parametrize("name", ["process_conversation_compile", "process_conversation_flush", "process_session_summary"])
def test_the_remaining_legacy_queue_writers_honor_the_fence(name) -> None:
    import worker as worker_module

    assert hasattr(getattr(worker_module, name), "__wrapped__"), name


def test_the_backfill_route_answers_503_when_the_memory_service_is_not_initialized(monkeypatch, tmp_path) -> None:
    from fastapi.testclient import TestClient

    import main
    from config import settings

    controller = CutoverController(SqlitePayloadStore(str(tmp_path / "kv.db")), cache_seconds=0)

    class Broken:
        def _get_memory(self):
            raise RuntimeError("not initialized")

    manager = SimpleNamespace(cutover=controller, ledger=None, producer=SimpleNamespace(payloads=None),
                              authority=None, _legacy_inflight_sync=lambda: 0)
    monkeypatch.setattr(main, "_task_manager", manager)
    monkeypatch.setattr(main, "_service", Broken())
    monkeypatch.setattr(settings, "ledger_cutover_enabled", True)
    monkeypatch.setattr(settings, "dictator_user_ids", "default_user")
    response = TestClient(main.app, raise_server_exceptions=False).get("/v1/admin/ledger-cutover")
    assert response.status_code == 503


# ================================================================== second review round
# ---- waiting callers are told "fenced" too
async def fenced_job(ctx):  # module level: ARQ names a function by its qualified name
    return dict(FENCED_RESULT)


async def normal_job(ctx):
    return {"stored": 1}


def _redis_url():
    import os

    return os.environ.get("V1_TEST_REDIS_URL")


@pytest.mark.skipif(not _redis_url(), reason="V1_TEST_REDIS_URL not set (real ARQ and Redis)")
def test_wait_for_result_reports_a_fenced_job_as_fenced_on_real_arq(monkeypatch) -> None:
    import uuid

    from arq import create_pool
    from arq.worker import create_worker

    import task_manager
    from config import parse_redis_settings, settings

    queue = f"fenced-test-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(settings, "redis_url", _redis_url())
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


def test_get_status_and_wait_share_one_fence_mapping() -> None:
    import task_manager

    assert callable(task_manager._fenced_view)
    assert task_manager._fenced_view(dict(FENCED_RESULT)) == ("fenced", "legacy_write_after_cutover")
    assert task_manager._fenced_view({"status": "fenced"}) is None  # only the fence's own marker


# ---- revision-1 baselines converge
def test_a_revision_one_write_path_baseline_edited_behind_the_ledger_is_rebaselined_and_converges(env) -> None:
    ledger, kv = env
    MemoryRevisions(ledger).register_baseline(TENANT, "mem-001", content_checksum("what the ledger saw"), "intent-x")
    bf, _ = backfill(env, [row(1, content="changed by a legacy writer")])
    assert bf.reconcile().drifted == ["mem-001"]
    for _ in range(3):
        bf.run()
    assert bf.reconcile().ok and bf.reconcile().drifted == []  # used to stay drifted after any number of passes
    assert MemoryRevisions(ledger).latest(TENANT, "mem-001") == 2
    controller = CutoverController(kv, backfill=bf, legacy_inflight=lambda: 0, cache_seconds=0)
    controller.start()
    assert controller.advance().phase.value == "ledger_only"


def test_a_ledger_edit_revision_is_never_rebaselined_over(env) -> None:
    ledger, _ = env
    MemoryRevisions(ledger).register_baseline(TENANT, "mem-001", content_checksum("original"), "intent-x")
    ledger.put_object_ref(ObjectRef(TENANT, "mem-001", 2, 0, (), "sha256:a-ledger-owned-edit"))
    bf, _ = backfill(env, [row(1, content="anything")])
    assert bf.run().counts.rebaselined == 0 and MemoryRevisions(ledger).latest(TENANT, "mem-001") == 2


# ---- stray references, one tenant and unscanned tenants
def test_a_quarantined_rows_stray_reference_is_found_with_a_single_mapped_tenant(env) -> None:
    ledger, _ = env
    bf, _ = backfill(env, [row(1, owner="alice"), row(2, owner="ghost")], resolver=lambda owner: TENANT if owner == "alice" else None)
    bf.run()
    ledger.put_object_ref(ObjectRef(TENANT, "mem-002", 1, 0, (), "sha256:invented"))
    assert bf.reconcile().wrong_tenant == ["mem-002"]  # used to need a second tenant before it looked


def test_the_admin_wiring_feeds_the_authority_tenants_to_the_stray_check() -> None:
    import main

    assert callable(main._known_tenants)
    authority = SimpleNamespace(tenants=lambda: ("tenant-a", "tenant-b"))
    assert set(main._known_tenants(authority)) == {"tenant-a", "tenant-b"}
    assert main._known_tenants(None) == ()


def test_a_stray_in_an_unscanned_tenant_is_found_through_the_extra_tenants(env) -> None:
    ledger, _ = env
    bf, _ = backfill(env, [row(1, owner="alice")], extra_tenants=("tenant-elsewhere",))
    bf.run()
    ledger.put_object_ref(ObjectRef("tenant-elsewhere", "mem-001", 1, 0, (), "sha256:stray"))
    assert bf.reconcile().wrong_tenant == ["mem-001"]


# ---- an authority subclass cannot mint a token
def test_an_erasure_authority_subclass_is_refused() -> None:
    from ledger.erasure import ErasureAuthority

    with pytest.raises(TypeError):
        class Forged(ErasureAuthority):  # noqa: D401 - the class statement itself must fail
            pass


def test_a_look_alike_of_the_authority_cannot_mint_a_token(tmp_path) -> None:
    from ledger.erasure import issue_erasure_token
    from ledger.port import AuthorityDenied

    class Alike:
        tenant_id, principal_id, credential_id, witness_id = "t", "p", "c", "1:1"

    verifier = SimpleNamespace(verify_owner_erasure=lambda authority, record_id: True)
    with pytest.raises(AuthorityDenied):
        issue_erasure_token(verifier, Alike(), record_id="r", content_revision=1, intent_id="i")


# ---- the on-demand connector sync job carries the fence
def test_the_connector_sync_queue_job_honors_the_fence() -> None:
    import worker as worker_module

    assert hasattr(worker_module.process_connector_sync, "__wrapped__")
