"""Online cutover: phase machine, persistence, race safety, fences and routing."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ledger.backfill import Backfill, LegacyRow, single_tenant_resolver
from ledger.cutover import (
    FENCED_RESULT,
    CutoverConflict,
    CutoverController,
    CutoverError,
    Phase,
    SequencedLog,
    legacy_admission,
)
from ledger.sqlite import SqliteLedger
from tests.test_ledger_backfill import FakeScanner, row
from write_producer import SqlitePayloadStore

TENANT = "tenant-default"


class Clock:
    now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def env(tmp_path):
    ledger = SqliteLedger(str(tmp_path / "ledger.db"))
    kv = SqlitePayloadStore(str(tmp_path / "kv.db"))
    yield ledger, kv
    kv.close()
    ledger.close()


def build(env, rows=None, inflight=None, clock=None):
    ledger, kv = env
    scanner = FakeScanner(rows if rows is not None else [row(i) for i in range(6)])
    backfill = Backfill(ledger, scanner, single_tenant_resolver(TENANT), kv, page_size=4)
    queue = SimpleNamespace(n=0)
    controller = CutoverController(
        kv, backfill=backfill, legacy_inflight=inflight or (lambda: queue.n), clock=clock or Clock(), cache_seconds=0
    )
    return controller, backfill, scanner, queue


def test_the_sequenced_log_finds_its_end_for_any_length(env) -> None:
    _, kv = env
    log = SequencedLog(kv, "probe")
    assert log.length() == 0
    for n in range(1, 40):
        log.append(n - 1, {"n": n})
        assert log.length() == n and log.latest()[1]["n"] == n


def test_two_controllers_cannot_both_take_the_next_phase(env) -> None:
    _, kv = env
    log_a, log_b = SequencedLog(kv, "race"), SequencedLog(kv, "race")
    log_a.append(0, {"who": "a"})
    with pytest.raises(CutoverConflict):
        log_b.append(0, {"who": "b"})
    assert log_a.latest()[1]["who"] == "a"


def test_the_full_path_legacy_dual_ledger_only_complete(env) -> None:
    controller, backfill, _, _ = build(env)
    assert controller.state().phase is Phase.LEGACY and not controller.writes_use_ledger()
    assert controller.start().phase is Phase.DUAL_ACCEPT and controller.writes_use_ledger()
    assert controller.start().phase is Phase.DUAL_ACCEPT  # idempotent
    ledger_only = controller.advance()
    assert ledger_only.phase is Phase.LEDGER_ONLY and ledger_only.fence_at == Clock.now
    assert backfill.reconcile().ok
    assert controller.complete().phase is Phase.COMPLETE
    assert controller.state(fresh=True).phase is Phase.COMPLETE and controller.complete().phase is Phase.COMPLETE


def test_transitions_out_of_order_are_refused(env) -> None:
    controller, _, _, _ = build(env)
    with pytest.raises(CutoverError):
        controller.advance()
    with pytest.raises(CutoverError):
        controller.complete()
    controller.start()
    with pytest.raises(CutoverError):
        controller.complete()


def test_advance_waits_for_the_legacy_queue_to_drain(env) -> None:
    controller, _, _, queue = build(env)
    controller.start()
    queue.n = 2
    with pytest.raises(CutoverError, match="drain"):
        controller.advance()
    assert controller.state().phase is Phase.DUAL_ACCEPT  # nothing moved
    queue.n = 0
    assert controller.advance().phase is Phase.LEDGER_ONLY


def test_a_legacy_job_that_starts_during_reconciliation_blocks_the_move(env) -> None:
    calls = iter([0, 1])  # drained at the first check, running at the second
    controller, _, _, _ = build(env, inflight=lambda: next(calls))
    controller.start()
    with pytest.raises(CutoverError, match="during reconciliation"):
        controller.advance()
    assert controller.state().phase is Phase.DUAL_ACCEPT


def test_an_unclean_reconciliation_blocks_advance_and_complete(env) -> None:
    controller, _, scanner, _ = build(env, rows=[row(1), row(2)])
    scanner.rows.append(row(2))  # the legacy store reports a duplicate row
    controller.start()
    with pytest.raises(CutoverError, match="not clean"):
        controller.advance()


def test_complete_is_blocked_by_a_row_the_final_reconcile_finds_missing(env) -> None:
    controller, _, scanner, _ = build(env)
    controller.start()
    controller.advance()
    scanner.rows.append(row(77))  # a straggler legacy write after the fence
    with pytest.raises(CutoverError, match="final reconciliation"):
        controller.complete()
    controller.backfill_step()
    assert controller.complete().phase is Phase.COMPLETE


@pytest.mark.parametrize("stop_after", ["start", "advance"])
def test_rollback_before_complete_returns_writes_to_legacy_and_loses_nothing(env, stop_after) -> None:
    ledger, _ = env
    controller, backfill, _, _ = build(env)
    controller.start()
    if stop_after == "advance":
        controller.advance()
    registered = backfill.report().counts.registered
    state = controller.rollback()
    assert state.phase is Phase.LEGACY and state.rolled_back and not controller.writes_use_ledger()
    assert ledger.get_object_ref(TENANT, "mem-000", 1) is not None or registered == 0  # canonical refs stay
    # a second cutover attempt reconciles again before it can pass the fence
    controller.start()
    assert controller.advance().phase is Phase.LEDGER_ONLY


def test_rollback_after_complete_is_refused(env) -> None:
    controller, _, _, _ = build(env)
    controller.start()
    controller.advance()
    controller.complete()
    with pytest.raises(CutoverError, match="refused after complete"):
        controller.rollback()
    assert controller.state(fresh=True).phase is Phase.COMPLETE


@pytest.mark.parametrize("phase", ["legacy", "dual_accept", "ledger_only", "complete"])
def test_a_new_controller_resumes_at_the_persisted_phase(env, phase) -> None:
    ledger, kv = env
    controller, _, _, _ = build(env)
    if phase != "legacy":
        controller.start()
    if phase in ("ledger_only", "complete"):
        controller.advance()
    if phase == "complete":
        controller.complete()
    restarted, _, _, _ = build(env)
    assert restarted.state(fresh=True).phase.value == phase
    assert restarted.status()["phase"] == phase


def test_legacy_admission_fences_only_what_arrives_after_the_fence() -> None:
    clock = Clock()
    store: dict = {}

    class Kv:
        def put(self, key, payload):
            store.setdefault(key, payload)

        def get(self, key):
            return store.get(key)

    controller = CutoverController(Kv(), clock=clock, cache_seconds=0)
    assert legacy_admission(controller, 5.0) == (True, "")
    assert legacy_admission(None, None) == (True, "")  # cutover not enabled
    controller.start()
    assert legacy_admission(controller, None) == (True, "")
    # move to ledger-only without a backfill: drive the log directly
    state = controller.state(fresh=True)
    controller._move(state, Phase.LEDGER_ONLY, fence_at=clock.now)
    before, after = clock.now - 1, clock.now + 1
    assert legacy_admission(controller, before) == (True, "")  # admitted before the fence: completes
    assert legacy_admission(controller, after) == (False, "legacy_write_after_cutover")
    assert legacy_admission(controller, None)[0] is False  # unknown admission time is not provably safe


def _ctx(controller, enqueued: float | None):
    ctx = {"cutover": controller}
    if enqueued is not None:
        ctx["enqueue_time"] = datetime.fromtimestamp(enqueued, timezone.utc)
    return ctx


def test_a_fenced_legacy_worker_job_reports_an_explicit_status_and_does_not_run(env) -> None:
    import worker as worker_module

    ran = []

    async def legacy_job(ctx, payload):
        ran.append(payload)
        return {"status": "done"}

    guarded = worker_module._cutover_fenced(legacy_job)
    controller, _, _, _ = build(env)
    controller.start()
    controller.advance()
    fence = controller.state(fresh=True).fence_at
    result = asyncio.run(guarded(_ctx(controller, fence + 5), {"n": 1}))
    assert result == FENCED_RESULT and ran == []
    assert asyncio.run(guarded(_ctx(controller, fence - 5), {"n": 2})) == {"status": "done"} and ran == [{"n": 2}]
    assert guarded.__wrapped__ is legacy_job  # the ledger's own stages call the unwrapped body


def test_real_legacy_write_jobs_are_fenced_and_the_ledger_stage_bodies_are_not() -> None:
    import worker as worker_module

    for name in ("process_memory_store", "process_memory_raw", "process_memory_raw_batch", "process_memory_retag",
                 "process_ingest_document", "process_ingest_file", "process_ingest_okf_bundle", "process_code_index"):
        assert hasattr(getattr(worker_module, name), "__wrapped__"), name


def test_the_task_manager_routes_writes_by_phase(env) -> None:
    import task_manager

    controller, _, _, _ = build(env)
    manager = task_manager.TaskManager()
    assert manager._durable_route() is False  # no producer: legacy
    manager.producer = object()
    assert manager._durable_route() is True  # cutover not enabled: the ledger, as before
    manager.cutover = controller
    assert manager._durable_route() is False  # cutover enabled but not started
    controller.start()
    assert manager._durable_route() is True
    controller.advance()
    controller.complete()
    assert manager._durable_route() is True


def test_cutover_routes_are_404_unless_enabled_and_dictator_only(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    import main
    from config import settings

    client = TestClient(main.app, raise_server_exceptions=False)
    monkeypatch.setattr(settings, "dictator_user_ids", "default_user")
    monkeypatch.setattr(settings, "ledger_cutover_enabled", False)
    assert client.get("/v1/admin/ledger-cutover").status_code == 404
    monkeypatch.setattr(settings, "dictator_user_ids", "someone-else")
    assert client.post("/v1/admin/ledger-cutover/start").status_code == 403
