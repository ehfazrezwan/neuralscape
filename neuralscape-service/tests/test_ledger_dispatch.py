"""Stage-set registry and dispatcher scenarios (SQLite here; PostgreSQL in the integration module)."""

from __future__ import annotations

import multiprocessing
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from contracts_lifecycle import IntentStatus, ProcessingStage, StageReceipt, StageStatus
from contracts_references import ReferenceHandle, SourceVersion
from ledger import stages as stage_registry
from ledger.dispatch import (
    Applied,
    Publisher,
    Skipped,
    StageContext,
    StageFailure,
    StageWorker,
)
from ledger.port import LedgerPort, StaleLease
from ledger.sqlite import SqliteLedger
from ledger.stages import REGISTRY, build_intent, build_requirements, stage_set

C, E, V, G, M = (
    ProcessingStage.CANONICAL,
    ProcessingStage.EXTRACT,
    ProcessingStage.VECTOR,
    ProcessingStage.GRAPH,
    ProcessingStage.MARKDOWN,
)
PRE = (SourceVersion(record_id="mem-1", content_revision=3, policy_epoch=7),)
POST = (SourceVersion(record_id="mem-1", content_revision=4, policy_epoch=7),)


# Generous guard for thread hand-offs: it only bounds how long a failing run takes, never how long a passing one
# may.  Leases are long (30 s and more) and their expiry is moved explicitly (``store.shift_leases``), so no scenario
# waits for wall-clock time to pass and a stalled runner cannot lapse a lease that a scenario still relies on.
WAIT = 120.0


class Crash(BaseException):
    """Simulated process death: not an Exception, so nothing in the worker swallows it."""


# ---------------------------------------------------------------- fake external store
class External:
    """Idempotent-by-effect-id external system shared across processes (SQLite file)."""

    def __init__(self, path: Path) -> None:
        self.path = str(path)
        conn = sqlite3.connect(self.path)
        conn.execute("CREATE TABLE IF NOT EXISTS effects(effect_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE IF NOT EXISTS calls(n INTEGER PRIMARY KEY AUTOINCREMENT, effect_id TEXT)")
        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def put(self, ctx: StageContext) -> tuple[ReferenceHandle, ...]:
        conn = self._conn()
        conn.execute("INSERT INTO calls(effect_id) VALUES(?)", (ctx.effect_id,))
        conn.execute("INSERT OR IGNORE INTO effects VALUES(?)", (ctx.effect_id,))
        conn.commit()
        conn.close()
        return (ReferenceHandle(kind="artifact", id=f"obj-{ctx.effect_id}"[:200], tenant_id=ctx.tenant_id, resolver="r"),)

    def has(self, effect_id: str) -> bool:
        conn = self._conn()
        row = conn.execute("SELECT 1 FROM effects WHERE effect_id=?", (effect_id,)).fetchone()
        conn.close()
        return row is not None

    def effects(self) -> int:
        conn = self._conn()
        n = conn.execute("SELECT count(*) FROM effects").fetchone()[0]
        conn.close()
        return n

    def calls(self, kind: str | None = None) -> int:
        conn = self._conn()
        sql, args = "SELECT count(*) FROM calls", ()
        if kind:
            sql, args = sql + " WHERE effect_id LIKE ?", (f"{kind}%",)
        n = conn.execute(sql, args).fetchone()[0]
        conn.close()
        return n


class Handler:
    def __init__(self, external: External, *, inspect: bool = True, hook: Callable[[StageContext], None] | None = None,
                 after_put: Callable[[StageContext], None] | None = None) -> None:
        self.external, self.hook, self.after_put = external, hook, after_put
        if not inspect:
            self.inspect = None  # type: ignore[assignment]

    def apply(self, ctx: StageContext) -> Applied:
        if self.hook:
            self.hook(ctx)
        refs = self.external.put(ctx)
        if self.after_put:
            self.after_put(ctx)
        return Applied(refs)

    def inspect(self, ctx: StageContext):  # noqa: D401
        if self.external.has(ctx.effect_id):
            return Applied((ReferenceHandle(kind="artifact", id=f"obj-{ctx.effect_id}"[:200], tenant_id=ctx.tenant_id, resolver="r"),))
        return None


def all_handlers(external: External, **overrides: Any) -> dict:
    handlers: dict = {stage: Handler(external) for stage in ProcessingStage}
    handlers.update(overrides)
    return handlers


def intent(operation: str = "memory.write.raw", name: str = "1", tenant: str = "tenant-1"):
    return build_intent(
        operation, intent_id=f"intent-{name}", tenant_id=tenant, actor_id="actor-1", credential_id="cred-1",
        idempotency_key=f"key-{name}", request_digest="sha256:a", correlation_id=f"corr-{name}",
        preconditions=PRE, committed=POST,
    )


# ------------------------------------------------------------------------------ stores
class SqliteStore:
    def __init__(self, root: Path) -> None:
        self.root, self.path, self.opened = root, root / "ledger.db", []

    def open(self) -> SqliteLedger:
        ledger = SqliteLedger(str(self.path))
        self.opened.append(ledger)
        return ledger

    open_admin = open

    def shift_leases(self, seconds: float) -> None:
        """Move every live lease's expiry by ``seconds`` (negative: as if that much time had passed)."""

        raw = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        raw.execute("UPDATE outbox SET lease_until = lease_until + ? WHERE state='leased'", (seconds,))
        raw.close()

    def external(self) -> External:
        return External(self.root / "external.db")

    def close(self) -> None:
        for ledger in self.opened:
            ledger.close()


# ------------------------------------------------------------------------- registry tests
def test_every_operation_builds_a_valid_intent_with_ordered_stages() -> None:
    assert len(REGISTRY) >= 18
    for operation, spec_set in REGISTRY.items():
        built = intent(operation)
        assert [r.stage for r in built.stage_requirements] == list(spec_set.stage_names())
        assert len({r.effect_id for r in built.stage_requirements}) == len(spec_set.stages)
        assert built.operation == operation
        for spec in spec_set.stages:
            assert set(spec.upstream) <= set(spec_set.stage_names()[: spec_set.stage_names().index(spec.stage)])


def test_registry_covers_every_worker_producer_and_marks_optional_projections() -> None:
    required_ops = {
        "memory.write.raw", "memory.write.extracted", "memory.edit", "memory.delete", "memory.retag",
        "memory.raw_batch", "ingest.text", "ingest.file", "ingest.okf", "connector.sync", "dream.sweep",
        "summary.session", "markdown.edit", "code.index", "conversation.compile", "memory.dedup",
        "memory.expire", "playbook.synthesize",
    }
    assert required_ops <= set(REGISTRY)
    assert stage_set("memory.raw_batch").children == "memory.write.raw"
    for op in ("memory.write.raw", "memory.edit", "dream.sweep"):
        optional = {s.stage for s in stage_set(op).stages if s.optional}
        assert optional == {G, M}
    assert not stage_set("memory.write.raw").stages[0].optional  # canonical commit is never optional
    with pytest.raises(stage_registry.UnknownOperation):
        stage_set("no.such.operation")


def test_requirements_keep_calculated_inputs_collision_free_and_ids_bounded() -> None:
    long_id = "x" * 250
    reqs = build_requirements("memory.edit", intent_id=long_id, preconditions=PRE, committed=POST)
    assert all(len(r.effect_id) <= 256 for r in reqs)
    assert len({r.effect_id for r in reqs}) == len(reqs)
    assert build_requirements("memory.edit", intent_id=long_id, preconditions=PRE, committed=POST) == reqs  # deterministic
    canonical = next(r for r in reqs if r.stage is C)
    assert canonical.committed_sources == POST and canonical.precondition_source_ids == ("mem-1",)
    assert all(r.precondition_source_ids == () for r in reqs if r.upstream_stages)


# --------------------------------------------------------------------------- scenarios
def drain(worker: StageWorker, limit: int = 20):
    results = []
    for _ in range(limit):
        got = worker.run_once()
        if got is None:
            break
        results.append(got)
    return results


def scenario_all_stages_apply(store) -> None:
    ledger, external = store.open(), store.external()
    for operation in REGISTRY:
        ledger.accept(intent(operation, name=operation))
    worker = StageWorker(ledger, all_handlers(external), owner="w")
    results = drain(worker, limit=len(REGISTRY) + 2)
    assert len(results) == len(REGISTRY) and {r.status for r in results} == {IntentStatus.APPLIED}
    total_stages = sum(len(s.stages) for s in REGISTRY.values())
    assert external.effects() == external.calls() == total_stages
    for operation in REGISTRY:
        stored = ledger.get_intent("tenant-1", f"intent-{operation}")
        assert stored.status is IntentStatus.APPLIED
        applied = {r.stage for r in ledger.receipts("tenant-1", stored.intent.id) if r.status is StageStatus.APPLIED}
        assert applied == set(stage_set(operation).stage_names())
    assert worker.run_once() is None


def scenario_disabled_optional_stages_are_skipped_not_applied(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    handlers = {C: Handler(external), V: Handler(external)}  # graph/markdown capability disabled
    result = StageWorker(ledger, handlers, owner="w").run_once()
    assert result.status is IntentStatus.PARTIAL  # never APPLIED
    assert result.outcomes[G] == "skipped (no handler)" and result.skip_reasons[G] == "capability disabled"
    statuses = {r.stage: r.status for r in ledger.receipts("tenant-1", "intent-1")}
    assert statuses[G] is statuses[M] is StageStatus.SKIPPED and statuses[C] is StageStatus.APPLIED
    assert external.effects() == 2  # only canonical + vector happened


def scenario_duplicate_queue_message_single_effect(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    queue: list = []
    boom = {"armed": True}

    def enqueue(entry) -> None:
        queue.append(entry)
        if boom["armed"]:
            boom["armed"] = False
            raise Crash  # died after enqueue, before mark_queued

    publisher = Publisher(ledger, enqueue)
    with pytest.raises(Crash):
        publisher.publish_unqueued()
    assert publisher.publish_unqueued() == 1  # recovery scan re-sends the same obligation
    assert len(queue) == 2 and queue[0] == queue[1]
    assert Publisher(ledger, queue.append).publish_unqueued() == 0
    workers = [StageWorker(store.open(), all_handlers(external), owner=f"w{i}") for i in range(2)]
    first = workers[0].run_entry(queue[0])
    second = workers[1].run_entry(queue[1])  # duplicate message: nothing left to claim
    assert first.status is IntentStatus.APPLIED and second is None
    assert external.effects() == external.calls() == len(stage_set("memory.write.raw").stages)


def scenario_kill_after_effect_before_receipt_reapplies_idempotently(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    state = {"died": False}

    def die_after_vector_put(ctx: StageContext) -> None:
        if ctx.stage is V and not state["died"]:
            state["died"] = True
            raise Crash

    handlers = all_handlers(external, **{V: Handler(external, after_put=die_after_vector_put)})
    with pytest.raises(Crash):
        StageWorker(ledger, handlers, owner="a", lease_seconds=30).run_once()
    assert external.has(next(r.effect_id for r in ledger.get_intent("tenant-1", "intent-1").intent.stage_requirements if r.stage is V))
    assert ledger.claim("probe", 5) is None  # still leased to the dead worker
    store.shift_leases(-60)  # the dead worker's lease has expired
    result = StageWorker(store.open(), all_handlers(external), owner="b", lease_seconds=5).run_once()
    assert result.status is IntentStatus.APPLIED and result.attempt == 2
    assert result.outcomes[C] == "applied (earlier attempt)"
    assert external.effects() == len(stage_set("memory.write.raw").stages)  # exactly once
    assert external.calls("vector_upsert") == 2  # re-applied once, deduplicated by effect id


class CrashingLedger:
    """Delegates to a real ledger and dies once, at a chosen point."""

    def __init__(self, inner: LedgerPort, *, on_receipt: Callable[[StageReceipt], bool] | None = None,
                 on_settle: bool = False) -> None:
        self._inner, self._on_receipt, self._on_settle, self.fired = inner, on_receipt, on_settle, False

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def record_receipt(self, receipt, tenant_id, claim=None):
        if self._on_receipt and not self.fired and self._on_receipt(receipt):
            self.fired = True
            raise Crash
        return self._inner.record_receipt(receipt, tenant_id, claim)

    def settle(self, *args, **kwargs):
        if self._on_settle and not self.fired:
            self.fired = True
            raise Crash
        return self._inner.settle(*args, **kwargs)


def scenario_kill_after_effect_recorded_recovers_without_reapply(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    crashing = CrashingLedger(
        store.open(), on_receipt=lambda r: r.stage is V and r.status is StageStatus.APPLIED
    )
    with pytest.raises(Crash):  # effect applied AND registered, applied receipt never written
        StageWorker(crashing, all_handlers(external), owner="a", lease_seconds=30).run_once()
    vector_effect = next(r.effect_id for r in ledger.get_intent("tenant-1", "intent-1").intent.stage_requirements if r.stage is V)
    assert ledger.effect_recorded("tenant-1", vector_effect)
    store.shift_leases(-60)  # the dead worker's lease has expired
    result = StageWorker(store.open(), all_handlers(external), owner="b", lease_seconds=5).run_once()
    assert result.status is IntentStatus.APPLIED
    assert external.calls("vector_upsert") == 1  # inspect() recovered it: no second apply
    assert external.effects() == len(stage_set("memory.write.raw").stages)


def _child_kill_during_stage(store, ready) -> None:
    external = store.external()

    def die(ctx: StageContext) -> None:
        if ctx.stage is G:
            os._exit(137)

    StageWorker(store.open(), all_handlers(external, **{G: Handler(external, after_put=die)}),
                owner="victim", lease_seconds=30).run_once()


def scenario_real_process_kill_during_stage_resumes_exactly_once(store) -> None:
    if sys.platform == "darwin":
        # A forked child that inherits an open SQLite connection segfaults on macOS (fork-unsafe system
        # libraries); the scenario is real on Linux, where CI and the gate run it.
        pytest.skip("os.fork after SQLite use is unsafe on macOS; runs on Linux")
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    child = multiprocessing.get_context("fork").Process(target=_child_kill_during_stage, args=(store, None))
    child.start()
    child.join()
    assert child.exitcode == 137
    assert ledger.get_intent("tenant-1", "intent-1").status is IntentStatus.PROCESSING
    store.shift_leases(-60)  # the killed process's lease has expired
    result = StageWorker(store.open(), all_handlers(external), owner="rescuer").run_once()
    assert result.status is IntentStatus.APPLIED and result.attempt == 2
    assert external.effects() == len(stage_set("memory.write.raw").stages)
    assert external.calls("canonical_commit") == 1 and external.calls("graph_episode") == 2


def scenario_stale_claim_is_fenced(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    entered, release = threading.Event(), threading.Event()

    def slow(ctx: StageContext) -> None:
        if ctx.stage is V:
            entered.set()
            release.wait(WAIT)

    slow_worker = StageWorker(store.open(), all_handlers(external, **{V: Handler(external, hook=slow)}),
                              owner="slow", lease_seconds=30)
    holder: list = []
    thread = threading.Thread(target=lambda: holder.append(slow_worker.run_once()))
    thread.start()
    assert entered.wait(WAIT)
    store.shift_leases(-60)  # slow's lease expires while its handler is stuck
    fast = StageWorker(store.open(), all_handlers(external), owner="fast", lease_seconds=5).run_once()
    assert fast.status is IntentStatus.APPLIED and fast.attempt == 2
    release.set()
    thread.join(WAIT)
    assert holder[0].lost_lease is True  # its receipts were refused by the fence
    stored = ledger.get_intent("tenant-1", "intent-1")
    assert stored.status is IntentStatus.APPLIED
    assert {r.attempt for r in ledger.receipts("tenant-1", "intent-1") if r.status is StageStatus.APPLIED} <= {1, 2}
    assert external.effects() == len(stage_set("memory.write.raw").stages)
    with pytest.raises(StaleLease):  # the slow worker's old token is dead for good
        ledger.complete(_stale_claim(ledger))


def _stale_claim(ledger: LedgerPort):
    from ledger.port import Claim

    stored = ledger.get_intent("tenant-1", "intent-1")
    return Claim(1, stored.intent.id, "tenant-1", "slow", 1, stored.accepted_at)


def scenario_mixed_failed_and_pending_stays_processing_until_settle(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    seen: dict = {}
    entered, release = threading.Event(), threading.Event()

    def hold_graph(ctx: StageContext) -> None:
        if ctx.stage is G:
            entered.set()
            assert release.wait(WAIT)

    def fail_vector(ctx: StageContext) -> None:
        if ctx.stage is V:
            raise StageFailure("vector_down", "vector store unavailable")

    handlers = all_handlers(external, **{V: Handler(external, hook=fail_vector), G: Handler(external, hook=hold_graph)})
    out: list = []
    thread = threading.Thread(target=lambda: out.append(StageWorker(store.open(), handlers, owner="w").run_once()))
    thread.start()
    assert entered.wait(WAIT)
    observer = store.open()
    stored = observer.get_intent("tenant-1", "intent-1")
    receipts = {r.stage: r.status for r in observer.receipts("tenant-1", "intent-1")}
    assert receipts[V] is StageStatus.FAILED and receipts[G] is StageStatus.PROCESSING  # mixed failed + running
    assert stored.status is IntentStatus.PROCESSING  # nonterminal until everything settles
    assert observer.derive_status("tenant-1", "intent-1") is IntentStatus.PROCESSING
    release.set()
    thread.join(WAIT)
    result = out[0]
    assert result.status is IntentStatus.PARTIAL  # canonical applied; vector failed; graph/markdown skipped upstream or applied
    assert result.outcomes[V] == "failed (vector_down)"
    assert observer.get_intent("tenant-1", "intent-1").status is IntentStatus.PARTIAL


def scenario_retryable_failure_retries_only_unfinished_stages(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    flaky = {"left": 1}

    def maybe_fail(ctx: StageContext) -> None:
        if ctx.stage is V and flaky["left"]:
            flaky["left"] -= 1
            raise StageFailure("vector_busy", "vector store busy", retryable=True)

    worker = StageWorker(ledger, all_handlers(external, **{V: Handler(external, hook=maybe_fail)}), owner="w")
    first = worker.run_once()
    assert first.released and first.status is IntentStatus.PROCESSING and first.attempt == 1
    second = worker.run_once()
    assert second.status is IntentStatus.APPLIED and second.attempt == 2 and not second.released
    assert external.calls("canonical_commit") == 1  # applied stages never re-run
    assert external.effects() == len(stage_set("memory.write.raw").stages)


def scenario_non_retryable_failure_is_terminal_and_downstream_skipped(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))

    def reject(ctx: StageContext) -> None:
        raise StageFailure("canonical_conflict", "revision conflict")

    result = StageWorker(ledger, all_handlers(external, **{C: Handler(external, hook=reject)}), owner="w").run_once()
    assert result.status is IntentStatus.FAILED and result.attempt == 1
    statuses = {r.stage: r.status for r in ledger.receipts("tenant-1", "intent-1")}
    assert statuses[C] is StageStatus.FAILED and statuses[V] is statuses[G] is statuses[M] is StageStatus.SKIPPED
    assert external.effects() == 0 and worker_idle(ledger)


def worker_idle(ledger: LedgerPort) -> bool:
    return ledger.claim("probe", 5) is None


def scenario_unexpected_error_is_failed_without_leaking_details(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))

    def explode(ctx: StageContext) -> None:
        raise RuntimeError("password=hunter2 stack detail")

    worker = StageWorker(ledger, all_handlers(external, **{C: Handler(external, hook=explode)}), owner="w", max_attempts=1)
    result = worker.run_once()
    assert result.status is IntentStatus.FAILED
    failed = next(r for r in ledger.receipts("tenant-1", "intent-1") if r.stage is C)
    assert failed.error.code == "handler_error" and "hunter2" not in failed.error.safe_message
    assert failed.error.retryable is True


def scenario_exhausted_retries_settle_terminal(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))

    def always(ctx: StageContext) -> None:
        if ctx.stage is V:
            raise StageFailure("vector_busy", "busy", retryable=True)

    worker = StageWorker(ledger, all_handlers(external, **{V: Handler(external, hook=always)}), owner="w", max_attempts=3)
    runs = drain(worker)
    assert [r.attempt for r in runs] == [1, 2, 3] and [r.released for r in runs] == [True, True, False]
    assert runs[-1].status is IntentStatus.PARTIAL  # canonical applied, vector failed for good
    assert worker.run_once() is None


def scenario_settled_failure_survives_a_crash_before_settle(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))

    def reject(ctx: StageContext) -> None:
        if ctx.stage is V:
            raise StageFailure("vector_rejected", "rejected")

    crashing = CrashingLedger(store.open(), on_settle=True)
    handlers = all_handlers(external, **{V: Handler(external, hook=reject)})
    with pytest.raises(Crash):
        StageWorker(crashing, handlers, owner="a", lease_seconds=30).run_once()
    store.shift_leases(-60)  # the crashed worker's lease has expired
    calls_before = external.calls("vector_upsert")
    result = StageWorker(store.open(), handlers, owner="b").run_once()
    assert result.attempt == 2 and result.status is IntentStatus.PARTIAL
    assert external.calls("vector_upsert") == calls_before  # a settled failure is not retried
    assert result.outcomes[V].endswith("earlier attempt)")


class RenewalCounter:
    """Delegating ledger that counts successful lease renewals and lets a scenario wait for the next ones."""

    def __init__(self, inner: LedgerPort) -> None:
        self._inner, self._renewed, self._refused, self._changed = inner, 0, 0, threading.Condition()

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def extend_lease(self, claim, lease_seconds):
        try:
            renewed = self._inner.extend_lease(claim, lease_seconds)
        except StaleLease:
            with self._changed:
                self._refused += 1
                self._changed.notify_all()
            raise
        with self._changed:
            self._renewed += 1
            self._changed.notify_all()
        return renewed

    def refused(self) -> bool:
        """Wait for a refused renewal (the lease had lapsed); ``False`` if none comes."""

        with self._changed:
            return self._changed.wait_for(lambda: self._refused > 0, timeout=WAIT)

    def renewals_after(self, count: int) -> bool:
        """Wait for ``count`` renewals beyond those already seen; ``False`` if they never come."""

        with self._changed:
            target = self._renewed + count
            return self._changed.wait_for(lambda: self._renewed >= target, timeout=WAIT)


def scenario_heartbeat_keeps_a_long_stage_leased(store) -> None:
    """A stage that outlives its lease keeps it: each beat renews the lease from the ledger's clock.

    The stored expiry is moved forward three times by two thirds of a 300 s lease (two whole lease lengths in
    total, with 100 s of margin after each move), and after each move the scenario waits for the heartbeat to renew it (the second renewal after
    the move started after it).  Without renewals the lease would have expired twice over, so a thief could
    claim it; nothing depends on how fast the runner is."""

    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    entered, finish = threading.Event(), threading.Event()

    def long_stage(ctx: StageContext) -> None:
        if ctx.stage is V:
            entered.set()
            finish.wait(WAIT)

    renewals = RenewalCounter(store.open())
    worker = StageWorker(renewals, all_handlers(external, **{V: Handler(external, hook=long_stage)}),
                         owner="w", lease_seconds=300, heartbeat_seconds=0.05)
    out: list = []
    thread = threading.Thread(target=lambda: out.append(worker.run_once()))
    thread.start()
    assert entered.wait(WAIT)
    for step in range(3):
        store.shift_leases(-200)  # 100 s of the 300 s lease left: only a renewal keeps it alive
        assert renewals.renewals_after(2), f"the heartbeat did not renew the lease (step {step + 1})"
    assert store.open().claim("thief", 5) is None  # the heartbeat kept the lease alive
    finish.set()
    thread.join(WAIT)
    assert out[0].status is IntentStatus.APPLIED and out[0].lost_lease is False and out[0].attempt == 1


def scenario_a_lease_lost_mid_stage_is_reclaimed_and_settled_exactly_once(store) -> None:
    """When renewals do not arrive before the lease lapses (a starved worker; here the stored expiry is moved
    past the lease while the stage runs), the next renewal is refused: the run reports a lost lease, can record
    nothing more (every write is fenced by the claim) and leaves the intent PROCESSING.  The obligation is not
    stranded: its lease has expired, so another worker claims it and settles it, each effect exactly once (the
    vector effect the first worker already put is found by ``inspect``, not applied again)."""

    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    entered, finish = threading.Event(), threading.Event()

    def long_stage(ctx: StageContext) -> None:
        if ctx.stage is V:
            entered.set()
            finish.wait(WAIT)

    renewals = RenewalCounter(store.open())
    worker = StageWorker(renewals, all_handlers(external, **{V: Handler(external, hook=long_stage)}),
                         owner="starved", lease_seconds=30, heartbeat_seconds=0.05)
    out: list = []
    thread = threading.Thread(target=lambda: out.append(worker.run_once()))
    thread.start()
    assert entered.wait(WAIT)
    store.shift_leases(-60)  # the renewals did not come in time: the lease has lapsed
    assert renewals.refused(), "the heartbeat never noticed the lapsed lease"
    finish.set()
    thread.join(WAIT)
    lost = out[0]
    assert lost.lost_lease is True and lost.status is IntentStatus.PROCESSING and lost.attempt == 1
    assert ledger.get_intent("tenant-1", "intent-1").status is IntentStatus.PROCESSING  # unsettled, not stranded
    rescued = StageWorker(store.open(), all_handlers(external), owner="rescuer", lease_seconds=30).run_once()
    assert rescued is not None and rescued.status is IntentStatus.APPLIED and rescued.attempt == 2
    assert rescued.outcomes[C] == "applied (earlier attempt)"
    assert external.effects() == len(stage_set("memory.write.raw").stages)  # each effect exactly once
    assert external.calls("vector_upsert") == 1  # the starved worker's put is recovered, not repeated
    assert ledger.get_intent("tenant-1", "intent-1").status is IntentStatus.APPLIED


def scenario_a_lease_that_lapses_during_settle_is_reclaimed_without_rerunning_stages(store) -> None:
    """Settling holds the ledger for its whole transaction and runs after the heartbeat stops, so it must finish
    within the lease.  If it does not (a slow settle on a loaded runner; here the stored expiry is moved past the
    lease just before it), the fenced settle is refused: the run reports a lost lease and the intent stays
    PROCESSING with every stage already applied.  Another worker claims the expired obligation and settles it
    without running any stage again."""

    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))

    class LapseBeforeSettle:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def settle(self, *args, **kwargs):
            store.shift_leases(-60)  # the settle outlived the lease
            return self._inner.settle(*args, **kwargs)

    first = StageWorker(LapseBeforeSettle(store.open()), all_handlers(external), owner="slow-settle",
                        lease_seconds=30, heartbeat_seconds=0.05).run_once()
    assert first.lost_lease is True and first.status is IntentStatus.PROCESSING and first.attempt == 1
    assert all(outcome == "applied" for outcome in first.outcomes.values())  # every stage ran and was recorded
    assert ledger.get_intent("tenant-1", "intent-1").status is IntentStatus.PROCESSING  # unsettled, not stranded
    calls_before = external.calls()
    rescued = StageWorker(store.open(), all_handlers(external), owner="rescuer", lease_seconds=30).run_once()
    assert rescued is not None and rescued.status is IntentStatus.APPLIED and rescued.attempt == 2
    assert all(outcome == "applied (earlier attempt)" for outcome in rescued.outcomes.values())
    assert external.calls() == calls_before  # no stage ran again
    assert external.effects() == len(stage_set("memory.write.raw").stages)


def scenario_skipped_by_handler_is_reported(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))

    class OptOut:
        def apply(self, ctx: StageContext):
            return Skipped("markdown views disabled for this tenant")

    result = StageWorker(ledger, all_handlers(external, **{M: OptOut()}), owner="w").run_once()
    assert result.status is IntentStatus.PARTIAL
    assert result.skip_reasons[M] == "markdown views disabled for this tenant"


def scenario_downstream_receipts_cite_the_upstream_applications_they_consumed(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    StageWorker(ledger, all_handlers(external), owner="w").run_once()
    receipts = {r.stage: r for r in ledger.receipts("tenant-1", "intent-1") if r.status is StageStatus.APPLIED}
    for requirement in ledger.get_intent("tenant-1", "intent-1").intent.stage_requirements:
        cited = receipts[requirement.stage].upstream_applications
        assert {w.stage for w in cited} == set(requirement.upstream_stages)
        for witness in cited:
            upstream = receipts[witness.stage]
            assert (witness.intent_id, witness.attempt, witness.effect_id) == (
                "intent-1", upstream.attempt, upstream.effect_id
            )
    assert any(r.upstream_applications for r in receipts.values())  # the raw write has dependent stages


def scenario_a_source_free_stage_applies_only_by_declaration(store) -> None:
    ledger, external = store.open(), store.external()
    free = build_intent(
        "memory.write.raw", intent_id="intent-free", tenant_id="tenant-1", actor_id="actor-1",
        credential_id="cred-1", idempotency_key="key-free", request_digest="sha256:a",
        correlation_id="corr-free", preconditions=(), committed=(),
    )
    assert all(r.no_source_witness for r in free.stage_requirements)
    ledger.accept(free)
    result = StageWorker(ledger, all_handlers(external), owner="w").run_once()
    assert result.status is IntentStatus.APPLIED


def _flood(n):
    class Flood(Handler):
        def apply(self, ctx: StageContext) -> Applied:
            return Applied(tuple(
                ReferenceHandle(kind="memory", id=f"m{i}", tenant_id=ctx.tenant_id, resolver="r") for i in range(n)
            ))

    return Flood


def scenario_a_stage_beyond_the_claim_budget_fails_visibly_at_record_time(store) -> None:
    """Record and claim agree: evidence the claim could never walk is refused before it is recorded."""

    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    worker = StageWorker(ledger, all_handlers(external, **{C: _flood(70_000)(external)}), owner="w", max_attempts=1)
    result = drain(worker, limit=3)[0]
    receipts = {r.stage: r for r in ledger.receipts("tenant-1", "intent-1")}
    assert receipts[C].status is StageStatus.FAILED and receipts[C].error.code == "evidence_budget_exceeded"
    assert not receipts[C].output_refs  # the oversized list was never recorded
    assert not ledger.effect_recorded("tenant-1", receipts[C].effect_id)  # refused before the effect was noted
    assert result.status is IntentStatus.FAILED  # a visible terminal state, not a stranded PROCESSING
    assert ledger.derive_status("tenant-1", "intent-1") is IntentStatus.FAILED


def scenario_a_large_stage_within_the_claim_budget_is_recorded_and_settles(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    worker = StageWorker(ledger, all_handlers(external, **{C: _flood(10_000)(external)}), owner="w", max_attempts=1)
    result = drain(worker, limit=3)[0]
    canonical = next(r for r in ledger.receipts("tenant-1", "intent-1") if r.stage is C)
    assert canonical.status is StageStatus.APPLIED and len(canonical.output_refs) == 10_000
    assert result.status is IntentStatus.APPLIED  # what was recorded, the claim can settle


def scenario_the_store_refuses_a_receipt_the_claim_could_never_validate(store) -> None:
    """A writer that bypasses the dispatcher still cannot store an unsettleable receipt set."""

    from datetime import datetime, timezone

    from contracts_lifecycle import StageReceipt
    from ledger.port import ReceiptMismatch

    ledger = store.open()
    built = intent("memory.write.raw")
    ledger.accept(built)
    claim = ledger.claim("w", 30.0)
    requirement = next(r for r in built.stage_requirements if r.stage is C)
    now = datetime.now(timezone.utc)
    oversized = StageReceipt(
        schema_version="candidate-v1", intent_id=built.id, attempt=claim.attempt, stage=C,
        effect_id=requirement.effect_id, status=StageStatus.APPLIED, applied_sources=PRE,
        committed_sources=requirement.committed_sources, output_refs=tuple(
            ReferenceHandle(kind="memory", id=f"m{i}", tenant_id="tenant-1", resolver="r") for i in range(70_000)
        ),
        error=None, started_at=now, finished_at=now,
    )
    with pytest.raises(ReceiptMismatch, match="cannot be validated by the claim"):
        ledger.record_receipt(oversized, "tenant-1", claim)
    assert ledger.receipts("tenant-1", built.id) == ()  # nothing was stored


def scenario_an_evidence_check_can_refuse_a_stage_before_its_receipt_is_recorded(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    seen = []

    def check(intent_, requirement, outcome):
        seen.append(requirement.stage)
        return "evidence_budget" if requirement.stage is V else None

    worker = StageWorker(ledger, all_handlers(external), owner="w", max_attempts=1, evidence_check=check)
    drain(worker, limit=3)
    receipts = {r.stage: r for r in ledger.receipts("tenant-1", "intent-1")}
    assert receipts[V].status is StageStatus.FAILED and receipts[V].error.code == "evidence_budget"
    assert receipts[C].status is StageStatus.APPLIED and C in seen  # the check runs for every applied stage


def _receipt_for(built, stage, claim, status, refs=0, tag="m"):
    from datetime import datetime, timezone

    from contracts_lifecycle import StageReceipt

    requirement = next(r for r in built.stage_requirements if r.stage is stage)
    now = datetime.now(timezone.utc)
    applied = status is StageStatus.APPLIED
    return StageReceipt(
        schema_version="candidate-v1", intent_id=built.id, attempt=claim.attempt, stage=stage,
        effect_id=requirement.effect_id, status=status,
        applied_sources=PRE if applied else (), committed_sources=requirement.committed_sources if applied else (),
        output_refs=tuple(ReferenceHandle(kind="memory", id=f"{tag}{i}", tenant_id="tenant-1", resolver="r")
                          for i in range(refs)),
        error=None, started_at=now, finished_at=None if status is StageStatus.PROCESSING else now,
    )


def scenario_ordinary_receipt_sets_are_recorded_without_walking_the_claim(store) -> None:
    import ledger.port as port

    ledger = store.open()
    walks = []
    import contracts_lifecycle

    original = contracts_lifecycle.validate_stage_receipts
    contracts_lifecycle.validate_stage_receipts = lambda i, r: walks.append(len(r)) or original(i, r)
    try:
        external = store.external()
        ledger.accept(intent("memory.write.raw"))
        flood = _flood(1_000)
        result = drain(StageWorker(ledger, all_handlers(external, **{C: flood(external)}), owner="w"), limit=3)[0]
    finally:
        contracts_lifecycle.validate_stage_receipts = original
    assert result.status is IntentStatus.APPLIED
    assert walks == []  # far below the claim budget: no full walk at record time
    assert port.claimability_problem  # the shared check is what decided


def scenario_a_large_receipt_is_walked_once_per_write_and_never_twice_for_one_receipt(store) -> None:
    import contracts_lifecycle

    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    walks = []
    original = contracts_lifecycle.validate_stage_receipts
    contracts_lifecycle.validate_stage_receipts = lambda i, r: walks.append(sum(len(x.output_refs) for x in r)) or original(i, r)
    try:
        worker = StageWorker(ledger, all_handlers(external, **{C: _flood(20_000)(external)}), owner="w", max_attempts=1)
        result = drain(worker, limit=3)[0]
    finally:
        contracts_lifecycle.validate_stage_receipts = original
    assert result.status is IntentStatus.APPLIED
    later_stages = len({r.stage for r in ledger.receipts("tenant-1", "intent-1")}) - 1
    # one walk for the canonical applied receipt (the dispatcher's check is reused by the ledger, not repeated),
    # then at most one per later write (a processing and an outcome receipt per later stage)
    assert walks and all(w >= 20_000 for w in walks), walks  # the large set is walked, and only it
    assert len(walks) <= 1 + 2 * later_stages, walks


def scenario_the_settle_derives_the_aggregate_with_one_validation_on_success(store) -> None:
    import ledger.port as port

    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    calls = []
    original = port.validate_required_stage_claim
    port.validate_required_stage_claim = lambda **k: calls.append(k["claimed_status"]) or original(**k)
    try:
        drain(StageWorker(ledger, all_handlers(external), owner="w"), limit=3)
        calls.clear()
        assert ledger.derive_status("tenant-1", "intent-1") is IntentStatus.APPLIED
    finally:
        port.validate_required_stage_claim = original
    assert calls == [IntentStatus.APPLIED]  # used to try every status


def scenario_the_history_moving_between_check_and_write_is_validated_again_in_the_transaction(store) -> None:
    from ledger.port import ReceiptMismatch

    ledger = store.open()
    built = intent("memory.write.raw")
    ledger.accept(built)
    claim = ledger.claim("w", 30.0)
    first = _receipt_for(built, C, claim, StageStatus.APPLIED, refs=40_000)
    ledger.record_receipt(first, "tenant-1", claim)
    big = _receipt_for(built, V, claim, StageStatus.APPLIED, refs=40_000, tag="v")
    big = type(big).model_validate({**big.model_dump(), "upstream_applications": ({
        "intent_id": built.id, "stage": C, "attempt": claim.attempt, "effect_id": first.effect_id},)})
    # a cache entry for this very receipt validated against an older history must not be trusted
    ledger.__dict__.setdefault("_claimable_cache", {})[("tenant-1", built.id)] = (0, ledger._receipt_digest(big))
    with pytest.raises(ReceiptMismatch, match="cannot be validated by the claim"):
        ledger.record_receipt(big, "tenant-1", claim)


def scenario_a_second_applied_receipt_with_different_evidence_is_refused_at_record_time(store) -> None:
    from ledger.port import ReceiptMismatch

    ledger = store.open()
    built = intent("memory.write.raw")
    ledger.accept(built)
    first_claim = ledger.claim("w", 30.0)
    ledger.record_receipt(_receipt_for(built, C, first_claim, StageStatus.APPLIED, refs=3), "tenant-1", first_claim)
    ledger.release(first_claim)
    second_claim = ledger.claim("w2", 30.0)
    with pytest.raises(ReceiptMismatch, match="same results"):
        ledger.record_receipt(_receipt_for(built, C, second_claim, StageStatus.APPLIED, refs=2), "tenant-1", second_claim)
    # identical evidence in a later attempt is still accepted
    ledger.record_receipt(_receipt_for(built, C, second_claim, StageStatus.APPLIED, refs=3), "tenant-1", second_claim)


def scenario_a_retry_reporting_the_same_results_in_another_order_is_accepted(store) -> None:
    from ledger.port import ReceiptMismatch

    ledger = store.open()
    built = intent("memory.write.raw")
    ledger.accept(built)
    first_claim = ledger.claim("w", 30.0)
    first = _receipt_for(built, C, first_claim, StageStatus.APPLIED, refs=5)
    ledger.record_receipt(first, "tenant-1", first_claim)
    ledger.release(first_claim)
    second_claim = ledger.claim("w2", 30.0)
    same_set = _receipt_for(built, C, second_claim, StageStatus.APPLIED, refs=5)
    reordered = type(same_set).model_validate({**same_set.model_dump(), "output_refs": tuple(reversed(same_set.output_refs))})
    other_ids = _receipt_for(built, C, second_claim, StageStatus.APPLIED, refs=5, tag="x")
    with pytest.raises(ReceiptMismatch, match="same results"):  # same count, different references
        ledger.record_receipt(other_ids, "tenant-1", second_claim)
    ledger.record_receipt(reordered, "tenant-1", second_claim)  # the claim compares sets, and so does the record
    applied = [r for r in ledger.receipts("tenant-1", built.id) if r.stage is C and r.status is StageStatus.APPLIED]
    assert [r.attempt for r in applied] == [first_claim.attempt, second_claim.attempt]


def _sized_intent(name: str, sources: int):
    pre = tuple(SourceVersion(record_id=f"rec-{i}", content_revision=3, policy_epoch=7) for i in range(sources))
    post = tuple(SourceVersion(record_id=f"rec-{i}", content_revision=4, policy_epoch=7) for i in range(sources))
    return build_intent(
        "memory.write.raw", intent_id=f"intent-{name}", tenant_id="tenant-1", actor_id="actor-1",
        credential_id="cred-1", idempotency_key=f"key-{name}", request_digest="sha256:a",
        correlation_id=f"corr-{name}", preconditions=pre, committed=post,
    )


def scenario_an_intent_too_large_to_ever_be_claimed_is_refused_at_accept(store) -> None:
    import contracts_lifecycle
    from contracts_limits import DiscoveryTier
    from ledger.port import IntentTooLarge

    ledger, external = store.open(), store.external()
    original = contracts_lifecycle.LIFECYCLE_TIER
    contracts_lifecycle.LIFECYCLE_TIER = DiscoveryTier("small", 2048, 12288, 256)  # a small claim budget
    try:
        with pytest.raises(IntentTooLarge):  # it used to be accepted, apply every stage and never settle
            ledger.accept(_sized_intent("huge", 400))
        assert ledger.get_intent("tenant-1", "intent-huge") is None and ledger.scan_unqueued() == ()
        ledger.accept(_sized_intent("large", 100))  # large but within the budget: walked, accepted, settles
        result = drain(StageWorker(ledger, all_handlers(external), owner="w"), limit=3)[0]
        assert result.status is IntentStatus.APPLIED and result.settle_error is None
    finally:
        contracts_lifecycle.LIFECYCLE_TIER = original


def scenario_a_receipt_validated_but_not_recorded_is_not_kept_by_the_worker(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))

    class LeaseLostBeforeRecord:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def record_effect(self, *args, **kwargs):  # after the receipt was validated, before it is recorded
            raise StaleLease("lease lost between validation and record")

    worker = StageWorker(LeaseLostBeforeRecord(ledger), all_handlers(external), owner="w")
    result = worker.run_once()
    assert result is not None and result.lost_lease
    assert worker._prepared == {}  # used to keep the validated receipt for the worker's life


def scenario_an_oversize_write_to_a_terminal_intent_is_fenced_first(store) -> None:
    from ledger.port import PublicationFenced

    ledger, external = store.open(), store.external()
    built = intent("memory.write.raw")
    ledger.accept(built)
    drain(StageWorker(ledger, all_handlers(external), owner="w"), limit=3)
    assert ledger.get_intent("tenant-1", built.id).status is IntentStatus.APPLIED  # terminal
    from types import SimpleNamespace

    late = _receipt_for(built, M, SimpleNamespace(attempt=1), StageStatus.CANCELLED, refs=70_000)
    with pytest.raises(PublicationFenced):
        ledger.record_receipt(late, "tenant-1")


def scenario_a_type_error_from_the_claimability_check_is_a_visible_stage_failure(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))

    class Raising:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def check_receipt_claimable(self, receipt, tenant_id):
            raise TypeError("unexpected shape")

    result = drain(StageWorker(Raising(ledger), all_handlers(external), owner="w", max_attempts=1), limit=3)[0]
    canonical = next(r for r in ledger.receipts("tenant-1", "intent-1") if r.stage is C)
    assert canonical.status is StageStatus.FAILED and canonical.error.code == "evidence_invalid"
    assert result.status is IntentStatus.FAILED


def scenario_concurrent_dispatchers_send_one_wake_per_obligation(store) -> None:
    """Every dispatcher scans the same rows before any of them hands one off: one send each."""

    rows, dispatchers, handles = 12, 16, 4
    first = store.open()
    for n in range(rows):
        first.accept(intent("memory.write.raw", name=str(n)))
    scanned = threading.Barrier(dispatchers, timeout=30)

    class ScanTogether:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def scan_unqueued(self, limit=100):
            entries = self._inner.scan_unqueued(limit)
            scanned.wait()
            return entries

    ledgers = [store.open() for _ in range(handles)]  # several API processes share the ledger
    sent: list = []
    lock = threading.Lock()

    def enqueue(entry) -> None:
        with lock:
            sent.append(entry)

    def dispatch(i: int) -> None:
        Publisher(ScanTogether(ledgers[i % handles]), enqueue).publish_unqueued()

    threads = [threading.Thread(target=dispatch, args=(i,)) for i in range(dispatchers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not any(thread.is_alive() for thread in threads)
    assert len(sent) == rows, f"{len(sent)} wake-ups for {rows} obligations"
    assert len({entry.outbox_id for entry in sent}) == rows
    assert first.scan_unqueued() == ()


def scenario_an_obligation_is_taken_once_per_hand_off(store) -> None:
    ledger = store.open()
    ledger.accept(intent("memory.write.raw"))
    (entry,) = ledger.scan_unqueued()
    assert entry.attempt == 0
    assert ledger.take_unqueued(entry.tenant_id, entry.outbox_id, 0) is True
    assert ledger.take_unqueued(entry.tenant_id, entry.outbox_id, 0) is False  # already handed off
    assert ledger.scan_unqueued() == ()

    ledger.unmark_queued(entry.tenant_id, entry.outbox_id, 0)  # the enqueue failed: back to the scan
    assert ledger.scan_unqueued() == (entry,)
    assert ledger.take_unqueued("tenant-other", entry.outbox_id, 0) is False  # tenant-bound
    assert ledger.take_unqueued(entry.tenant_id, entry.outbox_id, 1) is False  # not the scanned attempt
    assert ledger.scan_unqueued() == (entry,)

    claim = ledger.claim("w", 30.0)
    assert claim is not None
    assert ledger.take_unqueued(entry.tenant_id, entry.outbox_id, claim.attempt) is False  # leased: not a queue row
    ledger.unmark_queued(entry.tenant_id, entry.outbox_id, 0)  # and a late give-back leaves the lease alone
    assert ledger.claim("w2", 30.0) is None

    ledger.release(claim)  # a retry is a new hand-off with a new attempt
    (again,) = ledger.scan_unqueued()
    assert (again.outbox_id, again.attempt) == (entry.outbox_id, 1)
    assert ledger.take_unqueued(again.tenant_id, again.outbox_id, again.attempt) is True


def scenario_a_failed_publish_returns_the_row_to_the_scan(store) -> None:
    ledger = store.open()
    ledger.accept(intent("memory.write.raw"))

    def down(entry) -> None:
        raise ConnectionError("queue down")

    with pytest.raises(ConnectionError):
        Publisher(ledger, down).publish_unqueued()
    assert len(ledger.scan_unqueued()) == 1
    sent: list = []
    assert Publisher(ledger, sent.append).publish_unqueued() == 1
    assert len(sent) == 1 and ledger.scan_unqueued() == ()


def scenario_a_late_give_back_never_reopens_a_newer_hand_off(store) -> None:
    """A slow dispatcher's failed enqueue returns only its own hand-off, not one made after a retry."""

    ledger = store.open()
    ledger.accept(intent("memory.write.raw"))
    sent: list = []

    def slow_then_failing(entry) -> None:
        claim = ledger.claim("poll", 30.0)  # meanwhile the poll claims the row and releases it: attempt 1
        ledger.release(claim)
        Publisher(ledger, sent.append).publish_unqueued()  # a second dispatcher hands attempt 1 off
        raise ConnectionError("the first dispatcher's enqueue fails late")

    with pytest.raises(ConnectionError):
        Publisher(ledger, slow_then_failing).publish_unqueued()
    assert [(e.outbox_id, e.attempt) for e in sent] == [(sent[0].outbox_id, 1)]
    assert ledger.scan_unqueued() == ()  # attempt 1 is still handed off
    assert Publisher(ledger, sent.append).publish_unqueued() == 0  # used to send attempt 1 a second time


SCENARIOS = {
    name.removeprefix("scenario_"): fn for name, fn in sorted(globals().items()) if name.startswith("scenario_")
}


def test_registry_declares_no_source_witness_only_where_there_are_no_inputs() -> None:
    with_inputs = build_requirements("memory.write.raw", intent_id="i", preconditions=PRE, committed=POST)
    assert [r.no_source_witness for r in with_inputs if r.stage is C] == [False]
    assert all(r.no_source_witness is False for r in with_inputs if r.upstream_stages)  # fed by the commit
    bare = build_requirements("memory.write.raw", intent_id="i", preconditions=(), committed=())
    assert all(r.no_source_witness for r in bare)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_dispatch_scenarios_on_sqlite(name: str, tmp_path: Path) -> None:
    store = SqliteStore(tmp_path)
    try:
        SCENARIOS[name](store)
    finally:
        store.close()


def test_a_renewal_the_heartbeat_has_refused_is_reported_as_a_lost_lease() -> None:
    """The heartbeat reports a lapsed lease itself: a refused renewal sets ``lost_lease`` and ends the beat.

    The scenarios cannot pin this on their own, because a run's next fenced write reports the loss too; here the
    beat runs alone, so a beat that swallowed the refusal (or kept beating) would go unnoticed otherwise."""

    from ledger.dispatch import RunResult, _ClaimHolder

    refusals: list[int] = []

    class Refusing:
        def extend_lease(self, claim, lease_seconds):
            refusals.append(1)
            raise StaleLease("claim is not a live lease")

    worker = StageWorker(Refusing(), {}, owner="w", lease_seconds=30, heartbeat_seconds=0.01)
    result = RunResult("intent-1", "tenant-1", IntentStatus.PROCESSING, 1)
    stop = threading.Event()
    beat = threading.Thread(target=worker._beat, args=(_ClaimHolder(None), stop, result))
    beat.start()
    beat.join(WAIT)
    stop.set()
    assert not beat.is_alive(), "the beat kept running after its renewal was refused"
    assert result.lost_lease is True and refusals == [1]


def test_heartbeat_never_reports_a_lost_lease_for_a_run_that_completed(tmp_path: Path) -> None:
    """Regression: a beat that fired between ``complete`` and the thread stop extended a
    finished claim, got ``StaleLease`` and flagged ``lost_lease`` on a successful run."""

    store = SqliteStore(tmp_path)
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))

    class SlowComplete:
        """Delegating ledger whose ``complete`` lingers so several beats fire right after it."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def complete(self, claim):
            self._inner.complete(claim)
            time.sleep(0.25)

    worker = StageWorker(SlowComplete(store.open()), all_handlers(external), owner="w",
                         lease_seconds=5, heartbeat_seconds=0.01)
    result = worker.run_once()
    assert result.status is IntentStatus.APPLIED
    assert result.lost_lease is False
    store.close()


# ---------------------------------------------------------------- review fixes (H-B1-1, H-B1-2)
def _canonical_handler(external, result):
    class Canonical:
        def apply(self, ctx):
            return result(ctx)

    return {C: Canonical(), V: Handler(external), G: Handler(external), M: Handler(external)}


def test_a_handler_result_the_ledger_would_reject_fails_the_stage_and_the_intent_settles(tmp_path: Path) -> None:
    """H-B1-1: ``Applied()`` with no outputs for a stage that requires one used to be recorded,
    then ``settle`` raised, ``run_once`` crashed and the intent stayed PROCESSING forever."""

    store = SqliteStore(tmp_path)
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    worker = StageWorker(ledger, _canonical_handler(external, lambda ctx: Applied()), owner="w")

    result = worker.run_once()  # must not raise

    assert result.status is IntentStatus.FAILED and result.outcomes[C] == "failed (outputs_missing)"
    states = {r.stage: r.status for r in ledger.receipts("tenant-1", "intent-1")}
    assert states[C] is StageStatus.FAILED and states[V] is StageStatus.SKIPPED  # downstream not run
    assert external.effects() == 0  # nothing downstream ran on the rejected evidence
    assert ledger.claim("again", 5) is None  # settled: not a poison pill
    store.close()


def test_an_output_reference_from_another_tenant_fails_the_stage_without_crashing(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path)
    ledger, external = store.open(), store.external()
    ledger.accept(intent("memory.write.raw"))
    foreign = ReferenceHandle(kind="artifact", id="obj-1", tenant_id="tenant-other", resolver="r")
    worker = StageWorker(ledger, _canonical_handler(external, lambda ctx: Applied((foreign,))), owner="w")

    result = worker.run_once()

    assert result.status is IntentStatus.FAILED and result.outcomes[C] == "failed (output_tenant_mismatch)"
    assert ledger.claim("again", 5) is None
    assert all(r.output_refs == () for r in ledger.receipts("tenant-1", "intent-1"))
    store.close()


def test_a_settle_the_ledger_rejects_closes_the_obligation_instead_of_looping(tmp_path: Path) -> None:
    from ledger.port import LedgerError

    store = SqliteStore(tmp_path)
    inner, external = store.open(), store.external()
    inner.accept(intent("memory.write.raw"))

    class BrokenSettle:
        def __init__(self, ledger):
            self._ledger = ledger

        def __getattr__(self, name):
            return getattr(self._ledger, name)

        def settle(self, *args, **kwargs):
            raise LedgerError("receipts support 0 aggregates")

    result = StageWorker(BrokenSettle(inner), all_handlers(external), owner="w").run_once()

    assert result.settle_error == "receipts support 0 aggregates"
    assert inner.claim("again", 5) is None  # not re-leased forever
    assert inner.get_intent("tenant-1", "intent-1").status is IntentStatus.PROCESSING  # honest: unsettled
    store.close()


def test_effect_ids_are_unique_across_tenants_even_for_the_same_intent_id() -> None:
    def effects(tenant, operation="memory.write.raw"):
        built = build_intent(
            operation, intent_id="intent-same", tenant_id=tenant, actor_id="a", credential_id="c",
            idempotency_key="k", request_digest="d", correlation_id="x", preconditions=PRE, committed=POST,
        )
        return [r.effect_id for r in built.stage_requirements]

    one, two = effects("tenant-1"), effects("tenant-2")
    assert not set(one) & set(two)
    assert effects("tenant-1") == one  # deterministic
    assert all(len(e) <= 256 for e in one)
