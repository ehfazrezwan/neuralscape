"""Backend-independent ledger conformance cases.

Each case receives a ``Store`` for an empty, isolated ledger.  ``store.open()``
returns a new connection to that same store (so cases can model separate
workers) and ``store.fresh()`` returns another empty store.  Cases use plain
assertions so they run under pytest and as a script.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import signal
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator, Protocol

from contracts_lifecycle import (
    Intent,
    IntentStatus,
    ProcessingStage,
    StageError,
    StageEvidenceRequirement,
    StageReceipt,
    StageStatus,
    UpstreamApplication,
)
from contracts_references import ReferenceHandle, SourceVersion
from ledger.port import (
    DuplicateIntentId,
    HoldActive,
    IdempotencyConflict,
    Claim,
    IllegalTransition,
    LedgerError,
    LedgerPort,
    ObjectRef,
    ObjectRefConflict,
    PublicationFenced,
    ReceiptMismatch,
    ResurrectionDenied,
    StaleLease,
    UnknownIntent,
    snapshot_digest,
)

T0 = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


class Store(Protocol):
    def open(self) -> LedgerPort:
        """A runtime-privilege connection (non-superuser, no direct outbox updates)."""

    def open_admin(self) -> LedgerPort:
        """The owner connection (restore, migrations)."""

    def fresh(self) -> "Store": ...

    def shift_leases(self, seconds: float) -> None:
        """Move every stored lease expiry by ``seconds`` (negative = a forward clock step)."""


@contextlib.contextmanager
def expect(exc: type[BaseException]) -> Iterator[None]:
    try:
        yield
    except exc:
        return
    except BaseException as other:  # noqa: BLE001
        raise AssertionError(f"expected {exc.__name__}, got {type(other).__name__}: {other}") from other
    raise AssertionError(f"expected {exc.__name__}")


# ------------------------------------------------------------------ builders


def make_intent(
    name: str = "1",
    *,
    tenant: str = "tenant-1",
    actor: str = "actor-1",
    operation: str = "write",
    key: str | None = None,
    digest: str = "sha256:a",
    stages: tuple[ProcessingStage, ...] = (ProcessingStage.CANONICAL, ProcessingStage.VECTOR),
) -> Intent:
    return Intent(
        schema_version="candidate-v1",
        id=f"intent-{name}",
        tenant_id=tenant,
        actor_id=actor,
        credential_id="credential-1",
        operation=operation,
        target_refs=(),
        request_digest=digest,
        idempotency_key=key or f"key-{name}",
        source_preconditions=(SourceVersion(record_id="memory-1", content_revision=3, policy_epoch=7),),
        stage_requirements=tuple(
            StageEvidenceRequirement(
                stage=stage,
                effect_id=f"effect-{stage.value}-{name}",
                precondition_source_ids=("memory-1",),
                upstream_stages=(),
                committed_sources=(),
                minimum_output_refs=0,
            )
            for stage in stages
        ),
        correlation_id=f"correlation-{name}",
        accepted_at=datetime(2020, 1, 1, tzinfo=timezone.utc),  # caller clock, deliberately stale
    )


def make_receipt(
    intent: Intent,
    stage: ProcessingStage,
    status: StageStatus,
    *,
    attempt: int = 1,
) -> StageReceipt:
    effect = next(r.effect_id for r in intent.stage_requirements if r.stage is stage)
    terminal = status not in {StageStatus.PENDING, StageStatus.PROCESSING}
    pending = status is StageStatus.PENDING
    return StageReceipt(
        schema_version="candidate-v1",
        intent_id=intent.id,
        attempt=attempt,
        stage=stage,
        effect_id=effect,
        status=status,
        applied_sources=() if pending else intent.source_preconditions,
        committed_sources=(),
        upstream_applications=tuple(
            UpstreamApplication(intent_id=intent.id, stage=up, attempt=attempt, effect_id=up_effect)
            for requirement in intent.stage_requirements
            if requirement.stage is stage and status is StageStatus.APPLIED
            for up in requirement.upstream_stages
            for up_effect in [next(r.effect_id for r in intent.stage_requirements if r.stage is up)]
        ),
        output_refs=(),
        error=StageError(code="projection_failed", retryable=False, safe_message="failed")
        if status is StageStatus.FAILED
        else None,
        started_at=None if pending else T0,
        finished_at=T0 + timedelta(seconds=1) if terminal else None,
    )


def ref(record: str, revision: int = 1, epoch: int = 1, tenant: str = "tenant-1", sources: tuple = ()) -> ObjectRef:
    return ObjectRef(tenant, record, revision, epoch, tuple(sources), checksum=f"sha256:{record}-{revision}")


def _handle(ident: str, tenant: str = "tenant-1") -> ReferenceHandle:
    return ReferenceHandle(kind="source", id=ident, tenant_id=tenant, resolver="resolve_source")


def _drain(ledger: LedgerPort, owner: str = "drain") -> None:
    while (claim := ledger.claim(owner, 30)) is not None:
        ledger.complete(claim)


# --------------------------------------------------------------------- cases


def case_atomic_accept(store: Store) -> None:
    ledger = store.open()
    ledger.accept(make_intent("t"), [ref("rec-dead")])
    ledger.tombstone("tenant-1", "rec-gone", 1, "erased")
    with expect(ResurrectionDenied):  # fails after the intent row would be inserted
        ledger.accept(make_intent("a"), [ref("rec-ok"), ref("rec-gone")])
    assert ledger.get_intent("tenant-1", "intent-a") is None
    assert ledger.get_object_ref("tenant-1", "rec-ok", 1) is None
    assert [e.intent_id for e in ledger.scan_unqueued()] == ["intent-t"]  # no orphan outbox row
    result = ledger.accept(make_intent("a"), [ref("rec-ok")])
    assert result.created and result.status is IntentStatus.ACCEPTED
    assert ledger.get_object_ref("tenant-1", "rec-ok", 1) is not None
    assert {e.intent_id for e in ledger.scan_unqueued()} == {"intent-t", "intent-a"}


def case_idempotency(store: Store) -> None:
    ledger = store.open()
    first = ledger.accept(make_intent("1", key="k"))
    replay = ledger.accept(make_intent("other-id", key="k"))  # new id, same key+digest
    assert replay.intent_id == first.intent_id and not replay.created and replay.version == first.version == 1
    with expect(IdempotencyConflict):
        ledger.accept(make_intent("2", key="k", digest="sha256:b"))
    for variant in (
        make_intent("t2", key="k", tenant="tenant-2"),
        make_intent("a2", key="k", actor="actor-2"),
        make_intent("o2", key="k", operation="edit"),
    ):
        assert ledger.accept(variant).created
    with expect(DuplicateIntentId):  # same id under another idempotency scope
        ledger.accept(make_intent("1", key="different"))
    assert len(ledger.scan_unqueued()) == 4


def case_idempotency_race(store: Store) -> None:
    results: list[Any] = []
    # Every connection is open before any accept starts and all accepts are released
    # together: the contention is real, not left to thread start-up timing.
    barrier = threading.Barrier(16)

    def worker() -> None:
        try:
            ledger = store.open()
            barrier.wait(timeout=60)
            results.append(ledger.accept(make_intent(threading.current_thread().name, key="race")))
        except Exception as exc:  # noqa: BLE001
            barrier.abort()
            results.append(exc)

    threads = [threading.Thread(target=worker, name=f"w{i}") for i in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert all(not isinstance(r, Exception) for r in results), results
    assert sum(r.created for r in results) == 1
    assert len({r.intent_id for r in results}) == 1
    assert len(store.open().scan_unqueued()) == 1


def case_lease_steal_and_fence(store: Store) -> None:
    ledger, other = store.open(), store.open()
    ledger.accept(make_intent())
    a = ledger.claim("A", 0.4)
    assert a is not None and a.attempt == 1
    assert other.claim("B", 5) is None  # still leased
    time.sleep(0.7)
    b = other.claim("B", 5)
    assert b is not None and b.outbox_id == a.outbox_id and b.attempt == 2
    with expect(StaleLease):
        ledger.complete(a)
    with expect(StaleLease):
        ledger.extend_lease(a, 5)
    extended = other.extend_lease(b, 30)
    assert extended.lease_expires_at > b.lease_expires_at
    other.complete(b)
    other.complete(b)  # idempotent for the current owner
    assert other.claim("C", 5) is None
    c_ledger = store.open()
    c_ledger.accept(make_intent("2"))
    claim = c_ledger.claim("R", 30)
    c_ledger.release(claim)
    again = c_ledger.claim("R2", 30)
    assert again is not None and again.attempt == 2 and again.intent_id == "intent-2"
    with expect(StaleLease):
        c_ledger.release(claim)
    with expect(ValueError):
        c_ledger.claim("R", 0)


def _child_accept_then_die(store: Store) -> None:
    store.open().accept(make_intent("crash", key="crash"))
    os._exit(137)  # committed, never handed to the queue


def case_commit_before_enqueue_recovery(store: Store) -> None:
    ctx = multiprocessing.get_context("fork")
    p = ctx.Process(target=_child_accept_then_die, args=(store,))
    p.start()
    p.join()
    assert p.exitcode == 137
    ledger = store.open()
    pending = ledger.scan_unqueued()
    assert [e.intent_id for e in pending] == ["intent-crash"]
    ledger.mark_queued(pending[0].tenant_id, pending[0].outbox_id)
    assert ledger.scan_unqueued() == ()
    claim = ledger.claim("w", 30)  # queued or not, the obligation is still claimable
    assert claim is not None and claim.intent_id == "intent-crash"


def _child_accept_forever(store: Store) -> None:
    ledger = store.open()
    i = 0
    while True:
        ledger.accept(make_intent(f"k{i}"), [ref(f"rec-{i}")])
        i += 1


def case_kill_mid_accept_leaves_no_orphans(store: Store) -> None:
    ctx = multiprocessing.get_context("fork")
    p = ctx.Process(target=_child_accept_forever, args=(store,))
    p.start()
    time.sleep(0.6)
    os.kill(p.pid, signal.SIGKILL)
    p.join()
    ledger = store.open()
    snap = ledger.snapshot()
    intents, outbox = snap["tables"]["intents"], snap["tables"]["outbox"]
    assert intents and len(intents) == len(outbox)
    assert {r["id"] for r in intents} == {r["intent_id"] for r in outbox}
    for row in intents:  # object refs commit with the intent, never alone
        n = row["id"].removeprefix("intent-k")
        assert ledger.get_object_ref("tenant-1", f"rec-{n}", 1) is not None


def case_ambiguous_effect_replay(store: Store) -> None:
    a, b = store.open(), store.open()
    intent = make_intent()
    a.accept(intent)
    effect = intent.stage_requirements[0].effect_id
    first = a.claim("A", 0.4)
    assert a.record_effect("tenant-1", effect, intent.id) is True  # effect happened, A dies
    time.sleep(0.7)
    second = b.claim("B", 30)
    assert second is not None and second.attempt == 2
    assert b.effect_recorded("tenant-1", effect)  # inspect-by-effect-id before re-creating
    assert b.record_effect("tenant-1", effect, intent.id) is False
    b.complete(second)
    with expect(StaleLease):
        a.complete(first)
    with expect(ReceiptMismatch):
        b.record_effect("tenant-1", "effect-undeclared", intent.id)
    with expect(UnknownIntent):
        b.record_effect("tenant-2", effect, intent.id)


def case_status_machine(store: Store) -> None:
    ledger = store.open()
    intent = make_intent()
    ledger.accept(intent)
    claim = ledger.claim("w", 30)
    canonical, vector = ProcessingStage.CANONICAL, ProcessingStage.VECTOR
    with expect(IllegalTransition):  # nothing started
        ledger.set_status("tenant-1", intent.id, IntentStatus.PROCESSING, claim)
    with expect(IllegalTransition):  # no evidence of completion
        ledger.set_status("tenant-1", intent.id, IntentStatus.APPLIED, claim)
    ledger.record_receipt(make_receipt(intent, canonical, StageStatus.APPLIED), "tenant-1", claim)
    ledger.record_receipt(make_receipt(intent, vector, StageStatus.PENDING), "tenant-1", claim)
    with expect(StaleLease):  # evidence is fine but the writer holds no claim
        ledger.set_status("tenant-1", intent.id, IntentStatus.PROCESSING)
    assert ledger.set_status("tenant-1", intent.id, IntentStatus.PROCESSING, claim) is IntentStatus.PROCESSING
    # mixed failed + pending stays nonterminal PROCESSING
    ledger.record_receipt(make_receipt(intent, vector, StageStatus.FAILED), "tenant-1", claim)
    ledger.release(claim)  # a retry is a new claim attempt, and receipts follow the claim's attempt
    claim = ledger.claim("w", 30)
    assert claim.attempt == 2
    ledger.record_receipt(make_receipt(intent, vector, StageStatus.PROCESSING, attempt=2), "tenant-1", claim)
    ledger.set_status("tenant-1", intent.id, IntentStatus.PROCESSING, claim)  # revalidates only
    with expect(IllegalTransition):  # unfinished stage cannot be PARTIAL or APPLIED
        ledger.set_status("tenant-1", intent.id, IntentStatus.PARTIAL, claim)
    ledger.record_receipt(make_receipt(intent, vector, StageStatus.APPLIED, attempt=2), "tenant-1", claim)
    ledger.set_status("tenant-1", intent.id, IntentStatus.APPLIED, claim)
    stored = ledger.get_intent("tenant-1", intent.id)
    assert stored.status is IntentStatus.APPLIED
    with expect(IllegalTransition):  # terminal states are final
        ledger.set_status("tenant-1", intent.id, IntentStatus.FAILED, claim)
    ledger.complete(claim)  # outbox already done by the terminal write; completion is idempotent
    assert ledger.claim("w2", 30) is None  # an applied intent is never redispatched
    assert ledger.get_intent("tenant-2", intent.id) is None
    with expect(UnknownIntent):
        ledger.set_status("tenant-2", intent.id, IntentStatus.APPLIED, claim)


def case_cancel_fences_publication(store: Store) -> None:
    ledger = store.open()
    intent = make_intent()
    ledger.accept(intent)
    claim = ledger.claim("w", 30)
    canonical, vector = ProcessingStage.CANONICAL, ProcessingStage.VECTOR
    ledger.record_receipt(make_receipt(intent, canonical, StageStatus.PROCESSING), "tenant-1", claim)
    for stage in (canonical, vector):  # client cancel needs no lease
        ledger.record_receipt(make_receipt(intent, stage, StageStatus.CANCELLED, attempt=2), "tenant-1")
    assert ledger.set_status("tenant-1", intent.id, IntentStatus.CANCELLED) is IntentStatus.CANCELLED
    with expect(StaleLease):  # the holder's lease is fenced
        ledger.complete(claim)
    with expect(PublicationFenced):  # publication evidence is refused after the fence
        ledger.record_receipt(make_receipt(intent, vector, StageStatus.APPLIED, attempt=3), "tenant-1", claim)
    assert ledger.claim("w3", 30) is None  # the fenced obligation is never redispatched


def case_receipt_validation(store: Store) -> None:
    ledger = store.open()
    intent = make_intent()
    ledger.accept(intent)
    claim = ledger.claim("w", 30)
    canonical = ProcessingStage.CANONICAL
    good = make_receipt(intent, canonical, StageStatus.PROCESSING)
    version = ledger.record_receipt(good, "tenant-1", claim)
    assert ledger.record_receipt(good, "tenant-1", claim) == version == 1  # identical replay is a no-op
    assert len(ledger.receipts("tenant-1", intent.id)) == 1
    wrong_effect = good.model_copy(update={"effect_id": "effect-other"})
    with expect(ReceiptMismatch):
        ledger.record_receipt(wrong_effect, "tenant-1", claim)
    undeclared = make_receipt(intent, ProcessingStage.VECTOR, StageStatus.PROCESSING).model_copy(
        update={"stage": ProcessingStage.GRAPH}
    )
    with expect(ReceiptMismatch):
        ledger.record_receipt(undeclared, "tenant-1", claim)
    conflicting = StageReceipt.model_validate(
        {**good.model_dump(), "started_at": T0 + timedelta(seconds=5)}
    )
    with expect(ReceiptMismatch):
        ledger.record_receipt(conflicting, "tenant-1", claim)
    with expect(UnknownIntent):
        ledger.record_receipt(good, "tenant-2", claim)
    with expect(StaleLease):  # a live claim is required for processing evidence
        ledger.record_receipt(make_receipt(intent, ProcessingStage.VECTOR, StageStatus.PROCESSING), "tenant-1")


def case_dispatch_exactly_once(store: Store) -> None:
    n = 120
    ledger = store.open()
    for i in range(n):
        ledger.accept(make_intent(str(i), stages=(ProcessingStage.CANONICAL,)))
    claims: list[tuple[str, int]] = []
    guard = threading.Lock()

    def worker(name: str) -> None:
        mine = store.open()
        while (claim := mine.claim(name, 60)) is not None:
            intent = mine.get_intent(claim.tenant_id, claim.intent_id).intent
            fresh = mine.record_effect(claim.tenant_id, intent.stage_requirements[0].effect_id, intent.id)
            mine.complete(claim)
            with guard:
                claims.append((claim.intent_id, int(fresh)))

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(claims) == n and len({c[0] for c in claims}) == n and all(c[1] == 1 for c in claims)
    assert ledger.claim("late", 30) is None


def case_object_refs_and_tombstones(store: Store) -> None:
    ledger = store.open()
    source = _handle("src-1")
    cross = _handle("src-x", tenant="tenant-2")  # cross-tenant handle is data, never authority
    base = ref("rec-1", 1, 4, sources=(source, cross))
    assert ledger.put_object_ref(base) is True
    assert ledger.put_object_ref(base) is False
    with expect(ObjectRefConflict):
        ledger.put_object_ref(ref("rec-1", 1, 5, sources=(source, cross)))
    assert ledger.put_object_ref(ref("rec-1", 2, 4)) is True
    got = ledger.get_object_ref("tenant-1", "rec-1", 1)
    assert got == base and got.source_refs == (source, cross)
    assert ledger.get_object_ref("tenant-2", "rec-1", 1) is None
    assert ledger.tombstone("tenant-1", "rec-1", 2, "admin_delete") is True
    assert ledger.tombstone("tenant-1", "rec-1", 2, "admin_delete") is False
    assert ledger.get_object_ref("tenant-1", "rec-1", 1) is None  # lineage removed with the body
    tomb = ledger.get_tombstone("tenant-1", "rec-1")
    assert tomb is not None and tomb.reason_code == "admin_delete" and tomb.content_revision == 2
    assert ledger.get_tombstone("tenant-2", "rec-1") is None
    with expect(ResurrectionDenied):
        ledger.put_object_ref(ref("rec-1", 3, 4))
    assert ledger.put_object_ref(ref("rec-1", 1, 4, tenant="tenant-2")) is True  # other tenant unaffected
    with expect(ReceiptMismatch):  # accept refuses refs for another tenant
        ledger.accept(make_intent("x"), [ref("rec-z", tenant="tenant-2")])


def case_snapshot_restore(store: Store) -> None:
    ledger = store.open()
    done, active = make_intent("done"), make_intent("active")
    ledger.accept(done, [ref("rec-1", sources=(_handle("s1"),))])
    ledger.accept(active)
    claim = ledger.claim("w", 30)  # intent-done, leased at snapshot time
    first, second = (r.stage for r in done.stage_requirements)
    ledger.record_receipt(make_receipt(done, first, StageStatus.APPLIED), "tenant-1", claim)
    ledger.record_receipt(make_receipt(done, second, StageStatus.PROCESSING), "tenant-1", claim)
    ledger.set_status("tenant-1", done.id, IntentStatus.PROCESSING, claim)
    ledger.record_receipt(make_receipt(done, second, StageStatus.APPLIED), "tenant-1", claim)
    ledger.set_status("tenant-1", done.id, IntentStatus.APPLIED, claim)
    leased = ledger.claim("w2", 30)  # intent-active, leased and queued state must not survive
    ledger.record_effect("tenant-1", active.stage_requirements[0].effect_id, active.id)
    ledger.tombstone("tenant-1", "rec-erased", 4, "admin_delete")
    snapshot = ledger.snapshot()
    json.dumps(snapshot)  # backend-neutral and JSON-serializable
    assert snapshot["schema_version"] >= 1 and "sequence" not in snapshot

    target_store = store.fresh()
    target_store.open_admin().restore(snapshot)
    target = target_store.open()
    assert target.get_intent("tenant-1", done.id).status is IntentStatus.APPLIED
    assert target.get_intent("tenant-1", done.id).intent == done
    assert target.receipts("tenant-1", done.id) == ledger.receipts("tenant-1", done.id)
    assert target.get_object_ref("tenant-1", "rec-1", 1).source_refs == (_handle("s1"),)
    assert target.effect_recorded("tenant-1", active.stage_requirements[0].effect_id)
    assert target.get_tombstone("tenant-1", "rec-erased").reason_code == "admin_delete"
    with expect(ResurrectionDenied):  # fences replay before anything becomes publishable
        target.put_object_ref(ref("rec-erased", 5))
    with expect(ResurrectionDenied):
        target.accept(make_intent("new"), [ref("rec-erased", 5)])
    assert target.get_intent("tenant-1", "intent-new") is None
    redo = target.claim("restored", 30)  # lease reset: unfinished work becomes claimable again
    assert redo is not None and redo.intent_id == active.id and redo.attempt == leased.attempt + 1
    assert target.claim("restored", 30) is None  # the finished intent stays finished
    assert target.accept(make_intent("after")).version == 1
    assert [r for r in target.receipts("tenant-1", done.id)] == list(ledger.receipts("tenant-1", done.id))
    with expect(LedgerError):  # restore never merges into a populated ledger
        target_store.open_admin().restore(snapshot)
    broken = json.loads(json.dumps(snapshot))
    broken["tables"]["object_refs"].append(
        {**broken["tables"]["object_refs"][0], "record_id": "rec-erased", "content_revision": 77}
    )
    with expect(LedgerError):
        store.fresh().open_admin().restore(broken)


def case_per_aggregate_versions(store: Store) -> None:
    ledger = store.open()
    intent = make_intent()
    assert ledger.accept(intent).version == 1
    stored = ledger.get_intent("tenant-1", intent.id)
    assert stored.intent.accepted_at.year == 2020  # caller witness preserved in the body
    assert stored.accepted_at.year >= 2026  # ledger-evaluated time
    claim = ledger.claim("w", 30)
    first = ledger.record_receipt(make_receipt(intent, ProcessingStage.CANONICAL, StageStatus.PROCESSING), "tenant-1", claim)
    second = ledger.record_receipt(make_receipt(intent, ProcessingStage.VECTOR, StageStatus.PROCESSING), "tenant-1", claim)
    moved = ledger.record_receipt(make_receipt(intent, ProcessingStage.CANONICAL, StageStatus.APPLIED), "tenant-1", claim)
    assert (first, second, moved) == (1, 2, 3)  # contiguous per intent, bumped on every change
    ledger.set_status("tenant-1", intent.id, IntentStatus.PROCESSING, claim)
    assert ledger.get_intent("tenant-1", intent.id).version == 2
    other = make_intent("2")
    ledger.accept(other)
    assert ledger.get_intent("tenant-1", other.id).version == 1  # versions never cross aggregates
    assert ledger.schema_version() >= 1


_ALL_STAGES = tuple(ProcessingStage)


def case_concurrent_commits_leave_no_version_gaps(store: Store) -> None:
    ledger = store.open()
    intent = make_intent("v", stages=_ALL_STAGES)
    ledger.accept(intent)
    claim = ledger.claim("w", 120)
    versions: list[int] = []
    errors: list[BaseException] = []
    guard = threading.Lock()

    def writer(stage: ProcessingStage, status: StageStatus) -> None:
        try:
            got = store.open().record_receipt(
                make_receipt(intent, stage, status, attempt=claim.attempt), "tenant-1", claim
            )
            with guard:
                versions.append(got)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    for status in (StageStatus.PROCESSING, StageStatus.APPLIED):  # insert, then forward-move, concurrently
        threads = [threading.Thread(target=writer, args=(stage, status)) for stage in _ALL_STAGES]
        [t.start() for t in threads]
        [t.join() for t in threads]
    assert not errors, errors
    assert sorted(versions) == list(range(1, 2 * len(_ALL_STAGES) + 1))  # contiguous, unique


def case_status_based_drain_never_skips(store: Store) -> None:
    ledger = store.open()
    accepted: list[str] = []
    guard = threading.Lock()
    done = threading.Event()
    drained: list[str] = []

    def writer(n: int) -> None:
        mine = store.open()
        for i in range(20):
            result = mine.accept(make_intent(f"{n}-{i}"))
            with guard:
                accepted.append(result.intent_id)

    def drainer() -> None:
        mine = store.open()
        while True:
            finished = done.is_set()
            for entry in mine.scan_unqueued(limit=7):
                drained.append(entry.intent_id)
                mine.mark_queued(entry.tenant_id, entry.outbox_id)
            if finished and not mine.scan_unqueued(limit=1):
                return

    scanner = threading.Thread(target=drainer)
    scanner.start()
    writers = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    [t.start() for t in writers]
    [t.join() for t in writers]
    done.set()
    scanner.join()
    assert len(accepted) == 120
    assert sorted(drained) == sorted(accepted)  # nothing skipped, nothing twice
    assert ledger.scan_unqueued() == ()


def case_tenant_scoped_claim(store: Store) -> None:
    ledger = store.open()
    ledger.accept(make_intent("a", tenant="tenant-1"))
    ledger.accept(make_intent("b", tenant="tenant-2"))
    assert ledger.claim("w", 30, tenant_id="tenant-3") is None
    second = ledger.claim("w", 30, tenant_id="tenant-2")
    assert second is not None and second.tenant_id == "tenant-2" and second.intent_id == "intent-b"
    assert ledger.claim("w", 30, tenant_id="tenant-2") is None
    rest = ledger.claim("w", 30)  # unscoped visits remaining tenants
    assert rest is not None and rest.intent_id == "intent-a"
    wrong = Claim(second.outbox_id, second.intent_id, "tenant-1", second.owner, second.attempt, second.lease_expires_at)
    with expect(StaleLease):  # a claim presented under another tenant's scope is not a lease
        ledger.complete(wrong)
    ledger.complete(second)


def case_clock_step_is_fenced(store: Store) -> None:
    a, b = store.open(), store.open()
    intent = make_intent()
    a.accept(intent)
    effect = intent.stage_requirements[0].effect_id
    first = a.claim("A", 30)
    assert a.record_effect("tenant-1", effect, intent.id) is True
    store.shift_leases(-60)  # forward clock step: A's live lease now reads as expired
    second = b.claim("B", 30)
    assert second is not None and second.attempt == 2  # stolen early
    with expect(StaleLease):
        a.complete(first)
    with expect(StaleLease):  # the early-stolen holder cannot publish evidence either
        a.record_receipt(make_receipt(intent, ProcessingStage.CANONICAL, StageStatus.PROCESSING), "tenant-1", first)
    assert b.effect_recorded("tenant-1", effect) and b.record_effect("tenant-1", effect, intent.id) is False
    b.complete(second)  # safety comes from fencing + idempotent effects, not lease accuracy

    c, d = store.open(), store.open()
    c.accept(make_intent("2"))
    held = c.claim("C", 0.4)
    time.sleep(0.7)  # really expired...
    store.shift_leases(3600)  # ...until a backward step makes the stored expiry far in the future
    assert d.claim("D", 30) is None  # not stolen while the lease reads as live
    c.complete(held)
    assert held.attempt == 1 and d.claim("D", 30) is None


def case_concurrent_accepts_share_object_ref(store: Store) -> None:
    rounds, writers = 10, 4
    connections = [store.open() for _ in range(writers)]
    errors: list[BaseException] = []
    created: list[bool] = []
    guard = threading.Lock()
    for n in range(rounds):
        barrier = threading.Barrier(writers)

        def accept(i: int) -> None:
            barrier.wait()
            try:
                result = connections[i].accept(make_intent(f"{n}-{i}"), [ref(f"shared-{n}", sources=(_handle("s"),))])
                with guard:
                    created.append(result.created)
            except BaseException as exc:  # noqa: BLE001
                with guard:
                    errors.append(exc)

        threads = [threading.Thread(target=accept, args=(i,)) for i in range(writers)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    assert not errors, f"{len(errors)} spurious failures, e.g. {errors[0]!r}"
    assert created == [True] * (rounds * writers)
    assert len(connections[0].scan_unqueued(limit=1000)) == rounds * writers
    with expect(ObjectRefConflict):  # a different payload for the same revision still conflicts
        connections[0].put_object_ref(ref("shared-0", epoch=9, sources=(_handle("s"),)))


def case_receipt_attempt_is_bound_to_claim(store: Store) -> None:
    ledger = store.open()
    intent = make_intent()
    ledger.accept(intent)
    claim = ledger.claim("w", 30)
    canonical = ProcessingStage.CANONICAL
    for attempt in (claim.attempt + 1, claim.attempt + 6, claim.attempt + 49):
        with expect(ReceiptMismatch):
            ledger.record_receipt(make_receipt(intent, canonical, StageStatus.PROCESSING, attempt=attempt), "tenant-1", claim)
    assert ledger.receipts("tenant-1", intent.id) == ()
    ledger.record_receipt(make_receipt(intent, canonical, StageStatus.PROCESSING, attempt=claim.attempt), "tenant-1", claim)
    # a fence receipt needs no lease and may name any attempt
    ledger.record_receipt(make_receipt(intent, ProcessingStage.VECTOR, StageStatus.CANCELLED, attempt=9), "tenant-1")


def case_intent_ids_are_tenant_scoped(store: Store) -> None:
    ledger = store.open()
    mine, theirs = make_intent("same", tenant="tenant-1"), make_intent("same", tenant="tenant-2", key="other-key")
    assert ledger.accept(mine).created
    assert ledger.accept(theirs).created  # same id in another tenant is not an existence oracle
    with expect(DuplicateIntentId):  # still unique inside one tenant
        ledger.accept(make_intent("same", tenant="tenant-1", key="third-key"))
    assert ledger.get_intent("tenant-1", "intent-same").intent == mine
    assert ledger.get_intent("tenant-2", "intent-same").intent == theirs
    one = ledger.claim("w", 30, tenant_id="tenant-1")
    ledger.record_receipt(make_receipt(mine, ProcessingStage.CANONICAL, StageStatus.PROCESSING, attempt=one.attempt), "tenant-1", one)
    assert len(ledger.receipts("tenant-1", "intent-same")) == 1
    assert ledger.receipts("tenant-2", "intent-same") == ()
    effect = mine.stage_requirements[0].effect_id
    assert ledger.record_effect("tenant-1", effect, "intent-same") is True
    assert ledger.record_effect("tenant-2", effect, "intent-same") is True  # tenant-scoped effects
    assert ledger.set_status("tenant-1", "intent-same", IntentStatus.PROCESSING, one) is IntentStatus.PROCESSING
    assert ledger.get_intent("tenant-2", "intent-same").status is IntentStatus.ACCEPTED
    two = ledger.claim("w", 30, tenant_id="tenant-2")
    assert two is not None and two.tenant_id == "tenant-2" and two.intent_id == "intent-same"
    ledger.complete(two)


def case_snapshot_integrity(store: Store) -> None:
    ledger = store.open()
    intent = make_intent()
    ledger.accept(intent, [ref("rec-1")])
    claim = ledger.claim("w", 30)
    ledger.record_receipt(make_receipt(intent, ProcessingStage.CANONICAL, StageStatus.PROCESSING, attempt=claim.attempt), "tenant-1", claim)
    ledger.tombstone("tenant-1", "rec-erased", 2, "admin_delete")
    good = ledger.snapshot()
    assert good["digest"] == snapshot_digest(good) and good["counts"]["tombstones"] == 1

    def fresh_admin() -> LedgerPort:
        return store.fresh().open_admin()

    def rejected(mutate: Callable[[dict], None], *, resign: bool, **kwargs: Any) -> None:
        broken = json.loads(json.dumps(good))
        mutate(broken)
        if resign:
            broken["counts"] = {t: len(rows) for t, rows in broken["tables"].items()}
            broken["digest"] = snapshot_digest(broken)
        try:
            fresh_admin().restore(broken, **kwargs)
        except LedgerError:
            return
        raise AssertionError(f"tampered snapshot accepted (mutation at line {mutate.__code__.co_firstlineno})")

    rejected(lambda b: b["tables"]["intents"][0].update(status="applied"), resign=False)  # edited, digest stale
    rejected(lambda b: b["tables"]["tombstones"].clear(), resign=False)  # truncated fence
    rejected(lambda b: b.pop("digest"), resign=False)
    # a forger who recomputes counts and digest is only caught by the out-of-band digest
    rejected(lambda b: b["tables"]["tombstones"].clear(), resign=True, expected_digest=good["digest"])
    # internally consistent forgeries are caught by content checks, not the digest
    rejected(lambda b: b["tables"]["intents"][0].update(status="applied"), resign=True)  # no stage evidence
    rejected(lambda b: b["tables"]["intents"][0].update(tenant_id="tenant-9"), resign=True)  # row vs body
    rejected(lambda b: b["tables"]["stage_receipts"][0].update(stage="graph"), resign=True)
    rejected(lambda b: b["tables"]["outbox"][0].update(tenant_id="tenant-9"), resign=True)
    rejected(lambda b: b["tables"]["outbox"][0].update(state="fenced"), resign=True)  # fence without cancel
    rejected(lambda b: b["tables"]["outbox"].clear(), resign=True)  # intent without its obligation
    rejected(lambda b: b["tables"]["effects"].append(
        {"tenant_id": "tenant-1", "effect_id": "effect-undeclared", "intent_id": intent.id, "recorded_at": 1.0}
    ), resign=True)
    rejected(lambda b: b.update(schema_version=b["schema_version"] + 1), resign=True)
    rejected(lambda b: None, resign=False, expected_digest="0" * 64)  # out-of-band digest mismatch
    fresh_admin().restore(json.loads(json.dumps(good)), expected_digest=good["digest"])


def case_settled_stages_always_reach_a_terminal_status(store: Store) -> None:
    stages = (ProcessingStage.CANONICAL, ProcessingStage.VECTOR, ProcessingStage.GRAPH)
    S = StageStatus
    scenarios = {
        "cancelled+failed": ((S.CANCELLED, S.FAILED, S.FAILED), IntentStatus.FAILED),
        "applied+cancelled+superseded": ((S.APPLIED, S.CANCELLED, S.SUPERSEDED), IntentStatus.PARTIAL),
        "skipped+superseded+superseded": ((S.SKIPPED, S.SUPERSEDED, S.SUPERSEDED), IntentStatus.SUPERSEDED),
        "all-skipped": ((S.SKIPPED, S.SKIPPED, S.SKIPPED), IntentStatus.PARTIAL),
        "all-applied": ((S.APPLIED, S.APPLIED, S.APPLIED), IntentStatus.APPLIED),
    }
    ledger = store.open()
    for name, (states, expected) in scenarios.items():
        intent = make_intent(name.replace("+", "-"), stages=stages)
        ledger.accept(intent)
        claim = ledger.claim("w", 60, tenant_id="tenant-1")
        assert claim.intent_id == intent.id
        # one stage is still running so PROCESSING is legal, then everything settles
        ledger.record_receipt(make_receipt(intent, stages[0], S.PROCESSING, attempt=claim.attempt), "tenant-1", claim)
        ledger.set_status("tenant-1", intent.id, IntentStatus.PROCESSING, claim)
        for stage, state in zip(stages, states):
            ledger.record_receipt(make_receipt(intent, stage, state, attempt=claim.attempt), "tenant-1", claim)
        assert ledger.derive_status("tenant-1", intent.id) is expected, name
        assert ledger.settle("tenant-1", intent.id, claim) is expected, name
        assert ledger.get_intent("tenant-1", intent.id).status is expected
        assert ledger.claim("w2", 60, tenant_id="tenant-1") is None, name  # obligation closed, never re-leased
    # an unsettled intent keeps its derived status and cannot be settled early
    pending = make_intent("unsettled", stages=stages)
    ledger.accept(pending)
    claim = ledger.claim("w", 60, tenant_id="tenant-1")
    ledger.record_receipt(make_receipt(pending, stages[0], S.PROCESSING, attempt=claim.attempt), "tenant-1", claim)
    assert ledger.derive_status("tenant-1", pending.id) is IntentStatus.PROCESSING
    assert ledger.settle("tenant-1", pending.id, claim) is IntentStatus.PROCESSING


def case_terminal_intent_refuses_new_receipts(store: Store) -> None:
    ledger = store.open()
    intent = make_intent(stages=(ProcessingStage.CANONICAL,))
    ledger.accept(intent)
    claim = ledger.claim("w", 30)
    stage = ProcessingStage.CANONICAL
    ledger.record_receipt(make_receipt(intent, stage, StageStatus.PROCESSING, attempt=claim.attempt), "tenant-1", claim)
    ledger.set_status("tenant-1", intent.id, IntentStatus.PROCESSING, claim)
    applied = make_receipt(intent, stage, StageStatus.APPLIED, attempt=claim.attempt)
    version = ledger.record_receipt(applied, "tenant-1", claim)
    ledger.set_status("tenant-1", intent.id, IntentStatus.APPLIED, claim)
    assert ledger.receipts("tenant-1", intent.id)[0] == applied
    with expect(StaleLease):  # the closed claim cannot publish again, even an identical receipt
        ledger.record_receipt(applied, "tenant-1", claim)
    for late in (StageStatus.CANCELLED, StageStatus.SUPERSEDED):
        with expect(PublicationFenced):
            ledger.record_receipt(make_receipt(intent, stage, late, attempt=7), "tenant-1")
    assert len(ledger.receipts("tenant-1", intent.id)) == 1


def case_hold_and_ordinary_delete_never_both_commit(store: Store) -> None:
    """A hold placed while an ordinary delete runs must win or lose cleanly: never both."""

    trials, errors = 30, []
    a_conn, b_conn = store.open(), store.open()
    for n in range(trials):
        record = f"race-rec-{n}"
        barrier = threading.Barrier(2)
        outcome: dict[str, Any] = {}

        def place() -> None:
            barrier.wait(timeout=30)
            try:
                outcome["hold"] = a_conn.place_hold("tenant-1", f"hold-{n}", record, "legal", "counsel", "cred-c")
            except (ResurrectionDenied, ) as exc:
                outcome["hold"] = type(exc).__name__
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def delete() -> None:
            barrier.wait(timeout=30)
            try:
                outcome["tomb"] = b_conn.tombstone("tenant-1", record, 1, "user_delete")
            except HoldActive as exc:
                outcome["tomb"] = "HoldActive"
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=place), threading.Thread(target=delete)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        tombstoned = a_conn.get_tombstone("tenant-1", record) is not None
        held = bool(a_conn.active_holds("tenant-1", record))
        assert not (tombstoned and held), f"trial {n}: record tombstoned with an ACTIVE hold ({outcome})"
    assert not errors, errors[:3]


def case_concurrent_tombstones_of_one_record_have_one_winner(store: Store) -> None:
    results: list[Any] = []
    barrier = threading.Barrier(6)
    connections = [store.open() for _ in range(6)]

    def worker(conn) -> None:
        barrier.wait(timeout=30)
        try:
            results.append(conn.tombstone("tenant-1", "rec-one", 1, "user_delete"))
        except BaseException as exc:  # noqa: BLE001
            results.append(exc)

    threads = [threading.Thread(target=worker, args=(c,)) for c in connections]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert all(isinstance(r, bool) for r in results), results  # no raw driver errors
    assert results.count(True) == 1 and results.count(False) == 5


CASES: dict[str, Callable[[Store], None]] = {
    name.removeprefix("case_"): fn for name, fn in sorted(globals().items()) if name.startswith("case_")
}


def run_all(store_factory: Callable[[], Store]) -> dict[str, str]:
    """Script entry point: returns case -> 'ok' or the failure text."""

    out: dict[str, str] = {}
    for name, fn in CASES.items():
        try:
            fn(store_factory())
            out[name] = "ok"
        except BaseException as exc:  # noqa: BLE001
            out[name] = f"{type(exc).__name__}: {exc}"
    return out
