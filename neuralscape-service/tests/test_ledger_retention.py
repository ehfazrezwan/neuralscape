"""Retention enforcement: windows, compaction, stale work, tombstone refusal, holds and restore integrity.

The scenarios take a ``store`` (SQLite here, PostgreSQL as the runtime role in
``test_ledger_retention_pg_integration``).  Time is moved by shifting the stored timestamps, so the same
scenarios run on both backends without a fake clock.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from contracts_lifecycle import IntentStatus, ProcessingStage, StageStatus
from contracts_references import ReferenceHandle, SourceVersion
from ledger.dispatch import Applied, StageContext, StageFailure, StageWorker
from ledger.port import IdempotencyConflict, LedgerError, RETIRED_KEY_PREFIX
from ledger.retention import DAY, HOUR, RetentionPolicy, RetentionRefused, RetentionSweeper
from ledger.stages import build_intent
from tests.test_ledger_dispatch import External, Handler, SqliteStore, all_handlers, drain

T = "tenant-1"
G = ProcessingStage.GRAPH
POLICY = RetentionPolicy()  # 7 d idempotency, 30 d content, 24 h stale, tombstones and audit never


class RetentionSqliteStore(SqliteStore):
    def age(self, seconds: float) -> None:
        raw = sqlite3.connect(self.path, isolation_level=None)
        for table, column in (
            ("intents", "updated_at"), ("intents", "accepted_at"), ("stage_receipts", "recorded_at"),
            ("tombstones", "recorded_at"), ("effects", "recorded_at"), ("object_refs", "recorded_at"),
        ):
            raw.execute(f"UPDATE {table} SET {column} = {column} - ?", (seconds,))
        raw.close()

    def payload_store(self):
        from write_producer import SqlitePayloadStore

        store = SqlitePayloadStore(str(self.root / "payloads.db"))
        self.opened_payloads = getattr(self, "opened_payloads", []) + [store]
        return store

    def fresh_target(self):
        other = RetentionSqliteStore(self.root / "restore-target")
        other.root.mkdir(exist_ok=True)
        other.path = other.root / "ledger.db"
        return other


def make(name: str, *, record: str = "mem-1", key: str | None = None, digest: str = "sha256:a", tenant: str = T):
    return build_intent(
        "memory.write.raw", intent_id=f"intent-{name}", tenant_id=tenant, actor_id="actor-1", credential_id="cred-1",
        idempotency_key=key or f"key-{name}", request_digest=digest, correlation_id=f"corr-{name}",
        preconditions=(SourceVersion(record_id=record, content_revision=3, policy_epoch=7),),
        committed=(SourceVersion(record_id=record, content_revision=4, policy_epoch=7),),
        target_refs=(ReferenceHandle(kind="memory", id=record, tenant_id=tenant, resolver="r"),),
    )


def run_to_terminal(store, ledger, *names: str, **make_kwargs) -> None:
    external = store.external()
    for name in names:
        ledger.accept(make(name, **make_kwargs))
    drain(StageWorker(ledger, all_handlers(external), owner="w"), limit=len(names) + 2)
    for name in names:
        assert ledger.get_intent(T, f"intent-{name}").status is IntentStatus.APPLIED


class Payloads:
    """The payload store's delete surface, with the content an intent carried."""

    def __init__(self) -> None:
        self.rows = {}

    def put(self, key, body):
        self.rows[key] = body

    def delete_intent(self, intent_id):
        gone = [k for k in self.rows if k == intent_id or k.startswith(intent_id + ":")]
        for key in gone:
            del self.rows[key]
        return len(gone)

    def get(self, key):
        return self.rows.get(key)

    def delete_key(self, key):
        return 1 if self.rows.pop(key, None) is not None else 0

    def keys_with_suffix(self, suffix, limit, after=""):
        return tuple((k, v) for k, v in sorted(self.rows.items()) if k > after and k.endswith(suffix))[:limit]


# --------------------------------------------------------------------------- scenarios
def scenario_idempotency_replays_inside_the_window_and_is_new_work_after_it(store) -> None:
    ledger = store.open()
    run_to_terminal(store, ledger, "a")
    sweeper = RetentionSweeper(ledger, POLICY)
    store.age(6 * DAY)  # inside the window
    assert sweeper.sweep(T).idempotency_released == 0
    again = ledger.accept(make("a"))
    assert again.created is False and again.intent_id == "intent-a"  # the original receipt
    with pytest.raises(IdempotencyConflict):
        ledger.accept(make("a", digest="sha256:other"))  # a changed request is still a conflict inside the window
    store.age(2 * DAY)  # now past 7 days
    assert sweeper.sweep(T).idempotency_released == 1
    stored = ledger.get_intent(T, "intent-a")
    assert stored.intent.idempotency_key.startswith(RETIRED_KEY_PREFIX) and stored.status is IntentStatus.APPLIED
    fresh = ledger.accept(make("a2", key="key-a", digest="sha256:other"))  # the same client key: a new key now
    assert fresh.created is True and fresh.intent_id == "intent-a2"
    assert sweeper.sweep(T).idempotency_released == 0  # idempotent
    assert ledger.snapshot()["digest"]  # the ledger still snapshots


def scenario_content_is_compacted_after_its_window_and_the_ledger_still_restores(store) -> None:
    ledger = store.open()
    external = store.external()
    many = [ReferenceHandle(kind="artifact", id=f"extra-{i}", tenant_id=T, resolver="r") for i in range(4)]

    class Bulky(Handler):
        def apply(self, ctx: StageContext) -> Applied:
            return Applied(tuple(many) + super().apply(ctx).output_refs)

    class Refusing(Handler):
        def apply(self, ctx: StageContext):
            raise StageFailure("graph_down", "the graph said: secret-detail-123", retryable=False)

    ledger.accept(make("c"))
    handlers = all_handlers(external, **{ProcessingStage.VECTOR: Bulky(external), G: Refusing(external)})
    drain(StageWorker(ledger, handlers, owner="w"), limit=3)
    before = ledger.get_intent(T, "intent-c")
    assert before.status is IntentStatus.PARTIAL
    originals = ledger.receipts(T, "intent-c")
    assert any(r.error and "secret-detail-123" in r.error.safe_message for r in originals)
    payloads = Payloads()
    payloads.put("intent-c", {"content": "the memory text"})
    payloads.put("intent-c:delete:1", {"result": "blob"})
    payloads.put("intent-other", {"content": "someone else"})
    sweeper = RetentionSweeper(ledger, POLICY, payloads=payloads)
    store.age(20 * DAY)
    assert sweeper.sweep(T).content_compacted == 0  # inside the 30-day content window
    store.age(15 * DAY)
    report = sweeper.sweep(T)
    assert report.content_compacted == 1 and report.payloads_deleted == 2
    assert "intent-c" not in payloads.rows and "intent-c:delete:1" not in payloads.rows and "intent-other" in payloads.rows
    after = ledger.get_intent(T, "intent-c")
    assert after.status is before.status  # the aggregate is unchanged
    assert after.intent.target_refs == () and after.intent.correlation_id == "intent-c"
    assert after.intent.request_digest.startswith(RETIRED_KEY_PREFIX)
    compacted = ledger.receipts(T, "intent-c")
    assert [r.status for r in compacted] == [r.status for r in originals]
    assert all("secret-detail-123" not in (r.error.safe_message if r.error else "") for r in compacted)
    vector = next(r for r in compacted if r.stage is ProcessingStage.VECTOR)
    minimum = next(r.minimum_output_refs for r in after.intent.stage_requirements if r.stage is ProcessingStage.VECTOR)
    original = next(r for r in originals if r.stage is ProcessingStage.VECTOR)
    assert len(original.output_refs) > minimum and len(vector.output_refs) == minimum  # only the required minimum stays
    assert ledger.derive_status(T, "intent-c") is before.status
    assert sweeper.sweep(T).content_compacted == 0  # idempotent
    # snapshot and restore still verify after pruning
    snapshot = ledger.snapshot()
    target = store.fresh_target().open()
    target.restore(snapshot, expected_digest=snapshot["digest"])
    assert target.get_intent(T, "intent-c").status is before.status
    assert target.snapshot()["digest"] == snapshot["digest"]


def scenario_a_held_record_blocks_pruning_of_its_intents_and_release_unblocks_it(store) -> None:
    ledger = store.open()
    run_to_terminal(store, ledger, "held", record="mem-held")
    run_to_terminal(store, ledger, "free", record="mem-free")
    ledger.place_hold(T, "hold-1", "mem-held", "legal", "counsel", "cred-c")
    store.age(40 * DAY)
    report = RetentionSweeper(ledger, POLICY).sweep(T)
    assert report.idempotency_released == 1 and report.content_compacted == 1
    assert not ledger.get_intent(T, "intent-held").intent.idempotency_key.startswith(RETIRED_KEY_PREFIX)
    assert ledger.get_intent(T, "intent-held").intent.target_refs  # content untouched while held
    assert ledger.get_intent(T, "intent-free").intent.idempotency_key.startswith(RETIRED_KEY_PREFIX)
    ledger.release_hold(T, "hold-1", "counsel", "case closed")
    again = RetentionSweeper(ledger, POLICY).sweep(T)
    assert again.idempotency_released == 1 and again.content_compacted == 1
    assert ledger.get_intent(T, "intent-held").intent.target_refs == ()


def _processing_intent(store, ledger, name: str, *, hook=None):
    """An intent whose canonical stage applied and whose vector stage is left PROCESSING (a dead worker)."""

    external = store.external()

    class Dies(Handler):
        def apply(self, ctx: StageContext):
            raise KeyboardInterrupt("worker died")

    ledger.accept(make(name))
    worker = StageWorker(ledger, all_handlers(external, **{ProcessingStage.VECTOR: Dies(external)}), owner="dead",
                         lease_seconds=30)
    with pytest.raises(KeyboardInterrupt):
        worker.run_once()
    return external


def scenario_stale_processing_is_expired_only_when_every_effect_is_confirmed_absent(store) -> None:
    ledger = store.open()
    _processing_intent(store, ledger, "dead")
    store.age(2 * HOUR)
    inside = RetentionSweeper(ledger, POLICY).sweep(T)  # lease still held by the dead worker, no horizon passed
    assert inside.expired == [] and inside.needs_reconciliation == []
    store.shift_leases(-3 * HOUR)  # the dead worker's lease has expired
    store.age(30 * HOUR)
    applied_before = {(r.stage, r.attempt) for r in ledger.receipts(T, "intent-dead") if r.status is StageStatus.APPLIED}
    # no inspector: absence cannot be shown, so nothing is written
    unknown = RetentionSweeper(ledger, POLICY).sweep(T)
    assert unknown.expired == [] and unknown.needs_reconciliation == ["intent-dead"]
    assert ledger.get_intent(T, "intent-dead").status is IntentStatus.PROCESSING
    # an effect that is recorded must be recovered by the dispatcher, not failed
    ledger.record_effect(T, next(r.effect_id for r in ledger.get_intent(T, "intent-dead").intent.stage_requirements
                                 if r.stage is ProcessingStage.VECTOR), "intent-dead")
    present = RetentionSweeper(ledger, POLICY, inspector=lambda *_: False).sweep(T)
    assert present.expired == [] and present.needs_reconciliation == ["intent-dead"]


def scenario_expiry_writes_explicit_failure_receipts_and_never_an_applied_one(store) -> None:
    ledger = store.open()
    _processing_intent(store, ledger, "gone")
    store.age(32 * HOUR)
    store.shift_leases(-3 * HOUR)
    asked = []
    sweeper = RetentionSweeper(ledger, POLICY, inspector=lambda tenant, stage, effect: asked.append(stage) or False)
    report = sweeper.sweep(T)
    assert report.expired == ["intent-gone"] and report.needs_reconciliation == []
    stored = ledger.get_intent(T, "intent-gone")
    assert stored.status is IntentStatus.PARTIAL  # canonical applied, the rest failed: honest, never APPLIED
    receipts = ledger.receipts(T, "intent-gone")
    expired = [r for r in receipts if r.error and r.error.code == "stage_expired"]
    assert {r.stage.value for r in expired} == {"vector", "graph", "markdown"}
    assert all(r.status is StageStatus.FAILED and r.error.retryable is False for r in expired)
    assert not [r for r in receipts if r.status is StageStatus.APPLIED and r.stage.value != "canonical"]
    assert ledger.derive_status(T, "intent-gone") is IntentStatus.PARTIAL
    assert sweeper.sweep(T).expired == []  # a settled intent is not revisited


def scenario_expiry_refuses_when_the_history_moved_or_a_lease_is_live(store) -> None:
    ledger = store.open()
    run_to_terminal(store, ledger, "done")
    _processing_intent(store, ledger, "race")
    store.age(30 * HOUR)
    assert ledger.stale_processing(T, older_than_seconds=24 * HOUR, limit=10) == ()  # the dead worker's lease is still live
    store.shift_leases(-3 * HOUR)
    (candidate,) = ledger.stale_processing(T, older_than_seconds=24 * HOUR, limit=10)
    with pytest.raises(LedgerError, match="receipt history changed"):
        ledger.expire_stale(T, "intent-race", older_than_seconds=24 * HOUR,
                            expected_receipt_version=candidate.receipt_version + 1)
    with pytest.raises(LedgerError, match="only a PROCESSING intent"):
        ledger.expire_stale(T, "intent-done", older_than_seconds=1, expected_receipt_version=0)
    assert ledger.get_intent(T, "intent-race").status is IntentStatus.PROCESSING


def scenario_tombstones_and_audit_are_never_pruned_by_default(store) -> None:
    ledger = store.open()
    ledger.accept(make("t"), [])
    ledger.tombstone(T, "mem-gone", 3, "user_delete")
    store.age(10 * 365 * DAY)
    report = RetentionSweeper(ledger, POLICY).sweep(T)
    assert report.tombstones_pruned == 0 and ledger.get_tombstone(T, "mem-gone") is not None


def scenario_tombstone_pruning_is_refused_without_a_backup_expiry_and_a_safe_margin(store) -> None:
    ledger = store.open()
    for policy in (
        RetentionPolicy(tombstone_seconds=400 * DAY),  # no backup expiry configured
        RetentionPolicy(tombstone_seconds=100 * DAY, backup_expiry_seconds=90 * DAY),  # 100 < 90 + 30
        RetentionPolicy(tombstone_seconds=120 * DAY, backup_expiry_seconds=90 * DAY, audit_seconds=60 * DAY),
        RetentionPolicy(audit_seconds=500 * DAY),  # audit needs a tombstone retention
        RetentionPolicy(idempotency_seconds=10 * DAY, content_seconds=5 * DAY),
    ):
        with pytest.raises(RetentionRefused):
            RetentionSweeper(ledger, policy)
    ledger.tombstone(T, "mem-gone", 3, "user_delete")
    store.age(10 * 365 * DAY)
    assert ledger.get_tombstone(T, "mem-gone") is not None  # a refused configuration pruned nothing


def scenario_configured_tombstone_pruning_keeps_audited_owner_erasure_fences(store) -> None:
    from ledger.conformance import make_intent, ref
    from tests.test_ledger_erasure import erase, good_owner

    ledger = store.open()
    pruner = store.open_pruner() if hasattr(store, "open_pruner") else ledger  # PostgreSQL: a separate role
    ledger.tombstone(T, "mem-ordinary", 3, "user_delete")
    ledger.accept(make_intent("seed"), [ref("rec-1")])
    erase(ledger, good_owner())  # an audited owner erasure: tombstone reason owner_erasure, plus an audit row
    store.age(400 * DAY)
    ledger.tombstone(T, "mem-recent", 3, "user_delete")  # fenced just now: inside every window
    policy = RetentionPolicy(tombstone_seconds=120 * DAY, backup_expiry_seconds=90 * DAY)
    report = RetentionSweeper(pruner, policy, fences_only=pruner is not ledger).sweep(T)
    assert report.tombstones_pruned == 1
    assert ledger.get_tombstone(T, "mem-ordinary") is None  # aged out
    assert ledger.get_tombstone(T, "mem-recent") is not None  # too young
    assert ledger.get_tombstone(T, "rec-1") is not None  # audited erasure fence is kept
    assert ledger.audit_records(T, "rec-1")  # and so is its audit
    snapshot = ledger.snapshot()  # the remaining ledger still validates
    store.fresh_target().open().restore(snapshot, expected_digest=snapshot["digest"])


def scenario_the_ledger_snapshot_and_restore_still_verify_after_every_stage_of_pruning(store) -> None:
    ledger = store.open()
    run_to_terminal(store, ledger, "p1", "p2", "p3")
    store.age(60 * DAY)
    RetentionSweeper(ledger, POLICY).sweep(T)
    snapshot = ledger.snapshot()
    target = store.fresh_target().open()
    target.restore(snapshot, expected_digest=snapshot["digest"])
    assert target.snapshot()["digest"] == snapshot["digest"]
    for name in ("p1", "p2", "p3"):
        assert target.get_intent(T, f"intent-{name}").status is IntentStatus.APPLIED
        assert target.derive_status(T, f"intent-{name}") is IntentStatus.APPLIED


def scenario_a_concurrent_replay_during_pruning_sees_the_original_or_clean_new_work(store) -> None:
    ledger = store.open()
    run_to_terminal(store, ledger, "r")
    store.age(8 * DAY)
    errors: list[BaseException] = []
    outcomes: list[tuple[str, bool]] = []
    gate = threading.Barrier(5)

    def replay(n: int) -> None:
        try:
            gate.wait()
            for i in range(10):
                try:
                    result = store.open().accept(make("r"))  # same id and key as the original
                    outcomes.append((result.intent_id, result.created))
                except LedgerError as exc:
                    if type(exc).__name__ != "DuplicateIntentId":
                        raise
                    outcomes.append(("duplicate-after-retirement", False))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def prune() -> None:
        try:
            gate.wait()
            RetentionSweeper(store.open(), POLICY).sweep(T)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=replay, args=(n,)) for n in range(4)] + [threading.Thread(target=prune)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    # every replay either returned the original receipt or hit the retired id (a clean refusal the producer retries)
    assert all(o in (("intent-r", False), ("duplicate-after-retirement", False)) for o in outcomes), outcomes
    stored = ledger.get_intent(T, "intent-r")
    assert stored.status is IntentStatus.APPLIED and stored.intent.idempotency_key.startswith(RETIRED_KEY_PREFIX)
    snapshot = ledger.snapshot()
    store.fresh_target().open().restore(snapshot, expected_digest=snapshot["digest"])


SCENARIOS = {
    name.removeprefix("scenario_"): fn for name, fn in sorted(globals().items()) if name.startswith("scenario_")
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_retention_scenarios_on_sqlite(name: str, tmp_path: Path) -> None:
    store = RetentionSqliteStore(tmp_path)

    def shift_leases(seconds: float) -> None:
        raw = sqlite3.connect(store.path, isolation_level=None)
        raw.execute("UPDATE outbox SET lease_until = lease_until + ? WHERE state='leased'", (seconds,))
        raw.close()

    store.shift_leases = shift_leases  # type: ignore[attr-defined]
    try:
        SCENARIOS[name](store)
    finally:
        store.close()
