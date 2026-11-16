"""Corrections to retention found by review.  Scenarios run on SQLite here and on PostgreSQL as the runtime role
in ``test_ledger_retention_pg_integration`` (which imports ``FIX_SCENARIOS``).  Each fails on the code before the fix."""

from __future__ import annotations

from pathlib import Path

import pytest

from contracts_lifecycle import IntentStatus, ProcessingStage
from ledger.port import RETIRED_KEY_PREFIX
from ledger.retention import DAY, HOUR, LEGACY_ORPHANS_DONE as LEGACY_DONE, RetentionPolicy, RetentionRefused, RetentionSweeper
from tests.test_ledger_retention import (
    POLICY,
    T,
    Payloads,
    RetentionSqliteStore,
    _processing_intent,
    make,
    run_to_terminal,
)


# ------------------------------------------------------------------ the held record's payload (data loss)
class HoldAfterSelection:
    """Places the hold the moment the candidates were chosen, as a concurrent operator would."""

    def __init__(self, inner, record_id):
        self._inner, self._record = inner, record_id

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def retention_candidates(self, *args, **kwargs):
        found = self._inner.retention_candidates(*args, **kwargs)
        if found and kwargs.get("stage") == "content" and not self._inner.active_holds(T, self._record):
            self._inner.place_hold(T, "late-hold", self._record, "legal", "counsel", "cred-c")
        return found


def scenario_a_hold_placed_after_candidate_selection_keeps_the_payload(store) -> None:
    ledger = store.open()
    run_to_terminal(store, ledger, "keep", record="mem-keep")
    payloads = Payloads()
    payloads.put("intent-keep", {"content": "the memory a hold exists to keep"})
    payloads.put("intent-keep:vector-write-started:1", {})
    store.age(60 * DAY)
    report = RetentionSweeper(HoldAfterSelection(ledger, "mem-keep"), POLICY, payloads=payloads).sweep(T)
    assert report.content_compacted == 0
    assert "intent-keep" in payloads.rows, "a held record's payload was deleted"  # used to be gone before the re-check
    assert ledger.get_intent(T, "intent-keep").intent.target_refs  # and the intent was left whole
    ledger.release_hold(T, "late-hold", "counsel", "closed")
    again = RetentionSweeper(ledger, POLICY, payloads=payloads).sweep(T)
    assert again.content_compacted == 1 and "intent-keep" not in payloads.rows  # released: pruned in full


def scenario_payloads_are_deleted_only_for_intents_that_were_actually_compacted(store) -> None:
    ledger = store.open()
    run_to_terminal(store, ledger, "gone", record="mem-gone")
    payloads = Payloads()
    payloads.put("intent-gone", {"content": "x"})
    store.age(60 * DAY)
    report = RetentionSweeper(ledger, POLICY, payloads=payloads).sweep(T)
    assert report.content_compacted == 1 and report.payloads_deleted == 1 and "intent-gone" not in payloads.rows


# ------------------------------------------------------------------ retired rows must not starve the scan
def scenario_retired_intents_never_starve_later_candidates(store) -> None:
    ledger = store.open()
    names = [f"n{i:03d}" for i in range(45)]
    run_to_terminal(store, ledger, *names)
    store.age(60 * DAY)
    sweeper = RetentionSweeper(ledger, POLICY, batch=2)
    total_idem = total_content = 0
    for _ in range(40):  # far more sweeps than needed; each handles a batch of 2
        report = sweeper.sweep(T)
        total_idem += report.idempotency_released
        total_content += report.content_compacted
        if not report.idempotency_released and not report.content_compacted:
            break
    assert total_idem == 45 and total_content == 45, (total_idem, total_content)  # used to stop at 40
    assert all(ledger.get_intent(T, f"intent-{n}").intent.idempotency_key.startswith(RETIRED_KEY_PREFIX) for n in names)


# ------------------------------------------------------------------ unreconcilable stale intents must not starve expirable ones
def scenario_unreconcilable_stale_intents_do_not_starve_an_expirable_one(store) -> None:
    ledger = store.open()
    for name in ("a1", "a2", "z9"):  # id order puts the two unknowns ahead of the expirable one
        _processing_intent(store, ledger, name)
    store.age(32 * HOUR)
    store.shift_leases(-3 * HOUR)
    expirable_effects = {
        r.effect_id for r in ledger.get_intent(T, "intent-z9").intent.stage_requirements
    }

    def inspector(tenant, stage, effect_id):
        return False if effect_id in expirable_effects else None  # the others cannot be shown absent

    report = RetentionSweeper(ledger, POLICY, inspector=inspector, batch=2).sweep(T)
    assert report.expired == ["intent-z9"], report  # used to see only the first two and never reach it
    assert sorted(report.needs_reconciliation) == ["intent-a1", "intent-a2"]
    assert ledger.get_intent(T, "intent-a1").status is IntentStatus.PROCESSING


# ------------------------------------------------------------------ an effect that lands during review is not failed
def scenario_an_effect_recorded_during_review_is_not_failed_by_expiry(store) -> None:
    ledger = store.open()
    _processing_intent(store, ledger, "late")
    store.age(32 * HOUR)
    store.shift_leases(-3 * HOUR)
    vector_effect = next(
        r.effect_id for r in ledger.get_intent(T, "intent-late").intent.stage_requirements
        if r.stage is ProcessingStage.VECTOR
    )

    def revived(tenant, stage, effect_id):
        if effect_id == vector_effect:  # the worker revives and lands its effect while we look
            ledger.record_effect(tenant, effect_id, "intent-late")
        return False

    report = RetentionSweeper(ledger, POLICY, inspector=revived).sweep(T)
    assert report.expired == [] and report.needs_reconciliation == ["intent-late"]
    stored = ledger.get_intent(T, "intent-late")
    assert stored.status is IntentStatus.PROCESSING  # the dispatcher recovers the outputs
    assert not [r for r in ledger.receipts(T, "intent-late") if r.error and r.error.code == "stage_expired"]


def pending_markers(payloads):
    """Every pending-delete marker, in the current (tenant in the key) and the earlier format."""

    found = [k for k, _ in payloads.keys_with_suffix(f":retention-pending:{T}", 1000)]
    return found + [k for k, _ in payloads.keys_with_suffix(":retention-pending", 1000)]


# ------------------------------------------------------------------ a payload whose delete fails or crashes is retried
class FlakyPayloads:
    """A real payload store whose ``delete_intent`` can fail once (an outage) or die (a crash)."""

    def __init__(self, inner, *, fail_times=0, crash=False):
        self._inner, self.fail_times, self.crash = inner, fail_times, crash

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def delete_intent(self, intent_id):
        if self.crash:
            self.crash = False
            raise KeyboardInterrupt("process died before the payload delete")
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("payload store unavailable")
        return self._inner.delete_intent(intent_id)


class CrashAfterCompaction:
    """The ledger commits the compaction, then the process dies before any payload work."""

    def __init__(self, inner):
        self._inner, self.armed = inner, True

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def apply_retention(self, *args, **kwargs):
        changed = self._inner.apply_retention(*args, **kwargs)
        if self.armed and kwargs.get("stage") == "content" and changed:
            self.armed = False
            raise KeyboardInterrupt("process died after the compaction committed")
        return changed


def _seed_payload(store, ledger, name):
    run_to_terminal(store, ledger, name, record=f"mem-{name}")
    payloads = store.payload_store()
    payloads.put(f"intent-{name}", {"content": "memory text that must not outlive its window"})
    payloads.put(f"intent-{name}:vector:1", {"blob": 1})
    store.age(60 * DAY)
    return payloads


def scenario_a_payload_delete_that_fails_after_compaction_is_retried_until_it_succeeds(store) -> None:
    ledger = store.open()
    payloads = _seed_payload(store, ledger, "p1")
    flaky = FlakyPayloads(payloads, fail_times=1)
    first = RetentionSweeper(ledger, POLICY, payloads=flaky).sweep(T)
    assert first.content_compacted == 1 and first.payloads_deleted == 0
    assert payloads.get("intent-p1") is not None  # the outage left the text behind...
    second = RetentionSweeper(ledger, POLICY, payloads=flaky).sweep(T)
    assert payloads.get("intent-p1") is None and payloads.get("intent-p1:vector:1") is None  # ...and the next sweep removed it
    assert second.payloads_deleted >= 2
    third = RetentionSweeper(ledger, POLICY, payloads=flaky).sweep(T)
    assert third.payloads_deleted == 0 and pending_markers(payloads) == []  # confirmed: nothing left to retry


def scenario_a_crash_between_compaction_and_payload_delete_is_recovered_by_the_next_sweep(store) -> None:
    ledger = store.open()
    payloads = _seed_payload(store, ledger, "p2")
    with pytest.raises(KeyboardInterrupt):
        RetentionSweeper(CrashAfterCompaction(ledger), POLICY, payloads=payloads).sweep(T)
    assert ledger.get_intent(T, "intent-p2").intent.request_digest.startswith(RETIRED_KEY_PREFIX)  # compacted
    assert payloads.get("intent-p2") is not None
    RetentionSweeper(ledger, POLICY, payloads=payloads).sweep(T)  # a fresh process
    assert payloads.get("intent-p2") is None and pending_markers(payloads) == []


def scenario_a_crash_inside_the_payload_delete_is_recovered_by_the_next_sweep(store) -> None:
    ledger = store.open()
    payloads = _seed_payload(store, ledger, "p3")
    with pytest.raises(KeyboardInterrupt):
        RetentionSweeper(ledger, POLICY, payloads=FlakyPayloads(payloads, crash=True)).sweep(T)
    assert payloads.get("intent-p3") is not None
    RetentionSweeper(ledger, POLICY, payloads=payloads).sweep(T)
    assert payloads.get("intent-p3") is None


def scenario_a_held_intents_payload_is_kept_across_sweeps_and_pending_markers_never_delete_it(store) -> None:
    ledger = store.open()
    payloads = _seed_payload(store, ledger, "p4")
    ledger.place_hold(T, "h4", "mem-p4", "legal", "counsel", "cred-c")
    for _ in range(3):
        RetentionSweeper(ledger, POLICY, payloads=payloads).sweep(T)
        assert payloads.get("intent-p4") is not None  # a marker alone is never a licence to delete
    ledger.release_hold(T, "h4", "counsel", "closed")
    RetentionSweeper(ledger, POLICY, payloads=payloads).sweep(T)
    assert payloads.get("intent-p4") is None and pending_markers(payloads) == []


def scenario_pending_markers_of_another_tenant_are_left_to_that_tenants_sweep(store) -> None:
    ledger = store.open()
    payloads = store.payload_store()
    payloads.put("intent-foreign:retention-pending", {"tenant": "tenant-2"})
    payloads.put("intent-foreign", {"content": "belongs to another tenant's sweep"})
    RetentionSweeper(ledger, POLICY, payloads=payloads).sweep(T)
    assert payloads.get("intent-foreign") is not None


# ------------------------------------------------------------------ markers of other tenants never starve a drain
def scenario_other_tenants_and_malformed_markers_do_not_starve_this_tenants_drain(store) -> None:
    ledger = store.open()
    payloads = _seed_payload(store, ledger, "p5")
    for i in range(60):  # markers that sort first: another tenant's, and malformed ones
        payloads.put(f"intent-0{i:03d}:retention-pending", {"tenant": "tenant-other"})
        payloads.put(f"intent-00{i:03d}:retention-pending", {})
    flaky = FlakyPayloads(payloads, fail_times=1)
    RetentionSweeper(ledger, POLICY, payloads=flaky, batch=2).sweep(T)  # the delete fails once
    assert payloads.get("intent-p5") is not None
    RetentionSweeper(ledger, POLICY, payloads=payloads, batch=2).sweep(T)
    assert payloads.get("intent-p5") is None  # used to stay behind the 120 foreign and malformed markers


# ------------------------------------------------------------------ a racing sweeper cannot remove the marker another needs
class RaceAfterSelection:
    """Sweeper B selected the intent; sweeper A then compacts it before B's own compaction runs."""

    def __init__(self, inner, other_sweep):
        self._inner, self._other, self.armed = inner, other_sweep, True

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def apply_retention(self, *args, **kwargs):
        if self.armed and kwargs.get("stage") == "content":
            self.armed = False
            self._other()  # A compacts first (and its payload delete fails, leaving the marker pending)
        return self._inner.apply_retention(*args, **kwargs)


def scenario_a_racing_sweeper_never_removes_the_marker_of_an_intent_the_other_one_compacted(store) -> None:
    ledger = store.open()
    payloads = _seed_payload(store, ledger, "p6")
    sweeper_a = RetentionSweeper(ledger, POLICY, payloads=FlakyPayloads(payloads, fail_times=1))
    sweeper_b = RetentionSweeper(RaceAfterSelection(ledger, lambda: sweeper_a.sweep(T)), POLICY, payloads=payloads)
    sweeper_b.sweep(T)  # B finds nothing to change: A compacted it in between
    for _ in range(2):
        RetentionSweeper(ledger, POLICY, payloads=payloads).sweep(T)
    assert payloads.get("intent-p6") is None and payloads.get("intent-p6:vector:1") is None  # used to stay forever


class BeforeCompaction:
    """Runs ``before`` once, just before this sweeper's content compaction."""

    def __init__(self, inner, before):
        self._inner, self._before, self.armed = inner, before, True

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def apply_retention(self, *args, **kwargs):
        if self.armed and kwargs.get("stage") == "content":
            self.armed = False
            self._before()
        return self._inner.apply_retention(*args, **kwargs)


def scenario_a_hold_placed_and_released_inside_a_racing_sweep_never_loses_the_pending_delete(store) -> None:
    ledger = store.open()
    payloads = _seed_payload(store, ledger, "p9")

    def hold() -> None:
        ledger.place_hold(T, "h9", "mem-p9", "legal", "counsel", "cred-c")

    # B selected the intent; a hold lands before B compacts, so B compacts nothing.
    sweeper_b = RetentionSweeper(BeforeCompaction(ledger, hold), POLICY, payloads=payloads)

    def b_sweeps_then_the_hold_is_released() -> None:
        sweeper_b.sweep(T)
        ledger.release_hold(T, "h9", "counsel", "closed")

    # A wrote its marker; B runs in full and the hold is released before A's compaction.
    RetentionSweeper(BeforeCompaction(ledger, b_sweeps_then_the_hold_is_released), POLICY, payloads=payloads).sweep(T)
    assert ledger.get_intent(T, "intent-p9").intent.request_digest.startswith(RETIRED_KEY_PREFIX)
    for _ in range(2):
        RetentionSweeper(ledger, POLICY, payloads=payloads).sweep(T)
    assert payloads.get("intent-p9") is None and payloads.get("intent-p9:vector:1") is None  # used to stay forever
    assert pending_markers(payloads) == []


# ------------------------------------------------------------------ payloads orphaned before markers existed are cleaned once
def scenario_payloads_of_intents_compacted_before_markers_existed_are_cleaned(store) -> None:
    ledger = store.open()
    payloads = _seed_payload(store, ledger, "p7")
    RetentionSweeper(ledger, POLICY).sweep(T)  # an older release: compacted without any payload work
    assert ledger.get_intent(T, "intent-p7").intent.request_digest.startswith(RETIRED_KEY_PREFIX)
    assert payloads.get("intent-p7") is not None
    report = RetentionSweeper(ledger, POLICY, payloads=payloads).sweep(T)
    assert payloads.get("intent-p7") is None and payloads.get("intent-p7:vector:1") is None
    assert report.payloads_deleted >= 2
    again = RetentionSweeper(ledger, POLICY, payloads=payloads).sweep(T)
    assert again.payloads_deleted == 0  # one pass: later sweeps do not rescan the history


def scenario_a_marker_in_the_earlier_key_format_is_still_finished(store) -> None:
    ledger = store.open()
    payloads = _seed_payload(store, ledger, "p8")
    RetentionSweeper(ledger, POLICY).sweep(T)  # compacted without payload work...
    payloads.put(f"{LEGACY_DONE}{T}", {"tenant": T})  # ...and the one-time pass already ran
    payloads.put("intent-p8:retention-pending", {"tenant": T})  # a marker written by the previous release
    RetentionSweeper(ledger, POLICY, payloads=payloads).sweep(T)
    assert payloads.get("intent-p8") is None and pending_markers(payloads) == []


# ------------------------------------------------------------------ P-08 (C13): a hold on what a write created
def scenario_a_hold_on_the_record_a_write_created_keeps_the_creating_intents_payload(store) -> None:
    """A raw write names no memory id when it is accepted (its source record is ``write-<intent>``); the id it
    created exists only as a stage output.  A hold placed on that memory must protect the intent that created
    it, or its original content is pruned at the end of the window although the record is held."""

    ledger = store.open()
    run_to_terminal(store, ledger, "created", record="write-intent-created")
    outputs = [ref.id for receipt in ledger.receipts(T, "intent-created") for ref in receipt.output_refs]
    assert outputs and "write-intent-created" not in outputs  # the created id is only in the receipts
    payloads = Payloads()
    payloads.put("intent-created", {"content": "the memory the hold exists to keep"})
    ledger.place_hold(T, "hold-created", outputs[0], "legal", "counsel", "cred-c")
    store.age(60 * DAY)
    report = RetentionSweeper(ledger, POLICY, payloads=payloads).sweep(T)
    assert report.content_compacted == 0 and report.idempotency_released == 0
    assert "intent-created" in payloads.rows, "the held memory's creating payload was deleted"
    assert [ref.id for receipt in ledger.receipts(T, "intent-created") for ref in receipt.output_refs] == outputs
    ledger.release_hold(T, "hold-created", "counsel", "closed")
    again = RetentionSweeper(ledger, POLICY, payloads=payloads).sweep(T)
    assert again.content_compacted == 1 and "intent-created" not in payloads.rows  # released: pruned in full


def scenario_an_aggregated_output_list_is_held_while_any_hold_exists_in_the_tenant(store) -> None:
    """Past the receipt's output bound the list lives behind one handle the ledger cannot read: while the
    tenant has any active hold, that intent is left whole rather than guessed at."""

    from contracts_references import ReferenceHandle
    from ledger.dispatch import Applied, StageContext
    from tests.test_ledger_dispatch import Handler, all_handlers, drain
    from ledger.dispatch import StageWorker
    from receipts import STAGE_OUTPUTS_RESOLVER

    class Aggregated(Handler):
        def apply(self, ctx: StageContext) -> Applied:
            self.external.put(ctx)
            return Applied((ReferenceHandle(kind="artifact", id=f"{ctx.intent.id}:outputs-job:1", tenant_id=ctx.tenant_id,
                                            resolver=STAGE_OUTPUTS_RESOLVER),))

    ledger = store.open()
    external = store.external()
    ledger.accept(make("big", record="write-big"))
    drain(StageWorker(ledger, all_handlers(external, **{ProcessingStage.VECTOR: Aggregated(external)}), owner="w"), limit=4)
    assert ledger.get_intent(T, "intent-big").status is IntentStatus.APPLIED
    ledger.place_hold(T, "hold-other", "mem-somewhere", "legal", "counsel", "cred-c")
    store.age(60 * DAY)
    assert RetentionSweeper(ledger, POLICY).sweep(T).content_compacted == 0
    ledger.release_hold(T, "hold-other", "counsel", "closed")
    assert RetentionSweeper(ledger, POLICY).sweep(T).content_compacted == 1


FIX_SCENARIOS = {
    name.removeprefix("scenario_"): fn for name, fn in sorted(globals().items()) if name.startswith("scenario_")
}


@pytest.mark.parametrize("name", sorted(FIX_SCENARIOS))
def test_retention_fixes_on_sqlite(name: str, tmp_path: Path) -> None:
    import sqlite3

    store = RetentionSqliteStore(tmp_path)

    def shift_leases(seconds: float) -> None:
        raw = sqlite3.connect(store.path, isolation_level=None)
        raw.execute("UPDATE outbox SET lease_until = lease_until + ? WHERE state='leased'", (seconds,))
        raw.close()

    store.shift_leases = shift_leases  # type: ignore[attr-defined]
    try:
        FIX_SCENARIOS[name](store)
    finally:
        store.close()


# ------------------------------------------------------------------ a non-finite window is refused at startup
@pytest.mark.parametrize("field", ["idempotency_seconds", "content_seconds", "stale_processing_seconds",
                                   "restore_window_seconds"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_window_is_refused_when_the_policy_is_built(field, bad) -> None:
    with pytest.raises(RetentionRefused, match=field):
        RetentionPolicy(**{field: bad}).validate()


@pytest.mark.parametrize("field", ["tombstone_seconds", "audit_seconds", "backup_expiry_seconds"])
def test_a_non_finite_optional_window_is_refused_too(field) -> None:
    with pytest.raises(RetentionRefused, match=field):
        RetentionPolicy(**{field: float("nan")}).validate()


def test_a_nan_setting_is_refused_by_from_settings_not_by_the_nightly_sweep() -> None:
    from config import Settings

    with pytest.raises(RetentionRefused, match="idempotency_seconds"):
        RetentionPolicy.from_settings(Settings(_env_file=None, ledger_retention_idempotency_days=float("nan")))
