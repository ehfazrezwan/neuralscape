"""Generic dispatcher over the ledger port.

Producer side (``Publisher``): scan committed-but-unqueued obligations, hand
each to an enqueue callback, then mark it queued.  Delivery is at-least-once:
a crash between enqueue and mark produces a duplicate message, which is
harmless because the worker trusts the ledger, not the message.

Worker side (``StageWorker``): claim an obligation from the ledger (the queue
message is only a hint), run the registered stage handlers in dependency
order, record receipts that follow the claim's attempt, reconcile ambiguous
effects through ``record_effect``, extend the lease, and settle the aggregate
status.  "Applied" is only ever written from stored receipts, never because a
callback returned.

Handler contract
----------------
``apply(ctx)`` must be idempotent by ``ctx.effect_id`` (the external store
deduplicates on it).  The worker registers the effect in the ledger only
after ``apply`` returned, so ``effect_recorded`` means "the effect is
confirmed".  If a previous attempt confirmed the effect but died before the
applied receipt, the worker calls ``inspect(ctx)`` (when provided) to recover
the outputs without re-applying; otherwise it re-applies, relying on the
idempotency above.

Retry model: a retry is a new claim attempt.  After a retryable failure with
attempts left the worker releases the obligation (status stays PROCESSING);
stages already applied are never re-run.  Failed stages with no attempts left
settle to the terminal aggregate chosen by the lifecycle rule.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping, Protocol

logger = logging.getLogger(__name__)

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
from contracts_limits import DISCOVERY_BUDGET_MESSAGE
from contracts_references import ReferenceHandle, SourceVersion
from ledger.port import (
    Claim,
    LedgerError,
    LedgerPort,
    OutboxEntry,
    StaleLease,
    TERMINAL_STATUSES,
    expected_inputs,
)


class StageFailure(Exception):
    """Raised by a handler for an expected failure with a stable code."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code, self.safe_message, self.retryable = code, message, retryable


@dataclass(frozen=True)
class Applied:
    output_refs: tuple[ReferenceHandle, ...] = ()


@dataclass(frozen=True)
class Skipped:
    reason: str


@dataclass(frozen=True)
class Pending:
    """Not ready yet (e.g. waiting for child intents): release the claim and try again.

    Writes no outcome receipt, never counts as a failure and never consumes a retry.
    """

    delay: float = 1.0


@dataclass(frozen=True)
class StageContext:
    tenant_id: str
    intent: Intent
    requirement: StageEvidenceRequirement
    effect_id: str
    attempt: int
    owner: str
    inputs: tuple[SourceVersion, ...]  # exactly the versions this stage may act on
    prior_outputs: Mapping[ProcessingStage, tuple[ReferenceHandle, ...]]

    @property
    def stage(self) -> ProcessingStage:
        return self.requirement.stage


class StageHandler(Protocol):
    def apply(self, ctx: StageContext) -> Applied | Skipped | Pending: ...


Handlers = Mapping[object, StageHandler]


@dataclass
class RunResult:
    intent_id: str
    tenant_id: str
    status: IntentStatus
    attempt: int
    outcomes: dict[ProcessingStage, str] = field(default_factory=dict)
    released: bool = False  # a retryable failure left the obligation for another attempt
    lost_lease: bool = False
    skip_reasons: dict[ProcessingStage, str] = field(default_factory=dict)
    settle_error: str | None = None  # the ledger rejected the aggregate; the obligation was closed, not looped


def _topological(intent: Intent) -> list[StageEvidenceRequirement]:
    remaining = {item.stage: item for item in intent.stage_requirements}
    done: set[ProcessingStage] = set()
    order: list[StageEvidenceRequirement] = []
    while remaining:
        ready = [
            item for item in intent.stage_requirements
            if item.stage in remaining and set(item.upstream_stages) <= done
        ]
        if not ready:
            raise LedgerError("stage requirements contain a cycle")
        for item in ready:
            order.append(item)
            done.add(item.stage)
            del remaining[item.stage]
    return order


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Publisher:
    """Moves committed obligations onto a queue, at least once."""

    def __init__(self, ledger: LedgerPort, enqueue: Callable[[OutboxEntry], None]) -> None:
        self._ledger, self._enqueue = ledger, enqueue

    def publish_unqueued(self, limit: int = 100) -> int:
        sent = 0
        for entry in self._ledger.scan_unqueued(limit):
            if not self._ledger.take_unqueued(entry.tenant_id, entry.outbox_id, entry.attempt):
                continue  # another publisher sent it, or a worker claimed it
            try:
                self._enqueue(entry)
            except BaseException:
                self._ledger.unmark_queued(entry.tenant_id, entry.outbox_id, entry.attempt)
                raise
            sent += 1
        return sent


class StageWorker:
    def __init__(
        self,
        ledger: LedgerPort,
        handlers: Handlers,
        *,
        owner: str,
        lease_seconds: float = 30.0,
        heartbeat_seconds: float | None = None,
        max_attempts: int = 3,
        evidence_check: "Callable[[Intent, StageEvidenceRequirement, Applied], str | None] | None" = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        # Optional extra record-time check of a stage's evidence, run after the built-in ones (which already
        # include the claim validator's own budgeted validation): a failure code refuses the stage visibly.
        self._evidence_check = evidence_check
        self._prepared: dict = {}  # applied receipts validated before recording, keyed by (intent, stage, attempt)
        self._ledger, self._handlers, self._owner = ledger, handlers, owner
        self._lease, self._heartbeat, self._max_attempts = lease_seconds, heartbeat_seconds, max_attempts

    def _handler(self, operation: str, stage: ProcessingStage) -> StageHandler | None:
        return self._handlers.get((operation, stage)) or self._handlers.get(stage)

    # -------------------------------------------------------------- entry points
    def run_entry(self, entry: OutboxEntry) -> RunResult | None:
        """Queue message hook: the entry is only a hint, the claim decides."""

        return self.run_once(tenant_id=entry.tenant_id)

    def run_once(self, tenant_id: str | None = None) -> RunResult | None:
        claim = self._ledger.claim(self._owner, self._lease, tenant_id)
        if claim is None:
            return None  # duplicate or stale queue message: nothing claimable
        return self._run_claim(claim)

    # ------------------------------------------------------------------- internals
    def _run_claim(self, claim: Claim) -> RunResult:
        ledger, tenant = self._ledger, claim.tenant_id
        stored = ledger.get_intent(tenant, claim.intent_id)
        if stored is None:  # cannot happen through the port; fail safe
            raise LedgerError("claimed obligation has no intent")
        intent = stored.intent
        result = RunResult(intent.id, tenant, stored.status, claim.attempt)
        holder = _ClaimHolder(claim)
        stop = threading.Event()
        beat = None
        if self._heartbeat:
            beat = threading.Thread(target=self._beat, args=(holder, stop, result), daemon=True)
            beat.start()
        try:
            self._run_stages(intent, holder, result)
        finally:
            stop.set()
            if beat is not None:
                beat.join()
            # A receipt validated but never recorded (the lease was lost in between) must not stay for the
            # worker's life: it can hold tens of thousands of references.
            prepared = self.__dict__.get("_prepared", {})
            for key in [k for k in prepared if k[0] == intent.id and k[2] == claim.attempt]:
                prepared.pop(key, None)
        return result

    def _beat(self, holder: "_ClaimHolder", stop: threading.Event, result: RunResult) -> None:
        while not stop.wait(self._heartbeat):
            # The run freezes the holder before it settles and completes, so a
            # beat can never extend (and then misreport) a claim that finished.
            with holder.beat_lock:
                if holder.frozen:
                    return
                try:
                    holder.set(self._ledger.extend_lease(holder.get(), self._lease))
                except StaleLease:
                    result.lost_lease = True
                    return

    def _latest(self, receipts: tuple[StageReceipt, ...]) -> dict[ProcessingStage, StageReceipt]:
        latest: dict[ProcessingStage, StageReceipt] = {}
        for receipt in receipts:
            held = latest.get(receipt.stage)
            if held is None or receipt.attempt >= held.attempt:
                latest[receipt.stage] = receipt
        return latest

    def _run_stages(self, intent: Intent, holder: "_ClaimHolder", result: RunResult) -> None:
        ledger, tenant = self._ledger, intent.tenant_id
        claim = holder.get()
        applied: set[ProcessingStage] = {
            r.stage for r in ledger.receipts(tenant, intent.id) if r.status is StageStatus.APPLIED
        }
        outputs: dict[ProcessingStage, tuple[ReferenceHandle, ...]] = {
            r.stage: r.output_refs for r in ledger.receipts(tenant, intent.id) if r.status is StageStatus.APPLIED
        }
        failed_retryable = False
        pending: Pending | None = None
        latest = self._latest(ledger.receipts(tenant, intent.id))
        for requirement in _topological(intent):
            stage = requirement.stage
            if stage in applied:
                result.outcomes[stage] = "applied (earlier attempt)"
                continue
            prior = latest.get(stage)
            if (
                prior is not None
                and prior.status is StageStatus.FAILED
                and (not prior.error.retryable or prior.attempt >= self._max_attempts)
            ):
                # a settled failure from an earlier attempt is never re-run
                result.outcomes[stage] = f"failed ({prior.error.code}, earlier attempt)"
                continue
            claim = holder.get()
            try:
                holder.set(ledger.extend_lease(claim, self._lease))
            except StaleLease:
                result.lost_lease = True
                return
            claim = holder.get()
            if not set(requirement.upstream_stages) <= applied:
                self._record(intent, requirement, claim, StageStatus.SKIPPED, result)
                result.outcomes[stage] = "skipped (upstream not applied)"
                continue
            handler = self._handler(intent.operation, stage)
            if handler is None:
                self._record(intent, requirement, claim, StageStatus.SKIPPED, result)
                result.outcomes[stage] = "skipped (no handler)"
                result.skip_reasons[stage] = "capability disabled"
                continue
            try:
                self._record(intent, requirement, claim, StageStatus.PROCESSING, result)
                if result.status is IntentStatus.ACCEPTED:
                    result.status = ledger.set_status(tenant, intent.id, IntentStatus.PROCESSING, claim)
            except StaleLease:
                result.lost_lease = True
                return
            ctx = StageContext(
                tenant, intent, requirement, requirement.effect_id, claim.attempt, self._owner,
                expected_inputs(intent, requirement), dict(outputs),
            )
            try:
                outcome = self._execute(handler, ctx)
            except StageFailure as failure:
                code, message, retryable = failure.code, failure.safe_message, failure.retryable
                outcome = None
            except Exception:  # noqa: BLE001  # raw messages never reach receipts
                code, message, retryable = "handler_error", "stage handler raised an unexpected error", True
                outcome = None
            if isinstance(outcome, Applied):
                # Evidence the ledger would reject must never be recorded: it would otherwise
                # crash settle later and strand the intent (a poison pill re-claimed forever).
                rejected = self._rejected_evidence(intent, requirement, outcome, holder.get())
                if rejected is not None:
                    code, message, retryable = rejected, "stage evidence rejected", False
                    outcome = None
            try:
                if isinstance(outcome, Applied):
                    ledger.record_effect(tenant, requirement.effect_id, intent.id)
                    self._record(intent, requirement, holder.get(), StageStatus.APPLIED, result,
                                 output_refs=outcome.output_refs)
                    applied.add(stage)
                    outputs[stage] = outcome.output_refs
                    result.outcomes[stage] = "applied"
                elif isinstance(outcome, Skipped):
                    self._record(intent, requirement, holder.get(), StageStatus.SKIPPED, result)
                    result.outcomes[stage] = "skipped"
                    result.skip_reasons[stage] = outcome.reason
                elif isinstance(outcome, Pending):
                    result.outcomes[stage] = "pending"
                    pending = outcome
                else:
                    self._record(intent, requirement, holder.get(), StageStatus.FAILED, result,
                                 error=StageError(code=code, retryable=retryable, safe_message=message))
                    result.outcomes[stage] = f"failed ({code})"
                    failed_retryable = failed_retryable or retryable
            except StaleLease:
                result.lost_lease = True
                return
            except LedgerError:
                # The ledger refused the receipt itself: close the stage as failed, never crash the run.
                try:
                    self._record(intent, requirement, holder.get(), StageStatus.FAILED, result,
                                 error=StageError(code="receipt_rejected", retryable=False,
                                                  safe_message="stage evidence rejected"))
                except StaleLease:
                    result.lost_lease = True
                    return
                result.outcomes[stage] = "failed (receipt_rejected)"
            if pending is not None:
                break
        holder.freeze()  # no heartbeat may touch the lease once settling starts
        if pending is not None:
            time.sleep(pending.delay)
            try:
                ledger.release(holder.get())  # not ready: no outcome, no failure, no retry consumed
                result.released = True
            except StaleLease:
                result.lost_lease = True
            return
        self._finish(intent, holder, result, failed_retryable)

    def _rejected_evidence(
        self, intent: Intent, requirement: StageEvidenceRequirement, outcome: Applied, claim: Claim
    ) -> str | None:
        refs = outcome.output_refs
        if len(refs) < requirement.minimum_output_refs:
            return "outputs_missing"
        if any(ref.tenant_id != intent.tenant_id for ref in refs):
            return "output_tenant_mismatch"
        if len({(ref.kind, ref.id) for ref in refs}) != len(refs):
            return "output_refs_duplicate"
        rejected = self._claimable(intent, requirement, outcome, claim)
        if rejected is None and self._evidence_check is not None:
            rejected = self._evidence_check(intent, requirement, outcome)
        return rejected

    def _claimable(
        self, intent: Intent, requirement: StageEvidenceRequirement, outcome: Applied, claim: Claim
    ) -> str | None:
        """Whether the receipt set, with this applied receipt in it, is one the claim can still validate.

        The claim validator walks the intent and every receipt under one budget (the lifecycle tier), so
        evidence it cannot walk would be recorded and then never settle.  The same validation runs here,
        before the effect or the receipt is recorded, and turns a refusal into a visible stage failure.
        """

        prospective = self._build_receipt(
            intent, requirement, claim, StageStatus.APPLIED, output_refs=outcome.output_refs
        )
        try:
            problem = self._ledger.check_receipt_claimable(prospective, intent.tenant_id)
        except (ValueError, TypeError) as exc:
            problem = str(exc)
        if problem is not None:
            return "evidence_budget_exceeded" if problem == DISCOVERY_BUDGET_MESSAGE else "evidence_invalid"
        # Record exactly what was validated, so the ledger can confirm it without walking it again.
        self._prepared[(intent.id, requirement.stage, claim.attempt)] = prospective
        return None

    # Operations that act on a record's own erasure/deletion and so must run after its tombstone.
    ERASE_EXEMPT = frozenset({"memory.erase", "memory.delete"})

    def _guard_not_erased(self, ctx: StageContext) -> None:
        """Work accepted before an erasure must not re-create content for an erased record."""

        if ctx.intent.operation in self.ERASE_EXEMPT or not ctx.intent.source_preconditions:
            return
        record = ctx.intent.source_preconditions[0].record_id
        if self._ledger.get_tombstone(ctx.tenant_id, record) is not None:
            raise StageFailure("record_erased", "the target record was erased", retryable=False)

    def _execute(self, handler: StageHandler, ctx: StageContext) -> Applied | Skipped | Pending:
        self._guard_not_erased(ctx)
        ledger = self._ledger
        inspect = getattr(handler, "inspect", None)
        if ledger.effect_recorded(ctx.tenant_id, ctx.effect_id) and inspect is not None:
            found = inspect(ctx)  # a previous attempt confirmed the effect: recover, do not re-apply
            if found is not None:
                return found
        return handler.apply(ctx)

    def _build_receipt(
        self,
        intent: Intent,
        requirement: StageEvidenceRequirement,
        claim: Claim,
        status: StageStatus,
        *,
        output_refs: tuple[ReferenceHandle, ...] = (),
        error: StageError | None = None,
    ) -> StageReceipt:
        now = _now()
        applied = status is StageStatus.APPLIED
        processing = status is StageStatus.PROCESSING
        receipt = StageReceipt(
            schema_version="candidate-v1",
            intent_id=intent.id,
            attempt=claim.attempt,
            stage=requirement.stage,
            effect_id=requirement.effect_id,
            status=status,
            applied_sources=expected_inputs(intent, requirement) if applied else (),
            committed_sources=requirement.committed_sources if applied else (),
            upstream_applications=self._witnesses(intent, requirement) if applied else (),
            output_refs=output_refs,
            error=error,
            started_at=now if status in {StageStatus.PROCESSING, StageStatus.APPLIED, StageStatus.FAILED} else None,
            finished_at=None if processing else now,
        )
        if status in {StageStatus.APPLIED, StageStatus.FAILED}:
            # keep the original start of this attempt when one was recorded
            for prior in self._ledger.receipts(intent.tenant_id, intent.id):
                if prior.stage is requirement.stage and prior.attempt == claim.attempt and prior.started_at:
                    receipt = StageReceipt.model_validate(
                        {**receipt.model_dump(), "started_at": prior.started_at}
                    )
        return receipt

    def _record(
        self,
        intent: Intent,
        requirement: StageEvidenceRequirement,
        claim: Claim,
        status: StageStatus,
        result: RunResult,
        *,
        output_refs: tuple[ReferenceHandle, ...] = (),
        error: StageError | None = None,
    ) -> None:
        prepared = self.__dict__.setdefault("_prepared", {}).pop((intent.id, requirement.stage, claim.attempt), None)
        if prepared is not None and status is StageStatus.APPLIED and prepared.output_refs == output_refs:
            receipt = prepared
        else:
            receipt = self._build_receipt(intent, requirement, claim, status, output_refs=output_refs, error=error)
        self._ledger.record_receipt(receipt, intent.tenant_id, claim)

    def _witnesses(
        self, intent: Intent, requirement: StageEvidenceRequirement
    ) -> tuple[UpstreamApplication, ...]:
        """The applied upstream receipts this stage consumed, one per upstream stage."""

        if not requirement.upstream_stages:
            return ()
        latest: dict[ProcessingStage, StageReceipt] = {}
        for prior in self._ledger.receipts(intent.tenant_id, intent.id):
            if prior.status is StageStatus.APPLIED and prior.stage in requirement.upstream_stages:
                held = latest.get(prior.stage)
                if held is None or prior.attempt >= held.attempt:
                    latest[prior.stage] = prior
        return tuple(
            UpstreamApplication(
                intent_id=intent.id, stage=stage, attempt=latest[stage].attempt, effect_id=latest[stage].effect_id
            )
            for stage in requirement.upstream_stages
            if stage in latest
        )

    def _finish(self, intent: Intent, holder: "_ClaimHolder", result: RunResult, failed_retryable: bool) -> None:
        ledger, tenant, claim = self._ledger, intent.tenant_id, holder.get()
        try:
            if failed_retryable and claim.attempt < self._max_attempts:
                ledger.release(claim)  # next claim is a new attempt; applied stages are not re-run
                result.released = True
                result.status = ledger.get_intent(tenant, intent.id).status
                return
            result.status = ledger.settle(tenant, intent.id, claim)
            ledger.complete(claim)
        except StaleLease:
            result.lost_lease = True
        except LedgerError as exc:
            # The stored receipts support no legal aggregate.  Re-leasing would repeat this forever,
            # so close the obligation, keep the intent honestly unsettled and surface the reason.
            result.settle_error = str(exc)
            logger.error("settle rejected for intent %s: %s", intent.id, exc)
            try:
                ledger.complete(claim)
            except LedgerError:
                pass


class _ClaimHolder:
    """Thread-safe current claim (the heartbeat replaces it with a fresher expiry)."""

    def __init__(self, claim: Claim) -> None:
        self._claim, self._lock = claim, threading.Lock()
        self.beat_lock = threading.Lock()
        self.frozen = False

    def freeze(self) -> None:
        """Wait for an in-flight heartbeat and forbid any later one."""

        with self.beat_lock:
            self.frozen = True

    def get(self) -> Claim:
        with self._lock:
            return self._claim

    def set(self, claim: Claim) -> None:
        with self._lock:
            self._claim = claim
