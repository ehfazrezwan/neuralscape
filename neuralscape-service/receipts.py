"""Public receipt facade over ledger intents and stage receipts.

Maps the ledger's validated state into the shape clients poll.  The aggregate
status is the status the ledger stored, which can only have been written after
``validate_required_stage_claim`` accepted it against the stage receipts; this
module never derives ``applied`` from a queue result or from the absence of
errors.  Stage states come from the latest receipt per stage; a required stage
with no receipt is ``not_started``.

``legacy_task_status`` maps the aggregate onto the task-status vocabulary the
existing REST/MCP pollers understand (``queued``/``processing``/``completed``/
``failed``) so they keep working unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from contracts_lifecycle import IntentStatus, StageReceipt, StageStatus
from contracts_references import ReferenceHandle
from ledger.port import RETIRED_KEY_PREFIX, StoredIntent

#: Aggregated stage outputs (see ``write_producer.stage_output_refs``) carry this resolver.  Their id is
#: an internal storage key and is never shown to clients; the receipt lists the memories behind it.
STAGE_OUTPUTS_RESOLVER = "ns:stage-outputs"
#: The most outputs one receipt lists.  Past it the receipt says ``outputs_truncated`` and gives the
#: total; the full list is read page by page from ``GET /v1/memories/status/{task_id}/outputs``.
RECEIPT_OUTPUT_LIMIT = 256

# Poll-vocabulary mapping.  ``partial`` is reported as completed work with an
# explicit per-stage breakdown (some stage did not apply): today a best-effort
# graph failure also leaves a completed vector write.
_LEGACY_STATUS = {
    IntentStatus.ACCEPTED: "queued",
    IntentStatus.PROCESSING: "processing",
    IntentStatus.APPLIED: "completed",
    IntentStatus.PARTIAL: "completed",
    IntentStatus.FAILED: "failed",
    IntentStatus.CANCELLED: "failed",
    IntentStatus.SUPERSEDED: "failed",
}


def _latest_per_stage(receipts: Sequence[StageReceipt]) -> dict[str, StageReceipt]:
    """Greatest attempt per stage, with an APPLIED receipt always winning."""

    latest: dict[str, StageReceipt] = {}
    for receipt in receipts:
        key = receipt.stage.value
        held = latest.get(key)
        if held is None:
            latest[key] = receipt
        elif held.status is StageStatus.APPLIED:
            continue
        elif receipt.status is StageStatus.APPLIED or receipt.attempt >= held.attempt:
            latest[key] = receipt
    return latest


def stage_view(stored: StoredIntent, receipts: Sequence[StageReceipt]) -> list[dict[str, Any]]:
    latest = _latest_per_stage(receipts)
    out: list[dict[str, Any]] = []
    for requirement in stored.intent.stage_requirements:
        receipt = latest.get(requirement.stage.value)
        entry: dict[str, Any] = {
            "stage": requirement.stage.value,
            "effect_id": requirement.effect_id,
            "state": "not_started" if receipt is None else receipt.status.value,
            "attempt": 0 if receipt is None else receipt.attempt,
            "error_code": None,
            "retryable": None,
        }
        if receipt is not None and receipt.error is not None:
            entry["error_code"] = receipt.error.code
            entry["retryable"] = receipt.error.retryable
        out.append(entry)
    return out


def _expanded_outputs(
    receipts: Sequence[StageReceipt],
    resolve_outputs: Callable[[ReferenceHandle], Sequence[ReferenceHandle]] | None,
) -> tuple[list[ReferenceHandle], bool]:
    """Every applied output in stage order, with aggregated lists expanded; ``True`` if one was unavailable."""

    refs: list[ReferenceHandle] = []
    unavailable = False
    for receipt in _latest_per_stage(receipts).values():
        if receipt.status is not StageStatus.APPLIED:
            continue
        for ref in receipt.output_refs:
            if ref.resolver != STAGE_OUTPUTS_RESOLVER:
                refs.append(ref)
                continue
            if resolve_outputs is None:
                unavailable = True
                continue
            try:
                refs.extend(resolve_outputs(ref))
            except (LookupError, ValueError):  # gone, or not bound to this tenant and intent
                unavailable = True
    return refs, unavailable


def receipt_outputs(
    receipts: Sequence[StageReceipt],
    resolve_outputs: Callable[[ReferenceHandle], Sequence[ReferenceHandle]] | None = None,
) -> tuple[list[tuple[str, str]], bool]:
    """Unique ``(kind, id)`` outputs of the applied stages, in order, and whether any list was unavailable."""

    refs, unavailable = _expanded_outputs(receipts, resolve_outputs)
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for ref in refs:
        key = (ref.kind, ref.id)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out, unavailable


def output_view(kind: str, identifier: str, revision_of: Callable[[str], int] | None) -> dict[str, Any]:
    output: dict[str, Any] = {"kind": kind, "id": identifier}
    if kind == "memory" and revision_of is not None:
        revision = revision_of(identifier)
        if revision:
            output["revision"] = revision
    return output


def outputs_compacted(stored: StoredIntent) -> bool:
    """Whether retention has compacted this intent's content (its output list is no longer complete)."""

    return stored.intent.request_digest.startswith(RETIRED_KEY_PREFIX)


def build_receipt(
    stored: StoredIntent,
    receipts: Sequence[StageReceipt],
    revision_of: Callable[[str], int] | None = None,
    resolve_outputs: Callable[[ReferenceHandle], Sequence[ReferenceHandle]] | None = None,
) -> dict[str, Any]:
    """Build the public receipt for one intent.

    ``revision_of`` maps a memory id to its current revision (the number a later
    correction must name as its expected revision); omitted, no revision is
    reported.

    A large stage records one aggregated reference instead of its outputs; ``resolve_outputs`` reads the
    list behind it.  The receipt lists at most ``RECEIPT_OUTPUT_LIMIT`` outputs: past that it adds
    ``outputs_truncated``, ``outputs_total`` and ``outputs_page`` (the paging route).  A list that is no
    longer available adds ``outputs_unavailable`` instead of pretending the stage had no outputs.  The
    internal storage key of an aggregated list is never shown.
    """

    pairs, unavailable = receipt_outputs(receipts, resolve_outputs)
    # Retention compacts an intent's content after its window, outputs included: what is left is not the
    # full list, so it is reported as unavailable rather than as a (smaller) answer.
    unavailable = unavailable or outputs_compacted(stored)
    outputs = [output_view(kind, identifier, revision_of) for kind, identifier in pairs[:RECEIPT_OUTPUT_LIMIT]]
    receipt: dict[str, Any] = {
        "intent_id": stored.intent.id,
        "idempotency_key": stored.intent.idempotency_key,
        "operation": stored.intent.operation,
        "status": stored.status.value,
        "stages": stage_view(stored, receipts),
        "outputs": outputs,
    }
    if len(pairs) > RECEIPT_OUTPUT_LIMIT:
        receipt["outputs_truncated"] = True
        receipt["outputs_total"] = len(pairs)
        receipt["outputs_page"] = f"/v1/memories/status/{stored.intent.id}/outputs"
    if unavailable:
        receipt["outputs_unavailable"] = True
    return receipt


def legacy_task_status(
    task_id: str,
    stored: StoredIntent | None,
    receipts: Sequence[StageReceipt],
    revision_of: Callable[[str], int] | None = None,
    resolve_outputs: Callable[[ReferenceHandle], Sequence[ReferenceHandle]] | None = None,
) -> dict[str, Any]:
    """The existing ``get_status`` result shape, carrying the receipt."""

    if stored is None:
        return {"task_id": task_id, "status": "not_found", "result": None, "error": None}
    receipt = build_receipt(stored, receipts, revision_of, resolve_outputs)
    status = _LEGACY_STATUS[stored.status]
    error = None
    if status == "failed":
        failed = [s for s in receipt["stages"] if s["state"] == "failed"]
        error = failed[0]["error_code"] if failed else stored.status.value
    return {
        "task_id": task_id,
        "status": status,
        "result": {"receipt": receipt},
        "error": error,
    }


__all__ = [
    "RECEIPT_OUTPUT_LIMIT",
    "STAGE_OUTPUTS_RESOLVER",
    "build_receipt",
    "legacy_task_status",
    "output_view",
    "outputs_compacted",
    "receipt_outputs",
    "stage_view",
]
