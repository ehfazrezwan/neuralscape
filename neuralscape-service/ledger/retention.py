"""Retention enforcement for the durable ledger.

What is pruned, and when (windows count from an intent's terminal time on the ledger clock):

* **Idempotency records** (default 7 days).  The client key is freed: a replay inside the window
  returns the original receipt, a use after it is a new key and runs as new work.  The retired
  intent keeps its row and receipts; only the key and the request digest are neutralized.
* **Content** (default 30 days).  The payload (the memory text the intent carried) is deleted from
  the payload store, and the intent and its receipts are compacted to minimal terminal records:
  reference lists and free text are dropped, while everything the evidence rules and the snapshot
  validator need (declared effects, sources, committed generations, the required minimum outputs)
  stays, so ``snapshot()`` and ``restore()`` still verify.
* **Stale PROCESSING work** (default 24 hours without a lease or receipt).  Reviewed, not blindly
  failed: an unfinished stage whose effect is recorded, or whose effect cannot be shown absent, is
  left alone and reported as needing reconciliation.  Only when every unfinished stage is confirmed
  absent does the ledger record ``stage_expired`` failure receipts and settle.  Nothing is ever
  recorded as applied by expiry.
* **Tombstones and the audit record** are never pruned by default.  Pruning a tombstone must be
  configured, needs a configured backup expiry, and is refused unless its retention is at least that
  expiry plus the restore window (a restore of an older backup would otherwise resurrect the
  record).  Audited owner-erasure fences are never pruned here, and audit rows are append-only in
  this schema, so audit pruning is accepted as configuration but reported as unsupported.
* **Holds** block pruning of every intent that touched a held record and of its tombstone.

A sweep is idempotent and safe to repeat or run concurrently with replays.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import math

from ledger.port import RETIRED_KEY_PREFIX, ExpiryRefused, LedgerError, LedgerPort

logger = logging.getLogger(__name__)

PENDING_SUFFIX = ":retention-pending"  # markers written before the tenant was part of the key
LEGACY_ORPHANS_DONE = "retention-legacy-orphans-cleaned:"


def _tenant_suffix(tenant: str) -> str:
    return f"{PENDING_SUFFIX}:{tenant}"


def _pending_key(intent_id: str, tenant: str) -> str:
    """``<intent>:retention-pending:<tenant>``: in the intent's key space (removed with its payload) and
    selectable by tenant, so other tenants' markers never take a place in this tenant's drain."""

    return intent_id + _tenant_suffix(tenant)


DAY = 86_400.0
HOUR = 3_600.0


class RetentionRefused(LedgerError):
    """The retention configuration would risk resurrecting erased data or is inconsistent."""


@dataclass(frozen=True)
class RetentionPolicy:
    idempotency_seconds: float = 7 * DAY
    content_seconds: float = 30 * DAY
    stale_processing_seconds: float = 24 * HOUR
    # Never pruned until configured.
    tombstone_seconds: float | None = None
    audit_seconds: float | None = None
    backup_expiry_seconds: float | None = None  # the promise "backups expire within N days"; unset until a deployment states it
    restore_window_seconds: float = 30 * DAY

    def validate(self) -> "RetentionPolicy":
        for name in ("idempotency_seconds", "content_seconds", "stale_processing_seconds", "restore_window_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise RetentionRefused(f"{name} must be a positive, finite number of seconds")
        for name in ("tombstone_seconds", "audit_seconds", "backup_expiry_seconds"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            ):
                raise RetentionRefused(f"{name} must be unset or a positive, finite number of seconds")
        if self.content_seconds < self.idempotency_seconds:
            raise RetentionRefused("content retention must not end before the idempotency window")
        if self.tombstone_seconds is not None:
            if self.backup_expiry_seconds is None:
                raise RetentionRefused(
                    "tombstone retention needs a configured backup expiry: refusing to prune the fences"
                )
            floor = self.backup_expiry_seconds + self.restore_window_seconds
            if self.tombstone_seconds < floor:
                raise RetentionRefused(
                    "tombstone retention must be at least the backup expiry plus the restore window "
                    "(an older backup could otherwise resurrect an erased record)"
                )
        if self.audit_seconds is not None:
            if self.tombstone_seconds is None:
                raise RetentionRefused("audit retention needs a configured tombstone retention")
            if self.audit_seconds < self.tombstone_seconds:
                raise RetentionRefused("audit retention must be at least the tombstone retention")
        return self

    @classmethod
    def from_settings(cls, cfg: Any) -> "RetentionPolicy":
        def days(name: str) -> float | None:
            value = getattr(cfg, name, None)
            return None if value is None else float(value) * DAY

        return cls(
            idempotency_seconds=float(cfg.ledger_retention_idempotency_days) * DAY,
            content_seconds=float(cfg.ledger_retention_content_days) * DAY,
            stale_processing_seconds=float(cfg.ledger_retention_stale_processing_hours) * HOUR,
            tombstone_seconds=days("ledger_retention_tombstone_days"),
            audit_seconds=days("ledger_retention_audit_days"),
            backup_expiry_seconds=days("ledger_retention_backup_expiry_days"),
            restore_window_seconds=float(cfg.ledger_retention_restore_window_days) * DAY,
        ).validate()


#: Reports whether a stage's external effect exists: True, False, or None when it cannot be told.
EffectInspector = Callable[[str, str, str], "bool | None"]  # (tenant_id, stage, effect_id)


@dataclass
class RetentionReport:
    tenant_id: str
    expired: list[str] = field(default_factory=list)
    needs_reconciliation: list[str] = field(default_factory=list)
    idempotency_released: int = 0
    content_compacted: int = 0
    payloads_deleted: int = 0
    tombstones_pruned: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class RetentionSweeper:
    def __init__(
        self,
        ledger: LedgerPort,
        policy: RetentionPolicy,
        *,
        payloads: Any = None,
        inspector: EffectInspector | None = None,
        batch: int = 200,
        fences_only: bool = False,
    ) -> None:
        self._ledger, self._policy = ledger, policy.validate()
        self._payloads, self._inspector, self._batch = payloads, inspector, int(batch)
        self._fences_only = fences_only  # a maintenance role that may only delete tombstones
        if self._batch < 1:
            raise RetentionRefused("batch must be at least 1")

    def sweep(self, tenant_id: str) -> RetentionReport:
        report = RetentionReport(tenant_id)
        if not self._fences_only:
            self._expire_stale(tenant_id, report)
            self._idempotency(tenant_id, report)
            self._content(tenant_id, report)
        self._tombstones(tenant_id, report)
        if self._policy.audit_seconds is not None:
            report.notes.append("audit retention is configured but audit rows are append-only in this schema: not pruned")
        return report

    # ----------------------------------------------------------------- steps
    def _expire_stale(self, tenant: str, report: RetentionReport) -> None:
        """Review every stale intent in id order (paged), so unreconcilable ones never starve the rest."""

        ledger, horizon = self._ledger, self._policy.stale_processing_seconds
        after = ""
        while True:
            page = ledger.stale_processing(tenant, older_than_seconds=horizon, limit=self._batch, after_intent_id=after)
            if not page:
                return
            for stale in page:
                after = stale.intent_id
                if all(self._confirmed_absent(tenant, stage, effect) for stage, effect in stale.unfinished):
                    try:
                        ledger.expire_stale(
                            tenant, stale.intent_id, older_than_seconds=horizon,
                            expected_receipt_version=stale.receipt_version,
                        )
                    except ExpiryRefused as exc:
                        logger.warning("stale intent %s was not expired: %s", stale.intent_id, exc)
                        if exc.reason == "effect_recorded":
                            report.needs_reconciliation.append(stale.intent_id)  # it must be recovered, not failed
                        continue
                    report.expired.append(stale.intent_id)
                else:
                    report.needs_reconciliation.append(stale.intent_id)

    def _confirmed_absent(self, tenant: str, stage: str, effect_id: str) -> bool:
        if self._ledger.effect_recorded(tenant, effect_id):
            return False  # the effect was confirmed: the dispatcher must recover its outputs
        if self._inspector is None:
            return False  # absence cannot be shown from the ledger alone
        try:
            return self._inspector(tenant, stage, effect_id) is False
        except Exception:  # noqa: BLE001 - an unreadable store is "unknown", never "absent"
            logger.warning("effect inspection failed for %s", effect_id, exc_info=True)
            return False

    def _idempotency(self, tenant: str, report: RetentionReport) -> None:
        window = self._policy.idempotency_seconds
        ids = self._ledger.retention_candidates(
            tenant, stage="idempotency", older_than_seconds=window, limit=self._batch
        )
        if ids:
            report.idempotency_released += len(self._ledger.apply_retention(
                tenant, stage="idempotency", intent_ids=ids, older_than_seconds=window
            ))

    def _content(self, tenant: str, report: RetentionReport) -> None:
        if self._payloads is not None:
            # First, so it sees only intents compacted before this sweep (by a release without markers).
            self._clean_legacy_orphans(tenant, report)
        window = self._policy.content_seconds
        ids = self._ledger.retention_candidates(tenant, stage="content", older_than_seconds=window, limit=self._batch)
        changed: tuple[str, ...] = ()
        if ids:
            # A pending-delete marker goes into the payload store BEFORE the compaction commits, under the
            # intent's own key space (``<intent>:retention-pending``).  If the process dies or the store
            # fails after the compaction, the marker is what the next sweep finds and finishes; the one
            # ``delete_intent`` that removes the payload removes the marker with it.
            if self._payloads is not None:
                for intent_id in ids:
                    self._payloads.put(_pending_key(intent_id, tenant), {"tenant": tenant})
            # Compact first: the ledger re-checks terminal state and holds inside its transaction and reports
            # which intents it actually retired, so a hold placed after selection keeps its record's content.
            # A marker this sweep wrote for an intent it did not compact is left in place, never removed: a
            # concurrent sweeper may compact the intent (now or after a hold is released) and rely on the same
            # marker.  The drain ignores markers of intents that are not compacted, so a left marker licenses
            # nothing; it is finished with the payload when the intent is compacted, or dropped if it is gone.
            changed = self._ledger.apply_retention(tenant, stage="content", intent_ids=ids, older_than_seconds=window)
            report.content_compacted += len(changed)
        if self._payloads is not None:
            self._drain_pending_payloads(tenant, report)

    def _drain_pending_payloads(self, tenant: str, report: RetentionReport) -> None:
        """Finish every payload delete whose intent the ledger has already compacted, until confirmed.

        Pages through this tenant's markers only (the tenant is in the key), so markers of other tenants,
        or of tenants that are never swept again, cannot crowd them out.  Markers in the earlier key format
        are read too; malformed ones are removed (a marker never licenses a delete on its own).
        """

        for suffix, legacy in ((_tenant_suffix(tenant), False), (PENDING_SUFFIX, True)):
            after = ""
            while True:
                page = self._payloads.keys_with_suffix(suffix, self._batch * 10, after)
                if not page:
                    break
                for key, body in page:
                    after = key
                    if legacy:
                        owner = body.get("tenant") if isinstance(body, dict) else None
                        if not isinstance(owner, str) or not owner:
                            self._payloads.delete_key(key)  # malformed: nothing can ever finish it
                            continue
                        if owner != tenant:
                            continue  # that tenant's own sweep finishes it
                    self._finish_pending(tenant, key[: -len(suffix)], key, report)

    def _finish_pending(self, tenant: str, intent_id: str, key: str, report: RetentionReport) -> None:
        stored = self._ledger.get_intent(tenant, intent_id)
        if stored is None:
            self._payloads.delete_key(key)  # the intent is gone: only the marker is left
            return
        if not stored.intent.request_digest.startswith(RETIRED_KEY_PREFIX):
            return  # not compacted (yet, or held): a marker alone is never a licence to delete
        try:
            removed = int(self._payloads.delete_intent(intent_id) or 0)  # also removes the marker row
            report.payloads_deleted += max(removed - 1, 0)
        except Exception:  # noqa: BLE001 - an outage: the marker stays, the next sweep retries
            logger.warning("payload delete for %s failed; it stays pending", intent_id, exc_info=True)
            report.notes.append(f"payload delete pending for {intent_id}")

    def _clean_legacy_orphans(self, tenant: str, report: RetentionReport) -> None:
        """Once per tenant: delete the payloads of intents compacted before pending-delete markers existed.

        Deleting is idempotent, so an interrupted pass simply runs again; the completion record makes later
        sweeps skip it.
        """

        done_key = LEGACY_ORPHANS_DONE + tenant
        if self._payloads.get(done_key) is not None:
            return
        after = ""
        while True:
            ids = self._ledger.retired_intent_ids(tenant, after_intent_id=after, limit=self._batch * 10)
            if not ids:
                break
            for intent_id in ids:
                after = intent_id
                try:
                    report.payloads_deleted += int(self._payloads.delete_intent(intent_id) or 0)
                except Exception:  # noqa: BLE001 - an outage: not done; the next sweep runs the pass again
                    logger.warning("legacy payload cleanup stopped at %s", intent_id, exc_info=True)
                    report.notes.append("legacy payload cleanup incomplete")
                    return
        self._payloads.put(done_key, {"tenant": tenant})

    def _tombstones(self, tenant: str, report: RetentionReport) -> None:
        if self._policy.tombstone_seconds is None:
            return
        report.tombstones_pruned += self._ledger.prune_tombstones(
            tenant, older_than_seconds=self._policy.tombstone_seconds, limit=self._batch
        )


__all__ = ["EffectInspector", "RetentionPolicy", "RetentionRefused", "RetentionReport", "RetentionSweeper"]
