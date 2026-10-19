"""Online cutover of legacy writers to the durable ledger.

Phases (one direction, with a rollback until ``complete``):

``legacy``       writes use the legacy queue; nothing is accepted by the ledger.
``dual_accept``  new writes are accepted by the ledger; legacy jobs already queued still run.
``ledger_only``  the legacy queue is drained and fenced: a legacy job admitted after the fence is
                 not executed and reports an explicit ``fenced`` status; one admitted before the
                 fence completes.
``complete``     ``ledger_only`` plus a passed final reconciliation; rollback is refused because
                 the legacy path can no longer interpret the ledger's revisions and fences.

State is an append-only sequence of put-once records in the payload store (a record's key is its
sequence number, so two controllers racing for the next phase cannot both win).  No ledger
migration is needed, and the same code runs on the SQLite and PostgreSQL payload stores.

The fence is enforced where the work runs: a legacy worker asks :func:`legacy_admission` before
executing a write job.  A stale phase in a lagging API replica can therefore enqueue one more legacy
job, but never execute it unfenced and never lose it silently.
"""

from __future__ import annotations

import enum
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol


class Phase(str, enum.Enum):
    LEGACY = "legacy"
    DUAL_ACCEPT = "dual_accept"
    LEDGER_ONLY = "ledger_only"
    COMPLETE = "complete"


class CutoverError(Exception):
    """A transition the current state does not allow, or a failed precondition."""


class CutoverConflict(CutoverError):
    """Another controller recorded the next state first."""


class KeyValueStore(Protocol):
    """The payload store: put-once keys (the first value for a key wins)."""

    def put(self, key: str, payload: dict[str, Any]) -> None: ...

    def get(self, key: str) -> dict[str, Any] | None: ...


class SequencedLog:
    """Append-only sequence of put-once records ``<name>:<n>`` (n starts at 1)."""

    def __init__(self, store: KeyValueStore, name: str) -> None:
        self._store, self._name = store, name

    def _key(self, n: int) -> str:
        return f"{self._name}:{n}"

    def length(self) -> int:
        """The highest written sequence number (records are contiguous): gallop, then bisect."""

        if self._store.get(self._key(1)) is None:
            return 0
        high = 1
        while self._store.get(self._key(high * 2)) is not None:
            high *= 2
        low, high = high, high * 2  # low exists, high does not
        while high - low > 1:
            mid = (low + high) // 2
            if self._store.get(self._key(mid)) is not None:
                low = mid
            else:
                high = mid
        return low

    def latest(self) -> tuple[int, dict[str, Any] | None]:
        n = self.length()
        return n, (self._store.get(self._key(n)) if n else None)

    def get(self, n: int) -> dict[str, Any] | None:
        return self._store.get(self._key(n))

    def append(self, after: int, record: dict[str, Any]) -> int:
        """Write record ``after + 1``; raises :class:`CutoverConflict` if someone else did first."""

        nonce = secrets.token_hex(8)
        mine = {**record, "_nonce": nonce}
        self._store.put(self._key(after + 1), mine)
        stored = self._store.get(self._key(after + 1))
        if stored is None or stored.get("_nonce") != nonce:
            raise CutoverConflict(f"{self._name} sequence {after + 1} was written by another controller")
        return after + 1


def _utc_iso(now: float) -> str:
    return datetime.fromtimestamp(now, timezone.utc).isoformat()


@dataclass(frozen=True)
class CutoverState:
    phase: Phase
    sequence: int
    fence_at: float | None  # epoch seconds of the move to ``ledger_only`` (legacy jobs enqueued later are fenced)
    rolled_back: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "sequence": self.sequence,
            "fence_at": None if self.fence_at is None else _utc_iso(self.fence_at),
            "rolled_back": self.rolled_back,
            "note": self.note,
        }


_NEW_STATE = CutoverState(Phase.LEGACY, 0, None)


class BackfillPort(Protocol):
    def run(self, max_pages: int | None = None) -> Any: ...

    def reconcile(self) -> Any: ...


class CutoverController:
    """Persistent, race-safe phase machine over a backfill and a legacy in-flight probe."""

    def __init__(
        self,
        store: KeyValueStore,
        *,
        backfill: BackfillPort | None = None,
        legacy_inflight: Callable[[], int] | None = None,
        clock: Callable[[], float] = time.time,
        cache_seconds: float = 1.0,
    ) -> None:
        self._log = SequencedLog(store, "cutover")
        self._backfill, self._inflight, self._clock = backfill, legacy_inflight, clock
        self._cache_seconds = cache_seconds
        self._cached: tuple[float, CutoverState] | None = None

    # ------------------------------------------------------------------ reading
    def state(self, *, fresh: bool = False) -> CutoverState:
        now = self._clock()
        if not fresh and self._cached is not None and now - self._cached[0] < self._cache_seconds:
            return self._cached[1]
        n, record = self._log.latest()
        state = _NEW_STATE if record is None else CutoverState(
            Phase(record["phase"]), n, record.get("fence_at"), bool(record.get("rolled_back")), record.get("note", "")
        )
        self._cached = (now, state)
        return state

    def writes_use_ledger(self) -> bool:
        return self.state().phase is not Phase.LEGACY

    def status(self) -> dict[str, Any]:
        state = self.state(fresh=True)
        out = state.as_dict()
        out["legacy_inflight"] = None if self._inflight is None else int(self._inflight())
        return out

    # --------------------------------------------------------------- transitions
    def _move(self, current: CutoverState, phase: Phase, **fields: Any) -> CutoverState:
        record = {"phase": phase.value, "at": _utc_iso(self._clock()), **fields}
        n = self._log.append(current.sequence, record)
        self._cached = None
        return CutoverState(
            phase, n, fields.get("fence_at", current.fence_at if phase is Phase.COMPLETE else None),
            bool(fields.get("rolled_back")), fields.get("note", ""),
        )

    def start(self) -> CutoverState:
        """``legacy`` -> ``dual_accept``: new writes go through the ledger.  Idempotent once started."""

        current = self.state(fresh=True)
        if current.phase is Phase.DUAL_ACCEPT:
            return current
        if current.phase is not Phase.LEGACY:
            raise CutoverError(f"cannot start from {current.phase.value}")
        return self._move(current, Phase.DUAL_ACCEPT)

    def backfill_step(self, max_pages: int | None = None) -> Any:
        if self._backfill is None:
            raise CutoverError("no backfill is configured")
        return self._backfill.run(max_pages)

    def advance(self) -> CutoverState:
        """``dual_accept`` -> ``ledger_only``: needs a drained legacy queue and a clean reconciliation.

        The legacy queue is checked before and after the backfill pass and reconciliation, so a job
        that finishes during the pass cannot leave a row the reconciliation never saw.
        """

        current = self.state(fresh=True)
        if current.phase is Phase.LEDGER_ONLY:
            return current
        if current.phase is not Phase.DUAL_ACCEPT:
            raise CutoverError(f"cannot advance from {current.phase.value}")
        if self._backfill is None or self._inflight is None:
            raise CutoverError("advance needs a backfill and a legacy in-flight probe")
        if self._inflight() != 0:
            raise CutoverError("legacy jobs are still queued or running; drain them first")
        self._backfill.run(None)
        report = self._backfill.reconcile()
        if not getattr(report, "ok", False):
            raise CutoverError(f"reconciliation is not clean: {getattr(report, 'summary', lambda: report)()}")
        if self._inflight() != 0:
            raise CutoverError("a legacy job started during reconciliation; retry")
        return self._move(current, Phase.LEDGER_ONLY, fence_at=self._clock())

    def complete(self) -> CutoverState:
        """``ledger_only`` -> ``complete`` after a clean final reconciliation.

        It reconciles but does not run a backfill pass: when the reconciliation is unclean the
        operator runs the backfill step first and retries.
        """

        current = self.state(fresh=True)
        if current.phase is Phase.COMPLETE:
            return current
        if current.phase is not Phase.LEDGER_ONLY:
            raise CutoverError(f"cannot complete from {current.phase.value}")
        if self._backfill is None or self._inflight is None:
            raise CutoverError("complete needs a backfill and a legacy in-flight probe")
        if self._inflight() != 0:
            raise CutoverError("legacy jobs are still queued or running")
        report = self._backfill.reconcile()
        if not getattr(report, "ok", False):
            raise CutoverError(f"final reconciliation is not clean: {getattr(report, 'summary', lambda: report)()}")
        return self._move(current, Phase.COMPLETE, fence_at=current.fence_at)

    def rollback(self) -> CutoverState:
        """Return to the legacy path.  Safe only before ``complete``.

        Intents the ledger already accepted stay in its outbox and are still applied by the ledger
        worker; only the routing of new writes changes.  A rolled-back state must be reconciled
        again (``advance`` always does) before the cutover can proceed, because legacy writes do not
        advance ledger revisions.
        """

        current = self.state(fresh=True)
        if current.phase is Phase.COMPLETE:
            raise CutoverError("rollback is refused after complete: the legacy path cannot interpret the ledger's fences")
        if current.phase is Phase.LEGACY:
            return current
        return self._move(current, Phase.LEGACY, rolled_back=True, note="rolled back before complete")


def legacy_admission(controller: CutoverController | None, enqueued_at: float | None) -> tuple[bool, str]:
    """Whether a legacy write job may execute, and the explicit reason when it may not.

    ``enqueued_at`` is the job's enqueue time in epoch seconds; ``None`` (unknown) is treated as
    after the fence, because an unknown admission time cannot be proven safe.
    """

    if controller is None:
        return True, ""
    state = controller.state()
    if state.phase in (Phase.LEGACY, Phase.DUAL_ACCEPT):
        return True, ""
    if state.fence_at is not None and enqueued_at is not None and enqueued_at <= float(_epoch(state.fence_at)):
        return True, ""
    return False, "legacy_write_after_cutover"


def _epoch(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(str(value)).timestamp()


FENCED_RESULT = {"status": "fenced", "reason": "legacy_write_after_cutover", "executed": False}
