"""Generation-fenced read/write routing with a dual-write window and rollback.

Replaces the engine behind ``VectorStorePort`` without losing, duplicating or
resurrecting state.  The router never treats an engine as the source of truth:
the canonical store (the ledger, behind F's coordinator) is.  An engine is a
projection that the router can rebuild, so every step below is idempotent and
every uncertainty is resolved from canonical state.

Phases (one monotonic routing generation, see ``RoutingGenerationFence``):

``STEADY``      one backend serves reads and writes.
``DUAL_WRITE``  the active backend still serves reads and is authoritative for
                write results; every accepted write is also applied to a
                *shadow* backend.  A shadow failure never fails the caller; the
                key is marked dirty and must be repaired before a flip.
catch-up        ``catch_up`` brings the shadow to canonical state (and replays
                the journal for writes accepted before the window opened).
``flip``        under the write lock: lag must be zero, the generation advances
                and the shadow becomes active.  The old backend is kept as the
                rollback target and stops receiving writes.
``rollback``    replays the journal entries accepted since the flip onto the
                old backend, then binds it to a NEW generation (generations
                never go backwards).  The newer backend becomes the rollback
                target.

Generation fence: every operation is issued with a ``RouteToken`` carrying the
generation it was routed under.  Writes are checked under the write lock, so a
write routed before a flip is rejected (``STALE_REVISION``) once the flip has
happened and can never reach the old backend afterwards.  Reads with a stale
token are rejected too, so a client re-routes instead of reading a backend that
no longer serves.

Write journal / replay contract:

* the journal is content-free: an entry carries identity, revision and kind,
  never record text, so it holds nothing an erasure would have to chase;
* it exists only to bound rollback: writes are journaled only while a rollback
  target exists (from a flip until ``release_rollback_target``), so steady
  state and the dual-write window keep no journal at all, and a release or a
  rollback trims what is no longer needed;
* a write is applied to the active backend first and journaled only after it
  succeeded, under the same write lock, so the journal is totally ordered and
  contains only accepted writes; a crash between apply and append loses an
  entry, and canonical state repairs it;
* every repair copies *canonical* state: catch-up copies the canonical source
  and re-copies the keys whose dual write failed; rollback re-copies each key
  written since the flip (a removal needs only its journaled revision).  A
  repair that needs content and has no canonical source is refused, never
  guessed;
* replay is idempotent through the backend's revision fence: a revision older
  than the backend state is "already superseded", an identical replay is a
  no-op, a same-revision payload mismatch is reported as a conflict.

Shadow writes: the shadow write runs outside the write lock with a short
deadline and a bounded number of outstanding calls.  A shadow that is slow,
hung or saturated never delays the live backend or other writers: the key is
marked dirty (and counted as lag while the call is in flight) so a flip cannot
overtake it, and the key is repaired from canonical state.

Not provided here: per-tenant routing, graph or memory-engine routing,
multi-process coordination (one router owns the write lock), and durability of
the reference journal and cursor store.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, NamedTuple, Optional, Protocol

from contracts_errors import InternalErrorCode
from engine_ports import (
    EngineRecord,
    OperationContext,
    PortFailure,
    ProjectionReceipt,
    RoutingGenerationFence,
    SearchHit,
    TenantFilter,
    VectorStorePort,
)


class RoutePhase(str, enum.Enum):
    STEADY = "steady"
    DUAL_WRITE = "dual_write"


class RouteToken(NamedTuple):
    """The generation an operation was routed under."""

    generation: int


# --------------------------------------------------------------------------
# Hooks for the migration coordinator
# --------------------------------------------------------------------------


class CanonicalState(NamedTuple):
    """One record as the canonical store holds it (removed records keep their fence)."""

    tenant_id: str
    record_id: str
    revision: int
    removed: bool
    text: str = ""
    labels: tuple[str, ...] = ()


class CanonicalSource(Protocol):
    """Supplied by the coordinator; the authority for catch-up and repair.

    ``states()`` enumerates everything.  A source may also offer
    ``state_for(tenant_id, record_id) -> Optional[CanonicalState]`` so a repair
    of a few keys does not need a full pass.
    """

    def states(self) -> Iterable[CanonicalState]: ...


class CursorStore(Protocol):
    """Where the coordinator persists catch-up progress."""

    def get(self, name: str) -> int: ...
    def set(self, name: str, position: int) -> None: ...


class InMemoryCursorStore:
    """Reference cursor store (NOT durable)."""

    def __init__(self) -> None:
        self._values: dict[str, int] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> int:
        with self._lock:
            return self._values.get(name, 0)

    def set(self, name: str, position: int) -> None:
        with self._lock:
            self._values[name] = max(position, self._values.get(name, 0))


class RoutingHooks(Protocol):
    """Optional observer; exceptions raised here never affect routing."""

    def on_phase_change(self, phase: str, generation: int, detail: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class JournalEntry:
    seq: int
    generation: int
    kind: str  # "apply" | "remove" | "redacted"
    tenant_id: str
    record_id: str
    revision: int


class WriteJournal(Protocol):
    def append(self, entry: JournalEntry) -> JournalEntry: ...
    def head(self) -> int: ...
    def entries_after(self, seq: int, upto: Optional[int] = None) -> list[JournalEntry]: ...
    def trim_through(self, seq: int) -> None: ...


class InMemoryJournal:
    """Reference journal (NOT durable).

    Sequence numbers are assigned on append and never reused.  Entries are
    coalesced to the newest one per (tenant, record): rollback re-copies each
    record from canonical state, so only the latest identity and revision of a
    record matter.  The journal is therefore bounded by the number of distinct
    records written since the flip, not by the number of writes.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], JournalEntry] = {}
        self._seq = 0
        self._base = 0  # highest sequence number trimmed away
        self._lock = threading.Lock()

    def append(self, entry: JournalEntry) -> JournalEntry:
        with self._lock:
            self._seq += 1
            stored = JournalEntry(
                seq=self._seq,
                generation=entry.generation,
                kind=entry.kind,
                tenant_id=entry.tenant_id,
                record_id=entry.record_id,
                revision=entry.revision,
            )
            key = (stored.tenant_id, stored.record_id)
            self._entries.pop(key, None)  # re-insert so iteration order stays by sequence
            self._entries[key] = stored
            return stored

    def head(self) -> int:
        with self._lock:
            return self._seq

    def entries_after(self, seq: int, upto: Optional[int] = None) -> list[JournalEntry]:
        with self._lock:
            end = self._seq if upto is None else upto
            return [e for e in self._entries.values() if max(seq, self._base) < e.seq <= end]

    def trim_through(self, seq: int) -> None:
        """Drop every entry up to and including ``seq``; later sequence numbers are unchanged."""

        with self._lock:
            self._base = max(self._base, seq)
            for key in [k for k, e in self._entries.items() if e.seq <= seq]:
                del self._entries[key]


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------


@dataclass
class Backend:
    """One engine generation.

    ``synced`` means the engine has been brought to canonical state by a
    canonical catch-up or backfill (and is kept there by dual write); recall
    from the engine is only safe when it is true.  ``destroy`` drops the
    generation's stored data when its rollback target is released; a backend
    without one cannot be destroyed by the router (the adapter's own
    ``destroy`` is used when it has one).
    """

    label: str
    adapter: VectorStorePort
    synced: bool = False
    destroy: Optional[Callable[[], None]] = None


@dataclass
class CatchUpReport:
    copied: int = 0
    tombstoned: int = 0
    superseded: int = 0
    replayed: int = 0
    conflicts: list[tuple[str, str, int]] = field(default_factory=list)
    repaired_dirty: int = 0
    unrepaired: int = 0  # dirty keys that need content and had no canonical state to copy


class RoutingError(Exception):
    """A switch step was refused (lag, wrong phase); the serving generation is unchanged."""


def _ctx(tenant_id: str) -> OperationContext:
    return OperationContext(tenant_id=tenant_id, correlation_id="routing", deadline_ms=60_000)


_TOLERATED = (InternalErrorCode.STALE_REVISION,)


class EngineRouter:
    """Routes ``VectorStorePort`` operations by generation (see module docstring)."""

    def __init__(
        self,
        active: Backend,
        *,
        journal: Optional[WriteJournal] = None,
        cursors: Optional[CursorStore] = None,
        hooks: Optional[RoutingHooks] = None,
        fence: Optional[RoutingGenerationFence] = None,
        shadow_timeout_s: float = 2.0,
        shadow_max_outstanding: int = 8,
        rollback_window_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fence = fence or RoutingGenerationFence(1)
        if self._fence.current < 1:
            self._fence.advance()
        self._active = active
        self._previous: Optional[Backend] = None
        self._shadow: Optional[Backend] = None
        self._phase = RoutePhase.STEADY
        self._journal = journal or InMemoryJournal()
        self._cursors = cursors or InMemoryCursorStore()
        self._hooks = hooks
        self._write_lock = threading.RLock()
        self._state_lock = threading.Lock()  # guards the dirty/in-flight bookkeeping only
        self._cursor_name = "catchup@0"
        self._flip_seq = 0
        # key -> (kind, revision) of the latest write whose shadow copy is unconfirmed
        self._dirty: dict[tuple[str, str], tuple[str, int]] = {}
        self._inflight: dict[tuple[str, str], int] = {}
        self._conflicts: set[tuple[str, str]] = set()
        self._window = 0  # identifies the dual-write window a late shadow result belongs to
        self._unsynced_steady_writes = False  # writes accepted outside any window and any journal
        self._needs_canonical_sync = False
        self._previous_dirty: dict[tuple[str, str], tuple[str, int]] = {}
        self._previous_inflight: dict[tuple[str, str], int] = {}
        self._previous_window = 0  # identifies the previous generation a late purge result belongs to
        self._rollback_window_s = rollback_window_s
        self._clock = clock
        self._flipped_at: Optional[float] = None
        self.release_log: list[dict[str, Any]] = []
        self._shadow_timeout_s = shadow_timeout_s
        self._shadow_slots = threading.BoundedSemaphore(max(1, shadow_max_outstanding))

    # -- introspection -------------------------------------------------------

    @property
    def generation(self) -> int:
        return self._fence.current

    @property
    def phase(self) -> RoutePhase:
        return self._phase

    @property
    def active_label(self) -> str:
        return self._active.label

    @property
    def previous_label(self) -> Optional[str]:
        return self._previous.label if self._previous else None

    @property
    def active_synced(self) -> bool:
        """Whether the serving engine has been brought to canonical state."""

        return self._active.synced

    @property
    def previous_purge_pending(self) -> int:
        """Erasures not yet confirmed on the previous generation (its text may still be there)."""

        with self._state_lock:
            return len(self._previous_dirty) + len(self._previous_inflight)

    def route(self) -> RouteToken:
        return RouteToken(self._fence.current)

    def lag(self) -> int:
        """Pending shadow work: unsynced canonical state, dirty and in-flight keys, conflicts."""

        if self._shadow is None:
            return 0
        with self._state_lock:
            pending = 1 if self._needs_canonical_sync else 0
            return pending + len(self._dirty) + len(self._inflight) + len(self._conflicts)

    def _notify(self, phase: str, detail: Optional[dict[str, Any]] = None) -> None:
        if self._hooks is not None:
            try:
                self._hooks.on_phase_change(phase, self._fence.current, detail or {})
            except Exception:  # noqa: BLE001 - observers cannot affect routing
                pass

    # -- routed operations -----------------------------------------------------

    def _record_write(
        self, token: RouteToken, kind: str, tenant: str, record_id: str, revision: int
    ) -> list[tuple[str, Backend, int]]:
        """Under the write lock: journal for rollback and reserve the secondary writes.

        A removal is also sent to the previous generation (the rollback target)
        so erased text does not outlive the erasure there; applies are not, the
        rollback repairs them from canonical state.
        """

        self._maybe_release_after_window()
        reservations: list[tuple[str, Backend, int]] = []
        key = (tenant, record_id)
        if self._previous is not None:
            self._journal.append(JournalEntry(0, token.generation, kind, tenant, record_id, revision))
            if kind == "remove":
                with self._state_lock:
                    self._previous_inflight[key] = self._previous_inflight.get(key, 0) + 1
                reservations.append(("previous", self._previous, self._previous_window))
        elif self._phase is RoutePhase.STEADY:
            self._unsynced_steady_writes = True
        if self._phase is RoutePhase.DUAL_WRITE and self._shadow is not None:
            with self._state_lock:
                self._inflight[key] = self._inflight.get(key, 0) + 1
            reservations.append(("shadow", self._shadow, self._window))
        return reservations

    def apply(self, token: RouteToken, ctx: OperationContext, record: EngineRecord) -> ProjectionReceipt:
        with self._write_lock:
            self._fence.check(token.generation)
            receipt = self._active.adapter.apply(ctx, record)
            reservations = self._record_write(token, "apply", record.tenant_id, record.record_id, record.revision)
        for reserved in reservations:
            self._shadow_write(
                reserved, lambda adapter: adapter.apply(ctx, record), "apply", record.tenant_id, record.record_id, record.revision
            )
        return receipt

    def remove(self, token: RouteToken, ctx: OperationContext, record_id: str, revision: int) -> ProjectionReceipt:
        with self._write_lock:
            self._fence.check(token.generation)
            receipt = self._active.adapter.remove(ctx, record_id, revision)
            reservations = self._record_write(token, "remove", ctx.tenant_id, record_id, revision)
        for reserved in reservations:
            self._shadow_write(
                reserved, lambda adapter: adapter.remove(ctx, record_id, revision), "remove", ctx.tenant_id, record_id, revision
            )
        return receipt

    def get(self, token: RouteToken, ctx: OperationContext, record_id: str) -> Optional[EngineRecord]:
        self._fence.check(token.generation)
        return self._active.adapter.get(ctx, record_id)

    def search(
        self, token: RouteToken, ctx: OperationContext, query: str, flt: TenantFilter, limit: int
    ) -> tuple[SearchHit, ...]:
        self._fence.check(token.generation)
        return self._active.adapter.search(ctx, query, flt, limit)

    def port(self) -> "RoutedVectorStore":
        """A ``VectorStorePort`` that routes every call under the current generation."""

        return RoutedVectorStore(self)

    def _mark_dirty(self, target: str, window: int, key: tuple[str, str], kind: str, revision: int) -> None:
        with self._state_lock:
            if target == "previous":
                if window == self._previous_window:  # a late result for a released generation is irrelevant
                    self._previous_dirty[key] = (kind, revision)
            elif window == self._window:  # a late result from a closed window is irrelevant
                self._dirty[key] = (kind, revision)

    def _shadow_write(
        self,
        reserved: tuple[str, Backend, int],
        call: Callable[[VectorStorePort], object],
        kind: str,
        tenant: str,
        record_id: str,
        revision: int,
    ) -> None:
        """Apply to a secondary generation outside the write lock; never fail or stall the caller for long.

        The key stays counted as in flight until this returns.  A failure, a
        timeout or a saturated target marks it dirty for canonical repair.
        """

        target, backend, window = reserved
        key = (tenant, record_id)
        inflight = self._previous_inflight if target == "previous" else self._inflight
        try:
            if not self._shadow_slots.acquire(blocking=False):
                self._mark_dirty(target, window, key, kind, revision)
                return
            finished = threading.Event()

            def run() -> None:
                try:
                    call(backend.adapter)
                except PortFailure as failure:
                    if failure.code not in _TOLERATED:  # a stale write: the target holds something newer
                        self._mark_dirty(target, window, key, kind, revision)
                except Exception:  # noqa: BLE001 - a secondary failure must not fail the caller
                    self._mark_dirty(target, window, key, kind, revision)
                finally:
                    self._shadow_slots.release()
                    finished.set()

            threading.Thread(target=run, name="ns-shadow-write", daemon=True).start()
            if not finished.wait(self._shadow_timeout_s):
                self._mark_dirty(target, window, key, kind, revision)  # unconfirmed: repair from canonical state
        finally:
            with self._state_lock:
                remaining = inflight.get(key, 0) - 1
                if remaining > 0:
                    inflight[key] = remaining
                else:
                    inflight.pop(key, None)

    # -- switch protocol ----------------------------------------------------------

    def begin_dual_write(self, shadow: Backend) -> None:
        """Open the window: from now on accepted writes also go to ``shadow``."""

        with self._write_lock:
            if self._phase is not RoutePhase.STEADY:
                raise RoutingError("a switch is already in progress")
            if shadow.adapter is self._active.adapter:
                raise RoutingError("the shadow must be a different backend")
            with self._state_lock:
                self._window += 1
                self._dirty = {}
                self._inflight = {}
                self._conflicts = set()
                # Writes this router accepted without any journal can only be copied from canonical state.
                self._needs_canonical_sync = self._unsynced_steady_writes
            self._shadow = shadow
            self._phase = RoutePhase.DUAL_WRITE
            self._cursor_name = f"catchup@{self._fence.current}"  # one fresh cursor per switch attempt
            self._notify("dual_write_started", {"journal_head": self._journal.head()})

    def abandon_dual_write(self) -> None:
        with self._write_lock:
            if self._phase is not RoutePhase.DUAL_WRITE:
                raise RoutingError("no dual-write window is open")
            with self._state_lock:
                self._window += 1
                self._dirty, self._conflicts, self._inflight = {}, set(), {}
                self._needs_canonical_sync = False
            self._shadow, self._phase = None, RoutePhase.STEADY
            self._notify("dual_write_abandoned")

    @staticmethod
    def _tombstone(adapter: VectorStorePort, tenant_id: str, record_id: str, revision: int) -> None:
        """Remove a record and leave the revision fence even if it was never applied."""

        ctx = _ctx(tenant_id)
        try:
            adapter.remove(ctx, record_id, revision)
        except PortFailure as failure:
            if failure.code is not InternalErrorCode.MISSING_RESOURCE:
                raise
            # Never applied here (or already removed): place a content-free record so the fence exists.
            try:
                adapter.apply(ctx, EngineRecord(record_id=record_id, revision=revision, tenant_id=tenant_id, text=""))
            except PortFailure as inner:
                if inner.code not in (InternalErrorCode.STALE_REVISION, InternalErrorCode.IDEMPOTENCY_CONFLICT):
                    raise
                return  # already fenced at this revision or newer
            adapter.remove(ctx, record_id, revision)

    @staticmethod
    def _lookup(source: Optional[CanonicalSource]) -> Callable[[str, str], Optional[CanonicalState]]:
        """Canonical state for one key: the source's own lookup, else one pass over ``states()``."""

        if source is None:
            return lambda tenant, record_id: None
        direct = getattr(source, "state_for", None)
        if callable(direct):
            return direct
        index: dict[tuple[str, str], CanonicalState] = {}
        loaded = False

        def find(tenant: str, record_id: str) -> Optional[CanonicalState]:
            nonlocal loaded
            if not loaded:
                index.update({(st.tenant_id, st.record_id): st for st in source.states()})
                loaded = True
            return index.get((tenant, record_id))

        return find

    def _copy_state(self, adapter: VectorStorePort, state: CanonicalState, report: CatchUpReport) -> None:
        ctx = _ctx(state.tenant_id)
        try:
            if state.removed:
                self._tombstone(adapter, state.tenant_id, state.record_id, state.revision)
                report.tombstoned += 1
            else:
                adapter.apply(
                    ctx,
                    EngineRecord(
                        record_id=state.record_id, revision=state.revision, tenant_id=state.tenant_id,
                        text=state.text, labels=state.labels,
                    ),
                )
                report.copied += 1
        except PortFailure as failure:
            if failure.code is InternalErrorCode.STALE_REVISION:
                report.superseded += 1  # something newer already reached this backend
            elif failure.code is InternalErrorCode.IDEMPOTENCY_CONFLICT:
                report.conflicts.append((state.tenant_id, state.record_id, state.revision))
            else:
                raise

    def _repair_key(
        self,
        adapter: VectorStorePort,
        tenant: str,
        record_id: str,
        kind: str,
        revision: int,
        lookup: Callable[[str, str], Optional[CanonicalState]],
        report: CatchUpReport,
    ) -> bool:
        """Bring one key to canonical state; ``False`` when content is needed but unavailable."""

        state = lookup(tenant, record_id)
        if state is not None:
            self._copy_state(adapter, state, report)
            return True
        if kind == "remove":  # a removal needs only identity and revision
            try:
                self._tombstone(adapter, tenant, record_id, revision)
                report.tombstoned += 1
            except PortFailure as failure:
                if failure.code is InternalErrorCode.STALE_REVISION:
                    report.superseded += 1
                elif failure.code is InternalErrorCode.IDEMPOTENCY_CONFLICT:
                    report.conflicts.append((tenant, record_id, revision))
                else:
                    raise
            return True
        return False

    def catch_up(self, source: Optional[CanonicalSource] = None) -> CatchUpReport:
        """Bring the shadow to canonical state; idempotent and safe under live writes."""

        if self._phase is not RoutePhase.DUAL_WRITE or self._shadow is None:
            raise RoutingError("catch-up needs an open dual-write window")
        shadow = self._shadow.adapter
        report = CatchUpReport()
        self._conflicts = set()
        if source is not None:
            for state in source.states():
                self._copy_state(shadow, state, report)
            with self._state_lock:
                self._needs_canonical_sync = False
        self._cursors.set(self._cursor_name, self._journal.head())
        self._conflicts = {(tenant, record_id) for tenant, record_id, _rev in report.conflicts}
        # Repair keys whose dual write is unconfirmed, from canonical state (never from stored text).
        lookup = self._lookup(source)
        with self._state_lock:
            dirty = sorted(self._dirty.items())
        for (tenant, record_id), (kind, revision) in dirty:
            if self._repair_key(shadow, tenant, record_id, kind, revision, lookup, report):
                with self._state_lock:
                    if self._dirty.get((tenant, record_id)) == (kind, revision):
                        del self._dirty[(tenant, record_id)]
                report.repaired_dirty += 1
            else:
                report.unrepaired += 1
        if source is not None and not report.conflicts and not report.unrepaired:
            self._shadow.synced = True  # brought to canonical state; dual write keeps it there
        self._notify("caught_up", {"lag": self.lag()})
        return report

    def backfill(self, source: CanonicalSource) -> CatchUpReport:
        """Bring the *serving* engine to canonical state (no window open).

        Memories written before the engine was routed, or by pipelines that do
        not project, are copied from the canonical source; writes racing the
        copy are safe through the revision fence.  Only a clean backfill marks
        the engine synced, which is what routed recall requires.
        """

        if self._phase is not RoutePhase.STEADY:
            raise RoutingError("backfill needs no open dual-write window")
        report = CatchUpReport()
        for state in source.states():
            self._copy_state(self._active.adapter, state, report)
        if not report.conflicts:
            self._active.synced = True
        self._notify("backfilled", {"copied": report.copied, "tombstoned": report.tombstoned, "conflicts": len(report.conflicts)})
        return report

    def flip(self, *, parity_check: Optional[Callable[[VectorStorePort, VectorStorePort], bool]] = None) -> int:
        """Make the shadow active under a new generation; refuse while lagging."""

        with self._write_lock:  # no write is in flight and none can start
            if self._phase is not RoutePhase.DUAL_WRITE or self._shadow is None:
                raise RoutingError("flip needs an open dual-write window")
            if self.lag() != 0:
                raise RoutingError(f"shadow is lagging (lag={self.lag()})")
            if parity_check is not None and not parity_check(self._active.adapter, self._shadow.adapter):
                raise RoutingError("parity check failed")
            if self._previous is not None:
                # A flip would otherwise orphan the older generation with its text; destroy it explicitly first.
                self.release_rollback_target(reason="superseded")
            self._previous, self._active = self._active, self._shadow
            self._shadow, self._phase = None, RoutePhase.STEADY
            with self._state_lock:
                self._window += 1
                self._dirty, self._inflight = {}, {}
                self._previous_window += 1
                self._previous_dirty, self._previous_inflight = {}, {}
            self._flipped_at = self._clock()
            self._flip_seq = self._journal.head()
            generation = self._fence.advance()
            self._notify("flipped", {"flip_seq": self._flip_seq})
            return generation

    def rollback(self, source: Optional[CanonicalSource] = None) -> int:
        """Re-copy post-flip writes onto the previous backend, then serve it under a new generation.

        The journal holds identity and revision only, so a write that needs
        content is repaired from ``source``; without one the rollback is refused.
        """

        with self._write_lock:
            if self._previous is None or self._phase is not RoutePhase.STEADY:
                raise RoutingError("nothing to roll back to")
            target = self._previous
            report = CatchUpReport()
            latest: dict[tuple[str, str], JournalEntry] = {}
            for entry in self._journal.entries_after(self._flip_seq):
                latest[(entry.tenant_id, entry.record_id)] = entry
            lookup = self._lookup(source)
            missing = [
                key for key, entry in latest.items()
                if entry.kind == "apply" and lookup(key[0], key[1]) is None
            ]
            if missing:
                raise RoutingError(
                    "rollback needs the canonical source to repair "
                    f"{len(missing)} write(s) accepted since the flip"
                )
            for (tenant, record_id), entry in sorted(latest.items()):
                self._repair_key(target.adapter, tenant, record_id, entry.kind, entry.revision, lookup, report)
            if report.conflicts:
                raise RoutingError(f"rollback replay found conflicts: {report.conflicts[:3]}")
            self._previous, self._active = self._active, target
            with self._state_lock:
                self._previous_window += 1
                self._previous_dirty, self._previous_inflight = {}, {}
            self._flipped_at = self._clock()
            self._journal.trim_through(self._journal.head())  # what was replayed is no longer needed
            self._flip_seq = self._journal.head()
            generation = self._fence.advance()
            self._notify("rolled_back", {"replayed": report.copied + report.tombstoned, "superseded": report.superseded})
            return generation

    def release_rollback_target(self, reason: str = "explicit") -> Optional[dict[str, Any]]:
        """Give up the ability to roll back and destroy the previous generation.

        The previous generation holds record text.  It is destroyed (its
        ``destroy`` hook, or the adapter's own) so retained content does not
        outlive the rollback window, and the release is recorded in
        ``release_log``.  It is refused while an erasure is not yet confirmed on
        that generation, and when the generation cannot be destroyed, in which
        case it stays the rollback target.
        """

        with self._write_lock:
            previous = self._previous
            if previous is None:
                return None
            if self.previous_purge_pending:
                raise RoutingError("the previous generation still holds unpurged erasures; repair it first")
            destroy = previous.destroy or getattr(previous.adapter, "destroy", None)
            destroyed = False
            if callable(destroy):
                try:
                    destroy()
                except Exception as error:  # noqa: BLE001 - the generation stays the rollback target
                    raise RoutingError("the previous generation could not be destroyed") from error
                destroyed = True
            audit = {
                "label": previous.label,
                "destroyed": destroyed,
                "reason": reason,
                "generation": self._fence.current,
                "journal_entries_dropped": len(self._journal.entries_after(0)),
            }
            self._previous = None
            with self._state_lock:
                self._previous_window += 1
                self._previous_dirty, self._previous_inflight = {}, {}
            self._journal.trim_through(self._journal.head())
            self._flip_seq = self._journal.head()
            self._flipped_at = None
            self.release_log.append(audit)
            self._notify("rollback_released", audit)
            return audit

    def _maybe_release_after_window(self) -> None:
        """Close the rollback window lazily: the first write after it has elapsed releases and destroys."""

        if self._previous is None or self._rollback_window_s is None or self._flipped_at is None:
            return
        if self._clock() - self._flipped_at < self._rollback_window_s:
            return
        try:
            self.release_rollback_target(reason="window_elapsed")
        except RoutingError:
            pass  # still pending or not destroyable: it stays and is visible in the status


class RoutedVectorStore:
    """``VectorStorePort`` over an ``EngineRouter``; each call is routed under the current generation."""

    def __init__(self, router: EngineRouter) -> None:
        self._router = router

    def manifest(self):
        return self._router._active.adapter.manifest()

    def apply(self, ctx: OperationContext, record: EngineRecord) -> ProjectionReceipt:
        return self._router.apply(self._router.route(), ctx, record)

    def remove(self, ctx: OperationContext, record_id: str, revision: int) -> ProjectionReceipt:
        return self._router.remove(self._router.route(), ctx, record_id, revision)

    def get(self, ctx: OperationContext, record_id: str) -> Optional[EngineRecord]:
        return self._router.get(self._router.route(), ctx, record_id)

    def search(
        self, ctx: OperationContext, query: str, flt: TenantFilter, limit: int
    ) -> tuple[SearchHit, ...]:
        return self._router.search(self._router.route(), ctx, query, flt, limit)


__all__ = [
    "Backend",
    "CanonicalSource",
    "CanonicalState",
    "CatchUpReport",
    "CursorStore",
    "EngineRouter",
    "InMemoryCursorStore",
    "InMemoryJournal",
    "JournalEntry",
    "RoutePhase",
    "RouteToken",
    "RoutedVectorStore",
    "RoutingError",
    "RoutingHooks",
    "WriteJournal",
]
