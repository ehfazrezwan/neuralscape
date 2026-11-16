"""Service wiring for the engine switch: projection, canonical source, recall, admin operations.

The memory service's own vector store (mem0 over Qdrant) stays the system of
record for memory rows in V1.  This module adds a *routed engine projection*
beside it: the ledger's vector stage projects every accepted write, edit and
delete into an ``EngineRouter`` whose generation 1 is a Qdrant projection
collection and whose shadow can be pgvector.  An operator moves traffic with
dual-write -> catch-up -> flip (and rollback); recall can then be served from
the routed engine.  Nothing here is imported or active unless
``ENGINE_ROUTING_ENABLED`` is set.

Canonical source for catch-up = the ledger's object references and tombstones
(identity, revision, erasure fences) joined with the *current memory rows* from
the memory service; never the engine being replaced.

Honest limits (also in the report):

* the router keeps its phase and generation in one process; run it where the
  API and the ledger worker share a process until a durable control record
  exists;
* only memories that went through the ledger (raw writes, edits, deletes) are
  projected; memories written before the ledger, or created by the ingest and
  connector pipelines, enter the projection only if they are registered in the
  ledger;
* a record whose edit was accepted but not yet applied is projected with its
  current text at the registered revision; a catch-up in that window reports a
  same-revision conflict and the flip is refused until it is resolved.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Iterable, Optional

import third_party_telemetry  # noqa: F401 — must precede any mem0/Graphiti import (defaults their telemetry off)

from contracts_errors import InternalErrorCode
from engine_ports import EngineRecord, OperationContext, PortFailure, SearchHit, TenantFilter
from engine_routing import (
    Backend,
    CanonicalSource,
    CanonicalState,
    CatchUpReport,
    EngineRouter,
    RoutePhase,
    RoutingError,
)

logger = logging.getLogger("neuralscape.engine_switch")

_GLOBAL_OVERFETCH = 4  # depth of the project-free leg of a routed project recall, in multiples of the limit
CANONICAL_PAGE_SIZE = 500  # ledger rows read per page by a catch-up or backfill


def labels_for(user_id: str, visibility: Optional[str], project_id: Optional[str]) -> tuple[str, ...]:
    """Filter labels that let recall reproduce the personal and shared pools."""

    labels = [f"owner:{user_id}", f"vis:{visibility or 'private'}"]
    if project_id:
        labels.append(f"project:{project_id}")
    return tuple(label[:256] for label in labels)


def _bounded(label: str) -> str:
    """The same truncation the projection applies, so recall filters match what was projected."""

    return label[:256]


def _ctx(tenant_id: str) -> OperationContext:
    return OperationContext(tenant_id=tenant_id, correlation_id="engine-switch", deadline_ms=60_000)


class LedgerCanonicalSource:
    """Catch-up source: ledger object refs and tombstones joined with current rows."""

    def __init__(self, ledger: Any, service: Any, labeler: Callable[[Any], tuple[str, ...]]) -> None:
        self._ledger, self._service, self._labeler = ledger, service, labeler

    def state_for(self, tenant_id: str, record_id: str) -> Optional[CanonicalState]:
        """One key's canonical state, for repairs that do not need a full pass."""

        revision = 0
        while self._ledger.get_object_ref(tenant_id, record_id, revision + 1) is not None:
            revision += 1
        tombstone = self._ledger.get_tombstone(tenant_id, record_id)
        if tombstone is not None:
            return CanonicalState(tenant_id, record_id, int(tombstone.content_revision), True)
        if revision == 0:
            return None
        row = self._service.get_memory(record_id)
        if row is None:
            return None
        return CanonicalState(tenant_id, record_id, revision, False, row.memory, self._labeler(row))

    def states(self) -> Iterable[CanonicalState]:
        heads = getattr(self._ledger, "object_heads", None)
        tombstones = getattr(self._ledger, "tombstone_heads", None)
        if callable(heads) and callable(tombstones):
            yield from self._paged_states(heads, tombstones)
            return
        # A ledger without paged reads: one snapshot (memory grows with the whole ledger).
        yield from self._snapshot_states()

    def _paged_states(self, heads: Callable[..., list], tombstones: Callable[..., list]) -> Iterable[CanonicalState]:
        """Walk object references and tombstones a page at a time: memory is bounded by the page,
        not by the ledger (whose intents and receipts a catch-up never needs).

        The states come in the ledger's key order (the database's collation), which can differ from
        the snapshot path's code-point order; catch-up and backfill do not depend on the order.  The
        walk is lazy, so writes accepted meanwhile may be seen at an older revision: the engine's
        revision fence refuses such a copy once the newer write has reached it.
        """

        after = None
        while page := heads(after, CANONICAL_PAGE_SIZE):
            for tenant, record_id, revision in page:
                if self._ledger.get_tombstone(tenant, record_id) is not None:
                    continue
                row = self._service.get_memory(record_id)
                if row is None:
                    continue  # neither tombstoned nor readable: nothing to project
                yield CanonicalState(tenant, record_id, revision, False, row.memory, self._labeler(row))
            after = page[-1][:2]
        after = None
        while page := tombstones(after, CANONICAL_PAGE_SIZE):
            for tenant, record_id, revision in page:
                yield CanonicalState(tenant, record_id, revision, True)
            after = page[-1][:2]

    def _snapshot_states(self) -> Iterable[CanonicalState]:
        tables = self._ledger.snapshot()["tables"]
        latest: dict[tuple[str, str], int] = {}
        for row in tables["object_refs"]:
            key = (row["tenant_id"], row["record_id"])
            latest[key] = max(latest.get(key, 0), int(row["content_revision"]))
        tombstones = {
            (row["tenant_id"], row["record_id"]): int(row["content_revision"]) for row in tables["tombstones"]
        }
        for (tenant, record_id), revision in sorted(latest.items()):
            if (tenant, record_id) in tombstones:
                continue
            row = self._service.get_memory(record_id)
            if row is None:
                continue  # neither tombstoned nor readable: nothing to project
            yield CanonicalState(tenant, record_id, revision, False, row.memory, self._labeler(row))
        for (tenant, record_id), revision in sorted(tombstones.items()):
            yield CanonicalState(tenant, record_id, revision, True)


class EngineSwitch:
    """Projection, recall and the privileged switch operations over one ``EngineRouter``."""

    def __init__(
        self,
        router: EngineRouter,
        *,
        ledger: Any,
        service: Any,
        tenant_for: Callable[[str], Optional[str]],
        shadow_factory: Optional[Callable[[], Backend]] = None,
        recall_enabled: bool = False,
        closers: Iterable[Callable[[], None]] = (),
    ) -> None:
        self._router = router
        self._router_of: Optional[Callable[[], Optional[EngineRouter]]] = None
        self._ledger = ledger
        self._service = service
        self._tenant_for = tenant_for
        self._shadow_factory = shadow_factory
        self.recall_enabled = recall_enabled
        self._admin_lock = threading.Lock()
        self._closers = list(closers)

    def close(self) -> None:
        """Release the clients this switch opened (each at most once; a failure is logged, not raised)."""

        closers, self._closers = self._closers, []
        for close in closers:
            try:
                close()
            except Exception:  # noqa: BLE001 - shutdown must go on
                logger.warning("closing an engine switch client failed", exc_info=True)

    # -- the serving router ------------------------------------------------------

    @property
    def router(self) -> EngineRouter:
        """The router projections and recall go through (a migration controller's when followed)."""

        if self._router_of is None:
            return self._router
        router = self._router_of()
        if router is None:
            # Between a controller's detach and its next attach: fail the caller (a stage is
            # retried) rather than projecting into an engine nobody serves.
            raise RoutingError("the migration controller has no serving router")
        return router

    def follow(self, router_of: Callable[[], Optional[EngineRouter]]) -> None:
        """Route projection and recall through a migration controller's current router.

        The controller replaces its router object on a rollback, so the router is looked
        up on every call rather than captured.  While followed, the switch's own admin
        operations are refused: the controller owns the window, the flip and rollback.
        """

        self._router_of = router_of

    def _admin_allowed(self) -> None:
        if self._router_of is not None:
            raise RoutingError("the engine switch is driven by a migration controller")

    # -- helpers -----------------------------------------------------------------

    def _row_labels(self, row: Any) -> tuple[str, ...]:
        return labels_for(
            getattr(row, "owner_user_id", None) or "unknown",
            getattr(row, "visibility", None),
            getattr(row, "project_id", None),
        )

    def canonical_source(self) -> CanonicalSource:
        return LedgerCanonicalSource(self._ledger, self._service, self._row_labels)

    def _revision(self, tenant_id: str, memory_id: str) -> int:
        revision = 0
        while self._ledger.get_object_ref(tenant_id, memory_id, revision + 1) is not None:
            revision += 1
        return max(revision, 1)

    # -- projection (called from the ledger vector stage) --------------------------

    def project_apply(
        self,
        tenant_id: str,
        memory_id: str,
        *,
        text: str,
        user_id: str,
        visibility: Optional[str] = None,
        project_id: Optional[str] = None,
        revision: Optional[int] = None,
    ) -> None:
        applied_revision = revision if revision is not None else self._revision(tenant_id, memory_id)
        tombstone = self._ledger.get_tombstone(tenant_id, memory_id)
        if tombstone is not None and applied_revision <= tombstone.content_revision:
            return  # erased: an old or replayed projection must never bring the record back
        record = EngineRecord(
            record_id=memory_id,
            revision=applied_revision,
            tenant_id=tenant_id,
            text=text,
            labels=labels_for(user_id, visibility, project_id),
        )
        router = self.router  # one router for the token and the call
        try:
            router.apply(router.route(), _ctx(tenant_id), record)
        except PortFailure as failure:
            if failure.code is InternalErrorCode.STALE_REVISION:
                return  # the engine already holds this or a newer revision (a retried stage)
            raise

    def project_row(self, tenant_id: str, memory_id: str) -> None:
        """Project the memory's current row at its registered revision (after an edit)."""

        row = self._service.get_memory(memory_id)
        if row is None:
            return
        self.project_apply(
            tenant_id, memory_id, text=row.memory, user_id=getattr(row, "owner_user_id", None) or "unknown",
            visibility=getattr(row, "visibility", None), project_id=getattr(row, "project_id", None),
        )

    def project_remove(self, tenant_id: str, memory_id: str) -> None:
        tombstone = self._ledger.get_tombstone(tenant_id, memory_id)
        revision = tombstone.content_revision if tombstone is not None else self._revision(tenant_id, memory_id)
        router = self.router
        try:
            router.remove(router.route(), _ctx(tenant_id), memory_id, revision)
        except PortFailure as failure:
            if failure.code in (InternalErrorCode.MISSING_RESOURCE, InternalErrorCode.STALE_REVISION):
                return  # never projected, already removed, or fenced at a newer revision
            raise

    # -- recall ---------------------------------------------------------------------

    def recall(
        self,
        user_id: str,
        query: str,
        limit: int,
        *,
        project_id: Optional[str] = None,
        include_shared: bool = True,
        visibility: Optional[str] = None,
        include_global: bool = False,
    ) -> list[SearchHit]:
        """Vector recall from the serving engine: the caller's own rows plus the shared pool.

        With a ``project_id`` the project's rows carry its label, but the caller's global rows (no
        project) do not, so ``include_global`` adds a second, project-free search per pool and
        returns the union (``_GLOBAL_OVERFETCH`` times ``limit`` deep, since that search also meets
        other projects' rows).  The caller keeps only rows of the project or of global scope.
        """

        tenant_id = self._tenant_for(user_id)
        if tenant_id is None:
            return []
        router = self.router
        token = router.route()
        seen: dict[str, SearchHit] = {}
        pools = []
        if visibility != "shared":
            pools.append((_bounded(f"owner:{user_id}"),))
        if include_shared and visibility != "private":
            pools.append(("vis:shared",))
        wide = bool(project_id and include_global)
        for labels in pools:
            legs = [(labels + ((_bounded(f"project:{project_id}"),) if project_id else ()), limit)]
            if wide:
                legs.append((labels, limit * _GLOBAL_OVERFETCH))
            for leg_labels, leg_limit in legs:
                flt = TenantFilter(tenant_id=tenant_id, labels_all=leg_labels)
                for hit in router.search(token, _ctx(tenant_id), query, flt, leg_limit):
                    if hit.record_id not in seen or hit.score > seen[hit.record_id].score:
                        seen[hit.record_id] = hit
        ordered = sorted(seen.values(), key=lambda h: (-h.score, h.record_id))
        return ordered[: limit * _GLOBAL_OVERFETCH if wide else limit]

    def recall_ready(self) -> tuple[bool, Optional[str]]:
        """Whether routed recall may serve: enabled, and the engine holds canonical state.

        An engine that has not been brought to canonical state (memories written
        before routing, or by pipelines that do not project) would silently hide
        those memories, so routed recall is refused with a reason instead.
        """

        if not self.recall_enabled:
            return False, "recall_disabled"
        try:
            synced = self.router.active_synced
        except RoutingError:
            return False, "no_serving_router"
        if not synced:
            return False, "projection_not_synced"
        return True, None

    # -- privileged switch operations ------------------------------------------------

    def status(self) -> dict[str, Any]:
        ready, reason = self.recall_ready()
        router = self.router
        return {
            "generation": router.generation,
            "phase": router.phase.value,
            "active": router.active_label,
            "previous": router.previous_label,
            "lag": router.lag(),
            "recall_enabled": self.recall_enabled,
            "recall_ready": ready,
            "recall_blocked_reason": reason,
            "active_synced": router.active_synced,
            "previous_purge_pending": router.previous_purge_pending,
            "rollback_released": list(router.release_log),
        }

    def start_dual_write(self) -> dict[str, Any]:
        self._admin_allowed()
        if self._shadow_factory is None:
            raise RoutingError("no shadow backend is configured")
        with self._admin_lock:
            self.router.begin_dual_write(self._shadow_factory())
        return self.status()

    def catch_up(self) -> dict[str, Any]:
        """Bring the shadow to canonical state; with no window open, backfill the serving engine.

        The backfill is what makes routed recall safe to enable on an engine that
        was not populated from every memory (no new admin route is needed).
        """

        self._admin_allowed()
        with self._admin_lock:
            backfilled = self.router.phase is RoutePhase.STEADY
            report: CatchUpReport = (
                self.router.backfill(self.canonical_source())
                if backfilled
                else self.router.catch_up(self.canonical_source())
            )
        return {**self.status(), "backfilled": backfilled, "report": {
            "copied": report.copied, "tombstoned": report.tombstoned, "superseded": report.superseded,
            "replayed": report.replayed, "repaired_dirty": report.repaired_dirty, "conflicts": len(report.conflicts),
        }}

    def flip(self) -> dict[str, Any]:
        self._admin_allowed()
        with self._admin_lock:
            self.router.flip()
        return self.status()

    def rollback(self) -> dict[str, Any]:
        self._admin_allowed()
        with self._admin_lock:
            self.router.rollback(self.canonical_source())
        return self.status()

    def abandon(self) -> dict[str, Any]:
        self._admin_allowed()
        with self._admin_lock:
            self.router.abandon_dual_write()
        return self.status()


# --------------------------------------------------------------------------
# Process-wide wiring
# --------------------------------------------------------------------------

_switch: Optional[EngineSwitch] = None
_unavailable_reason: Optional[str] = "not_started"


def configure(switch: Optional[EngineSwitch]) -> None:
    global _switch, _unavailable_reason
    _switch = switch
    _unavailable_reason = None if switch is not None else "not_configured"


def get_switch() -> Optional[EngineSwitch]:
    return _switch


def unavailable_reason() -> Optional[str]:
    """Why this process has no switch (``None`` when it has one): reported by the admin routes."""

    return None if _switch is not None else _unavailable_reason


def start(
    settings: Any,
    *,
    ledger: Any,
    service: Any,
    tenant_for: Callable[[str], Optional[str]],
    embed: Callable[[str], list[float]],
    qdrant_client: Any = None,
) -> Optional[EngineSwitch]:
    """Build and install this process's switch at startup, or record why it is not available.

    Nothing is built unless ``ENGINE_ROUTING_ENABLED is True``.  The switch reads the
    ledger the process has open (its canonical source and erasure fences), so with no
    ledger open it stays unavailable; so it does when the routed engine cannot be built
    (the service keeps serving from its own store, and the admin routes say why).
    """

    global _unavailable_reason
    stop()
    if getattr(settings, "engine_routing_enabled", False) is not True:
        _unavailable_reason = "routing_disabled"
        return None
    if ledger is None:
        _unavailable_reason = "ledger_not_open"
    elif int(getattr(settings, "engine_routing_dimensions", 0) or 0) <= 0:
        _unavailable_reason = "dimensions_not_set"
    else:
        try:
            switch = build_default_switch(
                settings, ledger=ledger, service=service, tenant_for=tenant_for, embed=embed, qdrant_client=qdrant_client,
            )
        except Exception as error:  # noqa: BLE001 - startup goes on without the switch
            logger.warning("engine switch could not be built (%s); the routed engine stays off", type(error).__name__,
                           exc_info=True)
            _unavailable_reason = "build_failed"
        else:
            configure(switch)
            logger.info("engine switch started (generation %s, %s)", switch.router.generation, switch.router.phase.value)
            return switch
    logger.warning("ENGINE_ROUTING_ENABLED is set but the engine switch is unavailable: %s", _unavailable_reason)
    return None


def stop() -> None:
    """Uninstall this process's switch and release its clients (safe to call more than once)."""

    global _unavailable_reason
    switch = _switch
    configure(None)
    _unavailable_reason = "stopped"
    if switch is not None:
        switch.close()


def projection_switch(settings: Any) -> Optional[EngineSwitch]:
    """The switch to project into, or ``None`` unless ``ENGINE_ROUTING_ENABLED is True``."""

    if getattr(settings, "engine_routing_enabled", False) is not True:
        return None
    return _switch


_refusal_logged: set[str] = set()


def recall_switch(settings: Any) -> Optional[EngineSwitch]:
    """The switch to recall from, or ``None`` (the legacy path serves) unless recall is both
    enabled and ready.  A refusal for readiness is logged once per reason, never silent."""

    switch = projection_switch(settings)
    if switch is None or getattr(settings, "engine_routing_recall_enabled", False) is not True:
        return None
    ready, reason = switch.recall_ready()
    if not ready:
        if reason is not None and reason not in _refusal_logged:
            _refusal_logged.add(reason)
            logger.warning(
                "routed recall is refused (%s): the serving engine has not been brought to canonical state; "
                "the legacy path keeps serving. Run catch-up with no dual-write window open to backfill it.",
                reason,
            )
        return None
    _refusal_logged.discard("projection_not_synced")  # a later regression is flagged again
    return switch


def build_default_switch(
    settings: Any,
    *,
    ledger: Any,
    service: Any,
    tenant_for: Callable[[str], Optional[str]],
    embed: Callable[[str], list[float]],
    qdrant_client: Any = None,
) -> EngineSwitch:
    """Generation 1 = a Qdrant projection collection; the shadow = pgvector (configured separately).

    ``qdrant_client`` (the memory service's own client) is reused when given: a local on-disk
    Qdrant admits one client per storage folder.  A reused client is never closed by the switch.
    """

    from engine_adapters_current import QdrantVectorAdapter
    from engine_adapters_pgvector import PgVectorAdapter
    from mem0.vector_stores.qdrant import Qdrant

    dims = settings.engine_routing_dimensions
    if qdrant_client is not None:
        store = Qdrant(
            collection_name=settings.engine_routing_qdrant_collection, embedding_model_dims=dims, client=qdrant_client,
        )
        client_close = None
    else:
        store = Qdrant(
            collection_name=settings.engine_routing_qdrant_collection,
            embedding_model_dims=dims,
            url=settings.qdrant_url or None,
            path=None if settings.qdrant_url else settings.qdrant_path,
        )
        client_close = getattr(getattr(store, "client", None), "close", None)
    active = Backend("qdrant", QdrantVectorAdapter(store, embed, implementation_id="engine-qdrant"))

    def shadow() -> Backend:
        adapter = PgVectorAdapter(
            settings.engine_pgvector_dsn, embed=embed, dimensions=dims,
            schema=settings.engine_pgvector_schema, implementation_id="engine-pgvector",
        )
        adapter.ensure_schema()
        return Backend("pgvector", adapter)

    return EngineSwitch(
        EngineRouter(
            active,
            rollback_window_s=(getattr(settings, "engine_routing_rollback_window_seconds", 0.0) or None),
        ),
        ledger=ledger, service=service, tenant_for=tenant_for,
        shadow_factory=shadow if settings.engine_pgvector_dsn else None,
        recall_enabled=settings.engine_routing_recall_enabled is True,
        closers=(client_close,) if callable(client_close) else (),
    )


__all__ = [
    "EngineSwitch",
    "LedgerCanonicalSource",
    "build_default_switch",
    "configure",
    "get_switch",
    "labels_for",
    "projection_switch",
    "recall_switch",
    "start",
    "stop",
    "unavailable_reason",
]
