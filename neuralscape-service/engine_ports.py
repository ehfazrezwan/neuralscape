"""Engine ports, adapter registry, reference fakes and a conformance suite.

Replacing a vector store, a graph store and a whole memory engine are
different operations, so each has its own ``typing.Protocol`` port.  Inference
is split the same way: bounded decisions and schema-constrained generation are
separate ports.  Every operation carries an explicit tenant-bearing
``OperationContext``; an engine hit is only a candidate and never grants access.

The reference fakes are deterministic in-memory implementations used to
exercise the conformance suite.  They are *reference* fakes: they are not
qualified engines, make no retrieval-quality claim, and nothing here wires
into the running service.  ``activate`` checks declared capability facts only;
it does not resolve qualification evidence.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Protocol, runtime_checkable

from pydantic import model_validator

from contracts_common import ContractModel, OpaqueId, SafeCounter
from contracts_engines import (
    CapabilityHealth,
    CapabilityManifest,
    CapabilityOperationState,
    CapabilityRequirement,
    CapabilityViolation,
    EngineAdapterKind,
    validate_capability_requirements,
)
from contracts_errors import InternalErrorCode, PublicError, public_error_for
from contracts_inference import (
    AttemptUsage,
    DecisionHeadResult,
    DecisionOutcome,
    DecisionRequest,
    DecisionResult,
    GenerationOutcome,
    GenerationProposal,
    GenerationRequest,
    GenerationResult,
    InferenceResourceLimits,
    JSON_SCHEMA_DIALECT,
    SourceEvidence,
    SourceSpan,
    DecisionCandidate,
    DecisionHead,
    validate_decision_result,
    validate_generation_result,
)
from contracts_references import ReferenceHandle, SourceVersion

CONTRACT_VERSION = "candidate-v1"

OPERATIONS_BY_KIND: dict[EngineAdapterKind, tuple[str, ...]] = {
    EngineAdapterKind.VECTOR_STORE: ("apply", "remove", "get", "search"),
    EngineAdapterKind.GRAPH_STORE: (
        "apply_episode",
        "remove_by_source",
        "resolve_edge",
        "neighbors",
    ),
    EngineAdapterKind.MEMORY_ENGINE: ("ingest", "delete", "resolve", "retrieve"),
    EngineAdapterKind.INFERENCE: ("decide", "generate"),
}
"""Exact operation names an adapter of each kind declares in its manifest."""


# --------------------------------------------------------------------------
# Shared value models
# --------------------------------------------------------------------------


class OperationContext(ContractModel):
    """Tenant, correlation and deadline carried by every port operation."""

    tenant_id: OpaqueId
    correlation_id: OpaqueId
    deadline_ms: SafeCounter

    @model_validator(mode="after")
    def validate_deadline(self) -> "OperationContext":
        if self.deadline_ms == 0:
            raise ValueError("deadline_ms must be positive")
        return self


class EngineRecord(ContractModel):
    """A canonical record projected into an engine, under its canonical ID."""

    record_id: OpaqueId
    revision: SafeCounter
    tenant_id: OpaqueId
    text: str
    labels: tuple[OpaqueId, ...] = ()


class TenantFilter(ContractModel):
    """Engine-neutral filter; the tenant constraint is mandatory.

    ``authorized_record_ids`` is the policy-narrowed set pushed down to the
    engine (``None`` means no narrowing beyond tenant and labels).  It must be
    applied *before* ranking and limiting; filtering only the top hits can drop
    authorized results or leak counts.  For graph ports the ids are *source*
    record ids: an edge is visible when at least one live supporting source is
    authorized, and invisible when none is.
    """

    tenant_id: OpaqueId
    labels_all: tuple[OpaqueId, ...] = ()
    authorized_record_ids: Optional[tuple[OpaqueId, ...]] = None


class SearchHit(ContractModel):
    record_id: OpaqueId
    revision: SafeCounter
    score: float


class ProjectionReceipt(ContractModel):
    """Evidence that one canonical object revision was applied or removed.

    A receipt is not proof of retrievability or of current authorization.
    """

    object_id: OpaqueId
    revision: SafeCounter
    operation: str
    engine_id: OpaqueId
    engine_version: OpaqueId
    index_version: OpaqueId

    @model_validator(mode="after")
    def validate_operation(self) -> "ProjectionReceipt":
        if self.operation not in ("applied", "removed"):
            raise ValueError("receipt operation must be applied or removed")
        return self


class GraphEdge(ContractModel):
    edge_id: OpaqueId
    subject_id: OpaqueId
    predicate: OpaqueId
    object_id: OpaqueId


class GraphEpisode(ContractModel):
    """An episode derived from exactly one canonical source record."""

    episode_id: OpaqueId
    revision: SafeCounter
    tenant_id: OpaqueId
    source_record_id: OpaqueId
    edges: tuple[GraphEdge, ...]


class PortFailure(Exception):
    """Normalized failure carrying a canonical internal error code."""

    def __init__(self, code: InternalErrorCode, detail: str = "") -> None:
        if type(code) is not InternalErrorCode:
            raise TypeError("code must be an InternalErrorCode")
        super().__init__(code.value)
        self.code = code
        self.detail = detail

    def public(self) -> PublicError:
        """Disclosure-safe form; detail is never included."""

        return public_error_for(self.code)


# --------------------------------------------------------------------------
# Ports
# --------------------------------------------------------------------------


@runtime_checkable
class VectorStorePort(Protocol):
    """Revisioned projection store; owns neither membership nor truth."""

    def manifest(self) -> CapabilityManifest: ...
    def apply(self, ctx: OperationContext, record: EngineRecord) -> ProjectionReceipt: ...
    def remove(
        self, ctx: OperationContext, record_id: str, revision: int
    ) -> ProjectionReceipt: ...
    def get(self, ctx: OperationContext, record_id: str) -> Optional[EngineRecord]: ...
    def search(
        self, ctx: OperationContext, query: str, flt: TenantFilter, limit: int
    ) -> tuple[SearchHit, ...]: ...


@runtime_checkable
class GraphStorePort(Protocol):
    """Episode/edge projection with source-supported removal.

    ``remove_by_source`` contract: removing a source with live episodes removes
    them and raises the source's fence.  Removing a source that is already
    removed is idempotent (a receipt, the fence only rises); a removal older
    than the fence is ``stale_revision``.  The first removal of a source that
    was never applied reports ``missing_resource`` and still records the fence,
    so a delayed older apply is refused.
    """

    def manifest(self) -> CapabilityManifest: ...
    def apply_episode(
        self, ctx: OperationContext, episode: GraphEpisode
    ) -> ProjectionReceipt: ...
    def remove_by_source(
        self, ctx: OperationContext, source_record_id: str, revision: int
    ) -> ProjectionReceipt: ...
    def resolve_edge(self, ctx: OperationContext, edge_id: str) -> Optional[GraphEdge]: ...
    def neighbors(
        self, ctx: OperationContext, node_id: str, flt: TenantFilter
    ) -> tuple[GraphEdge, ...]: ...


@runtime_checkable
class MemoryEnginePort(Protocol):
    """Engine that owns extraction/index/query; IDs stay canonical."""

    def manifest(self) -> CapabilityManifest: ...
    def ingest(self, ctx: OperationContext, record: EngineRecord) -> ProjectionReceipt: ...
    def delete(
        self, ctx: OperationContext, record_id: str, revision: int
    ) -> ProjectionReceipt: ...
    def resolve(self, ctx: OperationContext, record_id: str) -> Optional[EngineRecord]: ...
    def retrieve(
        self, ctx: OperationContext, query: str, flt: TenantFilter, limit: int
    ) -> tuple[SearchHit, ...]: ...


@runtime_checkable
class DecisionPort(Protocol):
    """Bounded selection over supplied candidates; never widens scope."""

    def manifest(self) -> CapabilityManifest: ...
    def decide(self, ctx: OperationContext, request: DecisionRequest) -> DecisionResult: ...


@runtime_checkable
class GenerationPort(Protocol):
    """Schema-constrained machine-authored proposals with source support."""

    def manifest(self) -> CapabilityManifest: ...
    def generate(
        self, ctx: OperationContext, request: GenerationRequest
    ) -> GenerationResult: ...


PORTS_BY_KIND: dict[EngineAdapterKind, tuple[type, ...]] = {
    EngineAdapterKind.VECTOR_STORE: (VectorStorePort,),
    EngineAdapterKind.GRAPH_STORE: (GraphStorePort,),
    EngineAdapterKind.MEMORY_ENGINE: (MemoryEnginePort,),
    EngineAdapterKind.INFERENCE: (DecisionPort, GenerationPort),
}
"""An inference adapter must implement both inference ports."""


# --------------------------------------------------------------------------
# Registry and activation validator
# --------------------------------------------------------------------------


class ActivationRejected(Exception):
    """Activation refused before any data was sent to the adapter."""

    def __init__(self, reason: str, violations: tuple[CapabilityViolation, ...] = ()):
        super().__init__(reason)
        self.reason = reason
        self.violations = violations


class ActivationReceipt(ContractModel):
    adapter_id: OpaqueId
    adapter_kind: EngineAdapterKind
    implementation_version: OpaqueId
    routing_generation: SafeCounter


def _read_manifest(port: object) -> CapabilityManifest:
    """Call ``manifest()`` once; any failure or wrong type is a rejection."""

    try:
        manifest = port.manifest()  # type: ignore[attr-defined]
    except Exception as error:
        raise ActivationRejected("adapter manifest could not be read") from error
    if not isinstance(manifest, CapabilityManifest):
        raise ActivationRejected("adapter manifest is not a CapabilityManifest")
    return manifest


def validate_activation(
    port: object,
    kind: EngineAdapterKind,
    requirements: tuple[CapabilityRequirement, ...],
) -> CapabilityManifest:
    """Return the port's manifest if it structurally satisfies ``requirements``.

    Only ``manifest()`` is called on the port.  Satisfied declarations are not
    runtime authorization or proof of qualification.
    """

    if not isinstance(kind, EngineAdapterKind):
        raise TypeError("kind must be an EngineAdapterKind")
    if not all(isinstance(port, protocol) for protocol in PORTS_BY_KIND[kind]):
        raise ActivationRejected("adapter does not implement the required port")
    manifest = _read_manifest(port)
    if manifest.adapter_kind is not kind:
        raise ActivationRejected("manifest adapter kind does not match the port kind")
    declared = {state.operation for state in manifest.operations}
    missing = tuple(
        name for name in OPERATIONS_BY_KIND[kind] if name not in declared
    )
    if missing:
        raise ActivationRejected("manifest omits operations of the port")
    required_names = {requirement.operation for requirement in requirements}
    unknown = required_names - set(OPERATIONS_BY_KIND[kind])
    if unknown:
        raise ActivationRejected("requirement names an operation outside the port")
    violations = validate_capability_requirements(manifest, requirements)
    if violations:
        raise ActivationRejected("capability requirements are not met", violations)
    return manifest


class RoutingGenerationFence:
    """A monotonic routing generation that stale callers are checked against.

    Every read and write is routed under the generation current when it was
    routed and carries it to the point of use; after a switch the old
    generation is rejected there, so an operation routed to a previous adapter
    can never be applied (or served) after the flip.  ``advance`` is the only
    way the generation changes and it never goes backwards: a rollback is a new
    generation bound to the previous adapter, not a return to an old number.
    """

    def __init__(self, initial: int = 0) -> None:
        if type(initial) is not int or initial < 0:
            raise ValueError("initial generation must be a non-negative integer")
        self._lock = threading.Lock()
        self._generation = initial

    @property
    def current(self) -> int:
        with self._lock:
            return self._generation

    def advance(self) -> int:
        with self._lock:
            self._generation += 1
            return self._generation

    def check(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                raise PortFailure(InternalErrorCode.STALE_REVISION, "routing generation is stale")


class RoutedPort(NamedTuple):
    """The active port together with the generation it was routed under."""

    generation: int
    port: object


class AdapterRegistry:
    """Registered adapters and at most one active adapter per kind.

    Activation changes the routing generation; a deactivated adapter must be
    reactivated (and re-validated) rather than silently resuming service.
    ``route`` hands out ``(generation, port)`` atomically; ``check_generation``
    rejects a generation that is no longer current.
    """

    def __init__(self) -> None:
        self._ports: dict[str, tuple[EngineAdapterKind, object]] = {}
        self._active: dict[EngineAdapterKind, str] = {}
        self._fence = RoutingGenerationFence()
        self._lock = threading.RLock()

    @property
    def routing_generation(self) -> int:
        return self._fence.current

    def check_generation(self, generation: int) -> None:
        """Raise ``PortFailure(STALE_REVISION)`` unless ``generation`` is current."""

        self._fence.check(generation)

    def route(self, kind: EngineAdapterKind) -> RoutedPort:
        with self._lock:
            return RoutedPort(self._fence.current, self.active_port(kind))

    def register(self, port: object, kind: EngineAdapterKind) -> str:
        if not isinstance(kind, EngineAdapterKind):
            raise TypeError("kind must be an EngineAdapterKind")
        if not all(isinstance(port, protocol) for protocol in PORTS_BY_KIND[kind]):
            raise ActivationRejected("adapter does not implement the required port")
        manifest = _read_manifest(port)
        adapter_id = manifest.implementation_id
        if adapter_id in self._ports:
            raise ActivationRejected("adapter id is already registered")
        self._ports[adapter_id] = (kind, port)
        return adapter_id

    def activate(
        self,
        adapter_id: str,
        requirements: tuple[CapabilityRequirement, ...],
    ) -> ActivationReceipt:
        entry = self._ports.get(adapter_id)
        if entry is None:
            raise ActivationRejected("adapter is not registered")
        kind, port = entry
        manifest = validate_activation(port, kind, requirements)
        if manifest.implementation_id != adapter_id:
            raise ActivationRejected("manifest identity changed since registration")
        with self._lock:
            self._active[kind] = adapter_id
            generation = self._fence.advance()
        return ActivationReceipt(
            adapter_id=adapter_id,
            adapter_kind=kind,
            implementation_version=manifest.implementation_version,
            routing_generation=generation,
        )

    def deactivate(self, kind: EngineAdapterKind) -> None:
        with self._lock:
            if kind in self._active:
                del self._active[kind]
                self._fence.advance()

    def active_port(self, kind: EngineAdapterKind) -> object:
        adapter_id = self._active.get(kind)
        if adapter_id is None:
            raise PortFailure(InternalErrorCode.UNAVAILABLE_DEPENDENCY, "no active adapter")
        return self._ports[adapter_id][1]


# --------------------------------------------------------------------------
# Reference fakes (deterministic, in-memory, NOT qualified engines)
# --------------------------------------------------------------------------

_REFERENCE_VERSION = "reference-v1"


def reference_manifest(
    implementation_id: str,
    kind: EngineAdapterKind,
    *,
    health: CapabilityHealth = CapabilityHealth.HEALTHY,
) -> CapabilityManifest:
    """Manifest of a reference fake: supported and configured, never qualified."""

    return CapabilityManifest(
        schema_version=CONTRACT_VERSION,
        implementation_id=implementation_id,
        implementation_version=_REFERENCE_VERSION,
        adapter_kind=kind,
        contract_schema_versions=(CONTRACT_VERSION,),
        operations=tuple(
            CapabilityOperationState(
                operation=name,
                supported=True,
                configured=True,
                health=health,
                qualification=None,
            )
            for name in OPERATIONS_BY_KIND[kind]
        ),
    )


def _tokens(text: str) -> set[str]:
    return set(text.lower().split())


def _check_tenant(ctx: OperationContext, tenant_id: str) -> None:
    if ctx.tenant_id != tenant_id:
        raise PortFailure(InternalErrorCode.DENIED, "context tenant mismatch")


class _ReferenceRecordStore:
    """Shared record semantics for the vector and memory-engine references."""

    def __init__(self, implementation_id: str, kind: EngineAdapterKind) -> None:
        self._implementation_id = implementation_id
        self._kind = kind
        # (tenant, record_id) -> (record, removed)
        self._rows: dict[tuple[str, str], tuple[EngineRecord, bool]] = {}

    def manifest(self) -> CapabilityManifest:
        return reference_manifest(self._implementation_id, self._kind)

    def _receipt(self, record_id: str, revision: int, operation: str) -> ProjectionReceipt:
        return ProjectionReceipt(
            object_id=record_id,
            revision=revision,
            operation=operation,
            engine_id=self._implementation_id,
            engine_version=_REFERENCE_VERSION,
            index_version=_REFERENCE_VERSION,
        )

    def _apply(self, ctx: OperationContext, record: EngineRecord) -> ProjectionReceipt:
        _check_tenant(ctx, record.tenant_id)
        key = (record.tenant_id, record.record_id)
        existing = self._rows.get(key)
        if existing is not None:
            stored, removed = existing
            if record.revision < stored.revision:
                raise PortFailure(InternalErrorCode.STALE_REVISION)
            # The same revision with the same text and other labels is a relabel (the projection's labels
            # are derived from the memory row and may be corrected); other text at the revision is a conflict.
            if record.revision == stored.revision and (removed or record.text != stored.text):
                raise PortFailure(InternalErrorCode.IDEMPOTENCY_CONFLICT)
        self._rows[key] = (record, False)
        return self._receipt(record.record_id, record.revision, "applied")

    def _remove(
        self, ctx: OperationContext, record_id: str, revision: int
    ) -> ProjectionReceipt:
        key = (ctx.tenant_id, record_id)
        existing = self._rows.get(key)
        if existing is None:
            raise PortFailure(InternalErrorCode.MISSING_RESOURCE)
        stored, _removed = existing
        if revision < stored.revision:
            raise PortFailure(InternalErrorCode.STALE_REVISION)
        # A removal keeps the revision fence and nothing of the content.
        self._rows[key] = (stored.model_copy(update={"revision": revision, "text": "", "labels": ()}), True)
        return self._receipt(record_id, revision, "removed")

    def _get(self, ctx: OperationContext, record_id: str) -> Optional[EngineRecord]:
        existing = self._rows.get((ctx.tenant_id, record_id))
        if existing is None or existing[1]:
            return None
        return existing[0]

    def _search(
        self, ctx: OperationContext, query: str, flt: TenantFilter, limit: int
    ) -> tuple[SearchHit, ...]:
        _check_tenant(ctx, flt.tenant_id)
        wanted = _tokens(query)
        allowed = None if flt.authorized_record_ids is None else set(flt.authorized_record_ids)
        hits = []
        for (tenant_id, record_id), (record, removed) in self._rows.items():
            if removed or tenant_id != flt.tenant_id:
                continue
            if allowed is not None and record_id not in allowed:
                continue
            if not set(flt.labels_all).issubset(record.labels):
                continue
            score = len(wanted & _tokens(record.text))
            if score:
                hits.append(SearchHit(record_id=record_id, revision=record.revision, score=float(score)))
        hits.sort(key=lambda hit: (-hit.score, hit.record_id))
        return tuple(hits[: max(limit, 0)])


class ReferenceVectorStore(_ReferenceRecordStore):
    """Reference vector store (token-overlap ranking). Not a qualified engine."""

    def __init__(self, implementation_id: str = "reference-vector") -> None:
        super().__init__(implementation_id, EngineAdapterKind.VECTOR_STORE)

    apply = _ReferenceRecordStore._apply
    remove = _ReferenceRecordStore._remove
    get = _ReferenceRecordStore._get
    search = _ReferenceRecordStore._search


class ReferenceMemoryEngine(_ReferenceRecordStore):
    """Reference full engine; keeps canonical IDs. Not a qualified engine."""

    def __init__(self, implementation_id: str = "reference-engine") -> None:
        super().__init__(implementation_id, EngineAdapterKind.MEMORY_ENGINE)

    ingest = _ReferenceRecordStore._apply
    delete = _ReferenceRecordStore._remove
    resolve = _ReferenceRecordStore._get
    retrieve = _ReferenceRecordStore._search


class ReferenceGraphStore:
    """Reference graph store with source-supported edge removal."""

    def __init__(self, implementation_id: str = "reference-graph") -> None:
        self._implementation_id = implementation_id
        # (tenant, episode_id) -> (episode, removed)
        self._episodes: dict[tuple[str, str], tuple[GraphEpisode, bool]] = {}
        # (tenant, source_record_id) -> revision of the latest removal
        self._fences: dict[tuple[str, str], int] = {}
        self._lock = threading.RLock()  # every operation is atomic

    def manifest(self) -> CapabilityManifest:
        return reference_manifest(self._implementation_id, EngineAdapterKind.GRAPH_STORE)

    def _receipt(self, object_id: str, revision: int, operation: str) -> ProjectionReceipt:
        return ProjectionReceipt(
            object_id=object_id,
            revision=revision,
            operation=operation,
            engine_id=self._implementation_id,
            engine_version=_REFERENCE_VERSION,
            index_version=_REFERENCE_VERSION,
        )

    def apply_episode(self, ctx: OperationContext, episode: GraphEpisode) -> ProjectionReceipt:
        _check_tenant(ctx, episode.tenant_id)
        with self._lock:
            key = (episode.tenant_id, episode.episode_id)
            existing = self._episodes.get(key)
            if existing is not None and episode.revision < existing[0].revision:
                raise PortFailure(InternalErrorCode.STALE_REVISION)
            fence = self._fences.get((episode.tenant_id, episode.source_record_id))
            if fence is not None and episode.revision <= fence:
                raise PortFailure(InternalErrorCode.STALE_REVISION)
            self._episodes[key] = (episode, False)
            return self._receipt(episode.episode_id, episode.revision, "applied")

    def remove_by_source(
        self, ctx: OperationContext, source_record_id: str, revision: int
    ) -> ProjectionReceipt:
        with self._lock:
            found = False
            for key, (episode, removed) in list(self._episodes.items()):
                if (
                    key[0] == ctx.tenant_id
                    and episode.source_record_id == source_record_id
                    and not removed
                ):
                    if revision < episode.revision:
                        raise PortFailure(InternalErrorCode.STALE_REVISION)
                    self._episodes[key] = (episode, True)
                    found = True
            # The fence only rises and is recorded even when nothing was found,
            # so a delayed older apply cannot resurrect the source.
            fence_key = (ctx.tenant_id, source_record_id)
            prior = self._fences.get(fence_key)
            if not found and prior is not None and revision < prior:
                raise PortFailure(InternalErrorCode.STALE_REVISION)  # older than a removal already recorded
            self._fences[fence_key] = max(revision, prior if prior is not None else revision)
            if not found and prior is None:
                raise PortFailure(InternalErrorCode.MISSING_RESOURCE)  # nothing was ever there
            # Otherwise the source is removed (now, or already): a repeat is idempotent.
            return self._receipt(source_record_id, revision, "removed")

    def _live_edges(self, tenant_id: str, authorized: Optional[set[str]] = None):
        for (tenant, _episode_id), (episode, removed) in sorted(self._episodes.items()):
            if tenant != tenant_id or removed:
                continue
            if authorized is not None and episode.source_record_id not in authorized:
                continue
            yield from episode.edges

    def resolve_edge(self, ctx: OperationContext, edge_id: str) -> Optional[GraphEdge]:
        with self._lock:
            return next((e for e in self._live_edges(ctx.tenant_id) if e.edge_id == edge_id), None)

    def neighbors(
        self, ctx: OperationContext, node_id: str, flt: TenantFilter
    ) -> tuple[GraphEdge, ...]:
        _check_tenant(ctx, flt.tenant_id)
        authorized = (
            None if flt.authorized_record_ids is None else set(flt.authorized_record_ids)
        )
        with self._lock:
            seen: dict[str, GraphEdge] = {}
            for edge in self._live_edges(flt.tenant_id, authorized):
                if node_id in (edge.subject_id, edge.object_id):
                    seen.setdefault(edge.edge_id, edge)
            return tuple(sorted(seen.values(), key=lambda e: e.edge_id))


def _span_text(request: "DecisionRequest | GenerationRequest", span: SourceSpan) -> str:
    for evidence in request.sources:
        if evidence.source_version == span.source_version:
            raw = evidence.text.encode("utf-8")[span.start_utf8_byte : span.end_utf8_byte]
            return raw.decode("utf-8", errors="ignore")
    return ""


def _attempt(request_id: str, input_bytes: int, output_bytes: int) -> AttemptUsage:
    return AttemptUsage(
        attempt_id=f"{request_id}-attempt-1",
        model_version=_REFERENCE_VERSION,
        policy_version=_REFERENCE_VERSION,
        input_utf8_bytes=input_bytes,
        output_utf8_bytes=output_bytes,
        late=False,
    )


class ReferenceInference:
    """Reference decision and generation rules (deterministic, not a model).

    Decision: blank source -> ABSTAINED; no candidate id appears in the span ->
    valid-empty; otherwise SELECTED.  Generation: blank -> ABSTAINED, comment
    only (``#``) -> valid-empty, otherwise one proposal with source support.
    """

    def __init__(self, implementation_id: str = "reference-inference") -> None:
        self._implementation_id = implementation_id

    def manifest(self) -> CapabilityManifest:
        return reference_manifest(self._implementation_id, EngineAdapterKind.INFERENCE)

    def decide(self, ctx: OperationContext, request: DecisionRequest) -> DecisionResult:
        spans = {span.span_id: span for span in request.spans}
        text = " ".join(_span_text(request, spans[sid]) for sid in request.head.span_ids)
        if not text.strip():
            head = DecisionHeadResult(
                head_id=request.head.head_id,
                outcome=DecisionOutcome.ABSTAINED,
                selected_candidate_ids=(),
                distribution=None,
                reason_code="unsupported_input",
            )
        else:
            lowered = text.lower()
            chosen = tuple(
                cid for cid in request.head.candidate_ids if cid.lower() in lowered
            )
            head = DecisionHeadResult(
                head_id=request.head.head_id,
                outcome=DecisionOutcome.SELECTED if chosen else DecisionOutcome.EMPTY,
                selected_candidate_ids=chosen,
                distribution=None,
                reason_code=None,
            )
        return DecisionResult(
            schema_version=CONTRACT_VERSION,
            request_id=request.request_id,
            head_result=head,
            attempts=(_attempt(request.request_id, len(text.encode("utf-8")), 0),),
        )

    def generate(self, ctx: OperationContext, request: GenerationRequest) -> GenerationResult:
        text = " ".join(_span_text(request, span) for span in request.spans)
        proposals: tuple[GenerationProposal, ...] = ()
        reason = None
        if not text.strip():
            outcome, reason = GenerationOutcome.ABSTAINED, "unsupported_input"
        elif text.strip().startswith("#"):
            outcome = GenerationOutcome.EMPTY
        else:
            outcome = GenerationOutcome.PROPOSED
            proposals = (
                GenerationProposal(
                    proposal_id=f"{request.request_id}-p1",
                    output={"summary": text.strip()},
                    support_span_ids=tuple(span.span_id for span in request.spans),
                    machine_authored=True,
                ),
            )
        return GenerationResult(
            schema_version=CONTRACT_VERSION,
            request_id=request.request_id,
            outcome=outcome,
            proposals=proposals,
            reason_code=reason,
            attempts=(_attempt(request.request_id, len(text.encode("utf-8")), 0),),
        )


# --------------------------------------------------------------------------
# Conformance suite
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConformanceFailure:
    check: str
    detail: str


def _ctx(tenant_id: str = "tenant-a") -> OperationContext:
    return OperationContext(tenant_id=tenant_id, correlation_id="corr-1", deadline_ms=1000)


def _record(record_id: str, revision: int, text: str, tenant_id: str = "tenant-a", labels=()) -> EngineRecord:
    return EngineRecord(
        record_id=record_id, revision=revision, tenant_id=tenant_id, text=text, labels=tuple(labels)
    )


class _RecordSurface:
    """Operation names of a record-owning port, so one suite covers two ports."""

    def __init__(self, port: object, apply: str, remove: str, get: str, search: str) -> None:
        self.apply = getattr(port, apply)
        self.remove = getattr(port, remove)
        self.get = getattr(port, get)
        self.search = getattr(port, search)


def _run_checks(checks: list[tuple[str, Callable[[], None]]]) -> tuple[ConformanceFailure, ...]:
    failures = []
    for name, check in checks:
        try:
            check()
        except AssertionError as error:
            failures.append(ConformanceFailure(name, str(error) or "assertion failed"))
        except Exception as error:  # an unexpected error is a conformance failure
            failures.append(ConformanceFailure(name, f"{type(error).__name__}: {error}"))
    return tuple(failures)


def _expect_failure(code: InternalErrorCode, call: Callable[[], object]) -> None:
    try:
        call()
    except PortFailure as failure:
        assert failure.code is code, f"expected {code.value}, got {failure.code.value}"
        return
    raise AssertionError(f"expected PortFailure({code.value}) but the call succeeded")


def _record_checks(factory: Callable[[], object], names: tuple[str, str, str, str]):
    def surface() -> _RecordSurface:
        return _RecordSurface(factory(), *names)

    def ids_and_receipts_exact() -> None:
        s = surface()
        receipt = s.apply(_ctx(), _record("rec-1", 1, "red apple"))
        assert receipt.object_id == "rec-1" and receipt.revision == 1, "receipt must name the canonical id/revision"
        stored = s.get(_ctx(), "rec-1")
        assert stored is not None and stored.record_id == "rec-1", "canonical id must be preserved"
        hits = s.search(_ctx(), "apple", TenantFilter(tenant_id="tenant-a"), 10)
        assert [hit.record_id for hit in hits] == ["rec-1"], "hits must carry canonical ids"

    def tenant_filter_enforced() -> None:
        s = surface()
        s.apply(_ctx("tenant-a"), _record("rec-a", 1, "shared words", "tenant-a"))
        s.apply(_ctx("tenant-b"), _record("rec-b", 1, "shared words", "tenant-b"))
        hits = s.search(_ctx("tenant-a"), "shared", TenantFilter(tenant_id="tenant-a"), 10)
        assert [hit.record_id for hit in hits] == ["rec-a"], "search leaked another tenant's record"
        assert s.get(_ctx("tenant-a"), "rec-b") is None, "get leaked another tenant's record"

    def context_tenant_mismatch_denied() -> None:
        s = surface()
        _expect_failure(
            InternalErrorCode.DENIED,
            lambda: s.apply(_ctx("tenant-a"), _record("rec-x", 1, "text", "tenant-b")),
        )

    def authorized_ids_applied_before_limit() -> None:
        s = surface()
        s.apply(_ctx(), _record("rec-1", 1, "alpha alpha alpha"))
        s.apply(_ctx(), _record("rec-2", 1, "alpha"))
        flt = TenantFilter(tenant_id="tenant-a", authorized_record_ids=("rec-2",))
        hits = s.search(_ctx(), "alpha", flt, 1)
        assert [hit.record_id for hit in hits] == ["rec-2"], "inaccessible distractor returned or authorized hit dropped"
        empty = TenantFilter(tenant_id="tenant-a", authorized_record_ids=())
        assert s.search(_ctx(), "alpha", empty, 10) == (), "empty authorization must return nothing"

    def delete_is_exact() -> None:
        s = surface()
        s.apply(_ctx(), _record("rec-1", 1, "gone soon"))
        s.apply(_ctx(), _record("rec-2", 1, "gone later"))
        s.remove(_ctx(), "rec-1", 2)
        assert s.get(_ctx(), "rec-1") is None, "removed record is still readable"
        ids = [h.record_id for h in s.search(_ctx(), "gone", TenantFilter(tenant_id="tenant-a"), 10)]
        assert ids == ["rec-2"], "removed record is still searchable or neighbours were removed"

    def stale_revision_rejected() -> None:
        s = surface()
        s.apply(_ctx(), _record("rec-1", 2, "new text"))
        _expect_failure(
            InternalErrorCode.STALE_REVISION,
            lambda: s.apply(_ctx(), _record("rec-1", 1, "old text")),
        )
        current = s.get(_ctx(), "rec-1")
        assert current is not None and current.text == "new text", "stale apply replaced current state"

    def same_revision_conflict_detected() -> None:
        s = surface()
        s.apply(_ctx(), _record("rec-1", 1, "first"))
        s.apply(_ctx(), _record("rec-1", 1, "first"))  # identical replay is idempotent
        _expect_failure(
            InternalErrorCode.IDEMPOTENCY_CONFLICT,
            lambda: s.apply(_ctx(), _record("rec-1", 1, "different")),
        )

    def update_replaces_postings() -> None:
        s = surface()
        s.apply(_ctx(), _record("rec-1", 1, "oldterm shared"))
        s.apply(_ctx(), _record("rec-1", 2, "newterm shared"))
        hits = s.search(_ctx(), "oldterm", TenantFilter(tenant_id="tenant-a"), 10)
        assert all(hit.revision == 2 for hit in hits), "a superseded revision is still served"
        current = s.get(_ctx(), "rec-1")
        assert current is not None and current.text == "newterm shared", "update did not replace the record"

    def no_resurrection_after_remove() -> None:
        s = surface()
        s.apply(_ctx(), _record("rec-1", 3, "alive"))
        s.remove(_ctx(), "rec-1", 4)
        _expect_failure(
            InternalErrorCode.STALE_REVISION,
            lambda: s.apply(_ctx(), _record("rec-1", 2, "older")),
        )
        assert s.get(_ctx(), "rec-1") is None, "an older revision was resurrected after removal"

    def stale_remove_refused() -> None:
        s = surface()
        s.apply(_ctx(), _record("rec-1", 5, "keep me"))
        _expect_failure(InternalErrorCode.STALE_REVISION, lambda: s.remove(_ctx(), "rec-1", 4))
        assert s.get(_ctx(), "rec-1") is not None, "a stale remove deleted the current record"

    def remove_receipt_exact() -> None:
        s = surface()
        s.apply(_ctx(), _record("rec-1", 1, "text"))
        receipt = s.remove(_ctx(), "rec-1", 2)
        assert receipt.operation == "removed", "remove must return a removed receipt"
        assert (receipt.object_id, receipt.revision) == ("rec-1", 2), "remove receipt must name the id/revision"

    def search_filter_tenant_mismatch_denied() -> None:
        s = surface()
        s.apply(_ctx("tenant-b"), _record("rec-b", 1, "secret words", "tenant-b"))
        _expect_failure(
            InternalErrorCode.DENIED,
            lambda: s.search(_ctx("tenant-a"), "secret", TenantFilter(tenant_id="tenant-b"), 10),
        )

    def same_id_two_tenants_isolated() -> None:
        s = surface()
        s.apply(_ctx("tenant-a"), _record("rec-1", 5, "alpha", "tenant-a"))
        s.apply(_ctx("tenant-b"), _record("rec-1", 1, "beta", "tenant-b"))  # not stale for tenant-b
        a = s.get(_ctx("tenant-a"), "rec-1")
        b = s.get(_ctx("tenant-b"), "rec-1")
        assert a is not None and a.text == "alpha" and a.revision == 5, "tenant A state was clobbered"
        assert b is not None and b.text == "beta" and b.revision == 1, "tenant B state is missing or wrong"
        s.remove(_ctx("tenant-a"), "rec-1", 6)
        assert s.get(_ctx("tenant-a"), "rec-1") is None, "tenant A removal failed"
        assert s.get(_ctx("tenant-b"), "rec-1") is not None, "removal crossed tenants"

    def labels_honored() -> None:
        s = surface()
        s.apply(_ctx(), _record("rec-1", 1, "cat", labels=("a",)))
        s.apply(_ctx(), _record("rec-2", 1, "cat", labels=("a", "b")))
        flt = TenantFilter(tenant_id="tenant-a", labels_all=("a", "b"))
        assert [h.record_id for h in s.search(_ctx(), "cat", flt, 10)] == ["rec-2"], "labels_all was not applied"

    def limit_honored() -> None:
        s = surface()
        s.apply(_ctx(), _record("rec-1", 1, "cat"))
        s.apply(_ctx(), _record("rec-2", 1, "cat"))
        flt = TenantFilter(tenant_id="tenant-a")
        assert len(s.search(_ctx(), "cat", flt, 1)) == 1, "limit was not honored"
        assert s.search(_ctx(), "cat", flt, 0) == (), "a zero limit must return nothing"

    return [
        ("ids_and_receipts_exact", ids_and_receipts_exact),
        ("tenant_filter_enforced", tenant_filter_enforced),
        ("context_tenant_mismatch_denied", context_tenant_mismatch_denied),
        ("authorized_ids_applied_before_limit", authorized_ids_applied_before_limit),
        ("delete_is_exact", delete_is_exact),
        ("stale_revision_rejected", stale_revision_rejected),
        ("same_revision_conflict_detected", same_revision_conflict_detected),
        ("update_replaces_postings", update_replaces_postings),
        ("no_resurrection_after_remove", no_resurrection_after_remove),
        ("stale_remove_refused", stale_remove_refused),
        ("remove_receipt_exact", remove_receipt_exact),
        ("search_filter_tenant_mismatch_denied", search_filter_tenant_mismatch_denied),
        ("same_id_two_tenants_isolated", same_id_two_tenants_isolated),
        ("labels_honored", labels_honored),
        ("limit_honored", limit_honored),
    ]


def run_vector_store_conformance(
    factory: Callable[[], VectorStorePort],
) -> tuple[ConformanceFailure, ...]:
    return _run_checks(_record_checks(factory, ("apply", "remove", "get", "search")))


def run_memory_engine_conformance(
    factory: Callable[[], MemoryEnginePort],
) -> tuple[ConformanceFailure, ...]:
    return _run_checks(_record_checks(factory, ("ingest", "delete", "resolve", "retrieve")))


def _parallel(*calls: Callable[[], object]) -> list[object]:
    """Run calls concurrently from a barrier; return results or raised exceptions."""

    barrier = threading.Barrier(len(calls))
    results: list[object] = [None] * len(calls)

    def runner(index: int, call: Callable[[], object]) -> None:
        barrier.wait()
        try:
            results[index] = call()
        except Exception as error:  # reported to the caller, not raised here
            results[index] = error

    threads = [threading.Thread(target=runner, args=(i, c)) for i, c in enumerate(calls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def run_graph_store_conformance(
    factory: Callable[[], GraphStorePort],
) -> tuple[ConformanceFailure, ...]:
    def episode(eid: str, source: str, revision: int, tenant: str = "tenant-a", edge: str = "e1"):
        return GraphEpisode(
            episode_id=eid,
            revision=revision,
            tenant_id=tenant,
            source_record_id=source,
            edges=(GraphEdge(edge_id=edge, subject_id="n1", predicate="likes", object_id="n2"),),
        )

    flt = TenantFilter(tenant_id="tenant-a")

    def tenant_isolated() -> None:
        g = factory()
        g.apply_episode(_ctx("tenant-a"), episode("ep-a", "src-a", 1))
        g.apply_episode(_ctx("tenant-b"), episode("ep-b", "src-b", 1, "tenant-b", "e-b"))
        assert g.resolve_edge(_ctx("tenant-a"), "e-b") is None, "resolve leaked another tenant's edge"
        assert [e.edge_id for e in g.neighbors(_ctx("tenant-a"), "n1", flt)] == ["e1"], "neighbors leaked another tenant"

    def removal_is_source_supported() -> None:
        g = factory()
        g.apply_episode(_ctx(), episode("ep-1", "src-1", 1))
        g.apply_episode(_ctx(), episode("ep-2", "src-2", 1))  # same edge, second source
        g.remove_by_source(_ctx(), "src-1", 2)
        assert g.resolve_edge(_ctx(), "e1") is not None, "edge removed while another source still supports it"
        g.remove_by_source(_ctx(), "src-2", 2)
        assert g.resolve_edge(_ctx(), "e1") is None, "edge still resolvable after all support was removed"
        assert g.neighbors(_ctx(), "n1", flt) == (), "neighbors still return a removed edge"

    def stale_episode_rejected() -> None:
        g = factory()
        g.apply_episode(_ctx(), episode("ep-1", "src-1", 2, edge="e-new"))
        _expect_failure(
            InternalErrorCode.STALE_REVISION,
            lambda: g.apply_episode(_ctx(), episode("ep-1", "src-1", 1, edge="e-old")),
        )
        assert g.resolve_edge(_ctx(), "e-old") is None, "stale episode was applied"

    def receipt_exact() -> None:
        g = factory()
        receipt = g.apply_episode(_ctx(), episode("ep-1", "src-1", 3))
        assert (receipt.object_id, receipt.revision) == ("ep-1", 3), "receipt must name the episode id/revision"

    def context_tenant_mismatch_denied() -> None:
        g = factory()
        _expect_failure(
            InternalErrorCode.DENIED,
            lambda: g.apply_episode(_ctx("tenant-a"), episode("ep-x", "src-x", 1, "tenant-b")),
        )

    def removal_scoped_to_tenant() -> None:
        g = factory()
        g.apply_episode(_ctx("tenant-a"), episode("ep-a", "src-shared", 1, "tenant-a", "e-a"))
        g.apply_episode(_ctx("tenant-b"), episode("ep-b", "src-shared", 1, "tenant-b", "e-b"))
        g.remove_by_source(_ctx("tenant-a"), "src-shared", 2)
        assert g.resolve_edge(_ctx("tenant-a"), "e-a") is None, "tenant A removal did not remove its edge"
        assert g.resolve_edge(_ctx("tenant-b"), "e-b") is not None, "removal crossed tenants via a shared source id"

    def no_resurrection_after_source_removal() -> None:
        g = factory()
        g.apply_episode(_ctx(), episode("ep-1", "src-1", 1))
        g.remove_by_source(_ctx(), "src-1", 4)
        _expect_failure(
            InternalErrorCode.STALE_REVISION,
            lambda: g.apply_episode(_ctx(), episode("ep-1", "src-1", 2)),
        )
        assert g.resolve_edge(_ctx(), "e1") is None, "an older episode was resurrected after its source was removed"

    def neighbors_filter_tenant_mismatch_denied() -> None:
        g = factory()
        g.apply_episode(_ctx("tenant-b"), episode("ep-b", "src-b", 1, "tenant-b", "e-b"))
        _expect_failure(
            InternalErrorCode.DENIED,
            lambda: g.neighbors(_ctx("tenant-a"), "n1", TenantFilter(tenant_id="tenant-b")),
        )

    def neighbors_respect_authorized_sources() -> None:
        g = factory()
        g.apply_episode(_ctx(), episode("ep-secret", "src-secret", 1, edge="e-secret"))
        g.apply_episode(_ctx(), episode("ep-pub", "src-public", 1, edge="e-both"))
        g.apply_episode(_ctx(), episode("ep-sec2", "src-secret", 1, edge="e-both"))
        only_public = TenantFilter(tenant_id="tenant-a", authorized_record_ids=("src-public",))
        ids = [e.edge_id for e in g.neighbors(_ctx(), "n1", only_public)]
        assert ids == ["e-both"], "an edge supported only by an unauthorized source was returned (or a partly supported edge hidden)"
        nobody = TenantFilter(tenant_id="tenant-a", authorized_record_ids=())
        assert g.neighbors(_ctx(), "n1", nobody) == (), "empty authorization must return nothing"

    def removal_order_independent() -> None:
        for first, second in (("src-1", "src-2"), ("src-2", "src-1")):
            g = factory()
            g.apply_episode(_ctx(), episode("ep-1", "src-1", 1, edge="e-shared"))
            g.apply_episode(_ctx(), episode("ep-2", "src-2", 1, edge="e-shared"))
            g.remove_by_source(_ctx(), first, 2)
            assert g.resolve_edge(_ctx(), "e-shared") is not None, f"edge lost when {first} was removed first"
            g.remove_by_source(_ctx(), second, 2)
            assert g.resolve_edge(_ctx(), "e-shared") is None, "edge survives its last supporting source"

    def unknown_source_removal_fences_later_apply() -> None:
        g = factory()
        try:
            g.remove_by_source(_ctx(), "src-never", 5)
        except PortFailure as failure:
            assert failure.code is InternalErrorCode.MISSING_RESOURCE, "unexpected failure code"
        _expect_failure(
            InternalErrorCode.STALE_REVISION,
            lambda: g.apply_episode(_ctx(), episode("ep-late", "src-never", 4)),
        )
        assert g.resolve_edge(_ctx(), "e1") is None, "a delayed older apply resurrected a removed source"
        g.apply_episode(_ctx(), episode("ep-new", "src-never", 6))
        assert g.resolve_edge(_ctx(), "e1") is not None, "a newer revision must be accepted"

    def repeated_removal_keeps_the_fence() -> None:
        g = factory()
        g.apply_episode(_ctx(), episode("ep-1", "src-1", 1))
        g.remove_by_source(_ctx(), "src-1", 5)
        # A repeat at the fence revision or later is idempotent: the source is already removed.
        g.remove_by_source(_ctx(), "src-1", 5)
        g.remove_by_source(_ctx(), "src-1", 7)
        # A removal older than the fence is stale information and never lowers it.
        _expect_failure(
            InternalErrorCode.STALE_REVISION,
            lambda: g.remove_by_source(_ctx(), "src-1", 3),
        )
        _expect_failure(
            InternalErrorCode.STALE_REVISION,
            lambda: g.apply_episode(_ctx(), episode("ep-1", "src-1", 6)),
        )
        g.apply_episode(_ctx(), episode("ep-1", "src-1", 8))
        assert g.resolve_edge(_ctx(), "e1") is not None, "a newer revision must be accepted"

    def concurrent_applies_keep_all_support() -> None:
        for _ in range(5):
            g = factory()
            _parallel(
                lambda: g.apply_episode(_ctx(), episode("ep-a", "src-a", 1, edge="e-shared")),
                lambda: g.apply_episode(_ctx(), episode("ep-b", "src-b", 1, edge="e-shared")),
            )
            for source in ("src-a", "src-b"):
                only = TenantFilter(tenant_id="tenant-a", authorized_record_ids=(source,))
                assert [e.edge_id for e in g.neighbors(_ctx(), "n1", only)] == ["e-shared"], (
                    f"a concurrent apply lost the support of {source}"
                )
            g.remove_by_source(_ctx(), "src-a", 2)
            assert g.resolve_edge(_ctx(), "e-shared") is not None, "removal deleted an edge another source supports"

    def concurrent_apply_and_removal_respect_the_fence() -> None:
        for _ in range(5):
            g = factory()
            g.apply_episode(_ctx(), episode("ep-0", "src-1", 1, edge="e-old"))
            def delayed_removal() -> object:
                time.sleep(0.01)  # the apply is already in flight
                return g.remove_by_source(_ctx(), "src-1", 5)

            results = _parallel(
                delayed_removal,
                lambda: g.apply_episode(_ctx(), episode("ep-1", "src-1", 3, edge="e-new")),
            )
            for outcome in results:
                if isinstance(outcome, Exception):
                    assert isinstance(outcome, PortFailure), f"unexpected {type(outcome).__name__}"
            assert g.resolve_edge(_ctx(), "e-new") is None, "an apply older than the removal survived it"
            assert g.resolve_edge(_ctx(), "e-old") is None, "the removed source's edge survived"

    return _run_checks(
        [
            ("tenant_isolated", tenant_isolated),
            ("removal_is_source_supported", removal_is_source_supported),
            ("stale_episode_rejected", stale_episode_rejected),
            ("receipt_exact", receipt_exact),
            ("context_tenant_mismatch_denied", context_tenant_mismatch_denied),
            ("removal_scoped_to_tenant", removal_scoped_to_tenant),
            ("no_resurrection_after_source_removal", no_resurrection_after_source_removal),
            ("neighbors_filter_tenant_mismatch_denied", neighbors_filter_tenant_mismatch_denied),
            ("neighbors_respect_authorized_sources", neighbors_respect_authorized_sources),
            ("removal_order_independent", removal_order_independent),
            ("unknown_source_removal_fences_later_apply", unknown_source_removal_fences_later_apply),
            ("repeated_removal_keeps_the_fence", repeated_removal_keeps_the_fence),
            ("concurrent_applies_keep_all_support", concurrent_applies_keep_all_support),
            ("concurrent_apply_and_removal_respect_the_fence", concurrent_apply_and_removal_respect_the_fence),
        ]
    )


def _decision_request(text: str, candidates: tuple[str, ...]) -> DecisionRequest:
    version = SourceVersion(record_id="src-1", content_revision=1, policy_epoch=1)
    length = len(text.encode("utf-8"))
    return DecisionRequest(
        schema_version=CONTRACT_VERSION,
        request_id="decision-1",
        operation="select",
        sources=(SourceEvidence(source_version=version, text=text),),
        spans=(SourceSpan(span_id="span-1", source_version=version, start_utf8_byte=0, end_utf8_byte=length),),
        candidates=tuple(DecisionCandidate(candidate_id=c) for c in candidates),
        processing_policy=ReferenceHandle(kind="artifact", id="policy-1", tenant_id="tenant-a", resolver="policy"),
        resource_limits=InferenceResourceLimits(
            max_total_input_utf8_bytes=10_000, max_total_output_utf8_bytes=10_000, max_attempts=3
        ),
        deadline_ms=1000,
        head=DecisionHead(head_id="head-1", span_ids=("span-1",), candidate_ids=candidates, required=True),
    )


def run_decision_conformance(
    factory: Callable[[], DecisionPort],
) -> tuple[ConformanceFailure, ...]:
    def selects_supplied_candidates() -> None:
        request = _decision_request("the alpha team", ("alpha", "beta"))
        result = factory().decide(_ctx(), request)
        validate_decision_result(request, result)  # identities, candidate membership
        head = result.head_result
        assert head.outcome is DecisionOutcome.SELECTED and head.selected_candidate_ids == ("alpha",), "expected the supplied matching candidate"

    def valid_empty_is_not_abstain() -> None:
        request = _decision_request("nothing relevant", ("alpha", "beta"))
        result = factory().decide(_ctx(), request)
        validate_decision_result(request, result)
        assert result.head_result.outcome is DecisionOutcome.EMPTY, "no match must be a valid-empty result"
        assert result.head_result.reason_code is None, "valid-empty must not carry a failure reason"

    def inability_is_abstain_with_reason() -> None:
        request = _decision_request("   ", ("alpha",))
        result = factory().decide(_ctx(), request)
        validate_decision_result(request, result)
        assert result.head_result.outcome is DecisionOutcome.ABSTAINED, "unusable input must abstain, not return empty"
        assert result.head_result.reason_code, "abstention requires a reason code"

    return _run_checks(
        [
            ("selects_supplied_candidates", selects_supplied_candidates),
            ("valid_empty_is_not_abstain", valid_empty_is_not_abstain),
            ("inability_is_abstain_with_reason", inability_is_abstain_with_reason),
        ]
    )


_GENERATION_SCHEMA = {
    "$schema": JSON_SCHEMA_DIALECT,
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


def _generation_request(text: str) -> GenerationRequest:
    version = SourceVersion(record_id="src-1", content_revision=1, policy_epoch=1)
    length = len(text.encode("utf-8"))
    return GenerationRequest(
        schema_version=CONTRACT_VERSION,
        request_id="generation-1",
        operation="summarize",
        sources=(SourceEvidence(source_version=version, text=text),),
        spans=(SourceSpan(span_id="span-1", source_version=version, start_utf8_byte=0, end_utf8_byte=length),),
        processing_policy=ReferenceHandle(kind="artifact", id="policy-1", tenant_id="tenant-a", resolver="policy"),
        output_schema_dialect=JSON_SCHEMA_DIALECT,
        output_schema=_GENERATION_SCHEMA,
        resource_limits=InferenceResourceLimits(
            max_total_input_utf8_bytes=10_000, max_total_output_utf8_bytes=10_000, max_attempts=3
        ),
        deadline_ms=1000,
    )


def run_generation_conformance(
    factory: Callable[[], GenerationPort],
) -> tuple[ConformanceFailure, ...]:
    def proposes_schema_valid_supported_output() -> None:
        request = _generation_request("prefers dark mode")
        result = factory().generate(_ctx(), request)
        validate_generation_result(request, result)
        assert result.outcome is GenerationOutcome.PROPOSED and result.proposals, "expected a proposal"

    def valid_empty_is_not_abstain() -> None:
        request = _generation_request("# only a comment")
        result = factory().generate(_ctx(), request)
        validate_generation_result(request, result)
        assert result.outcome is GenerationOutcome.EMPTY, "nothing to propose must be valid-empty"

    def inability_is_abstain_with_reason() -> None:
        request = _generation_request("   ")
        result = factory().generate(_ctx(), request)
        validate_generation_result(request, result)
        assert result.outcome is GenerationOutcome.ABSTAINED and result.reason_code, "unusable input must abstain with a reason"

    return _run_checks(
        [
            ("proposes_schema_valid_supported_output", proposes_schema_valid_supported_output),
            ("valid_empty_is_not_abstain", valid_empty_is_not_abstain),
            ("inability_is_abstain_with_reason", inability_is_abstain_with_reason),
        ]
    )


__all__ = [
    "ActivationReceipt",
    "ActivationRejected",
    "AdapterRegistry",
    "ConformanceFailure",
    "DecisionPort",
    "EngineRecord",
    "GenerationPort",
    "GraphEdge",
    "GraphEpisode",
    "GraphStorePort",
    "MemoryEnginePort",
    "OPERATIONS_BY_KIND",
    "OperationContext",
    "PortFailure",
    "ProjectionReceipt",
    "RoutedPort",
    "RoutingGenerationFence",
    "ReferenceGraphStore",
    "ReferenceInference",
    "ReferenceMemoryEngine",
    "ReferenceVectorStore",
    "SearchHit",
    "TenantFilter",
    "VectorStorePort",
    "reference_manifest",
    "run_decision_conformance",
    "run_generation_conformance",
    "run_graph_store_conformance",
    "run_memory_engine_conformance",
    "run_vector_store_conformance",
    "validate_activation",
]
