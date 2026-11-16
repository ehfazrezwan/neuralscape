"""Reference fakes pass the conformance suite; deliberately broken ones fail it."""

import ast
from pathlib import Path

import pytest

import engine_ports
from contracts_engines import (
    CapabilityHealth,
    CapabilityRequirement,
    EngineAdapterKind,
)
from contracts_errors import InternalErrorCode, PublicErrorCode
from contracts_inference import (
    DecisionHeadResult,
    DecisionOutcome,
    DecisionResult,
    GenerationOutcome,
    GenerationProposal,
    GenerationResult,
)
from contracts_references import ReferenceHandle
from engine_ports import (
    ActivationRejected,
    AdapterRegistry,
    EngineRecord,
    GraphEpisode,
    OperationContext,
    PortFailure,
    ReferenceGraphStore,
    ReferenceInference,
    ReferenceMemoryEngine,
    ReferenceVectorStore,
    TenantFilter,
    reference_manifest,
    run_decision_conformance,
    run_generation_conformance,
    run_graph_store_conformance,
    run_memory_engine_conformance,
    run_vector_store_conformance,
)


def failed(failures):
    return {failure.check for failure in failures}


# ---- broken record-store fakes (negative controls) -------------------------


class NoTenantFilterVector(ReferenceVectorStore):
    """Searches and reads across tenants."""

    def get(self, ctx, record_id):
        for (_tenant, rid), (record, removed) in self._rows.items():
            if rid == record_id and not removed:
                return record
        return None

    def search(self, ctx, query, flt, limit):
        wide = flt.model_copy(update={"tenant_id": "ignored"})
        hits = []
        for (_t, rid), (record, removed) in self._rows.items():
            if removed:
                continue
            score = len(set(query.lower().split()) & set(record.text.lower().split()))
            if score:
                hits.append(engine_ports.SearchHit(record_id=rid, revision=record.revision, score=float(score)))
        hits.sort(key=lambda h: (-h.score, h.record_id))
        return tuple(hits[:limit])


class IgnoresDeleteVector(ReferenceVectorStore):
    def remove(self, ctx, record_id, revision):
        return self._receipt(record_id, revision, "removed")


class ChangesIdVector(ReferenceVectorStore):
    def apply(self, ctx, record):
        return super().apply(ctx, record.model_copy(update={"record_id": "engine-" + record.record_id}))


class DistractorVector(ReferenceVectorStore):
    """Ignores the authorization pushdown."""

    def search(self, ctx, query, flt, limit):
        return super().search(ctx, query, flt.model_copy(update={"authorized_record_ids": None}), limit)


class PostFilterVector(ReferenceVectorStore):
    """Limits first, then filters: drops authorized hits behind a distractor."""

    def search(self, ctx, query, flt, limit):
        top = super().search(ctx, query, flt.model_copy(update={"authorized_record_ids": None}), limit)
        allowed = None if flt.authorized_record_ids is None else set(flt.authorized_record_ids)
        return tuple(h for h in top if allowed is None or h.record_id in allowed)


class StaleApplyVector(ReferenceVectorStore):
    """Last write wins, even for an older revision."""

    def apply(self, ctx, record):
        self._rows[(record.tenant_id, record.record_id)] = (record, False)
        return self._receipt(record.record_id, record.revision, "applied")


class ConflictBlindVector(ReferenceVectorStore):
    """Accepts a different payload at the same revision."""

    def apply(self, ctx, record):
        self._rows[(record.tenant_id, record.record_id)] = (record, False)
        return self._receipt(record.record_id, record.revision, "applied")


class WrongReceiptVector(ReferenceVectorStore):
    def apply(self, ctx, record):
        super().apply(ctx, record)
        return self._receipt(record.record_id, record.revision + 1, "applied")


class StalePostingsVector(ReferenceVectorStore):
    """Keeps every superseded revision's postings searchable."""

    def __init__(self):
        super().__init__()
        self._history = []

    def apply(self, ctx, record):
        receipt = super().apply(ctx, record)
        self._history.append(record)
        return receipt

    def search(self, ctx, query, flt, limit):
        hits = list(super().search(ctx, query, flt, limit))
        words = set(query.lower().split())
        for old in self._history:
            if old.tenant_id == flt.tenant_id and words & set(old.text.lower().split()):
                hits.append(engine_ports.SearchHit(record_id=old.record_id, revision=old.revision, score=1.0))
        return tuple(hits[:limit])


class ResurrectingVector(ReferenceVectorStore):
    """Forgets a removed record entirely, including its revision fence."""

    def remove(self, ctx, record_id, revision):
        receipt = super().remove(ctx, record_id, revision)
        del self._rows[(ctx.tenant_id, record_id)]
        return receipt


class StaleRemoveVector(ReferenceVectorStore):
    """Honors a remove whatever its revision."""

    def remove(self, ctx, record_id, revision):
        stored = self._rows[(ctx.tenant_id, record_id)][0]
        self._rows[(ctx.tenant_id, record_id)] = (
            stored.model_copy(update={"revision": max(revision, stored.revision)}),
            True,
        )
        return self._receipt(record_id, revision, "removed")


class WrongRemoveReceiptVector(ReferenceVectorStore):
    def remove(self, ctx, record_id, revision):
        super().remove(ctx, record_id, revision)
        return self._receipt(record_id, revision, "applied")


class LenientFilterTenantVector(ReferenceVectorStore):
    """Serves the filter's tenant instead of denying a mismatched context."""

    def search(self, ctx, query, flt, limit):
        return super().search(ctx.model_copy(update={"tenant_id": flt.tenant_id}), query, flt, limit)


class IdOnlyKeyVector(ReferenceVectorStore):
    """Keys records by id alone, so tenants collide."""

    def apply(self, ctx, record):
        key = ("*", record.record_id)
        self._rows[key] = (record, False)
        return self._receipt(record.record_id, record.revision, "applied")

    def get(self, ctx, record_id):
        row = self._rows.get(("*", record_id))
        return None if row is None or row[1] or row[0].tenant_id != ctx.tenant_id else row[0]

    def remove(self, ctx, record_id, revision):
        row = self._rows.get(("*", record_id))
        if row is not None:
            self._rows[("*", record_id)] = (row[0], True)
        return self._receipt(record_id, revision, "removed")

    def search(self, ctx, query, flt, limit):
        hits = []
        for (_t, rid), (record, removed) in self._rows.items():
            if not removed and record.tenant_id == flt.tenant_id and set(query.lower().split()) & set(record.text.lower().split()):
                hits.append(engine_ports.SearchHit(record_id=rid, revision=record.revision, score=1.0))
        return tuple(hits[:limit])


class IgnoresLabelsVector(ReferenceVectorStore):
    def search(self, ctx, query, flt, limit):
        return super().search(ctx, query, flt.model_copy(update={"labels_all": ()}), limit)


class IgnoresLimitVector(ReferenceVectorStore):
    def search(self, ctx, query, flt, limit):
        return super().search(ctx, query, flt, 1000)


# ---- vector store / memory engine ------------------------------------------


def test_reference_vector_store_passes_conformance():
    assert run_vector_store_conformance(ReferenceVectorStore) == ()


def test_reference_memory_engine_passes_conformance():
    assert run_memory_engine_conformance(ReferenceMemoryEngine) == ()


@pytest.mark.parametrize(
    ("fake", "expected"),
    [
        (NoTenantFilterVector, {"tenant_filter_enforced"}),
        (IgnoresDeleteVector, {"delete_is_exact"}),
        (ChangesIdVector, {"ids_and_receipts_exact"}),
        (DistractorVector, {"authorized_ids_applied_before_limit"}),
        (PostFilterVector, {"authorized_ids_applied_before_limit"}),
        (StaleApplyVector, {"stale_revision_rejected"}),
        (ConflictBlindVector, {"same_revision_conflict_detected"}),
        (WrongReceiptVector, {"ids_and_receipts_exact"}),
        (StalePostingsVector, {"update_replaces_postings"}),
        (ResurrectingVector, {"no_resurrection_after_remove"}),
        (StaleRemoveVector, {"stale_remove_refused"}),
        (WrongRemoveReceiptVector, {"remove_receipt_exact"}),
        (LenientFilterTenantVector, {"search_filter_tenant_mismatch_denied"}),
        (IdOnlyKeyVector, {"same_id_two_tenants_isolated"}),
        (IgnoresLabelsVector, {"labels_honored"}),
        (IgnoresLimitVector, {"limit_honored"}),
    ],
)
def test_broken_vector_stores_fail_the_expected_check(fake, expected):
    assert expected <= failed(run_vector_store_conformance(fake))


def test_every_record_check_has_a_failing_negative_control():
    names = {name for name, _ in engine_ports._record_checks(ReferenceVectorStore, ("apply", "remove", "get", "search"))}
    covered = failed_all_vector() | failed(
        run_vector_store_conformance(TrustsRecordTenantVector)
    )
    assert names <= covered, sorted(names - covered)


class TrustsRecordTenantVector(ReferenceVectorStore):
    def apply(self, ctx, record):
        return super().apply(ctx.model_copy(update={"tenant_id": record.tenant_id}), record)


def failed_all_vector():
    collected = set()
    for fake in (
        NoTenantFilterVector,
        IgnoresDeleteVector,
        ChangesIdVector,
        DistractorVector,
        StaleApplyVector,
        ConflictBlindVector,
        StalePostingsVector,
        ResurrectingVector,
        StaleRemoveVector,
        WrongRemoveReceiptVector,
        LenientFilterTenantVector,
        IdOnlyKeyVector,
        IgnoresLabelsVector,
        IgnoresLimitVector,
    ):
        collected |= failed(run_vector_store_conformance(fake))
    return collected


def test_context_tenant_mismatch_control_fails():
    class TrustsRecordTenant(ReferenceVectorStore):
        def apply(self, ctx, record):
            return super().apply(ctx.model_copy(update={"tenant_id": record.tenant_id}), record)

    assert "context_tenant_mismatch_denied" in failed(
        run_vector_store_conformance(TrustsRecordTenant)
    )


def test_broken_memory_engine_changing_ids_fails():
    class ChangesIdEngine(ReferenceMemoryEngine):
        def ingest(self, ctx, record):
            return super().ingest(ctx, record.model_copy(update={"record_id": "e-" + record.record_id}))

    assert "ids_and_receipts_exact" in failed(run_memory_engine_conformance(ChangesIdEngine))


def test_broken_memory_engine_ignoring_delete_fails():
    class KeepsDeleted(ReferenceMemoryEngine):
        def delete(self, ctx, record_id, revision):
            return self._receipt(record_id, revision, "removed")

    assert "delete_is_exact" in failed(run_memory_engine_conformance(KeepsDeleted))


# ---- graph store -------------------------------------------------------------


class CrossTenantGraph(ReferenceGraphStore):
    def resolve_edge(self, ctx, edge_id):
        for episode, removed in self._episodes.values():
            if not removed:
                for edge in episode.edges:
                    if edge.edge_id == edge_id:
                        return edge
        return None


class SupportBlindGraph(ReferenceGraphStore):
    """Removes an edge for every source once any source is removed."""

    def remove_by_source(self, ctx, source_record_id, revision):
        for key, (episode, removed) in list(self._episodes.items()):
            self._episodes[key] = (episode, True)
        return self._receipt(source_record_id, revision, "removed")


class IgnoresDeleteGraph(ReferenceGraphStore):
    def remove_by_source(self, ctx, source_record_id, revision):
        return self._receipt(source_record_id, revision, "removed")


class StaleEpisodeGraph(ReferenceGraphStore):
    def apply_episode(self, ctx, episode):
        self._episodes[(episode.tenant_id, episode.episode_id)] = (episode, False)
        return self._receipt(episode.episode_id, episode.revision, "applied")


class ResurrectGraph(ReferenceGraphStore):
    """Forgets the removal fence, so an older episode returns."""

    def remove_by_source(self, ctx, source_record_id, revision):
        receipt = super().remove_by_source(ctx, source_record_id, revision)
        self._fences.clear()
        return receipt


class CrossTenantRemoveGraph(ReferenceGraphStore):
    """Removes every tenant's episodes that share the source id."""

    def remove_by_source(self, ctx, source_record_id, revision):
        for key, (episode, removed) in list(self._episodes.items()):
            if episode.source_record_id == source_record_id:
                self._episodes[key] = (episode, True)
        return self._receipt(source_record_id, revision, "removed")


class LenientNeighborsGraph(ReferenceGraphStore):
    def neighbors(self, ctx, node_id, flt):
        return super().neighbors(ctx.model_copy(update={"tenant_id": flt.tenant_id}), node_id, flt)


class UnauthorizedLeakGraph(ReferenceGraphStore):
    """Ignores authorized source ids."""

    def neighbors(self, ctx, node_id, flt):
        return super().neighbors(ctx, node_id, flt.model_copy(update={"authorized_record_ids": None}))


class AllSourcesMustBeAuthorizedGraph(ReferenceGraphStore):
    """Hides an edge unless every supporting source is authorized."""

    def neighbors(self, ctx, node_id, flt):
        edges = super().neighbors(ctx, node_id, flt.model_copy(update={"authorized_record_ids": None}))
        if flt.authorized_record_ids is None:
            return edges
        allowed = set(flt.authorized_record_ids)
        visible = []
        for edge in edges:
            sources = {
                ep.source_record_id
                for (tenant, _), (ep, removed) in self._episodes.items()
                if tenant == flt.tenant_id and not removed and any(e.edge_id == edge.edge_id for e in ep.edges)
            }
            if sources <= allowed:
                visible.append(edge)
        return tuple(visible)


class NoFenceOnUnknownGraph(ReferenceGraphStore):
    """Writes no fence when a removal finds nothing."""

    def remove_by_source(self, ctx, source_record_id, revision):
        key = (ctx.tenant_id, source_record_id)
        had = key in self._fences
        try:
            return super().remove_by_source(ctx, source_record_id, revision)
        except PortFailure:
            if not had:
                self._fences.pop(key, None)
            raise


class LoweringFenceGraph(ReferenceGraphStore):
    """A repeated removal at a lower revision lowers the fence."""

    def remove_by_source(self, ctx, source_record_id, revision):
        try:
            return super().remove_by_source(ctx, source_record_id, revision)
        finally:
            self._fences[(ctx.tenant_id, source_record_id)] = revision


class RaisesOnRepeatedRemovalGraph(ReferenceGraphStore):
    """A repeated removal of an already removed source reports ``missing_resource``."""

    def remove_by_source(self, ctx, source_record_id, revision):
        key = (ctx.tenant_id, source_record_id)
        if key in self._fences:
            self._fences[key] = max(self._fences[key], revision)
            raise PortFailure(InternalErrorCode.MISSING_RESOURCE)
        return super().remove_by_source(ctx, source_record_id, revision)


class OrderDependentGraph(ReferenceGraphStore):
    """Native-Graphiti-like: removing the first-created supporter drops the edge for all."""

    def remove_by_source(self, ctx, source_record_id, revision):
        with self._lock:
            order = {}
            for key, (episode, removed) in self._episodes.items():
                if key[0] == ctx.tenant_id and not removed:
                    for edge in episode.edges:
                        order.setdefault(edge.edge_id, episode.source_record_id)
            doomed = {e for e, first in order.items() if first == source_record_id}
            for key, (episode, removed) in list(self._episodes.items()):
                if key[0] == ctx.tenant_id and not removed:
                    kept = tuple(e for e in episode.edges if e.edge_id not in doomed)
                    self._episodes[key] = (episode.model_copy(update={"edges": kept}), removed)
            return super().remove_by_source(ctx, source_record_id, revision)


class LostUpdateGraph(ReferenceGraphStore):
    """Keeps an explicit support set per edge with a non-atomic read-modify-write."""

    def __init__(self):
        super().__init__()
        self._support = {}
        self._edges = {}

    def apply_episode(self, ctx, episode):
        _check_tenant_for_test(ctx, episode.tenant_id)
        for edge in episode.edges:
            key = (episode.tenant_id, edge.edge_id)
            current = set(self._support.get(key, ()))
            time.sleep(0.02)  # a concurrent writer lands here
            current.add(episode.source_record_id)
            self._support[key] = current
            self._edges[key] = edge
        return self._receipt(episode.episode_id, episode.revision, "applied")

    def remove_by_source(self, ctx, source_record_id, revision):
        for key, sources in self._support.items():
            if key[0] == ctx.tenant_id:
                sources.discard(source_record_id)
        return self._receipt(source_record_id, revision, "removed")

    def resolve_edge(self, ctx, edge_id):
        key = (ctx.tenant_id, edge_id)
        return self._edges[key] if self._support.get(key) else None

    def neighbors(self, ctx, node_id, flt):
        allowed = None if flt.authorized_record_ids is None else set(flt.authorized_record_ids)
        found = []
        for key, sources in self._support.items():
            if key[0] != flt.tenant_id or not sources:
                continue
            if allowed is not None and not (sources & allowed):
                continue
            edge = self._edges[key]
            if node_id in (edge.subject_id, edge.object_id):
                found.append(edge)
        return tuple(sorted(found, key=lambda e: e.edge_id))


class CheckThenWriteGraph(ReferenceGraphStore):
    """Checks the fence, then writes later without holding anything in between."""

    def apply_episode(self, ctx, episode):
        _check_tenant_for_test(ctx, episode.tenant_id)
        fence = self._fences.get((episode.tenant_id, episode.source_record_id))
        if fence is not None and episode.revision <= fence:
            raise PortFailure(InternalErrorCode.STALE_REVISION)
        time.sleep(0.05)  # a removal commits here
        self._episodes[(episode.tenant_id, episode.episode_id)] = (episode, False)
        return self._receipt(episode.episode_id, episode.revision, "applied")


def _check_tenant_for_test(ctx, tenant_id):
    if ctx.tenant_id != tenant_id:
        raise PortFailure(InternalErrorCode.DENIED)


def test_reference_graph_store_passes_conformance():
    assert run_graph_store_conformance(ReferenceGraphStore) == ()


@pytest.mark.parametrize(
    ("fake", "expected"),
    [
        (CrossTenantGraph, "tenant_isolated"),
        (SupportBlindGraph, "removal_is_source_supported"),
        (IgnoresDeleteGraph, "removal_is_source_supported"),
        (StaleEpisodeGraph, "stale_episode_rejected"),
        (ResurrectGraph, "no_resurrection_after_source_removal"),
        (CrossTenantRemoveGraph, "removal_scoped_to_tenant"),
        (LenientNeighborsGraph, "neighbors_filter_tenant_mismatch_denied"),
        (UnauthorizedLeakGraph, "neighbors_respect_authorized_sources"),
        (AllSourcesMustBeAuthorizedGraph, "neighbors_respect_authorized_sources"),
        (NoFenceOnUnknownGraph, "unknown_source_removal_fences_later_apply"),
        (LoweringFenceGraph, "repeated_removal_keeps_the_fence"),
        (OrderDependentGraph, "removal_order_independent"),
        (LostUpdateGraph, "concurrent_applies_keep_all_support"),
        (CheckThenWriteGraph, "concurrent_apply_and_removal_respect_the_fence"),
        (RaisesOnRepeatedRemovalGraph, "repeated_removal_keeps_the_fence"),
    ],
)
def test_broken_graph_stores_fail_the_expected_check(fake, expected):
    assert expected in failed(run_graph_store_conformance(fake))


# ---- inference ---------------------------------------------------------------


class ConflatingDecision(ReferenceInference):
    """Reports a valid-empty result as an abstention."""

    def decide(self, ctx, request):
        result = super().decide(ctx, request)
        if result.head_result.outcome is DecisionOutcome.EMPTY:
            head = DecisionHeadResult(
                head_id=result.head_result.head_id,
                outcome=DecisionOutcome.ABSTAINED,
                selected_candidate_ids=(),
                distribution=None,
                reason_code="no_match",
            )
            return result.model_copy(update={"head_result": head})
        return result


class EmptyOnInabilityDecision(ReferenceInference):
    """Reports inability to decide as valid-empty."""

    def decide(self, ctx, request):
        result = super().decide(ctx, request)
        if result.head_result.outcome is DecisionOutcome.ABSTAINED:
            head = DecisionHeadResult(
                head_id=result.head_result.head_id,
                outcome=DecisionOutcome.EMPTY,
                selected_candidate_ids=(),
                distribution=None,
                reason_code=None,
            )
            return result.model_copy(update={"head_result": head})
        return result


class ForeignCandidateDecision(ReferenceInference):
    def decide(self, ctx, request):
        result = super().decide(ctx, request)
        head = DecisionHeadResult(
            head_id=result.head_result.head_id,
            outcome=DecisionOutcome.SELECTED,
            selected_candidate_ids=("not-a-candidate",),
            distribution=None,
            reason_code=None,
        )
        return result.model_copy(update={"head_result": head})


class WrongRequestIdDecision(ReferenceInference):
    def decide(self, ctx, request):
        result = super().decide(ctx, request)
        return result.model_copy(update={"request_id": "other"})


class ConflatingGeneration(ReferenceInference):
    def generate(self, ctx, request):
        result = super().generate(ctx, request)
        if result.outcome is GenerationOutcome.EMPTY:
            return result.model_copy(
                update={"outcome": GenerationOutcome.ABSTAINED, "reason_code": "no_output"}
            )
        return result


class SchemaViolatingGeneration(ReferenceInference):
    def generate(self, ctx, request):
        result = super().generate(ctx, request)
        if result.outcome is GenerationOutcome.PROPOSED:
            bad = GenerationProposal(
                proposal_id="p-bad",
                output={"summary": 7},
                support_span_ids=("span-1",),
                machine_authored=True,
            )
            return result.model_copy(update={"proposals": (bad,)})
        return result


class UnsupportedSourceGeneration(ReferenceInference):
    def generate(self, ctx, request):
        result = super().generate(ctx, request)
        if result.outcome is GenerationOutcome.PROPOSED:
            bad = GenerationProposal(
                proposal_id="p-bad",
                output={"summary": "x"},
                support_span_ids=("span-unknown",),
                machine_authored=True,
            )
            return result.model_copy(update={"proposals": (bad,)})
        return result


def test_reference_inference_passes_both_conformance_suites():
    assert run_decision_conformance(ReferenceInference) == ()
    assert run_generation_conformance(ReferenceInference) == ()


@pytest.mark.parametrize(
    ("fake", "expected"),
    [
        (ConflatingDecision, "valid_empty_is_not_abstain"),
        (EmptyOnInabilityDecision, "inability_is_abstain_with_reason"),
        (ForeignCandidateDecision, "selects_supplied_candidates"),
        (WrongRequestIdDecision, "selects_supplied_candidates"),
    ],
)
def test_broken_decision_adapters_fail(fake, expected):
    assert expected in failed(run_decision_conformance(fake))


@pytest.mark.parametrize(
    ("fake", "expected"),
    [
        (ConflatingGeneration, "valid_empty_is_not_abstain"),
        (SchemaViolatingGeneration, "proposes_schema_valid_supported_output"),
        (UnsupportedSourceGeneration, "proposes_schema_valid_supported_output"),
    ],
)
def test_broken_generation_adapters_fail(fake, expected):
    assert expected in failed(run_generation_conformance(fake))


# ---- failure taxonomy --------------------------------------------------------


def test_port_failure_reuses_canonical_codes_and_hides_detail():
    failure = PortFailure(InternalErrorCode.STALE_REVISION, "secret-record-id")
    public = failure.public()
    assert public.code is PublicErrorCode.NOT_FOUND  # default non-disclosure
    assert "secret" not in public.message
    assert PortFailure(InternalErrorCode.UNAVAILABLE_DEPENDENCY).public().retryable is True


@pytest.mark.parametrize("bad", ["stale_revision", None, 3])
def test_port_failure_rejects_non_canonical_codes(bad):
    with pytest.raises(TypeError):
        PortFailure(bad)


# ---- registry and activation -------------------------------------------------


def requirements_for(kind, *, qualified=False):
    profile = {}
    if qualified:
        profile = {
            "qualification_profile_reference": ReferenceHandle(
                kind="artifact", id="profile", tenant_id="tenant-a", resolver="evidence"
            ),
            "qualification_profile_version": "p1",
        }
    return tuple(
        CapabilityRequirement(
            operation=name,
            contract_schema_version="candidate-v1",
            require_configured=True,
            require_healthy=True,
            qualification_profile_reference=profile.get("qualification_profile_reference"),
            qualification_profile_version=profile.get("qualification_profile_version"),
        )
        for name in engine_ports.OPERATIONS_BY_KIND[kind]
    )


class SpyVector(ReferenceVectorStore):
    def __init__(self, *, health=CapabilityHealth.HEALTHY, **kwargs):
        super().__init__(**kwargs)
        self.health = health
        self.data_calls = 0

    def manifest(self):
        return reference_manifest(self._implementation_id, self._kind, health=self.health)

    def apply(self, ctx, record):
        self.data_calls += 1
        return super().apply(ctx, record)

    def search(self, ctx, query, flt, limit):
        self.data_calls += 1
        return super().search(ctx, query, flt, limit)


def test_activation_succeeds_and_routes_a_healthy_adapter():
    registry = AdapterRegistry()
    adapter_id = registry.register(SpyVector(), EngineAdapterKind.VECTOR_STORE)
    receipt = registry.activate(adapter_id, requirements_for(EngineAdapterKind.VECTOR_STORE))
    assert receipt.routing_generation == 1
    port = registry.active_port(EngineAdapterKind.VECTOR_STORE)
    assert port.apply(
        OperationContext(tenant_id="tenant-a", correlation_id="c", deadline_ms=5),
        EngineRecord(record_id="r", revision=1, tenant_id="tenant-a", text="t"),
    ).object_id == "r"


def test_activation_rejects_unhealthy_adapter_before_any_data_call():
    port = SpyVector(health=CapabilityHealth.UNHEALTHY)
    registry = AdapterRegistry()
    adapter_id = registry.register(port, EngineAdapterKind.VECTOR_STORE)
    with pytest.raises(ActivationRejected) as raised:
        registry.activate(adapter_id, requirements_for(EngineAdapterKind.VECTOR_STORE))
    assert {violation.fact for violation in raised.value.violations} == {"healthy"}
    assert port.data_calls == 0
    with pytest.raises(PortFailure) as unavailable:
        registry.active_port(EngineAdapterKind.VECTOR_STORE)
    assert unavailable.value.code is InternalErrorCode.UNAVAILABLE_DEPENDENCY
    assert registry.routing_generation == 0


def test_reference_adapters_are_not_reported_qualified():
    registry = AdapterRegistry()
    adapter_id = registry.register(SpyVector(), EngineAdapterKind.VECTOR_STORE)
    with pytest.raises(ActivationRejected) as raised:
        registry.activate(
            adapter_id, requirements_for(EngineAdapterKind.VECTOR_STORE, qualified=True)
        )
    assert "qualified" in {violation.fact for violation in raised.value.violations}


def test_registration_rejects_wrong_kind_missing_methods_and_duplicates():
    registry = AdapterRegistry()
    with pytest.raises(ActivationRejected):
        registry.register(object(), EngineAdapterKind.VECTOR_STORE)
    with pytest.raises(ActivationRejected):
        registry.register(ReferenceVectorStore(), EngineAdapterKind.GRAPH_STORE)
    registry.register(ReferenceVectorStore("dup"), EngineAdapterKind.VECTOR_STORE)
    with pytest.raises(ActivationRejected):
        registry.register(ReferenceVectorStore("dup"), EngineAdapterKind.VECTOR_STORE)


def test_activation_rejects_manifest_kind_mismatch_and_missing_operations():
    class WrongKindManifest(ReferenceVectorStore):
        def manifest(self):
            return reference_manifest(self._implementation_id, EngineAdapterKind.GRAPH_STORE)

    class MissingOperation(ReferenceVectorStore):
        def manifest(self):
            full = super().manifest()
            return full.model_copy(update={"operations": full.operations[:-1]})

    registry = AdapterRegistry()
    wrong = registry.register(WrongKindManifest("wrong"), EngineAdapterKind.VECTOR_STORE)
    with pytest.raises(ActivationRejected, match="kind"):
        registry.activate(wrong, ())
    missing = registry.register(MissingOperation("missing"), EngineAdapterKind.VECTOR_STORE)
    with pytest.raises(ActivationRejected, match="omits"):
        registry.activate(missing, ())


def test_activation_rejects_requirement_outside_the_port():
    registry = AdapterRegistry()
    adapter_id = registry.register(ReferenceVectorStore(), EngineAdapterKind.VECTOR_STORE)
    bogus = (
        CapabilityRequirement(
            operation="teleport",
            contract_schema_version="candidate-v1",
            require_configured=False,
            require_healthy=False,
            qualification_profile_reference=None,
            qualification_profile_version=None,
        ),
    )
    with pytest.raises(ActivationRejected, match="outside"):
        registry.activate(adapter_id, bogus)


def test_inference_adapter_requires_both_ports_and_reactivation_bumps_generation():
    registry = AdapterRegistry()
    with pytest.raises(ActivationRejected):
        registry.register(ReferenceVectorStore(), EngineAdapterKind.INFERENCE)
    adapter_id = registry.register(ReferenceInference(), EngineAdapterKind.INFERENCE)
    first = registry.activate(adapter_id, requirements_for(EngineAdapterKind.INFERENCE))
    registry.deactivate(EngineAdapterKind.INFERENCE)
    with pytest.raises(PortFailure):
        registry.active_port(EngineAdapterKind.INFERENCE)
    second = registry.activate(adapter_id, requirements_for(EngineAdapterKind.INFERENCE))
    assert second.routing_generation > first.routing_generation


def test_unknown_manifest_object_is_rejected():
    class ForgedManifest(ReferenceVectorStore):
        def manifest(self):
            return {"implementation_id": "forged"}

    with pytest.raises(ActivationRejected):
        AdapterRegistry().register(ForgedManifest(), EngineAdapterKind.VECTOR_STORE)


# ---- scope guards ------------------------------------------------------------


def test_filter_requires_tenant():
    with pytest.raises(Exception):
        TenantFilter()  # type: ignore[call-arg]


def test_module_has_no_runtime_wiring_imports():
    source = Path(engine_ports.__file__).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"memory", "memory_service", "mem0", "graphiti_core", "qdrant_client", "worker", "main", "knowledge", "adapters"}
    assert not (imported & forbidden)


def test_graph_episode_is_single_source_value():
    episode = GraphEpisode(
        episode_id="e", revision=1, tenant_id="t", source_record_id="s", edges=()
    )
    assert episode.source_record_id == "s"


def test_raising_manifest_is_normalized_to_activation_rejected():
    class Exploding(ReferenceVectorStore):
        def manifest(self):
            raise RuntimeError("backend exploded")

    with pytest.raises(ActivationRejected, match="could not be read"):
        AdapterRegistry().register(Exploding(), EngineAdapterKind.VECTOR_STORE)
    with pytest.raises(ActivationRejected, match="could not be read"):
        engine_ports.validate_activation(Exploding(), EngineAdapterKind.VECTOR_STORE, ())

    class ExplodesAfterRegistration(ReferenceVectorStore):
        broken = False

        def manifest(self):
            if self.broken:
                raise RuntimeError("went away")
            return super().manifest()

    port = ExplodesAfterRegistration("late")
    registry = AdapterRegistry()
    adapter_id = registry.register(port, EngineAdapterKind.VECTOR_STORE)
    port.broken = True
    with pytest.raises(ActivationRejected, match="could not be read"):
        registry.activate(adapter_id, ())
    assert registry.routing_generation == 0


def test_every_graph_check_has_a_failing_negative_control(monkeypatch):
    names = []
    monkeypatch.setattr(
        engine_ports, "_run_checks", lambda checks: names.extend(n for n, _ in checks) or ()
    )
    run_graph_store_conformance(ReferenceGraphStore)
    monkeypatch.undo()
    controls = (
        CrossTenantGraph, SupportBlindGraph, IgnoresDeleteGraph, StaleEpisodeGraph, ResurrectGraph,
        CrossTenantRemoveGraph, LenientNeighborsGraph, UnauthorizedLeakGraph,
        AllSourcesMustBeAuthorizedGraph, NoFenceOnUnknownGraph, LoweringFenceGraph,
        OrderDependentGraph, LostUpdateGraph, CheckThenWriteGraph, RaisesOnRepeatedRemovalGraph,
    )
    covered = set()
    for fake in controls:
        covered |= failed(run_graph_store_conformance(fake))
    # receipt/context checks are covered by dedicated inline fakes elsewhere
    missing = set(names) - covered - {"receipt_exact", "context_tenant_mismatch_denied"}
    assert not missing, sorted(missing)


def test_a_graph_that_raises_on_a_repeated_removal_fails_the_contract():
    assert "repeated_removal_keeps_the_fence" in failed(run_graph_store_conformance(RaisesOnRepeatedRemovalGraph))


def test_the_reference_graph_repeats_a_removal_idempotently_and_refuses_an_older_one():
    from engine_ports import GraphEdge, OperationContext

    ctx = OperationContext(tenant_id="tenant-a", correlation_id="c", deadline_ms=5)
    g = ReferenceGraphStore()
    g.apply_episode(ctx, GraphEpisode(
        episode_id="ep-1", revision=1, tenant_id="tenant-a", source_record_id="src-1",
        edges=(GraphEdge(edge_id="e1", subject_id="n1", predicate="likes", object_id="n2"),),
    ))
    assert g.remove_by_source(ctx, "src-1", 5).operation == "removed"
    assert g.remove_by_source(ctx, "src-1", 5).operation == "removed"
    assert g.remove_by_source(ctx, "src-1", 7).operation == "removed"
    with pytest.raises(PortFailure) as older:
        g.remove_by_source(ctx, "src-1", 3)
    assert older.value.code is InternalErrorCode.STALE_REVISION
