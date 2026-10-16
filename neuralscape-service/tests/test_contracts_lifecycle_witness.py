"""Application witness, clock rule, cross-tenant lineage and public-first pins.

Every test imports the new names lazily so each fails individually on a tree without the change.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from pydantic import BaseModel, ValidationError

import contracts_lifecycle as lc
from contracts_lifecycle import (
    Intent,
    IntentStatus,
    ProcessingStage,
    StageReceipt,
    StageStatus,
    validate_required_stage_claim,
)
from contracts_references import ReferenceHandle
from tests.test_contracts_lifecycle import (
    BASE_MODEL_DICT_DESCRIPTOR,
    NOW,
    DivergentTuple,
    NestedRepairingTuple,
    install_repairing_target_refs,
    intent,
    intent_with_requirements,
    memory_record,
    receipt,
    reference,
    requirement,
    source,
)

CANONICAL, GRAPH = ProcessingStage.CANONICAL, ProcessingStage.GRAPH


def with_witness(base: StageReceipt, *applications: tuple[ProcessingStage, int]) -> StageReceipt:
    values = base.model_dump()
    values["upstream_applications"] = tuple(
        lc.UpstreamApplication(intent_id="intent-1", stage=stage, attempt=attempt, effect_id=f"effect-{stage.value}")
        for stage, attempt in applications
    )
    return StageReceipt.model_validate(values)


def chain_intent():
    """canonical (commits memory-1 r4) -> graph (reads exactly what canonical committed)."""

    committed = (source("memory-1", revision=4),)
    return intent_with_requirements(
        requirement(CANONICAL, committed_sources=committed),
        requirement(GRAPH, precondition_source_ids=(), upstream_stages=(CANONICAL,)),
    ), committed


def chain_receipts(committed, *, graph_witness=((CANONICAL, 1),)):
    canonical = receipt(CANONICAL, StageStatus.APPLIED, committed=committed)
    graph = with_witness(receipt(GRAPH, StageStatus.APPLIED, sources=committed), *graph_witness)
    return canonical, graph


def claim(command, receipts, status=IntentStatus.APPLIED):
    # a tuple (including a subclass) is passed through untouched: copying it would read its public view
    receipts = receipts if isinstance(receipts, tuple) else tuple(receipts)
    validate_required_stage_claim(claimed_status=status, intent=command, receipts=receipts)


# ------------------------------------------------ item 12: empty source equality is not a witness
def test_an_applied_claim_with_an_empty_application_witness_is_rejected() -> None:
    """No preconditions + ``applied_sources=()`` used to pass."""

    command = intent_with_requirements(
        requirement(CANONICAL, precondition_source_ids=()), source_preconditions=()
    )
    empty = receipt(CANONICAL, StageStatus.APPLIED, sources=())
    with pytest.raises(ValueError, match="nonempty source witness"):
        claim(command, [empty])


def test_a_requirement_may_declare_that_it_has_no_source_witness() -> None:
    declared = lc.StageEvidenceRequirement(
        stage=CANONICAL, effect_id="effect-canonical", precondition_source_ids=(), upstream_stages=(),
        committed_sources=(), minimum_output_refs=0, no_source_witness=True,
    )
    command = intent_with_requirements(declared, source_preconditions=())
    claim(command, [receipt(CANONICAL, StageStatus.APPLIED, sources=())])  # explicit, not accidental


def test_a_nonempty_witness_still_has_to_match_the_calculated_inputs() -> None:
    command = intent(CANONICAL)
    wrong = receipt(CANONICAL, StageStatus.APPLIED, sources=(source("memory-1", revision=2),))
    with pytest.raises(ValueError, match="must match calculated inputs"):
        claim(command, [wrong])


# ------------------------------------- item 19: durable immutable upstream application references
def test_a_downstream_application_must_cite_the_upstream_application_it_consumed() -> None:
    """matching committed-source *sets* were the only evidence."""

    command, committed = chain_intent()
    canonical, graph = chain_receipts(committed)
    claim(command, [canonical, graph])  # cited: accepted

    uncited = receipt(GRAPH, StageStatus.APPLIED, sources=committed)  # same sets, no witness
    with pytest.raises(ValueError, match="upstream application"):
        claim(command, [canonical, uncited])


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(((CANONICAL, 2),), id="attempt-that-never-applied"),
        pytest.param(((GRAPH, 1),), id="not-the-upstream-stage"),
        pytest.param(((CANONICAL, 1), (CANONICAL, 1)), id="duplicate-witness"),
        pytest.param((), id="no-witness"),
    ],
)
def test_a_witness_must_name_exactly_one_real_applied_upstream_receipt(bad) -> None:
    command, committed = chain_intent()
    canonical = receipt(CANONICAL, StageStatus.APPLIED, committed=committed)
    graph = receipt(GRAPH, StageStatus.APPLIED, sources=committed)
    with pytest.raises((ValueError, ValidationError)):
        claim(command, [canonical, with_witness(graph, *bad)])


def test_a_witness_cannot_cite_an_upstream_receipt_that_did_not_apply() -> None:
    command, committed = chain_intent()
    failed = receipt(CANONICAL, StageStatus.FAILED, sources=())
    graph = with_witness(receipt(GRAPH, StageStatus.APPLIED, sources=committed), (CANONICAL, 1))
    with pytest.raises(ValueError):
        claim(command, [failed, graph])


def test_a_stage_without_upstream_cannot_carry_witnesses_and_only_applied_receipts_may() -> None:
    command, committed = chain_intent()
    canonical = with_witness(receipt(CANONICAL, StageStatus.APPLIED, committed=committed), (GRAPH, 1))
    with pytest.raises(ValueError, match="upstream application"):
        claim(command, [canonical])
    values = receipt(GRAPH, StageStatus.PROCESSING).model_dump()
    values["upstream_applications"] = (
        lc.UpstreamApplication(intent_id="intent-1", stage=CANONICAL, attempt=1, effect_id="effect-canonical"),
    )
    with pytest.raises(ValidationError, match="applied"):
        StageReceipt.model_validate(values)


def test_a_witness_for_another_intent_is_rejected() -> None:
    command, committed = chain_intent()
    canonical, graph = chain_receipts(committed)
    values = graph.model_dump()
    values["upstream_applications"] = (
        lc.UpstreamApplication(intent_id="intent-other", stage=CANONICAL, attempt=1, effect_id="effect-canonical"),
    )
    with pytest.raises(ValueError, match="upstream application"):
        claim(command, [canonical, StageReceipt.model_validate(values)])


def test_the_witness_survives_the_json_wire_form() -> None:
    command, committed = chain_intent()
    canonical, graph = chain_receipts(committed)
    again = StageReceipt.model_validate_json(graph.model_dump_json())
    assert again == graph and again.upstream_applications[0].stage is CANONICAL
    claim(command, [canonical, again])
    legacy = json.loads(graph.model_dump_json())
    legacy.pop("upstream_applications")  # a receipt written before the field existed still parses
    assert StageReceipt.model_validate_json(json.dumps(legacy)).upstream_applications == ()


# ------------------------------------------------------ item 17: wall-clock is not causal evidence
def test_receipt_timestamps_are_not_compared_with_acceptance() -> None:
    """Pinned by decision: wall-clock receipt times are not causal evidence.

    The contract deliberately does not compare worker clocks with ``Intent.accepted_at``: wall-clock
    receipt times are not causal evidence.  Order is proven by the ledger's per-aggregate versions
    and, between stages, by upstream application witnesses.  A receipt
    that claims to have run a day before acceptance is therefore accepted by the contract; a
    ledger that needs ordering reads versions, not these timestamps.
    """

    command = intent(CANONICAL)
    early = receipt(CANONICAL, StageStatus.APPLIED)
    day_before = StageReceipt.model_validate({
        **early.model_dump(),
        "started_at": NOW - timedelta(days=2),
        "finished_at": NOW - timedelta(days=2) + timedelta(seconds=1),
    })
    assert day_before.started_at < command.accepted_at
    claim(command, [day_before])


# --------------------------------------- cross-tenant lineage: data with a marker, never targets
def handle(tenant, *, cross=False, ident="src-1"):
    return ReferenceHandle(kind="source", id=ident, tenant_id=tenant, resolver="resolve_source",
                           cross_tenant=cross)


def test_cross_tenant_source_refs_validate_only_with_the_explicit_marker() -> None:
    """(cross-tenant lineage is allowed as data): a handle may name another
    tenant's source as *data*; resolution always needs a current authorization check."""

    record = memory_record(source_refs=(handle("tenant-2", cross=True), handle("tenant-1")))
    assert record.source_refs[0].cross_tenant is True and record.source_refs[1].cross_tenant is False
    with pytest.raises(ValidationError, match="cross_tenant"):
        memory_record(source_refs=(handle("tenant-2", cross=False),))  # foreign tenant, unmarked
    with pytest.raises(ValidationError, match="cross_tenant"):
        memory_record(source_refs=(handle("tenant-1", cross=True),))  # marker on a same-tenant handle
    assert memory_record(source_refs=(handle("tenant-2", cross=True),)).tenant_id == "tenant-1"


def test_targets_and_outputs_stay_tenant_scoped_even_with_the_marker() -> None:
    command = intent(CANONICAL)
    semantic = "target_refs must match intent tenant_id and cannot be cross_tenant"
    # tuples of handles: strict validation accepts the shape, so only the tenancy rule can reject
    for target in (handle("tenant-2", cross=True), handle("tenant-1", cross=True), handle("tenant-2", cross=False)):
        values = command.model_dump()
        values["target_refs"] = (target,)
        with pytest.raises(ValidationError, match=semantic):
            Intent.model_validate(values)
    allowed = command.model_dump()
    allowed["target_refs"] = (handle("tenant-1"),)
    assert Intent.model_validate(allowed).target_refs == (handle("tenant-1"),)
    output = receipt(CANONICAL, StageStatus.APPLIED, outputs=(handle("tenant-2", cross=True, ident="o"),))
    with pytest.raises(ValueError, match="output_refs"):
        claim(command, [output])


def test_the_marker_defaults_to_false_and_is_part_of_handle_identity() -> None:
    plain = ReferenceHandle(kind="source", id="a", tenant_id="t", resolver="r")
    assert plain.cross_tenant is False  # the default
    assert plain == ReferenceHandle(kind="source", id="a", tenant_id="t", resolver="r", cross_tenant=False)
    assert handle("t", cross=True) != handle("t", cross=False)
    assert ReferenceHandle.model_validate_json(handle("t", cross=True).model_dump_json()).cross_tenant is True


# ----------------------------------------------------------------------------------- the typo
def test_intent_type_errors_use_the_right_article() -> None:
    with pytest.raises(TypeError, match="value must be an Intent"):
        validate_required_stage_claim(claimed_status=IntentStatus.ACCEPTED, intent=object(), receipts=())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="value must be a StageReceipt"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED, intent=intent(CANONICAL), receipts=(object(),)  # type: ignore[arg-type]
        )


# ----------------------------- D1 pins: public view captured at first observation
class LazyPublicTuple(tuple):
    """Native backing differs from a *lazy* public iterator that runs a hook between items."""

    def __new__(cls, native, public_factory):
        instance = super().__new__(cls, native)
        instance.public_factory = public_factory
        instance.iteration_calls = 0
        return instance

    def __iter__(self):
        self.iteration_calls += 1
        return self.public_factory()


def _native_set(model: BaseModel, name: str, value: object) -> None:
    dict.__setitem__(BASE_MODEL_DICT_DESCRIPTOR.__get__(model, BaseModel), name, value)


def test_public_first_sibling_repair_before_first_observation_is_the_documented_limit() -> None:
    """a subclass's public view is a lazy generator, so an earlier
    sibling's callback can repair a later item before that item is captured.

    By design (public view is authoritative): the supported public view is captured **once, at
    first observation**, and replayed; items exposed only by a public iterator are captured when
    first exposed, and entry-time capture of them is not promised.  What is pinned: (1) an invalid
    later public-only receipt is rejected when nothing repairs it; (2) if an earlier item's hook
    repairs it before the iterator exposes it, the repaired value is what is validated (equal to
    plain JSON of the public view at first observation, never a native backing that was not
    exposed); (3) the public view is read exactly once.
    """

    command = intent(CANONICAL, GRAPH)
    first = receipt(CANONICAL, StageStatus.APPLIED)
    later = receipt(GRAPH, StageStatus.APPLIED)
    _native_set(later, "attempt", 0)  # invalid: attempt must be at least 1

    def public(repair: bool):
        def factory():
            def generate():
                yield first
                if repair:  # the earlier item's callback, simulated at the moment it would run
                    _native_set(later, "attempt", 1)
                yield later

            return generate()

        return factory

    plain = LazyPublicTuple((first,), public(False))
    with pytest.raises(ValidationError, match="attempt must be at least 1"):
        claim(command, plain)
    assert plain.iteration_calls == 1

    _native_set(later, "attempt", 0)
    repaired = LazyPublicTuple((first,), public(True))
    claim(command, repaired)  # the repaired public view is the first observation of that child
    assert repaired.iteration_calls == 1


def test_public_first_a_subclass_view_is_observed_once_per_edge_not_once_per_object() -> None:
    """By design (public view is authoritative): the public view of a container is captured once at
    first observation *of each edge that reaches it*.  One subclass object attached to a single
    field is read once; the same object attached to two fields of different receipts is read once
    for each of those edges, and every edge validates the view it observed.  Capture is cached by
    object identity for plain containers and models, not for a subclass's public view, because
    that view is exactly what a subclass may change between observations; nothing here promises a
    single read across edges.  This pins the accepted behavior so a change to it is deliberate.
    """

    ref = handle("tenant-1", ident="o1")
    single = LazyPublicTuple((ref,), lambda: iter((ref,)))
    only = receipt(CANONICAL, StageStatus.APPLIED)
    _native_set(only, "output_refs", single)
    claim(intent(CANONICAL), (only,))
    assert single.iteration_calls == 1

    shared = LazyPublicTuple((ref,), lambda: iter((ref,)))
    first, second = receipt(CANONICAL, StageStatus.APPLIED), receipt(GRAPH, StageStatus.APPLIED)
    _native_set(first, "output_refs", shared)
    _native_set(second, "output_refs", shared)
    claim(intent(CANONICAL, GRAPH), (first, second))
    assert shared.iteration_calls == 2  # one observation per edge


def test_public_first_a_shared_divergent_view_is_never_accepted_on_its_native_backing() -> None:
    """One tuple subclass object shared by ``target_refs`` and ``precondition_source_ids``, with an
    empty public view and a different native backing, is outside the supported envelope.

    By design the public view is authoritative and the native backing never decides anything, so
    two things are pinned.  (1) The backing is not consulted: a claim that is consistent only with
    the native items is rejected, never accepted.  (2) The backing cannot make an otherwise
    consistent claim fail through the shared capture: a claim consistent with the (empty) public
    view gives the same answer whether the backing is plain or reaches back to the intent itself.
    Rejection of the divergent shape, where it happens, is the fail-closed side of the envelope.
    """

    declared = lc.StageEvidenceRequirement(
        stage=CANONICAL, effect_id="effect-canonical", precondition_source_ids=("memory-1",),
        upstream_stages=(), committed_sources=(), minimum_output_refs=0,
    )
    command = intent_with_requirements(declared)
    shared = DivergentTuple(("memory-1",), ())  # native backing names the source; the public view is empty
    _native_set(command, "target_refs", shared)
    _native_set(command.stage_requirements[0], "precondition_source_ids", shared)
    needs_native = receipt(CANONICAL, StageStatus.APPLIED, sources=(source("memory-1"),))
    with pytest.raises(ValueError, match="must match calculated inputs"):
        claim(command, [needs_native])  # accepted only if the native backing had been read

    none_declared = lc.StageEvidenceRequirement(
        stage=CANONICAL, effect_id="effect-canonical", precondition_source_ids=(),
        upstream_stages=(), committed_sources=(), minimum_output_refs=0, no_source_witness=True,
    )
    for native in ((), ("memory-1",), None):
        quiet = intent_with_requirements(none_declared, source_preconditions=())
        backing = native if native is not None else ()
        view = DivergentTuple(backing, ())
        if native is None:  # a backing that reaches back to the intent being validated
            view = DivergentTuple((quiet,), ())
        _native_set(quiet, "target_refs", view)
        _native_set(quiet.stage_requirements[0], "precondition_source_ids", view)
        claim(quiet, [receipt(CANONICAL, StageStatus.APPLIED, sources=())])  # the empty public view decides


def test_public_first_public_only_receipt_repair_by_intent_callback_is_the_documented_limit() -> None:
    """receipts are traversed only after the intent is revalidated, so
    an intent callback can repair a public-only receipt first.

    By design (public view is authoritative): the intent and each public receipt view are captured at their own
    first observation, in source order; a callback that runs while the intent is revalidated may
    change a receipt that has not been observed yet, and the receipt is then validated as it is
    when first observed.  Pinned: unrepaired -> rejected; repaired by the intent's callback -> the
    repaired receipt is accepted; the divergent public tuple is read once.
    """

    command = intent(CANONICAL)
    valid = receipt(CANONICAL, StageStatus.APPLIED)
    public_receipt = receipt(CANONICAL, StageStatus.APPLIED)
    _native_set(public_receipt, "attempt", 0)
    divergent = DivergentTuple((valid,), (public_receipt,))

    with pytest.raises(ValidationError, match="attempt must be at least 1"):
        claim(command, divergent)
    assert divergent.iteration_calls == 1

    _native_set(public_receipt, "attempt", 0)
    trigger = install_repairing_target_refs(command, public_receipt, field_name="attempt", replacement=1)
    divergent = DivergentTuple((valid,), (public_receipt,))
    claim(command, divergent)  # the intent's callback repaired the later, not-yet-observed receipt
    assert divergent.iteration_calls == 1 and NestedRepairingTuple.iteration_calls == 1
    assert trigger.target is public_receipt
