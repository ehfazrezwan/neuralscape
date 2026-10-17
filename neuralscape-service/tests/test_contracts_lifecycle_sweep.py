"""Hostile-input sweep over the lifecycle contracts and the legacy observations.

Each class of caller-controlled behaviour is pinned: a class spoofing ``__class__``, a str subclass
used as a key or a value, a datetime subclass answering comparisons, shared or deeply nested graphs,
and a str subclass standing in for a response status.
"""

import ast
import inspect
import signal
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

import contracts_legacy_async
import contracts_lifecycle
from contracts_legacy_async import (
    LegacyAsyncObservationKind,
    observe_task_accepted,
    observe_task_status,
)
from contracts_lifecycle import (
    Intent,
    IntentStatus,
    ProcessingStage,
    StageError,
    StageReceipt,
    StageStatus,
    source_versions_match,
    validate_required_stage_claim,
)
from contracts_references import SourceVersion
from schemas import TaskAcceptedResponse, TaskStatusResponse
from tests.test_contracts_lifecycle import intent, receipt, source

EXTRA = vars(BaseModel)["__pydantic_extra__"]
BUDGET = "contract graph exceeds the native discovery budget"


class Spoof:
    """Delegates everything to ``target`` while claiming to be an instance of ``cls``."""

    def __init__(self, target, cls):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_cls", cls)

    def __getattribute__(self, name):
        if name == "__class__":
            return object.__getattribute__(self, "_cls")
        return getattr(object.__getattribute__(self, "_target"), name)


class Twin(str):
    """A str whose own equality and hash disagree with its text."""

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __hash__(self):
        return 424242


class Late(datetime):
    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return True


class Alarm:
    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        def expired(*_):
            raise TimeoutError("did not finish")

        signal.signal(signal.SIGALRM, expired)
        signal.alarm(self.seconds)

    def __exit__(self, *_):
        signal.alarm(0)


def claim(command):
    validate_required_stage_claim(claimed_status=IntentStatus.ACCEPTED, intent=command, receipts=())


# ---- a class that spoofs __class__ is not trusted
def test_a_spoofed_intent_is_refused():
    invalid = intent(ProcessingStage.CANONICAL).model_copy(update={"tenant_id": 5})
    with pytest.raises((TypeError, ValueError)):
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED, intent=Spoof(invalid, Intent), receipts=()
        )


def test_a_spoofed_receipt_is_refused():
    invalid = receipt(ProcessingStage.CANONICAL, StageStatus.PENDING).model_copy(update={"attempt": -4})
    with pytest.raises((TypeError, ValueError)):
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=intent(ProcessingStage.CANONICAL),
            receipts=(Spoof(invalid, StageReceipt),),
        )


def test_a_spoofed_source_version_is_refused_on_either_side():
    invalid = source("m").model_copy(update={"content_revision": -1})
    with pytest.raises((TypeError, ValueError)):
        source_versions_match(Spoof(invalid, SourceVersion), source("m"))
    with pytest.raises((TypeError, ValueError)):
        source_versions_match(source("m"), Spoof(invalid, SourceVersion))


def test_a_spoof_of_a_valid_contract_is_refused_at_every_entry():
    """``isinstance`` honours ``__class__``: a proxy of a valid model must not stand in for one."""

    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=(applied,))
    for entry in (
        lambda: contracts_lifecycle.validate_stage_receipts(Spoof(command, Intent), (applied,)),
        lambda: contracts_lifecycle.validate_stage_receipts(command, (Spoof(applied, StageReceipt),)),
        lambda: validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED, intent=Spoof(command, Intent), receipts=(applied,)
        ),
        lambda: validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED, intent=command, receipts=(Spoof(applied, StageReceipt),)
        ),
        lambda: contracts_lifecycle.validate_stage_receipts(command, Spoof((applied,), tuple)),
        lambda: source_versions_match(Spoof(source("m"), SourceVersion), source("m")),
        lambda: source_versions_match(source("m"), Spoof(source("m"), SourceVersion)),
    ):
        with pytest.raises(TypeError):
            entry()


def test_a_spoofed_receipt_cannot_carry_a_graph_past_the_claim_budget():
    import time

    from tests.test_contracts_lifecycle import intent_with_requirements, reference, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    large = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    set_native(large, "output_refs", tuple(reference(f"o{i}") for i in range(70_000)))
    with pytest.raises(ValueError, match=BUDGET):  # beyond the tier, read natively
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=(large,))
    started = time.monotonic()
    with pytest.raises((TypeError, ValueError)):  # the proxy's own view is never read outside the budget
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED, intent=command, receipts=(Spoof(large, StageReceipt),)
        )
    assert time.monotonic() - started < 30


class CountingSpoof:
    """A proxy that claims ``cls`` and counts every attribute read made through it."""

    def __init__(self, target, cls):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_cls", cls)
        object.__setattr__(self, "reads", 0)

    def __getattribute__(self, name):
        if name in ("_target", "_cls", "reads"):
            return object.__getattribute__(self, name)
        object.__setattr__(self, "reads", object.__getattribute__(self, "reads") + 1)
        if name == "__class__":
            return object.__getattribute__(self, "_cls")
        return getattr(object.__getattribute__(self, "_target"), name)


@pytest.mark.parametrize("position", ["output_ref", "source_precondition", "stage_requirement", "upstream_witness"])
def test_a_proxy_at_a_nested_position_is_refused_before_validation_reads_it(position):
    """A nested proxy is not a native value: capture refuses it, so no caller hook runs inside validation."""

    from contracts_lifecycle import StageEvidenceRequirement, UpstreamApplication
    from contracts_references import ReferenceHandle
    from tests.test_contracts_lifecycle import intent_with_requirements, reference, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    if position == "output_ref":
        proxy = CountingSpoof(reference("o1"), ReferenceHandle)
        set_native(applied, "output_refs", (proxy,))
    elif position == "source_precondition":
        proxy = CountingSpoof(command.source_preconditions[0], SourceVersion)
        set_native(command, "source_preconditions", (proxy,) + command.source_preconditions[1:])
    elif position == "stage_requirement":
        proxy = CountingSpoof(command.stage_requirements[0], StageEvidenceRequirement)
        set_native(command, "stage_requirements", (proxy,))
    else:
        witness = UpstreamApplication(intent_id=applied.intent_id, stage=ProcessingStage.CANONICAL, attempt=1, effect_id="e")
        proxy = CountingSpoof(witness, UpstreamApplication)
        set_native(applied, "upstream_applications", (proxy,))
    with pytest.raises(ValueError, match="native"):
        contracts_lifecycle.validate_stage_receipts(command, (applied,))
    with pytest.raises(ValueError, match="native"):
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=(applied,))
    assert object.__getattribute__(proxy, "reads") == 0


def test_a_proxy_yielded_by_a_public_view_is_refused_before_validation_reads_it():
    """A tuple subclass's public view is read item by item: a yielded proxy is refused like a nested one."""

    from contracts_references import ReferenceHandle
    from tests.test_contracts_lifecycle import intent_with_requirements, reference, requirement

    class View(tuple):
        def __iter__(self):
            yield proxy

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    proxy = CountingSpoof(reference("o1"), ReferenceHandle)
    set_native(applied, "output_refs", View((reference("o1"),)))  # native backing is a real handle
    with pytest.raises(ValueError, match="native"):
        contracts_lifecycle.validate_stage_receipts(command, (applied,))
    with pytest.raises(ValueError, match="native"):
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=(applied,))
    assert object.__getattribute__(proxy, "reads") == 0


def test_the_receipts_and_model_entry_checks_decide_by_the_real_type():
    """Pins the two entry checks no other test tells apart from ``isinstance``."""

    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    with pytest.raises(TypeError, match="receipts must be a tuple"):
        contracts_lifecycle.validate_stage_receipts(command, Spoof((applied,), tuple))
    with pytest.raises(TypeError, match="value must be an Intent"):
        contracts_lifecycle._revalidated_model(Intent, Spoof(command, Intent))


def test_a_noncanonical_enum_instance_is_refused_at_every_position():
    """``str.__new__(Enum, value)`` is an instance of the enum but no member: identity tests would misjudge it."""

    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    for field, odd in (
        ("status", str.__new__(StageStatus, "applied")),
        ("stage", str.__new__(ProcessingStage, "canonical")),
    ):
        applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
        set_native(applied, field, odd)
        with pytest.raises(ValueError, match="declared member"):
            contracts_lifecycle.validate_stage_receipts(command, (applied,))
        with pytest.raises(ValueError, match="declared member"):
            validate_required_stage_claim(claimed_status=IntentStatus.PARTIAL, intent=command, receipts=(applied,))
    odd_requirement = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    set_native(odd_requirement.stage_requirements[0], "stage", str.__new__(ProcessingStage, "canonical"))
    with pytest.raises(ValueError, match="declared member"):
        validate_required_stage_claim(claimed_status=IntentStatus.ACCEPTED, intent=odd_requirement, receipts=())
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    with pytest.raises(TypeError, match="claimed_status"):
        validate_required_stage_claim(
            claimed_status=str.__new__(IntentStatus, "applied"), intent=command, receipts=(applied,)
        )
    validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=(applied,))


def test_native_scalars_and_their_subclasses_still_reach_validation():
    """Only non-native leaves are refused: an int subclass or datetime subclass is still judged by validation."""

    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    set_native(applied, "attempt", type("Count", (int,), {})(1))
    kept_intent, kept = contracts_lifecycle.validate_stage_receipts(command, (applied,))
    assert kept[0].attempt == 1
    late = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    set_native(late, "finished_at", Late(2000, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="finished_at cannot precede started_at"):
        contracts_lifecycle.validate_stage_receipts(command, (late,))


class Alias(str):
    """A key whose text names one field while its hash and equality name another."""

    def __new__(cls, text, claimed):
        made = str.__new__(cls, text)
        made.claimed = claimed
        return made

    def __hash__(self):
        return hash(self.claimed)

    def __eq__(self, other):
        return str.__eq__(self.claimed, other) if isinstance(other, str) else NotImplemented

    def __ne__(self, other):
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal


def alias_storage(stored, first, second):
    """Keep each value under its own text, with the two keys' hash and equality swapped."""

    first_value, second_value = stored[first], stored[second]
    del stored[first]
    del stored[second]
    stored[Alias(first, second)] = first_value
    stored[Alias(second, first)] = second_value


def test_model_storage_keys_are_read_by_their_text_not_their_hash():
    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL)).model_copy(
        update={"actor_id": "tenant-evil"}
    )
    alias_storage(NATIVE_DICT.__get__(command, BaseModel), "tenant_id", "actor_id")
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    effect = applied.effect_id
    alias_storage(NATIVE_DICT.__get__(applied, BaseModel), "intent_id", "effect_id")
    kept_intent, kept = contracts_lifecycle.validate_stage_receipts(command, (applied,))
    assert (kept_intent.tenant_id, kept_intent.actor_id) == ("tenant-1", "tenant-evil")  # as the text says
    assert kept[0].intent_id == kept_intent.id and kept[0].effect_id == effect
    assert type(next(iter(NATIVE_DICT.__get__(kept_intent, BaseModel)))) is str


def test_raw_mapping_keys_are_read_by_their_text_not_their_hash():
    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    handle = {"kind": "artifact", "id": "tenant-1", "tenant_id": "other-tenant", "resolver": "resolve_artifact"}
    alias_storage(handle, "id", "tenant_id")  # by text the handle names another tenant
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    set_native(applied, "output_refs", (handle,))
    with pytest.raises(ValueError, match="output_refs must match intent tenant_id"):
        contracts_lifecycle.validate_stage_receipts(command, (applied,))
    with pytest.raises(ValueError, match="output_refs must match intent tenant_id"):
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=(applied,))


# ---- str subclasses
def test_two_source_ids_with_one_text_stay_duplicates_when_one_is_a_str_subclass():
    values = intent(ProcessingStage.CANONICAL).model_dump()
    values["source_preconditions"] = (
        SourceVersion(record_id=Twin("m"), content_revision=1, policy_epoch=1),
        SourceVersion(record_id="m", content_revision=1, policy_epoch=1),
    )
    with pytest.raises(ValueError):
        Intent.model_validate(values)


def test_a_str_subclass_native_key_cannot_smuggle_an_undeclared_field_past_a_declared_name():
    command = intent(ProcessingStage.CANONICAL)
    EXTRA.__set__(command, {Twin("tenant_id"): "other"})
    with pytest.raises(ValueError, match="conflicting stored contract field"):
        claim(command)


def test_a_str_subclass_extra_key_with_a_new_name_is_still_an_undeclared_field():
    command = intent(ProcessingStage.CANONICAL)
    EXTRA.__set__(command, {Twin("undeclared"): True})
    with pytest.raises(ValueError):
        claim(command)


def test_two_native_keys_that_spell_one_name_are_refused():
    first, second = Twin("k"), "k"
    assert len({first: 1, second: 2}) == 2  # they coexist natively only because their hooks differ
    command = intent(ProcessingStage.CANONICAL).model_copy(update={"id": {first: 1, second: 2}})
    with pytest.raises(ValueError, match="colliding keys"):
        claim(command)


# ---- timestamp ordering is not answered by a subclass
def test_a_datetime_subclass_cannot_hide_a_finish_before_the_start():
    started = Late(2026, 9, 28, 12, 0, 10, tzinfo=timezone.utc)
    finished = Late(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
    values = dict(
        schema_version="candidate-v1", intent_id="intent-1", attempt=1, stage=ProcessingStage.CANONICAL,
        effect_id="effect-canonical", status=StageStatus.SKIPPED, applied_sources=(), committed_sources=(),
        output_refs=(), error=None, started_at=started, finished_at=finished,
    )
    with pytest.raises(ValueError, match="cannot precede"):
        StageReceipt(**values)
    values.update(started_at=finished, finished_at=started)
    StageReceipt(**values)  # the ordered pair is accepted


def test_the_lifecycle_modules_never_read_the_clock():
    for module in (contracts_lifecycle, contracts_legacy_async):
        calls = {
            f"{getattr(node.func.value, 'id', '')}.{node.func.attr}"
            for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not calls & {"datetime.now", "datetime.utcnow", "time.time", "time.monotonic"}


# ---- reconstruction cost
def test_a_native_diamond_is_refused_not_reconstructed_per_path():
    """Built once per object (2^40 paths, 41 objects): each reuse is charged its expanded size, so the
    budget refuses it at once (arithmetic over 41 objects, not a walk of the paths)."""

    shared = ("x",)
    for _ in range(40):
        shared = (shared, shared)
    command = intent(ProcessingStage.CANONICAL).model_copy(update={"id": shared})
    with Alarm(20), pytest.raises(ValueError, match=BUDGET):
        claim(command)


def test_a_diamond_of_public_views_is_refused():
    class View(list):
        pass

    shared = View(["x"])
    for _ in range(40):
        shared = View([shared, shared])
    command = intent(ProcessingStage.CANONICAL).model_copy(update={"id": shared})
    with Alarm(20), pytest.raises(ValueError, match=BUDGET):
        claim(command)


def test_a_view_that_never_ends_is_refused():
    class Endless(tuple):
        def __iter__(self):
            while True:
                yield 1

    command = intent(ProcessingStage.CANONICAL).model_copy(update={"id": Endless((1,))})
    with Alarm(20), pytest.raises(ValueError, match=BUDGET):
        claim(command)


def test_deep_nesting_is_a_value_error_not_a_recursion_error():
    nested = ["x"]
    for _ in range(20_000):
        nested = [nested]
    command = intent(ProcessingStage.CANONICAL).model_copy(update={"id": nested})
    with pytest.raises(ValueError, match=BUDGET):
        claim(command)


def test_a_legitimate_large_receipt_set_and_a_small_shared_subtree_are_accepted():
    claim(intent(ProcessingStage.CANONICAL, ProcessingStage.VECTOR, ProcessingStage.GRAPH, ProcessingStage.MARKDOWN))
    shared = ("x",)
    for _ in range(3):
        shared = (shared, shared)
    # a small diamond is within the budget; it is then refused by the field type, not by cost
    command = intent(ProcessingStage.CANONICAL).model_copy(update={"id": shared})
    with pytest.raises(ValueError) as raised:
        claim(command)
    assert BUDGET not in str(raised.value)


# ---- legacy observations are not answered by a str subclass
class Always(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = str.__hash__


def test_a_spoofed_status_is_not_acknowledged_as_accepted():
    response = TaskAcceptedResponse.model_construct(status=Always("failed"), task_id="t")
    with pytest.raises(ValueError):
        observe_task_accepted(response)
    genuine = TaskAcceptedResponse.model_construct(status="accepted", task_id="t")
    assert observe_task_accepted(genuine).kind is LegacyAsyncObservationKind.QUEUE_ACKNOWLEDGED


def test_a_spoofed_polling_status_is_not_mapped_by_its_own_hooks():
    class Mimic(str):
        def __eq__(self, other):
            return other == "completed"

        def __hash__(self):
            return hash("completed")

    response = TaskStatusResponse.model_construct(status=Mimic("failed"), task_id=Always("t"))
    observed = observe_task_status(response)
    assert observed.kind is LegacyAsyncObservationKind.WORKER_FAILED
    assert type(observed.task_id) is str and observed.task_id == "t"


# ---- one shared budget from the shared limits module
def test_the_budget_message_and_tier_are_the_shared_ones(monkeypatch):
    import contracts_limits

    assert BUDGET == contracts_limits.DISCOVERY_BUDGET_MESSAGE
    assert not [name for name in vars(contracts_lifecycle) if name.startswith("_MAX_GRAPH") or name == "_GraphBudget"]
    made = []
    real = contracts_limits.DiscoveryBudget

    def recording(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(contracts_lifecycle, "DiscoveryBudget", recording)
    command = intent(ProcessingStage.CANONICAL)
    validate_required_stage_claim(
        claimed_status=IntentStatus.PROCESSING,
        intent=command,
        receipts=(receipt(ProcessingStage.CANONICAL, StageStatus.PROCESSING),) * 3,
    )
    assert len(made) == 1 and made[0].tier is contracts_lifecycle.LIFECYCLE_TIER  # one envelope for every root
    assert contracts_lifecycle.LIFECYCLE_TIER is contracts_limits.PORTABILITY
    made.clear()
    source_versions_match(source("m"), source("m"))
    assert len(made) == 1


def test_the_depth_limit_is_the_shared_standard_depth():
    import contracts_limits

    def nested(depth):
        value = ["x"]
        for _ in range(depth - 1):
            value = [value]
        return value

    within = intent(ProcessingStage.CANONICAL).model_copy(update={"id": nested(contracts_limits.STANDARD_MAX_DEPTH - 8)})
    with pytest.raises(ValueError) as raised:
        claim(within)
    assert BUDGET not in str(raised.value)  # refused by the field type, not by the budget
    beyond = intent(ProcessingStage.CANONICAL).model_copy(update={"id": nested(contracts_limits.STANDARD_MAX_DEPTH + 8)})
    with pytest.raises(ValueError, match=BUDGET):
        claim(beyond)


def test_a_flood_of_edges_in_one_root_is_refused_by_the_shared_envelope():
    flood = intent(ProcessingStage.CANONICAL).model_copy(
        update={"id": [list(range(200)) for _ in range(4_200)]}
    )
    with pytest.raises(ValueError, match=BUDGET):
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=flood,
            receipts=(receipt(ProcessingStage.CANONICAL, StageStatus.PROCESSING),),
        )


# ---- a public view is observed once per call, and a hidden native cycle is not a permanent mark
NATIVE_DICT = vars(BaseModel)["__dict__"]


def set_native(model, field, value):
    dict.__setitem__(NATIVE_DICT.__get__(model, BaseModel), field, value)


class Stateful(tuple):
    """A public view whose items change every time it is iterated."""

    def __new__(cls, rounds):
        view = super().__new__(cls, ())
        view.rounds = rounds
        view.passes = 0
        return view

    def __iter__(self):
        pass_number = self.passes
        self.passes += 1
        return iter(self.rounds[min(pass_number, len(self.rounds) - 1)])


def test_a_stateful_view_aliased_on_two_edges_is_observed_once():
    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    view = Stateful([(source("a"),), (source("b"),), (source("c"),)])
    set_native(command, "source_preconditions", view)
    set_native(
        command.stage_requirements[0], "committed_sources", view
    )
    set_native(command.stage_requirements[0], "precondition_source_ids", ())
    kept = contracts_lifecycle._revalidated_model(Intent, command)
    assert kept.source_preconditions == kept.stage_requirements[0].committed_sources
    assert view.passes == 1


def test_duplicate_receipts_with_a_stateful_output_view_stay_idempotent():
    from tests.test_contracts_lifecycle import intent_with_requirements, reference, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED, outputs=(reference("one"),))
    view = Stateful([(reference("one"),), (reference("one"), reference("two"))])
    set_native(applied, "output_refs", view)
    validate_required_stage_claim(
        claimed_status=IntentStatus.APPLIED, intent=command, receipts=(applied, applied)
    )
    assert view.passes == 1


def test_a_view_shared_by_the_intent_and_a_receipt_is_observed_once():
    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    view = Stateful([(source("memory-1"),), (source("other"),)])
    set_native(command, "source_preconditions", view)
    set_native(applied, "applied_sources", view)
    validate_required_stage_claim(
        claimed_status=IntentStatus.APPLIED, intent=command, receipts=(applied,)
    )
    assert view.passes == 1


def hidden_cycle():
    """A tuple subclass showing nothing publicly whose native backing runs back through a list to it."""

    class Quiet(tuple):
        def __iter__(self):
            return iter(())

    loop = []
    view = Quiet((loop,))
    loop.append(view)
    return view, loop


def failure(command):
    with pytest.raises(ValueError) as raised:
        claim(command)
    return str(raised.value)


def test_a_hidden_native_cycle_behind_an_acyclic_public_view_is_order_independent():
    view, loop = hidden_cycle()
    base = intent(ProcessingStage.CANONICAL)
    first = base.model_copy(update={"id": view, "actor_id": loop})
    view, loop = hidden_cycle()
    second = base.model_copy(update={"id": loop, "actor_id": view})
    one, two = failure(first), failure(second)
    assert "cyclic" not in one and "cyclic" not in two
    assert one.count("validation error") == two.count("validation error")


def test_a_real_native_cycle_is_still_refused_from_every_entry():
    ring = []
    ring.append(ring)
    command = intent(ProcessingStage.CANONICAL).model_copy(update={"id": ring})
    assert "cyclic" in failure(command)
    left, right = [], []
    left.append(right)
    right.append(left)
    for update in ({"id": left, "actor_id": right}, {"id": right, "actor_id": left}):
        assert "cyclic" in failure(intent(ProcessingStage.CANONICAL).model_copy(update=update))


# ---- receipt size: a large stage output settles, and record time agrees with the claim
def many_outputs(count, *, status=StageStatus.APPLIED, attempt=1, stage=ProcessingStage.CANONICAL):
    from tests.test_contracts_lifecycle import reference

    outputs = tuple(reference(f"o{attempt}-{i}") for i in range(count))
    return receipt(stage, status, outputs=outputs, attempt=attempt)


def test_a_stage_with_ten_thousand_output_refs_validates_and_settles_in_bounded_time():
    import time

    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    applied = many_outputs(10_000)
    started = time.monotonic()
    validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=(applied,))
    kept_intent, kept = contracts_lifecycle.validate_stage_receipts(command, (applied,))
    assert len(kept[0].output_refs) == 10_000 and kept_intent == command
    assert time.monotonic() - started < 30


def test_the_standard_tier_would_strand_a_large_stage_and_the_lifecycle_tier_does_not(monkeypatch):
    import contracts_limits
    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    large = many_outputs(20_000)  # more than the standard tier admits in one call
    validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=(large,))
    monkeypatch.setattr(contracts_lifecycle, "LIFECYCLE_TIER", contracts_limits.STANDARD)
    with pytest.raises(ValueError, match=BUDGET):
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=(large,))


def test_a_receipt_set_beyond_the_tier_is_refused_at_record_time_with_the_budget_message():
    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    too_many = many_outputs(70_000)  # beyond the roughly 65,500 handles the tier admits
    with pytest.raises(ValueError, match=BUDGET):
        contracts_lifecycle.validate_stage_receipts(command, (too_many,))
    with pytest.raises(ValueError, match=BUDGET):
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=(too_many,))


def test_record_time_counts_the_receipts_already_held_so_it_agrees_with_the_claim():
    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    earlier = tuple(
        many_outputs(9_000, status=StageStatus.SKIPPED, attempt=number) for number in range(1, 8)
    )  # each fits alone; together they exceed one call's envelope
    contracts_lifecycle.validate_stage_receipts(command, earlier[:6])
    with pytest.raises(ValueError, match=BUDGET):
        contracts_lifecycle.validate_stage_receipts(command, earlier + (many_outputs(9_000, status=StageStatus.SKIPPED, attempt=8),))


def test_record_time_applies_the_same_identity_checks_as_the_claim():
    from tests.test_contracts_lifecycle import intent_with_requirements, reference, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    wrong_tenant = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED, outputs=(reference("x", tenant_id="other"),))
    with pytest.raises(ValueError, match="output_refs must match intent tenant_id"):
        contracts_lifecycle.validate_stage_receipts(command, (wrong_tenant,))
    with pytest.raises(ValueError, match="output_refs must match intent tenant_id"):
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=(wrong_tenant,))
    wrong_effect = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED, effect_id="elsewhere")
    with pytest.raises(ValueError, match="declared effect_id"):
        contracts_lifecycle.validate_stage_receipts(command, (wrong_effect,))
    with pytest.raises(TypeError):
        contracts_lifecycle.validate_stage_receipts(command, [wrong_effect])  # type: ignore[arg-type]


# ---- the copies record time returns are qualified: schema defaults cannot carry them past the claim
def unchecked_outputs(count, form):
    """An applied receipt whose stored handles lack ``cross_tenant``, which validation adds."""

    from contracts_references import ReferenceHandle

    fields = [
        {"kind": "artifact", "id": f"o{i}", "tenant_id": "tenant-1", "resolver": "resolve_artifact"}
        for i in range(count)
    ]
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    if form == "dict":
        outputs = tuple(fields)
    else:
        outputs = tuple(ReferenceHandle.model_construct(_fields_set=set(item), **item) for item in fields)
        for handle in outputs:  # model_construct fills the default; drop it from storage
            dict.__delitem__(NATIVE_DICT.__get__(handle, BaseModel), "cross_tenant")
    set_native(applied, "output_refs", outputs)
    return applied


@pytest.mark.parametrize("form", ["dict", "constructed"])
def test_record_time_refuses_a_set_whose_normalized_copies_the_claim_could_not_read(form):
    import time
    import tracemalloc

    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    expanding = unchecked_outputs(70_000, form)  # fits as stored; one more field per handle once validated
    started = time.monotonic()
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match=BUDGET):
            contracts_lifecycle.validate_stage_receipts(command, (expanding,))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert time.monotonic() - started < 60
    assert peak < 512 * 2**20, peak


@pytest.mark.parametrize("form", ["dict", "constructed"])
def test_copies_returned_at_record_time_are_settled_by_the_claim_and_are_a_fixed_point(form):
    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    kept_intent, kept = contracts_lifecycle.validate_stage_receipts(command, (unchecked_outputs(60_000, form),))
    assert all(handle.cross_tenant is False for handle in kept[0].output_refs)
    validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=kept_intent, receipts=kept)
    assert contracts_lifecycle.validate_stage_receipts(kept_intent, kept) == (kept_intent, kept)
    received = tuple(StageReceipt.model_validate_json(item.model_dump_json()) for item in kept)
    validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=kept_intent, receipts=received)


# ---- a native container is sized against the remaining budget before it is copied
def test_a_wide_native_container_is_refused_before_it_is_copied():
    import tracemalloc

    import contracts_limits

    wide = {f"k{i}": i for i in range(400_000)}
    tight = contracts_limits.DiscoveryTier("tight", 1_000, 1_000, 64)
    budget = contracts_limits.DiscoveryBudget(tight)
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match=BUDGET):
            contracts_lifecycle._capture_native_contract_graph(wide, budget=budget)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000, peak  # a copy of its entries would be tens of megabytes
    with pytest.raises(ValueError, match=BUDGET):
        budget.charge_edge()  # and the budget stays exhausted


@pytest.mark.parametrize("kind", ["list", "tuple", "model"])
def test_wide_lists_tuples_and_model_storage_are_refused_before_they_are_copied(kind):
    import tracemalloc

    import contracts_limits

    if kind == "model":
        value = intent(ProcessingStage.CANONICAL)
        set_native(value, "target_refs", None)
        EXTRA.__set__(value, {f"x{i}": i for i in range(300_000)})
    else:
        value = list(range(400_000)) if kind == "list" else tuple(range(400_000))
    budget = contracts_limits.DiscoveryBudget(contracts_limits.DiscoveryTier("tight", 1_000, 1_000, 64))
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match=BUDGET):
            contracts_lifecycle._capture_native_contract_graph(value, budget=budget)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000, peak


def test_a_container_that_fits_the_remaining_budget_is_still_captured_whole():
    import contracts_limits

    budget = contracts_limits.DiscoveryBudget(contracts_limits.DiscoveryTier("tight", 1_000, 1_000, 64))
    captured = contracts_lifecycle._capture_native_contract_graph(list(range(900)), budget=budget)
    assert len(captured.items) == 900


# ---- the receipt inventory is charged and stops at exhaustion
def test_every_receipt_reference_is_charged_including_duplicates_and_non_receipts():
    command = intent(ProcessingStage.CANONICAL)
    pending = receipt(ProcessingStage.CANONICAL, StageStatus.PENDING)
    flood = (pending,) * 400_000 + (object(),) * 400_000  # more references than the tier's edges
    with pytest.raises(ValueError, match=BUDGET):
        contracts_lifecycle.validate_stage_receipts(command, flood)


def test_no_receipt_is_captured_once_the_intent_capture_failed(monkeypatch):
    real = contracts_lifecycle._capture_native_contract_graph
    receipts_captured = []

    def counting(value, *args, **kwargs):
        if isinstance(value, StageReceipt):
            receipts_captured.append(value)
        return real(value, *args, **kwargs)

    monkeypatch.setattr(contracts_lifecycle, "_capture_native_contract_graph", counting)
    flood = intent(ProcessingStage.CANONICAL).model_copy(update={"id": [list(range(200)) for _ in range(4_200)]})
    many = tuple(receipt(ProcessingStage.CANONICAL, StageStatus.PENDING) for _ in range(50))
    with pytest.raises(ValueError, match=BUDGET):
        contracts_lifecycle.validate_stage_receipts(flood, many)
    assert receipts_captured == []


def test_inventory_stops_at_exhaustion_and_the_first_error_keeps_its_precedence(monkeypatch):
    from tests.test_contracts_lifecycle import reference

    real = contracts_lifecycle._capture_native_contract_graph
    receipts_captured = []

    def counting(value, *args, **kwargs):
        if isinstance(value, StageReceipt):
            receipts_captured.append(value)
        return real(value, *args, **kwargs)

    monkeypatch.setattr(contracts_lifecycle, "_capture_native_contract_graph", counting)
    command = intent(ProcessingStage.CANONICAL)
    invalid_first = receipt(ProcessingStage.CANONICAL, StageStatus.PENDING)
    EXTRA.__set__(invalid_first, {Twin("note"): 1, "note": 2})  # refused while it is captured
    huge = receipt(ProcessingStage.CANONICAL, StageStatus.SKIPPED,
                   outputs=tuple(reference(f"o{i}") for i in range(140_000)))
    later = tuple(receipt(ProcessingStage.CANONICAL, StageStatus.PENDING, attempt=n) for n in range(2, 40))
    with pytest.raises(ValueError, match=BUDGET):  # one call, one envelope: the intent is revalidated first
        contracts_lifecycle.validate_stage_receipts(command, (huge, invalid_first) + later)
    assert len(receipts_captured) == 1  # nothing is captured after the budget ran out
    receipts_captured.clear()
    with pytest.raises(ValueError, match="colliding keys"):  # the inventory also stops at the first error,
        contracts_lifecycle.validate_stage_receipts(command, (invalid_first, huge) + later)
    assert len(receipts_captured) == 1  # so a later receipt is never captured (or charged) ahead of it
    receipts_captured.clear()
    with pytest.raises(ValueError, match="colliding keys"):  # within the budget the earlier capture error wins
        contracts_lifecycle.validate_stage_receipts(command, (invalid_first,) + later)


# ---- receipt ordering compares instants, not wall times
def test_finish_before_start_across_a_repeated_hour_is_refused_natively_and_on_the_wire():
    from zoneinfo import ZoneInfo

    new_york = ZoneInfo("America/New_York")
    started = datetime(2025, 11, 2, 1, 15, fold=1, tzinfo=new_york)   # 06:15 UTC (second 01:15, EST)
    finished = datetime(2025, 11, 2, 1, 45, fold=0, tzinfo=new_york)  # 05:45 UTC (first 01:45, EDT)
    values = dict(
        schema_version="candidate-v1", intent_id="intent-1", attempt=1, stage=ProcessingStage.CANONICAL,
        effect_id="effect-canonical", status=StageStatus.SKIPPED, applied_sources=(), committed_sources=(),
        output_refs=(), error=None, started_at=started, finished_at=finished,
    )
    with pytest.raises(ValueError, match="cannot precede"):
        StageReceipt(**values)
    wire = {**values, "stage": "canonical", "status": "skipped",
            "started_at": started.isoformat(), "finished_at": finished.isoformat()}
    import json

    with pytest.raises(ValueError, match="cannot precede"):
        StageReceipt.model_validate_json(json.dumps(wire))


def test_the_ordered_pair_across_a_repeated_hour_is_accepted_natively_and_on_the_wire():
    import json
    from zoneinfo import ZoneInfo

    new_york = ZoneInfo("America/New_York")
    started = datetime(2025, 11, 2, 1, 45, fold=0, tzinfo=new_york)   # 05:45 UTC
    finished = datetime(2025, 11, 2, 1, 15, fold=1, tzinfo=new_york)  # 06:15 UTC: later, though the wall time is earlier
    values = dict(
        schema_version="candidate-v1", intent_id="intent-1", attempt=1, stage=ProcessingStage.CANONICAL,
        effect_id="effect-canonical", status=StageStatus.SKIPPED, applied_sources=(), committed_sources=(),
        output_refs=(), error=None, started_at=started, finished_at=finished,
    )
    StageReceipt(**values)
    wire = {**values, "stage": "canonical", "status": "skipped",
            "started_at": started.isoformat(), "finished_at": finished.isoformat()}
    StageReceipt.model_validate_json(json.dumps(wire))


def test_enum_membership_is_read_without_metaclass_hooks_so_a_later_value_stays_frozen():
    """A genuine member of a caller's enum whose metaclass hooks attribute reads must not let capture run them:
    the hook would repair a later source's content revision before that source is frozen."""

    import enum

    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    later = command.source_preconditions[0]
    armed = [False]
    hooked: list[str] = []

    class HookedMeta(type(enum.Enum)):
        def __getattribute__(cls, name):
            if armed[0] and name in ("__members__", "_member_map_", "_value2member_map_", "__iter__"):
                hooked.append(name)
                set_native(later, "content_revision", 1)
            return super().__getattribute__(name)

    class Digest(str, enum.Enum, metaclass=HookedMeta):  # creating the class reads its maps: not armed yet
        ABC = "sha256:abc"

    member = Digest.ABC
    set_native(command, "request_digest", member)  # an early field, captured before the source
    set_native(later, "content_revision", -1)  # invalid, and it must stay so
    armed[0] = True
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    with pytest.raises(ValueError):
        contracts_lifecycle.validate_stage_receipts(command, (applied,))
    with pytest.raises(ValueError):
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=(applied,))
    assert hooked == []


def test_a_metaclass_that_claims_a_forged_instance_as_a_member_cannot_make_it_one():
    """The member map is read from the class's own namespace, not from an overridable ``__members__``."""

    import enum

    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    forged: list[object] = []

    class ClaimingMeta(type(enum.Enum)):
        @property
        def __members__(cls):
            return {"ABC": forged[0]} if forged else super().__members__

    class Digest(str, enum.Enum, metaclass=ClaimingMeta):
        ABC = "sha256:abc"

    forged.append(str.__new__(Digest, "sha256:abc"))  # an instance of the enum, but not its member
    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    set_native(command, "request_digest", forged[0])
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    with pytest.raises(ValueError, match="declared member"):
        contracts_lifecycle.validate_stage_receipts(command, (applied,))
    with pytest.raises(ValueError, match="declared member"):
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=(applied,))


def _scalar_proxy(kind: str, target: object, reads: list[str]):
    """A ``str`` subclass that claims ``type(target)`` and delegates its storage to ``target``."""

    claimed = type(target)
    if kind == "class-property":

        class Proxy(str):
            @property
            def __class__(self):
                return claimed

            def __getattr__(self, name):
                reads.append(name)
                return getattr(target, name)

    elif kind == "getattribute":

        class Proxy(str):
            def __getattribute__(self, name):
                if name == "__class__":
                    return claimed
                reads.append(name)
                return getattr(target, name)

    elif kind == "enum-member":
        import enum

        class Proxy(str, enum.Enum):  # a genuine member of the caller's own enum
            REQUIREMENT = "requirement"

            def __getattribute__(self, name):
                if name == "__class__":
                    return claimed
                if name in ("__dict__", "__pydantic_fields_set__", "__pydantic_extra__"):
                    reads.append(name)
                    return getattr(target, name)
                return super().__getattribute__(name)

        return Proxy.REQUIREMENT
    else:

        class Proxy(str):
            @property
            def __class__(self):
                return claimed

            def __getattribute__(self, name):
                if name == "__class__":
                    return claimed
                reads.append(name)
                return getattr(target, name)

    return Proxy("requirement")


@pytest.mark.parametrize("kind", ["class-property", "getattribute", "both", "enum-member"])
def test_a_scalar_subclass_claiming_a_model_is_refused_before_its_delegated_graph_is_read(kind):
    """A ``str`` subclass passes as a scalar leaf; claiming a model class, it would hand validation a delegated
    graph that capture never charged.  It is refused at capture, and nothing is read through it."""

    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    delegate = command.stage_requirements[0]
    source = command.source_preconditions[0]
    set_native(delegate, "committed_sources", (source,) * 140_000)  # past the claim's discovery budget
    reads: list[str] = []
    set_native(command, "stage_requirements", (_scalar_proxy(kind, delegate, reads),))
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    with pytest.raises(ValueError, match="native"):
        contracts_lifecycle.validate_stage_receipts(command, (applied,))
    with pytest.raises(ValueError, match="native"):
        validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=(applied,))
    assert reads == []


def test_plain_scalar_subclasses_are_still_admitted_in_scalar_fields():
    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    class Digest(str):
        pass

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    set_native(command, "request_digest", Digest("sha256:abc"))
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    kept_intent, _kept = contracts_lifecycle.validate_stage_receipts(command, (applied,))
    assert kept_intent.request_digest == "sha256:abc"


def _large_enum(size: int):
    import enum

    return enum.Enum("Large", {f"M{index}": f"m{index}" for index in range(size)}, type=str)


def _lifecycle_lines(call) -> int:
    """Lines executed inside ``contracts_lifecycle`` while ``call`` runs: the module's own work, by count."""

    import sys

    module_file = contracts_lifecycle.__file__
    count = [0]

    def tracer(frame, event, arg):
        if frame.f_code.co_filename != module_file:
            return None

        def local(frame, event, arg):
            if event == "line":
                count[0] += 1
            return local

        return local(frame, event, arg)

    sys.settrace(tracer)
    try:
        call()
    finally:
        sys.settrace(None)
    return count[0]


def test_repeating_a_member_of_a_large_enum_does_not_rescan_its_members():
    """Membership is decided from an identity index built once per class and call: 400 repeats of the last
    member of a 500-member enum cost about the same as 400 repeats of a member of a 2-member enum."""

    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    def run(member) -> int:
        command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
        set_native(command, "request_digest", (member,) * 400)  # captured (and charged) before validation refuses it
        applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)

        def call():
            with pytest.raises(ValueError):
                contracts_lifecycle.validate_stage_receipts(command, (applied,))

        return _lifecycle_lines(call)

    small = run(_large_enum(2).M1)
    large = run(_large_enum(500).M499)
    assert large - small < 10_000, (small, large)  # a rescan per leaf is about 200,000 lines


def test_the_member_index_is_charged_to_the_discovery_budget_once_per_class():
    from contracts_limits import DiscoveryBudget

    large = _large_enum(500)
    budget = DiscoveryBudget(contracts_lifecycle.LIFECYCLE_TIER)
    contracts_lifecycle._capture_native_contract_graph((large.M499,) * 400, budget=budget)
    mro_length = len(type.__dict__["__mro__"].__get__(large))
    assert budget.edges == 1 + 400 + mro_length + 500  # the tuple, its items, one class walk, one member index


def test_an_enum_with_more_members_than_the_budget_allows_is_refused():
    from contracts_limits import DiscoveryBudget

    budget = DiscoveryBudget(contracts_lifecycle.LIFECYCLE_TIER)
    budget.charge_edge(budget.tier.max_edges - 10)
    with pytest.raises(ValueError, match=BUDGET):
        contracts_lifecycle._capture_native_contract_graph(_large_enum(20).M0, budget=budget)


def test_a_forged_instance_of_a_large_enum_is_still_refused_after_its_index_is_built():
    large = _large_enum(500)
    forged = str.__new__(large, "m499")
    shared: dict = {}
    contracts_lifecycle._capture_native_contract_graph(large.M499, captured_by_identity=shared)
    with pytest.raises(ValueError, match="declared member"):
        contracts_lifecycle._capture_native_contract_graph(forged, captured_by_identity=shared)


def _deep_str_class(depth: int) -> type:
    kind = str
    for _ in range(depth):
        kind = type("Deeper", (kind,), {})
    return kind


def test_repeating_a_leaf_of_a_deeply_derived_class_does_not_rewalk_its_mro():
    """A leaf's class is classified once per call (its MRO walked and charged once), not once per leaf."""

    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    def run(kind) -> int:
        command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
        set_native(command, "request_digest", (kind("sha256:abc"),) * 400)
        applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)

        def call():
            with pytest.raises(ValueError):
                contracts_lifecycle.validate_stage_receipts(command, (applied,))

        return _lifecycle_lines(call)

    shallow = run(_deep_str_class(1))
    deep = run(_deep_str_class(300))
    assert deep - shallow < 10_000, (shallow, deep)


def test_a_leaf_class_mro_is_charged_to_the_discovery_budget_once_per_class():
    from contracts_limits import DiscoveryBudget

    kind = _deep_str_class(300)
    mro_length = len(type.__dict__["__mro__"].__get__(kind))
    budget = DiscoveryBudget(contracts_lifecycle.LIFECYCLE_TIER)
    contracts_lifecycle._capture_native_contract_graph((kind("a"),) * 400, budget=budget)
    assert budget.edges == 1 + 400 + mro_length


def test_a_leaf_class_changed_by_a_view_is_classified_again_for_items_captured_after_it():
    """A class verdict kept in the call's memo must not outlive caller code: a receipts view that gives an
    admitted ``str`` subclass a ``__class__`` claim, then yields a receipt its backing never held, does not get
    that receipt's instance past capture on a stale verdict."""

    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    error = StageError(code="projection_failed", retryable=False, safe_message="Projection failed")
    reads: list[str] = []

    class Plain(str):
        pass

    def impersonate(self, name):
        if name == "__class__":
            return StageError
        if name in ("__dict__", "__pydantic_fields_set__", "__pydantic_extra__"):
            reads.append(name)
            return getattr(error, name)
        return str.__getattribute__(self, name)

    def receipts():
        first = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
        set_native(first, "effect_id", Plain(first.effect_id))  # admitted at inventory, its class verdict kept
        unseen = receipt(ProcessingStage.CANONICAL, StageStatus.FAILED)
        set_native(unseen, "error", Plain("error"))

        class Receipts(tuple):
            def __iter__(self):
                yield first
                Plain.__getattribute__ = impersonate  # caller code between items
                yield unseen  # only in the public view: captured after the class changed

        return Receipts((first,))

    try:
        with pytest.raises(ValueError, match="native"):
            contracts_lifecycle.validate_stage_receipts(command, receipts())
        del Plain.__getattribute__
        with pytest.raises(ValueError, match="native"):
            validate_required_stage_claim(claimed_status=IntentStatus.FAILED, intent=command, receipts=receipts())
    finally:
        if "__getattribute__" in vars(Plain):
            del Plain.__getattribute__
    assert reads == []


def test_an_enum_member_map_changed_by_a_view_is_indexed_again_for_items_captured_after_it():
    """The member index holds only until caller code runs: a member a receipts view removes from its class's
    map is no longer a declared member for a receipt captured after the view stepped."""

    import enum

    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    first = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    Ids = enum.Enum("Ids", {"INTENT": first.intent_id, "OTHER": "other"}, type=str)
    members = vars(Ids)["_member_map_"]
    set_native(first, "intent_id", Ids.INTENT)  # indexed at inventory

    def receipts():
        unseen = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
        set_native(unseen, "intent_id", Ids.INTENT)

        class Receipts(tuple):
            def __iter__(self):
                yield first
                del members["INTENT"]  # caller code between items
                yield unseen  # only in the public view: captured after the map changed

        return Receipts((first,))

    try:
        with pytest.raises(ValueError, match="declared member"):
            contracts_lifecycle.validate_stage_receipts(command, receipts())
        members["INTENT"] = Ids.INTENT
        with pytest.raises(ValueError, match="declared member"):
            validate_required_stage_claim(claimed_status=IntentStatus.APPLIED, intent=command, receipts=receipts())
    finally:
        members["INTENT"] = Ids.INTENT


class _VersionLookalike(str):
    """Spells something else, but hashes and compares like the schema version."""

    calls = 0

    def __hash__(self):
        type(self).calls += 1
        return str.__hash__("candidate-v1")

    def __eq__(self, other):
        type(self).calls += 1
        return True

    def __ne__(self, other):
        return False


@pytest.mark.parametrize("position", ["receipt", "intent"])
def test_a_text_subclass_cannot_pass_a_literal_it_does_not_spell(position):
    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    set_native(applied if position == "receipt" else command, "schema_version", _VersionLookalike("attacker"))
    _VersionLookalike.calls = 0

    with pytest.raises(ValueError, match="schema_version"):  # used to be accepted as "candidate-v1"
        contracts_lifecycle.validate_stage_receipts(command, (applied,))
    assert _VersionLookalike.calls == 0  # its hash and equality never ran


def test_a_text_subclass_that_spells_its_value_reaches_validation_as_plain_text():
    from tests.test_contracts_lifecycle import intent_with_requirements, requirement

    class Text(str):
        pass

    command = intent_with_requirements(requirement(ProcessingStage.CANONICAL))
    applied = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    set_native(applied, "schema_version", Text("candidate-v1"))
    set_native(applied, "effect_id", Text("effect-canonical"))

    _, (kept,) = contracts_lifecycle.validate_stage_receipts(command, (applied,))

    assert kept == contracts_lifecycle.validate_stage_receipts(command, (receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),))[1][0]
    assert type(kept.schema_version) is str and type(kept.effect_id) is str
