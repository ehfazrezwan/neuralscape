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
    assert len(made) == 1 and made[0].tier is contracts_limits.STANDARD  # one envelope for every root
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
        update={"id": [list(range(200)) for _ in range(900)]}
    )
    with pytest.raises(ValueError, match=BUDGET):
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=flood,
            receipts=(receipt(ProcessingStage.CANONICAL, StageStatus.PROCESSING),),
        )
