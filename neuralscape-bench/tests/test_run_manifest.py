"""Contract tests for reproducible, product-independent benchmark manifests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from abc import ABCMeta
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Callable

import pydantic
import pytest
from pydantic import ValidationError

import neuralscape_bench.run_manifest as run_manifest_module
from neuralscape_bench.run_manifest import (
    BuildIdentity,
    CacheCondition,
    ConcurrencySetting,
    FailureOutcome,
    LimitKind,
    MeasurementProvenance,
    ObservationKind,
    ResourceReading,
    RunManifest,
    RunState,
    SampleExclusion,
    TimingMeasurement,
    WorkloadIdentity,
    finish_run,
    serialize_run_manifest,
    validate_run_manifest_json,
)
from neuralscape_bench.run_manifest import (
    _freeze_model_graphs,
    _protect_yielded_model_storage,
    _snapshot_native,
    _traverse_model_storage,
)


VERSION = "candidate-v1"
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
BASE_MODEL_DICT_DESCRIPTOR = vars(pydantic.BaseModel)["__dict__"]
BASE_MODEL_EXTRA_DESCRIPTOR = vars(pydantic.BaseModel)["__pydantic_extra__"]


def test_standalone_import_and_validation_without_product_modules(tmp_path):
    bench_root = Path(__file__).resolve().parents[1]
    site_packages = Path(pydantic.__file__).resolve().parent.parent
    script = """
import builtins
import sys

# Make installed dependencies visible without running site initialization or
# processing editable .pth files from unrelated packages.
sys.path.append(sys.argv[1])

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "contracts_common" or name.startswith("contracts_common."):
        raise AssertionError("product contract import attempted")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from pydantic import ValidationError
from neuralscape_bench.run_manifest import ConcurrencySetting

valid = ConcurrencySetting(
    schema_version="candidate-v1",
    scope="standalone-driver",
    value=1,
)
assert valid.value == 1
try:
    ConcurrencySetting(
        schema_version="candidate-v1",
        scope="standalone-driver",
        value=True,
    )
except ValidationError:
    pass
else:
    raise AssertionError("strict counter accepted a boolean")
print("standalone validation passed")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(bench_root)
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(site_packages)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "standalone validation passed"


def build_identity() -> BuildIdentity:
    return BuildIdentity(
        schema_version=VERSION,
        git_commit="d" * 40,
        image_digest=SHA_A,
        dependency_lock_sha256=SHA_B,
    )


def workload_identity() -> WorkloadIdentity:
    return WorkloadIdentity(
        schema_version=VERSION,
        workload_id="synthetic-read-v1",
        workload_sha256=SHA_B,
        corpus_sha256=SHA_C,
    )


def concurrency() -> tuple[ConcurrencySetting, ...]:
    return (
        ConcurrencySetting(schema_version=VERSION, scope="driver", value=8),
        ConcurrencySetting(schema_version=VERSION, scope="api-process", value=2),
    )


def pending_memory() -> ResourceReading:
    return ResourceReading(
        schema_version=VERSION,
        scope="api-container",
        resource="memory",
        unit="bytes",
        limit_kind=LimitKind.BOUNDED,
        limit_value=536_870_912,
        observation_kind=ObservationKind.PENDING,
    )


def measured_memory(value: int = 402_653_184) -> ResourceReading:
    return ResourceReading(
        schema_version=VERSION,
        scope="api-container",
        resource="memory",
        unit="bytes",
        limit_kind=LimitKind.BOUNDED,
        limit_value=536_870_912,
        observation_kind=ObservationKind.MEASURED,
        observed_value=value,
    )


def timing(*, warmup: bool = True) -> TimingMeasurement:
    exclusions = (
        SampleExclusion(
            schema_version=VERSION,
            kind="warmup",
            count=2,
            reason="discarded before steady-state sampling",
        ),
    ) if warmup else ()
    return TimingMeasurement(
        schema_version=VERSION,
        name="exact-id-read",
        time_unit="milliseconds",
        attempted_sample_count=12 if warmup else 10,
        included_sample_count=10,
        exclusions=exclusions,
        provenance=MeasurementProvenance(
            schema_version=VERSION,
            collector="benchmark-driver",
            collector_version="1.0.0",
            clock="monotonic",
            placement="separate-driver-process",
        ),
    )


def planned_manifest(*, condition: CacheCondition = CacheCondition.WARM) -> RunManifest:
    return RunManifest(
        schema_version=VERSION,
        run_id="run-001",
        state=RunState.PLANNED,
        build=build_identity(),
        workload=workload_identity(),
        cache_condition=condition,
        concurrency=concurrency(),
        resources=(pending_memory(),),
    )


def completed_manifest() -> RunManifest:
    return finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=(measured_memory(),),
        measurements=(timing(),),
    )


def planned_manifest_with_extra_target(target_name: str) -> tuple[RunManifest, object]:
    manifest = planned_manifest()
    targets = {
        "manifest": manifest,
        "nested_resource": manifest.resources[0],
    }
    return manifest, targets[target_name]


class HiddenIterationExtras(dict[str, object]):
    """Expose stored entries through items(), but not ordinary iteration."""

    def __iter__(self):
        return iter(())


class FalseyPopulatedExtras(dict[str, object]):
    def __bool__(self):
        return False


class HiddenBackingExtras(dict[str, object]):
    """Hide native dict backing through every overridable public view."""

    def __init__(self, values: dict[str, object]):
        super().__init__(values)
        self.bool_calls = 0
        self.class_calls = 0
        self.len_calls = 0
        self.iteration_calls = 0
        self.keys_calls = 0
        self.items_calls = 0

    @property
    def __class__(self):
        self.class_calls += 1
        return object

    def __len__(self):
        self.len_calls += 1
        return 0

    def __bool__(self):
        self.bool_calls += 1
        return False

    def __iter__(self):
        self.iteration_calls += 1
        return iter(())

    def keys(self):
        self.keys_calls += 1
        return ()

    def items(self):
        self.items_calls += 1
        return ()


class HiddenItemsExtras(Mapping[str, object]):
    """Expose keys through iteration while hiding every items() pair."""

    def __init__(self, values: dict[str, object]):
        self.values = values
        self.iteration_calls = 0
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        self.iteration_calls += 1
        return iter(self.values)

    def items(self):
        self.items_calls += 1
        return ()


class ChangingItemsExtras(Mapping[str, object]):
    """Return a different entry inventory if a receiver consumes it twice."""

    def __init__(self):
        self.iteration_calls = 0
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __len__(self) -> int:
        return 0

    def __iter__(self):
        self.iteration_calls += 1
        return iter(())

    def items(self):
        self.items_calls += 1
        if self.items_calls == 1:
            return (("future_constraint", "first-view"),)
        return (("run_id", "second-view"),)


class DictEqualitySpoof:
    """Compare equal to dict if candidate-MRO membership consults it."""

    def __init__(self):
        self.calls = 0

    def __eq__(self, other: object) -> bool:
        self.calls += 1
        return other is dict


DICT_EQUALITY_SPOOF = DictEqualitySpoof()


class MroSpoofMeta(ABCMeta):
    def __getattribute__(cls, name: str):
        if name == "__mro__":
            return (dict,)
        return super().__getattribute__(name)


class EqualitySpoofMeta(ABCMeta):
    def __getattribute__(cls, name: str):
        if name == "__mro__":
            return (DICT_EQUALITY_SPOOF,)
        return super().__getattribute__(name)


class MetaclassSpoofExtras(Mapping[str, object]):
    def __init__(self, values: dict[str, object]):
        self.values = values
        self.iteration_calls = 0
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        self.iteration_calls += 1
        return iter(self.values)

    def items(self):
        self.items_calls += 1
        return tuple(self.values.items())


class MroSpoofExtras(MetaclassSpoofExtras, metaclass=MroSpoofMeta):
    pass


class EqualitySpoofExtras(MetaclassSpoofExtras, metaclass=EqualitySpoofMeta):
    pass


class HiddenModelBacking(dict[str, object]):
    """Retain native entries while hiding them from overridable iteration."""

    def __init__(self, values: dict[str, object]):
        super().__init__(values)
        self.iteration_calls = 0

    def __iter__(self):
        self.iteration_calls += 1
        return (
            key for key in dict.keys(self) if key != "future_constraint"
        )


class HiddenStorageViewManifest(RunManifest):
    """Hide native stored and extra entries from instance-dispatched views."""

    def __getattribute__(self, name: str):
        if name == "__dict__":
            native = object.__getattribute__(self, "__dict__")
            return {
                key: value
                for key, value in dict.items(native)
                if key != "future_constraint"
            }
        if name == "__pydantic_extra__":
            return None
        return super().__getattribute__(name)


class DescriptorMaskedRunManifest(RunManifest):
    """Hide actual Pydantic storage behind subclass data descriptors."""

    @property
    def __dict__(self):
        native = BASE_MODEL_DICT_DESCRIPTOR.__get__(self, pydantic.BaseModel)
        return {
            key: value
            for key, value in dict.items(native)
            if key != "future_constraint"
        }

    @__dict__.setter
    def __dict__(self, value):
        BASE_MODEL_DICT_DESCRIPTOR.__set__(self, value)

    @property
    def __pydantic_extra__(self):
        return None

    @__pydantic_extra__.setter
    def __pydantic_extra__(self, value):
        BASE_MODEL_EXTRA_DESCRIPTOR.__set__(self, value)


class DescriptorMaskedResourceReading(ResourceReading):
    """Nested counterpart to DescriptorMaskedRunManifest."""

    @property
    def __dict__(self):
        native = BASE_MODEL_DICT_DESCRIPTOR.__get__(self, pydantic.BaseModel)
        return {
            key: value
            for key, value in dict.items(native)
            if key != "future_constraint"
        }

    @__dict__.setter
    def __dict__(self, value):
        BASE_MODEL_DICT_DESCRIPTOR.__set__(self, value)

    @property
    def __pydantic_extra__(self):
        return None

    @__pydantic_extra__.setter
    def __pydantic_extra__(self, value):
        BASE_MODEL_EXTRA_DESCRIPTOR.__set__(self, value)


class MutatingTuple(tuple[object, ...]):
    """Apply one model-storage mutation when native tuple traversal starts."""

    def __new__(
        cls,
        values: tuple[object, ...],
        action: Callable[[], None],
    ):
        instance = super().__new__(cls, values)
        instance.action = action
        return instance

    def __iter__(self):
        action = self.action
        self.action = lambda: None
        action()
        return super().__iter__()


class PublicViewTuple(tuple[object, ...]):
    """Expose a distinct, authoritative public tuple view exactly once."""

    def __new__(
        cls,
        values: tuple[object, ...],
        public_values: tuple[object, ...],
        action: Callable[[], None] | None = None,
    ):
        instance = super().__new__(cls, values)
        instance.public_values = public_values
        instance.action = action
        instance.iteration_calls = 0
        return instance

    def __iter__(self):
        self.iteration_calls += 1
        if self.action is not None:
            self.action()
        return iter(self.public_values)


class EventPublicTuple(tuple[object, ...]):
    """Expose a passive public tuple view and record observation order."""

    def __new__(
        cls,
        values: tuple[object, ...],
        public_values: tuple[object, ...],
        events: list[str],
        name: str,
    ):
        instance = super().__new__(cls, values)
        instance.public_values = public_values
        instance.events = events
        instance.name = name
        instance.iteration_calls = 0
        return instance

    def __iter__(self):
        self.iteration_calls += 1
        self.events.append(f"{self.name}:iter")
        return iter(self.public_values)


class IteratorAcquisitionErrorTuple(tuple[object, ...]):
    """Raise one stable ordinary error while acquiring the public iterator."""

    def __new__(cls, _prefix: object):
        instance = super().__new__(cls, ())
        instance.iteration_calls = 0
        instance.error = RuntimeError(
            "ordinary public sequence iterator acquisition failed"
        )
        return instance

    def __iter__(self):
        self.iteration_calls += 1
        raise self.error


class IteratorAdvancementErrorTuple(tuple[object, ...]):
    """Yield one public value, then raise one stable ordinary error."""

    def __new__(cls, prefix: object):
        instance = super().__new__(cls, ())
        instance.prefix = prefix
        instance.iteration_calls = 0
        instance.advancement_calls = 0
        instance.error = RuntimeError(
            "ordinary public sequence iterator advancement failed"
        )
        return instance

    def __iter__(self):
        self.iteration_calls += 1

        def generate():
            self.advancement_calls += 1
            yield self.prefix
            self.advancement_calls += 1
            raise self.error

        return generate()


class IteratorAcquisitionErrorList(list[object]):
    """Raise one stable ordinary error while acquiring the public iterator."""

    def __init__(self, _prefix: object):
        super().__init__()
        self.iteration_calls = 0
        self.error = RuntimeError(
            "ordinary public list iterator acquisition failed"
        )

    def __iter__(self):
        self.iteration_calls += 1
        raise self.error


class IteratorAdvancementErrorList(list[object]):
    """Yield one public value, then raise one stable ordinary error."""

    def __init__(self, prefix: object):
        super().__init__()
        self.prefix = prefix
        self.iteration_calls = 0
        self.advancement_calls = 0
        self.error = RuntimeError(
            "ordinary public list iterator advancement failed"
        )

    def __iter__(self):
        self.iteration_calls += 1

        def generate():
            self.advancement_calls += 1
            yield self.prefix
            self.advancement_calls += 1
            raise self.error

        return generate()


class PassiveProjectionKey(str):
    """Count ordinary hashing when a retained mapping prefix is projected."""

    def __new__(cls, value: str):
        instance = super().__new__(cls, value)
        instance.hash_calls = 0
        return instance

    def __hash__(self) -> int:
        self.hash_calls += 1
        return str.__hash__(self)


class PublicViewList(list[object]):
    """Keep hidden native list backing distinct from a passive public view."""

    def __init__(
        self,
        values: list[object],
        public_values: list[object],
    ):
        super().__init__(values)
        self.public_values = public_values
        self.iteration_calls = 0

    def __iter__(self):
        self.iteration_calls += 1
        return iter(self.public_values)


class PublicResourceWithDeepNativeBacking(dict[str, object]):
    """Expose a valid resource while retaining a deep native-only graph."""

    def __init__(self, depth: int):
        hidden: dict[str, object] = {"leaf": "value"}
        for index in range(depth):
            hidden = {f"level-{index}": hidden}
        super().__init__(hidden=hidden)
        self.public_values = measured_memory().model_dump(mode="python")
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        return self.public_values[key]

    def __len__(self) -> int:
        return len(self.public_values)

    def __iter__(self):
        return iter(self.public_values)

    def items(self):
        self.items_calls += 1
        return self.public_values.items()


class YieldRepairMapping(Mapping[str, object]):
    """Repair the first yielded child when the next pair is requested."""

    def __init__(
        self,
        values: dict[str, object],
        action: Callable[[], None],
    ):
        self.values = values
        self.action = action
        self.items_calls = 0
        self.callback_calls = 0
        self.events: list[str] = []

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def items(self):
        self.items_calls += 1
        self.events.append("items")

        def generate():
            self.events.append("yield:exclusions")
            yield "exclusions", self.values["exclusions"]
            self.events.append("callback")
            self.callback_calls += 1
            self.action()
            for key, value in self.values.items():
                if key != "exclusions":
                    self.events.append(f"yield:{key}")
                    yield key, value

        return generate()


class LazyProvenanceMapping(Mapping[str, object]):
    """Expose provenance before the remaining measurement fields."""

    def __init__(self, values: dict[str, object]):
        self.values = values
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def items(self):
        self.items_calls += 1

        def generate():
            yield "provenance", self.values["provenance"]
            for key, value in self.values.items():
                if key != "provenance":
                    yield key, value

        return generate()


class InventoryMapping(Mapping[object, object]):
    """Expose a benign, counted public pair stream for inventory tests."""

    def __init__(
        self,
        pairs: tuple[tuple[object, ...], ...],
        *,
        name: str = "mapping",
        events: list[str] | None = None,
    ):
        self.pairs = pairs
        self.name = name
        self.events = events if events is not None else []
        self.items_calls = 0
        self.yielded_indexes: list[int] = []

    def __getitem__(self, key: object) -> object:
        for pair in self.pairs:
            if len(pair) == 2 and pair[0] == key:
                return pair[1]
        raise KeyError(key)

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return (pair[0] for pair in self.pairs if pair)

    def items(self):
        self.items_calls += 1
        self.events.append(f"{self.name}:items")

        def generate():
            for index, pair in enumerate(self.pairs):
                self.yielded_indexes.append(index)
                self.events.append(f"{self.name}:yield:{index}")
                yield pair

        return generate()


def dual_interface_timing(
    provenance: Mapping[object, object],
) -> tuple[TimingMeasurement, dict[str, int]]:
    """Build a model whose Mapping view must not outrank model storage."""

    calls = {"getitem": 0, "iter": 0, "len": 0, "items": 0}

    class DualInterfaceTimingMeasurement(
        TimingMeasurement, Mapping[str, object]
    ):
        def __getitem__(self, key: str) -> object:
            calls["getitem"] += 1
            return getattr(self, key)

        def __iter__(self):
            calls["iter"] += 1
            return iter(type(self).model_fields)

        def __len__(self) -> int:
            calls["len"] += 1
            return len(type(self).model_fields)

        def items(self):
            calls["items"] += 1
            return (
                (name, getattr(self, name))
                for name in type(self).model_fields
            )

    measurement = DualInterfaceTimingMeasurement.model_validate(
        timing().model_dump(mode="python")
    ).model_copy(update={"provenance": provenance})
    return measurement, calls


class PublicItemsDict(dict[object, object]):
    """Keep native dict backing distinct from its public items authority."""

    def __init__(
        self,
        native: dict[object, object],
        public_pairs: tuple[tuple[object, object], ...],
    ):
        super().__init__(native)
        self.public_pairs = public_pairs
        self.items_calls = 0

    def items(self):
        self.items_calls += 1
        return iter(self.public_pairs)


class FailingItemsMapping(Mapping[object, object]):
    def __init__(self):
        self.items_calls = 0

    def __getitem__(self, key: object) -> object:
        raise KeyError(key)

    def __len__(self) -> int:
        return 0

    def __iter__(self):
        return iter(())

    def items(self):
        self.items_calls += 1
        raise RuntimeError("ordinary public mapping observation failed")


class FailingIteratorMapping(Mapping[object, object]):
    def __init__(self):
        self.items_calls = 0
        self.yielded = 0

    def __getitem__(self, key: object) -> object:
        raise KeyError(key)

    def __len__(self) -> int:
        return 0

    def __iter__(self):
        return iter(())

    def items(self):
        self.items_calls += 1

        def generate():
            self.yielded += 1
            yield "prefix", 1
            raise RuntimeError("ordinary public mapping iterator failed")

        return generate()


class ArmedFieldName(str):
    """Run one callback when a stored field name is next hashed."""

    def __new__(cls, value: str):
        instance = super().__new__(cls, value)
        instance.action = None
        instance.hash_calls = 0
        return instance

    def arm(self, action: Callable[[], None]) -> None:
        self.action = action

    def __hash__(self) -> int:
        if self.action is not None:
            action = self.action
            self.action = None
            self.hash_calls += 1
            action()
        return str.__hash__(self)


def receive_planned_manifest(boundary: str, manifest: RunManifest) -> RunManifest:
    if boundary == "serialize":
        return validate_run_manifest_json(serialize_run_manifest(manifest))
    return finish_run(
        manifest,
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=(measured_memory(),),
        measurements=(timing(),),
    )


def descriptor_masked_manifest(
    target_name: str,
) -> tuple[RunManifest, RunManifest | ResourceReading]:
    if target_name == "manifest":
        manifest = DescriptorMaskedRunManifest.model_validate(
            planned_manifest().model_dump()
        )
        return manifest, manifest

    resource = DescriptorMaskedResourceReading.model_validate(
        pending_memory().model_dump()
    )
    manifest = planned_manifest().model_copy(update={"resources": (resource,)})
    return manifest, resource


def assert_received(boundary: str, manifest: RunManifest) -> None:
    received = receive_planned_manifest(boundary, manifest)
    expected_state = (
        RunState.PLANNED if boundary == "serialize" else RunState.COMPLETED
    )
    assert received.state is expected_state


def test_replay_inventory_retains_first_entries_and_strong_owners():
    class ReplayDict(dict[str, object]):
        pass

    class ReplayList(list[object]):
        pass

    class ReplayTuple(tuple[object, ...]):
        pass

    dict_child: dict[str, object] = {}
    list_child: dict[str, object] = {}
    tuple_child: dict[str, object] = {}
    dict_parent = ReplayDict(child=dict_child)
    list_parent = ReplayList((list_child,))
    tuple_parent = ReplayTuple((tuple_child,))
    manifest = planned_manifest()

    frozen = _freeze_model_graphs(
        manifest,
        dict_parent,
        list_parent,
        tuple_parent,
    )

    assert frozen.replay_entries[id(manifest)][0] is manifest
    assert manifest.build in frozen.replay_entries[id(manifest)][1]
    assert frozen.replay_entries[id(dict_parent)] == (
        dict_parent,
        (dict_child,),
    )
    assert frozen.replay_entries[id(list_parent)] == (
        list_parent,
        (list_child,),
    )
    assert frozen.replay_entries[id(tuple_parent)] == (
        tuple_parent,
        (tuple_child,),
    )

    dict_parent.clear()
    list_parent.clear()
    visited: set[int] = set()
    _traverse_model_storage(dict_parent, frozen, visited)
    _traverse_model_storage(list_parent, frozen, visited)
    _traverse_model_storage(tuple_parent, frozen, visited)
    _traverse_model_storage(manifest, frozen, visited)

    assert id(dict_child) in visited
    assert id(list_child) in visited
    assert id(tuple_child) in visited
    assert id(manifest.build) in visited
    assert frozen.replay_entries[id(dict_parent)][1] == (dict_child,)
    assert frozen.replay_entries[id(list_parent)][1] == (list_child,)


def test_yielded_child_guard_traverses_captured_storage_then_reuses_completion():
    class ReplayDict(dict[str, object]):
        pass

    child: dict[str, object] = {}
    value = ReplayDict(child=child)
    frozen = _freeze_model_graphs(value)

    assert frozen.replay_entries[id(value)][0] is value
    assert id(value) not in frozen.completed_traversals
    assert _protect_yielded_model_storage(value, frozen) is True
    assert frozen.completed_traversals[id(value)] is value
    assert frozen.completed_traversals[id(child)] is child

    value.clear()
    assert _protect_yielded_model_storage(value, frozen) is False
    assert frozen.replay_entries[id(value)] == (value, (child,))

    visited: set[int] = set()
    _traverse_model_storage(value, frozen, visited)
    assert id(child) in visited
    assert _protect_yielded_model_storage("ordinary scalar", frozen) is False


def test_yielded_child_guard_rejects_wrong_capture_and_completion_owners():
    class ReplayDict(dict[str, object]):
        pass

    child: dict[str, object] = {}
    value = ReplayDict(child=child)
    impostor = ReplayDict()
    frozen = _freeze_model_graphs(value)
    frozen.completed_traversals[id(value)] = impostor

    assert _protect_yielded_model_storage(value, frozen) is True
    assert frozen.completed_traversals[id(value)] is value

    replacement = ReplayDict(child=child)
    replacement_graph = _freeze_model_graphs(())
    replacement_graph.replay_entries[id(replacement)] = (impostor, ())
    replacement_graph.completed_traversals[id(replacement)] = replacement

    assert _protect_yielded_model_storage(replacement, replacement_graph) is True
    owner, entries = replacement_graph.replay_entries[id(replacement)]
    assert owner is replacement
    assert entries == (child,)
    assert replacement_graph.completed_traversals[id(replacement)] is replacement


def test_failed_yielded_child_traversal_is_not_marked_complete():
    deep: list[object] = []
    for _index in range(sys.getrecursionlimit() + 100):
        deep = [deep]
    alias: dict[str, object] = {}
    value: list[object] = [alias, deep, alias]
    frozen = _freeze_model_graphs(value)

    with pytest.raises(RecursionError, match="maximum recursion depth exceeded"):
        _protect_yielded_model_storage(value, frozen)

    assert frozen.completed_traversals == {}


def test_successful_alias_and_cycle_traversals_record_exact_owners():
    child: dict[str, object] = {}
    aliased: list[object] = [child, child]
    cycle: list[object] = []
    cycle.append(cycle)
    frozen = _freeze_model_graphs(aliased, cycle)

    assert _protect_yielded_model_storage(aliased, frozen) is True
    assert frozen.completed_traversals[id(aliased)] is aliased
    assert frozen.completed_traversals[id(child)] is child
    assert _protect_yielded_model_storage(cycle, frozen) is True
    assert frozen.completed_traversals[id(cycle)] is cycle
    assert _protect_yielded_model_storage(aliased, frozen) is False
    assert _protect_yielded_model_storage(cycle, frozen) is False


def test_public_mapping_inventory_replays_one_recursive_alias_projection():
    events: list[str] = []
    inner = InventoryMapping((("leaf", 1),), name="inner", events=events)
    outer = InventoryMapping(
        (("inner", inner), ("tail", 2)),
        name="outer",
        events=events,
    )
    proxy = MappingProxyType({"outer": outer})
    value = {"proxy": proxy, "alias": outer}

    frozen = _freeze_model_graphs(value)

    assert outer.items_calls == 1
    assert inner.items_calls == 1
    assert events.index("inner:items") < events.index("outer:yield:1")
    assert frozen.public_mapping_owners[id(outer)] is outer
    assert frozen.public_mappings[id(outer)].owner is outer
    assert frozen.public_mapping_queue == [proxy, outer, inner]

    assert _snapshot_native(value, _frozen_graph=frozen) == {
        "proxy": {"outer": {"inner": {"leaf": 1}, "tail": 2}},
        "alias": {"inner": {"leaf": 1}, "tail": 2},
    }
    assert outer.items_calls == 1
    assert inner.items_calls == 1


def test_public_mapping_queue_visits_each_retained_owner_once():
    mappings = tuple(
        InventoryMapping((), name=f"mapping-{index}") for index in range(32)
    )

    frozen = _freeze_model_graphs(mappings)

    assert frozen.public_mapping_queue == list(mappings)
    assert len(frozen.public_mapping_owners) == len(mappings)
    assert len(frozen.public_mappings) == len(mappings)
    assert sum(mapping.items_calls for mapping in mappings) == len(mappings)


def test_hidden_native_alias_does_not_create_a_public_mapping_cycle():
    mapping = InventoryMapping(())
    child = PublicViewList([mapping], ["public-value"])
    mapping.pairs = (("child", child),)

    frozen = _freeze_model_graphs(mapping)

    assert frozen.public_mappings[id(mapping)].complete is True
    assert frozen.public_sequences[id(child)].complete is True
    assert _snapshot_native(mapping, _frozen_graph=frozen) == {
        "child": ["public-value"]
    }
    assert mapping.items_calls == 1
    assert child.iteration_calls == 1


@pytest.mark.parametrize("invalid", [False, True], ids=["valid", "invalid"])
def test_mapping_inventory_observes_yielded_public_tuple_before_parent_advance(
    invalid: bool,
):
    events: list[str] = []
    exclusion = timing().exclusions[0]
    if invalid:
        exclusion = exclusion.model_copy(update={"count": True})
    public_exclusions = EventPublicTuple(
        (),
        (exclusion,),
        events,
        "exclusions",
    )
    values = timing().model_dump(mode="python")
    values["exclusions"] = public_exclusions
    ordered_pairs = (("exclusions", public_exclusions),) + tuple(
        (key, item) for key, item in values.items() if key != "exclusions"
    )
    measurement = InventoryMapping(
        ordered_pairs,
        name="measurement",
        events=events,
    )

    if invalid:
        with pytest.raises(ValidationError) as exc_info:
            finish_run(
                planned_manifest(),
                state=RunState.COMPLETED,
                started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
                finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
                resources=(measured_memory(),),
                measurements=(measurement,),
            )
        assert exc_info.value.errors()[0]["loc"] == (
            "measurements",
            0,
            "exclusions",
            0,
            "count",
        )
    else:
        result = finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(measurement,),
        )
        assert result.state is RunState.COMPLETED

    assert events.index("exclusions:iter") < events.index(
        "measurement:yield:1"
    )
    assert measurement.items_calls == 1
    assert public_exclusions.iteration_calls == 1


def test_mapping_inventory_completes_nested_public_tuple_before_parent_advance():
    events: list[str] = []
    public_exclusions = EventPublicTuple(
        (),
        timing().exclusions,
        events,
        "nested-exclusions",
    )
    measurement = timing().model_copy(
        update={"exclusions": public_exclusions}
    )
    mapping = InventoryMapping(
        (("measurement", measurement), ("tail", 1)),
        name="parent",
        events=events,
    )

    frozen = _freeze_model_graphs(mapping)

    assert events.index("nested-exclusions:iter") < events.index(
        "parent:yield:1"
    )
    assert _snapshot_native(mapping, _frozen_graph=frozen)["tail"] == 1
    assert public_exclusions.iteration_calls == 1


@pytest.mark.parametrize(
    ("sequence_type", "retains_prefix", "expected_advancement_calls"),
    [
        (IteratorAcquisitionErrorTuple, False, 0),
        (IteratorAdvancementErrorTuple, True, 2),
        (IteratorAcquisitionErrorList, False, 0),
        (IteratorAdvancementErrorList, True, 2),
    ],
    ids=[
        "tuple-iterator-acquisition",
        "tuple-iterator-advancement",
        "list-iterator-acquisition",
        "list-iterator-advancement",
    ],
)
def test_public_sequence_iterator_failure_is_retained_and_replayed_once(
    sequence_type,
    retains_prefix: bool,
    expected_advancement_calls: int,
):
    projection_key = PassiveProjectionKey("projected")
    prefix = InventoryMapping(((projection_key, "prefix-value"),))
    sequence = sequence_type(prefix)
    primary = InventoryMapping((("sequence", sequence),), name="primary")
    alias = InventoryMapping((("sequence", sequence),), name="alias")

    frozen = _freeze_model_graphs(primary, alias)

    retained = frozen.public_sequences[id(sequence)]
    assert retained.owner is sequence
    assert retained.entries == ((prefix,) if retains_prefix else ())
    assert retained.failure is sequence.error
    assert retained.complete is False
    assert sequence.iteration_calls == 1
    assert getattr(sequence, "advancement_calls", 0) == (
        expected_advancement_calls
    )
    assert primary.items_calls == 1
    assert alias.items_calls == 0
    assert prefix.items_calls == (1 if retains_prefix else 0)
    assert projection_key.hash_calls == 0
    assert _snapshot_native(
        {"safe": "value"}, _frozen_graph=frozen
    ) == {"safe": "value"}
    assert projection_key.hash_calls == 0

    with pytest.raises(RuntimeError) as alias_error:
        _snapshot_native(alias, _frozen_graph=frozen)
    assert projection_key.hash_calls == (1 if retains_prefix else 0)
    with pytest.raises(RuntimeError) as primary_error:
        _snapshot_native(primary, _frozen_graph=frozen)

    assert alias_error.value is sequence.error
    assert primary_error.value is sequence.error
    assert sequence.iteration_calls == 1
    assert getattr(sequence, "advancement_calls", 0) == (
        expected_advancement_calls
    )
    assert primary.items_calls == 1
    assert alias.items_calls == 1
    assert prefix.items_calls == (1 if retains_prefix else 0)
    assert projection_key.hash_calls == (2 if retains_prefix else 0)


@pytest.mark.parametrize(
    "sequence_type",
    [
        IteratorAcquisitionErrorTuple,
        IteratorAdvancementErrorTuple,
        IteratorAcquisitionErrorList,
        IteratorAdvancementErrorList,
    ],
    ids=[
        "tuple-iterator-acquisition",
        "tuple-iterator-advancement",
        "list-iterator-acquisition",
        "list-iterator-advancement",
    ],
)
def test_public_sequence_iterator_failure_preserves_receiver_error_order(
    sequence_type,
):
    def measurement_with(sequence):
        values = timing().model_dump(mode="python")
        values["exclusions"] = sequence
        return InventoryMapping(tuple(values.items()))

    invalid_planned = planned_manifest()
    invalid_concurrency = invalid_planned.concurrency[0].model_copy(
        update={"value": True}
    )
    invalid_planned = invalid_planned.model_copy(
        update={"concurrency": (invalid_concurrency,)}
    )
    deferred_sequence = sequence_type(timing().exclusions[0])
    deferred_measurement = measurement_with(deferred_sequence)

    with pytest.raises(ValidationError) as invalid_error:
        finish_run(
            invalid_planned,
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(deferred_measurement,),
        )

    assert invalid_error.value.errors()[0]["loc"] == (
        "concurrency",
        0,
        "value",
    )
    assert deferred_sequence.iteration_calls == 1
    assert deferred_measurement.items_calls == 1

    reached_sequence = sequence_type(timing().exclusions[0])
    reached_measurement = measurement_with(reached_sequence)
    with pytest.raises(RuntimeError) as reached_error:
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(reached_measurement,),
        )

    assert reached_error.value is reached_sequence.error
    assert reached_sequence.iteration_calls == 1
    assert reached_measurement.items_calls == 1


@pytest.mark.parametrize("boundary", ["mapping", "sequence"])
def test_recursive_public_shape_failure_is_deferred_and_replayed_once(
    boundary: str,
    monkeypatch,
):
    sentinel = object()
    failure = RuntimeError("ordinary recursive public-shape helper failed")
    failure_calls = 0
    original = run_manifest_module._inventory_yielded_public_shape

    def inject_failure(value, frozen_graph, observing):
        nonlocal failure_calls
        if value is sentinel:
            failure_calls += 1
            raise failure
        return original(value, frozen_graph, observing)

    monkeypatch.setattr(
        run_manifest_module,
        "_inventory_yielded_public_shape",
        inject_failure,
    )
    projection_key = PassiveProjectionKey("projected")
    prefix = InventoryMapping(((projection_key, "prefix-value"),))
    if boundary == "mapping":
        failing_child = InventoryMapping(
            (("prefix", prefix), ("terminal", sentinel)),
            name="failing-child",
        )
    else:
        failing_child = EventPublicTuple(
            (),
            (prefix, sentinel),
            [],
            "failing-child",
        )
    primary = InventoryMapping((("child", failing_child),), name="primary")
    alias = InventoryMapping((("child", failing_child),), name="alias")

    frozen = _freeze_model_graphs(primary, alias)

    if boundary == "mapping":
        retained = frozen.public_mappings[id(failing_child)]
        assert failing_child.items_calls == 1
    else:
        retained = frozen.public_sequences[id(failing_child)]
        assert failing_child.iteration_calls == 1
    assert retained.owner is failing_child
    assert retained.entries == (
        (("prefix", prefix), ("terminal", sentinel))
        if boundary == "mapping"
        else (prefix, sentinel)
    )
    assert retained.failure is failure
    assert retained.complete is False
    assert failure_calls == 1
    assert prefix.items_calls == 1
    assert projection_key.hash_calls == 0
    assert primary.items_calls == 1
    assert alias.items_calls == 0
    assert _snapshot_native(
        {"safe": "value"}, _frozen_graph=frozen
    ) == {"safe": "value"}

    with pytest.raises(RuntimeError) as alias_error:
        _snapshot_native(alias, _frozen_graph=frozen)
    assert projection_key.hash_calls == 1
    with pytest.raises(RuntimeError) as primary_error:
        _snapshot_native(primary, _frozen_graph=frozen)

    assert alias_error.value is failure
    assert primary_error.value is failure
    assert failure_calls == 1
    assert prefix.items_calls == 1
    assert projection_key.hash_calls == 2
    assert primary.items_calls == 1
    assert alias.items_calls == 1
    if boundary == "mapping":
        assert failing_child.items_calls == 1
    else:
        assert failing_child.iteration_calls == 1


@pytest.mark.parametrize("boundary", ["mapping", "sequence"])
def test_recursive_public_shape_failure_preserves_receiver_error_order(
    boundary: str,
    monkeypatch,
):
    sentinel = object()
    failure = RuntimeError("ordinary recursive public-shape helper failed")
    failure_calls = 0
    original = run_manifest_module._inventory_yielded_public_shape

    def inject_failure(value, frozen_graph, observing):
        nonlocal failure_calls
        if value is sentinel:
            failure_calls += 1
            raise failure
        return original(value, frozen_graph, observing)

    monkeypatch.setattr(
        run_manifest_module,
        "_inventory_yielded_public_shape",
        inject_failure,
    )
    prefix = InventoryMapping((("prefix", "value"),))
    if boundary == "mapping":
        failing_child = InventoryMapping(
            (("prefix", prefix), ("terminal", sentinel))
        )
    else:
        failing_child = PublicViewList([], [prefix, sentinel])
    values = timing().model_dump(mode="python")
    values["exclusions"] = failing_child
    measurement = InventoryMapping(tuple(values.items()))
    invalid_planned = planned_manifest()
    invalid_concurrency = invalid_planned.concurrency[0].model_copy(
        update={"value": True}
    )
    invalid_planned = invalid_planned.model_copy(
        update={"concurrency": (invalid_concurrency,)}
    )

    with pytest.raises(ValidationError) as invalid_error:
        finish_run(
            invalid_planned,
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(measurement,),
        )

    assert invalid_error.value.errors()[0]["loc"] == (
        "concurrency",
        0,
        "value",
    )
    assert failure_calls == 1
    assert measurement.items_calls == 1

    with pytest.raises(RuntimeError) as reached_error:
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(measurement,),
        )

    assert reached_error.value is failure
    assert failure_calls == 2
    assert measurement.items_calls == 2
    assert prefix.items_calls == 2
    if boundary == "mapping":
        assert failing_child.items_calls == 2
    else:
        assert failing_child.iteration_calls == 2


def test_model_reachable_public_mappings_validate_from_one_inventory():
    resource = MappingProxyType(measured_memory().model_dump(mode="python"))
    measurement = InventoryMapping(
        tuple(
            (key, item)
            for key, item in timing().model_dump(mode="python").items()
        )
    )
    manifest = completed_manifest().model_copy(
        update={
            "resources": (resource,),
            "measurements": (measurement,),
        }
    )

    received = validate_run_manifest_json(serialize_run_manifest(manifest))

    assert received.resources[0] == measured_memory()
    assert received.measurements[0] == timing()
    assert measurement.items_calls == 1


def test_dual_interface_model_prefers_storage_and_inventories_mapping_child():
    events: list[str] = []
    expected = timing()
    provenance = InventoryMapping(
        tuple(expected.provenance.model_dump(mode="python").items()),
        name="provenance",
        events=events,
    )
    measurement, mapping_calls = dual_interface_timing(provenance)
    manifest = completed_manifest().model_copy(
        update={"measurements": (measurement,)}
    )

    received = validate_run_manifest_json(serialize_run_manifest(manifest))

    assert received.measurements[0] == expected
    assert mapping_calls == {"getitem": 0, "iter": 0, "len": 0, "items": 0}
    assert provenance.items_calls == 1


def test_parent_yielded_dual_interface_model_keeps_model_precedence():
    events: list[str] = []
    expected = timing()
    provenance = InventoryMapping(
        tuple(expected.provenance.model_dump(mode="python").items()),
        name="provenance",
        events=events,
    )
    measurement, mapping_calls = dual_interface_timing(provenance)
    parent = InventoryMapping(
        (("measurement", measurement), ("tail", 1)),
        name="parent",
        events=events,
    )

    frozen = _freeze_model_graphs(parent)
    snapshot = _snapshot_native(parent, _frozen_graph=frozen)

    assert snapshot["measurement"]["provenance"] == (
        expected.provenance.model_dump(mode="python")
    )
    assert snapshot["tail"] == 1
    assert mapping_calls == {"getitem": 0, "iter": 0, "len": 0, "items": 0}
    assert id(measurement) not in frozen.public_mapping_owners
    assert frozen.public_mapping_owners[id(provenance)] is provenance
    assert provenance.items_calls == 1
    assert events.index("provenance:items") < events.index("parent:yield:1")


def test_dict_subclass_uses_public_inventory_but_exact_dict_uses_native_entries():
    public = PublicItemsDict(
        {"native": "backing"},
        (("public", "projection"),),
    )
    exact = {"native": "authority"}
    value = {"public": public, "exact": exact}

    frozen = _freeze_model_graphs(value)

    assert frozen.replay_entries[id(public)] == (public, ("backing",))
    assert frozen.public_mappings[id(public)].entries == (
        ("public", "projection"),
    )
    assert id(exact) not in frozen.public_mapping_owners
    assert _snapshot_native(value, _frozen_graph=frozen) == {
        "public": {"public": "projection"},
        "exact": {"native": "authority"},
    }
    assert public.items_calls == 1


def test_public_mapping_inventory_defers_malformed_pair_and_cycle_diagnostics():
    malformed = InventoryMapping(
        (("prefix", 1), ("malformed",), ("unreached", 3)),
    )
    malformed_graph = _freeze_model_graphs(malformed)

    assert malformed.items_calls == 1
    assert malformed.yielded_indexes == [0, 1]
    assert malformed_graph.public_mappings[id(malformed)].entries == (
        ("prefix", 1),
    )
    with pytest.raises(ValueError, match="not enough values to unpack"):
        _snapshot_native(malformed, _frozen_graph=malformed_graph)

    terminal = FailingIteratorMapping()
    terminal_graph = _freeze_model_graphs(terminal)

    assert terminal.items_calls == 1
    assert terminal.yielded == 1
    assert terminal_graph.public_mappings[id(terminal)].entries == (
        ("prefix", 1),
    )
    with pytest.raises(RuntimeError, match="public mapping iterator failed"):
        _snapshot_native(terminal, _frozen_graph=terminal_graph)

    unhashable_key: list[object] = []
    deferred_key = InventoryMapping(((unhashable_key, 1),))
    deferred_key_graph = _freeze_model_graphs(deferred_key)

    assert deferred_key.items_calls == 1
    with pytest.raises(TypeError, match="unhashable type"):
        _snapshot_native(deferred_key, _frozen_graph=deferred_key_graph)

    cycle = InventoryMapping(())
    cycle.pairs = (("self", cycle), ("unreached", 2))
    cycle_graph = _freeze_model_graphs(cycle)

    assert cycle.items_calls == 1
    assert cycle.yielded_indexes == [0]
    assert cycle_graph.public_mappings[id(cycle)].complete is False
    with pytest.raises(ValueError, match="cyclic native input"):
        _snapshot_native(cycle, _frozen_graph=cycle_graph)


def test_public_mapping_observation_failure_stays_deferred_to_receiver_order():
    invalid_planned = planned_manifest()
    invalid_concurrency = invalid_planned.concurrency[0].model_copy(
        update={"value": True}
    )
    invalid_planned = invalid_planned.model_copy(
        update={"concurrency": (invalid_concurrency,)}
    )
    invalid_mapping = FailingItemsMapping()

    with pytest.raises(ValidationError) as invalid_error:
        finish_run(
            invalid_planned,
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(invalid_mapping,),
        )
    assert invalid_error.value.errors()[0]["loc"] == (
        "concurrency",
        0,
        "value",
    )
    assert invalid_mapping.items_calls == 1

    nonplanned_mapping = FailingItemsMapping()
    with pytest.raises(ValueError, match="only a planned run can be finished"):
        finish_run(
            completed_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(nonplanned_mapping,),
        )
    assert nonplanned_mapping.items_calls == 1

    reached_mapping = FailingItemsMapping()
    with pytest.raises(
        RuntimeError,
        match="ordinary public mapping observation failed",
    ):
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(reached_mapping,),
        )
    assert reached_mapping.items_calls == 1


def test_public_only_tuple_mapping_keeps_lazy_first_observation():
    mapping = InventoryMapping((("value", 1),))
    parent = PublicViewTuple((), (mapping,))

    frozen = _freeze_model_graphs(parent)

    assert mapping.items_calls == 0
    assert id(mapping) not in frozen.public_mapping_owners
    assert _snapshot_native(parent, _frozen_graph=frozen) == ({"value": 1},)
    assert parent.iteration_calls == 1
    assert mapping.items_calls == 1


@pytest.mark.parametrize("boundary", ["serialize", "finish"])
@pytest.mark.parametrize(
    ("target_name", "expected_path"),
    [("manifest", "$"), ("nested_resource", "$.resources[0]")],
)
@pytest.mark.parametrize("storage_kind", ["stored", "extra"])
def test_receivers_use_base_model_descriptors_for_native_storage(
    boundary: str,
    target_name: str,
    expected_path: str,
    storage_kind: str,
):
    manifest, target = descriptor_masked_manifest(target_name)

    if storage_kind == "stored":
        backing = BASE_MODEL_DICT_DESCRIPTOR.__get__(target, pydantic.BaseModel)
        dict.__setitem__(backing, "future_constraint", "deny")
        assert tuple(dict.items(backing))[-1] == ("future_constraint", "deny")
        assert "future_constraint" not in target.__dict__
    else:
        BASE_MODEL_EXTRA_DESCRIPTOR.__set__(
            target, {"future_constraint": "deny"}
        )
        backing = BASE_MODEL_EXTRA_DESCRIPTOR.__get__(target, pydantic.BaseModel)
        assert tuple(dict.items(backing)) == (("future_constraint", "deny"),)
        assert target.__pydantic_extra__ is None

    with pytest.raises(ValueError) as exc_info:
        receive_planned_manifest(boundary, manifest)

    assert str(exc_info.value) == (
        f"undeclared stored fields at {expected_path}: 'future_constraint'"
    )


@pytest.mark.parametrize("boundary", ["serialize", "finish"])
@pytest.mark.parametrize(
    ("target_name", "expected_path"),
    [("manifest", "$"), ("nested_resource", "$.resources[0]")],
)
@pytest.mark.parametrize("native_state", ["valid", "unknown"])
@pytest.mark.parametrize("extra_state", ["absent", "none", "empty", "populated"])
def test_receivers_preserve_base_model_extra_storage_states(
    boundary: str,
    target_name: str,
    expected_path: str,
    native_state: str,
    extra_state: str,
):
    manifest, target = descriptor_masked_manifest(target_name)
    stored = BASE_MODEL_DICT_DESCRIPTOR.__get__(target, pydantic.BaseModel)
    if native_state == "unknown":
        dict.__setitem__(stored, "future_constraint", "stored")
        assert ("future_constraint", "stored") in tuple(dict.items(stored))

    if extra_state == "absent":
        BASE_MODEL_EXTRA_DESCRIPTOR.__delete__(target)
        with pytest.raises(AttributeError):
            BASE_MODEL_EXTRA_DESCRIPTOR.__get__(target, pydantic.BaseModel)
    else:
        extras = {
            "none": None,
            "empty": {},
            "populated": {"future_extra": "extra"},
        }[extra_state]
        BASE_MODEL_EXTRA_DESCRIPTOR.__set__(target, extras)
        assert (
            BASE_MODEL_EXTRA_DESCRIPTOR.__get__(target, pydantic.BaseModel)
            is extras
        )

    rejected_names = []
    if native_state == "unknown":
        rejected_names.append("'future_constraint'")
    if extra_state == "populated":
        rejected_names.append("'future_extra'")

    if rejected_names:
        with pytest.raises(ValueError) as exc_info:
            receive_planned_manifest(boundary, manifest)
        assert str(exc_info.value) == (
            f"undeclared stored fields at {expected_path}: "
            + ", ".join(rejected_names)
        )
    else:
        assert_received(boundary, manifest)


@pytest.mark.parametrize("boundary", ["serialize", "finish"])
@pytest.mark.parametrize(
    "storage_kind",
    ["ordinary", "native-dict-subclass", "model-stored-view", "model-extra-view"],
)
def test_receivers_reject_unknown_native_model_storage(
    boundary: str,
    storage_kind: str,
):
    valid = receive_planned_manifest(boundary, planned_manifest())
    assert valid.state is (
        RunState.PLANNED if boundary == "serialize" else RunState.COMPLETED
    )

    if storage_kind.startswith("model-"):
        manifest = HiddenStorageViewManifest.model_validate(
            planned_manifest().model_dump()
        )
    else:
        manifest = planned_manifest()

    native = object.__getattribute__(manifest, "__dict__")
    hidden_backing = None
    if storage_kind == "native-dict-subclass":
        hidden_backing = HiddenModelBacking(dict(dict.items(native)))
        dict.__setitem__(hidden_backing, "future_constraint", "deny")
        object.__setattr__(manifest, "__dict__", hidden_backing)
        native = object.__getattribute__(manifest, "__dict__")
    elif storage_kind == "model-extra-view":
        object.__setattr__(
            manifest,
            "__pydantic_extra__",
            {"future_constraint": "deny"},
        )
        native_extras = object.__getattribute__(manifest, "__pydantic_extra__")
        assert tuple(dict.items(native_extras)) == (("future_constraint", "deny"),)
    else:
        dict.__setitem__(native, "future_constraint", "deny")

    if storage_kind != "model-extra-view":
        assert ("future_constraint", "deny") in tuple(dict.items(native))

    with pytest.raises(ValueError, match="undeclared stored fields.*future_constraint"):
        receive_planned_manifest(boundary, manifest)

    if hidden_backing is not None:
        assert hidden_backing.iteration_calls == 0


@pytest.mark.parametrize("boundary", ["serialize", "finish"])
def test_receivers_snapshot_declared_values_before_nested_traversal(boundary: str):
    manifest = planned_manifest()
    native = object.__getattribute__(manifest, "__dict__")
    original_resources = native["resources"]
    original_measurements = native["measurements"]
    replacement_measurements = (timing(),)
    dict.__setitem__(
        native,
        "resources",
        MutatingTuple(
            original_resources,
            lambda: dict.__setitem__(
                native, "measurements", replacement_measurements
            ),
        ),
    )

    received = receive_planned_manifest(boundary, manifest)

    assert native["measurements"] is replacement_measurements
    assert original_measurements == ()
    if boundary == "serialize":
        assert received.measurements == original_measurements
    else:
        assert received.measurements == (timing(),)


@pytest.mark.parametrize("boundary", ["serialize", "finish"])
def test_receivers_reject_initial_invalid_value_changed_during_traversal(
    boundary: str,
):
    manifest = planned_manifest()
    native = object.__getattribute__(manifest, "__dict__")
    original_resources = native["resources"]
    invalid_measurements = [timing()]
    dict.__setitem__(native, "measurements", invalid_measurements)
    dict.__setitem__(
        native,
        "resources",
        MutatingTuple(
            original_resources,
            lambda: dict.__setitem__(native, "measurements", ()),
        ),
    )

    with pytest.raises(ValidationError):
        receive_planned_manifest(boundary, manifest)

    assert native["measurements"] == ()
    assert isinstance(invalid_measurements, list)


@pytest.mark.parametrize("boundary", ["serialize", "finish"])
def test_receivers_freeze_nested_models_before_overridable_traversal(
    boundary: str,
):
    manifest = planned_manifest()
    invalid_resource = pending_memory().model_copy(
        update={"observed_value": 1}
    )
    invalid_native = BASE_MODEL_DICT_DESCRIPTOR.__get__(
        invalid_resource, pydantic.BaseModel
    )
    manifest_native = BASE_MODEL_DICT_DESCRIPTOR.__get__(
        manifest, pydantic.BaseModel
    )
    original_concurrency = manifest_native["concurrency"]
    manifest_native["resources"] = (invalid_resource,)
    manifest_native["concurrency"] = MutatingTuple(
        original_concurrency,
        lambda: dict.__setitem__(invalid_native, "observed_value", None),
    )

    with pytest.raises(ValidationError) as exc_info:
        receive_planned_manifest(boundary, manifest)

    assert invalid_native["observed_value"] is None
    assert exc_info.value.errors()[0]["loc"] == ("resources", 0)
    assert "observed_value is valid only for a measured resource" in str(
        exc_info.value
    )


@pytest.mark.parametrize("boundary", ["serialize", "finish"])
def test_receivers_preserve_valid_overridable_tuple_traversal(boundary: str):
    manifest = planned_manifest()
    native = BASE_MODEL_DICT_DESCRIPTOR.__get__(manifest, pydantic.BaseModel)
    original_concurrency = native["concurrency"]
    calls: list[str] = []
    native["concurrency"] = MutatingTuple(
        original_concurrency,
        lambda: calls.append("iterated"),
    )

    assert_received(boundary, manifest)

    assert calls == ["iterated"]


@pytest.mark.parametrize("boundary", ["serialize", "finish"])
def test_receivers_preserve_benign_string_subclass_field_names(boundary: str):
    class FieldName(str):
        pass

    manifest = planned_manifest()
    native = BASE_MODEL_DICT_DESCRIPTOR.__get__(manifest, pydantic.BaseModel)
    run_id = dict.pop(native, "run_id")
    stored_name = FieldName("run_id")
    dict.__setitem__(native, stored_name, run_id)

    assert any(
        type(name) is FieldName
        for name, _field_value in dict.items(native)
        if name == "run_id"
    )
    assert_received(boundary, manifest)


@pytest.mark.parametrize("boundary", ["serialize", "finish"])
@pytest.mark.parametrize(
    ("invalid_target", "expected_location"),
    [("nested", ("resources", 0)), ("root", ("state",))],
)
def test_receivers_freeze_models_before_stored_name_hash_callbacks(
    boundary: str,
    invalid_target: str,
    expected_location: tuple[object, ...],
):
    assert_received(boundary, planned_manifest())

    for armed in (False, True):
        manifest = planned_manifest()
        native = BASE_MODEL_DICT_DESCRIPTOR.__get__(
            manifest, pydantic.BaseModel
        )
        if invalid_target == "nested":
            invalid_resource = pending_memory().model_copy(
                update={"observed_value": 1}
            )
            invalid_native = BASE_MODEL_DICT_DESCRIPTOR.__get__(
                invalid_resource, pydantic.BaseModel
            )
            native["resources"] = (invalid_resource,)

            def repair() -> None:
                dict.__setitem__(invalid_native, "observed_value", None)

            def repaired_value() -> object:
                return dict.__getitem__(invalid_native, "observed_value")

        else:
            dict.__setitem__(native, "state", "invalid-state")

            def repair() -> None:
                dict.__setitem__(native, "state", RunState.PLANNED)

            def repaired_value() -> object:
                return dict.__getitem__(native, "state")

        stored_name = None
        if armed:
            run_id = dict.pop(native, "run_id")
            stored_name = ArmedFieldName("run_id")
            dict.__setitem__(native, stored_name, run_id)
            stored_name.arm(repair)

        with pytest.raises(ValidationError) as exc_info:
            receive_planned_manifest(boundary, manifest)

        assert exc_info.value.errors()[0]["loc"] == expected_location
        if armed:
            assert stored_name is not None
            assert stored_name.hash_calls == 1
            expected_repaired = (
                None if invalid_target == "nested" else RunState.PLANNED
            )
            assert repaired_value() == expected_repaired


@pytest.mark.parametrize("boundary", ["serialize", "finish"])
def test_receivers_do_not_retroactively_include_late_unknowns(boundary: str):
    manifest = planned_manifest()
    native = object.__getattribute__(manifest, "__dict__")
    original_resources = native["resources"]
    dict.__setitem__(
        native,
        "resources",
        MutatingTuple(
            original_resources,
            lambda: dict.__setitem__(native, "late_unknown", "deny"),
        ),
    )

    received = receive_planned_manifest(boundary, manifest)

    assert ("late_unknown", "deny") in tuple(dict.items(native))
    received_native = object.__getattribute__(received, "__dict__")
    assert "late_unknown" not in received_native


def test_planned_to_completed_round_trip_is_canonical_and_stable():
    manifest = completed_manifest()

    first = serialize_run_manifest(manifest)
    reparsed = validate_run_manifest_json(first)

    assert reparsed == manifest
    assert serialize_run_manifest(reparsed) == first
    assert first.startswith(b'{"build":')
    assert b'"observed_value":402653184' in first


@pytest.mark.parametrize("target_name", ["manifest", "nested_resource"])
@pytest.mark.parametrize(
    "malformed_extras",
    [[], (), "", 0, False, ["unexpected"]],
    ids=[
        "empty-list",
        "empty-tuple",
        "empty-string",
        "zero",
        "false",
        "nonempty-list",
    ],
)
def test_serializer_rejects_malformed_extra_storage(
    target_name: str,
    malformed_extras: object,
):
    manifest, target = planned_manifest_with_extra_target(target_name)
    object.__setattr__(target, "__pydantic_extra__", malformed_extras)

    with pytest.raises(ValueError, match="malformed stored extras"):
        serialize_run_manifest(manifest)


@pytest.mark.parametrize("target_name", ["manifest", "nested_resource"])
@pytest.mark.parametrize(
    "valid_extras",
    [None, {}, MappingProxyType({})],
    ids=["none", "empty-dict", "empty-mapping"],
)
def test_serializer_accepts_absent_or_empty_mapping_extra_storage(
    target_name: str,
    valid_extras: object,
):
    manifest, target = planned_manifest_with_extra_target(target_name)
    object.__setattr__(target, "__pydantic_extra__", valid_extras)

    assert serialize_run_manifest(manifest) == serialize_run_manifest(
        planned_manifest()
    )


@pytest.mark.parametrize(
    ("target_name", "extra_key", "extra_value"),
    [
        ("manifest", "future_constraint", "deny"),
        ("manifest", "run_id", "run-001"),
        ("manifest", "run_id", "shadow-run"),
        ("nested_resource", "future_constraint", "deny"),
        ("nested_resource", "scope", "api-container"),
        ("nested_resource", "scope", "shadow-scope"),
    ],
    ids=[
        "direct-unknown",
        "direct-declared-equal",
        "direct-declared-conflicting",
        "nested-unknown",
        "nested-declared-equal",
        "nested-declared-conflicting",
    ],
)
def test_serializer_rejects_nonempty_stable_extra_entry_snapshot(
    target_name: str,
    extra_key: str,
    extra_value: str,
):
    manifest, target = planned_manifest_with_extra_target(target_name)
    object.__setattr__(
        target,
        "__pydantic_extra__",
        HiddenIterationExtras({extra_key: extra_value}),
    )

    with pytest.raises(ValueError, match="undeclared stored fields"):
        serialize_run_manifest(manifest)


@pytest.mark.parametrize("target_name", ["manifest", "nested_resource"])
def test_serializer_rejects_falsey_populated_extra_storage(target_name: str):
    manifest, target = planned_manifest_with_extra_target(target_name)
    extras = FalseyPopulatedExtras({"future_constraint": "deny"})
    assert not extras
    assert tuple(extras.items())
    object.__setattr__(target, "__pydantic_extra__", extras)

    with pytest.raises(ValueError, match="undeclared stored fields"):
        serialize_run_manifest(manifest)


@pytest.mark.parametrize(
    ("target_name", "extra_key", "extra_value"),
    [
        ("manifest", "future_constraint", "deny"),
        ("manifest", "run_id", "run-001"),
        ("manifest", "run_id", "shadow-run"),
        ("nested_resource", "future_constraint", "deny"),
        ("nested_resource", "scope", "api-container"),
        ("nested_resource", "scope", "shadow-scope"),
    ],
    ids=[
        "direct-unknown",
        "direct-declared-equal",
        "direct-declared-conflicting",
        "nested-unknown",
        "nested-declared-equal",
        "nested-declared-conflicting",
    ],
)
def test_serializer_rejects_hidden_native_dict_backing(
    target_name: str,
    extra_key: str,
    extra_value: str,
):
    manifest, target = planned_manifest_with_extra_target(target_name)
    extras = HiddenBackingExtras({extra_key: extra_value})
    object.__setattr__(target, "__pydantic_extra__", extras)

    with pytest.raises(ValueError, match="undeclared stored fields") as exc_info:
        serialize_run_manifest(manifest)

    assert repr(extra_key) in str(exc_info.value)
    assert extras.bool_calls == 0
    assert extras.class_calls == 0
    assert extras.len_calls == 0
    assert extras.iteration_calls == 0
    assert extras.keys_calls == 0
    assert extras.items_calls == 0


@pytest.mark.parametrize("target_name", ["manifest", "nested_resource"])
def test_serializer_accepts_empty_native_dict_subclass(target_name: str):
    manifest, target = planned_manifest_with_extra_target(target_name)
    extras = HiddenBackingExtras({})
    object.__setattr__(target, "__pydantic_extra__", extras)

    assert serialize_run_manifest(manifest) == serialize_run_manifest(
        planned_manifest()
    )
    assert extras.bool_calls == 0
    assert extras.class_calls == 0
    assert extras.len_calls == 0
    assert extras.iteration_calls == 0
    assert extras.keys_calls == 0
    assert extras.items_calls == 0


@pytest.mark.parametrize("target_name", ["manifest", "nested_resource"])
@pytest.mark.parametrize(
    "extra_type",
    [MroSpoofExtras, EqualitySpoofExtras],
    ids=["spoofed-mro", "spoofed-mro-equality"],
)
@pytest.mark.parametrize("populated", [False, True], ids=["empty", "unknown"])
def test_serializer_keeps_metaclass_spoofs_on_non_dict_mapping_path(
    target_name: str,
    extra_type: type[MetaclassSpoofExtras],
    populated: bool,
):
    manifest, target = planned_manifest_with_extra_target(target_name)
    values = {"future_constraint": "deny"} if populated else {}
    extras = extra_type(values)
    object.__setattr__(target, "__pydantic_extra__", extras)
    DICT_EQUALITY_SPOOF.calls = 0

    if populated:
        with pytest.raises(ValueError, match="undeclared stored fields"):
            serialize_run_manifest(manifest)
    else:
        assert serialize_run_manifest(manifest) == serialize_run_manifest(
            planned_manifest()
        )

    assert extras.iteration_calls == 1
    assert extras.items_calls == 1
    assert DICT_EQUALITY_SPOOF.calls == 0


@pytest.mark.parametrize(
    ("target_name", "expected_path"),
    [
        ("manifest", "$"),
        ("nested_resource", "$.resources[0]"),
    ],
)
def test_serializer_rejects_inverse_items_view_once(
    target_name: str,
    expected_path: str,
):
    manifest, target = planned_manifest_with_extra_target(target_name)
    extras = HiddenItemsExtras({"future_constraint": "deny"})
    object.__setattr__(target, "__pydantic_extra__", extras)

    with pytest.raises(ValueError) as exc_info:
        serialize_run_manifest(manifest)

    message = str(exc_info.value)
    assert f"undeclared stored fields at {expected_path}:" in message
    assert "'future_constraint'" in message
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1


def test_serializer_consumes_one_stable_extra_entry_inventory():
    manifest = planned_manifest()
    extras = ChangingItemsExtras()
    object.__setattr__(manifest, "__pydantic_extra__", extras)

    with pytest.raises(
        ValueError,
        match="undeclared stored fields.*future_constraint",
    ):
        serialize_run_manifest(manifest)

    assert extras.iteration_calls == 1
    assert extras.items_calls == 1


def test_serializer_rejects_extra_storage_overlapping_declared_fields():
    manifest = planned_manifest()
    object.__setattr__(
        manifest,
        "__pydantic_extra__",
        {"run_id": "shadowed-run"},
    )

    with pytest.raises(ValueError, match="undeclared stored fields.*run_id"):
        serialize_run_manifest(manifest)


def test_finish_rejects_malformed_nested_extra_storage():
    manifest, target = planned_manifest_with_extra_target("nested_resource")
    object.__setattr__(target, "__pydantic_extra__", [])

    with pytest.raises(ValueError, match="malformed stored extras"):
        finish_run(
            manifest,
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(timing(),),
        )


def test_validated_evidence_is_immutable():
    manifest = completed_manifest()
    with pytest.raises(ValidationError, match="frozen"):
        manifest.state = RunState.ABORTED


@pytest.mark.parametrize("state", [RunState.COMPLETED, RunState.ABORTED])
def test_missing_resource_observation_is_explicitly_unknown_not_zero(state):
    unknown = measured_memory().model_copy(
        update={
            "observation_kind": ObservationKind.UNKNOWN,
            "observed_value": None,
            "unknown_reason": "cgroup metrics unavailable",
        }
    )
    manifest = finish_run(
        planned_manifest(),
        state=state,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=(ResourceReading.model_validate(unknown.model_dump()),),
        measurements=(timing(),) if state is RunState.COMPLETED else (),
    )

    reparsed = validate_run_manifest_json(serialize_run_manifest(manifest))
    assert reparsed == manifest
    payload = reparsed.model_dump(mode="json")
    assert payload["resources"][0]["observation_kind"] == "unknown"
    assert payload["resources"][0]["observed_value"] is None
    assert payload["resources"][0]["unknown_reason"] == "cgroup metrics unavailable"


def test_run_requires_resource_evidence_at_every_receiving_boundary():
    for manifest in (planned_manifest(), completed_manifest()):
        data = manifest.model_dump()
        data["resources"] = ()
        with pytest.raises(ValidationError, match="at least one resource reading"):
            RunManifest.model_validate(data)

    payload = json.loads(serialize_run_manifest(completed_manifest()))
    payload["resources"] = []
    with pytest.raises(ValidationError, match="at least one resource reading"):
        validate_run_manifest_json(json.dumps(payload))

    unchecked = completed_manifest().model_copy(update={"resources": ()})
    with pytest.raises(ValidationError, match="at least one resource reading"):
        serialize_run_manifest(unchecked)

    with pytest.raises(ValidationError, match="at least one resource reading"):
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(),
            measurements=(timing(),),
        )


@pytest.mark.parametrize("field,value", [
    ("git_commit", "deadbee"),
    ("git_commit", "D" * 40),
    ("image_digest", "latest"),
    ("image_digest", "sha256:" + "a" * 63),
])
def test_build_identity_rejects_non_exact_identifiers(field: str, value: str):
    data = build_identity().model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        BuildIdentity.model_validate(data)


def test_contracts_reject_unknown_fields_and_missing_schema_version():
    data = workload_identity().model_dump()
    data["branch"] = "dev"
    with pytest.raises(ValidationError):
        WorkloadIdentity.model_validate(data)

    data = workload_identity().model_dump()
    del data["schema_version"]
    with pytest.raises(ValidationError):
        WorkloadIdentity.model_validate(data)


def test_bool_is_not_accepted_as_a_counter():
    with pytest.raises(ValidationError):
        ConcurrencySetting(schema_version=VERSION, scope="driver", value=True)


def test_zero_concurrency_is_rejected_instead_of_describing_a_noop_run():
    with pytest.raises(ValidationError, match="at least one"):
        ConcurrencySetting(schema_version=VERSION, scope="driver", value=0)


def test_local_identifier_and_counter_bounds_are_enforced():
    with pytest.raises(ValidationError):
        ConcurrencySetting(schema_version=VERSION, scope="", value=1)
    with pytest.raises(ValidationError):
        ConcurrencySetting(schema_version=VERSION, scope="x" * 257, value=1)
    with pytest.raises(ValidationError):
        ConcurrencySetting(
            schema_version=VERSION,
            scope="driver",
            value=9_007_199_254_740_992,
        )


def test_resource_limit_and_observation_states_are_consistent():
    with pytest.raises(ValidationError, match="positive limit_value"):
        ResourceReading(
            schema_version=VERSION,
            scope="worker",
            resource="cpu",
            unit="millicores",
            limit_kind=LimitKind.BOUNDED,
            observation_kind=ObservationKind.PENDING,
        )

    with pytest.raises(ValidationError, match="requires observed_value"):
        ResourceReading(
            schema_version=VERSION,
            scope="worker",
            resource="cpu",
            unit="millicores",
            limit_kind=LimitKind.UNKNOWN,
            observation_kind=ObservationKind.MEASURED,
        )

    with pytest.raises(ValidationError, match="requires unknown_reason"):
        ResourceReading(
            schema_version=VERSION,
            scope="worker",
            resource="cpu",
            unit="millicores",
            limit_kind=LimitKind.UNBOUNDED,
            observation_kind=ObservationKind.UNKNOWN,
        )


def test_sample_accounting_requires_every_attempt_to_have_one_disposition():
    data = timing().model_dump()
    data["included_sample_count"] = 9
    with pytest.raises(ValidationError, match="must equal attempted"):
        TimingMeasurement.model_validate(data)


def test_timeout_failures_are_counted_without_becoming_exclusions():
    measurement = TimingMeasurement(
        schema_version=VERSION,
        name="read-with-timeouts",
        time_unit="milliseconds",
        attempted_sample_count=10,
        included_sample_count=8,
        provenance=timing(warmup=False).provenance,
        failure_outcomes=(
            FailureOutcome(schema_version=VERSION, kind="timeout", count=2),
        ),
    )

    assert measurement.included_sample_count == 8
    assert measurement.failure_outcomes[0].count == 2
    assert measurement.exclusions == ()

    manifest = finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=(measured_memory(),),
        measurements=(measurement,),
    )
    reparsed = validate_run_manifest_json(serialize_run_manifest(manifest))
    assert reparsed.measurements[0].failure_outcomes == measurement.failure_outcomes


def test_failure_outcomes_reject_unknown_kinds_and_accounting_mismatches():
    with pytest.raises(ValidationError):
        FailureOutcome(schema_version=VERSION, kind="unavailable", count=1)

    data = timing(warmup=False).model_dump()
    data["included_sample_count"] = 8
    data["failure_outcomes"] = (
        FailureOutcome(schema_version=VERSION, kind="error", count=1),
    )
    with pytest.raises(ValidationError, match="must equal attempted"):
        TimingMeasurement.model_validate(data)


def test_timeouts_and_errors_cannot_be_hidden_as_exclusions():
    data = timing().model_dump()
    data["exclusions"][0]["kind"] = "timeout"
    with pytest.raises(ValidationError):
        TimingMeasurement.model_validate(data)


def test_cold_run_cannot_discard_warmup_samples():
    with pytest.raises(ValidationError, match="cold runs"):
        finish_run(
            planned_manifest(condition=CacheCondition.COLD),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(timing(warmup=True),),
        )


def test_completed_run_requires_measurement_and_resolved_resource_observations():
    with pytest.raises(ValidationError, match="at least one measurement"):
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
        )

    with pytest.raises(ValidationError, match="pending observations"):
        RunManifest(
            schema_version=VERSION,
            run_id="run-001",
            state=RunState.ABORTED,
            build=build_identity(),
            workload=workload_identity(),
            cache_condition=CacheCondition.WARM,
            concurrency=concurrency(),
            resources=(pending_memory(),),
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        )


def test_finish_rejects_resource_envelope_changes_and_repeat_transitions():
    changed_limit = measured_memory().model_copy(update={"limit_value": 1_073_741_824})
    with pytest.raises(ValueError, match="resource envelope"):
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(changed_limit,),
            measurements=(timing(),),
        )

    with pytest.raises(ValueError, match="only a planned run"):
        finish_run(
            completed_manifest(),
            state=RunState.ABORTED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 2, tzinfo=timezone.utc),
            resources=(measured_memory(),),
        )


def test_timestamps_must_be_ordered_and_timezone_aware():
    data = completed_manifest().model_dump()
    data["finished_at"] = datetime(2026, 9, 28, 7, 59, tzinfo=timezone.utc)
    with pytest.raises(ValidationError, match="cannot precede"):
        RunManifest.model_validate(data)

    data = completed_manifest().model_dump()
    data["started_at"] = datetime(2026, 9, 28, 8, 0)
    with pytest.raises(ValidationError, match="explicit UTC offset"):
        RunManifest.model_validate(data)


def test_duplicate_scopes_names_and_json_keys_are_rejected():
    data = planned_manifest().model_dump()
    data["concurrency"] = (data["concurrency"][0], data["concurrency"][0])
    with pytest.raises(ValidationError, match="scopes must be unique"):
        RunManifest.model_validate(data)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        validate_run_manifest_json('{"schema_version":"candidate-v1","run_id":"a","run_id":"b"}')


@pytest.mark.parametrize(
    "resource,measurement",
    [
        (
            measured_memory().model_copy(update={"observed_value": True}),
            timing(),
        ),
        (
            measured_memory(),
            timing().model_copy(update={"time_unit": "fortnights"}),
        ),
        (
            measured_memory(),
            timing().model_copy(
                update={
                    "provenance": timing().provenance.model_copy(
                        update={"clock": "wall_clock"}
                    )
                }
            ),
        ),
        (
            measured_memory(),
            TimingMeasurement.model_construct(
                schema_version=VERSION,
                name="constructed-invalid",
                time_unit="fortnights",
                attempted_sample_count=1,
                included_sample_count=1,
                provenance=timing().provenance,
                exclusions=(),
                failure_outcomes=(),
            ),
        ),
        (
            measured_memory().model_copy(update={"future_semantics": "deny"}),
            timing(),
        ),
    ],
)
def test_finish_deeply_revalidates_unchecked_nested_models(resource, measurement):
    with pytest.raises((ValidationError, ValueError)):
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(resource,),
            measurements=(measurement,),
        )


@pytest.mark.parametrize(
    ("callback_source", "invalid_target", "expected_location"),
    [
        ("planned", "resource", ("resources", 0, "observed_value")),
        ("planned", "measurement", ("measurements", 0, "time_unit")),
        ("resource", "measurement", ("measurements", 0, "time_unit")),
    ],
)
def test_finish_freezes_all_input_models_before_earlier_argument_callbacks(
    callback_source: str,
    invalid_target: str,
    expected_location: tuple[object, ...],
):
    def inputs(invalid: bool):
        resource = measured_memory()
        measurement = timing()
        if invalid_target == "resource":
            if invalid:
                resource = resource.model_copy(update={"observed_value": True})
            target = resource
            field_name = "observed_value"
            repaired_value: object = 402_653_184
        else:
            if invalid:
                measurement = measurement.model_copy(
                    update={"time_unit": "fortnights"}
                )
            target = measurement
            field_name = "time_unit"
            repaired_value = "milliseconds"
        return resource, measurement, target, field_name, repaired_value

    def callback_inputs(invalid: bool):
        planned = planned_manifest()
        resource, measurement, target, field_name, repaired_value = inputs(invalid)
        target_native = BASE_MODEL_DICT_DESCRIPTOR.__get__(
            target, pydantic.BaseModel
        )
        callback_calls: list[str] = []

        def repair_target() -> None:
            callback_calls.append("called")
            dict.__setitem__(target_native, field_name, repaired_value)

        resources: tuple[ResourceReading, ...]
        if callback_source == "planned":
            planned_native = BASE_MODEL_DICT_DESCRIPTOR.__get__(
                planned, pydantic.BaseModel
            )
            planned_native["concurrency"] = MutatingTuple(
                planned_native["concurrency"],
                repair_target,
            )
            resources = (resource,)
        else:
            resources = MutatingTuple((resource,), repair_target)

        arguments = {
            "planned": planned,
            "state": RunState.COMPLETED,
            "started_at": datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            "finished_at": datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            "resources": resources,
            "measurements": (measurement,),
        }
        return arguments, callback_calls, target_native, field_name, repaired_value

    ordinary_resource, ordinary_measurement, *_unused = inputs(True)
    with pytest.raises(ValidationError) as ordinary_error:
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(ordinary_resource,),
            measurements=(ordinary_measurement,),
        )
    assert ordinary_error.value.errors()[0]["loc"] == expected_location

    invalid_callback = callback_inputs(True)
    with pytest.raises(ValidationError) as callback_error:
        finish_run(**invalid_callback[0])
    assert callback_error.value.errors() == ordinary_error.value.errors()
    assert invalid_callback[1] == ["called"]
    assert dict.__getitem__(invalid_callback[2], invalid_callback[3]) == (
        invalid_callback[4]
    )

    valid_callback = callback_inputs(False)
    result = finish_run(**valid_callback[0])
    assert result.state is RunState.COMPLETED
    assert valid_callback[1] == ["called"]
    assert dict.__getitem__(valid_callback[2], valid_callback[3]) == valid_callback[4]


@pytest.mark.parametrize("callback_source", ["planned", "resources"])
def test_finish_freezes_plain_resource_dict_before_callbacks(
    callback_source: str,
):
    def resource_data(invalid: bool) -> dict[str, object]:
        data = measured_memory().model_dump(mode="python")
        if invalid:
            data["observed_value"] = True
        return data

    ordinary_invalid = resource_data(True)
    with pytest.raises(ValidationError) as ordinary_error:
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(ordinary_invalid,),
            measurements=(timing(),),
        )
    assert ordinary_error.value.errors()[0]["loc"] == (
        "resources",
        0,
        "observed_value",
    )

    def callback_arguments(invalid: bool):
        planned = planned_manifest()
        resource = resource_data(invalid)
        callback_calls: list[str] = []

        def repair_resource() -> None:
            callback_calls.append("called")
            dict.__setitem__(resource, "observed_value", 402_653_184)

        if callback_source == "planned":
            native = BASE_MODEL_DICT_DESCRIPTOR.__get__(
                planned,
                pydantic.BaseModel,
            )
            native["concurrency"] = MutatingTuple(
                native["concurrency"],
                repair_resource,
            )
            resources = (resource,)
        else:
            resources = MutatingTuple((resource,), repair_resource)
        return planned, resources, resource, callback_calls

    attacked = callback_arguments(True)
    with pytest.raises(ValidationError) as callback_error:
        finish_run(
            attacked[0],
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=attacked[1],
            measurements=(timing(),),
        )
    assert callback_error.value.errors() == ordinary_error.value.errors()
    assert attacked[2]["observed_value"] == 402_653_184
    assert attacked[3] == ["called"]

    valid = callback_arguments(False)
    result = finish_run(
        valid[0],
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=valid[1],
        measurements=(timing(),),
    )
    assert result.resources[0].observed_value == 402_653_184
    assert valid[3] == ["called"]


def test_finish_preserves_authoritative_public_resource_tuple_view():
    invalid_native = measured_memory().model_dump(mode="python")
    invalid_native["observed_value"] = True
    visible = measured_memory(value=123)
    resources = PublicViewTuple((invalid_native,), (visible,))

    result = finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=resources,
        measurements=(timing(),),
    )

    assert result.resources[0].observed_value == 123
    assert resources.iteration_calls == 1


def test_finish_preserves_deep_later_native_failure_for_valid_planned():
    shallow = PublicResourceWithDeepNativeBacking(2)
    result = finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=(shallow,),
        measurements=(timing(),),
    )
    assert result.resources[0] == measured_memory()
    assert shallow.items_calls == 1

    deep = PublicResourceWithDeepNativeBacking(sys.getrecursionlimit() + 600)
    with pytest.raises(RecursionError, match="maximum recursion depth exceeded"):
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(deep,),
            measurements=(timing(),),
        )
    assert deep.items_calls == 1


def test_finish_validates_planned_before_deep_later_native_failure():
    invalid_planned = planned_manifest()
    invalid_concurrency = invalid_planned.concurrency[0].model_copy(
        update={"value": True}
    )
    invalid_planned = invalid_planned.model_copy(
        update={"concurrency": (invalid_concurrency,)}
    )
    deep_for_invalid = PublicResourceWithDeepNativeBacking(
        sys.getrecursionlimit() + 600
    )
    with pytest.raises(ValidationError) as invalid_error:
        finish_run(
            invalid_planned,
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(deep_for_invalid,),
            measurements=(timing(),),
        )
    assert invalid_error.value.errors()[0]["type"] == "int_type"
    assert invalid_error.value.errors()[0]["loc"] == (
        "concurrency",
        0,
        "value",
    )
    assert deep_for_invalid.items_calls == 1

    deep_for_nonplanned = PublicResourceWithDeepNativeBacking(
        sys.getrecursionlimit() + 600
    )
    with pytest.raises(
        ValueError,
        match="only a planned run can be finished",
    ):
        finish_run(
            completed_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(deep_for_nonplanned,),
            measurements=(timing(),),
        )
    assert deep_for_nonplanned.items_calls == 1


def test_finish_snapshots_public_mapping_child_before_requesting_next_pair():
    def finish_with(measurements):
        return finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=measurements,
        )

    def mapping_input(*, invalid: bool, name: str):
        measurement = timing()
        child = measurement.exclusions[0]
        if invalid:
            child = child.model_copy(update={"count": True})
        values = measurement.model_dump(mode="python")
        values["name"] = name
        values["exclusions"] = (child,)
        native = BASE_MODEL_DICT_DESCRIPTOR.__get__(child, pydantic.BaseModel)
        mapping = YieldRepairMapping(
            values,
            lambda: dict.__setitem__(native, "count", 2),
        )
        return mapping, child, native

    ordinary_measurement = timing()
    ordinary_invalid = ordinary_measurement.model_dump(mode="python")
    ordinary_invalid["exclusions"] = (
        ordinary_measurement.exclusions[0].model_copy(update={"count": True}),
    )
    with pytest.raises(ValidationError) as ordinary_error:
        finish_with((ordinary_invalid,))
    assert ordinary_error.value.errors()[0]["loc"] == (
        "measurements",
        0,
        "exclusions",
        0,
        "count",
    )

    attacked, _child, attacked_native = mapping_input(
        invalid=True,
        name="public-invalid",
    )
    with pytest.raises(ValidationError) as callback_error:
        finish_with((attacked,))
    assert callback_error.value.errors() == ordinary_error.value.errors()
    assert attacked.items_calls == 1
    assert attacked.callback_calls == 1
    assert attacked.events[:3] == [
        "items",
        "yield:exclusions",
        "callback",
    ]
    assert dict.__getitem__(attacked_native, "count") == 2

    valid, _child, valid_native = mapping_input(
        invalid=False,
        name="public-valid",
    )
    result = finish_with((valid,))
    assert result.state is RunState.COMPLETED
    assert valid.items_calls == 1
    assert valid.callback_calls == 1
    assert valid.events[:3] == [
        "items",
        "yield:exclusions",
        "callback",
    ]
    assert dict.__getitem__(valid_native, "count") == 2

    shared, shared_child, shared_native = mapping_input(
        invalid=True,
        name="shared-second",
    )
    first = timing().model_dump(mode="python")
    first["name"] = "shared-first"
    first["exclusions"] = (shared_child,)
    with pytest.raises(ValidationError) as shared_error:
        finish_with((first, shared))
    assert shared_error.value.errors()[0] == ordinary_error.value.errors()[0]
    assert shared.items_calls == 1
    assert shared.callback_calls == 1
    assert dict.__getitem__(shared_native, "count") == 2


def test_finish_replays_new_exact_dict_before_snapshotting_its_entries():
    unexpected_callbacks: list[str] = []

    def reject_callback() -> None:
        unexpected_callbacks.append("called")
        raise RuntimeError("callback ran before native traversal")

    trigger = PublicViewTuple((), (), action=reject_callback)
    deep: dict[str, object] = {"leaf": "value"}
    for index in range(sys.getrecursionlimit() + 600):
        deep = {f"level-{index}": deep}
    provenance = {"trigger": trigger}
    provenance.update(timing().provenance.model_dump(mode="python"))
    provenance["deep"] = deep
    values = timing().model_dump(mode="python")
    values["provenance"] = provenance
    measurement = LazyProvenanceMapping(values)

    with pytest.raises(RecursionError, match="maximum recursion depth exceeded"):
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(measurement,),
        )

    assert measurement.items_calls == 1
    assert trigger.iteration_calls == 0
    assert unexpected_callbacks == []


def test_finish_preserves_new_exact_dict_shallow_and_valid_controls():
    shallow_callbacks: list[str] = []
    trigger = PublicViewTuple(
        (),
        (),
        action=lambda: shallow_callbacks.append("called"),
    )
    shallow = {"leaf": "value"}
    for index in range(2):
        shallow = {f"level-{index}": shallow}
    provenance = {"trigger": trigger}
    provenance.update(timing().provenance.model_dump(mode="python"))
    provenance["deep"] = shallow
    invalid_values = timing().model_dump(mode="python")
    invalid_values["provenance"] = provenance
    invalid = LazyProvenanceMapping(invalid_values)

    with pytest.raises(ValidationError) as invalid_error:
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(invalid,),
        )
    assert invalid_error.value.errors()[0]["type"] == "extra_forbidden"
    assert invalid_error.value.errors()[0]["loc"] == (
        "measurements",
        0,
        "provenance",
        "trigger",
    )
    assert invalid.items_calls == 1
    assert trigger.iteration_calls == 1
    assert shallow_callbacks == ["called"]

    key_callbacks: list[str] = []
    collector_key = ArmedFieldName("collector")
    valid_provenance = {
        collector_key: "benchmark-driver",
        "schema_version": VERSION,
        "collector_version": "1.0.0",
        "clock": "monotonic",
        "placement": "separate-driver-process",
    }
    collector_key.arm(lambda: key_callbacks.append("called"))
    valid_values = timing().model_dump(mode="python")
    valid_values["provenance"] = valid_provenance
    valid = LazyProvenanceMapping(valid_values)

    result = finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=(measured_memory(),),
        measurements=(valid,),
    )
    assert result.state is RunState.COMPLETED
    assert valid.items_calls == 1
    assert collector_key.hash_calls == 1
    assert key_callbacks == ["called"]


def test_finish_preserves_lazy_exact_dict_priority_and_shared_capture():
    def deep_measurement():
        callbacks: list[str] = []
        trigger = PublicViewTuple(
            (),
            (),
            action=lambda: callbacks.append("called"),
        )
        deep: dict[str, object] = {"leaf": "value"}
        for index in range(sys.getrecursionlimit() + 600):
            deep = {f"level-{index}": deep}
        provenance = {"trigger": trigger}
        provenance.update(timing().provenance.model_dump(mode="python"))
        provenance["deep"] = deep
        values = timing().model_dump(mode="python")
        values["provenance"] = provenance
        return LazyProvenanceMapping(values), trigger, callbacks, provenance

    invalid_planned = planned_manifest()
    invalid_concurrency = invalid_planned.concurrency[0].model_copy(
        update={"value": True}
    )
    invalid_planned = invalid_planned.model_copy(
        update={"concurrency": (invalid_concurrency,)}
    )
    invalid, invalid_trigger, invalid_callbacks, _provenance = deep_measurement()
    with pytest.raises(ValidationError) as invalid_error:
        finish_run(
            invalid_planned,
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(invalid,),
        )
    assert invalid_error.value.errors()[0]["loc"] == (
        "concurrency",
        0,
        "value",
    )
    assert invalid.items_calls == 1
    assert invalid_trigger.iteration_calls == 0
    assert invalid_callbacks == []

    nonplanned, nonplanned_trigger, nonplanned_callbacks, _ = deep_measurement()
    with pytest.raises(
        ValueError,
        match="only a planned run can be finished",
    ):
        finish_run(
            completed_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(nonplanned,),
        )
    assert nonplanned.items_calls == 1
    assert nonplanned_trigger.iteration_calls == 0
    assert nonplanned_callbacks == []

    public, shared_trigger, shared_callbacks, shared_provenance = (
        deep_measurement()
    )
    native = timing().model_dump(mode="python")
    native["provenance"] = shared_provenance
    measurements = PublicViewTuple((native,), (public,))
    with pytest.raises(RecursionError, match="maximum recursion depth exceeded"):
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=measurements,
        )
    assert measurements.iteration_calls == 0
    assert public.items_calls == 0
    assert shared_trigger.iteration_calls == 0
    assert shared_callbacks == []


def test_public_only_parent_cannot_refreeze_native_exposed_plain_child():
    native_measurement = timing().model_dump(mode="python")
    visible_measurement = timing().model_dump(mode="python")
    shared_provenance = native_measurement["provenance"]
    assert isinstance(shared_provenance, dict)
    shared_provenance["clock"] = "wall_clock"
    visible_measurement["provenance"] = shared_provenance

    ordinary_measurement = timing().model_dump(mode="python")
    ordinary_provenance = ordinary_measurement["provenance"]
    assert isinstance(ordinary_provenance, dict)
    ordinary_provenance["clock"] = "wall_clock"
    with pytest.raises(ValidationError) as ordinary_error:
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(ordinary_measurement,),
        )

    measurements = PublicViewTuple(
        (native_measurement,),
        (visible_measurement,),
        action=lambda: dict.__setitem__(
            shared_provenance,
            "clock",
            "monotonic",
        ),
    )
    with pytest.raises(ValidationError) as callback_error:
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=measurements,
        )

    assert callback_error.value.errors() == ordinary_error.value.errors()
    assert shared_provenance["clock"] == "monotonic"
    assert measurements.iteration_calls == 1


def test_finish_rejects_undeclared_planned_fields_and_returns_fresh_graph():
    with pytest.raises(ValueError, match="undeclared stored fields"):
        finish_run(
            planned_manifest().model_copy(update={"future_semantics": "deny"}),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(timing(),),
        )

    resource = measured_memory()
    measurement = timing()
    result = finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=(resource,),
        measurements=(measurement,),
    )
    assert result.resources[0] is not resource
    assert result.measurements[0] is not measurement
    assert result.measurements[0].provenance is not measurement.provenance


def test_finish_revalidates_scalar_and_container_arguments():
    common = {
        "planned": planned_manifest(),
        "state": RunState.COMPLETED,
        "started_at": datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        "finished_at": datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        "resources": (measured_memory(),),
        "measurements": (timing(),),
    }

    with pytest.raises(ValidationError):
        finish_run(**(common | {"state": "completed"}))
    with pytest.raises(ValidationError, match="explicit UTC offset"):
        finish_run(
            **(common | {"started_at": datetime(2026, 9, 28, 8, 0)})
        )
    with pytest.raises(ValidationError):
        finish_run(**(common | {"resources": [measured_memory()]}))
    with pytest.raises(ValidationError):
        finish_run(**(common | {"measurements": [timing()]}))


@pytest.mark.parametrize(
    "manifest",
    [
        completed_manifest().model_copy(update={"state": RunState.PLANNED}),
        completed_manifest().model_copy(update={"future_semantics": "deny"}),
        completed_manifest().model_copy(
            update={
                "measurements": (
                    timing().model_copy(update={"time_unit": "fortnights"}),
                )
            }
        ),
        completed_manifest().model_copy(
            update={
                "measurements": (
                    TimingMeasurement.model_construct(
                        schema_version=VERSION,
                        name="constructed-invalid",
                        time_unit="fortnights",
                        attempted_sample_count=1,
                        included_sample_count=1,
                        provenance=timing().provenance,
                        exclusions=(),
                        failure_outcomes=(),
                    ),
                )
            }
        ),
    ],
)
def test_serializer_rejects_unchecked_invalid_manifest_copies(manifest):
    with pytest.raises((ValidationError, ValueError)):
        serialize_run_manifest(manifest)


def test_native_snapshot_preserves_mapping_keys_and_sequence_types():
    value = {"01": [1, 2], 1: ("a", "b")}
    snapshot = _snapshot_native(value)

    assert tuple(snapshot) == ("01", 1)
    assert isinstance(snapshot["01"], list)
    assert isinstance(snapshot[1], tuple)


def test_receiving_boundaries_reject_lists_malformed_models_and_cycles():
    list_resources = completed_manifest().model_copy(
        update={"resources": [measured_memory()]}
    )
    with pytest.raises(ValidationError):
        serialize_run_manifest(list_resources)

    malformed = RunManifest.model_construct(schema_version=VERSION, run_id="missing")
    with pytest.raises(ValidationError):
        serialize_run_manifest(malformed)

    cycle = []
    cycle.append(cycle)
    cyclic = completed_manifest().model_copy(update={"resources": cycle})
    with pytest.raises(ValueError, match="cyclic native input"):
        serialize_run_manifest(cyclic)
