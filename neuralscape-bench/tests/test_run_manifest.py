"""Contract tests for reproducible, product-independent benchmark manifests."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from abc import ABCMeta
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo
from pathlib import Path
from types import MappingProxyType
from typing import Callable

import pydantic
import pytest
from pydantic import BaseModel, ValidationError

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


class StableFrozenExtras(Mapping[str, object]):
    """Expose stable, separately counted key and item inventories."""

    def __init__(
        self,
        iterated_names: tuple[str, ...],
        item_entries: tuple[tuple[str, object], ...],
        events: list[str] | None = None,
    ):
        self.iterated_names = iterated_names
        self.item_entries = item_entries
        self.values = dict(item_entries)
        self.events = events if events is not None else []
        self.iteration_calls = 0
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        self.iteration_calls += 1
        self.events.append("extras:iter")
        return iter(self.iterated_names)

    def items(self):
        self.items_calls += 1
        self.events.append("extras:items")
        return iter(self.item_entries)


class TerminalFrozenExtras(Mapping[str, object]):
    """Yield one stable prefix before one ordinary terminal exception."""

    def __init__(self, failure_stage: str):
        self.failure_stage = failure_stage
        self.error = RuntimeError("ordinary model-extra inventory failed")
        self.iteration_calls = 0
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __len__(self) -> int:
        return 0

    def __iter__(self):
        self.iteration_calls += 1
        if self.failure_stage == "keys":
            error = self.error

            def generate():
                yield "future_constraint"
                raise error

            return generate()
        return iter(())

    def items(self):
        self.items_calls += 1
        error = self.error

        def generate():
            yield "future_constraint", "deny"
            raise error

        return generate()


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


class EventPublicList(list[object]):
    """Expose a passive public list view and record observation order."""

    def __init__(
        self,
        values: list[object],
        public_values: list[object],
        events: list[str],
        name: str,
    ):
        super().__init__(values)
        self.public_values = public_values
        self.events = events
        self.name = name
        self.iteration_calls = 0

    def __iter__(self):
        self.iteration_calls += 1
        self.events.append(f"{self.name}:iter")
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

    with pytest.raises(ValueError, match="exceeds the native snapshot budget"):
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


@pytest.mark.parametrize("sequence_kind", ["tuple", "list"])
def test_initial_public_sequence_inventory_replays_alias_once(
    sequence_kind: str,
):
    public_values = ({"value": 1},)
    if sequence_kind == "tuple":
        sequence = PublicViewTuple((), public_values)
        expected = public_values
    else:
        sequence = PublicViewList([], list(public_values))
        expected = list(public_values)

    frozen = _freeze_model_graphs(sequence, {"alias": sequence})

    assert frozen.public_sequence_owners[id(sequence)] is sequence
    assert sequence.iteration_calls == 1
    assert _snapshot_native(sequence, _frozen_graph=frozen) == expected
    assert _snapshot_native(
        {"alias": sequence}, _frozen_graph=frozen
    ) == {"alias": expected}
    assert sequence.iteration_calls == 1


@pytest.mark.parametrize("sequence_kind", ["tuple", "list"])
def test_initial_public_sequence_inventory_ignores_hidden_subclass_backing(
    sequence_kind: str,
):
    events: list[str] = []
    hidden = EventPublicTuple((), ("hidden-public",), events, "hidden")
    if sequence_kind == "tuple":
        outer = EventPublicTuple(
            (hidden,),
            ("outer-public",),
            events,
            "outer",
        )
        expected = ("outer-public",)
    else:
        outer = EventPublicList(
            [hidden],
            ["outer-public"],
            events,
            "outer",
        )
        expected = ["outer-public"]

    frozen = _freeze_model_graphs(outer)

    assert _snapshot_native(outer, _frozen_graph=frozen) == expected
    assert outer.iteration_calls == 1
    assert hidden.iteration_calls == 0
    assert id(hidden) not in frozen.public_sequence_owners


@pytest.mark.parametrize("sequence_kind", ["tuple", "list"])
def test_initial_public_sequence_inventory_precedes_public_mapping_observation(
    sequence_kind: str,
):
    events: list[str] = []
    if sequence_kind == "tuple":
        sequence = EventPublicTuple((), (1,), events, "sequence")
    else:
        sequence = EventPublicList([], [1], events, "sequence")
    mapping = InventoryMapping(
        (("value", 1),),
        name="mapping",
        events=events,
    )

    _freeze_model_graphs(mapping, sequence)

    assert events.index("sequence:iter") < events.index("mapping:items")
    assert sequence.iteration_calls == 1
    assert mapping.items_calls == 1


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
def test_initial_public_sequence_inventory_defers_terminal_failure(
    sequence_type,
    retains_prefix: bool,
    expected_advancement_calls: int,
):
    sequence = sequence_type("prefix")

    frozen = _freeze_model_graphs(sequence)

    retained = frozen.public_sequences[id(sequence)]
    assert retained.owner is sequence
    assert retained.entries == (("prefix",) if retains_prefix else ())
    assert retained.failure is sequence.error
    assert retained.complete is False
    assert sequence.iteration_calls == 1
    assert getattr(sequence, "advancement_calls", 0) == (
        expected_advancement_calls
    )
    assert _snapshot_native(
        {"safe": "value"}, _frozen_graph=frozen
    ) == {"safe": "value"}

    with pytest.raises(RuntimeError) as first_error:
        _snapshot_native(sequence, _frozen_graph=frozen)
    with pytest.raises(RuntimeError) as alias_error:
        _snapshot_native({"alias": sequence}, _frozen_graph=frozen)

    assert first_error.value is sequence.error
    assert alias_error.value is sequence.error
    assert sequence.iteration_calls == 1
    assert getattr(sequence, "advancement_calls", 0) == (
        expected_advancement_calls
    )


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
def test_initial_public_sequence_inventory_preserves_finish_error_order(
    sequence_type,
):
    resources = sequence_type(measured_memory())
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
            resources=resources,
            measurements=(timing(),),
        )

    assert invalid_error.value.errors()[0]["loc"] == (
        "concurrency",
        0,
        "value",
    )
    assert resources.iteration_calls == 1

    with pytest.raises(RuntimeError) as reached_error:
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=resources,
            measurements=(timing(),),
        )

    assert reached_error.value is resources.error
    assert resources.iteration_calls == 2


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
    # the replay is pinned by the retained failure below; the key is read as exact text, so no key hook runs
    assert projection_key.hash_calls == 0
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
    assert projection_key.hash_calls == 0   # read as exact text: no key hook runs on either replay


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
    assert projection_key.hash_calls == 0   # read as exact text: no key hook runs; the replay is pinned below
    with pytest.raises(RuntimeError) as primary_error:
        _snapshot_native(primary, _frozen_graph=frozen)

    assert alias_error.value is failure
    assert primary_error.value is failure
    assert failure_calls == 1
    assert prefix.items_calls == 1
    assert projection_key.hash_calls == 0   # read as exact text: no key hook runs on either replay
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
    with pytest.raises(ValueError, match="unsupported key type"):   # refused by type before any hashing
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


def test_authoritative_tuple_mapping_is_inventoried_during_shared_freeze():
    mapping = InventoryMapping((("value", 1),))
    parent = PublicViewTuple((), (mapping,))

    frozen = _freeze_model_graphs(parent)

    assert mapping.items_calls == 1
    assert frozen.public_mapping_owners[id(mapping)] is mapping
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

        before_receiving = repaired_value()
        with pytest.raises(ValidationError) as exc_info:
            receive_planned_manifest(boundary, manifest)

        assert exc_info.value.errors()[0]["loc"] == expected_location
        if armed:
            assert stored_name is not None
            # The stored name is copied to its exact text at capture, so its hash callback is never run by the
            # snapshot (it ran once here before): nothing can repair the captured state, and the invalid state
            # that was captured is what is judged.
            assert stored_name.hash_calls == 0
            assert repaired_value() == before_receiving   # the repair callback never ran


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
    "inventory_source",
    [
        "empty",
        "key-undeclared",
        "key-declared",
        "item-undeclared",
        "item-declared",
    ],
)
def test_frozen_extra_inventory_replays_key_item_union_once(
    target_name: str,
    inventory_source: str,
):
    manifest, target = planned_manifest_with_extra_target(target_name)
    declared_name = "run_id" if target_name == "manifest" else "scope"
    extra_name = (
        declared_name
        if inventory_source in ("key-declared", "item-declared")
        else "future_constraint"
    )
    iterated_names = (
        (extra_name,) if inventory_source.startswith("key-") else ()
    )
    item_entries = (
        ((extra_name, "deny"),)
        if inventory_source.startswith("item-")
        else ()
    )
    extras = StableFrozenExtras(iterated_names, item_entries)
    BASE_MODEL_EXTRA_DESCRIPTOR.__set__(target, extras)

    frozen = _freeze_model_graphs(manifest)

    assert extras.iteration_calls == 1
    assert extras.items_calls == 1
    if inventory_source == "empty":
        snapshot = _snapshot_native(manifest, _frozen_graph=frozen)
        assert snapshot["run_id"] == manifest.run_id
    else:
        with pytest.raises(
            ValueError,
            match="undeclared stored fields",
        ) as exc_info:
            _snapshot_native(manifest, _frozen_graph=frozen)
        assert repr(extra_name) in str(exc_info.value)
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1


def test_frozen_extra_inventory_reuses_one_owner_across_model_aliases():
    manifest = planned_manifest()
    resource = manifest.resources[0]
    extras = StableFrozenExtras((), ())
    BASE_MODEL_EXTRA_DESCRIPTOR.__set__(manifest, extras)
    BASE_MODEL_EXTRA_DESCRIPTOR.__set__(resource, extras)

    frozen = _freeze_model_graphs(manifest)
    snapshot = _snapshot_native(manifest, _frozen_graph=frozen)

    assert snapshot["run_id"] == manifest.run_id
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1


def test_frozen_extra_inventory_precedes_public_mapping_observation():
    events: list[str] = []
    extras = StableFrozenExtras((), (), events)
    public_resource = InventoryMapping(
        tuple(pending_memory().model_dump(mode="python").items()),
        name="resource",
        events=events,
    )
    manifest = planned_manifest().model_copy(
        update={"resources": (public_resource,)}
    )
    BASE_MODEL_EXTRA_DESCRIPTOR.__set__(manifest, extras)

    frozen = _freeze_model_graphs(manifest)
    snapshot = _snapshot_native(manifest, _frozen_graph=frozen)

    assert events.index("extras:iter") < events.index("resource:items")
    assert events.index("extras:items") < events.index("resource:items")
    assert snapshot["resources"][0] == pending_memory().model_dump(
        mode="python"
    )
    assert extras.iteration_calls == 1
    assert extras.items_calls == 1
    assert public_resource.items_calls == 1


@pytest.mark.parametrize("failure_stage", ["keys", "items"])
def test_frozen_extra_inventory_failure_preserves_finish_error_order(
    failure_stage: str,
):
    extras = TerminalFrozenExtras(failure_stage)
    resource = measured_memory()
    BASE_MODEL_EXTRA_DESCRIPTOR.__set__(resource, extras)
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
            resources=(resource,),
            measurements=(timing(),),
        )

    assert invalid_error.value.errors()[0]["loc"] == (
        "concurrency",
        0,
        "value",
    )
    assert extras.iteration_calls == 1
    assert extras.items_calls == (0 if failure_stage == "keys" else 1)

    with pytest.raises(RuntimeError) as reached_error:
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(resource,),
            measurements=(timing(),),
        )

    assert reached_error.value is extras.error
    assert extras.iteration_calls == 2
    assert extras.items_calls == (0 if failure_stage == "keys" else 2)


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
    with pytest.raises(ValueError, match="exceeds the native snapshot budget"):
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

    with pytest.raises(ValueError, match="exceeds the native snapshot budget"):
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
    assert collector_key.hash_calls == 0   # a text key is copied to its exact text at capture: its hook never runs
    assert key_callbacks == []


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
    with pytest.raises(ValueError, match="exceeds the native snapshot budget"):
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=measurements,
        )
    assert measurements.iteration_calls == 1
    assert public.items_calls == 1
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
    value = {"01": [1, 2], "1": ("a", "b")}
    snapshot = _snapshot_native(value)

    assert tuple(snapshot) == ("01", "1")
    assert isinstance(snapshot["01"], list)
    assert isinstance(snapshot["1"], tuple)


def test_native_snapshot_refuses_a_key_that_is_not_text():
    with pytest.raises(ValueError, match="unsupported key type"):
        _snapshot_native({"01": [1, 2], 1: ("a", "b")})


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


class HybridMappingTuple(tuple, Mapping):
    """A tuple that is also a Mapping, with a side effect in sequence iteration."""

    def __new__(cls, values: dict[str, object], action: Callable[[], None]):
        instance = super().__new__(cls, ())
        instance.values = values
        instance.action = action
        instance.sequence_iterations = 0
        return instance

    def __getitem__(self, key: object) -> object:
        return self.values[key]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        self.sequence_iterations += 1
        self.action()
        return iter(self.values)

    def items(self):
        return iter(tuple(self.values.items()))


class HybridMappingList(list, Mapping):
    """A list that is also a Mapping, with a side effect in sequence iteration."""

    def __init__(self, values: dict[str, object], action: Callable[[], None]):
        super().__init__()
        self.values = values
        self.action = action
        self.sequence_iterations = 0

    def __getitem__(self, key: object) -> object:
        return self.values[key]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        self.sequence_iterations += 1
        self.action()
        return iter(self.values)

    def items(self):
        return iter(tuple(self.values.items()))




def _finish_with_measurement(measurement: object) -> RunManifest:
    return finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=(measured_memory(),),
        measurements=(measurement,),
    )


@pytest.mark.parametrize("hybrid_type", [HybridMappingTuple, HybridMappingList])
def test_mapping_view_of_sequence_hybrid_is_observed_before_sequence_iteration(
    hybrid_type,
):
    invalid_values = timing().model_dump(mode="python")
    invalid_values["provenance"] = dict(
        invalid_values["provenance"], clock="wall-clock"
    )

    # Equivalent plain data is rejected.
    with pytest.raises(ValidationError):
        _finish_with_measurement(invalid_values)

    # The hybrid exposes the same mapping at first observation; its sequence
    # iterator would repair the value if it were consulted first.
    hybrid_values = dict(invalid_values)

    def repair() -> None:
        hybrid_values["provenance"] = timing().provenance.model_dump(
            mode="python"
        )

    hybrid = hybrid_type(hybrid_values, repair)
    with pytest.raises(ValidationError):
        _finish_with_measurement(hybrid)
    assert hybrid.sequence_iterations == 0


@pytest.mark.parametrize("hybrid_type", [HybridMappingTuple, HybridMappingList])
def test_mapping_view_of_sequence_hybrid_valid_control_is_accepted(hybrid_type):
    values = timing().model_dump(mode="python")
    hybrid = hybrid_type(values, lambda: None)

    manifest = _finish_with_measurement(hybrid)

    assert manifest.measurements[0] == timing()


class HiddenAliasItemsDict(dict[object, object]):
    """Public items are the supplied pairs; native backing is separate."""

    def __init__(self, public_pairs: tuple[tuple[object, object], ...]):
        super().__init__()
        self.public_pairs = public_pairs

    def items(self):
        return iter(self.public_pairs)


class PairsMapping(Mapping[str, object]):
    def __init__(self, values: dict[str, object]):
        self.values = values

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def _shared_exclusions_alias_shape(exclusion: dict[str, object]):
    """Two measurements share one exclusions tuple subclass.

    The tuple's public exclusion item is a dict subclass whose hidden native
    backing refers to the first measurement Mapping, whose exclusions point
    back to the tuple.  The public graph has no cycle.
    """

    first_values = timing().model_dump(mode="python")
    first_values["name"] = "first"
    first = PairsMapping(first_values)
    item = HiddenAliasItemsDict(tuple(exclusion.items()))
    item["hidden-alias"] = first
    shared = PublicViewTuple((), (item,))
    first_values["exclusions"] = shared
    second = timing().model_dump(mode="python")
    second["name"] = "second"
    second["exclusions"] = shared
    return first, second, shared


def _plain_pair(exclusion: dict[str, object]):
    plain = []
    for name in ("first", "second"):
        values = timing().model_dump(mode="python")
        values["name"] = name
        values["exclusions"] = (exclusion,)
        plain.append(values)
    return tuple(plain)


@pytest.mark.parametrize("second_first", [False, True])
def test_shared_exclusions_alias_through_hidden_backing_is_rejected(second_first):
    """Pin: conservative rejection, never acceptance (input-envelope fail-closed).

    Equivalent plain JSON is accepted; the hidden-backing alias shape is
    rejected with the stable incomplete-inventory error rather than yielding a
    manifest.  If a later change starts accepting this shape, the accepted
    data must equal the plain public view and this test must be updated
    deliberately.
    """

    exclusion = timing().model_dump(mode="python")["exclusions"][0]
    plain = _plain_pair(exclusion)
    assert _finish_with_measurements(
        tuple(reversed(plain)) if second_first else plain
    ).measurements

    first, second, _shared = _shared_exclusions_alias_shape(exclusion)
    measurements = (second, first) if second_first else (first, second)

    with pytest.raises(ValueError, match="incomplete public mapping inventory"):
        _finish_with_measurements(measurements)


@pytest.mark.parametrize("second_first", [False, True])
def test_shared_exclusions_alias_through_hidden_backing_never_accepts_invalid_data(
    second_first,
):
    invalid = dict(timing().model_dump(mode="python")["exclusions"][0], count=0)
    plain = _plain_pair(invalid)
    with pytest.raises(ValidationError):
        _finish_with_measurements(tuple(reversed(plain)) if second_first else plain)

    first, second, _shared = _shared_exclusions_alias_shape(invalid)
    measurements = (second, first) if second_first else (first, second)

    with pytest.raises(ValueError):
        _finish_with_measurements(measurements)


def _finish_with_measurements(measurements: tuple[object, ...]) -> RunManifest:
    return finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=(measured_memory(),),
        measurements=measurements,
    )


NEW_YORK = ZoneInfo("America/New_York")
# 2026-11-01 01:30 happens twice in New York: first at -04:00 (fold=0), then at -05:00 (fold=1).


def _finish_between(started_at: datetime, finished_at: datetime) -> RunManifest:
    return finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=started_at,
        finished_at=finished_at,
        resources=(measured_memory(),),
        measurements=(timing(),),
    )


def test_run_interval_across_a_dst_rollback_is_ordered_in_utc():
    first_pass = datetime(2026, 11, 1, 1, 45, tzinfo=NEW_YORK)  # 05:45 UTC
    second_pass = datetime(2026, 11, 1, 1, 15, fold=1, tzinfo=NEW_YORK)  # 06:15 UTC

    manifest = _finish_between(first_pass, second_pass)

    assert manifest.finished_at.astimezone(timezone.utc) - manifest.started_at.astimezone(
        timezone.utc
    ) == timedelta(minutes=30)
    reparsed = validate_run_manifest_json(serialize_run_manifest(manifest))
    # Ambiguous wall times never compare equal across different tzinfo objects
    # (PEP 495), so compare the instants.
    for name in ("started_at", "finished_at"):
        assert getattr(reparsed, name).astimezone(timezone.utc) == getattr(
            manifest, name
        ).astimezone(timezone.utc)
    assert serialize_run_manifest(reparsed) == serialize_run_manifest(manifest)


def test_run_interval_ending_before_it_starts_across_a_rollback_is_rejected():
    later = datetime(2026, 11, 1, 1, 15, fold=1, tzinfo=NEW_YORK)  # 06:15 UTC
    earlier = datetime(2026, 11, 1, 1, 45, tzinfo=NEW_YORK)  # 05:45 UTC

    with pytest.raises(ValidationError, match="cannot precede"):
        _finish_between(later, earlier)


def test_run_interval_mixing_offsets_is_ordered_by_instant_not_wall_time():
    started = datetime(2026, 9, 28, 9, 0, tzinfo=timezone(timedelta(hours=2)))  # 07:00 UTC
    finished = datetime(2026, 9, 28, 8, 30, tzinfo=timezone.utc)

    assert _finish_between(started, finished).finished_at > _finish_between(
        started, finished
    ).started_at


@pytest.mark.parametrize("state", [RunState.PLANNED, "planned", "completed", None])
def test_finish_run_rejects_a_nonterminal_or_untyped_state(state):
    planned = planned_manifest()

    with pytest.raises((ValueError, ValidationError)):
        finish_run(
            planned,
            state=state,
            started_at=None,
            finished_at=None,
            resources=planned.resources,
        )


def test_finish_run_with_planned_state_does_not_return_a_planned_manifest():
    planned = planned_manifest()

    with pytest.raises(ValueError, match="terminal"):
        finish_run(
            planned,
            state=RunState.PLANNED,
            started_at=None,
            finished_at=None,
            resources=planned.resources,
        )


def _hostile_metaclass(events: list[str]):
    """A metaclass whose ``__eq__`` records a call made *directly by the manifest module*.

    Comparisons issued inside the standard library (the ABC machinery behind
    ``isinstance(value, Mapping)``) are not what an identity test replaces, so only a call
    whose immediate caller is the manifest module counts.
    """

    def from_manifest() -> bool:
        caller = sys._getframe(2).f_code.co_filename
        return caller == run_manifest_module.__file__

    class Meta(type):
        def __eq__(cls, other):
            if from_manifest():
                events.append("eq")
            return True

        def __ne__(cls, other):
            if from_manifest():
                events.append("ne")
            return False

        __hash__ = type.__hash__

    return Meta


def test_exact_type_helper_uses_identity_not_metaclass_equality() -> None:
    events: list[str] = []
    meta = _hostile_metaclass(events)
    hostile = meta("Hostile", (list,), {})
    assert hostile == list and hostile in (dict, list, tuple)  # what ``==`` membership would believe
    assert run_manifest_module._is_exact_type(hostile, (dict, list, tuple)) is False
    assert run_manifest_module._is_exact_type(list, (dict, list, tuple)) is True
    assert run_manifest_module._is_exact_type(type("Sub", (list,), {}), (list,)) is False
    assert events == []


@pytest.mark.parametrize("base", [list, tuple, dict])
@pytest.mark.parametrize("field", ["resources", "measurements"])
def test_a_hostile_metaclass_never_gets_its_eq_run_by_the_manifest_module(base, field) -> None:
    events: list[str] = []
    hostile = _hostile_metaclass(events)("Hostile", (base,), {})
    planned = planned_manifest()
    seed = planned.resources if field == "resources" else (timing(),)
    value = hostile(seed) if base is not dict else hostile()
    arguments = {"resources": (measured_memory(),), "measurements": (timing(),)}
    arguments[field] = value

    with contextlib.suppress(ValueError, TypeError, ValidationError):
        finish_run(
            planned,
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            **arguments,
        )

    assert events == []


def _class_spy(base, events: list[str]):
    """A container subclass whose ``__class__`` is a callback; ``type()`` is unaffected."""

    namespace = {
        "__class__": property(lambda self: events.append("__class__") or base),
    }
    if base is Mapping:
        namespace.update(
            __getitem__=lambda self, key: {}[key],
            __iter__=lambda self: iter(()),
            __len__=lambda self: 0,
        )
        return type("ClassSpy", (Mapping,), namespace)
    return type("ClassSpy", (base,), namespace)


@pytest.mark.parametrize("base", [list, tuple, dict, Mapping])
def test_discovery_never_reads_a_containers_class_attribute(base) -> None:
    events: list[str] = []
    spy = _class_spy(base, events)
    value = spy(()) if base is not dict and base is not Mapping else spy()

    run_manifest_module._freeze_model_graphs(value)

    assert events == []


@pytest.mark.parametrize("field", ["resources", "measurements"])
@pytest.mark.parametrize("base", [list, tuple, dict, Mapping])
def test_finish_run_never_runs_a_class_callback_before_capture(base, field) -> None:
    events: list[str] = []
    spy = _class_spy(base, events)
    value = spy(()) if base is not dict and base is not Mapping else spy()
    planned = planned_manifest()
    arguments = {"resources": (measured_memory(),), "measurements": (timing(),)}
    arguments[field] = value

    with contextlib.suppress(ValueError, TypeError, ValidationError):
        finish_run(
            planned,
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            **arguments,
        )

    assert events == []


def test_a_class_callback_cannot_repair_invalid_evidence_before_capture() -> None:
    """A hostile ``__class__`` that rewrites a resource's invalid observed_value before it is captured."""

    planned = planned_manifest()
    resource = measured_memory()
    repaired: list[str] = []

    class Repairing(tuple):
        @property
        def __class__(self):  # noqa: D401 - the callback under test
            repaired.append("repair")
            dict.__setitem__(BaseModel.__dict__["__dict__"].__get__(resource, BaseModel), "observed_value", 1)
            return tuple

    dict.__setitem__(BaseModel.__dict__["__dict__"].__get__(resource, BaseModel), "observed_value", True)

    with pytest.raises((ValueError, TypeError, ValidationError)):
        finish_run(
            planned,
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=Repairing((resource,)),
            measurements=(timing(),),
        )

    assert repaired == []


def test_an_extras_class_callback_cannot_repair_invalid_evidence_before_capture() -> None:
    """A hostile ``__class__`` on a model's non-dict extras that rewrites a sibling resource's
    invalid observed_value before that resource is captured."""

    planned = planned_manifest()
    carrier = measured_memory()
    invalid = measured_memory()
    repaired: list[str] = []
    stored = BaseModel.__dict__["__dict__"].__get__

    class RepairingExtras(Mapping[str, object]):
        @property
        def __class__(self):  # noqa: D401 - the callback under test
            repaired.append("repair")
            dict.__setitem__(stored(invalid, BaseModel), "observed_value", 1)
            return dict

        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self):
            return iter(())

        def __len__(self) -> int:
            return 0

    dict.__setitem__(stored(invalid, BaseModel), "observed_value", True)
    object.__setattr__(carrier, "__pydantic_extra__", RepairingExtras())

    with pytest.raises((ValueError, TypeError, ValidationError)):
        finish_run(
            planned,
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(carrier, invalid),
            measurements=(timing(),),
        )

    assert repaired == []


def _hooked_metaclass(base_meta, events: list[str]):
    """A metaclass whose hash, equality and subclass-check hooks all record a call."""

    class Meta(base_meta):
        def __hash__(cls):
            events.append("hash")
            return id(cls)

        def __eq__(cls, other):
            events.append("eq")
            return False

        def __ne__(cls, other):
            events.append("ne")
            return True

        def __subclasscheck__(cls, subclass):
            events.append("subclasscheck")
            return False

        def __instancecheck__(cls, instance):
            events.append("instancecheck")
            return False

    return Meta


def _hostile_instances(events: list[str]) -> dict[str, object]:
    from abc import ABCMeta

    plain = _hooked_metaclass(type, events)
    abstract = _hooked_metaclass(ABCMeta, events)
    return {
        "list": plain("HostileList", (list,), {})(),
        "tuple": plain("HostileTuple", (tuple,), {})(),
        "dict": plain("HostileDict", (dict,), {})(),
        "object": plain("HostileObject", (), {})(),
        "mapping-abc": abstract(
            "HostileMapping",
            (Mapping,),
            {"__getitem__": lambda s, k: {}[k], "__iter__": lambda s: iter(()), "__len__": lambda s: 0},
        )(),
    }


def test_type_checks_never_run_metaclass_hooks() -> None:
    from pydantic import BaseModel

    events: list[str] = []
    values = _hostile_instances(events)
    events.clear()  # building the classes is not under test

    for value in values.values():
        run_manifest_module._is_instance(value, (BaseModel, Mapping, tuple, list, dict))
        run_manifest_module._is_instance(value, BaseModel)
        run_manifest_module._is_instance(value, Mapping)
        run_manifest_module._is_subclass(type(value), (BaseModel, Mapping, tuple, list, dict))

    assert events == []


def test_type_checks_decide_by_native_inheritance() -> None:
    from types import MappingProxyType

    from pydantic import BaseModel

    is_instance = run_manifest_module._is_instance
    assert is_instance({}, Mapping) and is_instance(MappingProxyType({}), Mapping)
    assert is_instance(planned_manifest(), BaseModel)
    assert is_instance((), tuple) and is_instance([], list) and is_instance((), (list, tuple))
    assert not is_instance("text", Mapping) and not is_instance(5, (tuple, list)) and not is_instance(None, BaseModel)

    class Custom(Mapping):
        def __getitem__(self, key):
            raise KeyError(key)

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    assert is_instance(Custom(), Mapping)


def test_a_class_only_registered_as_a_mapping_is_not_a_mapping() -> None:
    iterated: list[str] = []

    class RegisteredOnly:
        def __getitem__(self, key):
            iterated.append("getitem")
            return 1

        def __iter__(self):
            iterated.append("iter")
            return iter(())

        def __len__(self):
            return 0

        def items(self):
            iterated.append("items")
            return ()

    Mapping.register(RegisteredOnly)
    value = RegisteredOnly()
    assert isinstance(value, Mapping)  # the ABC machinery believes the registration
    assert run_manifest_module._is_instance(value, Mapping) is False  # fail closed

    planned = planned_manifest()
    with pytest.raises((ValueError, TypeError, ValidationError)):
        finish_run(
            planned,
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(value,),
            measurements=(timing(),),
        )
    assert iterated == []  # never iterated as if it were a mapping


class DelegatingProxy:
    """Reports ``ResourceReading`` through ``__class__`` and delegates every attribute to a target."""

    def __init__(self, target: object, events: list[str]) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_events", events)

    @property
    def __class__(self):  # noqa: D401 - the callback under test
        object.__getattribute__(self, "_events").append("__class__")
        return ResourceReading

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_target"), name)


def _finish_with_resources(resources):
    return finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=resources,
        measurements=(timing(),),
    )


def test_a_delegating_proxy_cannot_smuggle_invalid_evidence_past_validation() -> None:
    events: list[str] = []
    invalid = measured_memory().model_copy(update={"observed_value": True})  # unchecked copy: a bool observation
    with pytest.raises((ValueError, TypeError, ValidationError)):
        _finish_with_resources((DelegatingProxy(invalid, events),))
    assert events == []  # and rejecting it never ran the proxy's class callback


def test_a_delegating_proxy_over_valid_evidence_is_also_refused() -> None:
    events: list[str] = []
    with pytest.raises((ValueError, TypeError, ValidationError)):
        _finish_with_resources((DelegatingProxy(measured_memory(), events),))
    assert events == []


@pytest.mark.parametrize("leaf", [object(), b"bytes", {1, 2}, 1 + 2j, type, len])
def test_an_unsupported_leaf_type_is_a_stable_error_before_validation(leaf) -> None:
    from pydantic import BaseModel as _Base

    resource = measured_memory()
    dict.__setitem__(_Base.__dict__["__dict__"].__get__(resource, _Base), "unit", leaf)
    with pytest.raises(ValueError, match="unsupported"):
        _finish_with_resources((resource,))


def test_genuine_scalars_enums_and_datetimes_still_pass() -> None:
    manifest = _finish_with_resources((measured_memory(),))
    assert manifest.state is RunState.COMPLETED and manifest.started_at.tzinfo is not None


def _shared_pairs(depth: int):
    node = (measured_memory(),)
    for _ in range(depth):
        node = (node, node)  # 2**depth paths over depth+1 tuples
    return node


def test_shared_exact_subtrees_cannot_force_exponential_reconstruction() -> None:
    import time

    started = time.monotonic()
    with pytest.raises(ValueError, match="native snapshot budget"):
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=_shared_pairs(40),
        )
    assert time.monotonic() - started < 10


def test_a_nesting_depth_past_the_bound_is_a_stable_budget_error() -> None:
    node: object = ()
    for _ in range(run_manifest_module.MAX_SNAPSHOT_DEPTH + 5):
        node = (node,)
    with pytest.raises(ValueError, match="native snapshot budget"):
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=node,
        )


def test_an_ordinary_manifest_is_far_inside_the_snapshot_budget() -> None:
    manifest = completed_manifest()
    frozen = run_manifest_module._freeze_model_graphs(manifest)
    run_manifest_module._snapshot_native(manifest, _frozen_graph=frozen)
    assert frozen.visits[0] < run_manifest_module.MAX_SNAPSHOT_VISITS // 50


def test_a_tzinfo_whose_offset_changes_between_calls_cannot_split_validation_from_serialization() -> None:
    from datetime import timedelta, tzinfo

    calls: list[int] = []

    class Flipping(tzinfo):
        def utcoffset(self, dt):
            calls.append(1)
            return timedelta(0) if len(calls) % 2 else timedelta(hours=5)

        def dst(self, dt):
            return None

        def tzname(self, dt):
            return "flip"

    manifest = finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=Flipping()),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=(measured_memory(),),
        measurements=(timing(),),
    )
    after_capture = len(calls)
    encoded = serialize_run_manifest(manifest)
    reparsed = validate_run_manifest_json(encoded)
    assert len(calls) == after_capture  # the hostile tzinfo is never consulted again
    assert reparsed.started_at == manifest.started_at and serialize_run_manifest(reparsed) == encoded


def test_a_tzinfo_with_an_out_of_range_offset_is_a_stable_error() -> None:
    from datetime import timedelta, tzinfo

    class TooFar(tzinfo):
        def utcoffset(self, dt):
            return timedelta(hours=30)

        def dst(self, dt):
            return None

    with pytest.raises((ValueError, TypeError)):
        finish_run(
            planned_manifest(),
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=TooFar()),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(measured_memory(),),
            measurements=(timing(),),
        )


def test_children_exposed_only_by_a_view_stay_distinct_when_nothing_else_holds_them() -> None:
    import gc

    base = timing()

    class Fresh(tuple):
        def __new__(cls, count):
            instance = super().__new__(cls, ())
            instance.count = count
            return instance

        def __iter__(self):
            for index in range(self.count):
                child = base.model_copy(update={"name": f"fresh-{index}"})
                gc.collect()  # the child is referenced only by this generator frame
                yield child

    manifest = finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=(measured_memory(),),
        measurements=Fresh(120),
    )
    assert [m.name for m in manifest.measurements] == [f"fresh-{i}" for i in range(120)]


class _ClassCallbackMixin:
    """An enum-member base that reports ``ResourceReading`` through ``__class__`` and delegates to a target."""

    events: list[str] = []
    target: object = None

    @property
    def __class__(self):  # noqa: D401 - the callback under test
        type(self).events.append("__class__")
        return ResourceReading

    def __getattr__(self, name: str):
        return getattr(type(self).target, name)


def _hostile_enum_member(target: object, events: list[str]):
    from enum import Enum as _Enum

    class Hostile(_ClassCallbackMixin, str, _Enum):
        spoof = "measured"

    Hostile.events = events
    Hostile.target = target
    return Hostile.spoof


def test_an_enum_member_of_a_foreign_enum_cannot_carry_a_proxy_to_validation() -> None:
    events: list[str] = []
    invalid = measured_memory().model_copy(update={"observed_value": True})  # unchecked copy: a bool observation
    member = _hostile_enum_member(invalid, events)
    with pytest.raises(ValueError, match="unsupported value type"):
        _finish_with_resources((member,))
    assert events == []


def test_a_foreign_enum_member_in_a_scalar_position_is_refused_without_a_callback() -> None:
    events: list[str] = []
    resource = measured_memory()
    member = _hostile_enum_member(resource, events)
    from pydantic import BaseModel as _Base

    dict.__setitem__(_Base.__dict__["__dict__"].__get__(resource, _Base), "observation_kind", member)
    with pytest.raises(ValueError, match="unsupported value type"):
        _finish_with_resources((resource,))
    assert events == []


@pytest.mark.parametrize(
    "member", [RunState.COMPLETED, CacheCondition.WARM, LimitKind.BOUNDED, ObservationKind.MEASURED],
    ids=lambda m: type(m).__name__,
)
def test_members_of_the_projects_own_enums_are_supported_leaves(member) -> None:
    assert run_manifest_module._is_supported_leaf(member)


def test_the_allowed_enums_are_exactly_the_projects_own_by_identity() -> None:
    from enum import Enum as _Enum

    class RunState(_Enum):  # same name, different class
        COMPLETED = "completed"

    assert not run_manifest_module._is_supported_leaf(RunState.COMPLETED)
    assert not run_manifest_module._is_supported_leaf(_Enum("Anything", {"a": 1}).a)


# ── the capture phase is bounded, depth is a stable error, and offsets and extremes are stable ──

_FINISH_WINDOW = dict(
    started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
    finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
)


def _finish(**overrides):
    values = dict(
        state=RunState.COMPLETED, resources=(measured_memory(),), measurements=(timing(),), **_FINISH_WINDOW
    )
    values.update(overrides)
    return finish_run(planned_manifest(), **values)


def test_a_public_sequence_that_never_ends_is_stopped_by_the_budget_while_it_is_read():
    produced = []

    class Endless(tuple):
        def __iter__(self):
            for index in range(5_000_000):
                produced.append(index)
                yield measured_memory()

    with pytest.raises(ValueError, match="exceeds the native snapshot budget"):
        _finish(resources=Endless())
    assert len(produced) <= run_manifest_module.MAX_SNAPSHOT_VISITS + 2


def test_a_public_mapping_that_never_ends_is_stopped_by_the_budget_while_it_is_read():
    produced = []

    class Endless(Mapping):
        def __getitem__(self, key):
            raise KeyError(key)

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

        def items(self):
            for index in range(5_000_000):
                produced.append(index)
                yield f"key-{index}", index

    with pytest.raises(ValueError, match="exceeds the native snapshot budget"):
        _finish(measurements=(Endless(),))
    assert len(produced) <= run_manifest_module.MAX_SNAPSHOT_VISITS + 2


def test_deep_nesting_is_a_budget_error_and_not_a_recursion_error():
    deep: tuple = ()
    for _ in range(5_000):
        deep = (deep,)
    values = timing().model_dump(mode="python")
    values["provenance"] = {**timing().provenance.model_dump(mode="python"), "deep": deep}
    with pytest.raises(ValueError, match="exceeds the native snapshot budget"):
        _finish(measurements=(values,))


@pytest.mark.parametrize(
    "tzinfo_factory",
    [
        pytest.param(lambda: _OffsetRaises(), id="raises"),
        pytest.param(lambda: _OffsetWrongType(), id="not-a-timedelta"),
    ],
)
def test_a_tzinfo_that_fails_to_report_an_offset_is_a_stable_error(tzinfo_factory):
    when = datetime(2026, 9, 28, 8, 1, tzinfo=tzinfo_factory())
    with pytest.raises(ValueError, match="unsupported UTC offset"):
        _finish(finished_at=when)


class _OffsetRaises(__import__("datetime").tzinfo):
    def utcoffset(self, value):
        raise RuntimeError("no offset")

    def dst(self, value):
        return None


class _OffsetWrongType(__import__("datetime").tzinfo):
    def utcoffset(self, value):
        return 3600

    def dst(self, value):
        return None


def test_extreme_instants_are_ordered_without_overflow_in_either_position():
    minus_five = timezone(-timedelta(hours=5))
    late = datetime(9999, 12, 31, 23, 0, tzinfo=minus_five)
    early = datetime(9999, 12, 31, 20, 0, tzinfo=minus_five)
    assert _finish(started_at=early, finished_at=late).finished_at == late
    with pytest.raises(ValueError, match="finished_at cannot precede started_at"):
        _finish(started_at=late, finished_at=early)
    plus_three = timezone(timedelta(hours=3))
    first = datetime(1, 1, 1, 3, 30, tzinfo=plus_three)
    assert _finish(started_at=first, finished_at=datetime(1, 1, 1, 12, tzinfo=timezone.utc)).started_at == first
    with pytest.raises(ValueError, match="finished_at cannot precede started_at"):
        _finish(started_at=datetime(1, 1, 1, 12, tzinfo=timezone.utc), finished_at=first)


def test_every_enum_a_manifest_field_uses_is_in_the_allow_list():
    import typing

    used = set()
    for model in (RunManifest, ResourceReading, TimingMeasurement):
        for field in model.model_fields.values():
            for part in (field.annotation, *typing.get_args(field.annotation)):
                if isinstance(part, type) and issubclass(part, run_manifest_module.Enum):
                    used.add(part)
    assert used and used <= set(run_manifest_module._ALLOWED_ENUMS)


@pytest.mark.parametrize("endless", ["names", "items"])
def test_a_model_extra_mapping_that_never_ends_is_stopped_by_the_budget_while_it_is_read(endless):
    produced = []

    class EndlessExtra(Mapping):
        def __getitem__(self, key):
            raise KeyError(key)

        def __len__(self):
            return 0

        def __iter__(self):
            if endless == "items":
                return iter(())
            for index in range(3_000_000):
                produced.append(index)
                yield f"extra-{index}"

        def items(self):
            for index in range(3_000_000):
                produced.append(index)
                yield f"extra-{index}", index

    resource = measured_memory()
    object.__setattr__(resource, "__pydantic_extra__", EndlessExtra())
    with pytest.raises(ValueError, match="exceeds the native snapshot budget"):
        _finish(resources=(resource,))
    assert len(produced) <= run_manifest_module.MAX_SNAPSHOT_VISITS + 2


def test_a_model_extra_mapping_within_the_budget_is_still_read_once():
    calls = []

    class SmallExtra(Mapping):
        def __getitem__(self, key):
            raise KeyError(key)

        def __len__(self):
            return 0

        def __iter__(self):
            calls.append("iter")
            return iter(())

        def items(self):
            calls.append("items")
            return iter(())

    resource = measured_memory()
    object.__setattr__(resource, "__pydantic_extra__", SmallExtra())
    try:
        _finish(resources=(resource,))
    except ValueError:
        pass  # a closed model may refuse extras; only the single read is pinned here
    assert calls.count("iter") <= 1 and calls.count("items") <= 1


# ── an offset is read once and is an exact timedelta; a missing offset at capture stays missing ──


class _EvilOffset(timedelta):
    """A timedelta subclass whose subtraction lies: any difference it takes part in is minus one day."""

    def __sub__(self, other):
        return timedelta(days=-1)

    def __rsub__(self, other):
        return timedelta(days=-1)


class _FixedTz(__import__("datetime").tzinfo):
    def __init__(self, offset):
        self._offset = offset

    def utcoffset(self, value):
        return self._offset

    def dst(self, value):
        return None


class _ScriptedTz(__import__("datetime").tzinfo):
    """Answers each utcoffset call from a script, then repeats the last answer."""

    def __init__(self, answers):
        self._answers, self.calls = list(answers), 0

    def utcoffset(self, value):
        answer = self._answers[min(self.calls, len(self._answers) - 1)]
        self.calls += 1
        return answer

    def dst(self, value):
        return None


def _revalidated(manifest, **fields):
    return RunManifest.model_validate({**manifest.model_dump(mode="python"), **fields})


def _reversed_by_a_lying_offset():
    finished = datetime(2026, 9, 28, 8, 0, tzinfo=_FixedTz(_EvilOffset(hours=1)))  # 07:00 UTC
    started = datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)  # 09:00 UTC: the run ends before it starts
    return started, finished


def test_a_timedelta_subclass_offset_cannot_reverse_a_run_interval_when_captured():
    started, finished = _reversed_by_a_lying_offset()
    with pytest.raises(ValueError, match="finished_at cannot precede started_at"):
        _finish(started_at=started, finished_at=finished)


def test_a_timedelta_subclass_offset_cannot_reverse_a_run_interval_in_direct_construction():
    manifest = _finish()
    started, finished = _reversed_by_a_lying_offset()
    with pytest.raises(ValueError, match="finished_at cannot precede started_at"):
        _revalidated(manifest, started_at=started, finished_at=finished)
    with pytest.raises(ValueError, match="finished_at cannot precede started_at"):
        _revalidated(manifest, started_at=finished.replace(tzinfo=_FixedTz(_EvilOffset(hours=-1))), finished_at=started.replace(tzinfo=timezone.utc) - timedelta(hours=3))


def test_a_stored_timestamp_keeps_an_exact_fixed_offset_after_validation():
    manifest = _finish()
    shifted = datetime(2026, 9, 28, 9, 30, tzinfo=_FixedTz(_EvilOffset(hours=1)))
    kept = _revalidated(manifest, started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc), finished_at=shifted)
    offset = kept.finished_at.utcoffset()
    assert type(offset) is timedelta and offset == timedelta(hours=1)
    assert kept.finished_at.tzinfo is not shifted.tzinfo  # the caller's tzinfo is not retained


def test_a_tzinfo_that_reports_no_offset_once_and_a_valid_one_afterwards_is_refused_at_capture():
    tz = _ScriptedTz([None, timedelta(0), timedelta(0)])
    when = datetime(2026, 9, 28, 8, 1, tzinfo=tz)
    with pytest.raises(ValueError, match="explicit UTC offset"):
        _finish(finished_at=when)


def test_a_tzinfo_whose_offset_changes_between_reads_is_read_once_in_direct_construction():
    manifest = _finish()
    tz = _ScriptedTz([timedelta(hours=1), timedelta(hours=-5), timedelta(hours=9)])
    when = datetime(2026, 9, 28, 9, 30, tzinfo=tz)
    kept = _revalidated(manifest, started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc), finished_at=when)
    assert tz.calls == 1  # one read by the validator; nothing later asks the caller's tzinfo again
    assert kept.finished_at.utcoffset() == timedelta(hours=1) == kept.finished_at.utcoffset()


@pytest.mark.parametrize(
    "first,then",
    [(timedelta(0), timedelta(hours=-5)), (timedelta(hours=-5), timedelta(0))],
    ids=["utc-then-minus-five", "minus-five-then-utc"],
)
@pytest.mark.parametrize("entry", ["constructor", "model_validate"])
def test_one_datetime_supplied_twice_is_read_with_one_offset_in_direct_construction(
    entry, first, then
) -> None:
    fields = _finish().model_dump(mode="python")
    shared = datetime(2026, 9, 28, 8, 0, tzinfo=_ScriptedTz([first, then]))
    fields.update(started_at=shared, finished_at=shared)
    kept = RunManifest(**fields) if entry == "constructor" else RunManifest.model_validate(fields)
    assert kept.started_at == kept.finished_at  # identical inputs: a zero-length interval
    assert kept.started_at.utcoffset() == first  # the first observed offset is kept
    assert kept.finished_at.utcoffset() == first


def test_the_offset_is_pinned_per_object_only_within_one_validation_call() -> None:
    fields = _finish().model_dump(mode="python")
    tz = _ScriptedTz([timedelta(hours=1), timedelta(hours=-5)])
    shared = datetime(2026, 9, 28, 8, 0, tzinfo=tz)
    fields.update(started_at=shared, finished_at=shared)
    assert RunManifest.model_validate(fields).started_at.utcoffset() == timedelta(hours=1)
    assert tz.calls == 1
    # A later call reads the caller's tzinfo afresh: nothing is remembered between calls.
    assert RunManifest.model_validate(fields).started_at.utcoffset() == timedelta(hours=-5)
    assert tz.calls == 2


def test_two_distinct_datetimes_are_each_read_in_direct_construction() -> None:
    fields = _finish().model_dump(mode="python")
    tz_a, tz_b = _ScriptedTz([timedelta(hours=1)]), _ScriptedTz([timedelta(hours=2)])
    fields.update(
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=tz_a),
        finished_at=datetime(2026, 9, 28, 9, 0, tzinfo=tz_b),
    )
    kept = RunManifest.model_validate(fields)
    assert (tz_a.calls, tz_b.calls) == (1, 1)
    assert kept.started_at.utcoffset() == timedelta(hours=1)
    assert kept.finished_at.utcoffset() == timedelta(hours=2)


class _WrapperView(tuple):
    """A tuple whose public view yields fresh wrappers, each around the same shared subtree."""

    def __iter__(self):
        return iter(self.wrappers())


def _wrapper_view(make_wrappers):
    view_type = type("WrapperView", (_WrapperView,), {"wrappers": staticmethod(make_wrappers)})
    return view_type()


def _nested(depth: int, leaf: object = 0) -> list[object]:
    value: list[object] = [leaf]
    for _index in range(depth - 1):
        value = [value]
    return value


def _enclosed(depth: int, inner: object) -> object:
    for _index in range(depth):
        inner = (inner,)
    return inner


def test_wrappers_around_one_shared_subtree_do_not_replay_it(monkeypatch) -> None:
    shared = list(range(1_000))
    counted = [0]
    original = run_manifest_module._is_subclass

    def counting(*args):
        counted[0] += 1
        return original(*args)

    monkeypatch.setattr(run_manifest_module, "_is_subclass", counting)

    def work(wrappers: int) -> int:
        counted[0] = 0
        graph = run_manifest_module._freeze_model_graphs(
            _wrapper_view(lambda: [(shared,) for _ in range(wrappers)])
        )
        assert graph.visits[0] == wrappers
        return counted[0]

    few, many = work(20), work(80)
    # Each added wrapper costs a constant number of checks; replaying the 1,000-element subtree
    # for every wrapper would add about a thousand.
    assert (many - few) / 60 < 100


@pytest.mark.parametrize("shallow_first", [True, False], ids=["shallow-first", "deep-first"])
def test_a_shared_subtree_reached_deep_is_depth_checked_in_either_order(shallow_first) -> None:
    shared = _nested(40)  # 40 levels: fine alone, too deep below a 40-level wrapper
    shallow, deep = (shared,), _enclosed(40, shared)

    def wrappers():
        return [shallow, deep] if shallow_first else [deep, shallow]

    view = _wrapper_view(wrappers)
    graph = run_manifest_module._freeze_model_graphs(view)
    failure = graph.public_sequences[id(view)].failure
    assert isinstance(failure, ValueError)
    assert run_manifest_module.SNAPSHOT_BUDGET_MESSAGE in str(failure)
    with pytest.raises(ValueError, match=run_manifest_module.SNAPSHOT_BUDGET_MESSAGE):
        run_manifest_module._snapshot_native(view, _frozen_graph=graph)


def test_a_shared_subtree_reached_only_within_the_depth_bound_is_accepted() -> None:
    shared = _nested(20)
    view = _wrapper_view(lambda: [(shared,), _enclosed(20, shared), (shared,)])
    graph = run_manifest_module._freeze_model_graphs(view)
    assert graph.public_sequences[id(view)].failure is None
    assert graph.visits[0] == 3


# ── capture is charged before it retains; one datetime gets one offset per call ──


def _model_with_stored_entries(n: int):
    model = measured_memory()
    stored = BaseModel.__dict__["__dict__"].__get__(model, BaseModel)
    for index in range(n):
        dict.__setitem__(stored, f"extra_{index}", index)
    return model


@pytest.mark.parametrize(
    "build",
    [
        lambda n: list(range(n)),
        lambda n: tuple(range(n)),
        lambda n: {f"k{i}": i for i in range(n)},
        lambda n: _model_with_stored_entries(n),
    ],
    ids=["list", "tuple", "dict", "model"],
)
def test_native_capture_is_charged_before_its_entries_are_retained(build) -> None:
    oversized = build(run_manifest_module.MAX_SNAPSHOT_VISITS + 1)
    with pytest.raises(ValueError, match=run_manifest_module.SNAPSHOT_BUDGET_MESSAGE):
        run_manifest_module._freeze_model_graphs(oversized)


def test_native_capture_within_the_budget_is_kept() -> None:
    graph = run_manifest_module._freeze_model_graphs([[] for _ in range(1_000)])
    assert len(graph.replay_entries) == 1_001


class _FlippingOffset(tzinfo):
    """Reports one UTC offset on the first call and another on every later call."""

    def __init__(self, first: timedelta, then: timedelta) -> None:
        self.answers = [first, then]
        self.calls = 0

    def utcoffset(self, dt):
        self.calls += 1
        return self.answers[0] if self.calls == 1 else self.answers[1]

    def dst(self, dt):
        return None


@pytest.mark.parametrize(
    "first,then",
    [(timedelta(0), timedelta(hours=-5)), (timedelta(hours=-5), timedelta(0))],
    ids=["utc-then-minus-five", "minus-five-then-utc"],
)
def test_one_datetime_supplied_twice_is_read_with_one_offset(first, then) -> None:
    shared = datetime(2026, 9, 28, 8, 0, tzinfo=_FlippingOffset(first, then))
    finished = finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=shared,
        finished_at=shared,
        resources=(measured_memory(),),
        measurements=(timing(),),
    )
    assert finished.started_at == finished.finished_at  # identical inputs: a zero-length interval
    assert finished.started_at.utcoffset() == first  # the first observed offset is kept


def test_an_oversized_native_container_is_refused_before_its_entries_are_copied() -> None:
    import tracemalloc

    oversized = [[] for _ in range(run_manifest_module.MAX_SNAPSHOT_VISITS + 1)]
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match=run_manifest_module.SNAPSHOT_BUDGET_MESSAGE):
            run_manifest_module._freeze_model_graphs(oversized)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Copying the 65,537 entries would allocate a tuple of about half a megabyte; refusing first
    # allocates only bookkeeping.
    assert peak < 128 * 1024, peak


# ---- the pre-copy size check does not spend the snapshot's own budget a second time

def test_a_flat_native_list_within_the_declared_budget_is_still_accepted() -> None:
    """``MAX_SNAPSHOT_VISITS`` bounds the snapshot; a 40,000-entry list inside it keeps passing capture and snapshot."""

    value = list(range(40_000))
    frozen = run_manifest_module._freeze_model_graphs(value)
    run_manifest_module._snapshot_native(value, _frozen_graph=frozen)


def test_the_largest_accepted_native_list_is_the_snapshots_own_limit() -> None:
    limit = run_manifest_module.MAX_SNAPSHOT_VISITS
    for size, accepted in ((limit - 1, True), (limit, False)):
        value = list(range(size))
        try:
            frozen = run_manifest_module._freeze_model_graphs(value)
            run_manifest_module._snapshot_native(value, _frozen_graph=frozen)
            verdict = True
        except ValueError as error:
            assert str(error) == run_manifest_module.SNAPSHOT_BUDGET_MESSAGE
            verdict = False
        assert verdict is accepted, size


def test_finish_run_keeps_its_capacity_for_timing_measurements() -> None:
    """3,000 distinct timing measurements were accepted before the pre-copy charge (the limit was 3,444)."""

    base = timing()
    measurements = tuple(base.model_copy(update={"name": f"m{i}"}) for i in range(3_000))
    finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc),
        resources=(measured_memory(),),
        measurements=measurements,
    )


def test_native_entries_across_containers_are_bounded_together_before_any_is_copied() -> None:
    """Many containers that are each within the budget add up: the freeze refuses the graph, not just one container."""

    import tracemalloc

    graph = [list(range(1_000)) for _ in range(100)]  # 100,100 entries in total
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match=run_manifest_module.SNAPSHOT_BUDGET_MESSAGE):
            run_manifest_module._freeze_model_graphs(graph)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2_000_000, peak  # about 66 of the 100 lists were counted and copied, not all of them plus a snapshot


# ---- one offset reading per datetime on every construction path

class _FlipZone(tzinfo):
    """Answers one offset on the first read and another afterwards (outside the envelope)."""

    def __init__(self, first, later):
        self.reads, self.first, self.later = 0, first, later

    def utcoffset(self, dt):
        self.reads += 1
        return self.first if self.reads == 1 else self.later

    def dst(self, dt):
        return timedelta(0)

def _completed_with_one_instant(zone):
    data = completed_manifest().model_dump()
    when = datetime(2026, 10, 1, 8, 0, tzinfo=zone)
    data["started_at"] = data["finished_at"] = when
    return data

class _Envelope(BaseModel):
    manifest: RunManifest

@pytest.mark.parametrize("first, later", [(timedelta(hours=-5), timedelta(0)), (timedelta(0), timedelta(hours=-5))])
@pytest.mark.parametrize("path", ["type_adapter", "nested_model", "list_of_manifests"])
def test_one_datetime_supplied_twice_is_read_once_on_every_construction_path(path, first, later):
    """A TypeAdapter or an enclosing model does not run RunManifest's own entry points; the same instant must
    still be one reading (neither refused as reversed nor accepted as a five-hour run)."""

    zone = _FlipZone(first, later)
    data = _completed_with_one_instant(zone)
    if path == "type_adapter":
        manifest = pydantic.TypeAdapter(RunManifest).validate_python(data)
    elif path == "nested_model":
        manifest = _Envelope.model_validate({"manifest": data}).manifest
    else:
        manifest = pydantic.TypeAdapter(list[RunManifest]).validate_python([data, _completed_with_one_instant(_FlipZone(first, later))])[0]
    assert manifest.started_at == manifest.finished_at
    assert zone.reads == 1

def test_two_distinct_datetimes_are_each_read_once_on_a_type_adapter():
    data = completed_manifest().model_dump()
    start_zone, finish_zone = _FlipZone(timedelta(0), timedelta(0)), _FlipZone(timedelta(0), timedelta(0))
    data["started_at"] = datetime(2026, 10, 1, 8, 0, tzinfo=start_zone)
    data["finished_at"] = datetime(2026, 10, 1, 9, 0, tzinfo=finish_zone)
    manifest = pydantic.TypeAdapter(RunManifest).validate_python(data)
    assert manifest.finished_at - manifest.started_at == timedelta(hours=1)
    assert (start_zone.reads, finish_zone.reads) == (1, 1)


def _refuse_started_at_only() -> Callable[[], tzinfo | None]:
    import weakref

    zone = _FlipZone(timedelta(0), timedelta(0))
    data = completed_manifest().model_dump()
    data["started_at"] = datetime(2026, 10, 1, 8, 0, tzinfo=zone)
    data.pop("finished_at")
    # The record is refused (a finished run needs both); pytest.raises would keep the traceback alive.
    refused = False
    try:
        pydantic.TypeAdapter(RunManifest).validate_python(data)
    except ValidationError as error:
        refused = "requires started_at and finished_at" in str(error)
    assert refused
    return weakref.ref(zone)


def test_started_at_without_finished_at_does_not_keep_the_caller_datetime():
    """The pairing slot must not hold the caller's datetime past the manifest it came from."""
    import gc

    gone = _refuse_started_at_only()
    gc.collect()
    assert gone() is None


# ---- datetime subclasses are kept as exact datetimes


class _HookedDatetime(datetime):
    hooks = 0

    def __new__(cls, *args, **kwargs):
        _HookedDatetime.hooks += 1
        return super().__new__(cls, *args, **kwargs)

    def __sub__(self, other):
        _HookedDatetime.hooks += 1
        return timedelta(days=-1)

    def replace(self, *args, **kwargs):
        _HookedDatetime.hooks += 1
        return super().replace(*args, **kwargs)


@pytest.mark.parametrize("path", ["init", "model_validate", "type_adapter"])
def test_datetime_subclass_runs_no_caller_code_and_is_stored_as_an_exact_datetime(path):
    start = _HookedDatetime(2026, 10, 1, 8, 0, 5, 7, tzinfo=timezone(timedelta(hours=2)), fold=1)
    finish = _HookedDatetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    data = {**completed_manifest().model_dump(), "started_at": start, "finished_at": finish}
    _HookedDatetime.hooks = 0
    if path == "init":
        manifest = RunManifest(**data)
    elif path == "model_validate":
        manifest = RunManifest.model_validate(data)
    else:
        manifest = pydantic.TypeAdapter(RunManifest).validate_python(data)
    assert _HookedDatetime.hooks == 0
    assert type(manifest.started_at) is datetime and type(manifest.finished_at) is datetime
    assert manifest.started_at == datetime(2026, 10, 1, 8, 0, 5, 7, tzinfo=timezone(timedelta(hours=2)))
    assert manifest.started_at.fold == 1
    assert manifest.finished_at - manifest.started_at == timedelta(hours=3, seconds=-5, microseconds=-7)


def test_exact_datetimes_keep_wall_time_offset_and_fold():
    zone = ZoneInfo("America/New_York")
    start = datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=1)
    finish = datetime(2026, 11, 1, 1, 45, tzinfo=zone, fold=1)
    manifest = RunManifest(**{**completed_manifest().model_dump(), "started_at": start, "finished_at": finish})
    assert manifest.started_at.utcoffset() == timedelta(hours=-5)
    assert (manifest.started_at.hour, manifest.started_at.minute, manifest.started_at.fold) == (1, 30, 1)
    assert manifest.finished_at - manifest.started_at == timedelta(minutes=15)


# ---- error locations never run a caller key's __repr__


class _LoudKey(str):
    reprs = 0

    def __repr__(self):
        _LoudKey.reprs += 1
        return "X" * 1_000_000


def test_an_undeclared_stored_key_is_named_without_running_its_repr():
    manifest = completed_manifest()
    dict.__setitem__(BASE_MODEL_DICT_DESCRIPTOR.__get__(manifest, BaseModel), _LoudKey("sneaky"), 1)
    _LoudKey.reprs = 0
    with pytest.raises(ValueError, match=r"undeclared stored fields at \$: 'sneaky'$"):
        serialize_run_manifest(manifest)
    assert _LoudKey.reprs == 0


def test_mapping_keys_on_the_walk_are_never_asked_for_their_repr():
    manifest = completed_manifest()
    reading = manifest.resources[0].model_dump()

    class KeyedResource(dict):
        pass

    resource = KeyedResource({_LoudKey(name): value for name, value in reading.items()})
    dict.__setitem__(BASE_MODEL_DICT_DESCRIPTOR.__get__(manifest, BaseModel), "resources", (resource,))
    _LoudKey.reprs = 0
    with contextlib.suppress(ValueError, ValidationError):
        serialize_run_manifest(manifest)
    assert _LoudKey.reprs == 0


@pytest.mark.parametrize(
    "key, label",
    [
        ("plain", "$['plain']"),
        (_LoudKey("loud"), "$['loud']"),
        (7, "$[7]"),
        (True, "$[True]"),
        (2**10_000, "$[<int key of 10001 bits>]"),
        ("k" * 500, "$['" + "k" * 64 + "'…]"),
        (frozenset(), "$[<frozenset key>]"),
    ],
    ids=["str", "str-subclass", "int", "bool", "huge-int", "long-str", "other"],
)
def test_error_locations_label_keys_from_their_exact_text(key, label):
    _LoudKey.reprs = 0
    with pytest.raises(ValueError) as refused:
        run_manifest_module._require_native_snapshot({key: object()})
    assert str(refused.value) == f"unsupported value type at {label}"
    assert _LoudKey.reprs == 0


def test_error_locations_still_name_nested_fields_and_indexes():
    with pytest.raises(ValueError) as refused:
        run_manifest_module._require_native_snapshot({"resources": [{"name": [1, object()]}]})
    assert str(refused.value) == "unsupported value type at $['resources'][0]['name'][1]"


# ---- the undeclared-field message lists a bounded number of names


def test_many_undeclared_keys_give_a_bounded_error_naming_the_first_few_and_a_count():
    manifest = completed_manifest()
    storage = BASE_MODEL_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
    for index in range(20_000):
        dict.__setitem__(storage, f"undeclared_{index:05d}_" + "y" * 200, 1)
    with pytest.raises(ValueError) as refused:
        serialize_run_manifest(manifest)
    message = str(refused.value)
    assert len(message) < 1_000
    assert message.startswith("undeclared stored fields at $: 'undeclared_00000_")
    assert message.endswith("… and 19995 more")
    assert message.count("'undeclared_") == 5


def test_a_few_undeclared_keys_are_all_listed():
    manifest = completed_manifest()
    storage = BASE_MODEL_DICT_DESCRIPTOR.__get__(manifest, BaseModel)
    for name in ("zeta", "alpha", "mid"):
        dict.__setitem__(storage, name, 1)
    with pytest.raises(ValueError) as refused:
        serialize_run_manifest(manifest)
    assert str(refused.value) == "undeclared stored fields at $: 'alpha', 'mid', 'zeta'"


# ---- a rejected validation leaves nothing behind (the pairing slot is scoped to one manifest)


def _validate_on(path, data):
    if path == "type_adapter":
        return pydantic.TypeAdapter(RunManifest).validate_python(data)
    if path == "nested_model":
        return _Envelope.model_validate({"manifest": data}).manifest
    if path == "list_of_manifests":
        return pydantic.TypeAdapter(list[RunManifest]).validate_python([data])[0]
    if path == "model_validate":
        return RunManifest.model_validate(data)
    return RunManifest(**data)


_ALL_PATHS = ["type_adapter", "nested_model", "list_of_manifests", "model_validate", "init"]
_BAD_FINISHES = [object(), "not a date", 12345, [], datetime(2026, 10, 1, 9, 0)]  # the last is naive


def _reject_with_bad_finish(path, bad):
    import weakref

    zone = _FlipZone(timedelta(0), timedelta(0))
    data = completed_manifest().model_dump()
    data["started_at"] = datetime(2026, 10, 1, 8, 0, tzinfo=zone)
    data["finished_at"] = bad
    refused = False
    try:
        _validate_on(path, data)
    except (ValidationError, ValueError):
        refused = True
    assert refused
    return weakref.ref(zone)


@pytest.mark.parametrize("bad", range(len(_BAD_FINISHES)))
@pytest.mark.parametrize("path", _ALL_PATHS)
def test_a_valid_start_with_an_invalid_finish_does_not_keep_the_caller_datetime(path, bad):
    import gc

    gone = _reject_with_bad_finish(path, _BAD_FINISHES[bad])
    gc.collect()
    assert gone() is None


@pytest.mark.parametrize("path", ["type_adapter", "nested_model", "list_of_manifests"])
def test_a_rejected_validation_cannot_change_the_next_objects_reading(path):
    """The first call reads ``shared`` once (+1 h) and is refused for its finished_at.  In the second call the
    started_at is refused and finished_at is the same object: it must be read afresh (-5 h), never taken from
    the first call's pairing."""

    zone = _FlipZone(timedelta(hours=1), timedelta(hours=-5))
    shared = datetime(2026, 10, 1, 8, 0, tzinfo=zone)
    first = completed_manifest().model_dump()
    first["started_at"], first["finished_at"] = shared, object()
    with pytest.raises((ValidationError, ValueError)):
        _validate_on(path, first)
    assert zone.reads == 1
    second = completed_manifest().model_dump()
    second["started_at"], second["finished_at"] = object(), shared
    with pytest.raises((ValidationError, ValueError)):
        _validate_on(path, second)
    assert zone.reads == 2  # read again for this object's own validation, not replayed from the rejected one


# ---- a key's class name is read natively, copied to an exact str and cut


def _key_of_type_named(name):
    key_type = type("K", (), {"__hash__": object.__hash__})
    type.__dict__["__name__"].__set__(key_type, name)
    return key_type()


class _LoudName(str):
    formats = 0

    def __format__(self, spec):
        _LoudName.formats += 1
        raise RuntimeError("caller __format__ ran")


def test_a_key_class_name_that_is_a_str_subclass_runs_no_caller_format():
    from neuralscape_bench.run_manifest import _key_label

    _LoudName.formats = 0
    label = _key_label(_key_of_type_named(_LoudName("Sneaky")))
    assert _LoudName.formats == 0
    assert label == "<Sneaky key>" and type(label) is str


def test_a_million_character_key_class_name_is_cut_to_the_label_bound():
    from neuralscape_bench.run_manifest import _KEY_LABEL_CHARS, _key_label

    label = _key_label(_key_of_type_named("N" * 1_000_000))
    assert len(label) <= _KEY_LABEL_CHARS + len("<…")  + len(" key>")
    assert label.startswith("<" + "N" * 10) and label.endswith("… key>")


def test_a_str_subclass_class_name_cut_and_a_short_one_unchanged():
    from neuralscape_bench.run_manifest import _key_label

    assert _key_label(_key_of_type_named("Short")) == "<Short key>"
    long_loud = _key_label(_key_of_type_named(_LoudName("L" * 500)))
    assert type(long_loud) is str and len(long_loud) < 100


def test_a_diagnostic_naming_a_hostile_key_type_stays_bounded_and_runs_no_caller_code():
    manifest = completed_manifest()
    _LoudName.formats = 0
    dict.__setitem__(BASE_MODEL_DICT_DESCRIPTOR.__get__(manifest, BaseModel), _key_of_type_named(_LoudName("X" * 1_000_000)), 1)
    with pytest.raises(ValueError) as caught:
        serialize_run_manifest(manifest)
    assert _LoudName.formats == 0 and len(str(caught.value)) < 2_000


class _HostileName(str):
    """A class name whose every str method is caller code."""

    calls = 0

    def _hit(self, *args, **kwargs):
        _HostileName.calls += 1
        raise RuntimeError("caller code ran")

    __format__ = __str__ = __repr__ = __len__ = __getitem__ = __eq__ = __hash__ = __iter__ = __contains__ = _hit
    startswith = endswith = encode = join = split = replace = lower = upper = strip = _hit


def test_a_class_name_with_every_str_method_hostile_is_read_with_the_base_methods_only():
    from neuralscape_bench.run_manifest import _key_label

    key = _key_of_type_named(_HostileName("Quiet" * 40))
    _HostileName.calls = 0
    label = _key_label(key)
    assert _HostileName.calls == 0
    assert type(label) is str and label.startswith("<QuietQuiet") and label.endswith("… key>")


def test_concurrent_validations_never_pair_across_threads():
    import threading

    errors: list[BaseException] = []

    def worker(seed: int) -> None:
        try:
            for _ in range(25):
                zone = _FlipZone(timedelta(hours=-5), timedelta(0))
                manifest = pydantic.TypeAdapter(RunManifest).validate_python(_completed_with_one_instant(zone))
                assert manifest.started_at == manifest.finished_at and zone.reads == 1
                bad = completed_manifest().model_dump()
                bad["started_at"], bad["finished_at"] = datetime(2026, 10, 1, 8, 0, tzinfo=_FlipZone(timedelta(0), timedelta(0))), object()
                with pytest.raises(ValidationError):
                    pydantic.TypeAdapter(RunManifest).validate_python(bad)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_json_validation_is_unaffected_by_the_pairing_step():
    manifest = completed_manifest()
    again = RunManifest.model_validate_json(serialize_run_manifest(manifest))
    assert again == manifest
    assert pydantic.TypeAdapter(RunManifest).validate_json(serialize_run_manifest(manifest)) == manifest


# ---- JSON validation never fills the pairing slot, so no input mode leaves anything behind


def _pending_slot():
    """What a finished validation leaves behind: nothing (the scope table is restored to ``None``)."""

    from neuralscape_bench.run_manifest import _TIMESTAMP_PINS

    return _TIMESTAMP_PINS.get()


@pytest.mark.parametrize("entry", ["model_validate_json", "type_adapter_validate_json"])
@pytest.mark.parametrize("outcome", ["valid", "rejected"])
def test_json_validation_leaves_the_pairing_slot_empty(entry, outcome):
    manifest = completed_manifest()
    text = serialize_run_manifest(manifest)
    if outcome == "rejected":
        document = json.loads(text)
        document["finished_at"] = "not a date"
        text = json.dumps(document)
    run = (
        (lambda: RunManifest.model_validate_json(text))
        if entry == "model_validate_json"
        else (lambda: pydantic.TypeAdapter(RunManifest).validate_json(text))
    )
    if outcome == "valid":
        run()
    else:
        with pytest.raises(ValidationError):
            run()
    assert _pending_slot() is None


def test_the_pairing_slot_is_empty_after_every_python_outcome_too():
    for bad in (object(), "x", datetime(2026, 10, 1, 9, 0)):
        data = completed_manifest().model_dump()
        data["finished_at"] = bad
        with pytest.raises((ValidationError, ValueError)):
            pydantic.TypeAdapter(RunManifest).validate_python(data)
        assert _pending_slot() is None
    pydantic.TypeAdapter(RunManifest).validate_python(completed_manifest().model_dump())
    assert _pending_slot() is None


# ---- a manifest built from datetime subclasses serializes and round-trips (the stored timestamps are exact)


@pytest.mark.parametrize("path", ["init", "model_validate", "type_adapter"])
def test_a_manifest_built_from_datetime_subclasses_serializes_and_round_trips(path):
    class Benign(datetime):
        pass

    data = {
        **completed_manifest().model_dump(),
        "started_at": Benign(2026, 10, 1, 8, 0, tzinfo=timezone.utc),
        "finished_at": Benign(2026, 10, 1, 12, 0, tzinfo=timezone(timedelta(hours=2))),
    }
    if path == "init":
        manifest = RunManifest(**data)
    elif path == "model_validate":
        manifest = RunManifest.model_validate(data)
    else:
        manifest = pydantic.TypeAdapter(RunManifest).validate_python(data)
    assert type(manifest.started_at) is datetime and type(manifest.finished_at) is datetime
    text = serialize_run_manifest(manifest)  # an inexact leaf would be refused here as an unsupported value type
    assert validate_run_manifest_json(text) == manifest


# ---- validation that nests inside another validation never erases the outer one's pending reading


def _source_whose_finish_builds_another_manifest(zone, *, nest):
    """An attribute source (for ``from_attributes``) whose ``finished_at`` property validates another manifest first."""

    fields = completed_manifest().model_dump()
    shared = datetime(2026, 10, 1, 8, 0, tzinfo=zone)
    namespace = {name: value for name, value in fields.items() if name not in ("started_at", "finished_at")}
    namespace["started_at"] = shared

    def finished_at(self):
        nest()  # a whole other manifest is validated while this one's finished_at is being read
        return shared

    namespace["finished_at"] = property(finished_at)
    return type("Source", (), namespace)()


def _another_manifest_through(route):
    data = completed_manifest().model_dump()
    if route == "init":
        return RunManifest(**data)
    if route == "model_validate":
        return RunManifest.model_validate(data)
    return pydantic.TypeAdapter(RunManifest).validate_python(data)


@pytest.mark.parametrize("first, later", [(timedelta(0), timedelta(hours=-5)), (timedelta(hours=-5), timedelta(0))])
@pytest.mark.parametrize("route", ["init", "model_validate", "type_adapter"])
def test_a_manifest_built_while_reading_finished_at_leaves_the_outer_reading_alone(route, first, later):
    zone = _FlipZone(first, later)
    source = _source_whose_finish_builds_another_manifest(zone, nest=lambda: _another_manifest_through(route))
    manifest = pydantic.TypeAdapter(RunManifest).validate_python(source, from_attributes=True)
    assert manifest.started_at == manifest.finished_at  # one object, one reading
    assert manifest.started_at.utcoffset() == first
    assert zone.reads == 1


def test_a_deeper_nest_and_a_nest_that_raises_still_leave_the_outer_reading_alone():
    zone = _FlipZone(timedelta(hours=-5), timedelta(0))

    def nest():
        inner_zone = _FlipZone(timedelta(hours=1), timedelta(hours=2))
        inner = _source_whose_finish_builds_another_manifest(inner_zone, nest=lambda: _another_manifest_through("type_adapter"))
        pydantic.TypeAdapter(RunManifest).validate_python(inner, from_attributes=True)
        assert inner_zone.reads == 1
        with pytest.raises(ValidationError):
            data = completed_manifest().model_dump()
            data["finished_at"] = object()
            pydantic.TypeAdapter(RunManifest).validate_python(data)

    manifest = pydantic.TypeAdapter(RunManifest).validate_python(
        _source_whose_finish_builds_another_manifest(zone, nest=nest), from_attributes=True
    )
    assert manifest.started_at == manifest.finished_at and zone.reads == 1


def test_threads_that_nest_validations_never_cross_readings():
    import threading

    errors: list[BaseException] = []

    def worker():
        try:
            for _ in range(20):
                zone = _FlipZone(timedelta(hours=-5), timedelta(0))
                source = _source_whose_finish_builds_another_manifest(zone, nest=lambda: _another_manifest_through("type_adapter"))
                manifest = pydantic.TypeAdapter(RunManifest).validate_python(source, from_attributes=True)
                assert manifest.started_at == manifest.finished_at and zone.reads == 1
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_nothing_is_pending_after_nested_validations_in_any_order():
    zone = _FlipZone(timedelta(0), timedelta(hours=-5))
    pydantic.TypeAdapter(RunManifest).validate_python(
        _source_whose_finish_builds_another_manifest(zone, nest=lambda: _another_manifest_through("type_adapter")), from_attributes=True
    )
    assert _pending_slot() is None


def test_validations_aborted_while_reading_finished_at_leave_nothing_and_pair_nothing_with_the_next():
    def explode():
        raise RuntimeError("the attribute source failed")

    for _ in range(100):
        source = _source_whose_finish_builds_another_manifest(_FlipZone(timedelta(0), timedelta(0)), nest=explode)
        with pytest.raises(ValidationError):  # pydantic reports a failing attribute as a validation error
            pydantic.TypeAdapter(RunManifest).validate_python(source, from_attributes=True)
        assert _pending_slot() is None  # the scope is restored in a finally, however the validation ended
    zone = _FlipZone(timedelta(hours=-5), timedelta(0))  # the next validation reads and pairs on its own
    manifest = pydantic.TypeAdapter(RunManifest).validate_python(
        _source_whose_finish_builds_another_manifest(zone, nest=lambda: None), from_attributes=True
    )
    assert manifest.started_at == manifest.finished_at and zone.reads == 1


@pytest.mark.parametrize("aborted", [1, 15, 16, 17, 40, 100])
def test_many_aborted_nested_validations_cannot_evict_the_outer_validations_reading(aborted):
    """The outer ``finished_at`` read runs ``aborted`` nested validations that each fail while reading an attribute."""

    def explode():
        raise RuntimeError("the attribute source failed")

    def nest():
        for _ in range(aborted):
            inner = _source_whose_finish_builds_another_manifest(_FlipZone(timedelta(0), timedelta(0)), nest=explode)
            with pytest.raises(ValidationError):
                pydantic.TypeAdapter(RunManifest).validate_python(inner, from_attributes=True)

    zone = _FlipZone(timedelta(hours=-5), timedelta(0))
    manifest = pydantic.TypeAdapter(RunManifest).validate_python(
        _source_whose_finish_builds_another_manifest(zone, nest=nest), from_attributes=True
    )
    assert manifest.started_at == manifest.finished_at  # one object, one reading, however many nested failures
    assert manifest.started_at.utcoffset() == timedelta(hours=-5) and zone.reads == 1


# ---- a public subclass shared by many wrappers is discovered once, and a repeated key is refused

class _HiddenBackingTuple(tuple):
    """A tuple subclass whose public view is empty: its native backing is reachable only defensively."""

    def __iter__(self):
        return iter(())


def test_wrappers_around_one_shared_public_subclass_do_not_rescan_its_hidden_backing(monkeypatch) -> None:
    shared = _HiddenBackingTuple(range(1_000))
    counted = [0]

    def counting(original):
        def wrapper(*args):
            counted[0] += 1
            return original(*args)
        return wrapper

    for name in ("_is_subclass", "_is_instance", "_is_exact_type"):
        monkeypatch.setattr(run_manifest_module, name, counting(getattr(run_manifest_module, name)))

    def work(wrappers: int) -> int:
        counted[0] = 0
        run_manifest_module._freeze_model_graphs(_wrapper_view(lambda: [(shared,) for _ in range(wrappers)]))
        return counted[0]

    few, many = work(20), work(80)
    # Each added wrapper costs a constant number of checks; rescanning the 1,000-entry backing for every
    # wrapper would add about a thousand per wrapper.
    assert (many - few) / 60 < 100, (few, many)


class _PairsMapping(Mapping):
    """A public mapping whose ``items()`` yields exactly the pairs given, repeats included."""

    def __init__(self, pairs):
        self.pairs = list(pairs)

    def __iter__(self):
        return iter(dict(self.pairs))

    def __len__(self):
        return len(dict(self.pairs))

    def __getitem__(self, key):
        return dict(self.pairs)[key]

    def items(self):
        return iter(self.pairs)


def _finish_with_resource(resource):
    return finish_run(
        planned_manifest(),
        state=RunState.COMPLETED,
        started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
        resources=(resource,),
        measurements=(timing(),),
    )


@pytest.mark.parametrize("pair_order", [(True, 1), (1, True), (5, 7)], ids=["bool-then-int", "int-then-bool", "two-ints"])
def test_a_repeated_key_in_a_public_mapping_is_refused_not_overwritten(pair_order) -> None:
    base = measured_memory().model_dump()
    others = [(key, value) for key, value in base.items() if key != "observed_value"]
    first, second = pair_order
    with pytest.raises(ValueError, match="duplicate key"):
        _finish_with_resource(_PairsMapping(others + [("observed_value", first), ("observed_value", second)]))


def test_keys_that_compare_equal_but_are_distinct_objects_are_a_repeat() -> None:
    class SpelledAgain(str):
        """Equal to, and hashing like, its base text, but a different object."""

    base = measured_memory().model_dump()
    others = [(key, value) for key, value in base.items() if key != "observed_value"]
    pairs = others + [("observed_value", True), (SpelledAgain("observed_value"), 1)]
    with pytest.raises(ValueError, match="duplicate key"):
        _finish_with_resource(_PairsMapping(pairs))


def test_non_text_keys_that_compare_equal_are_refused_by_type() -> None:
    base = measured_memory().model_dump()
    pairs = list(base.items()) + [(1, "a"), (True, "b")]   # 1 == True: never reaches a rebuild
    with pytest.raises(ValueError, match="unsupported key type"):
        _finish_with_resource(_PairsMapping(pairs))


def test_a_public_mapping_without_a_repeat_is_still_accepted() -> None:
    base = measured_memory().model_dump()
    manifest = _finish_with_resource(_PairsMapping(list(base.items())))
    assert manifest.resources[0].observed_value == base["observed_value"]


def test_a_text_subclass_key_in_a_public_mapping_is_never_hashed_by_the_snapshot() -> None:
    calls: list[int] = []

    class CountedKey(str):
        def __hash__(self):
            calls.append(1)
            return str.__hash__(self)

    base = measured_memory().model_dump()
    pairs = [(CountedKey(key), value) for key, value in base.items()]
    resource = _PairsMapping(pairs)
    calls.clear()                      # building the caller's own pairs is not the snapshot's doing
    _finish_with_resource(resource)    # accepted: the keys are the model's own field names
    assert calls == [], len(calls)     # read as exact text: the snapshot runs no key hook


# ---- a text key whose hash changes after capture cannot collapse two captured entries

class _ShiftyKey(str):
    """Equal to its base text, but it hashes differently until ``settle()``; then it hashes like the plain text."""

    settled = False

    def __hash__(self):
        return str.__hash__(self) if self.settled else str.__hash__(self) ^ 0x5A5A5A

    def settle(self):
        self.settled = True


def _with_shifty_duplicate(container, field="observed_value"):
    """Store ``field: True`` and an equal-text shifty key holding 1 in ``container`` while their hashes differ, then settle."""

    shifty = _ShiftyKey(field)
    dict.__setitem__(container, field, True)
    dict.__setitem__(container, shifty, 1)
    assert dict.__len__(container) >= 2
    shifty.settle()
    return shifty


def test_model_storage_with_a_key_that_changes_hash_is_refused_as_a_repeat() -> None:
    reading = measured_memory()
    _with_shifty_duplicate(BASE_MODEL_DICT_DESCRIPTOR.__get__(reading, pydantic.BaseModel))
    with pytest.raises(ValueError, match="duplicate key"):
        _finish_with_resource(reading)


def test_an_exact_dict_with_a_key_that_changes_hash_is_refused_as_a_repeat() -> None:
    fields = dict(measured_memory().model_dump())
    _with_shifty_duplicate(fields)
    with pytest.raises(ValueError, match="duplicate key"):
        _finish_with_resource(fields)


def test_a_public_mapping_with_equal_text_keys_of_different_hashes_is_refused_as_a_repeat() -> None:
    base = measured_memory().model_dump()
    others = [(key, value) for key, value in base.items() if key != "observed_value"]
    pairs = others + [("observed_value", True), (_ShiftyKey("observed_value"), 1)]
    with pytest.raises(ValueError):   # refused (by the model's own validation: the exact key wins, the other is extra)
        _finish_with_resource(_PairsMapping(pairs))


def test_a_single_text_subclass_key_is_read_as_its_exact_text_and_accepted() -> None:
    fields = {(_ShiftyKey(key) if key == "observed_value" else key): value for key, value in measured_memory().model_dump().items()}
    manifest = _finish_with_resource(fields)
    assert manifest.resources[0].observed_value == measured_memory().observed_value


def test_the_snapshot_runs_no_hook_of_a_text_subclass_key() -> None:
    log: list[str] = []

    class Counting(str):
        def __hash__(self):
            log.append("hash")
            return str.__hash__(self)

        def __eq__(self, other):
            log.append("eq")
            return str.__eq__(self, other)

    fields = {(Counting(key) if key == "observed_value" else key): value for key, value in measured_memory().model_dump().items()}
    log.clear()   # building the caller's own dict is not the snapshot's doing
    _finish_with_resource(fields)
    assert log == [], log



def test_a_text_key_is_copied_with_the_base_method_not_the_keys_own_str() -> None:
    log: list[str] = []

    class Hostile(str):
        def __str__(self):
            log.append("__str__")
            return "something-else"

    fields = {(Hostile(key) if key == "observed_value" else key): value for key, value in measured_memory().model_dump().items()}
    manifest = _finish_with_resource(fields)
    assert manifest.resources[0].observed_value == measured_memory().observed_value and log == []


def test_an_object_that_only_claims_to_be_text_is_not_copied_as_text() -> None:
    class Claims:
        @property
        def __class__(self):   # type: ignore[override]
            return str

        def __hash__(self):
            return 1

    fields = dict(measured_memory().model_dump())
    fields[Claims()] = 1
    with pytest.raises(ValueError):   # an unsupported key is refused as an undeclared field, never a TypeError from a bad copy
        _finish_with_resource(fields)


# ---- the public-mapping projection reads keys as exact text too: a twin of a field name cannot hide a value

class _NeverEqualKey(str):
    """Hashes like its text but never compares equal to anything: two keys of the same text stay two entries."""

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __hash__(self):
        return str.__hash__(self)


def _twin_pairs(kind, exact_value, twin_value, twin_first):
    base = measured_memory().model_dump()
    others = [(key, value) for key, value in base.items() if key != "observed_value"]
    twin = _NeverEqualKey("observed_value") if kind == "never-equal" else _ShiftyKey("observed_value")
    pairs = [("observed_value", exact_value), (twin, twin_value)]
    if twin_first:
        pairs.reverse()
    return others, pairs, twin


class _PublicDict(dict):
    """A dict subclass: it is read through the public-mapping route, not as an exact dictionary."""


def kind_settles(twin):
    return getattr(twin, "will_settle", True)


def _public_dict_from(others, pairs, twin):
    container = _PublicDict()
    for key, value in others:
        dict.__setitem__(container, key, value)
    for key, value in pairs:
        dict.__setitem__(container, key, value)
    if isinstance(twin, _ShiftyKey) and kind_settles(twin):
        twin.settle()
    return container


@pytest.mark.parametrize("container", ["mapping", "dict-subclass"])
@pytest.mark.parametrize("twin_first", [False, True], ids=["exact-first", "twin-first"])
@pytest.mark.parametrize("values", [(True, 1), (1, True)], ids=["exact-invalid", "twin-invalid"])
@pytest.mark.parametrize("kind", ["shifty-settled", "shifty-unsettled", "never-equal"])
def test_a_twin_of_a_field_name_in_a_public_mapping_is_refused_as_a_repeat(kind, values, twin_first, container) -> None:
    others, pairs, twin = _twin_pairs(kind, values[0], values[1], twin_first)
    if kind == "shifty-unsettled":
        twin.will_settle = False       # its hash still differs from the text's hash while the input is read
    if container == "mapping":
        if isinstance(twin, _ShiftyKey) and kind_settles(twin):
            twin.settle()
        resource = _PairsMapping(others + pairs)
    else:
        resource = _public_dict_from(others, pairs, twin)
    with pytest.raises(ValueError, match="duplicate key"):
        _finish_with_resource(resource)


def test_a_lone_text_subclass_key_in_a_public_mapping_is_read_as_its_text_and_runs_no_hook() -> None:
    log: list[str] = []

    class Counting(str):
        def __hash__(self):
            log.append("hash"); return str.__hash__(self)

        def __eq__(self, other):
            log.append("eq"); return str.__eq__(self, other)

    pairs = [(Counting(key) if key == "observed_value" else key, value) for key, value in measured_memory().model_dump().items()]
    resource = _PairsMapping(pairs)
    log.clear()
    manifest = _finish_with_resource(resource)
    assert manifest.resources[0].observed_value == measured_memory().observed_value
    assert log == [], log


def test_a_public_mapping_key_is_copied_with_the_base_method_not_its_own_str() -> None:
    log: list[str] = []

    class Hostile(str):
        def __str__(self):
            log.append("__str__")
            return "something-else"

    pairs = [(Hostile(key) if key == "observed_value" else key, value) for key, value in measured_memory().model_dump().items()]
    manifest = _finish_with_resource(_PairsMapping(pairs))
    assert manifest.resources[0].observed_value == measured_memory().observed_value and log == []


def test_an_object_claiming_to_be_text_in_a_public_mapping_is_refused_not_copied() -> None:
    class Claims:
        @property
        def __class__(self):   # type: ignore[override]
            return str

        def __hash__(self):
            return 1

    pairs = list(measured_memory().model_dump().items()) + [(Claims(), 1)]
    with pytest.raises(ValueError):
        _finish_with_resource(_PairsMapping(pairs))


def test_a_text_subclass_key_in_model_extras_runs_no_hook_and_is_refused_as_undeclared() -> None:
    log: list[str] = []

    class Counting(str):
        def __hash__(self):
            log.append("hash")
            return str.__hash__(self)

        def __eq__(self, other):
            log.append("eq")
            return str.__eq__(self, other)

    reading = measured_memory()
    BASE_MODEL_EXTRA_DESCRIPTOR.__set__(reading, {Counting("surprise"): 1})
    log.clear()
    with pytest.raises(ValueError):
        _finish_with_resource(reading)
    assert log == [], log


# ---- a key that is not text cannot be rebuilt into a dictionary: a stateful key must not hide a value

class _StatefulKey:
    """Hashes like ``observed_value``; it compares unequal to the text for its first two comparisons and equal after that."""

    def __init__(self):
        self.comparisons = 0

    def __hash__(self):
        return str.__hash__("observed_value")

    def __eq__(self, other):
        self.comparisons += 1
        return self.comparisons > 2


def _stateful_pairs():
    others = [(key, value) for key, value in measured_memory().model_dump().items() if key != "observed_value"]
    return others, [(_StatefulKey(), 1), ("observed_value", True)]   # the invalid boolean comes after the stateful twin


def test_a_stateful_non_text_key_in_model_storage_is_refused_not_collapsed() -> None:
    reading = measured_memory()
    storage = BASE_MODEL_DICT_DESCRIPTOR.__get__(reading, pydantic.BaseModel)
    dict.pop(storage, "observed_value")
    _others, pairs = _stateful_pairs()
    for key, value in pairs:
        dict.__setitem__(storage, key, value)
    with pytest.raises(ValueError, match="unsupported key type"):
        _finish_with_resource(reading)


def test_a_stateful_non_text_key_in_an_exact_dict_is_refused_not_collapsed() -> None:
    others, pairs = _stateful_pairs()
    fields: dict = {}
    for key, value in pairs + others:
        dict.__setitem__(fields, key, value)
    with pytest.raises(ValueError, match="unsupported key type"):
        _finish_with_resource(fields)


def test_a_stateful_non_text_key_in_a_public_mapping_is_refused_not_collapsed() -> None:
    others, pairs = _stateful_pairs()
    with pytest.raises(ValueError, match="unsupported key type"):
        _finish_with_resource(_PairsMapping(pairs + others))


def test_a_stateful_non_text_key_never_yields_a_manifest_that_serializes() -> None:
    """Through serialization, the reported route: the manifest is either refused or keeps the invalid boolean's refusal."""
    reading = measured_memory()
    storage = BASE_MODEL_DICT_DESCRIPTOR.__get__(reading, pydantic.BaseModel)
    dict.pop(storage, "observed_value")
    for key, value in _stateful_pairs()[1]:
        dict.__setitem__(storage, key, value)
    manifest = completed_manifest()
    object.__setattr__(manifest, "resources", (reading,))
    with pytest.raises(ValueError, match="unsupported key type"):
        serialize_run_manifest(manifest)


def test_a_stateful_non_text_key_in_model_extras_is_refused_by_type() -> None:
    reading = measured_memory()
    BASE_MODEL_EXTRA_DESCRIPTOR.__set__(reading, {_StatefulKey(): 1})
    with pytest.raises(ValueError, match="unsupported key type"):
        _finish_with_resource(reading)


def test_a_stateful_key_that_claims_to_be_text_is_refused_by_type_not_by_its_claim() -> None:
    class ClaimsText(_StatefulKey):
        @property
        def __class__(self):   # type: ignore[override]
            return str

    reading = measured_memory()
    storage = BASE_MODEL_DICT_DESCRIPTOR.__get__(reading, pydantic.BaseModel)
    dict.pop(storage, "observed_value")
    dict.__setitem__(storage, ClaimsText(), 1)
    dict.__setitem__(storage, "observed_value", True)
    with pytest.raises(ValueError, match="unsupported key type"):
        _finish_with_resource(reading)


# ---- caller-overridable inventories are read once: a stateful field list cannot drop a captured value

class _ShrinkingFields(dict):
    """The model's field inventory; every iteration after the first leaves ``limit_value`` out."""

    def __init__(self, fields):
        super().__init__(fields)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        names = list(dict.__iter__(self))
        return iter(names if self.iterations == 1 else [name for name in names if name != "limit_value"])

    def __contains__(self, name):
        self.iterations += 1
        return dict.__contains__(self, name) and (self.iterations == 1 or name != "limit_value")


def _resource_with_a_stateful_field_inventory():
    base_meta = type(ResourceReading)

    class StatefulMeta(base_meta):
        @property
        def model_fields(cls):   # type: ignore[override]
            return cls._inventory

    class StatefulReading(ResourceReading, metaclass=StatefulMeta):
        pass

    StatefulReading._inventory = _ShrinkingFields(ResourceReading.model_fields)
    values = measured_memory().model_dump()
    values.update(limit_kind=LimitKind.UNBOUNDED, limit_value=True, observed_kind=ObservationKind.MEASURED)
    values["observation_kind"] = ObservationKind.MEASURED
    values.pop("observed_kind", None)
    return StatefulReading.model_construct(**values), StatefulReading._inventory


def _error_locations(error) -> set[str]:
    return {".".join(str(part) for part in item["loc"]) for item in error.errors()}


def test_a_field_inventory_that_changes_between_reads_cannot_drop_a_captured_value_through_serialization() -> None:
    resource, inventory = _resource_with_a_stateful_field_inventory()
    manifest = completed_manifest()
    object.__setattr__(manifest, "resources", (resource,))
    with pytest.raises(ValidationError) as caught:   # limit_value=True is invalid; a dropped value would let the default None through
        serialize_run_manifest(manifest)
    assert any(location.endswith("limit_value") for location in _error_locations(caught.value)), caught.value
    assert inventory.iterations >= 1


def test_a_field_inventory_that_changes_between_reads_cannot_drop_a_captured_value_through_finish() -> None:
    """The planned envelope is unbounded too, so a dropped ``limit_value`` would be the only difference: it would be accepted."""

    resource, _inventory = _resource_with_a_stateful_field_inventory()
    unbounded = ResourceReading(
        schema_version=VERSION, scope="api-container", resource="memory", unit="bytes",
        limit_kind=LimitKind.UNBOUNDED, limit_value=None, observation_kind=ObservationKind.PENDING,
    )
    planned = planned_manifest().model_copy(update={"resources": (unbounded,)})
    with pytest.raises(ValidationError) as caught:
        finish_run(
            planned,
            state=RunState.COMPLETED,
            started_at=datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 9, 28, 8, 1, tzinfo=timezone.utc),
            resources=(resource,),
            measurements=(timing(),),
        )
    assert any(location.endswith("limit_value") for location in _error_locations(caught.value)), caught.value


def test_the_snapshot_keeps_every_captured_stored_entry_whatever_the_field_inventory_says() -> None:
    resource, inventory = _resource_with_a_stateful_field_inventory()
    snapshot = _snapshot_native(resource)    # the snapshot does not validate: it must simply keep what was captured
    assert snapshot["limit_value"] is True and set(snapshot) == set(dict.keys(BASE_MODEL_DICT_DESCRIPTOR.__get__(resource, pydantic.BaseModel))), snapshot


def test_an_honest_resource_snapshot_still_lists_its_fields_in_declaration_order() -> None:
    snapshot = _snapshot_native(measured_memory())
    assert list(snapshot) == [name for name in ResourceReading.model_fields if name in snapshot]


# ---- an input document is copied to exact text or bytes once, and every parser reads that copy

def _duplicate_key_document() -> bytes:
    text = serialize_run_manifest(completed_manifest()).decode()
    return text.replace('"run_id":"run-001"', '"run_id":"run-001","run_id":"run-001"', 1).encode()


def test_a_bytes_subclass_with_its_own_decode_cannot_show_one_parser_a_cleaner_document() -> None:
    clean = serialize_run_manifest(completed_manifest()).decode()
    dirty = _duplicate_key_document()
    assert dirty != clean.encode()

    class CleanDecode(bytes):
        def decode(self, *args, **kwargs):
            return clean

    with pytest.raises(ValueError, match="duplicate JSON key"):
        validate_run_manifest_json(CleanDecode(dirty))


def test_a_bytes_subclass_with_its_own_bytes_method_is_copied_from_its_buffer() -> None:
    clean = serialize_run_manifest(completed_manifest())
    dirty = _duplicate_key_document()

    class CleanBytes(bytes):
        def __bytes__(self):
            return clean

    with pytest.raises(ValueError, match="duplicate JSON key"):
        validate_run_manifest_json(CleanBytes(dirty))


def test_a_text_subclass_with_hooks_is_copied_and_runs_none_of_them() -> None:
    log: list[str] = []
    document = serialize_run_manifest(completed_manifest()).decode()

    class Hooked(str):
        def startswith(self, *args, **kwargs):
            log.append("startswith")
            return False

        def __len__(self):
            log.append("__len__")
            return 1

        def __str__(self):
            log.append("__str__")
            return "{}"

    manifest = validate_run_manifest_json(Hooked(document))
    assert manifest.run_id == completed_manifest().run_id and log == [], log


def test_a_bytes_subclass_is_read_as_its_own_bytes() -> None:
    class Plain(bytes):
        pass

    document = serialize_run_manifest(completed_manifest())
    assert validate_run_manifest_json(Plain(document)) == validate_run_manifest_json(document)


def test_other_input_types_are_refused() -> None:
    for bad in (None, 3, [], bytearray(b"{}")):
        with pytest.raises(TypeError, match="text or bytes"):
            validate_run_manifest_json(bad)   # type: ignore[arg-type]


def test_a_bytes_subclass_with_its_own_buffer_view_is_copied_from_its_real_bytes() -> None:
    # A Python-level __buffer__ is consulted from Python 3.12 only; on earlier versions the hook is simply never asked.
    log: list[str] = []
    clean = serialize_run_manifest(completed_manifest())
    dirty = _duplicate_key_document()

    class CleanBuffer(bytes):
        def __buffer__(self, flags):
            log.append("__buffer__")
            return memoryview(clean)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        validate_run_manifest_json(CleanBuffer(dirty))
    assert log == [], log


def test_a_bytes_subclass_overriding_item_access_is_copied_from_its_real_bytes() -> None:
    log: list[str] = []
    clean = serialize_run_manifest(completed_manifest())
    dirty = _duplicate_key_document()

    class Slippery(bytes):
        def __getitem__(self, index):
            log.append("__getitem__")
            return clean[index]

        def __iter__(self):
            log.append("__iter__")
            return iter(clean)

        def __len__(self):
            log.append("__len__")
            return len(clean)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        validate_run_manifest_json(Slippery(dirty))
    assert log == [], log


# ---- the single read of a caller-overridable field inventory is bounded

import signal as _signal


class _Alarm(Exception):
    pass


def _within(seconds, call):
    def on_alarm(signum, frame):
        raise _Alarm()

    previous = _signal.signal(_signal.SIGALRM, on_alarm)
    _signal.setitimer(_signal.ITIMER_REAL, seconds)
    try:
        return call()
    finally:
        _signal.setitimer(_signal.ITIMER_REAL, 0)
        _signal.signal(_signal.SIGALRM, previous)


class _InventoryFrom(dict):
    """A field inventory whose iteration yields whatever ``factory`` makes (after the real names, when asked)."""

    def __init__(self, fields, factory):
        super().__init__(fields)
        self.factory = factory
        self.reads = 0

    def __iter__(self):
        self.reads += 1
        return iter(self.factory(list(dict.__iter__(self))))


def _resource_with_inventory(factory):
    base_meta = type(ResourceReading)

    class InventoryMeta(base_meta):
        @property
        def model_fields(cls):   # type: ignore[override]
            return cls._inventory

    class InventoryReading(ResourceReading, metaclass=InventoryMeta):
        pass

    InventoryReading._inventory = _InventoryFrom(ResourceReading.model_fields, factory)
    return InventoryReading.model_construct(**measured_memory().model_dump()), InventoryReading._inventory


def _endless(real):
    def names():
        yield from real
        index = 0
        while True:
            yield f"extra-{index}"
            index += 1
    return names()


def _huge_distinct(real):
    return iter([*real, *(f"extra-{index}" for index in range(200_000))])


def _huge_repeated(real):
    return iter(["observed_value"] * 200_000)


@pytest.mark.parametrize("factory", [_endless, _huge_distinct, _huge_repeated], ids=["nonterminating", "huge-distinct", "huge-repeated"])
def test_an_oversized_or_endless_field_inventory_is_refused_by_the_snapshot_budget(factory) -> None:
    resource, inventory = _resource_with_inventory(factory)
    with pytest.raises(ValueError) as caught:
        _within(10, lambda: _snapshot_native(resource))
    assert str(caught.value) == run_manifest_module.SNAPSHOT_BUDGET_MESSAGE
    assert inventory.reads == 1


def test_an_oversized_inventory_is_refused_through_serialization_too() -> None:
    resource, _inventory = _resource_with_inventory(_huge_distinct)
    manifest = completed_manifest()
    object.__setattr__(manifest, "resources", (resource,))
    with pytest.raises(ValueError) as caught:
        _within(10, lambda: serialize_run_manifest(manifest))
    assert str(caught.value) == run_manifest_module.SNAPSHOT_BUDGET_MESSAGE


def test_an_inventory_at_the_limit_is_read_once_and_one_more_is_refused() -> None:
    limit = run_manifest_module._MAX_FIELD_INVENTORY
    real = list(ResourceReading.model_fields)

    def exactly(count):
        def make(names):
            return iter([*names, *(f"pad-{index}" for index in range(count - len(names)))])
        return make

    at_limit, inventory = _resource_with_inventory(exactly(limit))
    try:
        _snapshot_native(at_limit)   # padding names are not stored fields: refused as undeclared, but not by the budget
    except ValueError as error:
        assert str(error) != run_manifest_module.SNAPSHOT_BUDGET_MESSAGE
    assert inventory.reads == 1
    over, _inventory = _resource_with_inventory(exactly(limit + 1))
    with pytest.raises(ValueError) as caught:
        _snapshot_native(over)
    assert str(caught.value) == run_manifest_module.SNAPSHOT_BUDGET_MESSAGE
    assert limit > len(real)


def test_an_honest_resource_reads_its_inventory_once_and_is_unchanged() -> None:
    resource, inventory = _resource_with_inventory(lambda names: iter(names))
    snapshot = _snapshot_native(resource)
    assert inventory.reads == 1 and set(snapshot) == set(measured_memory().model_dump())


# ---- an inventory is bounded in the size of its names and read once per class per call

class _BigName(str):
    """A text name of a given length (a str subclass, so copying it is linear in its length)."""


def _resource_class_with_names(names):
    base_meta = type(ResourceReading)

    class NamesMeta(base_meta):
        @property
        def model_fields(cls):   # type: ignore[override]
            cls._reads += 1
            return {**ResourceReading.model_fields, **{name: ResourceReading.model_fields["scope"] for name in names}}

    class NamedReading(ResourceReading, metaclass=NamesMeta):
        pass

    NamedReading._reads = 0
    return NamedReading


def _manifest_of_instances(cls, count):
    values = measured_memory().model_dump()
    resources = tuple(cls.model_construct(**{**values, "scope": f"scope-{index}"}) for index in range(count))
    manifest = completed_manifest()
    object.__setattr__(manifest, "resources", resources)
    return manifest


def test_oversized_inventory_names_are_refused_before_they_are_copied_or_hashed() -> None:
    cls = _resource_class_with_names([_BigName("n" * (1 << 20)) + str(index) for index in range(55)])
    manifest = _manifest_of_instances(cls, 100)
    started = time.monotonic()
    with pytest.raises(ValueError) as caught:
        serialize_run_manifest(manifest)
    elapsed = time.monotonic() - started
    assert str(caught.value) == run_manifest_module.SNAPSHOT_BUDGET_MESSAGE
    assert elapsed < 2, elapsed      # unfixed: about 5 s for 100 instances of 55 one-mebibyte names


def test_a_name_that_is_not_text_is_refused_by_the_inventory_bound() -> None:
    cls = _resource_class_with_names([(1 << 2_000_000) + index for index in range(8)])
    with pytest.raises(ValueError) as caught:
        serialize_run_manifest(_manifest_of_instances(cls, 20))
    assert str(caught.value) == run_manifest_module.SNAPSHOT_BUDGET_MESSAGE


def test_a_class_inventory_is_read_once_per_call_however_many_instances() -> None:
    cls = _resource_class_with_names([f"pad-{index}" for index in range(10)])
    manifest = _manifest_of_instances(cls, 200)
    serialize_run_manifest(manifest)                 # the padding names are inventory entries only: nothing is stored under them
    assert cls._reads == 1, cls._reads


def test_each_call_reads_the_inventory_afresh() -> None:
    cls = _resource_class_with_names([f"pad-{index}" for index in range(3)])
    for call in range(3):
        serialize_run_manifest(_manifest_of_instances(cls, 5))
    assert cls._reads == 3, cls._reads


def test_field_names_of_ordinary_length_are_still_accepted() -> None:
    manifest = _manifest_of_instances(_resource_class_with_names([]), 50)
    manifest_bytes = serialize_run_manifest(manifest)
    assert b"scope-49" in manifest_bytes


def test_an_inventory_name_that_lies_about_its_length_is_measured_by_the_base_method() -> None:
    log: list[str] = []

    class Liar(str):
        def __len__(self):
            log.append("__len__")
            return 1

    cls = _resource_class_with_names([Liar("x" * (1 << 20) + str(index)) for index in range(3)])
    with pytest.raises(ValueError) as caught:
        serialize_run_manifest(_manifest_of_instances(cls, 5))
    assert str(caught.value) == run_manifest_module.SNAPSHOT_BUDGET_MESSAGE and log == [], log
