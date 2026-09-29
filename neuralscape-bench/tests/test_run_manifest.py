"""Contract tests for reproducible, product-independent benchmark manifests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pydantic
import pytest
from pydantic import ValidationError

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
from neuralscape_bench.run_manifest import _snapshot_native


VERSION = "candidate-v1"
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


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


def test_planned_to_completed_round_trip_is_canonical_and_stable():
    manifest = completed_manifest()

    first = serialize_run_manifest(manifest)
    reparsed = validate_run_manifest_json(first)

    assert reparsed == manifest
    assert serialize_run_manifest(reparsed) == first
    assert first.startswith(b'{"build":')
    assert b'"observed_value":402653184' in first


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
