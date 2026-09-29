"""Contract tests for topology-neutral tenant operation state."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from contracts_tenancy import (
    ResourceManifestReference,
    TenantOperationState,
    TenantPlacement,
    validate_operation_transition,
    validate_placement_publication,
)


VERSION = "candidate-v1"


def manifest(
    *, tenant_id: str = "tenant-a", generation: int = 7, manifest_id: str = "primary"
) -> dict[str, object]:
    return {
        "schema_version": VERSION,
        "tenant_id": tenant_id,
        "manifest_id": manifest_id,
        "manifest_revision": 2,
        "placement_generation": generation,
    }


def operation(**updates: object) -> TenantOperationState:
    document: dict[str, object] = {
        "schema_version": VERSION,
        "tenant_id": "tenant-a",
        "operation_id": "operation-1",
        "operation": "provision",
        "desired_state": "active",
        "observed_state": "pending",
        "placement_generation": 7,
        "resource_manifests": [manifest()],
    }
    document.update(updates)
    return TenantOperationState.model_validate_json(json.dumps(document))


def test_placement_preserves_explicit_tenant_isolation_identity() -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [manifest()],
            }
        )
    )

    assert placement.tenant_id == "tenant-a"
    assert placement.resource_manifests[0].tenant_id == "tenant-a"


def test_placement_rejects_cross_tenant_manifest_reference() -> None:
    with pytest.raises(ValidationError, match="tenant_id must match"):
        TenantPlacement.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "generation": 7,
                    "resource_manifests": [manifest(tenant_id="tenant-b")],
                }
            )
        )


def test_placement_rejects_manifest_from_mismatched_generation_epoch() -> None:
    with pytest.raises(ValidationError, match="placement_generation must match"):
        TenantPlacement.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "generation": 8,
                    "resource_manifests": [manifest(generation=7)],
                }
            )
        )


def test_placement_rejects_duplicate_manifest_identity_and_revision() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        TenantPlacement.model_validate_json(
            json.dumps(
                {
                    "schema_version": VERSION,
                    "tenant_id": "tenant-a",
                    "generation": 7,
                    "resource_manifests": [manifest(), deepcopy(manifest())],
                }
            )
        )


@pytest.mark.parametrize(
    ("kind", "desired"),
    [
        ("provision", "active"),
        ("suspend", "suspended"),
        ("resume", "active"),
        ("export", "unchanged"),
        ("restore", "active"),
        ("upgrade", "active"),
        ("delete", "deleted"),
    ],
)
def test_each_operation_has_an_explicit_desired_result(kind: str, desired: str) -> None:
    state = operation(
        operation=kind,
        desired_state=desired,
        resource_manifests=[],
    )

    assert state.desired_state == desired


def test_operation_rejects_semantically_wrong_desired_result() -> None:
    with pytest.raises(ValidationError, match="suspend requires desired_state=suspended"):
        operation(operation="suspend", desired_state="active")


def test_operation_rejects_manifest_from_another_tenant() -> None:
    with pytest.raises(ValidationError, match="tenant_id must match"):
        operation(resource_manifests=[manifest(tenant_id="tenant-b")])


def test_operation_rejects_exact_duplicate_manifest_references() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        operation(resource_manifests=[manifest(), deepcopy(manifest())])


def test_valid_operation_progress_is_accepted() -> None:
    pending = operation(observed_state="pending")
    running = operation(observed_state="running")
    succeeded = operation(observed_state="succeeded")

    running_result = validate_operation_transition(pending, running)
    succeeded_result = validate_operation_transition(running, succeeded)

    assert running_result == running
    assert running_result is not running
    assert succeeded_result == succeeded
    assert succeeded_result is not succeeded


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        ("pending", "succeeded"),
        ("running", "pending"),
        ("succeeded", "running"),
        ("failed", "running"),
        ("cancelled", "running"),
    ],
)
def test_invalid_operation_progress_is_rejected(previous: str, current: str) -> None:
    with pytest.raises(ValueError, match="invalid observed_state transition"):
        validate_operation_transition(
            operation(observed_state=previous),
            operation(observed_state=current),
        )


@pytest.mark.parametrize(
    ("field", "updates"),
    [
        ("tenant_id", {"tenant_id": "tenant-b"}),
        ("operation_id", {"operation_id": "operation-2"}),
        ("operation", {"operation": "resume", "desired_state": "active"}),
    ],
)
def test_transition_rejects_changed_operation_identity(
    field: str, updates: dict[str, str]
) -> None:
    previous = operation(resource_manifests=[])
    current = operation(
        observed_state="running", resource_manifests=[], **updates
    )
    with pytest.raises(ValueError, match=f"immutable {field}"):
        validate_operation_transition(previous, current)


def test_transition_rejects_generation_rollback() -> None:
    with pytest.raises(ValueError, match="cannot decrease"):
        validate_operation_transition(
            operation(placement_generation=8, resource_manifests=[]),
            operation(
                placement_generation=7,
                observed_state="running",
                resource_manifests=[],
            ),
        )


@pytest.mark.parametrize("terminal_state", ["succeeded", "failed", "cancelled"])
def test_terminal_republication_must_be_exactly_idempotent(
    terminal_state: str,
) -> None:
    terminal = operation(observed_state=terminal_state)

    assert validate_operation_transition(terminal, terminal.model_copy()) == terminal

    with pytest.raises(ValueError, match="must be identical"):
        validate_operation_transition(
            terminal,
            operation(
                observed_state=terminal_state,
                placement_generation=8,
                resource_manifests=[manifest(generation=8)],
            ),
        )
    with pytest.raises(ValueError, match="must be identical"):
        validate_operation_transition(
            terminal,
            operation(
                observed_state=terminal_state,
                resource_manifests=[manifest(manifest_id="replacement")],
            ),
        )


def test_publication_requires_exact_current_generation() -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [manifest()],
            }
        )
    )

    result = validate_placement_publication(
        placement, expected_tenant_id="tenant-a", current_generation=7
    )
    assert result == placement
    assert result is not placement
    with pytest.raises(ValueError, match="stale or unexpected"):
        validate_placement_publication(
            placement, expected_tenant_id="tenant-a", current_generation=8
        )


def test_publication_rejects_cross_tenant_context() -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [],
            }
        )
    )

    with pytest.raises(ValueError, match="tenant_id does not match"):
        validate_placement_publication(
            placement, expected_tenant_id="tenant-b", current_generation=7
        )


def test_publication_revalidates_copied_nested_manifest_graph() -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [manifest()],
            }
        )
    )
    cross_tenant = ResourceManifestReference.model_validate_json(
        json.dumps(manifest(tenant_id="tenant-b"))
    )
    corrupted = placement.model_copy(
        update={"resource_manifests": (cross_tenant,)}
    )

    with pytest.raises(ValidationError, match="tenant_id must match"):
        validate_placement_publication(
            corrupted,
            expected_tenant_id="tenant-a",
            current_generation=7,
        )


def test_publication_rejects_retained_copied_extra_fields() -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [manifest()],
            }
        )
    )
    top_level_extra = placement.model_copy(
        update={"unreviewed_topology": "shared-plane"}
    )
    nested_extra = placement.resource_manifests[0].model_copy(
        update={"authority": "admin"}
    )
    nested_corruption = placement.model_copy(
        update={"resource_manifests": (nested_extra,)}
    )

    for corrupted in (top_level_extra, nested_corruption):
        with pytest.raises(ValueError, match="undeclared fields"):
            validate_placement_publication(
                corrupted,
                expected_tenant_id="tenant-a",
                current_generation=7,
            )


@pytest.mark.parametrize("corrupt_previous", [False, True])
def test_transition_revalidates_copied_operation_graph(
    corrupt_previous: bool,
) -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    cross_tenant = ResourceManifestReference.model_validate_json(
        json.dumps(manifest(tenant_id="tenant-b"))
    )
    corrupted = (previous if corrupt_previous else current).model_copy(
        update={"resource_manifests": (cross_tenant,)}
    )

    with pytest.raises(ValidationError, match="tenant_id must match"):
        validate_operation_transition(
            corrupted if corrupt_previous else previous,
            current if corrupt_previous else corrupted,
        )


@pytest.mark.parametrize("corrupt_previous", [False, True])
def test_transition_rejects_retained_copied_top_level_extra(
    corrupt_previous: bool,
) -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    corrupted = (previous if corrupt_previous else current).model_copy(
        update={"unreviewed_policy": "allow"}
    )

    with pytest.raises(ValueError, match="undeclared fields"):
        validate_operation_transition(
            corrupted if corrupt_previous else previous,
            current if corrupt_previous else corrupted,
        )


def test_transition_rejects_retained_copied_nested_extra() -> None:
    previous = operation()
    current = operation(observed_state="running")
    nested_extra = current.resource_manifests[0].model_copy(
        update={"authority": "admin"}
    )
    corrupted = current.model_copy(
        update={"resource_manifests": (nested_extra,)}
    )

    with pytest.raises(ValueError, match="undeclared fields"):
        validate_operation_transition(previous, corrupted)


def test_transition_rejects_cyclic_mutated_input_graph() -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    current.resource_manifests = (current,)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="must be acyclic"):
        validate_operation_transition(previous, current)


def test_transition_revalidates_copied_operation_fields() -> None:
    previous = operation(resource_manifests=[])
    current = operation(observed_state="running", resource_manifests=[])
    invalid = current.model_copy(update={"desired_state": "suspended"})

    with pytest.raises(ValidationError, match="requires desired_state"):
        validate_operation_transition(previous, invalid)


def test_publication_revalidates_post_construction_nested_mutation() -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [manifest()],
            }
        )
    )
    placement.resource_manifests[0].tenant_id = "tenant-b"

    with pytest.raises(ValidationError, match="tenant_id must match"):
        validate_placement_publication(
            placement,
            expected_tenant_id="tenant-a",
            current_generation=7,
        )


@pytest.mark.parametrize(
    "invalid_generation",
    [True, 7.0, -1, 9_007_199_254_740_992],
)
def test_publication_helper_strictly_validates_generation(
    invalid_generation: object,
) -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [],
            }
        )
    )

    with pytest.raises(ValidationError):
        validate_placement_publication(
            placement,
            expected_tenant_id="tenant-a",
            current_generation=invalid_generation,
        )


@pytest.mark.parametrize("invalid_tenant_id", [1, b"tenant-a", ""])
def test_publication_helper_strictly_validates_tenant_id(
    invalid_tenant_id: object,
) -> None:
    placement = TenantPlacement.model_validate_json(
        json.dumps(
            {
                "schema_version": VERSION,
                "tenant_id": "tenant-a",
                "generation": 7,
                "resource_manifests": [],
            }
        )
    )

    with pytest.raises(ValidationError):
        validate_placement_publication(
            placement,
            expected_tenant_id=invalid_tenant_id,
            current_generation=7,
        )


def test_contracts_require_version_and_reject_unknown_fields() -> None:
    document = manifest()
    document.pop("schema_version")
    with pytest.raises(ValidationError):
        ResourceManifestReference.model_validate_json(json.dumps(document))

    with pytest.raises(ValidationError):
        operation(unreviewed_topology="shared-plane")


@pytest.mark.parametrize("invalid_generation", [True, -1, 9_007_199_254_740_992])
def test_generation_uses_safe_counter_bounds(invalid_generation: object) -> None:
    with pytest.raises(ValidationError):
        operation(
            placement_generation=invalid_generation,
            resource_manifests=[],
        )
