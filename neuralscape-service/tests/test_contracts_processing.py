"""Tests for processing-policy and plaintext-recipient boundaries."""

import pytest
from pydantic import ValidationError

from contracts_processing import (
    PlaintextDispatchRejected,
    ProcessingPolicy,
    validate_plaintext_dispatch,
)


def _strict_local_policy(**changes: object) -> ProcessingPolicy:
    values: dict[str, object] = {
        "schema_version": "candidate-v1",
        "mode": "strict_local",
        "allowed_execution_locations": ("endpoint",),
        "approved_recipient_ids": (),
        "fallback_policy": "deny",
        "policy_epoch": 7,
    }
    values.update(changes)
    return ProcessingPolicy.model_validate(values)


def _external_policy(**changes: object) -> ProcessingPolicy:
    values: dict[str, object] = {
        "schema_version": "candidate-v1",
        "mode": "operator_trusted",
        "allowed_execution_locations": ("endpoint", "external_provider"),
        "approved_recipient_ids": ("provider-a",),
        "fallback_policy": "within_approved_recipients",
        "policy_epoch": 7,
    }
    values.update(changes)
    return ProcessingPolicy.model_validate(values)


def test_strict_local_accepts_only_local_primary_dispatch() -> None:
    policy = _strict_local_policy()

    assert (
        validate_plaintext_dispatch(
            policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )
        is None
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"allowed_execution_locations": ("endpoint", "external_provider")},
        {"allowed_execution_locations": ("external_provider",)},
        {"approved_recipient_ids": ("provider-a",)},
        {"fallback_policy": "within_approved_recipients"},
    ],
)
def test_strict_local_rejects_external_or_fallback_policy(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="strict_local|recipient fallback"):
        _strict_local_policy(**changes)


def test_non_endpoint_policy_requires_explicit_recipient() -> None:
    with pytest.raises(ValidationError, match="approved recipient"):
        _external_policy(approved_recipient_ids=())


def test_recipient_without_non_endpoint_location_is_rejected() -> None:
    with pytest.raises(ValidationError, match="non-endpoint execution location"):
        _external_policy(allowed_execution_locations=("endpoint",))


def test_operator_blind_policy_cannot_select_operator_worker() -> None:
    with pytest.raises(ValidationError, match="operator_worker"):
        _external_policy(
            mode="operator_blind",
            allowed_execution_locations=("operator_worker",),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("allowed_execution_locations", ("endpoint", "endpoint")),
        ("approved_recipient_ids", ("provider-a", "provider-a")),
    ],
)
def test_policy_rejects_duplicate_boundary_entries(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match="unique"):
        _external_policy(**{field: value})


def test_policy_rejects_unknown_fields_and_boolean_epoch() -> None:
    with pytest.raises(ValidationError):
        _strict_local_policy(policy_epoch=True)
    with pytest.raises(ValidationError, match="extra"):
        _strict_local_policy(read_access_implies_processing=True)


def test_policy_requires_explicit_candidate_version() -> None:
    policy = _strict_local_policy()
    values = policy.model_dump(mode="python")
    values.pop("schema_version")

    assert policy.schema_version == "candidate-v1"
    assert policy.model_json_schema()["properties"]["schema_version"]["const"] == (
        "candidate-v1"
    )
    assert "schema_version" in policy.model_json_schema()["required"]
    with pytest.raises(ValidationError):
        ProcessingPolicy.model_validate(values)
    with pytest.raises(ValidationError):
        _strict_local_policy(schema_version=None)
    with pytest.raises(ValidationError):
        _strict_local_policy(schema_version="future-v2")


def test_policy_and_boundary_collections_are_frozen() -> None:
    policy = _strict_local_policy()

    with pytest.raises(ValidationError, match="frozen"):
        policy.mode = "operator_trusted"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        policy.allowed_execution_locations.append("external_provider")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        policy.approved_recipient_ids.append("provider-a")  # type: ignore[attr-defined]


def test_dispatch_revalidates_unchecked_strict_local_copy() -> None:
    unchecked = _strict_local_policy().model_copy(
        update={
            "allowed_execution_locations": ("endpoint", "external_provider"),
            "approved_recipient_ids": ("provider-a",),
        }
    )

    with pytest.raises(PlaintextDispatchRejected, match="policy is invalid"):
        validate_plaintext_dispatch(
            unchecked,
            execution_location="external_provider",
            recipient_id="provider-a",
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a"},
            is_fallback=False,
        )


def test_external_dispatch_requires_policy_and_current_authority() -> None:
    policy = _external_policy()

    validate_plaintext_dispatch(
        policy,
        execution_location="external_provider",
        recipient_id="provider-a",
        current_policy_epoch=7,
        currently_authorized_recipient_ids={"provider-a"},
        is_fallback=False,
    )

    with pytest.raises(PlaintextDispatchRejected, match="current processing authority"):
        validate_plaintext_dispatch(
            policy,
            execution_location="external_provider",
            recipient_id="provider-a",
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )


def test_current_authority_cannot_add_recipient_missing_from_policy() -> None:
    with pytest.raises(PlaintextDispatchRejected, match="not approved by policy"):
        validate_plaintext_dispatch(
            _external_policy(),
            execution_location="external_provider",
            recipient_id="provider-b",
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a", "provider-b"},
            is_fallback=False,
        )


def test_stale_policy_epoch_fails_closed() -> None:
    with pytest.raises(PlaintextDispatchRejected, match="not current"):
        validate_plaintext_dispatch(
            _external_policy(),
            execution_location="external_provider",
            recipient_id="provider-a",
            current_policy_epoch=8,
            currently_authorized_recipient_ids={"provider-a"},
            is_fallback=False,
        )


def test_unapproved_location_fails_even_for_authorized_recipient() -> None:
    with pytest.raises(PlaintextDispatchRejected, match="location"):
        validate_plaintext_dispatch(
            _external_policy(),
            execution_location="customer_worker",
            recipient_id="provider-a",
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a"},
            is_fallback=False,
        )


@pytest.mark.parametrize(
    ("execution_location", "recipient_id"),
    [("endpoint", None), ("external_provider", "provider-a")],
)
def test_deny_policy_blocks_fallback_after_primary_failure(
    execution_location: str,
    recipient_id: str | None,
) -> None:
    policy = _external_policy(fallback_policy="deny")

    with pytest.raises(PlaintextDispatchRejected, match="fallback is denied"):
        validate_plaintext_dispatch(
            policy,
            execution_location=execution_location,  # type: ignore[arg-type]
            recipient_id=recipient_id,
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a"},
            is_fallback=True,
        )


def test_dispatch_requires_explicit_fallback_classification() -> None:
    policy = _external_policy(fallback_policy="deny")

    with pytest.raises(TypeError, match="is_fallback"):
        validate_plaintext_dispatch(  # type: ignore[call-arg]
            policy,
            execution_location="external_provider",
            recipient_id="provider-a",
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a"},
        )

    assert (
        validate_plaintext_dispatch(
            policy,
            execution_location="external_provider",
            recipient_id="provider-a",
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a"},
            is_fallback=False,
        )
        is None
    )
    with pytest.raises(PlaintextDispatchRejected, match="fallback is denied"):
        validate_plaintext_dispatch(
            policy,
            execution_location="external_provider",
            recipient_id="provider-a",
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a"},
            is_fallback=True,
        )


def test_approved_fallback_is_revalidated_against_current_authority() -> None:
    policy = _external_policy(
        approved_recipient_ids=("provider-a", "provider-b"),
    )

    validate_plaintext_dispatch(
        policy,
        execution_location="external_provider",
        recipient_id="provider-b",
        current_policy_epoch=7,
        currently_authorized_recipient_ids={"provider-b"},
        is_fallback=True,
    )


def test_recipient_only_policy_rejects_endpoint_fallback_but_allows_primary() -> None:
    policy = _external_policy()

    assert (
        validate_plaintext_dispatch(
            policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=False,
        )
        is None
    )
    with pytest.raises(PlaintextDispatchRejected, match="non-endpoint"):
        validate_plaintext_dispatch(
            policy,
            execution_location="endpoint",
            recipient_id=None,
            current_policy_epoch=7,
            currently_authorized_recipient_ids=set(),
            is_fallback=True,
        )


def test_endpoint_dispatch_cannot_smuggle_a_recipient_identity() -> None:
    with pytest.raises(PlaintextDispatchRejected, match="must not name"):
        validate_plaintext_dispatch(
            _external_policy(),
            execution_location="endpoint",
            recipient_id="provider-a",
            current_policy_epoch=7,
            currently_authorized_recipient_ids={"provider-a"},
            is_fallback=False,
        )
