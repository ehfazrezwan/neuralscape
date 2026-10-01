"""Plaintext processing-policy contracts and dispatch-boundary validation.

These types describe allowed routing.  They do not establish read permission,
processing authority, runtime assurance, or successful attestation.  Callers
must supply current processing authority when validating each dispatch,
including fallback attempts.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator

from contracts_common import OpaqueId, SafeCounter, VersionedContract


ProcessingMode = Literal["operator_trusted", "operator_blind", "strict_local"]
ExecutionLocation = Literal[
    "endpoint",
    "customer_worker",
    "operator_worker",
    "external_provider",
    "attested_worker",
]
FallbackPolicy = Literal["deny", "within_approved_recipients"]

_NON_ENDPOINT_LOCATIONS = frozenset(
    {"customer_worker", "operator_worker", "external_provider", "attested_worker"}
)
_EXECUTION_LOCATION_ADAPTER = TypeAdapter(ExecutionLocation)
_OPAQUE_ID_ADAPTER = TypeAdapter(OpaqueId)
_SAFE_COUNTER_ADAPTER = TypeAdapter(SafeCounter)
_BASE_MODEL_DICT_DESCRIPTOR = vars(BaseModel)["__dict__"]
_BASE_MODEL_EXTRA_DESCRIPTOR = vars(BaseModel)["__pydantic_extra__"]


def _with_native_dict_extra_backing(value: object) -> object:
    """Project a model with native dict extra storage for closed revalidation."""
    value_type = type(value)
    if not issubclass(value_type, BaseModel):
        return value
    extras = _BASE_MODEL_EXTRA_DESCRIPTOR.__get__(value, BaseModel)
    if extras is None:
        captured = ()
    elif issubclass(type(extras), dict):
        # A dict subclass may lie through len/iteration/keys/items/truthiness.
        # Snapshot its concrete backing exactly once with the built-in operation.
        captured = tuple(dict.items(extras))
    else:
        return value

    # Detach every pair before projected-dict hashing can run a hostile stored
    # name callback that repairs a later value in the live model backing.
    stored = tuple(
        dict.items(_BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel))
    )
    projected = dict(stored)
    if captured:
        projected["native_extra_backing"] = dict(captured)
    return projected


class ProcessingPolicy(VersionedContract):
    """Declared locations and recipients that may receive plaintext.

    A recipient listed here is only policy-approved.  It must also be present
    in current independently evaluated processing authority at dispatch time.
    ``attested_worker`` is a routing category, not proof of attestation.
    """

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    mode: ProcessingMode
    allowed_execution_locations: tuple[ExecutionLocation, ...]
    approved_recipient_ids: tuple[OpaqueId, ...]
    fallback_policy: FallbackPolicy
    policy_epoch: SafeCounter

    @model_validator(mode="after")
    def validate_processing_boundary(self) -> "ProcessingPolicy":
        locations = self.allowed_execution_locations
        recipients = self.approved_recipient_ids

        if not locations:
            raise ValueError("at least one execution location is required")
        if len(locations) != len(set(locations)):
            raise ValueError("execution locations must be unique")
        if len(recipients) != len(set(recipients)):
            raise ValueError("approved recipient ids must be unique")

        if self.mode == "strict_local":
            if locations != ("endpoint",):
                raise ValueError("strict_local permits only endpoint execution")
            if recipients:
                raise ValueError("strict_local permits no plaintext recipients")
            if self.fallback_policy != "deny":
                raise ValueError("strict_local forbids fallback")

        non_endpoint_locations = set(locations) & _NON_ENDPOINT_LOCATIONS
        if non_endpoint_locations and not recipients:
            raise ValueError("non-endpoint execution requires an approved recipient")
        if recipients and not non_endpoint_locations:
            raise ValueError("approved recipients require a non-endpoint execution location")

        if self.mode == "operator_blind" and "operator_worker" in locations:
            raise ValueError("operator_blind forbids operator_worker execution")

        if (
            self.fallback_policy == "within_approved_recipients"
            and not non_endpoint_locations
        ):
            raise ValueError(
                "recipient fallback requires a non-endpoint execution location"
            )

        return self


class PlaintextDispatchRejected(ValueError):
    """Raised when a proposed plaintext dispatch fails a boundary check."""


def validate_plaintext_dispatch(
    policy: ProcessingPolicy,
    *,
    execution_location: ExecutionLocation,
    recipient_id: OpaqueId | None,
    current_policy_epoch: SafeCounter,
    currently_authorized_recipient_ids: Collection[OpaqueId],
    is_fallback: bool,
) -> None:
    """Reject a plaintext dispatch unless declared and current authority agree.

    ``currently_authorized_recipient_ids`` must come from a current processing-
    authority decision.  Read/decrypt access is deliberately not an input and
    cannot authorize forwarding plaintext.  Successful validation is only a
    routing precondition; it does not prove endpoint identity or runtime
    assurance.  Callers must explicitly classify every dispatch as primary or
    fallback; omission is not treated as a primary dispatch.  Recipient-only
    fallback must identify an approved, currently authorized non-endpoint
    recipient.  Execution-location string subclasses carrying a declared value
    are normalized to that built-in string; non-string values are rejected
    before policy comparisons.
    """

    try:
        policy = ProcessingPolicy.model_validate(
            _with_native_dict_extra_backing(policy)
        )
    except (TypeError, ValueError) as exc:
        raise PlaintextDispatchRejected("processing policy is invalid") from exc

    try:
        checked_epoch = _SAFE_COUNTER_ADAPTER.validate_python(
            current_policy_epoch, strict=True
        )
    except ValueError as exc:
        raise PlaintextDispatchRejected("current policy epoch is invalid") from exc

    if checked_epoch != policy.policy_epoch:
        raise PlaintextDispatchRejected("processing policy is not current")
    if type(is_fallback) is not bool:
        raise PlaintextDispatchRejected("fallback marker must be a boolean")
    try:
        checked_location = _EXECUTION_LOCATION_ADAPTER.validate_python(
            execution_location, strict=True
        )
    except ValueError as exc:
        raise PlaintextDispatchRejected("execution location is invalid") from exc

    if checked_location not in policy.allowed_execution_locations:
        raise PlaintextDispatchRejected("execution location is not allowed")
    if is_fallback:
        if policy.fallback_policy == "deny":
            raise PlaintextDispatchRejected("fallback is denied")
        if checked_location == "endpoint":
            raise PlaintextDispatchRejected(
                "recipient fallback requires a non-endpoint execution location"
            )

    if checked_location == "endpoint":
        if recipient_id is not None:
            raise PlaintextDispatchRejected(
                "endpoint execution must not name a plaintext recipient"
            )
        return

    if recipient_id is None:
        raise PlaintextDispatchRejected("non-endpoint execution requires a recipient")

    try:
        if isinstance(currently_authorized_recipient_ids, (str, bytes)):
            raise TypeError("recipient authority must be a collection of identifiers")
        checked_recipient = _OPAQUE_ID_ADAPTER.validate_python(recipient_id, strict=True)
        current_recipients = {
            _OPAQUE_ID_ADAPTER.validate_python(value, strict=True)
            for value in currently_authorized_recipient_ids
        }
    except (TypeError, ValueError) as exc:
        raise PlaintextDispatchRejected("current recipient authority is invalid") from exc

    if checked_recipient not in policy.approved_recipient_ids:
        raise PlaintextDispatchRejected("recipient is not approved by policy")
    if checked_recipient not in current_recipients:
        raise PlaintextDispatchRejected("recipient lacks current processing authority")


__all__ = [
    "ExecutionLocation",
    "FallbackPolicy",
    "PlaintextDispatchRejected",
    "ProcessingMode",
    "ProcessingPolicy",
    "validate_plaintext_dispatch",
]
