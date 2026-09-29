"""Typed error vocabulary and disclosure-safe public mapping.

``InternalErrorCode`` describes outcomes inside an authorized trust boundary.
``public_error_for`` maps those outcomes to stable public errors.  Its default
does not reveal whether a referenced object exists: denial, absence, a wrong
reference kind, and a stale resource revision all become ``not_found``.  Set
``disclose_existence`` only after an independent authorization decision permits
richer diagnostics.
"""

from enum import Enum

from contracts_common import ContractModel


class InternalErrorCode(str, Enum):
    """Typed internal outcomes; these names are not a public wire protocol."""

    INVALID_SCHEMA = "invalid_schema"
    UNAUTHENTICATED = "unauthenticated"
    DENIED = "denied"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    MISSING_RESOURCE = "missing_resource"
    WRONG_REFERENCE_KIND = "wrong_reference_kind"
    STALE_REVISION = "stale_revision"
    STALE_POLICY = "stale_policy"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    UNAVAILABLE_DEPENDENCY = "unavailable_dependency"
    PARTIAL_RESULT = "partial_result"
    CANCELLED = "cancelled"
    TERMINAL_FAILURE = "terminal_failure"


class PublicErrorCode(str, Enum):
    """Stable codes safe to expose after applying the disclosure policy."""

    INVALID_REQUEST = "invalid_request"
    UNAUTHENTICATED = "unauthenticated"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    WRONG_REFERENCE_KIND = "wrong_reference_kind"
    STALE_REVISION = "stale_revision"
    STALE_POLICY = "stale_policy"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    UNAVAILABLE_DEPENDENCY = "unavailable_dependency"
    PARTIAL_RESULT = "partial_result"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


class PublicError(ContractModel):
    """A content-free public error suitable for transport adapters."""

    code: PublicErrorCode
    message: str
    retryable: bool


_PUBLIC_ERRORS: dict[InternalErrorCode, PublicError] = {
    InternalErrorCode.INVALID_SCHEMA: PublicError(
        code=PublicErrorCode.INVALID_REQUEST,
        message="The request is invalid.",
        retryable=False,
    ),
    InternalErrorCode.UNAUTHENTICATED: PublicError(
        code=PublicErrorCode.UNAUTHENTICATED,
        message="Authentication is required.",
        retryable=False,
    ),
    InternalErrorCode.DENIED: PublicError(
        code=PublicErrorCode.NOT_FOUND,
        message="The requested resource was not found.",
        retryable=False,
    ),
    InternalErrorCode.UNSUPPORTED_CAPABILITY: PublicError(
        code=PublicErrorCode.UNSUPPORTED_CAPABILITY,
        message="The requested capability is not supported.",
        retryable=False,
    ),
    InternalErrorCode.MISSING_RESOURCE: PublicError(
        code=PublicErrorCode.NOT_FOUND,
        message="The requested resource was not found.",
        retryable=False,
    ),
    InternalErrorCode.WRONG_REFERENCE_KIND: PublicError(
        code=PublicErrorCode.NOT_FOUND,
        message="The requested resource was not found.",
        retryable=False,
    ),
    InternalErrorCode.STALE_REVISION: PublicError(
        code=PublicErrorCode.NOT_FOUND,
        message="The requested resource was not found.",
        retryable=False,
    ),
    InternalErrorCode.STALE_POLICY: PublicError(
        code=PublicErrorCode.STALE_POLICY,
        message="The policy context is stale.",
        retryable=False,
    ),
    InternalErrorCode.IDEMPOTENCY_CONFLICT: PublicError(
        code=PublicErrorCode.IDEMPOTENCY_CONFLICT,
        message="The idempotency key conflicts with an earlier request.",
        retryable=False,
    ),
    InternalErrorCode.UNAVAILABLE_DEPENDENCY: PublicError(
        code=PublicErrorCode.UNAVAILABLE_DEPENDENCY,
        message="A required dependency is unavailable.",
        retryable=True,
    ),
    InternalErrorCode.PARTIAL_RESULT: PublicError(
        code=PublicErrorCode.PARTIAL_RESULT,
        message="Only a partial result is available.",
        retryable=True,
    ),
    InternalErrorCode.CANCELLED: PublicError(
        code=PublicErrorCode.CANCELLED,
        message="The operation was cancelled.",
        retryable=False,
    ),
    InternalErrorCode.TERMINAL_FAILURE: PublicError(
        code=PublicErrorCode.INTERNAL_ERROR,
        message="The operation failed.",
        retryable=False,
    ),
}

_DISCLOSURE_ERRORS: dict[InternalErrorCode, PublicError] = {
    InternalErrorCode.DENIED: PublicError(
        code=PublicErrorCode.PERMISSION_DENIED,
        message="Permission was denied.",
        retryable=False,
    ),
    InternalErrorCode.WRONG_REFERENCE_KIND: PublicError(
        code=PublicErrorCode.WRONG_REFERENCE_KIND,
        message="The reference kind does not match the resource.",
        retryable=False,
    ),
    InternalErrorCode.STALE_REVISION: PublicError(
        code=PublicErrorCode.STALE_REVISION,
        message="The supplied revision is stale.",
        retryable=False,
    ),
}


def public_error_for(
    code: InternalErrorCode,
    *,
    disclose_existence: bool = False,
) -> PublicError:
    """Return the safe public representation for an internal outcome.

    The disclosure decision must be an actual boolean; truthy values are not
    accepted as authorization.  The returned model is a copy so a consumer
    cannot mutate the canonical mapping used by later calls.
    """

    if type(disclose_existence) is not bool:
        raise TypeError("disclose_existence must be a bool")

    if disclose_existence is True and code in _DISCLOSURE_ERRORS:
        return _DISCLOSURE_ERRORS[code].model_copy()
    return _PUBLIC_ERRORS[code].model_copy()


__all__ = [
    "InternalErrorCode",
    "PublicError",
    "PublicErrorCode",
    "public_error_for",
]
