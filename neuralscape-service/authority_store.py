"""Current-authority interface consumed by the policy enforcement path.

The store answers two questions for one tenant: what is the current
``AccessPolicy`` snapshot (policy epoch, membership versions, statements), and
what verified facts exist about a bound subject (tenant, kind, credential,
membership version).  It is read-only here; publication and durable state come
from the durable store implementation.

``InMemoryAuthorityStore`` is a REFERENCE implementation for tests and local
use.  It is not durable and makes no cross-process consistency promise.
"""

from __future__ import annotations

import threading
from typing import NamedTuple, Protocol

from contracts_policy import AccessPolicy, DelegationConstraints, SubjectKind


class AuthorityUnavailable(Exception):
    """The authority could not be read.  Callers must fail closed."""


class SubjectRecord(NamedTuple):
    """Verified facts about a bound subject at one tenant."""

    tenant_id: str
    subject_kind: SubjectKind
    credential_id: str
    membership_version: int
    delegation: DelegationConstraints | None = None


class AuthorityStore(Protocol):
    def current_policy(self, tenant_id: str) -> AccessPolicy | None:
        """The tenant's current snapshot, or ``None`` when it has no policy."""
        ...

    def subject_record(self, subject_id: str) -> SubjectRecord | None:
        """The subject's current record, or ``None`` when it is not a member."""
        ...


class InMemoryAuthorityStore:
    """Reference store (NOT durable)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._policies: dict[str, AccessPolicy] = {}
        self._subjects: dict[str, SubjectRecord] = {}
        self._available = True

    def set_policy(self, policy: AccessPolicy) -> None:
        with self._lock:
            self._policies[policy.tenant_id] = policy

    def set_subject(self, subject_id: str, record: SubjectRecord) -> None:
        with self._lock:
            self._subjects[subject_id] = record

    def remove_subject(self, subject_id: str) -> None:
        with self._lock:
            self._subjects.pop(subject_id, None)

    def set_available(self, available: bool) -> None:
        with self._lock:
            self._available = available

    def _check(self) -> None:
        if not self._available:
            raise AuthorityUnavailable("authority store unavailable")

    def current_policy(self, tenant_id: str) -> AccessPolicy | None:
        with self._lock:
            self._check()
            return self._policies.get(tenant_id)

    def tenants(self) -> tuple[str, ...]:
        with self._lock:
            self._check()
            return tuple(sorted(self._policies))

    def subject_record(self, subject_id: str) -> SubjectRecord | None:
        with self._lock:
            self._check()
            return self._subjects.get(subject_id)


__all__ = [
    "AuthorityStore",
    "AuthorityUnavailable",
    "InMemoryAuthorityStore",
    "SubjectRecord",
]
