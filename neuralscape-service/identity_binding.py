"""Collision-free identity binding, namespace codec and legacy adjudication.

Pure building blocks that need no durable ledger:

* ``resolve_binding`` / ``bind_identity`` map a verbatim ``(issuer,
  provider_subject)`` pair to an opaque, generated subject id over an injected
  :class:`BindingStore`.  No email, display name or slug takes part.
* ``encode_namespace`` / ``decode_namespace`` are a versioned, length-prefixed
  tuple encoding whose decode is the exact inverse of encode.  The encoding is
  an index key only; authorization compares tuples.
* ``validate_service_account_controller`` records and checks who may control a
  service account.
* ``classify_legacy_identities`` separates legacy ``user_id`` values that map
  to exactly one verified identity from ones that must be quarantined.

``InMemoryBindingStore`` is a REFERENCE implementation for tests.  A durable
store with an atomic unique constraint arrives with the selected ledger.
The authority store (``authority_store_sql``) is the durable ``BindingStore``; ``credentials`` and
``policy_enforcement`` resolve identities through it when they are switched on, and nothing here changes
authentication while they are off.
"""

from __future__ import annotations

import secrets
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Literal, NamedTuple, Protocol

from contracts_policy import PrincipalContext
from identity import slugify_email

MAX_COMPONENT_LENGTH = 256
SUBJECT_ID_PREFIX = "sub_"


# ── binding ──────────────────────────────────────────────────────────────


class BindingStore(Protocol):
    """Atomic storage for ``(issuer, provider_subject) -> subject_id``."""

    def get(self, issuer: str, provider_subject: str) -> str | None: ...

    def put_if_absent(self, issuer: str, provider_subject: str, subject_id: str) -> str:
        """Store the id unless the pair is bound; return the winning id."""
        ...


class InMemoryBindingStore:
    """Reference store (NOT durable).  Keys are compared verbatim."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bindings: dict[tuple[str, str], str] = {}

    def get(self, issuer: str, provider_subject: str) -> str | None:
        with self._lock:
            return self._bindings.get((issuer, provider_subject))

    def put_if_absent(self, issuer: str, provider_subject: str, subject_id: str) -> str:
        with self._lock:
            return self._bindings.setdefault((issuer, provider_subject), subject_id)


class BindingResult(NamedTuple):
    subject_id: str
    created: bool


def _checked_component(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    if not value or len(value) > MAX_COMPONENT_LENGTH:
        raise ValueError(f"{name} must be 1..{MAX_COMPONENT_LENGTH} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{name} must be valid unicode") from None
    return value


def new_subject_id() -> str:
    """Generate an opaque subject id.  Never derived from any identity data."""

    return SUBJECT_ID_PREFIX + secrets.token_hex(16)


def resolve_binding(
    store: BindingStore, issuer: str, provider_subject: str
) -> str | None:
    """Return the bound subject id, or ``None``.  Read-only; never creates."""

    return store.get(
        _checked_component("issuer", issuer),
        _checked_component("provider_subject", provider_subject),
    )


def bind_identity(
    store: BindingStore,
    issuer: str,
    provider_subject: str,
    *,
    generate_id: Callable[[], str] = new_subject_id,
) -> BindingResult:
    """Return the subject for the verbatim pair, creating it exactly once.

    Concurrent callers converge on the store's winning id.  The pair is never
    case-folded, trimmed or otherwise normalized: values that differ in any
    character are different identities.
    """

    issuer = _checked_component("issuer", issuer)
    provider_subject = _checked_component("provider_subject", provider_subject)
    existing = store.get(issuer, provider_subject)
    if existing is not None:
        return BindingResult(existing, False)
    candidate = generate_id()
    winner = store.put_if_absent(issuer, provider_subject, candidate)
    return BindingResult(winner, winner == candidate)


# ── namespace codec ──────────────────────────────────────────────────────

NAMESPACE_VERSION = "ns1"
PoolKind = Literal["private", "shared", "standard"]
_POOL_KINDS = ("private", "shared", "standard")
_FIELD_NAMES = ("tenant", "pool_kind", "owner", "project", "workspace")


class NamespaceKey(NamedTuple):
    """Structured namespace; ``owner`` is required for (only) private pools."""

    tenant: str
    pool_kind: str
    owner: str | None
    project: str | None
    workspace: str | None


def _validated_key(key: NamespaceKey) -> NamespaceKey:
    key = NamespaceKey(*key)
    _checked_component("tenant", key.tenant)
    if key.pool_kind not in _POOL_KINDS:
        raise ValueError("unsupported pool kind")
    for name in ("owner", "project", "workspace"):
        value = getattr(key, name)
        if value is not None:
            _checked_component(name, value)
    if (key.pool_kind == "private") != (key.owner is not None):
        raise ValueError("only private pools take an owner, and they require one")
    return key


def encode_namespace(key: NamespaceKey) -> str:
    """Encode as ``ns1`` followed by one segment per field.

    A segment is ``~`` for an absent value, else ``<utf8-byte-length>:<value>``.
    """

    key = _validated_key(key)
    parts = [NAMESPACE_VERSION]
    for value in key:
        if value is None:
            parts.append("~")
        else:
            raw = value.encode("utf-8")
            parts.append(f"{len(raw)}:{value}")
    return "".join(parts)


def decode_namespace(encoded: str) -> NamespaceKey:
    """Strict inverse of :func:`encode_namespace`; raises ``ValueError``."""

    if type(encoded) is not str:
        raise ValueError("namespace must be a string")
    try:
        data = encoded.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("namespace must be valid unicode") from None
    prefix = NAMESPACE_VERSION.encode()
    if not data.startswith(prefix):
        raise ValueError("unsupported namespace version")
    position = len(prefix)
    values: list[str | None] = []
    for _ in _FIELD_NAMES:
        if position >= len(data):
            raise ValueError("truncated namespace")
        if data[position : position + 1] == b"~":
            values.append(None)
            position += 1
            continue
        colon = data.find(b":", position)
        digits = data[position:colon] if colon != -1 else b""
        if not digits or not digits.isdigit() or (len(digits) > 1 and digits[:1] == b"0"):
            raise ValueError("malformed namespace length")
        length = int(digits)
        start = colon + 1
        end = start + length
        if length == 0 or length > MAX_COMPONENT_LENGTH * 4 or end > len(data):
            raise ValueError("malformed namespace length")
        try:
            values.append(data[start:end].decode("utf-8"))
        except UnicodeDecodeError:
            raise ValueError("malformed namespace value") from None
        position = end
    if position != len(data):
        raise ValueError("trailing namespace data")
    key = _validated_key(NamespaceKey(*values))
    if encode_namespace(key) != encoded:
        raise ValueError("non-canonical namespace")
    return key


# ── service accounts ─────────────────────────────────────────────────────


def validate_service_account_controller(
    controller: PrincipalContext, *, service_account_subject_id: str
) -> None:
    """Check that ``controller`` may create or control a service account.

    Least authority: only a human subject may control a service account, so a
    service account, worker or delegated agent can never mint another (no
    delegation upward), and an account cannot control itself.  The caller
    records ``controller.subject_id`` with the account; the controller gains no
    rights from that record and the account inherits none from the controller.
    """

    if controller.subject_kind != "human":
        raise ValueError("only a human subject may control a service account")
    if controller.subject_id == service_account_subject_id:
        raise ValueError("a service account cannot control itself")


# ── legacy adjudication ──────────────────────────────────────────────────


class LegacyObservation(NamedTuple):
    """One observed login for a legacy ``user_id``."""

    legacy_user_id: str
    issuer: str | None = None
    provider_subject: str | None = None
    email: str | None = None


class LegacyClassification(NamedTuple):
    status: Literal["unique", "ambiguous", "unbound"]
    reasons: tuple[str, ...]
    identities: tuple[tuple[str, str], ...]


_NAMESPACE_DELIMITERS = ("--project--", "--ws--")


def classify_legacy_identities(
    observations: Iterable[LegacyObservation],
) -> dict[str, LegacyClassification]:
    """Classify each legacy ``user_id`` as unique, ambiguous or unbound.

    * ``unique``: exactly one verified identity observed, that identity maps
      to no other legacy id, and no delimiter or slug collision applies.
    * ``ambiguous`` (quarantine): several identities share the legacy id, one
      identity appears under several legacy ids, two different emails fold to
      the same slug, or the id contains a legacy namespace delimiter.
    * ``unbound`` (quarantine): no verified identity was ever observed.

    Nothing is merged or guessed; the result only feeds an adjudication record.
    """

    identities: dict[str, set[tuple[str, str]]] = defaultdict(set)
    emails: dict[str, set[str]] = defaultdict(set)
    seen: list[str] = []
    for observation in observations:
        legacy = observation.legacy_user_id
        if legacy not in identities:
            seen.append(legacy)
        identities.setdefault(legacy, set())
        if observation.issuer is not None and observation.provider_subject is not None:
            identities[legacy].add((observation.issuer, observation.provider_subject))
        if observation.email is not None:
            emails[legacy].add(observation.email)

    owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    for legacy, pairs in identities.items():
        for pair in pairs:
            owners[pair].add(legacy)

    slug_sources: dict[str, set[str]] = defaultdict(set)
    for addresses in emails.values():
        for address in addresses:
            slug_sources[slugify_email(address)].add(address)

    result: dict[str, LegacyClassification] = {}
    for legacy in seen:
        pairs = identities[legacy]
        reasons: list[str] = []
        if len(pairs) > 1:
            reasons.append("multiple_identities")
        if any(len(owners[pair]) > 1 for pair in pairs):
            reasons.append("identity_under_multiple_legacy_ids")
        if any(len(slug_sources[slugify_email(address)]) > 1 for address in emails[legacy]):
            reasons.append("email_slug_collision")
        if any(delimiter in legacy for delimiter in _NAMESPACE_DELIMITERS):
            reasons.append("namespace_delimiter")
        if reasons:
            status = "ambiguous"
        elif not pairs:
            status, reasons = "unbound", ["no_verified_identity"]
        else:
            status = "unique"
        result[legacy] = LegacyClassification(
            status, tuple(sorted(reasons)), tuple(sorted(pairs))
        )
    return result


__all__ = [
    "BindingResult",
    "BindingStore",
    "InMemoryBindingStore",
    "LegacyClassification",
    "LegacyObservation",
    "NamespaceKey",
    "bind_identity",
    "classify_legacy_identities",
    "decode_namespace",
    "encode_namespace",
    "new_subject_id",
    "resolve_binding",
    "validate_service_account_controller",
]
