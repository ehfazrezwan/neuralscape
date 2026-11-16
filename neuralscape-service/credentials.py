"""Scoped credentials, single-use refresh tokens and service accounts.

A scoped credential is bound to ``(subject, tenant, scope, expiry)``:

* ids are opaque random values; the secret half of a token is random and only
  its SHA-256 is stored, so the database never holds a usable token;
* ``scope`` is a set of policy actions that must be a subset of what the
  subject is granted *at issue time*, and the enforcement layer additionally
  refuses any operation outside it at every request (narrowing only);
* a refresh token is single use.  Redeeming it issues a successor (new access
  and refresh) in the same family and retires the family's previous access
  credential.  Presenting an already-consumed refresh token is a replay and
  revokes the whole family;
* revoking a credential or offboarding the subject retires it in the authority
  store.  The publication fence (``publication_fence``/``hold_current``) is what will stop queued work at
  publication, but it has no caller in this slice (wired by slices 05/07), so queued work is not yet stopped;
* service accounts are owned by a recorded human controller, hold only their
  own statements, may have at most two active credential families (so a key
  can be rotated without downtime), and can be disabled.

Everything here is off unless ``CREDENTIALS_ENABLED`` is set, existing tokens
and the legacy profile are untouched, and scoped credentials are accepted only
while policy enforcement is on: a scope that nothing enforces would be a lie.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from typing import NamedTuple

from authority_store import AuthorityUnavailable
from authority_store_sql import (
    CredentialInvalid,
    CredentialLimit,
    RefreshReplay,
    SqlAuthorityStore,
    UnknownSubject,
    VerifiedCredential,
    _ident,
    build_authority_store,
)
from config import settings
from contracts_policy import SUPPORTED_POLICY_ACTIONS
from identity_binding import new_subject_id

ACCESS_PREFIX = "nsc_"
REFRESH_PREFIX = "nsr_"
SERVICE_ACCOUNT_MAX_FAMILIES = 2
DEFAULT_ACCESS_TTL = 15 * 60
DEFAULT_REFRESH_TTL = 30 * 24 * 3600
MAX_ACCESS_TTL = 24 * 3600
MAX_REFRESH_TTL = 90 * 24 * 3600
# A duplicate refresh submission within this many seconds of the first (a lost
# response, two tabs) returns the same successor instead of revoking the family.
DEFAULT_REFRESH_GRACE = 5.0
# The subject -> memory ``user_id`` lookup is cached (see ``CredentialService.memory_user_id``).
_ALIAS_CACHE_MAX = 10_000
_ALIAS_MISS_TTL = 5.0

# The credential verified for the current request (HTTP) or MCP call.
current_credential: ContextVar[VerifiedCredential | None] = ContextVar(
    "current_credential", default=None
)


class IssuedCredentials(NamedTuple):
    access_token: str
    refresh_token: str | None
    credential_id: str
    family_id: str
    scope: tuple[str, ...]
    access_expires_at: float
    refresh_expires_at: float | None


class ScopeError(ValueError):
    """The requested scope is empty, unknown or exceeds the subject's grants."""


def enabled() -> bool:
    return bool(settings.credentials_enabled)


def accepts_scoped_credentials() -> bool:
    """Scoped credentials authenticate only while policy enforcement is on."""

    return enabled() and settings.ns_policy_mode == "enforce"


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _new_token(prefix: str) -> tuple[str, str, str]:
    credential_id = "cred_" + secrets.token_hex(12)
    secret = secrets.token_urlsafe(32)
    return credential_id, _hash_secret(secret), f"{prefix}{credential_id}.{secret}"


def parse_token(token: str, prefix: str) -> tuple[str, str] | None:
    """Return ``(credential_id, secret_hash)`` for a well-formed token, else ``None``."""

    if type(token) is not str or not token.startswith(prefix) or len(token) > 512:
        return None
    credential_id, dot, secret = token[len(prefix):].partition(".")
    if not dot or not credential_id.startswith("cred_") or not secret:
        return None
    return credential_id, _hash_secret(secret)


class CredentialService:
    def __init__(
        self,
        store: SqlAuthorityStore,
        *,
        clock: Callable[[], float] = time.time,
        access_ttl: float = DEFAULT_ACCESS_TTL,
        refresh_ttl: float = DEFAULT_REFRESH_TTL,
        refresh_grace: float = DEFAULT_REFRESH_GRACE,
        allow_scoped_refresh_minting: bool = False,
    ) -> None:
        self._store = store
        self._clock = clock
        self._access_ttl = access_ttl
        self._refresh_ttl = refresh_ttl
        self._refresh_grace = refresh_grace
        self._allow_scoped_refresh_minting = allow_scoped_refresh_minting
        self._pepper: bytes | None = None
        self._aliases: dict[str, tuple[str | None, float]] = {}
        self._alias_lock = threading.Lock()

    # ── scope ────────────────────────────────────────────────────────────

    def granted_actions(self, subject_id: str) -> frozenset[str]:
        """Actions the subject currently holds an allow statement for."""

        _ident("subject_id", subject_id)  # compared below: a str subclass could claim every statement
        summary = self._store.subject_summary(subject_id)
        if summary is None:
            raise UnknownSubject(subject_id)
        policy = self._store.current_policy(summary[0])
        if policy is None:
            return frozenset()
        return frozenset(
            s.action
            for s in policy.statements
            if s.subject_id == subject_id and s.effect == "allow"
        )

    def _checked_scope(self, subject_id: str, scope: Sequence[str]) -> tuple[str, ...]:
        try:
            wanted = set(scope)
        except TypeError:
            raise ScopeError("scope must be a list of action names") from None
        if not wanted:
            raise ScopeError("scope must name at least one action")
        if not wanted <= SUPPORTED_POLICY_ACTIONS:
            raise ScopeError("scope names an unknown action")
        if not wanted <= self.granted_actions(subject_id):
            raise ScopeError("scope exceeds the subject's grants")
        return tuple(sorted(wanted))

    def _ttl(self, requested: float | None, default: float, ceiling: float) -> float:
        value = default if requested is None else float(requested)
        if not 0 < value <= ceiling:
            raise ValueError("ttl is outside the allowed range")
        return value

    # ── issue / verify / refresh / revoke ────────────────────────────────

    def issue(
        self,
        subject_id: str,
        scope: Sequence[str],
        *,
        access_ttl: float | None = None,
        refresh_ttl: float | None = None,
        with_refresh: bool = True,
        ceiling: Sequence[str] | None = None,
        issuer: VerifiedCredential | None = None,
    ) -> IssuedCredentials:
        """Issue a credential family.

        ``issuer`` is the scoped credential doing the minting, if any.  A
        minted credential can never outlive it (its access expiry is capped at
        the issuer's), cannot exceed its scope, cannot carry a refresh token
        unless that was explicitly permitted, and is revoked with the issuer's
        family.
        """

        checked = self._checked_scope(subject_id, scope)
        if issuer is not None:
            ceiling = issuer.scope if ceiling is None else ceiling
            if with_refresh and not self._allow_scoped_refresh_minting:
                raise ScopeError("a scoped credential cannot mint a refresh token")
        if ceiling is not None and not set(checked) <= set(ceiling):
            # A scoped credential can only mint credentials within its own scope.
            raise ScopeError("scope exceeds the issuing credential's scope")
        now = self._clock()
        access_expires = now + self._ttl(access_ttl, self._access_ttl, MAX_ACCESS_TTL)
        if issuer is not None:
            if issuer.expires_at <= now:
                raise ScopeError("the issuing credential has expired")
            access_expires = min(access_expires, issuer.expires_at)
        access_id, access_hash, access_token = _new_token(ACCESS_PREFIX)
        refresh = None
        refresh_expires = None
        refresh_token = None
        if with_refresh:
            refresh_expires = now + self._ttl(refresh_ttl, self._refresh_ttl, MAX_REFRESH_TTL)
            if issuer is not None:
                refresh_expires = min(refresh_expires, issuer.expires_at)
            refresh_id, refresh_hash, refresh_token = _new_token(REFRESH_PREFIX)
            refresh = (refresh_id, refresh_hash, refresh_expires)
        summary = self._store.subject_summary(subject_id)
        limit = SERVICE_ACCOUNT_MAX_FAMILIES if summary and summary[1] == "service_account" else None
        family_id = "fam_" + secrets.token_hex(12)
        self._store.issue_family(
            subject_id=subject_id,
            scope=checked,
            family_id=family_id,
            access=(access_id, access_hash, access_expires),
            refresh=refresh,
            max_families=limit,
            parent_family=issuer.family_id if issuer is not None else None,
            cap_expires_at=issuer.expires_at if issuer is not None else None,
        )
        return IssuedCredentials(
            access_token, refresh_token, access_id, family_id, checked, access_expires, refresh_expires
        )

    def verify_access(self, token: str) -> VerifiedCredential | None:
        parsed = parse_token(token, ACCESS_PREFIX)
        if parsed is None:
            return None
        return self._store.verify_access(*parsed)

    def _derive_successor(self, prefix: str, label: str, presented_secret: str) -> tuple[str, str, str]:
        """The successor token for a presented refresh secret, derived with a server secret.

        Deterministic, so a benign duplicate submission recomputes exactly the
        successor the first one created (nothing secret is stored to replay it),
        while nobody without the server secret can compute it from the stored hash.
        """

        pepper = self._pepper
        if pepper is None:
            pepper = self._pepper = self._store.pepper()

        def part(kind: str) -> bytes:
            return hmac.new(pepper, f"{label}|{kind}|{presented_secret}".encode(), hashlib.sha256).digest()

        credential_id = "cred_" + part("id")[:12].hex()
        secret = base64.urlsafe_b64encode(part("secret")).decode().rstrip("=")
        return credential_id, _hash_secret(secret), f"{prefix}{credential_id}.{secret}"

    def refresh(self, refresh_token: str) -> IssuedCredentials:
        """Redeem a refresh token once.  Raises ``CredentialInvalid`` or ``RefreshReplay``.

        A duplicate submission within the grace window returns the same
        successor; any later reuse of a consumed token is a replay and revokes
        the family.
        """

        parsed = parse_token(refresh_token, REFRESH_PREFIX)
        if parsed is None:
            raise CredentialInvalid("malformed refresh token")
        presented_secret = refresh_token[len(REFRESH_PREFIX):].split(".", 1)[1]
        now = self._clock()
        access_id, access_hash, access_token = self._derive_successor(ACCESS_PREFIX, "access", presented_secret)
        refresh_id, refresh_hash, new_refresh = self._derive_successor(REFRESH_PREFIX, "refresh", presented_secret)
        verified, refresh_expires = self._store.rotate_refresh(
            parsed[0],
            parsed[1],
            access=(access_id, access_hash, now + self._access_ttl),
            refresh=(refresh_id, refresh_hash, now + self._refresh_ttl),
            grace_seconds=self._refresh_grace,
        )
        return IssuedCredentials(
            access_token, new_refresh, access_id, verified.family_id,
            verified.scope, verified.expires_at, refresh_expires,
        )

    def revoke(self, credential_id: str, *, by_subject_id: str) -> bool:
        """Revoke a credential's whole family.

        Allowed for the credential's own subject, or for the human controller of
        the service account that owns it.  Returns False for anything else, so
        a caller learns nothing about credentials that are not theirs.
        """

        _ident("by_subject_id", by_subject_id)  # before the lookup: the same answer, known credential or not
        info = self._store.credential_info(credential_id)
        if info is None or not self._may_manage(info["subject_id"], by_subject_id):
            return False
        self._store.revoke_family(info["family_id"])
        return True

    def _may_manage(self, subject_id: str, by_subject_id: str) -> bool:
        _ident("by_subject_id", by_subject_id)  # a str subclass could claim to be the owner
        if subject_id == by_subject_id:
            return True
        account = self._store.service_account(subject_id)
        return account is not None and account["controller_subject_id"] == by_subject_id

    def subject_for_user_id(self, user_id: str) -> str | None:
        """The bound subject of an existing-token user (``("local", user_id)``)."""

        from identity_binding import resolve_binding

        return resolve_binding(self._store, "local", user_id)

    def memory_user_id(self, subject_id: str) -> str:
        """The ``user_id`` the memory layer keys this subject's rows by.

        Rows are stored under the legacy ``user_id`` an existing-token user wrote them with, and the
        ``("local", user_id)`` binding is what ties that id to the subject.  A scoped credential names
        the opaque subject, so the memory layer must be told the bound ``user_id`` or it would look in
        an empty namespace (reads empty, get 404, while the owner token sees the rows).  A subject
        with no such binding (a service account) owns rows under its own subject id.
        """

        with self._alias_lock:
            hit = self._aliases.get(subject_id)
            if hit is not None and (hit[0] is not None or hit[1] > self._clock()):
                return hit[0] or subject_id
        alias = self._store.binding_alias(subject_id, "local")
        with self._alias_lock:
            if len(self._aliases) >= _ALIAS_CACHE_MAX:
                self._aliases.clear()
            # A binding is never changed once made, so a hit is kept; a miss only briefly, because a
            # binding may be created for the subject later.
            self._aliases[subject_id] = (alias, self._clock() + _ALIAS_MISS_TTL)
        return alias or subject_id

    # ── service accounts ─────────────────────────────────────────────────

    def create_service_account(self, controller_subject_id: str, label: str) -> str:
        """Create an account owned by a human controller.  It has no credential yet."""

        summary = self._store.subject_summary(controller_subject_id)
        if summary is None:
            raise UnknownSubject(controller_subject_id)
        if not label or len(label) > 128:
            raise ValueError("label must be 1..128 characters")
        subject_id = new_subject_id()
        self._store.create_service_account(subject_id, summary[0], controller_subject_id, label)
        return subject_id

    def rotate_service_account(
        self, subject_id: str, controller_subject_id: str, scope: Sequence[str], **options: object
    ) -> IssuedCredentials:
        """Issue a new credential family.  At most two stay active, so the old
        key keeps working until the controller revokes it."""

        self._require_controller(subject_id, controller_subject_id)
        return self.issue(subject_id, scope, **options)  # type: ignore[arg-type]

    def disable_service_account(self, subject_id: str, controller_subject_id: str) -> bool:
        self._require_controller(subject_id, controller_subject_id)
        return self._store.disable_service_account(subject_id)

    def _require_controller(self, subject_id: str, controller_subject_id: str) -> None:
        account = self._store.service_account(subject_id)
        if account is None or account["controller_subject_id"] != controller_subject_id:
            raise UnknownSubject(subject_id)  # not disclosed as a permission error


# ── process-wide wiring ─────────────────────────────────────────────────────

_service: CredentialService | None = None


def configure(service: CredentialService | None) -> None:
    global _service
    _service = service


def get_service() -> CredentialService | None:
    """The configured service, or one built from ``AUTHORITY_*`` settings."""

    global _service
    if _service is None and enabled():
        store = build_authority_store(settings)
        if store is not None:
            _service = CredentialService(store)
    return _service


__all__ = [
    "AuthorityUnavailable",
    "CredentialInvalid",
    "CredentialLimit",
    "CredentialService",
    "IssuedCredentials",
    "RefreshReplay",
    "ScopeError",
    "UnknownSubject",
    "accepts_scoped_credentials",
    "configure",
    "current_credential",
    "enabled",
    "get_service",
    "parse_token",
]
