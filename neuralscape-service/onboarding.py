"""Membership: invite, join, list, remove, and the onboarding policy.

A new member needs more than content grants to use the product: tenant-level
``read`` (capabilities, console sign-in), and the relay
actions when encrypted sync is on.  Those defaults are an explicit, versioned
``OnboardingPolicy`` stored per tenant, not an accident of whoever set the member
up:

* ``hosted-v1`` is the default for a new hosted tenant: tenant read, own private
  pools in full (including export of their own data and the background derivation of what
  they save, which the write path's publication stage requires), read access to the shared
  and standard pools, and no right to create shared content or edit shared rows
  until a grant adds it;
* ``preserve-v1`` reproduces the sharing a migrated deployment already had (the
  migration stores it for the tenant): own private pools, shared read/create/share
  with author-only edits and team metadata edits, standard read, and tenant read.

A default policy can never grant ``manage_membership``, ``manage_credentials`` or
``publish_standard`` (administrator and standards authority stay separate), it
cannot give shared-pool ``update``/``delete`` except through the row rules, and it
can never give tenant-wide ``export`` (that exports every member's records and
files, so it is granted explicitly to a named member).  Tenant-level defaults
(``tenant_actions`` and ``relay_actions``) are limited to ``TENANT_DEFAULT_ACTIONS``, so
tenant-wide ``restore`` and the background and destructive actions are never defaults.

Console sign-in needs the tenant-level ``read`` grant: a member without it cannot
sign in to the console or read the capabilities, so every preset grants it and a
custom policy that drops it leaves its members unable to sign in.

A tenant always keeps at least one administrator: removing the last member who
holds an unconditional ``manage_membership`` grant is refused, atomically, so two
administrators removing each other at once leave one.  An administrator can invite
an identity they control; it joins with the ordinary default grants only (never
administrator authority, never another member's private content), the member list
shows every identity, and a tenant holds at most ``MAX_OUTSTANDING_INVITES`` unused
invites at a time.

Membership operations are the ``manage_membership`` action; joining is
authenticated by a verified identity plus a single-use invite.  Invites are
opaque, tenant-bound, expiring and stored hashed.  Everything is off unless
``ONBOARDING_ENABLED`` is set.  Grants cover the project-less pools; a project
pool needs its own grant.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from collections.abc import Callable
from typing import Literal, NamedTuple

from pydantic import model_validator

from authority_store import AuthorityUnavailable
from authority_store_sql import (
    LastAdministrator,
    SqlAuthorityStore,
    TooManyInvites,
    UnknownSubject,
    _ident,
    build_authority_store,
)
from config import settings
from contracts_common import ContractModel, OpaqueId
from contracts_policy import (
    SUPPORTED_POLICY_ACTIONS,
    PolicyCondition,
    PolicyStatement,
)
from identity_binding import new_subject_id, resolve_binding
from policy_enforcement import pool_handle, tenant_handle

INVITE_PREFIX = "nsi_"
DEFAULT_INVITE_TTL = 3 * 24 * 3600
MAX_INVITE_TTL = 14 * 24 * 3600
MAX_OUTSTANDING_INVITES = 20
LOCAL_ISSUER = "local"
FORBIDDEN_DEFAULT_ACTIONS = frozenset({"manage_membership", "manage_credentials", "publish_standard"})
_ROW_EDIT_ACTIONS = frozenset({"update", "delete"})
# The only actions a default may grant on the tenant handle (``tenant_actions`` and, when encrypted
# sync is on, ``relay_actions``).  Tenant-level ``export`` and ``restore`` act on every member's data,
# and the background and destructive actions act tenant-wide, so they are only ever explicit grants;
# an allowlist keeps a tenant-level action added later out of the defaults until it is chosen here.
TENANT_DEFAULT_ACTIONS = frozenset({"read", "count", "list", "create", "update"})
_OWNER = PolicyCondition(version="condition-v1", kind="row_owner_is_principal")
_METADATA = PolicyCondition(
    version="condition-v1",
    kind="fields_subset_of",
    values=("category", "concepts", "confidence", "domain", "expires_at",
            "observation_type", "project_id", "tags"),
)


class OnboardingPolicy(ContractModel):
    """The default grants a new member receives (versioned)."""

    version: Literal["onboarding-v1"]
    name: OpaqueId
    tenant_actions: tuple[OpaqueId, ...]
    private_actions: tuple[OpaqueId, ...]
    shared_actions: tuple[OpaqueId, ...]
    shared_row_rules: bool
    standard_read: bool
    relay_actions: tuple[OpaqueId, ...]

    @model_validator(mode="after")
    def _validate(self) -> "OnboardingPolicy":
        for name in ("tenant_actions", "private_actions", "shared_actions", "relay_actions"):
            actions = getattr(self, name)
            if len(actions) != len(set(actions)):
                raise ValueError(f"{name} must not repeat an action")
            if not set(actions) <= SUPPORTED_POLICY_ACTIONS:
                raise ValueError(f"{name} names an unknown action")
            if set(actions) & FORBIDDEN_DEFAULT_ACTIONS:
                raise ValueError(f"{name} cannot grant administrator or standards authority")
        # ``relay_actions`` are granted on the tenant handle too, so they are tenant-level grants.
        if "export" in self.tenant_actions or "export" in self.relay_actions:
            raise ValueError("tenant-wide export is never a default grant; grant it explicitly to a named member")
        for name in ("tenant_actions", "relay_actions"):
            if not set(getattr(self, name)) <= TENANT_DEFAULT_ACTIONS:
                raise ValueError(
                    f"{name} may grant only the tenant-level defaults {sorted(TENANT_DEFAULT_ACTIONS)};"
                    " restore and other tenant-wide actions are explicit grants"
                )
        if set(self.shared_actions) & _ROW_EDIT_ACTIONS:
            raise ValueError("shared edits and deletes come only from shared_row_rules")
        return self


HOSTED_V1 = OnboardingPolicy(
    version="onboarding-v1",
    name="hosted-v1",
    tenant_actions=("read",),
    private_actions=("read", "search", "list", "count", "create", "update", "delete", "export", "background_derive"),
    shared_actions=("read", "search", "list", "count"),
    shared_row_rules=False,
    standard_read=True,
    relay_actions=("create", "list", "update"),
)
PRESERVE_V1 = OnboardingPolicy(
    version="onboarding-v1",
    name="preserve-v1",
    tenant_actions=("read",),
    private_actions=("read", "search", "list", "count", "create", "update", "delete", "export", "background_derive"),
    shared_actions=("read", "search", "list", "count", "create", "share"),
    shared_row_rules=True,
    standard_read=True,
    relay_actions=("create", "list", "update"),
)
PRESETS = {"hosted-v1": HOSTED_V1, "preserve-v1": PRESERVE_V1}


def default_statements(
    policy: OnboardingPolicy, subject_id: str, tenant_id: str, *, relay_enabled: bool = False
) -> tuple[PolicyStatement, ...]:
    """The explicit statements a new member receives under ``policy``."""

    out: dict[str, PolicyStatement] = {}

    def grant(actions, handle, condition: PolicyCondition | None = None) -> None:
        tag = condition.kind if condition else ""
        for action in actions:
            digest = hashlib.sha256(f"onb|{subject_id}|{action}|{handle.id}|{tag}".encode()).hexdigest()
            sid = "onb-" + digest[:28]
            out[sid] = PolicyStatement(
                id=sid, effect="allow", subject_id=subject_id, credential_id=None, action=action,
                resource=handle, condition=condition, origin=f"onboarding:{policy.name}",
            )

    grant(policy.tenant_actions, tenant_handle(tenant_id))
    if relay_enabled:
        grant(policy.relay_actions, tenant_handle(tenant_id))
    grant(policy.private_actions, pool_handle(tenant_id, "private", subject_id))
    shared = pool_handle(tenant_id, "shared", None)
    grant(policy.shared_actions, shared)
    if policy.shared_row_rules:
        grant(("update", "delete"), shared, _OWNER)
        grant(("update",), shared, _METADATA)
    if policy.standard_read:
        grant(("read", "search", "list", "count"), pool_handle(tenant_id, "standard", None))
    return tuple(out[k] for k in sorted(out))


class JoinResult(NamedTuple):
    subject_id: str
    tenant_id: str
    policy_epoch: int
    policy_name: str


class IssuedInvite(NamedTuple):
    token: str
    invite_id: str
    expires_at: float


def enabled() -> bool:
    flag = getattr(settings, "onboarding_enabled", None)
    if flag is None:
        flag = os.environ.get("ONBOARDING_ENABLED", "").strip().lower() in {"1", "true", "yes"}
    return bool(flag)


def _relay_enabled() -> bool:
    return os.environ.get("ONBOARDING_RELAY_ENABLED", "").strip().lower() in {"1", "true", "yes"}


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def parse_invite(token: object) -> tuple[str, str] | None:
    """``(invite_id, secret_hash)`` for a well-formed invite token, else ``None``."""

    if type(token) is not str or not token.startswith(INVITE_PREFIX) or len(token) > 256:
        return None
    invite_id, dot, secret = token[len(INVITE_PREFIX):].partition(".")
    if not dot or not invite_id.startswith("inv_") or not secret:
        return None
    return invite_id, _hash(secret)


class OnboardingService:
    def __init__(
        self,
        store: SqlAuthorityStore,
        *,
        clock: Callable[[], float] = time.time,
        invite_ttl: float = DEFAULT_INVITE_TTL,
        relay_enabled: Callable[[], bool] = _relay_enabled,
    ) -> None:
        self._store = store
        self._clock = clock
        self._invite_ttl = invite_ttl
        self._relay_enabled = relay_enabled

    # ── identity helpers ─────────────────────────────────────────────────
    def subject_for_user_id(self, user_id: str) -> str | None:
        return resolve_binding(self._store, LOCAL_ISSUER, user_id)

    def tenant_of(self, subject_id: str) -> str | None:
        summary = self._store.subject_summary(subject_id)
        return summary[0] if summary else None

    # ── onboarding policy ────────────────────────────────────────────────
    def policy_for(self, tenant_id: str) -> OnboardingPolicy:
        """The tenant's stored policy, or ``hosted-v1`` for a tenant that has none."""

        stored = self._store.get_onboarding_policy(tenant_id)
        if stored is None:
            return HOSTED_V1
        return OnboardingPolicy.model_validate_json(stored[1])

    def set_policy(self, tenant_id: str, policy: OnboardingPolicy | str) -> OnboardingPolicy:
        """Set the tenant's policy from a preset name or a validated object."""

        if isinstance(policy, str):
            if policy not in PRESETS:
                raise ValueError("unknown onboarding policy")
            policy = PRESETS[policy]
        policy = OnboardingPolicy.model_validate(policy)
        self._store.set_onboarding_policy(tenant_id, policy.version, policy.model_dump_json())
        return policy

    # ── invites and joining ──────────────────────────────────────────────
    def invite(self, inviter_subject_id: str, *, ttl: float | None = None) -> IssuedInvite:
        tenant = self.tenant_of(inviter_subject_id)
        if tenant is None:
            raise UnknownSubject(inviter_subject_id)
        lifetime = self._invite_ttl if ttl is None else float(ttl)
        if not 0 < lifetime <= MAX_INVITE_TTL:
            raise ValueError("invite ttl is outside the allowed range")
        invite_id = "inv_" + secrets.token_hex(12)
        secret = secrets.token_urlsafe(32)
        expires = self._clock() + lifetime
        self._store.create_invite(
            tenant_id=tenant, created_by=inviter_subject_id, invite_id=invite_id,
            secret_hash=_hash(secret), expires_at=expires, max_outstanding=MAX_OUTSTANDING_INVITES,
        )
        return IssuedInvite(f"{INVITE_PREFIX}{invite_id}.{secret}", invite_id, expires)

    def join(self, user_id: str, invite_token: str) -> JoinResult:
        """Make the verified identity ``user_id`` a member through a single-use invite."""

        parsed = parse_invite(invite_token)
        if parsed is None:
            raise ValueError("invite is not usable")
        tenant = self._store.invite_tenant(*parsed)
        if tenant is None:
            raise ValueError("invite is not usable")
        policy = self.policy_for(tenant)
        relay = self._relay_enabled()
        subject_id = new_subject_id()
        _, epoch = self._store.join_with_invite(
            invite_id=parsed[0], secret_hash=parsed[1], alias_issuer=LOCAL_ISSUER,
            alias_subject=user_id, subject_id=subject_id, credential_id=f"token:{subject_id}",
            statements_for=lambda s, t: default_statements(policy, s, t, relay_enabled=relay),
        )
        return JoinResult(subject_id, tenant, epoch, policy.name)

    # ── members ──────────────────────────────────────────────────────────
    def grant_tenant_export(self, tenant_id: str, subject_id: str) -> int:
        """Explicitly let one member export the whole tenant; returns the new policy epoch."""

        _ident("tenant_id", tenant_id)  # compared below: a str subclass could match any tenant
        _ident("subject_id", subject_id)
        if self.tenant_of(subject_id) != tenant_id:
            raise UnknownSubject(subject_id)
        digest = hashlib.sha256(f"onb-export|{subject_id}|{tenant_id}".encode()).hexdigest()
        statement = PolicyStatement(
            id="onb-" + digest[:28], effect="allow", subject_id=subject_id, credential_id=None,
            action="export", resource=tenant_handle(tenant_id), origin="grant:tenant-export",
        )
        return self._store.grant_statements(tenant_id, [statement])

    def list_members(self, tenant_id: str) -> list[dict]:
        return self._store.list_members(tenant_id)

    def list_invites(self, tenant_id: str) -> list[dict]:
        """The tenant's outstanding invites: identifiers and times, never a secret."""

        return self._store.list_invites(tenant_id)

    def revoke_invite(self, tenant_id: str, invite_id: str) -> bool:
        """Revoke an outstanding invite of ``tenant_id``.  ``False`` for an invite that is unknown,
        of another tenant, used, expired or already revoked (all alike to the caller)."""

        _ident("tenant_id", tenant_id)
        if type(invite_id) is not str or not invite_id.startswith("inv_") or len(invite_id) > 64:
            return False
        if not any(row["invite_id"] == invite_id for row in self._store.list_invites(tenant_id)):
            return False
        return self._store.revoke_invite(invite_id, tenant_id=tenant_id)

    def remove_member(self, tenant_id: str, subject_id: str, *, by_subject_id: str) -> int:
        _ident("by_subject_id", by_subject_id)  # the store checks the other two
        if subject_id == by_subject_id:
            raise ValueError("a member cannot remove themselves")
        return self._store.remove_member(tenant_id, subject_id)


# ── process-wide wiring ─────────────────────────────────────────────────────

_service: OnboardingService | None = None


def configure(service: OnboardingService | None) -> None:
    global _service
    _service = service


def get_service() -> OnboardingService | None:
    global _service
    if _service is None and enabled():
        store = build_authority_store(settings)
        if store is not None:
            _service = OnboardingService(store)
    return _service


__all__ = [
    "AuthorityUnavailable",
    "FORBIDDEN_DEFAULT_ACTIONS",
    "TENANT_DEFAULT_ACTIONS",
    "HOSTED_V1",
    "LastAdministrator",
    "MAX_OUTSTANDING_INVITES",
    "OnboardingPolicy",
    "OnboardingService",
    "PRESERVE_V1",
    "PRESETS",
    "TooManyInvites",
    "UnknownSubject",
    "configure",
    "default_statements",
    "enabled",
    "get_service",
    "parse_invite",
]
