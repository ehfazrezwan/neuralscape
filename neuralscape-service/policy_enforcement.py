"""Flagged foreground enforcement of the explicit action policy.

One decision function serves both entry points (a FastAPI dependency and an
MCP call hook).  ``NS_POLICY_MODE`` selects the behavior:

* ``off`` (default): a no-op.  Nothing is read, resolved or logged.
* ``observe``: decide and write a structured decision log (no request
  content); never changes the outcome and never raises.
* ``enforce``: deny with an existence-safe public error.  Any failure to
  decide, including an unreadable authority store, an unresolved target or an
  unexpected exception, denies.  There is no availability fallback.

The operation's action and target-resolver spec come from ``policy_registry``;
the tenant comes only from the verified subject record, never from a request
field.  Row-dependent targets (a memory, episode or artifact id) need an
injected ``row_lookup`` that maps the row to its pool; without one they are
unresolved and deny in enforce mode.  Worker publication is enforced by the publication fence, not here.

Boundaries: the dependency runs for every API route (including routers mounted
later); targets are resolved from the same parsed request the handler receives
(see ``_handler_view_of_body``).  Framework documentation routes (``/docs``,
``/redoc``, ``/openapi.json``) carry no dependant, so the dependency never runs
for them; they are inventoried as exempt static metadata and remain behind the
bearer middleware.  The pre-principal OAuth endpoints are exempt here and
authorize issuance in their handlers.  Operations marked ``privileged`` are
checked against ``privileged_handle`` so an ordinary content grant never
authorizes them.
"""

from __future__ import annotations

import email.message
import logging
import re
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from fastapi import HTTPException, Request, params

from authority_store import AuthorityStore, AuthorityUnavailable, SubjectRecord
from config import settings
from contracts_errors import InternalErrorCode, PublicError, public_error_for
from contracts_policy import (
    AccessPolicy,
    ResourceContext,
    PolicyEvaluationInput,
    PrincipalContext,
)
from contracts_policy_reference import evaluate_policy
from contracts_references import ReferenceHandle
from identity_binding import (
    BindingStore,
    NamespaceKey,
    encode_namespace,
    resolve_binding,
)
from policy_registry import REGISTRY, Entry

logger = logging.getLogger("neuralscape.policy")

LEGACY_SERVICE_ISSUER = "local"
LEGACY_SERVICE_SUBJECT = "svc:legacy-shared-key"
LOCAL_ISSUER = "local"
# A body-asserted subject can never be a bound principal for these actions.
LEGACY_FORBIDDEN_ACTIONS = frozenset(
    {"manage_membership", "manage_credentials", "publish_standard"}
)
_HANDLE_KIND = {"memory": "memory", "artifact": "artifact", "graph_item": "episode"}
_ROW_KINDS = frozenset({"memory", "artifact", "graph_item"})
_FIXED_IDS = {"connector": "new", "tenant": "tenant"}
_POOL_KINDS = ("private", "shared", "standard")
# ``body.memories[*].visibility``: a field of each item of a list (per-item requests).
_ITEM_FIELD = re.compile(r"(body|arguments)\.([A-Za-z_]+)\[\*\]\.")
_TOKEN = re.compile(r"^(path|query|body|arguments)\.([A-Za-z_.\[\]*]+)$")


class RowInfo(NamedTuple):
    """Where a row lives and its current visibility, from the source row."""

    pool_kind: str
    owner: str | None
    project: str | None
    workspace: str | None
    visibility: str
    # The subject that authored the row (its owner for conditions such as
    # ``row_owner_is_principal``).  For private rows the pool owner is used when
    # this is absent.  Unknown means conditions on it are not satisfied.
    author: str | None = None
    # The row's current text, so an edit that merely echoes it is not a content edit.
    content: str | None = None


RowLookup = Callable[[str, str], "RowInfo | None"]


class JobOwner(NamedTuple):
    """Who submitted an accepted job: a bound subject, or a legacy user id to bind."""

    subject_id: str | None = None
    user_id: str | None = None


# ``(tenant_id, job_id) -> JobOwner | None``: the principal an accepted job is bound to.
JobOwnerLookup = Callable[[str, str], "JobOwner | None"]


class RequestContext(NamedTuple):
    """Verified credential facts plus the request fields a resolver may read."""

    user_id: str | None  # verified user id from a token, if any
    path_params: Mapping[str, Any] = {}
    query: Mapping[str, Any] = {}
    body: Mapping[str, Any] | None = None
    arguments: Mapping[str, Any] | None = None
    # True when the route takes a body that could not be read the way the
    # handler reads it.  Targets named in the body are then unknown: deny.
    body_unreadable: bool = False
    # A verified scoped credential (set by the auth layer, never by the
    # request): the subject it acts for, its own id, and its action scope.
    subject_id: str | None = None
    credential_id: str | None = None
    credential_scope: tuple[str, ...] | None = None


class Decision(NamedTuple):
    allowed: bool
    mode: str
    reason: str
    operation: str | None = None
    action: str | None = None
    tenant_id: str | None = None
    subject_id: str | None = None
    asserted_subject: str | None = None
    unavailable: bool = False


class _Unresolved(Exception):
    """A required target could not be resolved from trusted inputs."""


def pool_handle(
    tenant_id: str,
    pool_kind: str,
    owner: str | None,
    project: str | None = None,
    workspace: str | None = None,
) -> ReferenceHandle:
    """The pool handle shared by statement authors and the enforcer."""

    key = NamespaceKey(tenant_id, pool_kind, owner, project, workspace)
    return _handle("pool", encode_namespace(key), tenant_id)


def tenant_handle(tenant_id: str) -> ReferenceHandle:
    """The handle for tenant-scoped operations (capabilities, tenant export, relay)."""

    return _handle("tenant", tenant_id, tenant_id)


def privileged_handle(action: str, base: ReferenceHandle) -> ReferenceHandle:
    """The handle a privileged operation is checked against.

    It is distinct from the ordinary pool/settings handle, so a member's
    content grant on their own pool never authorizes a privileged operation;
    only an explicit statement naming this handle does.
    """

    return _handle("privileged", f"{action}|{base.resolver}|{base.id}", base.tenant_id)


def _handle(kind: str, identifier: str, tenant_id: str) -> ReferenceHandle:
    return ReferenceHandle(
        kind=_HANDLE_KIND.get(kind, "source"),
        id=identifier,
        tenant_id=tenant_id,
        resolver=f"ns:{kind}",
    )


class PolicyEnforcer:
    def __init__(
        self,
        authority: AuthorityStore,
        bindings: BindingStore,
        *,
        row_lookup: RowLookup | None = None,
        job_owner: JobOwnerLookup | None = None,
    ) -> None:
        self._authority = authority
        self._bindings = bindings
        self._row_lookup = row_lookup
        self._job_owner = job_owner

    def _submitted_by_principal(
        self, tenant_id: str, job_id: str, principal: PrincipalContext
    ) -> bool:
        """Whether the job was accepted for this very principal (an unreadable lookup is no)."""

        if self._job_owner is None:
            return False
        try:
            owner = self._job_owner(tenant_id, job_id)
            if owner is None:
                return False
            subject = owner.subject_id
            if subject is None and owner.user_id:
                subject = resolve_binding(self._bindings, LOCAL_ISSUER, owner.user_id)
        except Exception:  # noqa: BLE001 - an unreadable owner never grants anything
            return False
        return subject is not None and subject == principal.subject_id

    # ── principal ────────────────────────────────────────────────────────

    def _principal(
        self, context: RequestContext, profile: str
    ) -> tuple[PrincipalContext | None, str, bool]:
        """Return ``(principal, failure_reason, legacy_assertion)``."""

        if context.subject_id is not None:
            # A verified scoped credential names the subject directly; there is
            # no alias to resolve.  The principal acts under THAT credential, so
            # credential-specific statements and retirement apply to it.
            record = self._authority.subject_record(context.subject_id)
            if record is None:
                return None, "not_a_member", False
            if context.credential_id:
                record = record._replace(credential_id=context.credential_id)
            return _build_principal(context.subject_id, record), "", False
        token_user = context.user_id
        if token_user:
            issuer, subject_key, legacy = LOCAL_ISSUER, token_user, False
        elif settings.neuralscape_user_token_secret or settings.neuralscape_api_key:
            if profile != "legacy_compat":
                return None, "legacy_credential_refused", False
            issuer, subject_key, legacy = (
                LEGACY_SERVICE_ISSUER,
                LEGACY_SERVICE_SUBJECT,
                True,
            )
        else:
            if profile != "legacy_compat":
                return None, "unauthenticated_refused", False
            issuer, subject_key, legacy = (
                LOCAL_ISSUER,
                f"auth-disabled:{settings.default_user_id}",
                False,
            )
        subject_id = resolve_binding(self._bindings, issuer, subject_key)
        if subject_id is None:
            return None, "unbound_subject", legacy
        record = self._authority.subject_record(subject_id)
        if record is None:
            return None, "not_a_member", legacy
        return _build_principal(subject_id, record), "", legacy

    def principal_tenant(self, context: RequestContext, profile: str) -> str | None:
        """The tenant of the principal this context resolves to, or ``None``.

        Resolved exactly as ``decide`` resolves the acting principal, so a check
        built on it cannot disagree with the decision about who is calling.
        """

        principal, _reason, _legacy = self._principal(context, profile)
        return None if principal is None else principal.tenant_id

    # ── target resolution ────────────────────────────────────────────────

    def _field(self, token: str, context: RequestContext) -> Any:
        match = _TOKEN.match(token)
        if match is None:
            raise _Unresolved(token)
        source, path = match.groups()
        data: Any = {
            "path": context.path_params,
            "query": context.query,
            "body": context.body,
            "arguments": context.arguments,
        }[source]
        for part in path.split("."):
            if "[" in part:
                raise _Unresolved(token)
            if not isinstance(data, Mapping):
                return None
            data = data.get(part)
        return data

    def _owner_subject(self, user_id: str | None, principal: PrincipalContext) -> str:
        if user_id in (None, ""):
            return principal.subject_id
        bound = resolve_binding(self._bindings, LOCAL_ISSUER, str(user_id))
        if bound is None:
            raise _Unresolved("owner alias")
        return bound

    def _pool_for(
        self,
        entry: Entry,
        context: RequestContext,
        principal: PrincipalContext,
        legacy: bool,
    ) -> tuple[ReferenceHandle, tuple[str | None, str | None]]:
        target = entry.resolver_target
        names = {
            token.split(".")[-1]: token
            for token in re.findall(r"(?:path|query|body|arguments)\.[A-Za-z_.]+", target)
        }
        visibility = None
        if "visibility" in names:
            visibility = self._field(names["visibility"], context)
        if visibility in (None, ""):
            visibility = "private"
        if visibility not in _POOL_KINDS:
            raise _Unresolved("visibility")
        project = self._optional(names.get("project_id"), context)
        workspace = self._optional(names.get("workspace"), context)
        if workspace == "memory":
            workspace = None
        pool_name = self._optional(names.get("pool"), context)
        if pool_name in ("shared", "standard"):
            visibility = pool_name
        elif pool_name not in (None, ""):
            raise _Unresolved("pool")
        owner = None
        if visibility == "private":
            owner_token = names.get("target_user_id") or names.get("user_id")
            owner_alias = self._optional(owner_token, context)
            if owner_alias is None and legacy:
                # The shared key names the pool owner in the body, as it does
                # today.  That is a target selector, never the acting principal.
                owner_alias = _asserted_subject(context)
            owner = self._owner_subject(owner_alias, principal)
        handle = pool_handle(principal.tenant_id, visibility, owner, project, workspace)
        return handle, (project, workspace)

    def _optional(self, token: str | None, context: RequestContext) -> str | None:
        if token is None:
            return None
        value = self._field(token, context)
        if value is None:
            return None
        if type(value) is not str:
            raise _Unresolved(token)
        return value

    def _resolve(
        self,
        entry: Entry,
        context: RequestContext,
        principal: PrincipalContext,
        legacy: bool,
    ) -> tuple[list[ReferenceHandle], RowInfo | None, tuple[str | None, str | None] | None]:
        """Return ``(handles, row, (project, workspace))``; the last is the
        placement used to build the destination pool for ``share`` and
        ``publish_standard`` checks, or ``None`` when not pool-scoped."""

        kind, target = entry.resolver_kind, entry.resolver_target
        if kind == "pool":
            handle, placement = self._pool_for(entry, context, principal, legacy)
            return [handle], None, placement
        if kind in _ROW_KINDS:
            if self._row_lookup is None or target.endswith("|arguments.filter"):
                raise _Unresolved("row lookup")
            token = target.split(" ")[0].split("|")[0]
            if token.endswith("[*]"):
                values = self._list_field(token[:-3], context)
            else:
                values = [self._optional(token, context)]
            handles: list[ReferenceHandle] = []
            first: RowInfo | None = None
            for identifier in values:
                row = self._row_lookup(kind, identifier) if identifier else None
                if row is None:
                    raise _Unresolved("row")
                first = first or row
                handles.append(
                    pool_handle(
                        principal.tenant_id,
                        row.pool_kind,
                        row.owner,
                        row.project,
                        row.workspace,
                    )
                )
            if not handles or first is None:
                raise _Unresolved("row")
            return handles, first, (first.project, first.workspace)
        if kind in {"job", "connector", "settings", "tenant", "credential", "service_account", "membership"}:
            token = target.split(" ")[0].split("+")[0].split("|")[0]
            if token in {"credential.tenant", "tenant"}:
                identifier = principal.tenant_id
            elif token == "credential.subject":
                identifier = principal.subject_id
            elif kind in _FIXED_IDS and not _TOKEN.match(token):
                identifier = _FIXED_IDS[kind]
            else:
                identifier = self._optional(token, context)
            if not identifier:
                raise _Unresolved("target id")
            if kind == "job" and _TOKEN.match(token) and self._submitted_by_principal(
                principal.tenant_id, identifier, principal
            ):
                # The caller's own accepted job is checked against the caller's own private
                # pool, which every member may read; nobody else gains a read through it.
                return [pool_handle(principal.tenant_id, "private", principal.subject_id)], None, None
            return [_handle(kind, identifier, principal.tenant_id)], None, None
        raise _Unresolved(kind)

    def _list_field(self, token: str, context: RequestContext) -> list[str | None]:
        value = self._field(token, context)
        if not isinstance(value, list) or not all(type(v) is str for v in value):
            raise _Unresolved(token)
        return list(value)

    # ── decision ─────────────────────────────────────────────────────────

    def _required_actions(
        self, entry: Entry, context: RequestContext, row: RowInfo | None
    ) -> list[str]:
        actions = [entry.action] if entry.action else []
        asked = None
        for token in ("body.visibility", "arguments.visibility", "query.visibility"):
            asked = asked or self._field_safe(token, context)
        for extra, condition in entry.extra_actions:
            if extra == "background_replay":
                continue
            wanted, _, expected = condition.partition("=")
            if wanted == "visibility":
                required = asked == expected or (asked is None and expected == "shared")
            elif wanted == "visibility_change":
                required = asked == expected
            elif wanted == "target.visibility":
                required = row is None or row.visibility == expected
            elif wanted == "credential" and expected == "scoped":
                # Durable work is fenced under ``background_derive`` when it is published; a scoped
                # credential must hold it when the work is accepted, so a credential that may only
                # create is refused here instead of being accepted and fenced later.
                required = context.credential_scope is not None and settings.ledger_enabled
            else:
                required = True  # an unknown condition (or value) fails closed
            if required:
                actions.append(extra)
        return actions

    @staticmethod
    def _targets(
        action: str,
        entry: Entry,
        handles: list[ReferenceHandle],
        placement: tuple[str | None, str | None] | None,
        principal: PrincipalContext,
    ) -> list[ReferenceHandle]:
        """``share`` and ``publish_standard`` are checked on the destination
        pool; every other action on the operation's own target(s)."""

        if action != entry.action and placement is not None:
            kind = {"share": "shared", "publish_standard": "standard"}.get(action)
            if kind is not None:
                return [pool_handle(principal.tenant_id, kind, None, *placement)]
        return handles

    def _field_safe(self, token: str, context: RequestContext) -> Any:
        try:
            value = self._field(token, context)
        except _Unresolved:
            return None
        return value if type(value) is str else None

    def _decide_items(
        self,
        operation_id: str,
        entry: Entry,
        context: RequestContext,
        *,
        mode: str,
        profile: str,
    ) -> Decision:
        split = _item_contexts(entry, context)
        if split is None:
            return Decision(
                False, mode, "resolver_unavailable", operation_id,
                action=entry.action, asserted_subject=_asserted_subject(context),
            )
        single, contexts = split
        decision: Decision | None = None
        for item in contexts:
            decision = self._decide_entry(operation_id, single, item, mode=mode, profile=profile)
            if not decision.allowed:
                return decision
        assert decision is not None
        return decision

    def decide(
        self,
        operation_id: str,
        context: RequestContext,
        *,
        mode: str,
        profile: str,
    ) -> Decision:
        asserted = _asserted_subject(context)

        def done(allowed: bool, reason: str, **fields: Any) -> Decision:
            return Decision(
                allowed, mode, reason, operation_id, asserted_subject=asserted, **fields
            )

        entry = REGISTRY.get(operation_id)
        if entry is None:
            return done(False, "unregistered_operation")
        if entry.status == "exempt":
            return done(True, "exempt")
        try:
            if _ITEM_FIELD.search(entry.resolver_target):
                return self._decide_items(operation_id, entry, context, mode=mode, profile=profile)
            # A bare ``ids[*]`` is a list of row ids, resolved row by row below.
            return self._decide_entry(operation_id, entry, context, mode=mode, profile=profile)
        except AuthorityUnavailable:
            return done(False, "authority_unavailable", action=entry.action, unavailable=True)
        except Exception:  # noqa: BLE001 - nothing may escape the decision; a failure denies
            logger.exception("policy decision failed")
            return done(False, "decision_error", action=entry.action)

    def _decide_entry(
        self,
        operation_id: str,
        entry: Entry,
        context: RequestContext,
        *,
        mode: str,
        profile: str,
    ) -> Decision:
        asserted = _asserted_subject(context)

        def done(allowed: bool, reason: str, **fields: Any) -> Decision:
            return Decision(
                allowed, mode, reason, operation_id, asserted_subject=asserted, **fields
            )

        try:
            principal, failure, legacy = self._principal(context, profile)
            if principal is None:
                return done(False, failure, action=entry.action)
            if context.body_unreadable and "body." in entry.resolver_target:
                return done(
                    False,
                    "request_body_unreadable",
                    action=entry.action,
                    tenant_id=principal.tenant_id,
                    subject_id=principal.subject_id,
                )
            try:
                handles, row, placement = self._resolve(entry, context, principal, legacy)
            except _Unresolved:
                return done(
                    False,
                    "resolver_unavailable",
                    action=entry.action,
                    tenant_id=principal.tenant_id,
                    subject_id=principal.subject_id,
                )
            if entry.privileged and entry.action:
                handles = [privileged_handle(entry.action, h) for h in handles]
                placement = None
            if legacy and row is not None and row.pool_kind == "private":
                # The shared key acts as the user it asserts: it can address a
                # private row by id only when that user owns it (as before).
                acting_as = (
                    resolve_binding(self._bindings, LOCAL_ISSUER, asserted) if asserted else None
                )
                if acting_as is None or row.owner != acting_as:
                    return done(
                        False,
                        "legacy_assertion_mismatch",
                        action=entry.action,
                        tenant_id=principal.tenant_id,
                        subject_id=principal.subject_id,
                    )
            actions = self._required_actions(entry, context, row)
            if context.credential_scope is not None and not set(actions) <= set(
                context.credential_scope
            ):
                # A scope only narrows: any action outside it is refused, whatever
                # the subject itself may do.
                return done(
                    False,
                    "outside_credential_scope",
                    action=entry.action,
                    tenant_id=principal.tenant_id,
                    subject_id=principal.subject_id,
                )
            if legacy and LEGACY_FORBIDDEN_ACTIONS & set(actions):
                return done(
                    False,
                    "legacy_assertion_forbidden",
                    action=entry.action,
                    tenant_id=principal.tenant_id,
                    subject_id=principal.subject_id,
                )
            policy = self._authority.current_policy(principal.tenant_id)
            if policy is None:
                return done(
                    False,
                    "no_policy",
                    action=entry.action,
                    tenant_id=principal.tenant_id,
                    subject_id=principal.subject_id,
                )
            resource_context = _resource_context(entry, row, context)
            if legacy and resource_context is not None and asserted is not None:
                # The shared key acts for the user it asserts (as it always did):
                # the row-owner condition compares the row owner with THAT user.
                acting_as = resolve_binding(self._bindings, LOCAL_ISSUER, asserted)
                if acting_as is not None and acting_as == resource_context.row_owner_subject_id:
                    resource_context = ResourceContext(
                        row_owner_subject_id=principal.subject_id,
                        edited_fields=resource_context.edited_fields,
                    )
            for action in actions:
                for handle in self._targets(action, entry, handles, placement, principal):
                    result = evaluate_policy(
                        principal=principal,
                        evaluation=PolicyEvaluationInput(
                            schema_version="candidate-v1", action=action, resource=handle,
                            context=resource_context,
                        ),
                        policy=policy,
                    )
                    if result.outcome != "allow":
                        return done(
                            False,
                            result.reason_code,
                            action=action,
                            tenant_id=principal.tenant_id,
                            subject_id=principal.subject_id,
                        )
            return done(
                True,
                "explicit_grant",
                action=entry.action,
                tenant_id=principal.tenant_id,
                subject_id=principal.subject_id,
            )
        except AuthorityUnavailable:
            return done(False, "authority_unavailable", action=entry.action, unavailable=True)
        except Exception:  # noqa: BLE001 - any failure to decide denies
            logger.exception("policy decision failed")
            return done(False, "decision_error", action=entry.action)


_EDIT_IDENTITY_KEYS = frozenset({"user_id", "memory_id"})


def _patch_fields() -> frozenset[str]:
    from schemas import PatchMemoryRequest

    return frozenset(PatchMemoryRequest.model_fields) - {"user_id"}


def _resource_context(
    entry: Entry, row: RowInfo | None, context: RequestContext
) -> ResourceContext | None:
    """Current row facts for condition evaluation (row operations only).

    The owner comes from the CURRENT row the resolver just read, never from the
    request.  The edited fields are the fields the handler would actually apply:
    keys it ignores (unknown keys, ``scope``, ``owner_user_id``) are not edits,
    and ``content`` that merely echoes the row's current text is not a content
    edit.  An absent or unknown value can only narrow.
    """

    if row is None or entry.resolver_kind not in _ROW_KINDS:
        return None
    owner = row.author or (row.owner if row.pool_kind == "private" else None)
    edited = None
    if entry.action == "update":
        source = context.body if context.body is not None else context.arguments
        if isinstance(source, Mapping):
            known = _patch_fields()
            names = {str(k) for k in source if str(k) in known}
            if "content" in names and row.content is not None and source.get("content") == row.content:
                names.discard("content")
            edited = tuple(sorted(names))
    return ResourceContext(row_owner_subject_id=owner, edited_fields=edited)


def _item_contexts(
    entry: Entry, context: RequestContext
) -> tuple[Entry, list[RequestContext]] | None:
    """Split a batch request into per-item contexts, or ``None`` if malformed.

    A target such as ``body.memories[*].visibility`` names a field of each item
    of a list.  Each item is checked as its own single request (its own
    visibility, project and workspace), and every item must be allowed.  An
    absent or empty list is one default item.
    """

    match = _ITEM_FIELD.search(entry.resolver_target)
    if match is None:
        return None
    source, name = match.groups()
    container = context.body if source == "body" else context.arguments
    items = container.get(name) if isinstance(container, Mapping) else None
    if items in (None, []):
        items = [{}]
    if not isinstance(items, list) or not all(isinstance(i, Mapping) for i in items):
        return None
    target = entry.resolver_target.replace(f"{source}.{name}[*].", f"{source}.")
    single = entry._replace(resolver_target=target)
    contexts = [
        context._replace(body=item) if source == "body" else context._replace(arguments=item)
        for item in items
    ]
    return single, contexts


def _asserted_subject(context: RequestContext) -> str | None:
    for source in (context.body, context.arguments):
        if isinstance(source, Mapping) and type(source.get("user_id")) is str:
            return source["user_id"]
    return None


def _build_principal(subject_id: str, record: SubjectRecord) -> PrincipalContext:
    return PrincipalContext(
        schema_version="candidate-v1",
        tenant_id=record.tenant_id,
        subject_id=subject_id,
        subject_kind=record.subject_kind,
        credential_id=record.credential_id,
        membership_version=record.membership_version,
        device_id=None,
        delegation=record.delegation,
    )


# ── row lookups ─────────────────────────────────────────────────────────────

PayloadFetcher = Callable[[str], "Mapping[str, Any] | None"]


def memory_row_lookup(fetch_payload: PayloadFetcher, bindings: BindingStore) -> RowLookup:
    """A ``row_lookup`` for stored memories, from the CURRENT vector payload.

    Owner normalization follows the legacy rule: ``owner_user_id`` (possibly
    inside mem0's nested metadata), else ``user_id``.  A missing or unknown
    visibility is private.  The owner's legacy id is resolved through the
    ``local`` binding to a subject; an owner that is not a bound (unique)
    subject resolves to nothing, so the row is unreadable rather than guessed.
    Only the ``memory`` kind is resolved here.
    """

    def lookup(kind: str, identifier: str) -> RowInfo | None:
        if kind != "memory" or not identifier:
            return None
        payload = fetch_payload(identifier)
        if not isinstance(payload, Mapping):
            return None
        meta = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        if isinstance(meta.get("metadata"), Mapping):
            meta = meta["metadata"]

        def pick(key: str) -> Any:
            return meta.get(key) if meta.get(key) is not None else payload.get(key)

        legacy_owner = pick("owner_user_id") or payload.get("user_id")
        if type(legacy_owner) is not str or not legacy_owner:
            return None
        subject = resolve_binding(bindings, LOCAL_ISSUER, legacy_owner)
        if subject is None:
            return None
        visibility = pick("visibility") or "private"
        visibility = str(getattr(visibility, "value", visibility))
        if visibility not in _POOL_KINDS:
            visibility = "private"
        workspace = pick("workspace")
        content = payload.get("memory", payload.get("data"))
        return RowInfo(
            visibility,
            subject if visibility == "private" else None,
            pick("project_id") or None,
            None if workspace in (None, "", "memory") else str(workspace),
            visibility,
            author=subject,
            content=content if type(content) is str else None,
        )

    return lookup


def compose_row_lookups(*lookups: RowLookup) -> RowLookup:
    """The first lookup that resolves the row wins; none resolving is ``None``."""

    def lookup(kind: str, identifier: str) -> RowInfo | None:
        for candidate in lookups:
            row = candidate(kind, identifier)
            if row is not None:
                return row
        return None

    return lookup


# ── process-wide wiring ─────────────────────────────────────────────────────

_enforcer: PolicyEnforcer | None = None


def configure(enforcer: PolicyEnforcer | None) -> None:
    """Install (or clear) the enforcer used by the request and MCP hooks."""

    global _enforcer
    _enforcer = enforcer


def _mem0_payload_fetcher(service: Any) -> PayloadFetcher:
    """Read one memory's CURRENT stored payload from the running service."""

    def fetch(memory_id: str) -> Mapping[str, Any] | None:
        try:
            found = service._get_memory().get(memory_id)
        except Exception:  # noqa: BLE001 - an unreadable row is an unresolved one (deny)
            return None
        return found if isinstance(found, Mapping) else None

    return fetch


def configure_from_settings(
    service: Any = None, job_owner: JobOwnerLookup | None = None
) -> PolicyEnforcer | None:
    """Build and install the enforcer from settings; a no-op unless it is wanted.

    Off by default: nothing happens unless the policy mode is not ``off`` AND the
    authority store is enabled.  The store is built once and shared by the
    enforcer, the binding lookups and (when enabled) the credential service.  Row
    operations resolve through ``memory_row_lookup`` (current payload from
    ``service``) composed with ``artifact_row_lookup``.  ``job_owner`` binds an
    accepted job to the principal that submitted it, so the writer can read its own
    job status.  If the store cannot be built the enforcer is left unset, which
    denies every non-exempt operation; ``require_enforcer`` refuses to start then.
    """

    if current_mode() == "off" or not getattr(settings, "authority_enabled", False):
        return None
    from authority_store_sql import build_authority_store
    from policy_migration import artifact_row_lookup

    store = build_authority_store(settings)
    if store is None:
        return None
    lookups = [artifact_row_lookup(settings, store)]
    if service is not None:
        lookups.insert(0, memory_row_lookup(_mem0_payload_fetcher(service), store))
    enforcer = PolicyEnforcer(
        store, store, row_lookup=compose_row_lookups(*lookups), job_owner=job_owner
    )
    configure(enforcer)
    import credentials

    if credentials.enabled():
        credentials.configure(credentials.CredentialService(store))
    return enforcer


class EnforcementUnavailable(RuntimeError):
    """Enforce mode is selected but no enforcer could be installed."""


def install_or_refuse(service: Any = None, job_owner: JobOwnerLookup | None = None) -> PolicyEnforcer | None:
    """Install the enforcer at start-up; in enforce mode, refuse to start without one.

    ``observe`` keeps the previous behavior (a failure is logged, requests are not
    affected).  ``enforce`` with the authority store disabled, misconfigured or
    unreachable raises ``EnforcementUnavailable`` so the process does not serve
    traffic that would all be denied (or, worse, believed to be enforced).
    """

    try:
        enforcer = configure_from_settings(service, job_owner)
    except Exception as error:
        if current_mode() == "enforce":
            raise EnforcementUnavailable(
                "NS_POLICY_MODE=enforce but the authority store could not be built "
                f"({type(error).__name__}); refusing to start"
            ) from error
        logger.exception("policy enforcement setup failed; protected operations stay denied")
        return None
    if current_mode() == "enforce" and enforcer is None:
        raise EnforcementUnavailable(
            "NS_POLICY_MODE=enforce requires AUTHORITY_ENABLED=true and a configured "
            "authority store; refusing to start"
        )
    return enforcer


def current_mode() -> str:
    return settings.ns_policy_mode


def request_identity(request: Request) -> RequestContext:
    """The verified credential facts of an HTTP request, as the auth layer set them.

    The single place that turns a request's authenticated state into the identity
    part of a ``RequestContext``.  The enforcement dependency and every
    identity-based check derive the caller from it, so a new kind of credential
    added here is seen by all of them at once.
    """

    return RequestContext(
        user_id=getattr(request.state, "user_id", None),
        **_credential_fields(getattr(request.state, "credential", None)),
    )


def caller_tenant(request: Request) -> str | None:
    """The verified tenant of the request's principal, or ``None`` if it cannot be established.

    For routes that serve one tenant cell and must deny every other tenant's principal
    regardless of which tenant-level grants that principal holds.  The principal is
    resolved from ``request_identity`` exactly as the enforcement decision resolves
    it.  Any failure to establish the tenant (no enforcer, unbound subject,
    unreadable authority store) returns ``None``, which callers must treat as a
    denial.
    """

    enforcer = _enforcer
    if enforcer is None:
        return None
    try:
        return enforcer.principal_tenant(
            request_identity(request), settings.ns_auth_profile
        )
    except Exception:  # noqa: BLE001 - an unreadable authority store denies
        return None


def _log(decision: Decision) -> None:
    logger.info(
        "policy_decision",
        extra={
            "policy_mode": decision.mode,
            "policy_operation": decision.operation,
            "policy_action": decision.action,
            "policy_allowed": decision.allowed,
            "policy_reason": decision.reason,
            "policy_tenant": decision.tenant_id,
            "policy_subject": decision.subject_id,
            "policy_asserted_subject": decision.asserted_subject,
            "policy_assertion_provenance": (
                "legacy_assertion" if decision.asserted_subject else None
            ),
        },
    )


def evaluate_request(operation_id: str, context: RequestContext) -> PublicError | None:
    """Decide one operation.  Returns a public error to deny, else ``None``."""

    mode = current_mode()
    if mode == "off":
        return None
    profile = settings.ns_auth_profile
    entry = REGISTRY.get(operation_id)
    if _enforcer is None:
        if entry is not None and entry.status == "exempt":
            decision = Decision(True, mode, "exempt", operation_id)
        else:
            decision = Decision(
                False, mode, "authority_unavailable", operation_id, unavailable=True
            )
    else:
        try:
            decision = _enforcer.decide(operation_id, context, mode=mode, profile=profile)
        except Exception:  # noqa: BLE001 - observe mode must never raise; enforce denies
            logger.exception("policy decision failed")
            decision = Decision(False, mode, "decision_error", operation_id)
    _log(decision)
    if decision.allowed or mode != "enforce":
        return None
    code = (
        InternalErrorCode.UNAVAILABLE_DEPENDENCY
        if decision.unavailable
        else InternalErrorCode.DENIED
    )
    return public_error_for(code)


def operation_id_for_route(method: str, path: str) -> str | None:
    for surface in ("rest", "rest_extension", "rest_oauth", "rest_framework"):
        candidate = f"{surface}:{method.upper()} {path}"
        if candidate in REGISTRY:
            return candidate
    return None


async def _handler_view_of_body(request: Request, route: Any) -> tuple[Mapping[str, Any] | None, bool]:
    """Read the body exactly as FastAPI will hand it to the handler.

    The policy layer must resolve targets from the same parsed request the
    handler receives, for every content type FastAPI accepts for the route, or
    deny.  This mirrors ``fastapi.routing.get_request_handler``: a route
    without body parameters ignores the body; a form route reads the form
    (multipart or urlencoded); any other route parses JSON when there is no
    Content-Type or when it is ``application/json`` or ``application/*+json``
    (media type compared case-insensitively), and otherwise hands the handler
    raw bytes, which fail its validation.  The parsed values are cached on the
    request, so the handler sees what was read here.  Returns
    ``(body, unreadable)``; ``unreadable`` means a body that should have been
    read could not be, so body-named targets are unknown.
    """

    body_field = getattr(route, "body_field", None)
    if body_field is None:
        return None, False
    try:
        if isinstance(body_field.field_info, params.Form):
            form = await request.form()
            return {
                key: value for key, value in form.multi_items() if isinstance(value, str)
            }, False
        raw = await request.body()
        if not raw:
            return None, False
        content_type = request.headers.get("content-type")
        is_json = not content_type
        if content_type:
            message = email.message.Message()
            message["content-type"] = content_type
            if message.get_content_maintype() == "application":
                subtype = message.get_content_subtype()
                is_json = subtype == "json" or subtype.endswith("+json")
        if not is_json:
            return None, False  # raw bytes: the handler's validation rejects them
        parsed = await request.json()
        return (parsed if isinstance(parsed, dict) else None), False
    except Exception:  # noqa: BLE001 - an unreadable body is not an allowed one
        return None, True


async def enforce_request(request: Request) -> None:
    """FastAPI dependency installed on the app.  A no-op when the mode is off."""

    if current_mode() == "off":
        return
    route = request.scope.get("route")
    operation_id = operation_id_for_route(
        request.method, getattr(route, "path", "") or ""
    )
    if operation_id is None:
        # Only API routes carry this dependency.  Framework routes (docs,
        # redoc, openapi.json) have no dependant, so it never runs for them;
        # they are inventoried as exempt static metadata.  An API route missing
        # from the registry fails closed.
        if hasattr(route, "dependant") and current_mode() == "enforce":
            raise HTTPException(
                status_code=404,
                detail=_public_detail(public_error_for(InternalErrorCode.DENIED)),
            )
        return
    body, unreadable = await _handler_view_of_body(request, route)
    context = request_identity(request)._replace(
        path_params=dict(request.path_params),
        query=dict(request.query_params),
        body=body,
        body_unreadable=unreadable,
    )
    error = evaluate_request(operation_id, context)
    if error is not None:
        raise HTTPException(status_code=_status_for(error), detail=_public_detail(error))


def enforce_mcp_call(name: str, arguments: Mapping[str, Any] | None) -> dict | None:
    """MCP hook.  Returns a public error payload to deny, else ``None``."""

    if current_mode() == "off":
        return None
    from auth import current_user_id
    from credentials import current_credential

    context = RequestContext(
        user_id=current_user_id.get(),
        arguments=arguments or {},
        **_credential_fields(current_credential.get()),
    )
    error = evaluate_request(f"mcp_tool:{name}", context)
    return None if error is None else _public_detail(error)


def _credential_fields(verified: Any) -> dict[str, Any]:
    """Context fields from a verified scoped credential (empty when none)."""

    if verified is None:
        return {}
    return {
        "subject_id": verified.subject_id,
        "credential_id": verified.credential_id,
        "credential_scope": tuple(verified.scope),
    }


def _status_for(error: PublicError) -> int:
    return {"unavailable_dependency": 503, "permission_denied": 403}.get(
        error.code.value, 404
    )


def _public_detail(error: PublicError) -> dict:
    return {"error": error.model_dump(mode="json")}


__all__ = [
    "Decision",
    "PolicyEnforcer",
    "RequestContext",
    "RowInfo",
    "caller_tenant",
    "compose_row_lookups",
    "configure",
    "EnforcementUnavailable",
    "JobOwner",
    "configure_from_settings",
    "install_or_refuse",
    "enforce_mcp_call",
    "enforce_request",
    "evaluate_request",
    "memory_row_lookup",
    "operation_id_for_route",
    "pool_handle",
    "request_identity",
    "tenant_handle",
]
