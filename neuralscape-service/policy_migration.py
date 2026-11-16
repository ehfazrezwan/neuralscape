"""Legacy identity/namespace inventory, mapping report, and policy backfill.

Library only in this slice: it is run by an operator, not by the service; no route or worker calls it.

Read-only inventory of what the running service stores today (Redis identity
aliases, owner ids and pools in the vector payloads, group ids in the graph),
a classification with ``classify_legacy_identities`` (unique, ambiguous,
unbound; ambiguous and unbound are quarantined, never merged or guessed), a
dry-run diff, and an idempotent apply into the authority store with rollback.

What the generated policy reproduces (sharing=preserve, admin=separate):

* every unique user keeps all content actions on their own private pools;
* every unique user keeps read/create/share on the shared pools, and read of
  the standard pools when the tier is enabled;
* ``is_dictator`` users receive EXPLICIT ``publish_standard`` plus
  create/update/delete on the standard pools, and nothing else; no migrated
  grant ever includes ``manage_membership``/``manage_credentials``;
* the legacy shared key becomes one named service principal holding the same
  content grants, never the standard or administrator actions: it acts for
  the user it names, but it is not that user for those actions.

Resource conditions reproduce the per-row rules exactly: the author
of a shared row may edit or delete it (``row_owner_is_principal``) and any
member may edit organizational metadata (``fields_subset_of``), both judged
against the CURRENT row.  A dictator's right to edit or delete other users'
private and shared rows is kept as explicit content grants tagged
``origin=legacy_dictator`` (separate from ``publish_standard``, individually
revocable); a new deployment passes ``legacy_dictator_grants=False`` and never
receives them.

What it deliberately does NOT reproduce, all reported: quarantined users and
pools (identity safety; they get no grants), and a dictator's editing of
existing standard rows while the standards tier is switched off (a feature
switch).  Grants cover the pools observed at migration time; a project pool
created later needs a grant (or a wildcard extension).

Nothing here changes runtime authentication or behavior.  ``python -m
policy_migration`` is the CLI entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from authority_store_sql import PolicyConflict, SqlAuthorityStore
from contracts_policy import AccessPolicy, PolicyCondition, PolicyStatement
from identity_binding import (
    LegacyClassification,
    LegacyObservation,
    bind_identity,
    classify_legacy_identities,
    resolve_binding,
)
from policy_enforcement import RowInfo, pool_handle, privileged_handle, tenant_handle

LOCAL_ISSUER = "local"
LEGACY_SUBJECT_ISSUER = "legacy-subject"
LEGACY_EMAIL_ISSUER = "legacy-email"
LEGACY_KEY_SUBJECT = "svc:legacy-shared-key"
PRIVATE_ACTIONS = ("read", "search", "list", "count", "create", "update", "delete", "export")
SHARED_ACTIONS = ("read", "search", "list", "count", "create", "share")
STANDARD_READ_ACTIONS = ("read", "search", "list", "count")
DICTATOR_STANDARD_ACTIONS = ("create", "update", "delete", "publish_standard")
# Fields a shared row's non-owners may edit (legacy: organizational metadata).
METADATA_FIELDS = (
    "category", "concepts", "confidence", "domain", "expires_at",
    "observation_type", "project_id", "tags",
)
LEGACY_DICTATOR_ORIGIN = "legacy_dictator"
OWNER_CONDITION = PolicyCondition(version="condition-v1", kind="row_owner_is_principal")
METADATA_CONDITION = PolicyCondition(
    version="condition-v1", kind="fields_subset_of", values=METADATA_FIELDS
)
PROJECT_DELIM = "--project--"
WORKSPACE_DELIM = "--ws--"

# ── inventory ────────────────────────────────────────────────────────────


class PoolObservation(NamedTuple):
    owner: str | None
    visibility: str
    project: str | None
    workspace: str | None


@dataclass
class LegacyInventory:
    by_sub: dict[str, str] = field(default_factory=dict)
    by_email: dict[str, str] = field(default_factory=dict)
    user_ids: set[str] = field(default_factory=set)
    pools: set[PoolObservation] = field(default_factory=set)
    group_ids: set[str] = field(default_factory=set)


class RedisAliasSource:
    """Read-only: the identity alias hashes (``identity_store``)."""

    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(url, decode_responses=True)

    def read(self) -> tuple[dict[str, str], dict[str, str]]:
        from identity_store import EMAIL_HASH, SUB_HASH

        return dict(self._client.hgetall(SUB_HASH)), dict(self._client.hgetall(EMAIL_HASH))


class QdrantPoolSource:
    """Read-only scroll over vector payloads (owner, visibility, placement)."""

    def __init__(self, url: str, collection: str, *, batch: int = 256) -> None:
        from qdrant_client import QdrantClient

        self._client = QdrantClient(url=url)
        self._collection = collection
        self._batch = batch

    def read(self) -> Iterable[PoolObservation]:
        offset = None
        while True:
            points, offset = self._client.scroll(
                self._collection, limit=self._batch, offset=offset,
                with_payload=True, with_vectors=False,
            )
            for point in points:
                yield pool_from_payload(point.payload or {})
            if offset is None:
                return


class Neo4jGroupSource:
    """Read-only: distinct graph ``group_id`` values."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def read(self) -> set[str]:
        with self._driver.session(default_access_mode="READ") as session:
            rows = session.run(
                "MATCH (n) WHERE n.group_id IS NOT NULL RETURN DISTINCT n.group_id AS g"
            )
            return {row["g"] for row in rows if isinstance(row["g"], str)}

    def close(self) -> None:
        self._driver.close()


def pool_from_payload(payload: Mapping[str, Any]) -> PoolObservation:
    """Placement of one stored memory, tolerating mem0's nested metadata."""

    nested = payload.get("metadata")
    meta = nested if isinstance(nested, Mapping) else {}
    if isinstance(meta.get("metadata"), Mapping):
        meta = meta["metadata"]

    def pick(key: str) -> Any:
        return meta.get(key) if meta.get(key) is not None else payload.get(key)

    visibility = pick("visibility") or "private"
    owner = pick("owner_user_id") or payload.get("user_id")
    workspace = pick("workspace")
    return PoolObservation(
        owner if isinstance(owner, str) and owner else None,
        str(visibility),
        pick("project_id") or None,
        None if workspace in (None, "", "memory") else str(workspace),
    )


def collect_inventory(
    aliases: tuple[Mapping[str, str], Mapping[str, str]] | None = None,
    pools: Iterable[PoolObservation] = (),
    group_ids: Iterable[str] = (),
    extra_user_ids: Iterable[str] = (),
) -> LegacyInventory:
    inventory = LegacyInventory()
    if aliases is not None:
        inventory.by_sub, inventory.by_email = dict(aliases[0]), dict(aliases[1])
    inventory.user_ids |= set(inventory.by_sub.values()) | set(inventory.by_email.values())
    inventory.user_ids |= {u for u in extra_user_ids if u}
    for pool in pools:
        inventory.pools.add(pool)
        if pool.owner:
            inventory.user_ids.add(pool.owner)
    for group in group_ids:
        inventory.group_ids.add(group)
    return inventory


# ── legacy group names ───────────────────────────────────────────────────


class GroupParse(NamedTuple):
    name: str
    kind: str | None  # private | shared | standard
    owner: str | None
    project: str | None
    workspace: str | None
    problem: str | None  # None when the name decodes to exactly one placement


def parse_group_name(name: str, known_user_ids: set[str]) -> GroupParse:
    """Decode ``_build_group_id`` output, refusing ambiguous names.

    The legacy encoding joins free-form ids with ``--``, so a name can decode
    to more than one (owner, project, workspace) tuple.  Candidates are
    filtered by the users that actually exist; more than one survivor is
    ambiguous, none is an unknown owner.  Neither is ever guessed.
    """

    try:  # the injective encoding is owned by PR 501 (GRAPH_GROUP_ID_INJECTIVE_ENABLED)
        from memory.groups import ENCODED_GROUP_PREFIX, decode_group_id
    except ImportError:
        ENCODED_GROUP_PREFIX, decode_group_id = "nsg1--", (lambda _name: None)

    if name.startswith(ENCODED_GROUP_PREFIX):
        # A collision-free name decodes to exactly one placement, or to none (and without the
        # encoder on this tree it is never guessed: an unrecognized group is quarantined).
        key = decode_group_id(name)
        if key is None:
            return GroupParse(name, None, None, None, None, "unrecognized_group")
        return GroupParse(name, key.pool, key.owner, key.project, key.workspace, None)
    base, workspace, problem = name, None, None
    if base.count(WORKSPACE_DELIM) > 1:
        return GroupParse(name, None, None, None, None, "ambiguous_workspace")
    if WORKSPACE_DELIM in base:
        base, workspace = base.split(WORKSPACE_DELIM)
    for kind in ("shared", "standard"):
        if base == kind:
            return GroupParse(name, kind, None, None, workspace, None)
        if base.startswith(f"{kind}{PROJECT_DELIM}"):
            project = base[len(kind) + len(PROJECT_DELIM):]
            if PROJECT_DELIM in project:
                return GroupParse(name, kind, None, None, workspace, "ambiguous_project")
            return GroupParse(name, kind, None, project, workspace, None)
    if not base.startswith("user--"):
        return GroupParse(name, None, None, None, workspace, "unrecognized_group")
    rest = base[len("user--"):]
    candidates = [(rest, None)]
    start = 0
    while (i := rest.find(PROJECT_DELIM, start)) != -1:
        candidates.append((rest[:i], rest[i + len(PROJECT_DELIM):]))
        start = i + 1
    live = [c for c in candidates if c[0] in known_user_ids] if known_user_ids else candidates
    if len(live) > 1:
        return GroupParse(name, "private", None, None, workspace, "ambiguous_owner_project")
    if not live:
        return GroupParse(name, "private", None, None, workspace, "unknown_owner")
    owner, project = live[0]
    return GroupParse(name, "private", owner, project, workspace, problem)


# ── classification report ────────────────────────────────────────────────


@dataclass
class MigrationReport:
    classifications: dict[str, LegacyClassification]
    groups: dict[str, GroupParse]

    @property
    def unique(self) -> list[str]:
        return sorted(u for u, c in self.classifications.items() if c.status == "unique")

    @property
    def quarantined(self) -> list[str]:
        return sorted(u for u, c in self.classifications.items() if c.status != "unique")

    @property
    def quarantined_groups(self) -> list[str]:
        return sorted(n for n, g in self.groups.items() if g.problem)

    def counts(self) -> dict[str, int]:
        result = {"unique": 0, "ambiguous": 0, "unbound": 0}
        for classification in self.classifications.values():
            result[classification.status] += 1
        result["groups_total"] = len(self.groups)
        result["groups_quarantined"] = len(self.quarantined_groups)
        return result

    def documented_changes(
        self, inventory: LegacyInventory, *, standards_enabled: bool, dictators: Iterable[str] = (),
        legacy_key_export: bool = False,
    ) -> list[dict[str, Any]]:
        """Behavior that is deliberately NOT reproduced, for the migration report."""

        changes: list[dict[str, Any]] = []
        if self.quarantined or self.quarantined_groups:
            changes.append({"class": "quarantine", "users": len(self.quarantined),
                            "groups": len(self.quarantined_groups),
                            "effect": "ambiguous identities and group names get no grants"})
        standard_rows = any(p.visibility == "standard" for p in inventory.pools)
        if not standards_enabled and standard_rows and set(dictators):
            changes.append({"class": "standard_rows_with_the_tier_disabled",
                            "effect": "dictators can no longer edit existing standard rows while the tier is off"})
        changes.append({"class": "pools_created_after_migration",
                        "effect": "grants cover the pools observed now; a project or workspace pool created "
                                  "later (including a dictator's reach over it) needs a grant"})
        changes.append({"class": "legacy_key_dictator_and_standard_assertions",
                        "effect": "the shared key never publishes standards or acts with a dictator's "
                                  "cross-user powers, whatever user it asserts"})
        if not legacy_key_export:
            changes.append({"class": "legacy_key_export",
                            "effect": "the shared key is not granted bulk export of other users' private data"})
        changes.append({"class": "empty_or_ignored_edits",
                        "effect": "an edit that changes nothing the service applies is denied by policy "
                                  "(the service rejected an empty edit itself)"})
        return changes

    def to_json(self, *, redact: bool = True) -> dict[str, Any]:
        def label(value: str) -> str:
            return "u_" + hashlib.sha256(value.encode()).hexdigest()[:10] if redact else value

        return {
            "counts": self.counts(),
            "quarantined_users": [
                {"id": label(u), "status": self.classifications[u].status,
                 "reasons": list(self.classifications[u].reasons)}
                for u in self.quarantined
            ],
            "quarantined_groups": [
                {"group": label(n), "problem": self.groups[n].problem} for n in self.quarantined_groups
            ],
        }


def build_report(inventory: LegacyInventory, extra_user_ids: Iterable[str] = ()) -> MigrationReport:
    """Classify every legacy user id observed in any source."""

    ids = set(inventory.user_ids) | {u for u in extra_user_ids if u}
    observations: list[LegacyObservation] = []
    with_alias: set[str] = set()
    for sub, user in sorted(inventory.by_sub.items()):
        observations.append(LegacyObservation(user, LEGACY_SUBJECT_ISSUER, sub, None))
        with_alias.add(user)
    for email, user in sorted(inventory.by_email.items()):
        # An email link is an attribute of the person, not a second identity; it
        # only feeds the slug-collision check.
        observations.append(LegacyObservation(user, None, None, email))
    # A user with no provider-subject evidence is a token user: the identity is the id the
    # token carries (a ``local`` binding).  It is clean unless it collides.
    for user in sorted(ids - with_alias):
        observations.append(LegacyObservation(user, LOCAL_ISSUER, user, None))
    classifications = classify_legacy_identities(observations)
    groups = {
        name: parse_group_name(name, ids) for name in sorted(inventory.group_ids)
    }
    return MigrationReport(classifications, groups)


# ── policy generation ────────────────────────────────────────────────────


def _norm_workspace(workspace: str | None) -> str | None:
    return None if workspace in (None, "", "memory") else workspace


def statement_id(subject: str, action: str, handle_id: str, tag: str = "") -> str:
    digest = hashlib.sha256(f"{subject}|{action}|{handle_id}|{tag}".encode()).hexdigest()
    return "mig-" + digest[:28]


ADMIN_ORIGIN = "legacy_admin"
#: What the administrator gate guards today (dreaming runs, project knowledge-config writes).
ADMIN_ACTIONS_ON_TENANT = ("update", "background_derive")


def admin_user_ids(dictators: Iterable[str] = (), default_user: str | None = None) -> frozenset[str]:
    """Who is an administrator today (owner decision: with no listed dictators the default user is).

    Mirrors ``auth.is_admin_user``: a non-empty ``DICTATOR_USER_IDS`` names exactly the administrators;
    an empty one makes the default user the administrator, so single-user and legacy-key installs keep
    working once enforcement is on.
    """

    listed = {d for d in dictators if d}
    if listed:
        return frozenset(listed)
    if default_user is None:
        from config import settings

        default_user = settings.default_user_id
    return frozenset({default_user}) if default_user else frozenset()


def build_statements(
    report: MigrationReport,
    inventory: LegacyInventory,
    tenant_id: str,
    subject_ids: Mapping[str, str],
    *,
    dictators: Iterable[str] = (),
    standards_enabled: bool = False,
    legacy_key_subject: str | None = None,
    legacy_dictator_grants: bool = True,
    legacy_key_export: bool = False,
    admins: Iterable[str] | None = None,
) -> tuple[PolicyStatement, ...]:
    """The explicit grants that reproduce today's effective sharing.

    The shared key's administrator grant depends on who the key is bound to (``NEURALSCAPE_API_KEY_USER_ID``,
    else the default user) and on who the administrators are (``DICTATOR_USER_IDS``) AT MIGRATION TIME:
    re-run the migration after changing either.

    ``admins`` (default: ``admin_user_ids(dictators)``) receive the administrator operations, each on a
    privileged tenant handle that no content grant matches.
    """

    unique = set(report.unique) & set(subject_ids)
    quarantined_owners = set(report.quarantined)
    bad_groups = {n for n in report.quarantined_groups}

    def placements(visibility: str, owner: str | None = None) -> set[tuple[str | None, str | None]]:
        found = {(None, None)}
        for pool in inventory.pools:
            if pool.visibility == visibility and (owner is None or pool.owner == owner):
                if pool.owner in quarantined_owners:
                    continue
                found.add((pool.project, _norm_workspace(pool.workspace)))
        for name, parsed in report.groups.items():
            if parsed.problem or parsed.kind != visibility or name in bad_groups:
                continue
            if owner is None or parsed.owner == owner:
                found.add((parsed.project, _norm_workspace(parsed.workspace)))
        return found

    out: dict[str, PolicyStatement] = {}

    def grant(
        subject: str, actions: Iterable[str], handle, *,
        condition: PolicyCondition | None = None, origin: str | None = None,
    ) -> None:
        tag = f"{condition.kind if condition else ''}|{origin or ''}"
        for action in actions:
            sid = statement_id(subject, action, handle.id, tag)
            out[sid] = PolicyStatement(
                id=sid, effect="allow", subject_id=subject, credential_id=None,
                action=action, resource=handle, condition=condition, origin=origin,
            )

    private_pools = {u: placements("private", u) for u in unique}
    shared_pools = placements("shared")
    standard_pools = placements("standard") if standards_enabled else set()
    dictator_set = set(dictators) & unique

    for user in sorted(unique):
        subject = subject_ids[user]
        for project, workspace in sorted(private_pools[user], key=str):
            grant(subject, PRIVATE_ACTIONS, pool_handle(tenant_id, "private", subject, project, workspace))
        for project, workspace in sorted(shared_pools, key=str):
            shared = pool_handle(tenant_id, "shared", None, project, workspace)
            grant(subject, SHARED_ACTIONS, shared)
            # Legacy: the author may edit or delete their own shared rows, and
            # any member may edit organizational metadata.  Both are judged
            # against the CURRENT row at decision time.
            grant(subject, ("update", "delete"), shared, condition=OWNER_CONDITION)
            grant(subject, ("update",), shared, condition=METADATA_CONDITION)
            if user in dictator_set and legacy_dictator_grants:
                grant(subject, ("update", "delete"), shared, origin=LEGACY_DICTATOR_ORIGIN)
        if user in dictator_set and legacy_dictator_grants:
            # Legacy: a dictator may edit or delete any user's private rows.
            # Kept as explicit, tagged content grants (never publish_standard).
            for other in sorted(unique - {user}):
                for project, workspace in sorted(private_pools[other], key=str):
                    grant(subject, ("update", "delete"),
                          pool_handle(tenant_id, "private", subject_ids[other], project, workspace),
                          origin=LEGACY_DICTATOR_ORIGIN)
        for project, workspace in sorted(standard_pools, key=str):
            handle = pool_handle(tenant_id, "standard", None, project, workspace)
            grant(subject, STANDARD_READ_ACTIONS, handle)
            if user in dictator_set:
                grant(subject, DICTATOR_STANDARD_ACTIONS, handle)

    admin_set = set(admin_user_ids(dictators) if admins is None else admins) & unique
    for user in sorted(admin_set):
        for action in ADMIN_ACTIONS_ON_TENANT:
            grant(subject_ids[user], (action,), privileged_handle(action, tenant_handle(tenant_id)),
                  origin=ADMIN_ORIGIN)
    if legacy_key_subject:
        # Under legacy_compat the shared key acts as ONE bound identity (NEURALSCAPE_API_KEY_USER_ID, else
        # the default user).  When that user is an administrator the key keeps exactly that user's
        # administrator operations (and nothing else: still no standards, membership or credentials).
        from config import settings

        all_admins = set(admin_user_ids(dictators) if admins is None else admins)
        if settings.legacy_key_identity() in all_admins:
            for action in ADMIN_ACTIONS_ON_TENANT:
                grant(legacy_key_subject, (action,), privileged_handle(action, tenant_handle(tenant_id)),
                      origin=ADMIN_ORIGIN)

    if legacy_key_subject:
        # The shared key acted as any user it asserted: it holds each unique
        # user's content grants and the shared/standard reads, never the
        # standard-publication or administrator actions (the shared key is never a standards publisher or an administrator).
        # Bulk export of every user's private data is not granted by default.
        key_private = tuple(a for a in PRIVATE_ACTIONS if legacy_key_export or a != "export")
        for user in sorted(unique):
            for project, workspace in sorted(private_pools[user], key=str):
                grant(legacy_key_subject, key_private,
                      pool_handle(tenant_id, "private", subject_ids[user], project, workspace))
        for project, workspace in sorted(shared_pools, key=str):
            shared = pool_handle(tenant_id, "shared", None, project, workspace)
            grant(legacy_key_subject, SHARED_ACTIONS, shared)
            # The key edits and deletes shared rows exactly as the user it asserts
            # could: the enforcement layer compares the row owner with the
            # asserted user for the key principal.
            grant(legacy_key_subject, ("update", "delete"), shared, condition=OWNER_CONDITION)
            grant(legacy_key_subject, ("update",), shared, condition=METADATA_CONDITION)
        for project, workspace in sorted(standard_pools, key=str):
            grant(legacy_key_subject, STANDARD_READ_ACTIONS,
                  pool_handle(tenant_id, "standard", None, project, workspace))
    return tuple(out[k] for k in sorted(out))


# ── dry run, apply, rollback ─────────────────────────────────────────────


class PolicyDiff(NamedTuple):
    added: tuple[str, ...]
    removed: tuple[str, ...]
    unchanged: int


def _signature(statement: PolicyStatement) -> str:
    return statement.model_dump_json()


def diff_policy(
    current: Sequence[PolicyStatement], planned: Sequence[PolicyStatement]
) -> PolicyDiff:
    have = {s.id: _signature(s) for s in current}
    want = {s.id: _signature(s) for s in planned}
    added = tuple(sorted(i for i in want if have.get(i) != want[i]))
    removed = tuple(sorted(i for i in have if i not in want))
    return PolicyDiff(added, removed, sum(1 for i in want if have.get(i) == want[i]))


class ApplyResult(NamedTuple):
    changed: bool
    previous_epoch: int
    previous_statements: tuple[PolicyStatement, ...]
    applied_epoch: int
    subject_ids: dict[str, str]
    subjects_created: int


def _planned_subject_ids(store: SqlAuthorityStore, report: MigrationReport) -> dict[str, str]:
    """Existing subject ids, or a ``pending:`` placeholder (a dry run writes nothing)."""

    return {
        user: resolve_binding(store, LOCAL_ISSUER, user) or f"pending:{user}"
        for user in report.unique
    }


def plan_against_store(
    store: SqlAuthorityStore,
    inventory: LegacyInventory,
    tenant_id: str,
    *,
    dictators: Iterable[str] = (),
    standards_enabled: bool = False,
    include_legacy_key: bool = True,
    legacy_dictator_grants: bool = True,
    legacy_key_export: bool = False,
) -> tuple[MigrationReport, tuple[PolicyStatement, ...], PolicyDiff]:
    """Dry run: classify, generate, diff against the current policy.  Writes nothing."""

    dictators = tuple(dictators)
    report = build_report(inventory, extra_user_ids=(*dictators, *admin_user_ids(dictators)))
    ids = _planned_subject_ids(store, report)
    legacy = None
    if include_legacy_key:
        legacy = resolve_binding(store, LOCAL_ISSUER, LEGACY_KEY_SUBJECT) or f"pending:{LEGACY_KEY_SUBJECT}"
    statements = build_statements(
        report, inventory, tenant_id, ids, dictators=dictators,
        standards_enabled=standards_enabled, legacy_key_subject=legacy,
        legacy_dictator_grants=legacy_dictator_grants,
        legacy_key_export=legacy_key_export,
    )
    current = store.current_policy(tenant_id)
    return report, statements, diff_policy(current.statements if current else (), statements)


def apply_plan(
    store: SqlAuthorityStore,
    inventory: LegacyInventory,
    tenant_id: str,
    *,
    dictators: Iterable[str] = (),
    standards_enabled: bool = False,
    include_legacy_key: bool = True,
    legacy_dictator_grants: bool = True,
    legacy_key_export: bool = False,
    expected_epoch: int | None = None,
) -> ApplyResult:
    """Bind unique users, then publish the generated policy once.

    Idempotent: bindings and subjects are created at most once, and when the
    current policy already equals the plan nothing is published (the epoch does
    not move).  Quarantined users and pools receive no binding and no grants.
    """

    dictators = tuple(dictators)
    report = build_report(inventory, extra_user_ids=(*dictators, *admin_user_ids(dictators)))
    ids: dict[str, str] = {}
    created = 0
    for user in report.unique:
        subject = bind_identity(store, LOCAL_ISSUER, user).subject_id
        if store.subject_summary(subject) is None:
            store.add_subject(subject, tenant_id, "human", f"legacy-token:{user}")
            created += 1
        ids[user] = subject
    legacy = None
    if include_legacy_key:
        legacy = bind_identity(store, LOCAL_ISSUER, LEGACY_KEY_SUBJECT).subject_id
        if store.subject_summary(legacy) is None:
            store.add_subject(legacy, tenant_id, "service_account", "legacy-shared-key")
            created += 1
    statements = build_statements(
        report, inventory, tenant_id, ids, dictators=dictators,
        standards_enabled=standards_enabled, legacy_key_subject=legacy,
        legacy_dictator_grants=legacy_dictator_grants,
        legacy_key_export=legacy_key_export,
    )
    current = store.current_policy(tenant_id)
    previous_epoch = current.policy_epoch if current else 0
    previous = tuple(current.statements) if current else ()
    if current is not None and diff_policy(previous, statements) == PolicyDiff((), (), len(statements)):
        _store_preserve_onboarding(store, tenant_id)
        return ApplyResult(False, previous_epoch, previous, previous_epoch, ids, created)
    published = store.publish_policy(
        tenant_id, statements,
        expected_epoch=previous_epoch if expected_epoch is None else expected_epoch,
    )
    _store_preserve_onboarding(store, tenant_id)
    return ApplyResult(True, previous_epoch, previous, published.policy_epoch, ids, created)


def _store_preserve_onboarding(store: SqlAuthorityStore, tenant_id: str) -> None:
    """A migrated tenant's new members receive the sharing the deployment already had.

    Only when the tenant has no onboarding policy yet, so an operator's later
    choice is never overwritten by re-running the migration.
    """

    from onboarding import PRESERVE_V1

    if store.get_onboarding_policy(tenant_id) is None:
        store.set_onboarding_policy(tenant_id, PRESERVE_V1.version, PRESERVE_V1.model_dump_json())


def rollback(store: SqlAuthorityStore, tenant_id: str, result: ApplyResult) -> AccessPolicy:
    """Restore the policy content that preceded ``apply_plan``.

    The epoch only moves forward: the previous statements are republished under
    a new epoch, with compare-and-set against the applied epoch so a policy
    changed since the apply is never overwritten.  Bindings and subjects the
    apply created remain; without grants they authorize nothing.
    """

    if not result.changed:
        policy = store.current_policy(tenant_id)
        assert policy is not None
        return policy
    return store.publish_policy(
        tenant_id, result.previous_statements, expected_epoch=result.applied_epoch
    )


# ── artifact ownership (for the artifact resolver kind) ──────────────────

_FILE_ID = re.compile(r"[0-9a-f]{16}")
_GLOBAL_PROJECT = "_global"


def artifact_row_lookup(
    settings: Any, bindings: Any, *, tenant_check: Callable[[str], bool] | None = None
) -> Callable[[str, str], RowInfo | None]:
    """A ``row_lookup`` for artifacts: the owning pool of a stored file.

    Artifacts live under ``{user}/{project or _global}/{category}/{file_id}.ext``
    and are private to the uploader by construction.  The lookup returns that
    private pool (owner resolved through the ``local`` binding of the directory
    name), so a read requires ``read`` on the owning pool.  Unknown ids,
    malformed ids and unbound owners resolve to ``None`` and therefore deny,
    identically to a forbidden id.  Other row kinds are not resolved here.
    """

    root = Path(os.path.expanduser(settings.ingest_storage_dir)).resolve()

    def lookup(kind: str, identifier: str) -> RowInfo | None:
        if kind != "artifact" or not _FILE_ID.fullmatch(identifier or ""):
            return None
        if not root.is_dir():
            return None
        for user_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for match in sorted(user_dir.rglob("*")):
                if not (match.is_file() and (match.stem == identifier or match.name == identifier)):
                    continue
                parts = match.relative_to(user_dir).parts
                project = None if parts[0] == _GLOBAL_PROJECT else parts[0]
                owner = resolve_binding(bindings, LOCAL_ISSUER, user_dir.name)
                if owner is None or (tenant_check is not None and not tenant_check(owner)):
                    return None
                return RowInfo("private", owner, project, None, "private")
        return None

    return lookup


# ── CLI ──────────────────────────────────────────────────────────────────


def _load_inventory(args: argparse.Namespace) -> LegacyInventory:
    aliases = RedisAliasSource(args.redis_url).read() if args.redis_url else None
    pools = QdrantPoolSource(args.qdrant_url, args.collection).read() if args.qdrant_url else ()
    groups: set[str] = set()
    if args.neo4j_uri:
        source = Neo4jGroupSource(args.neo4j_uri, args.neo4j_user, os.environ.get(args.neo4j_password_env, ""))
        try:
            groups = source.read()
        finally:
            source.close()
    return collect_inventory(aliases, pools, groups)


def _dictators(args: argparse.Namespace) -> tuple[str, ...]:
    raw = args.dictators if args.dictators is not None else os.environ.get("DICTATOR_USER_IDS", "")
    return tuple(sorted({d.strip() for d in raw.split(",") if d.strip()}))


def _store(args: argparse.Namespace) -> SqlAuthorityStore:
    from authority_store_sql import PostgresAuthorityStore, SqliteAuthorityStore

    if args.authority_sqlite:
        return SqliteAuthorityStore(args.authority_sqlite)
    return PostgresAuthorityStore(os.environ[args.authority_pg_dsn_env])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="policy_migration", description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["inventory", "plan", "apply", "rollback"])
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--redis-url")
    parser.add_argument("--qdrant-url")
    parser.add_argument("--collection", default="neuralscape")
    parser.add_argument("--neo4j-uri")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password-env", default="NEO4J_PASSWORD")
    parser.add_argument("--authority-sqlite")
    parser.add_argument("--authority-pg-dsn-env", default="AUTHORITY_POSTGRES_DSN")
    parser.add_argument("--dictators", help="CSV of legacy user ids (default: DICTATOR_USER_IDS)")
    parser.add_argument("--standards-enabled", action="store_true")
    parser.add_argument("--no-legacy-key", action="store_true")
    parser.add_argument("--legacy-key-export", action="store_true",
                        help="let the legacy shared key export every user's private data")
    parser.add_argument("--no-legacy-dictator-grants", action="store_true",
                        help="new deployments: do not preserve dictators' cross-user edit/delete")
    parser.add_argument("--show-ids", action="store_true", help="print raw ids (default: hashed)")
    parser.add_argument("--rollback-file", help="where apply writes / rollback reads the restore point")
    args = parser.parse_args(argv)

    if args.command == "rollback":
        store = _store(args)
        saved = json.loads(Path(args.rollback_file).read_text())
        result = ApplyResult(
            True, saved["previous_epoch"],
            tuple(PolicyStatement.model_validate(s) for s in saved["previous_statements"]),
            saved["applied_epoch"], {}, 0,
        )
        policy = rollback(store, args.tenant, result)
        print(json.dumps({"restored_content_of_epoch": saved["previous_epoch"], "epoch": policy.policy_epoch}))
        return 0

    inventory = _load_inventory(args)
    dictators = _dictators(args)
    if args.command == "inventory":
        report = build_report(inventory, extra_user_ids=(*dictators, *admin_user_ids(dictators)))
        print(json.dumps(report.to_json(redact=not args.show_ids), indent=2, sort_keys=True))
        return 0
    store = _store(args)
    common = dict(dictators=dictators, standards_enabled=args.standards_enabled,
                  include_legacy_key=not args.no_legacy_key,
                  legacy_dictator_grants=not args.no_legacy_dictator_grants,
                  legacy_key_export=args.legacy_key_export)
    if args.command == "plan":
        report, statements, diff = plan_against_store(store, inventory, args.tenant, **common)
        print(json.dumps({
            "report": report.to_json(redact=not args.show_ids),
            "documented_changes": report.documented_changes(
                inventory, standards_enabled=args.standards_enabled, dictators=dictators,
                legacy_key_export=args.legacy_key_export),
            "statements": len(statements),
            "added": len(diff.added), "removed": len(diff.removed), "unchanged": diff.unchanged,
        }, indent=2, sort_keys=True))
        return 0
    result = apply_plan(store, inventory, args.tenant, **common)
    if args.rollback_file:
        Path(args.rollback_file).write_text(json.dumps({
            "previous_epoch": result.previous_epoch,
            "applied_epoch": result.applied_epoch,
            "previous_statements": [json.loads(s.model_dump_json()) for s in result.previous_statements],
        }))
    print(json.dumps({"changed": result.changed, "epoch": result.applied_epoch,
                      "subjects_created": result.subjects_created}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
