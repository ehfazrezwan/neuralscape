"""Durable authority and binding store (SQLite and PostgreSQL).

One SQL implementation backs both ``authority_store.AuthorityStore`` (current
policy snapshot, subject records) and ``identity_binding.BindingStore``
(``(issuer, provider_subject) -> subject_id``).  A dialect subclass supplies the
connection, transaction scope and placeholder style only, so both backends are
held to the same behavior by one conformance suite.

State kept here:

* bindings with a primary key on ``(issuer, provider_subject)``: first contact
  is one atomic insert, and a race loser receives the existing subject;
* subjects with a per-subject ``membership_version`` that is bumped on every
  change, and tombstones: an offboarded subject keeps its row, so its id is
  never reused and its binding keeps resolving to a denied subject;
* credentials with ``active|retired`` state and optional expiry;
* one policy per tenant, published atomically with the epoch incremented by the
  store (never taken from the caller), optionally compare-and-set.

``hold_current`` is the atomic check-and-publish primitive used by
``publication_fence``.  Migrations are expand-only, embedded here, and tracked
in ``authority_schema_version``.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple, TypeVar

from authority_store import AuthorityUnavailable, SubjectRecord
from contracts_policy import (
    AccessPolicy,
    DelegationConstraints,
    MembershipVersion,
    PolicyStatement,
)
from ledger.port import Migration, SchemaError, assert_expand_only
from ledger.sqlite import _statements as _split_sql_statements

T = TypeVar("T")

_MIGRATION_0001 = """
CREATE TABLE authority_bindings (
    issuer TEXT NOT NULL,
    provider_subject TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (issuer, provider_subject)
);
CREATE TABLE authority_subjects (
    subject_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    subject_kind TEXT NOT NULL,
    credential_id TEXT NOT NULL,
    delegation TEXT,
    membership_version BIGINT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'offboarded')),
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX authority_subjects_tenant ON authority_subjects (tenant_id, status);
CREATE TABLE authority_credentials (
    credential_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'retired')),
    expires_at DOUBLE PRECISION,
    created_at DOUBLE PRECISION NOT NULL,
    retired_at DOUBLE PRECISION
);
CREATE TABLE authority_policies (
    tenant_id TEXT PRIMARY KEY,
    policy_epoch BIGINT NOT NULL
);
CREATE TABLE authority_statements (
    tenant_id TEXT NOT NULL,
    statement_id TEXT NOT NULL,
    ordinal BIGINT NOT NULL,
    body TEXT NOT NULL,
    PRIMARY KEY (tenant_id, statement_id)
);
"""

_MIGRATION_0002 = """
CREATE TABLE authority_credential_secrets (
    credential_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('access', 'refresh')),
    secret_hash TEXT NOT NULL UNIQUE,
    scope TEXT NOT NULL,
    family_id TEXT NOT NULL,
    issued_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL,
    consumed_at DOUBLE PRECISION,
    revoked_at DOUBLE PRECISION
);
CREATE INDEX authority_credential_secrets_family ON authority_credential_secrets (family_id);
CREATE INDEX authority_credential_secrets_subject ON authority_credential_secrets (subject_id);
CREATE TABLE authority_service_accounts (
    subject_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    controller_subject_id TEXT NOT NULL,
    label TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
    created_at DOUBLE PRECISION NOT NULL
);
"""

_MIGRATION_0003 = """
ALTER TABLE authority_credential_secrets ADD COLUMN parent_family_id TEXT;
ALTER TABLE authority_credential_secrets ADD COLUMN successor_id TEXT;
CREATE INDEX authority_credential_secrets_parent ON authority_credential_secrets (parent_family_id);
CREATE TABLE authority_pepper (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    value TEXT NOT NULL
);
"""

_MIGRATION_0004 = """
ALTER TABLE authority_credential_secrets ADD COLUMN cap_expires_at DOUBLE PRECISION;
"""

_MIGRATION_0005 = """
CREATE TABLE authority_invites (
    invite_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    secret_hash TEXT NOT NULL UNIQUE,
    created_by TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL,
    used_at DOUBLE PRECISION,
    used_by TEXT,
    revoked_at DOUBLE PRECISION
);
CREATE TABLE authority_onboarding (
    tenant_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    body TEXT NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);
"""

MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        1,
        "authority_init",
        _MIGRATION_0001,
        hashlib.sha256(_MIGRATION_0001.encode("utf-8")).hexdigest(),
    ),
    Migration(
        2,
        "authority_scoped_credentials",
        _MIGRATION_0002,
        hashlib.sha256(_MIGRATION_0002.encode("utf-8")).hexdigest(),
    ),
    Migration(
        3,
        "authority_credential_lineage",
        _MIGRATION_0003,
        hashlib.sha256(_MIGRATION_0003.encode("utf-8")).hexdigest(),
    ),
    Migration(
        4,
        "authority_credential_cap",
        _MIGRATION_0004,
        hashlib.sha256(_MIGRATION_0004.encode("utf-8")).hexdigest(),
    ),
    Migration(
        5,
        "authority_membership_invites",
        _MIGRATION_0005,
        hashlib.sha256(_MIGRATION_0005.encode("utf-8")).hexdigest(),
    ),
)
_BOOTSTRAP = (
    "CREATE TABLE IF NOT EXISTS authority_schema_version("
    "version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL, "
    "applied_at DOUBLE PRECISION NOT NULL)"
)
_PG_MIGRATION_LOCK = 7_302_011_003


class PolicyConflict(Exception):
    """``publish_policy`` was given a stale ``expected_epoch``."""


class WitnessStale(Exception):
    """Authority state changed between evaluation and publication."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class UnknownSubject(Exception):
    pass


class LastAdministrator(ValueError):
    """The change would leave the tenant with no member who can manage membership."""


class TooManyInvites(ValueError):
    """The tenant already has the maximum number of unused, unexpired invites."""


class CredentialLimit(Exception):
    """A subject already has the maximum number of active credential families."""


class CredentialInvalid(Exception):
    """The presented credential is unknown, expired, revoked or not valid for this use."""


class RefreshReplay(Exception):
    """A consumed refresh token was presented again; its whole family was revoked."""

    def __init__(self, family_id: str) -> None:
        super().__init__("refresh token replay")
        self.family_id = family_id


class VerifiedCredential(NamedTuple):
    subject_id: str
    tenant_id: str
    credential_id: str
    scope: tuple[str, ...]
    family_id: str
    expires_at: float = 0.0


def plan_authority_migrations(applied: dict[int, str]) -> tuple[Migration, ...]:
    """Pending migrations; reject drift and schemas newer than this code."""

    known = {item.version: item for item in MIGRATIONS}
    for version, checksum in applied.items():
        if version not in known:
            raise SchemaError(f"authority schema version {version} is newer than this code")
        if known[version].checksum != checksum:
            raise SchemaError(f"applied authority migration {version} was modified")
    pending = tuple(item for item in MIGRATIONS if item.version not in applied)
    for item in pending:
        assert_expand_only(item)
    return pending


class SqlAuthorityStore:
    """Shared behavior; dialect subclasses provide connection and transactions."""

    dialect: str
    _errors: tuple[type[BaseException], ...]
    _unique_errors: tuple[type[BaseException], ...]

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._read_lock = threading.RLock()
        self._rconn: Any = None
        self._closed = False
        self._conn: Any = None

    # ── dialect hooks ────────────────────────────────────────────────────
    def _txn(self) -> Any:
        raise NotImplementedError

    def _read_txn(self) -> Any:
        raise NotImplementedError

    def _sql(self, sql: str) -> str:
        return sql

    _share = ""  # row-lock suffix for the check-and-publish verification
    _update_lock = ""

    # ── plumbing ─────────────────────────────────────────────────────────
    def _x(self, c: Any, sql: str, params: Sequence[Any] = ()) -> Any:
        return c.execute(self._sql(sql), tuple(params))

    def _one(self, c: Any, sql: str, params: Sequence[Any] = ()) -> Any:
        return self._x(c, sql, params).fetchone()

    @contextlib.contextmanager
    def _guard(self) -> Iterator[None]:
        """Surface any storage failure as an unreadable authority."""

        try:
            yield
        except self._unique_errors as error:
            # A uniqueness violation is a permanent client error, not an outage.
            raise ValueError("conflicting record") from error
        except self._errors as error:
            self._on_failure()
            raise AuthorityUnavailable("authority store failure") from error

    def _on_failure(self) -> None:
        """Hook: a storage error happened (backends may mark connections suspect)."""

    def _migrate(self) -> None:
        with self._lock, self._txn() as c:
            self._lock_for_migration(c)
            self._x(c, _BOOTSTRAP)
            applied = {
                r[0]: r[1]
                for r in self._x(
                    c, "SELECT version, checksum FROM authority_schema_version"
                ).fetchall()
            }
            for migration in plan_authority_migrations(applied):
                for statement in _split_sql_statements(migration.sql):
                    self._x(c, statement)
                self._x(
                    c,
                    "INSERT INTO authority_schema_version(version,name,checksum,applied_at)"
                    " VALUES(?,?,?,?)",
                    (migration.version, migration.name, migration.checksum, self._clock()),
                )

    def _lock_for_migration(self, c: Any) -> None:
        return None

    def schema_version(self) -> int:
        with self._guard(), self._read_lock, self._read_txn() as c:
            return self._one(c, "SELECT COALESCE(MAX(version), 0) FROM authority_schema_version")[0]

    def close(self) -> None:
        with self._lock, self._read_lock:
            self._closed = True
            for name in ("_conn", "_rconn"):
                conn = getattr(self, name)
                if conn is not None:
                    conn.close()
                    setattr(self, name, None)

    # ── BindingStore ─────────────────────────────────────────────────────
    def get(self, issuer: str, provider_subject: str) -> str | None:
        try:
            _ident("issuer", issuer)
            _ident("provider_subject", provider_subject)
        except ValueError:
            return None  # no such binding can exist
        with self._guard(), self._read_lock, self._read_txn() as c:
            row = self._one(
                c,
                "SELECT subject_id FROM authority_bindings WHERE issuer=? AND provider_subject=?",
                (issuer, provider_subject),
            )
            return row[0] if row else None

    def put_if_absent(self, issuer: str, provider_subject: str, subject_id: str) -> str:
        """Atomic first-contact insert; the loser receives the existing subject."""

        _ident("issuer", issuer)
        _ident("provider_subject", provider_subject)
        _ident("subject_id", subject_id)
        with self._guard(), self._lock, self._txn() as c:
            self._x(
                c,
                "INSERT INTO authority_bindings(issuer,provider_subject,subject_id,created_at)"
                " VALUES(?,?,?,?) ON CONFLICT (issuer, provider_subject) DO NOTHING",
                (issuer, provider_subject, subject_id, self._clock()),
            )
            return self._one(
                c,
                "SELECT subject_id FROM authority_bindings WHERE issuer=? AND provider_subject=?",
                (issuer, provider_subject),
            )[0]

    # ── subjects, credentials ────────────────────────────────────────────
    def add_subject(
        self,
        subject_id: str,
        tenant_id: str,
        subject_kind: str,
        credential_id: str,
        *,
        delegation: DelegationConstraints | None = None,
        expires_at: float | None = None,
    ) -> int:
        """Create a subject at membership version 1.  Ids are never reused."""

        for name, value in (("subject_id", subject_id), ("tenant_id", tenant_id),
                            ("subject_kind", subject_kind), ("credential_id", credential_id)):
            _ident(name, value)
        now = self._clock()
        with self._guard(), self._lock, self._txn() as c:
            if self._one(c, "SELECT 1 FROM authority_subjects WHERE subject_id=?", (subject_id,)):
                raise ValueError("subject id already exists or was offboarded")
            if self._one(c, "SELECT 1 FROM authority_credentials WHERE credential_id=?", (credential_id,)):
                raise ValueError("credential id already exists or was retired")
            self._x(
                c,
                "INSERT INTO authority_subjects(subject_id,tenant_id,subject_kind,credential_id,"
                "delegation,membership_version,status,created_at,updated_at)"
                " VALUES(?,?,?,?,?,1,'active',?,?)",
                (subject_id, tenant_id, subject_kind, credential_id, _dump(delegation), now, now),
            )
            self._x(
                c,
                "INSERT INTO authority_credentials(credential_id,subject_id,status,expires_at,created_at)"
                " VALUES(?,?,'active',?,?)",
                (credential_id, subject_id, expires_at, now),
            )
            return 1

    def update_subject(
        self,
        subject_id: str,
        *,
        subject_kind: str | None = None,
        credential_id: str | None = None,
        delegation: DelegationConstraints | None = None,
        expires_at: float | None = None,
    ) -> int:
        """Change a subject and bump its membership version.

        Passing a new ``credential_id`` is rotation: the previous credential is
        retired in the same transaction.
        """

        now = self._clock()
        with self._guard(), self._lock, self._txn() as c:
            row = self._one(
                c,
                "SELECT credential_id, subject_kind, delegation, status, membership_version"
                " FROM authority_subjects WHERE subject_id=?" + self._update_lock,
                (subject_id,),
            )
            if row is None:
                raise UnknownSubject(subject_id)
            if row[3] != "active":
                raise ValueError("offboarded subjects cannot change")
            new_credential = credential_id or row[0]
            if credential_id:
                _ident("credential_id", credential_id)
            if credential_id and credential_id != row[0]:
                if self._one(c, "SELECT 1 FROM authority_credentials WHERE credential_id=?", (credential_id,)):
                    raise ValueError("credential id already exists or was retired")
                self._x(
                    c,
                    "UPDATE authority_credentials SET status='retired', retired_at=?"
                    " WHERE credential_id=?",
                    (now, row[0]),
                )
                self._x(
                    c,
                    "INSERT INTO authority_credentials(credential_id,subject_id,status,expires_at,created_at)"
                    " VALUES(?,?,'active',?,?)",
                    (credential_id, subject_id, expires_at, now),
                )
            if expires_at is not None and not (credential_id and credential_id != row[0]):
                # No rotation: the new expiry applies to the subject's current credential.
                self._x(
                    c,
                    "UPDATE authority_credentials SET expires_at=? WHERE credential_id=? AND status='active'",
                    (expires_at, new_credential),
                )
            version = row[4] + 1
            self._x(
                c,
                "UPDATE authority_subjects SET subject_kind=?, credential_id=?, delegation=?,"
                " membership_version=?, updated_at=? WHERE subject_id=?",
                (
                    subject_kind or row[1],
                    new_credential,
                    _dump(delegation) if delegation is not None else row[2],
                    version,
                    now,
                    subject_id,
                ),
            )
            return version

    def offboard_subject(self, subject_id: str) -> int:
        """Tombstone a subject: bump its version, retire all its credentials."""

        now = self._clock()
        with self._guard(), self._lock, self._txn() as c:
            return self._offboard(c, subject_id, now)

    def _offboard(self, c: Any, subject_id: str, now: float) -> int:
        row = self._one(
            c,
            "SELECT membership_version, status FROM authority_subjects WHERE subject_id=?"
            + self._update_lock,
            (subject_id,),
        )
        if row is None:
            raise UnknownSubject(subject_id)
        if row[1] == "offboarded":
            return row[0]
        self._x(
            c,
            "UPDATE authority_credentials SET status='retired', retired_at=?"
            " WHERE subject_id=? AND status='active'",
            (now, subject_id),
        )
        # Invites the subject issued and nobody has used stop admitting anyone with them: a removed
        # or compromised administrator's pre-issued invites would otherwise outlive the removal.
        self._x(
            c,
            "UPDATE authority_invites SET revoked_at=?"
            " WHERE created_by=? AND used_at IS NULL AND revoked_at IS NULL",
            (now, subject_id),
        )
        self._x(
            c,
            "UPDATE authority_subjects SET status='offboarded', membership_version=?,"
            " updated_at=? WHERE subject_id=?",
            (row[0] + 1, now, subject_id),
        )
        # A service account is owned by a live human controller; when that
        # controller goes, what they controlled is disabled with them.
        for (account,) in self._x(
            c,
            "SELECT subject_id FROM authority_service_accounts"
            " WHERE controller_subject_id=? AND status='active'",
            (subject_id,),
        ).fetchall():
            self._disable_account(c, account, now)
        return row[0] + 1

    def retire_credential(self, credential_id: str) -> bool:
        """Record a retirement.  Work admitted under it is fenced at publication."""

        with self._guard(), self._lock, self._txn() as c:
            row = self._one(
                c,
                "SELECT status FROM authority_credentials WHERE credential_id=?" + self._update_lock,
                (credential_id,),
            )
            if row is None:
                raise UnknownSubject(credential_id)
            if row[0] == "retired":
                return False
            self._x(
                c,
                "UPDATE authority_credentials SET status='retired', retired_at=? WHERE credential_id=?",
                (self._clock(), credential_id),
            )
            return True

    def credential_state(self, credential_id: str) -> str:
        """``active``, ``retired``, ``expired`` or ``unknown``."""

        with self._guard(), self._read_lock, self._read_txn() as c:
            return self._credential_state(c, credential_id)

    def credential_owner(self, credential_id: str) -> str | None:
        """The subject a credential belongs to, or ``None`` if it is unknown."""

        with self._guard(), self._read_lock, self._read_txn() as c:
            row = self._one(
                c, "SELECT subject_id FROM authority_credentials WHERE credential_id=?", (credential_id,)
            )
            return row[0] if row else None

    def local_alias(self, subject_id: str) -> str | None:
        """The local user id bound to a subject (the smallest when several), or ``None``.

        Every alias resolves to the same subject, so any of them names the same owner.
        """

        with self._guard(), self._read_lock, self._read_txn() as c:
            row = self._one(
                c,
                "SELECT MIN(provider_subject) FROM authority_bindings WHERE issuer=? AND subject_id=?",
                ("local", subject_id),
            )
        return row[0] if row else None

    def credential_scope(self, credential_id: str) -> tuple[str, ...] | None:
        """The action scope a credential narrows to; ``None`` only for a subject's sign-in credential.

        A subject's current sign-in credential carries the subject's grants and has no
        scope.  Every other credential is an issued (scoped) one, decided by the
        credential itself rather than by whether its scope row can be found: when the
        scope cannot be read (a missing or malformed row) the scope is empty, so
        nothing is done under it.  Fail closed, never widened to the subject's grants.
        """

        with self._guard(), self._read_lock, self._read_txn() as c:
            if self._one(c, "SELECT 1 FROM authority_subjects WHERE credential_id=?", (credential_id,)):
                return None
            row = self._one(
                c, "SELECT scope FROM authority_credential_secrets WHERE credential_id=?", (credential_id,)
            )
        if row is None:
            return ()
        try:
            scope = json.loads(row[0])
        except (TypeError, ValueError):
            return ()
        if not isinstance(scope, list) or not all(isinstance(action, str) for action in scope):
            return ()
        return tuple(scope)

    def _credential_state(self, c: Any, credential_id: str) -> str:
        row = self._one(
            c,
            "SELECT status, expires_at FROM authority_credentials WHERE credential_id=?",
            (credential_id,),
        )
        if row is None:
            return "unknown"
        if row[0] != "active":
            return row[0]
        if row[1] is not None and row[1] <= self._clock():
            return "expired"
        return "active"

    # ── AuthorityStore ───────────────────────────────────────────────────
    def subject_record(self, subject_id: str) -> SubjectRecord | None:
        with self._guard(), self._read_lock, self._read_txn() as c:
            return self._subject_record(c, subject_id)

    def _subject_record(self, c: Any, subject_id: str) -> SubjectRecord | None:
        row = self._one(
            c,
            "SELECT tenant_id, subject_kind, credential_id, delegation, membership_version, status"
            " FROM authority_subjects WHERE subject_id=?",
            (subject_id,),
        )
        if row is None or row[5] != "active":
            return None
        # A subject is usable while it has at least one active credential: the
        # primary one, or (during a no-downtime rotation) a successor.
        credential_id = row[2]
        if self._credential_state(c, credential_id) != "active":
            credential_id = None
            for (candidate,) in self._x(
                c,
                "SELECT credential_id FROM authority_credentials WHERE subject_id=? AND status='active'"
                " ORDER BY created_at, credential_id",
                (subject_id,),
            ).fetchall():
                if self._credential_state(c, candidate) == "active":
                    credential_id = candidate
                    break
            if credential_id is None:
                return None
        return SubjectRecord(row[0], row[1], credential_id, row[4], _load_delegation(row[3]))

    def current_policy(self, tenant_id: str) -> AccessPolicy | None:
        with self._guard(), self._read_lock, self._read_txn() as c:
            return self._policy(c, tenant_id)

    def tenants(self) -> tuple[str, ...]:
        """Every tenant that has a policy or a subject (read only, sorted)."""

        with self._guard(), self._lock, self._read_txn() as c:
            rows = self._x(
                c,
                "SELECT tenant_id FROM authority_policies UNION SELECT tenant_id FROM authority_subjects"
                " ORDER BY tenant_id",
            ).fetchall()
        return tuple(r[0] for r in rows)

    def _policy(self, c: Any, tenant_id: str) -> AccessPolicy | None:
        head = self._one(
            c, "SELECT policy_epoch FROM authority_policies WHERE tenant_id=?", (tenant_id,)
        )
        if head is None:
            return None
        memberships = self._x(
            c,
            "SELECT subject_id, membership_version FROM authority_subjects"
            " WHERE tenant_id=? AND status='active' ORDER BY subject_id",
            (tenant_id,),
        ).fetchall()
        statements = self._x(
            c,
            "SELECT body FROM authority_statements WHERE tenant_id=? ORDER BY ordinal",
            (tenant_id,),
        ).fetchall()
        return AccessPolicy(
            schema_version="candidate-v1",
            tenant_id=tenant_id,
            policy_epoch=head[0],
            membership_versions=tuple(
                MembershipVersion(subject_id=r[0], version=r[1]) for r in memberships
            ),
            statements=tuple(PolicyStatement.model_validate_json(r[0]) for r in statements),
        )

    def publish_policy(
        self,
        tenant_id: str,
        statements: Sequence[PolicyStatement],
        *,
        expected_epoch: int | None = None,
    ) -> AccessPolicy:
        """Atomically replace the tenant's statements; the epoch increments by one.

        ``expected_epoch`` makes the publish compare-and-set (0 for the first
        policy).  Statements are validated as an ``AccessPolicy`` before any
        write, so an invalid set changes nothing.
        """

        if expected_epoch is not None and type(expected_epoch) is not int:
            # compared in Python below: an int subclass could claim to match and defeat the compare-and-set
            raise ValueError("expected_epoch must be an integer")
        statements = tuple(statements)
        AccessPolicy(
            schema_version="candidate-v1",
            tenant_id=tenant_id,
            policy_epoch=0,
            membership_versions=(),
            statements=statements,
        )
        with self._guard(), self._lock, self._txn() as c:
            self._x(
                c,
                "INSERT INTO authority_policies(tenant_id, policy_epoch) VALUES(?, 0)"
                " ON CONFLICT (tenant_id) DO NOTHING",
                (tenant_id,),
            )
            epoch = self._one(
                c,
                "SELECT policy_epoch FROM authority_policies WHERE tenant_id=?" + self._update_lock,
                (tenant_id,),
            )[0]
            if expected_epoch is not None and expected_epoch != epoch:
                raise PolicyConflict(f"policy epoch is {epoch}, expected {expected_epoch}")
            self._x(c, "DELETE FROM authority_statements WHERE tenant_id=?", (tenant_id,))
            for ordinal, statement in enumerate(statements):
                self._x(
                    c,
                    "INSERT INTO authority_statements(tenant_id,statement_id,ordinal,body)"
                    " VALUES(?,?,?,?)",
                    (tenant_id, statement.id, ordinal, statement.model_dump_json()),
                )
            self._x(
                c,
                "UPDATE authority_policies SET policy_epoch=? WHERE tenant_id=?",
                (epoch + 1, tenant_id),
            )
            policy = self._policy(c, tenant_id)
            assert policy is not None
            return policy

    # ── scoped credentials, refresh families, service accounts ───────────
    def subject_summary(self, subject_id: str) -> tuple[str, str] | None:
        """``(tenant_id, subject_kind)`` of an ACTIVE subject, credential or not.

        A new service account has no credential yet, so ``subject_record`` (which
        requires one) cannot be used to issue its first.
        """

        with self._guard(), self._read_lock, self._read_txn() as c:
            row = self._one(
                c,
                "SELECT tenant_id, subject_kind, status FROM authority_subjects WHERE subject_id=?",
                (subject_id,),
            )
            return (row[0], row[1]) if row and row[2] == "active" else None

    def create_service_account(
        self, subject_id: str, tenant_id: str, controller_subject_id: str, label: str
    ) -> int:
        """Create a service account with NO credential; it is unusable until one is issued."""

        for name, value in (("subject_id", subject_id), ("tenant_id", tenant_id),
                            ("controller_subject_id", controller_subject_id)):
            _ident(name, value)
        if not 1 <= len(label) <= 128 or any(ord(ch) < 32 or ord(ch) == 127 for ch in label):
            raise ValueError("label must be 1..128 printable characters")
        now = self._clock()
        with self._guard(), self._lock, self._txn() as c:
            controller = self._one(
                c,
                "SELECT tenant_id, subject_kind, status FROM authority_subjects WHERE subject_id=?",
                (controller_subject_id,),
            )
            if controller is None or controller[2] != "active" or controller[0] != tenant_id:
                raise ValueError("controller must be an active subject of the same tenant")
            if controller[1] != "human":
                raise ValueError("only a human subject may control a service account")
            if self._one(c, "SELECT 1 FROM authority_subjects WHERE subject_id=?", (subject_id,)):
                raise ValueError("subject id already exists or was offboarded")
            self._x(
                c,
                "INSERT INTO authority_subjects(subject_id,tenant_id,subject_kind,credential_id,"
                "delegation,membership_version,status,created_at,updated_at)"
                " VALUES(?,?,'service_account',?,NULL,1,'active',?,?)",
                (subject_id, tenant_id, f"unissued:{subject_id}", now, now),
            )
            self._x(
                c,
                "INSERT INTO authority_service_accounts(subject_id,tenant_id,controller_subject_id,label,status,created_at)"
                " VALUES(?,?,?,?,'active',?)",
                (subject_id, tenant_id, controller_subject_id, label, now),
            )
            return 1

    def service_account(self, subject_id: str) -> dict[str, Any] | None:
        with self._guard(), self._read_lock, self._read_txn() as c:
            row = self._one(
                c,
                "SELECT tenant_id, controller_subject_id, label, status FROM authority_service_accounts"
                " WHERE subject_id=?",
                (subject_id,),
            )
            if row is None:
                return None
            return {"tenant_id": row[0], "controller_subject_id": row[1], "label": row[2], "status": row[3]}

    def disable_service_account(self, subject_id: str) -> bool:
        """Retire every credential of the account and mark it disabled."""

        now = self._clock()
        with self._guard(), self._lock, self._txn() as c:
            row = self._one(
                c,
                "SELECT status FROM authority_service_accounts WHERE subject_id=?" + self._update_lock,
                (subject_id,),
            )
            if row is None:
                raise UnknownSubject(subject_id)
            if row[0] == "disabled":
                return False
            self._disable_account(c, subject_id, now)
            return True

    def _disable_account(self, c: Any, subject_id: str, now: float) -> None:
        self._x(c, "UPDATE authority_service_accounts SET status='disabled' WHERE subject_id=?", (subject_id,))
        self._retire_subject_credentials(c, subject_id, now)
        version = self._one(
            c, "SELECT membership_version FROM authority_subjects WHERE subject_id=?", (subject_id,)
        )[0]
        self._x(
            c,
            "UPDATE authority_subjects SET membership_version=?, updated_at=? WHERE subject_id=?",
            (version + 1, now, subject_id),
        )

    def _retire_subject_credentials(self, c: Any, subject_id: str, now: float) -> None:
        self._x(
            c,
            "UPDATE authority_credentials SET status='retired', retired_at=? WHERE subject_id=? AND status='active'",
            (now, subject_id),
        )
        self._x(
            c,
            "UPDATE authority_credential_secrets SET revoked_at=? WHERE subject_id=? AND revoked_at IS NULL",
            (now, subject_id),
        )

    def _active_families(self, c: Any, subject_id: str) -> int:
        now = self._clock()
        return self._one(
            c,
            "SELECT COUNT(DISTINCT family_id) FROM authority_credential_secrets"
            " WHERE subject_id=? AND revoked_at IS NULL AND expires_at > ?"
            " AND (kind='access' OR consumed_at IS NULL)",
            (subject_id, now),
        )[0]

    def issue_family(
        self,
        *,
        subject_id: str,
        scope: Sequence[str],
        family_id: str,
        access: tuple[str, str, float],
        refresh: tuple[str, str, float] | None,
        max_families: int | None = None,
        parent_family: str | None = None,
        cap_expires_at: float | None = None,
    ) -> None:
        """Insert an access credential (and its refresh token) as one family.

        ``access``/``refresh`` are ``(credential_id, secret_hash, expires_at)``.
        The access credential is also an ``authority_credentials`` row, so
        retirement fences work enqueued under it.  ``max_families`` bounds the
        subject's concurrently usable families (two for a service account).
        """

        for value in (subject_id, family_id, access[0], *(refresh[:1] if refresh else ())):
            _ident("id", value)
        now = self._clock()
        with self._guard(), self._lock, self._txn() as c:
            subject = self._one(
                c,
                "SELECT tenant_id, status FROM authority_subjects WHERE subject_id=?" + self._update_lock,
                (subject_id,),
            )
            if subject is None or subject[1] != "active":
                raise UnknownSubject(subject_id)
            for credential in (access[0], *(refresh[:1] if refresh else ())):
                if self._one(c, "SELECT 1 FROM authority_credential_secrets WHERE credential_id=?", (credential,)):
                    raise ValueError("credential id already exists")
            account = self._one(
                c, "SELECT status FROM authority_service_accounts WHERE subject_id=?", (subject_id,)
            )
            if account is not None and account[0] != "active":
                raise ValueError("service account is disabled")
            if max_families is not None and self._active_families(c, subject_id) >= max_families:
                raise CredentialLimit(max_families)
            self._insert_family_rows(
                c, subject_id, subject[0], scope, family_id, access, refresh, now, parent_family,
                cap_expires_at,
            )

    def _insert_family_rows(
        self, c: Any, subject_id: str, tenant_id: str, scope: Sequence[str], family_id: str,
        access: tuple[str, str, float], refresh: tuple[str, str, float] | None, now: float,
        parent_family: str | None = None,
        cap_expires_at: float | None = None,
    ) -> None:
        scope_text = json.dumps(sorted(set(scope)))
        if cap_expires_at is not None:
            # The family's lifetime cap travels with it: no successor may outlive it.
            access = (access[0], access[1], min(access[2], cap_expires_at))
            if refresh is not None:
                refresh = (refresh[0], refresh[1], min(refresh[2], cap_expires_at))
        self._x(
            c,
            "INSERT INTO authority_credentials(credential_id,subject_id,status,expires_at,created_at)"
            " VALUES(?,?,'active',?,?)",
            (access[0], subject_id, access[2], now),
        )
        rows = [(access, "access")] + ([(refresh, "refresh")] if refresh else [])
        for (credential_id, secret_hash, expires_at), kind in rows:
            self._x(
                c,
                "INSERT INTO authority_credential_secrets(credential_id,subject_id,tenant_id,kind,secret_hash,"
                "scope,family_id,issued_at,expires_at,parent_family_id,cap_expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (credential_id, subject_id, tenant_id, kind, secret_hash, scope_text, family_id, now,
                 expires_at, parent_family, cap_expires_at),
            )

    def verify_access(self, credential_id: str, secret_hash: str) -> VerifiedCredential | None:
        """The access credential's facts, or ``None`` unless it is valid right now."""

        with self._guard(), self._read_lock, self._read_txn() as c:
            row = self._one(
                c,
                "SELECT subject_id, tenant_id, scope, family_id, secret_hash, kind, revoked_at"
                " FROM authority_credential_secrets WHERE credential_id=?",
                (credential_id,),
            )
            if row is None or row[5] != "access" or row[6] is not None:
                return None
            if not hmac.compare_digest(row[4], secret_hash):
                return None
            if self._credential_state(c, credential_id) != "active":
                return None
            subject = self._one(
                c,
                "SELECT status, tenant_id FROM authority_subjects WHERE subject_id=?",
                (row[0],),
            )
            if subject is None or subject[0] != "active" or subject[1] != row[1]:
                return None
            account = self._one(
                c, "SELECT status FROM authority_service_accounts WHERE subject_id=?", (row[0],)
            )
            if account is not None and account[0] != "active":
                return None
            expiry = self._one(
                c, "SELECT expires_at FROM authority_credential_secrets WHERE credential_id=?", (credential_id,)
            )[0]
            return VerifiedCredential(
                row[0], row[1], credential_id, tuple(json.loads(row[2])), row[3], expiry
            )

    def rotate_refresh(
        self,
        refresh_id: str,
        refresh_hash: str,
        *,
        access: tuple[str, str, float],
        refresh: tuple[str, str, float],
        grace_seconds: float = 0.0,
    ) -> tuple[VerifiedCredential, float]:
        """Consume a refresh token once and issue its successor in the same family.

        Returns ``(access credential, refresh expiry)``.  Presenting a consumed
        refresh token is a replay and revokes the whole family (successor
        included), before :class:`RefreshReplay` is raised, with one exception: a
        duplicate submission within ``grace_seconds`` of the first redemption
        that asks for the SAME successor (the caller derives it deterministically
        from the presented token) while that successor is still unused and valid
        is an idempotent retry and returns the same result without writing.
        """

        now = self._clock()
        replay = False
        with self._guard(), self._lock, self._txn() as c:
            row = self._one(
                c,
                "SELECT subject_id, tenant_id, scope, family_id, secret_hash, kind, consumed_at,"
                " revoked_at, expires_at, successor_id, parent_family_id, cap_expires_at"
                " FROM authority_credential_secrets WHERE credential_id=?" + self._update_lock,
                (refresh_id,),
            )
            if row is None or row[5] != "refresh" or not hmac.compare_digest(row[4], refresh_hash):
                raise CredentialInvalid("unknown refresh token")
            subject_id, tenant_id, scope_text, family_id = row[0], row[1], row[2], row[3]
            parent_family, cap = row[10], row[11]
            if row[7] is not None or row[8] <= now:
                raise CredentialInvalid("refresh token is revoked or expired")
            scope = json.loads(scope_text)
            if row[6] is not None:
                if grace_seconds > 0 and now - row[6] <= grace_seconds and row[9] == refresh[0]:
                    successor = self._one(
                        c,
                        "SELECT consumed_at, revoked_at, expires_at FROM authority_credential_secrets"
                        " WHERE credential_id=?",
                        (refresh[0],),
                    )
                    live = self._one(
                        c,
                        "SELECT revoked_at, expires_at FROM authority_credential_secrets WHERE credential_id=?",
                        (access[0],),
                    )
                    if (successor is not None and successor[0] is None and successor[1] is None
                            and live is not None and live[0] is None):
                        return (
                            VerifiedCredential(subject_id, tenant_id, access[0], tuple(scope), family_id, live[1]),
                            successor[2],
                        )
                self._revoke_family(c, family_id, now)
                replay = True
            else:
                subject = self._one(
                    c, "SELECT status FROM authority_subjects WHERE subject_id=?", (subject_id,)
                )
                if subject is None or subject[0] != "active":
                    raise CredentialInvalid("subject is not active")
                account = self._one(
                    c, "SELECT status FROM authority_service_accounts WHERE subject_id=?", (subject_id,)
                )
                if account is not None and account[0] != "active":
                    raise CredentialInvalid("service account is disabled")
                self._x(
                    c,
                    "UPDATE authority_credential_secrets SET consumed_at=?, successor_id=? WHERE credential_id=?",
                    (now, refresh[0], refresh_id),
                )
                for (previous,) in self._x(
                    c,
                    "SELECT credential_id FROM authority_credential_secrets"
                    " WHERE family_id=? AND kind='access' AND revoked_at IS NULL",
                    (family_id,),
                ).fetchall():
                    self._x(
                        c,
                        "UPDATE authority_credentials SET status='retired', retired_at=? WHERE credential_id=?"
                        " AND status='active'",
                        (now, previous),
                    )
                    self._x(
                        c,
                        "UPDATE authority_credential_secrets SET revoked_at=? WHERE credential_id=?",
                        (now, previous),
                    )
                self._insert_family_rows(
                    c, subject_id, tenant_id, scope, family_id, access, refresh, now, parent_family, cap
                )
                if cap is not None:  # report what was actually stored
                    access = (access[0], access[1], min(access[2], cap))
                    refresh = (refresh[0], refresh[1], min(refresh[2], cap))
        if replay:
            raise RefreshReplay(family_id)
        return (
            VerifiedCredential(subject_id, tenant_id, access[0], tuple(scope), family_id, access[2]),
            refresh[2],
        )

    def _revoke_family(self, c: Any, family_id: str, now: float) -> None:
        """Revoke a family and, transitively, every family it minted."""

        pending, seen = [family_id], set()
        while pending:
            family = pending.pop()
            if family in seen:
                continue
            seen.add(family)
            self._x(
                c,
                "UPDATE authority_credentials SET status='retired', retired_at=? WHERE status='active'"
                " AND credential_id IN (SELECT credential_id FROM authority_credential_secrets WHERE family_id=?)",
                (now, family),
            )
            self._x(
                c,
                "UPDATE authority_credential_secrets SET revoked_at=? WHERE family_id=? AND revoked_at IS NULL",
                (now, family),
            )
            pending.extend(
                r[0]
                for r in self._x(
                    c,
                    "SELECT DISTINCT family_id FROM authority_credential_secrets WHERE parent_family_id=?",
                    (family,),
                ).fetchall()
            )

    def revoke_family(self, family_id: str) -> int:
        """Revoke a whole family (access and refresh).  Returns the rows revoked."""

        now = self._clock()
        with self._guard(), self._lock, self._txn() as c:
            before = self._one(
                c,
                "SELECT COUNT(*) FROM authority_credential_secrets WHERE family_id=? AND revoked_at IS NULL",
                (family_id,),
            )[0]
            self._revoke_family(c, family_id, now)
            return before

    def credential_info(self, credential_id: str) -> dict[str, Any] | None:
        """Non-secret facts about a scoped credential, for ownership checks."""

        with self._guard(), self._read_lock, self._read_txn() as c:
            row = self._one(
                c,
                "SELECT subject_id, tenant_id, kind, family_id FROM authority_credential_secrets"
                " WHERE credential_id=?",
                (credential_id,),
            )
            if row is None:
                return None
            return {"subject_id": row[0], "tenant_id": row[1], "kind": row[2], "family_id": row[3]}

    def pepper(self) -> bytes:
        """A random server secret, created once, for deriving idempotent successors."""

        import secrets as _secrets

        with self._guard(), self._lock, self._txn() as c:
            self._x(
                c,
                "INSERT INTO authority_pepper(id, value) VALUES(1, ?) ON CONFLICT (id) DO NOTHING",
                (_secrets.token_hex(32),),
            )
            return bytes.fromhex(self._one(c, "SELECT value FROM authority_pepper WHERE id=1")[0])

    def active_family_count(self, subject_id: str) -> int:
        with self._guard(), self._read_lock, self._read_txn() as c:
            return self._active_families(c, subject_id)

    def stored_secret_hashes(self) -> list[str]:
        """Hashes at rest (for tests proving no plaintext secret is stored)."""

        with self._guard(), self._read_lock, self._read_txn() as c:
            return [r[0] for r in self._x(c, "SELECT secret_hash FROM authority_credential_secrets").fetchall()]

    # ── membership: invites, join, list, remove, onboarding policy ───────
    def create_invite(
        self,
        *,
        tenant_id: str,
        created_by: str,
        invite_id: str,
        secret_hash: str,
        expires_at: float,
        max_outstanding: int | None = None,
    ) -> None:
        """Record a single-use invite for ``tenant_id``, created by an active member of it.

        ``max_outstanding`` bounds the tenant's unused, unexpired, unrevoked
        invites (``TooManyInvites``).  The count is exact within one inviter; two
        inviters racing on PostgreSQL can each pass the check, so the bound is
        soft by at most the number of concurrent inviters.
        """

        for name, value in (("tenant_id", tenant_id), ("created_by", created_by), ("invite_id", invite_id)):
            _ident(name, value)
        now = self._clock()
        if expires_at <= now:
            raise ValueError("an invite must expire in the future")
        with self._guard(), self._lock, self._txn() as c:
            creator = self._one(
                c, "SELECT tenant_id, status FROM authority_subjects WHERE subject_id=?" + self._update_lock,
                (created_by,),
            )
            if creator is None or creator[1] != "active" or creator[0] != tenant_id:
                raise ValueError("the inviter must be an active member of the tenant")
            if max_outstanding is not None:
                outstanding = self._one(
                    c,
                    "SELECT COUNT(*) FROM authority_invites WHERE tenant_id=? AND used_at IS NULL"
                    " AND revoked_at IS NULL AND expires_at>?",
                    (tenant_id, now),
                )[0]
                if outstanding >= max_outstanding:
                    raise TooManyInvites("too many outstanding invites")
            self._x(
                c,
                "INSERT INTO authority_invites(invite_id,tenant_id,secret_hash,created_by,created_at,expires_at)"
                " VALUES(?,?,?,?,?,?)",
                (invite_id, tenant_id, secret_hash, created_by, now, expires_at),
            )

    def invite_tenant(self, invite_id: str, secret_hash: str) -> str | None:
        """The tenant of a usable invite when the secret matches, else ``None``."""

        with self._guard(), self._read_lock, self._read_txn() as c:
            row = self._one(
                c,
                "SELECT tenant_id, secret_hash, used_at, revoked_at, expires_at FROM authority_invites"
                " WHERE invite_id=?",
                (invite_id,),
            )
            if row is None or not hmac.compare_digest(row[1], secret_hash):
                return None
            if row[2] is not None or row[3] is not None or row[4] <= self._clock():
                return None
            return row[0]

    def list_invites(self, tenant_id: str) -> list[dict[str, Any]]:
        """The tenant's outstanding invites (unused, unrevoked, unexpired), oldest first.

        Identifiers and times only: neither the secret nor its hash leaves the store.
        """

        with self._guard(), self._read_lock, self._read_txn() as c:
            rows = self._x(
                c,
                "SELECT invite_id, created_by, created_at, expires_at FROM authority_invites"
                " WHERE tenant_id=? AND used_at IS NULL AND revoked_at IS NULL AND expires_at>?"
                " ORDER BY created_at, invite_id",
                (tenant_id, self._clock()),
            ).fetchall()
        return [
            {"invite_id": r[0], "created_by": r[1], "created_at": r[2], "expires_at": r[3]} for r in rows
        ]

    def revoke_invite(self, invite_id: str, *, tenant_id: str) -> bool:
        with self._guard(), self._lock, self._txn() as c:
            row = self._one(
                c,
                "SELECT used_at, revoked_at FROM authority_invites WHERE invite_id=? AND tenant_id=?"
                + self._update_lock,
                (invite_id, tenant_id),
            )
            if row is None or row[0] is not None or row[1] is not None:
                return False
            self._x(c, "UPDATE authority_invites SET revoked_at=? WHERE invite_id=?", (self._clock(), invite_id))
            return True

    def _append_statements(self, c: Any, tenant_id: str, statements: Sequence[PolicyStatement]) -> int:
        """Append statements to the tenant policy (idempotent by id) and bump the epoch."""

        self._x(
            c,
            "INSERT INTO authority_policies(tenant_id, policy_epoch) VALUES(?, 0) ON CONFLICT (tenant_id) DO NOTHING",
            (tenant_id,),
        )
        epoch = self._one(
            c,
            "SELECT policy_epoch FROM authority_policies WHERE tenant_id=?" + self._update_lock,
            (tenant_id,),
        )[0]
        existing = self._x(
            c, "SELECT statement_id, body FROM authority_statements WHERE tenant_id=? ORDER BY ordinal",
            (tenant_id,),
        ).fetchall()
        known = {row[0]: row[1] for row in existing}
        fresh = []
        for statement in statements:
            body = statement.model_dump_json()
            if statement.id in known:
                if known[statement.id] != body:
                    raise ValueError("a different statement already has this id")
                continue
            fresh.append(statement)
        merged = [PolicyStatement.model_validate_json(row[1]) for row in existing] + fresh
        AccessPolicy(  # the whole set must still be a valid snapshot for this tenant
            schema_version="candidate-v1", tenant_id=tenant_id, policy_epoch=0,
            membership_versions=(), statements=tuple(merged),
        )
        base = len(existing)
        for offset, statement in enumerate(fresh):
            self._x(
                c,
                "INSERT INTO authority_statements(tenant_id,statement_id,ordinal,body) VALUES(?,?,?,?)",
                (tenant_id, statement.id, base + offset, statement.model_dump_json()),
            )
        self._x(c, "UPDATE authority_policies SET policy_epoch=? WHERE tenant_id=?", (epoch + 1, tenant_id))
        return epoch + 1

    def grant_statements(self, tenant_id: str, statements: Sequence[PolicyStatement]) -> int:
        """Atomically add statements to a tenant's policy; returns the new epoch."""

        _ident("tenant_id", tenant_id)
        with self._guard(), self._lock, self._txn() as c:
            return self._append_statements(c, tenant_id, statements)

    def join_with_invite(
        self,
        *,
        invite_id: str,
        secret_hash: str,
        alias_issuer: str,
        alias_subject: str,
        subject_id: str,
        credential_id: str,
        statements_for: Callable[[str, str], Sequence[PolicyStatement]],
    ) -> tuple[str, int]:
        """Consume an invite and create the member, binding and default grants atomically.

        One transaction: the invite is single use and unexpired, the identity is
        not already bound, the subject, credential and binding are created, the
        default statements are appended (epoch incremented), and the invite is
        marked used.  Returns ``(tenant_id, new policy epoch)``.  ``ValueError`` for
        an unusable invite or an identity that is already a member.
        """

        for name, value in (("subject_id", subject_id), ("credential_id", credential_id),
                            ("issuer", alias_issuer), ("provider_subject", alias_subject)):
            _ident(name, value)
        now = self._clock()
        with self._guard(), self._lock, self._txn() as c:
            row = self._one(
                c,
                "SELECT tenant_id, secret_hash, used_at, revoked_at, expires_at FROM authority_invites"
                " WHERE invite_id=?" + self._update_lock,
                (invite_id,),
            )
            if (row is None or not hmac.compare_digest(row[1], secret_hash)
                    or row[2] is not None or row[3] is not None or row[4] <= now):
                raise ValueError("invite is not usable")
            tenant_id = row[0]
            if self._one(
                c, "SELECT 1 FROM authority_bindings WHERE issuer=? AND provider_subject=?",
                (alias_issuer, alias_subject),
            ):
                raise ValueError("identity is already a member")
            if self._one(c, "SELECT 1 FROM authority_subjects WHERE subject_id=?", (subject_id,)):
                raise ValueError("subject id already exists")
            if self._one(c, "SELECT 1 FROM authority_credentials WHERE credential_id=?", (credential_id,)):
                raise ValueError("credential id already exists")
            self._x(
                c,
                "INSERT INTO authority_bindings(issuer,provider_subject,subject_id,created_at) VALUES(?,?,?,?)",
                (alias_issuer, alias_subject, subject_id, now),
            )
            self._x(
                c,
                "INSERT INTO authority_subjects(subject_id,tenant_id,subject_kind,credential_id,"
                "delegation,membership_version,status,created_at,updated_at)"
                " VALUES(?,?,'human',?,NULL,1,'active',?,?)",
                (subject_id, tenant_id, credential_id, now, now),
            )
            self._x(
                c,
                "INSERT INTO authority_credentials(credential_id,subject_id,status,expires_at,created_at)"
                " VALUES(?,?,'active',NULL,?)",
                (credential_id, subject_id, now),
            )
            epoch = self._append_statements(c, tenant_id, tuple(statements_for(subject_id, tenant_id)))
            self._x(
                c, "UPDATE authority_invites SET used_at=?, used_by=? WHERE invite_id=?",
                (now, subject_id, invite_id),
            )
            return tenant_id, epoch

    def list_members(self, tenant_id: str) -> list[dict[str, Any]]:
        with self._guard(), self._read_lock, self._read_txn() as c:
            rows = self._x(
                c,
                "SELECT subject_id, subject_kind, membership_version, created_at FROM authority_subjects"
                " WHERE tenant_id=? AND status='active'"
                " AND subject_id NOT IN (SELECT subject_id FROM authority_service_accounts WHERE status='disabled')"
                " ORDER BY created_at, subject_id",
                (tenant_id,),
            ).fetchall()
            return [
                {"subject_id": r[0], "kind": r[1], "membership_version": r[2], "created_at": r[3]}
                for r in rows
            ]

    def remove_member(self, tenant_id: str, subject_id: str) -> int:
        """Remove a member of ``tenant_id``: offboard them and drop their statements.

        One transaction.  Returns the new policy epoch.  ``UnknownSubject`` when
        the subject is not an active member of that tenant.
        """

        # Exact strings: the checks below compare in Python, where a str subclass can claim equality.
        _ident("tenant_id", tenant_id)
        _ident("subject_id", subject_id)
        now = self._clock()
        with self._guard(), self._lock, self._txn() as c:
            row = self._one(
                c, "SELECT tenant_id, status FROM authority_subjects WHERE subject_id=?" + self._update_lock,
                (subject_id,),
            )
            if row is None or row[1] != "active" or row[0] != tenant_id:
                raise UnknownSubject(subject_id)
            # The tenant's policy head is the serialization point: concurrent removals queue
            # here, and each one checks the administrators that are left AFTER the earlier
            # ones committed.
            self._x(
                c,
                "INSERT INTO authority_policies(tenant_id, policy_epoch) VALUES(?, 0) ON CONFLICT (tenant_id) DO NOTHING",
                (tenant_id,),
            )
            epoch = self._one(
                c, "SELECT policy_epoch FROM authority_policies WHERE tenant_id=?" + self._update_lock,
                (tenant_id,),
            )[0]
            bodies = self._x(
                c, "SELECT statement_id, body FROM authority_statements WHERE tenant_id=?", (tenant_id,)
            ).fetchall()
            statements = [(sid, PolicyStatement.model_validate_json(body)) for sid, body in bodies]
            admins = {
                st.subject_id for _, st in statements
                if st.effect == "allow" and st.action == "manage_membership" and st.condition is None
            }
            if subject_id in admins:
                active_humans = {
                    r[0] for r in self._x(
                        c,
                        "SELECT subject_id FROM authority_subjects WHERE tenant_id=? AND status='active'"
                        " AND subject_kind='human'",
                        (tenant_id,),
                    ).fetchall()
                }
                if not (admins & active_humans) - {subject_id}:
                    raise LastAdministrator("a tenant must keep at least one administrator")
            self._offboard(c, subject_id, now)
            for statement_id, statement in statements:
                if statement.subject_id == subject_id:
                    self._x(
                        c, "DELETE FROM authority_statements WHERE tenant_id=? AND statement_id=?",
                        (tenant_id, statement_id),
                    )
            self._x(c, "UPDATE authority_policies SET policy_epoch=? WHERE tenant_id=?", (epoch + 1, tenant_id))
            return epoch + 1

    def set_onboarding_policy(self, tenant_id: str, version: str, body: str) -> None:
        _ident("tenant_id", tenant_id)
        with self._guard(), self._lock, self._txn() as c:
            self._x(
                c,
                "INSERT INTO authority_onboarding(tenant_id,version,body,updated_at) VALUES(?,?,?,?)"
                " ON CONFLICT (tenant_id) DO UPDATE SET version=excluded.version, body=excluded.body,"
                " updated_at=excluded.updated_at",
                (tenant_id, version, body, self._clock()),
            )

    def get_onboarding_policy(self, tenant_id: str) -> tuple[str, str] | None:
        with self._guard(), self._read_lock, self._read_txn() as c:
            row = self._one(
                c, "SELECT version, body FROM authority_onboarding WHERE tenant_id=?", (tenant_id,)
            )
            return (row[0], row[1]) if row else None

    # ── atomic check-and-publish ─────────────────────────────────────────
    def hold_current(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        credential_id: str,
        membership_version: int,
        policy_epoch: int,
        body: Callable[[Any], T],
    ) -> T:
        """Verify the witnessed authority is unchanged, then run ``body(conn)``.

        Verification and ``body`` run in ONE transaction.  The rows that
        authorize the work (tenant policy, subject, credential) are locked
        (PostgreSQL ``FOR SHARE``; SQLite's single writer) until it commits, so a
        concurrent publish, version bump or retirement either happened before
        (``WitnessStale``) or waits until this commit.  Whatever ``body`` writes
        through the connection commits atomically with that verification; it
        can only guarantee more if the data it writes lives in this database.
        """

        # Exact types: every check below compares in Python, where a subclass can claim equality.
        for name, value in (("tenant_id", tenant_id), ("subject_id", subject_id), ("credential_id", credential_id)):
            _ident(name, value)
        for name, value in (("membership_version", membership_version), ("policy_epoch", policy_epoch)):
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        with self._guard(), self._lock, self._txn() as c:
            row = self._one(
                c,
                "SELECT policy_epoch FROM authority_policies WHERE tenant_id=?" + self._share,
                (tenant_id,),
            )
            if row is None or row[0] != policy_epoch:
                raise WitnessStale("policy_epoch_changed")
            subject = self._one(
                c,
                "SELECT tenant_id, credential_id, membership_version, status FROM authority_subjects"
                " WHERE subject_id=?" + self._share,
                (subject_id,),
            )
            if subject is None or subject[3] != "active" or subject[0] != tenant_id:
                raise WitnessStale("subject_not_active")
            if subject[2] != membership_version:
                raise WitnessStale("membership_changed")
            credential = self._one(
                c,
                "SELECT status, expires_at, subject_id FROM authority_credentials WHERE credential_id=?"
                + self._share,
                (credential_id,),
            )
            if credential is None:
                raise WitnessStale("credential_retired")
            if credential[2] != subject_id:
                raise WitnessStale("credential_subject_mismatch")
            if credential[0] != "active":
                raise WitnessStale("credential_retired")
            if credential[1] is not None and credential[1] <= self._clock():
                raise WitnessStale("credential_expired")
            return body(c)


def _ident(name: str, value: object) -> str:
    """An identifier accepted identically by every backend.

    One to 256 characters, no control characters (a NUL is stored by SQLite and
    rejected by PostgreSQL, which would otherwise surface as an outage).
    """

    if type(value) is not str or not 1 <= len(value) <= 256:
        raise ValueError(f"{name} must be 1..256 characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _dump(delegation: DelegationConstraints | None) -> str | None:
    return None if delegation is None else delegation.model_dump_json()


def _load_delegation(text: str | None) -> DelegationConstraints | None:
    return None if text is None else DelegationConstraints.model_validate_json(text)


_logger = logging.getLogger(__name__)
_SQLITE_SIDECARS = ("", "-wal", "-shm", "-journal")


def _owner_only(path: str) -> None:
    """Create the database file owner-only if it is missing, and tighten it and its sidecars if looser.

    The authority database holds invite and credential secret hashes and the credential pepper.  SQLite
    creates its WAL and shared-memory files with the database file's mode, so the main file is created
    0600 before the first connection; files left looser by an earlier version are tightened here.
    """

    if path == ":memory:" or path.startswith("file:"):
        return
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)  # 0600 & ~umask when it is created now
        os.close(descriptor)
    except OSError:
        return  # let sqlite report the path
    for suffix in _SQLITE_SIDECARS:
        target = path + suffix
        try:
            mode = stat.S_IMODE(os.stat(target).st_mode)
            if mode & 0o077:
                os.chmod(target, mode & 0o700)
        except FileNotFoundError:
            continue
        except OSError:
            _logger.warning("the authority database file %s could not be made owner-only", target)


class SqliteAuthorityStore(SqlAuthorityStore):
    """Embedded profile: one owning host, local disk, WAL, ``synchronous=FULL``."""

    dialect = "sqlite"
    _errors = (sqlite3.Error,)
    _unique_errors = (sqlite3.IntegrityError,)

    def __init__(
        self,
        path: str,
        *,
        clock: Callable[[], float] = time.time,
        busy_timeout_ms: int = 30_000,
    ) -> None:
        super().__init__(clock)
        _owner_only(path)
        self._conn = sqlite3.connect(
            path, isolation_level=None, timeout=busy_timeout_ms / 1000, check_same_thread=False
        )
        mode = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if mode.lower() != "wal":
            raise SchemaError(f"WAL is required (journal_mode={mode})")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        self._migrate()
        # Reads use their own connection: WAL readers never wait for the writer,
        # so a slow publication does not stall every authority read.
        self._rconn = sqlite3.connect(
            path, isolation_level=None, timeout=busy_timeout_ms / 1000, check_same_thread=False
        )
        _owner_only(path)  # sidecars left looser by an earlier version (new ones follow the main file)
        self._rconn.execute("PRAGMA query_only=ON")
        self._rconn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")

    @contextlib.contextmanager
    def _txn(self) -> Iterator[Any]:
        if self._closed:
            raise sqlite3.ProgrammingError("authority store is closed")
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    @contextlib.contextmanager
    def _read_txn(self) -> Iterator[Any]:
        if self._closed:
            raise sqlite3.ProgrammingError("authority store is closed")
        self._rconn.execute("BEGIN")
        try:
            yield self._rconn
        finally:
            self._rconn.execute("ROLLBACK")


class PostgresAuthorityStore(SqlAuthorityStore):
    """Hosted-cell profile: one database per tenant cell."""

    dialect = "postgres"
    _share = " FOR SHARE"
    _update_lock = " FOR UPDATE"

    def __init__(
        self,
        dsn: str,
        *,
        clock: Callable[[], float] = time.time,
        connect_timeout: int = 10,
        migrate: bool = True,
    ) -> None:
        """``migrate=True`` is the owner connection (it applies migrations).

        ``migrate=False`` is the least-privilege runtime connection: the role must not be a superuser or
        BYPASSRLS, and the schema must already be at this code's version (an owner ran the migrations
        and granted the role with :func:`grant_authority_runtime_role`).
        """

        import psycopg

        super().__init__(clock)
        self._errors = (psycopg.Error,)
        self._unique_errors = (psycopg.errors.UniqueViolation,)
        self._dsn, self._connect_timeout = dsn, connect_timeout
        self._suspect: set[str] = set()
        self._conn = psycopg.connect(dsn, autocommit=True, connect_timeout=connect_timeout)
        try:
            if migrate:
                self._migrate()
            else:
                self._verify_runtime_role()
                self._verify_schema_version()
        except BaseException:
            self._conn.close()
            raise
        try:
            self._rconn = psycopg.connect(dsn, autocommit=True, connect_timeout=connect_timeout)
        except BaseException:
            self._conn.close()
            raise

    def _live(self, name: str) -> Any:
        """The named connection, reconnecting if it was lost (failing closed if it cannot)."""

        import psycopg

        if self._closed:
            raise psycopg.OperationalError("authority store is closed")
        conn = getattr(self, name)
        if conn is not None and name in self._suspect and not (conn.closed or conn.broken):
            self._suspect.discard(name)
            try:
                conn.execute("SELECT 1")  # a failure marked it suspect: prove it still works
            except Exception:  # noqa: BLE001 - replaced below
                pass
        if conn is None or conn.closed or conn.broken:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001 - already broken
                    pass
            conn = psycopg.connect(self._dsn, autocommit=True, connect_timeout=self._connect_timeout)
            setattr(self, name, conn)
        return conn

    def _on_failure(self) -> None:
        self._suspect.update({"_conn", "_rconn"})

    def _verify_runtime_role(self) -> None:
        row = self._conn.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
        if row is None or row[0] or row[1]:
            raise AuthorityRuntimeError("runtime role must not be superuser or BYPASSRLS")
        from ledger.postgres import owner_privilege_problem

        problem = owner_privilege_problem(self._conn)
        if problem is not None:
            raise AuthorityRuntimeError(problem)

    def _verify_schema_version(self) -> None:
        import psycopg

        expected = max(m.version for m in MIGRATIONS)
        try:
            found = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM authority_schema_version"
            ).fetchone()[0]
        except psycopg.errors.UndefinedTable as exc:
            from ledger.postgres import ungranted_schemas

            if ungranted_schemas(self._conn):
                raise AuthorityRuntimeError("the runtime role was not granted the authority schema") from exc
            raise AuthorityRuntimeError(
                "authority schema is not installed; run the migrations as the owner role"
            ) from exc
        except psycopg.errors.InsufficientPrivilege as exc:
            raise AuthorityRuntimeError("the runtime role was not granted the authority schema") from exc
        if found != expected:
            raise AuthorityRuntimeError(f"authority schema version {found} does not match code version {expected}")

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s")

    def _lock_for_migration(self, c: Any) -> None:
        c.execute("SELECT pg_advisory_xact_lock(%s)", (_PG_MIGRATION_LOCK,))

    @contextlib.contextmanager
    def _txn(self) -> Iterator[Any]:
        conn = self._live("_conn")
        with conn.transaction():
            yield conn

    @contextlib.contextmanager
    def _read_txn(self) -> Iterator[Any]:
        conn = self._live("_rconn")
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            yield conn


def build_authority_store(settings: Any) -> SqlAuthorityStore | None:
    """Build the configured store, or ``None`` when ``AUTHORITY_ENABLED`` is off."""

    if not settings.authority_enabled:
        return None
    if settings.authority_backend == "sqlite":
        if not settings.authority_sqlite_path:
            raise ValueError("AUTHORITY_SQLITE_PATH is required for the sqlite backend")
        return SqliteAuthorityStore(settings.authority_sqlite_path)
    if settings.authority_backend == "postgres":
        if not settings.authority_postgres_dsn:
            raise ValueError("AUTHORITY_POSTGRES_DSN is required for the postgres backend")
        return PostgresAuthorityStore(
            settings.authority_postgres_dsn, migrate=getattr(settings, "authority_postgres_migrate_on_open", True)
        )
    raise ValueError("unsupported authority backend")


#: Least privileges the authority store's runtime role needs, per table.
_AUTHORITY_DML = {
    "authority_bindings": "SELECT, INSERT",
    "authority_subjects": "SELECT, INSERT, UPDATE",
    "authority_credentials": "SELECT, INSERT, UPDATE",
    "authority_policies": "SELECT, INSERT, UPDATE",
    "authority_statements": "SELECT, INSERT, DELETE",
    "authority_schema_version": "SELECT",
}


class AuthorityRuntimeError(RuntimeError):
    """The runtime connection cannot be used (privileged role, or schema not installed or not granted)."""


def authority_installation_problems(owner_conn: Any, runtime_role: str | None = None) -> list[str]:
    """Problems with the authority schema version and, when given, the runtime role's privileges.

    Run as the owner (read only).  The runtime role must exist, must not be a superuser, BYPASSRLS, the schema
    owner or a member of it, and must hold exactly the declared table privileges, no more and no less.
    """

    problems: list[str] = []
    expected = max(m.version for m in MIGRATIONS)
    found = owner_conn.execute("SELECT COALESCE(MAX(version), 0) FROM authority_schema_version").fetchone()[0]
    if found != expected:
        problems.append(f"authority schema version {found} != code version {expected}")
    if not runtime_role:
        return problems
    schema = owner_conn.execute("SELECT current_schema()").fetchone()[0]
    role = owner_conn.execute(
        "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname=%s", (runtime_role,)
    ).fetchone()
    if role is None:
        return problems + [f"role {runtime_role} does not exist"]
    if role[0] or role[1]:
        problems.append(f"role {runtime_role} is superuser or BYPASSRLS")
    if owner_conn.execute(
        "SELECT pg_has_role(%s, n.nspowner, 'MEMBER') FROM pg_namespace n WHERE n.nspname = %s",
        (runtime_role, schema),
    ).fetchone()[0]:
        problems.append(f"role {runtime_role} owns the authority schema or is a member of its owner")
    for table, wanted in _AUTHORITY_DML.items():
        want = {p.strip() for p in wanted.split(",")}
        for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE"):
            held = owner_conn.execute(
                "SELECT has_table_privilege(%s, %s, %s)", (runtime_role, f"{schema}.{table}", privilege)
            ).fetchone()[0]
            if held and privilege not in want:
                problems.append(f"role {runtime_role} holds {privilege} on {table}")
            if not held and privilege in want:
                problems.append(f"role {runtime_role} lacks {privilege} on {table}")
    return problems


def grant_authority_runtime_role(owner_conn: Any, role: str) -> None:
    """Grant a runtime role exactly what the store's runtime path uses; run as the owner role.

    ``role`` must already exist (created by the deployment, never here).
    """

    from psycopg import sql as pgsql

    ident = pgsql.Identifier(role)
    schema = owner_conn.execute("SELECT current_schema()").fetchone()[0]
    owner_conn.execute(pgsql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(pgsql.Identifier(schema), ident))
    for table, privileges in _AUTHORITY_DML.items():
        owner_conn.execute(
            pgsql.SQL("GRANT " + privileges + " ON {} TO {}").format(pgsql.Identifier(table), ident)
        )


__all__ = [
    "AuthorityRuntimeError",
    "CredentialInvalid",
    "CredentialLimit",
    "MIGRATIONS",
    "RefreshReplay",
    "VerifiedCredential",
    "PolicyConflict",
    "PostgresAuthorityStore",
    "SqlAuthorityStore",
    "SqliteAuthorityStore",
    "UnknownSubject",
    "WitnessStale",
    "build_authority_store",
    "authority_installation_problems",
    "grant_authority_runtime_role",
    "plan_authority_migrations",
]
