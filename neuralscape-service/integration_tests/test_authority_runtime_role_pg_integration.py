"""The authority store as the least-privilege PostgreSQL runtime role (integration).

Needs ``AUTHORITY_TEST_PG_DSN`` (a disposable PostgreSQL 16 server; see ``authority_pg_support``).
Every backend is its own schema with a non-superuser owner role that migrates and grants, and a
non-superuser runtime role that the store connects as with ``migrate=False``.  Skipped when the
variable is unset; that skip is the declared one.

Before ``grant_authority_runtime_role`` covered every table, the runtime role had no privilege on
``authority_credential_secrets``, ``authority_service_accounts``, ``authority_pepper``,
``authority_invites`` and ``authority_onboarding``, so credentials, service accounts, invites and
onboarding all failed as an authority outage on a least-privilege deployment while the owner-role
conformance runs stayed green.
"""

from __future__ import annotations

import pytest

from tests.authority_pg_support import RuntimeRoleBackend, require_pg  # noqa: F401  (autouse)

pytestmark = pytest.mark.integration

psycopg = pytest.importorskip("psycopg")

from authority_store_sql import (  # noqa: E402
    MIGRATIONS,
    AuthorityRuntimeError,
    PostgresAuthorityStore,
    _AUTHORITY_DML,
    authority_installation_problems,
    grant_authority_runtime_role,
)
from tests.test_authority_store_sql import CASES as AUTHORITY_CASES  # noqa: E402
from tests.test_credentials import CASES as CREDENTIAL_CASES  # noqa: E402
from tests.test_onboarding import CASES as ONBOARDING_CASES  # noqa: E402

# Owner-only behaviour: it tampers with the schema-version table (the runtime role may only read it) and
# checks the owner's migration drift detection.  The runtime role's own version check is tested below.
_OWNER_ONLY = {"case_schema_is_versioned_idempotent_and_drift_checked"}

ALL_CASES = [
    *[("authority", case) for case in AUTHORITY_CASES if case.__name__ not in _OWNER_ONLY],
    *[("credentials", case) for case in CREDENTIAL_CASES],
    *[("onboarding", case) for case in ONBOARDING_CASES],
]


@pytest.fixture
def pg_backend():
    backend = RuntimeRoleBackend()
    yield backend
    backend.close()


@pytest.mark.parametrize("suite,case", ALL_CASES, ids=lambda v: v if isinstance(v, str) else v.__name__)
def test_conformance_as_the_runtime_role(suite, case, pg_backend) -> None:
    with pg_backend.raw() as conn:  # the suite really runs unprivileged
        assert conn.execute(
            "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname=current_user"
        ).fetchone()[0] is False
    case(pg_backend)


def test_the_runtime_role_holds_exactly_the_declared_privileges(pg_backend) -> None:
    with pg_backend.raw(as_owner=True) as owner:
        assert authority_installation_problems(owner, pg_backend.runtime) == []
        # Every table of the schema is in the declared set (a new table must declare its grant).
        tables = {
            row[0]
            for row in owner.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (pg_backend.schema,)
            ).fetchall()
        }
        assert tables == set(_AUTHORITY_DML)


def test_a_missing_grant_is_reported_by_the_installation_check(pg_backend) -> None:
    with pg_backend.raw(as_owner=True) as owner:
        owner.execute(f'REVOKE UPDATE ON authority_invites FROM "{pg_backend.runtime}"')
        problems = authority_installation_problems(owner, pg_backend.runtime)
        assert any("lacks UPDATE on authority_invites" in p for p in problems), problems
        owner.execute(f'GRANT DELETE ON authority_credential_secrets TO "{pg_backend.runtime}"')
        problems = authority_installation_problems(owner, pg_backend.runtime)
        assert any("holds DELETE on authority_credential_secrets" in p for p in problems), problems


def test_the_runtime_role_cannot_delete_truncate_or_change_the_schema(pg_backend) -> None:
    store = pg_backend.open()
    store.add_subject("sub_a", "tenant-a", "human", "cred-a")
    for statement in (
        "DELETE FROM authority_credential_secrets",
        "DELETE FROM authority_invites",
        "DELETE FROM authority_service_accounts",
        "DELETE FROM authority_subjects",
        "TRUNCATE authority_pepper",
        "UPDATE authority_pepper SET value = 'x'",  # the pepper is written once
        "DROP TABLE authority_onboarding",
        "CREATE TABLE authority_rogue (x INTEGER)",
    ):
        with pg_backend.raw() as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(statement)


def test_an_ungranted_runtime_role_is_refused_at_open_with_a_clear_reason() -> None:
    backend = RuntimeRoleBackend()
    try:
        with backend.raw(as_owner=True) as owner:
            owner.execute(f'REVOKE USAGE ON SCHEMA "{backend.schema}" FROM "{backend.runtime}"')
        with pytest.raises(AuthorityRuntimeError, match="not granted"):
            backend.open()
        # regranting repairs it (idempotent): the owner re-runs the grant
        owner_store = backend.open_owner()
        grant_authority_runtime_role(owner_store._conn, backend.runtime)
        assert backend.open() is not None
    finally:
        backend.close()


def test_the_schema_version_matches_the_code(pg_backend) -> None:
    with pg_backend.raw() as conn:
        found = conn.execute("SELECT MAX(version) FROM authority_schema_version").fetchone()[0]
    assert found == max(m.version for m in MIGRATIONS)


def test_a_schema_version_that_differs_from_the_code_is_refused_at_open(pg_backend) -> None:
    with pg_backend.raw(as_owner=True) as owner:
        owner.execute("INSERT INTO authority_schema_version VALUES (99, 'future', 'x', 0)")
    with pytest.raises(AuthorityRuntimeError, match="does not match"):
        pg_backend.open()
