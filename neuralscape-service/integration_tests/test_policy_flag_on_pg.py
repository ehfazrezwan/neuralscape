"""The flag-on policy tests of ``tests/test_policy_flag_on_integration.py`` against PostgreSQL.

The authority store is a real PostgreSQL 16 schema reached as the least-privilege runtime role
(``migrate=False``, granted by ``grant_authority_runtime_role``).  Needs ``AUTHORITY_TEST_PG_DSN``; with
``NS_INTEGRATION_REQUIRED=1`` its absence is a failure, otherwise a skip.
"""

import pytest

from tests.authority_pg_support import RuntimeRoleBackend, require_pg  # noqa: F401  (autouse)
from tests.test_policy_flag_on_integration import *  # noqa: F401,F403  (the tests, and their fixtures)

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    backend = RuntimeRoleBackend()
    yield backend.open()
    backend.close()


@pytest.fixture
def configured_store(monkeypatch):
    from config import settings

    backend = RuntimeRoleBackend()
    monkeypatch.setattr(settings, "authority_enabled", True)
    monkeypatch.setattr(settings, "authority_backend", "postgres")
    monkeypatch.setattr(settings, "authority_postgres_dsn", backend.runtime_dsn)
    monkeypatch.setattr(settings, "authority_postgres_migrate_on_open", False)
    yield backend.open()
    backend.close()
