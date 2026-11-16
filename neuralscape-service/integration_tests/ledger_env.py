"""Store addresses and missing-infrastructure handling for the durable write path (ledger) integration tests.

The addresses are read from the write-path-specific variables first and from the shared integration
variables second, so the same tests run by hand and on the gate:

  Redis        V1_TEST_REDIS_URL | LEDGER_TEST_REDIS_URL | REDIS_URL (then database 14 of that server)
  PostgreSQL   LEDGER_TEST_PG_DSN | NS_TEST_PG_DSN
  Qdrant, Neo4j and the engine PostgreSQL: see ``engine_env``

The Redis these tests use is FLUSHED (``flushdb``) to model a queue that lost its wake-up.  Only a database
nobody else uses may be flushed, so when the address comes from the shared ``REDIS_URL`` the tests move to
database 14 of that server; a ``V1_TEST_REDIS_URL`` / ``LEDGER_TEST_REDIS_URL`` is taken exactly as given.

Missing infrastructure FAILS under ``NS_INTEGRATION_REQUIRED`` (the gate) and skips elsewhere
(``engine_env.require_env``).
"""

from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse

from engine_env import NEO4J_PASSWORD, NEO4J_URI, PG_DSN as ENGINE_PG_DSN, QDRANT_URL, _convention, _first

SHARED_REDIS_DB = 14


def _redis_url() -> str | None:
    explicit = _first("V1_TEST_REDIS_URL", "LEDGER_TEST_REDIS_URL")
    if explicit:
        return explicit
    shared = os.environ.get("REDIS_URL")
    if not shared:
        return None
    parsed = urlparse(shared)
    return urlunparse(parsed._replace(path=f"/{SHARED_REDIS_DB}"))


def require_env(**configured: object) -> None:
    """Fail (gate) or skip (elsewhere) unless every named store is configured: ``require_env(Redis=REDIS_URL)``."""

    missing = [name for name, value in configured.items() if not value]
    if missing:
        _convention().unavailable(f"{', '.join(missing)} not configured for the write-path integration tests")


REDIS_URL = _redis_url()
LEDGER_PG_DSN = _first("LEDGER_TEST_PG_DSN", "NS_TEST_PG_DSN")

__all__ = [
    "ENGINE_PG_DSN",
    "LEDGER_PG_DSN",
    "NEO4J_PASSWORD",
    "NEO4J_URI",
    "QDRANT_URL",
    "REDIS_URL",
    "require_env",
]
