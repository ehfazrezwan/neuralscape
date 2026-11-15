"""Store addresses and missing-infrastructure handling for the engine integration tests.

The addresses are read from the engine-specific variables first and from the shared
integration variables second, so the same tests run by hand (``ENGINE_TEST_*``) and on
the gate (``NS_TEST_PG_DSN``, ``QDRANT_URL``, ``NEO4J_URI``, ``NEO4J_PASSWORD``):

  PostgreSQL + pgvector   ENGINE_TEST_PG_DSN       or NS_TEST_PG_DSN
  Qdrant                  ENGINE_TEST_QDRANT_URL   or QDRANT_URL
  Neo4j                   ENGINE_TEST_NEO4J_URI    or NEO4J_URI
                          ENGINE_TEST_NEO4J_PASSWORD or NEO4J_PASSWORD (default ``neo4j``)

A test that needs a store that is not configured FAILS when ``NS_INTEGRATION_REQUIRED`` is
set (the gate) or on GitHub Actions, and SKIPS with the reason everywhere else.  It never
passes silently.  The stores must be disposable: the Neo4j tests wipe the database.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

# Every engine integration test imports this module first, so the vendored mem0/Graphiti telemetry is
# defaulted off before either library can load in an integration run (``tests/conftest.py`` does the same
# for the unit suites; this directory has no conftest of its own).
import third_party_telemetry  # noqa: E402,F401


def _first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


PG_DSN = _first("ENGINE_TEST_PG_DSN", "NS_TEST_PG_DSN")
QDRANT_URL = _first("ENGINE_TEST_QDRANT_URL", "QDRANT_URL")
NEO4J_URI = _first("ENGINE_TEST_NEO4J_URI", "NEO4J_URI")
NEO4J_PASSWORD = _first("ENGINE_TEST_NEO4J_PASSWORD", "NEO4J_PASSWORD") or "neo4j"


def integration_required() -> bool:
    if os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true":
        return True
    return os.environ.get("NS_INTEGRATION_REQUIRED", "").strip().lower() in {"1", "true", "yes", "on"}


def require_env(**configured: object) -> None:
    """Fail (gate) or skip (elsewhere) unless every named store is configured.

    Called as ``require_env(PostgreSQL=PG_DSN, Qdrant=QDRANT_URL)``; the keys only name the store in the message.
    """

    missing = [name for name, value in configured.items() if not value]
    if not missing:
        return
    reason = f"{', '.join(missing)} not configured for the engine integration tests"
    if integration_required():
        pytest.fail(f"NS_INTEGRATION_REQUIRED is set but {reason}", pytrace=False)
    pytest.skip(reason)
