"""Convention for flag-on integration tests (see README.md in this directory).

Everything under ``integration_tests/`` is automatically marked
``pytest.mark.integration``.  CI never collects this directory through
``pytest tests/`` (and the one job that runs the snapshot qualification names
its file explicitly), so tests here may need real Neo4j / Qdrant / Redis and
an LLM key without ever meeting the unit-suite skip gate.

Missing infrastructure is handled in one place:

* ``NS_INTEGRATION_REQUIRED=1`` (the VPS gate sets it): an unreachable store
  or a missing key FAILS the test.
* otherwise the test SKIPS with the reason, so a developer without stores
  gets a clear message instead of a stack trace.

Fixtures are opt-in: a test that asks for none of them (the pre-existing
``test_native_snapshot_neo4j.py``) is untouched apart from the marker.
Module level imports are stdlib + pytest only, so this file is importable
without the service on ``sys.path``.
"""

from __future__ import annotations

import os
import socket
import sys
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import pytest

INTEGRATION_DIR = Path(__file__).resolve().parent
SERVICE_ROOT = INTEGRATION_DIR.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

PROBE_TIMEOUT_S = 3.0


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: integration tests requiring running services (Redis, Qdrant, Neo4j, API, Worker)",
    )


def pytest_collection_modifyitems(config, items):
    """Mark every test collected from this directory ``integration``.

    A conftest hook sees the whole session's items, so filter by location.
    """

    for item in items:
        try:
            path = Path(str(item.path)).resolve()
        except Exception:  # noqa: BLE001 - never break collection over a marker
            continue
        if INTEGRATION_DIR == path.parent or INTEGRATION_DIR in path.parents:
            item.add_marker(pytest.mark.integration)


# --------------------------------------------------------------------------
# missing infrastructure: fail on the gate, skip elsewhere
# --------------------------------------------------------------------------


def integration_required() -> bool:
    return os.environ.get("NS_INTEGRATION_REQUIRED", "").strip().lower() in {"1", "true", "yes", "on"}


def unavailable(reason: str) -> None:
    """Fail (gate) or skip (anywhere else) because infrastructure is missing."""

    if integration_required():
        pytest.fail(f"NS_INTEGRATION_REQUIRED is set but {reason}", pytrace=False)
    pytest.skip(reason)


def _tcp_reachable(host: str, port: int) -> str | None:
    """None when a TCP connection succeeds, else a short error string."""

    try:
        with socket.create_connection((host, port), timeout=PROBE_TIMEOUT_S):
            return None
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"


def _probe_neo4j(uri: str) -> str | None:
    parsed = urlparse(uri)
    return _tcp_reachable(parsed.hostname or "127.0.0.1", parsed.port or 7687)


def _probe_redis(url: str) -> str | None:
    parsed = urlparse(url)
    return _tcp_reachable(parsed.hostname or "localhost", parsed.port or 6379)


def _probe_qdrant(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/collections", timeout=PROBE_TIMEOUT_S) as resp:  # noqa: S310
            return None if resp.status < 500 else f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return None if exc.code < 500 else f"HTTP {exc.code}"  # 401/403 means up but keyed
    except (OSError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class Stores:
    """Where the backing stores are.  Passwords and keys are never printed."""

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str
    qdrant_url: str
    redis_url: str

    def __repr__(self) -> str:  # keep secrets out of assertion output
        return f"Stores(neo4j={self.neo4j_uri}, qdrant={self.qdrant_url}, redis={self.redis_url})"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value else default


@pytest.fixture(scope="session")
def stores() -> Stores:
    """Store addresses from env, each probed once.  Unreachable -> fail/skip.

    Env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, QDRANT_URL,
    REDIS_URL.  QDRANT_URL has no default (the service falls back to an
    on-disk store without it, which is not what an integration test means).
    """

    qdrant_url = _env("QDRANT_URL")
    if not qdrant_url:
        unavailable("QDRANT_URL is not set")
    found = Stores(
        neo4j_uri=_env("NEO4J_URI", "neo4j://127.0.0.1:7687"),
        neo4j_user=_env("NEO4J_USER", "neo4j"),
        neo4j_password=_env("NEO4J_PASSWORD", ""),
        neo4j_database=_env("NEO4J_DATABASE", "memory"),
        qdrant_url=qdrant_url,
        redis_url=_env("REDIS_URL", "redis://localhost:6379"),
    )
    for name, error in (
        ("Neo4j", _probe_neo4j(found.neo4j_uri)),
        ("Qdrant", _probe_qdrant(found.qdrant_url)),
        ("Redis", _probe_redis(found.redis_url)),
    ):
        if error:
            unavailable(f"{name} is unreachable ({error})")
    return found


@pytest.fixture(scope="session")
def llm_key() -> str:
    """Whether a Gemini key is present.  Only tests that spend LLM calls ask for it.

    Returns the key so a test may pass it on; never print it.
    """

    key = _env("GOOGLE_API_KEY")
    if not key or key == "ci-dummy":
        unavailable("GOOGLE_API_KEY is not set to a real key")
    return key


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    """DSN of a PostgreSQL with pgvector (``NS_TEST_PG_DSN``), probed once.

    The gate also sets LEDGER_TEST_PG_DSN, ENGINE_TEST_PG_DSN,
    USAGE_TEST_PG_DSN and AUTHORITY_TEST_PG_DSN to the same value for the
    Phase 2 slices that read those names.  Use ``unique_ns`` names (a schema
    or table prefix) so tests never share rows.
    """

    dsn = _env("NS_TEST_PG_DSN")
    if not dsn:
        unavailable("NS_TEST_PG_DSN is not set")
    parsed = urlparse(dsn)
    error = _tcp_reachable(parsed.hostname or "localhost", parsed.port or 5432)
    if error:
        unavailable(f"PostgreSQL is unreachable ({error})")
    return dsn


@dataclass(frozen=True)
class LiveApi:
    """A running API (with all workers) and the credentials to call it."""

    url: str
    headers: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # never show the token
        return f"LiveApi(url={self.url})"


@pytest.fixture(scope="session")
def live_api() -> LiveApi:
    """The live API from ``NS_API_URL`` plus ``Authorization: Bearer $NS_API_TOKEN``.

    The token is minted by the gate for a per-run user; use that user's id in
    requests.  Probed with ``GET /health``.
    """

    url = _env("NS_API_URL")
    if not url:
        unavailable("NS_API_URL is not set")
    url = url.rstrip("/")
    try:
        with urllib.request.urlopen(url + "/health", timeout=PROBE_TIMEOUT_S) as resp:  # noqa: S310
            error = None if resp.status == 200 else f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        error = f"HTTP {exc.code}"
    except (OSError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    if error:
        unavailable(f"the API at {url} is unreachable ({error})")
    token = _env("NS_API_TOKEN")
    return LiveApi(url=url, headers={"Authorization": f"Bearer {token}"} if token else {})


# --------------------------------------------------------------------------
# turning a feature flag on for one test
# --------------------------------------------------------------------------


def _coerce(current: Any, value: Any) -> Any:
    """Match the type of the existing Settings field (bool/int/float/str)."""

    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


@pytest.fixture
def flags(monkeypatch):
    """Turn Settings fields on (or off) for ONE test, keyed by field name.

    ``flags(standards_enabled=True, dictator_user_ids="alice")`` does both:

    * ``monkeypatch.setenv("STANDARDS_ENABLED", "true")`` so any new
      ``Settings()`` (a subprocess, a re-imported module, a worker started by
      the test) sees it, and
    * ``monkeypatch.setattr(config.settings, field, value)`` on the
      process-wide ``config.settings`` singleton.  Modules that did
      ``from config import settings`` hold that same object, so the change is
      visible to them immediately.  This is the mechanism the unit suite uses.

    Caveat: a flag read ONCE at import time (copied into a module constant,
    a decorator argument, a default argument, a class attribute) is not
    changed by setattr.  For those, import the module after calling
    ``flags`` inside the test, or ``importlib.reload`` it, and say so in the
    test.  Unknown field names raise, so a typo cannot silently test
    nothing.  Everything is undone at teardown.
    """

    import config  # service root is on sys.path (see top of this file)

    def apply(**fields: Any) -> None:
        for name, value in fields.items():
            if not hasattr(config.settings, name):
                raise AttributeError(f"Settings has no field {name!r}")
            coerced = _coerce(getattr(config.settings, name), value)
            monkeypatch.setattr(config.settings, name, coerced)
            env_value = "true" if coerced is True else "false" if coerced is False else str(coerced)
            monkeypatch.setenv(name.upper(), env_value)

    return apply


# --------------------------------------------------------------------------
# isolation: unique names per test + cleanup
# --------------------------------------------------------------------------


@dataclass
class Namespace:
    """Names no other test (or earlier run) shares."""

    user_id: str
    project_id: str
    qdrant_collection: str
    group_prefix: str
    cleanups: list[Callable[[], Any]] = field(default_factory=list, repr=False)

    def on_cleanup(self, fn: Callable[[], Any]) -> None:
        """Register teardown work (delete the collection, DETACH DELETE the group...).

        Runs in reverse order after the test, even when it failed; an error in
        one cleanup does not stop the rest.
        """

        self.cleanups.append(fn)


@pytest.fixture
def unique_ns(request):
    """A unique user_id / project_id / Qdrant collection / Neo4j group prefix.

    Every name carries the same short random suffix and a ``nsit`` marker, so
    leftovers from a crashed run can be found and purged by prefix.  Write
    only inside these names; register cleanup with ``unique_ns.on_cleanup``.
    """

    token = uuid.uuid4().hex[:10]
    name = "".join(c if c.isalnum() else "_" for c in request.node.name.lower())[:24]
    ns = Namespace(
        user_id=f"nsit-user-{token}",
        project_id=f"nsit-proj-{token}",
        qdrant_collection=f"nsit_{name}_{token}",
        group_prefix=f"nsit_{token}",
    )
    yield ns
    errors = []
    for fn in reversed(ns.cleanups):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
    if errors:
        print(f"unique_ns cleanup errors (non-fatal): {errors}")
