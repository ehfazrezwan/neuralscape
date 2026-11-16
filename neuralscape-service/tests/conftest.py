"""Conftest for neuralscape-service tests — sys.path + auth isolation."""

import sys
from pathlib import Path

# Ensure the service root is importable (main.py, config.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Before anything can import mem0/Graphiti: keep their telemetry off in tests.
# Forced, not defaulted: a developer's own opt-in (shell or a service .env with
# MEM0_TELEMETRY=true) must not make the suite build a real analytics client.
# tests/test_third_party_telemetry.py runs its opt-in cases in subprocesses
# with their own environment, so they are unaffected.
import os  # noqa: E402

os.environ["MEM0_TELEMETRY"] = "false"
os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
import third_party_telemetry  # noqa: E402,F401

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _disable_auth_for_tests():
    """Run each test under the explicit auth opt-out.

    Auth is fail-closed: with no secret configured the middleware rejects
    every non-public request unless ``NEURALSCAPE_AUTH_DISABLED`` is set. The
    bulk of the suite exercises routes, not auth, so it runs with both
    secrets cleared AND the explicit opt-out on, i.e. the same state as a
    trusted local machine with ``NEURALSCAPE_AUTH_DISABLED=true``.

    Without clearing the secrets, the test process inherits the host's env (e.g. a
    ``NEURALSCAPE_USER_TOKEN_SECRET`` set for a running deployment),
    which makes every unauthenticated TestClient request return 401
    before reaching the route under test.

    Tests that specifically exercise auth (test_auth.py, including
    TestAuthFailClosed, and test_tokens.py) set their own credentials (which take
    precedence over the opt-out) or turn the opt-out back off.
    """
    from config import settings
    saved_token_secret = settings.neuralscape_user_token_secret
    saved_api_key = settings.neuralscape_api_key
    saved_default_user_id = settings.default_user_id
    saved_auth_disabled = settings.neuralscape_auth_disabled
    settings.neuralscape_user_token_secret = ""
    settings.neuralscape_api_key = ""
    settings.neuralscape_auth_disabled = True
    # Pin to the Pydantic class default so route fallbacks are deterministic
    # across deployment envs that set DEFAULT_USER_ID for production use.
    settings.default_user_id = "default_user"
    yield
    settings.neuralscape_user_token_secret = saved_token_secret
    settings.neuralscape_api_key = saved_api_key
    settings.default_user_id = saved_default_user_id
    settings.neuralscape_auth_disabled = saved_auth_disabled


class FakeRedis:
    """In-memory stand-in for the three Redis calls the per-memory edit lock makes (SET NX PX, GET, EVAL compare-and-delete)."""

    def __init__(self):
        self.store: dict[str, str] = {}

    def set(self, key, value, nx=False, px=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    def get(self, key):
        return self.store.get(key)

    def eval(self, script, numkeys, key, token):
        if self.store.get(key) == token:
            del self.store[key]
            return 1
        return 0


@pytest.fixture(autouse=True)
def edit_lock_redis(monkeypatch):
    """Patch the edit-lock Redis seam so no test ever reaches a real Redis; tests can read/steal locks through ``fake``."""
    import memory.edit as edit

    fake = FakeRedis()
    monkeypatch.setattr(edit, "_redis_client", lambda: fake)
    monkeypatch.setattr(edit, "_LOCK_WAIT_S", 0.0)     # a busy row answers immediately in tests (no sleeping)
    return fake


@pytest.fixture(autouse=True)
def _isolate_revocation_snapshot(monkeypatch):
    """Keep unit tests off any real Redis and clear the cached revocation snapshot.

    The bearer middleware reads the live Redis denylist; here it sees a
    reachable, empty one (so the cold-start 503 never fires) unless a test installs its own fake client through
    ``tokens._new_redis_client``.
    """
    import tokens

    class _EmptyRedis:
        """A reachable Redis whose revocation set is empty."""

        async def smembers(self, key):
            return set()

        async def aclose(self):
            return None

    monkeypatch.setattr(tokens, "_new_redis_client", _EmptyRedis)
    tokens.reset_revocation_snapshot()
    yield
    tokens.reset_revocation_snapshot()
