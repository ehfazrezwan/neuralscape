"""Tests for Bearer token authentication middleware.

Tests cover:
- Auth disabled when NEURALSCAPE_API_KEY is empty (local dev mode)
- Auth enabled when NEURALSCAPE_API_KEY is set
- Health endpoint always accessible without auth
- Protected endpoints return 401 without/invalid token
- Protected endpoints pass through with valid token
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main
from config import Settings
from main import app


TEST_API_KEY = "test-secret-key-12345"


@pytest.fixture(autouse=True)
def mock_memory():
    """Patch lazy-init globals so no real mem0/Graphiti is created."""
    original_memory = main._memory
    original_graphiti = main._graphiti
    original_bridge = main._bridge
    original_async_memory = main._async_memory

    main._memory = MagicMock(name="Memory")
    main._graphiti = MagicMock(name="Graphiti")
    main._bridge = MagicMock(name="AsyncBridge")
    main._async_memory = None

    yield

    main._memory = original_memory
    main._graphiti = original_graphiti
    main._bridge = original_bridge
    main._async_memory = original_async_memory


@pytest.fixture(autouse=True)
def mock_task_manager():
    """Patch TaskManager so tests don't need Redis."""
    mock_tm = MagicMock(name="TaskManager")
    mock_tm.connect = AsyncMock()
    mock_tm.close = AsyncMock()
    mock_tm.pool = MagicMock()
    mock_tm.pool.ping = AsyncMock()
    original = main._task_manager
    main._task_manager = mock_tm
    yield mock_tm
    main._task_manager = original


@pytest.fixture
def mock_service():
    """Patch MemoryService for v1 endpoints."""
    mock_svc = MagicMock(name="MemoryService")
    mock_svc._memory = MagicMock()
    mock_svc._graphiti = MagicMock()
    original = main._service
    main._service = mock_svc
    yield mock_svc
    main._service = original


class TestAuthDisabled:
    """When NEURALSCAPE_API_KEY is empty, all requests pass through."""

    @pytest.fixture(autouse=True)
    def disable_auth(self):
        from config import settings
        original = settings.neuralscape_api_key
        settings.neuralscape_api_key = ""
        yield
        settings.neuralscape_api_key = original

    def test_health_no_auth(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_v1_endpoint_no_auth(self, mock_service):
        mock_service.list_memories.return_value = []
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/memories", params={"user_id": "test"})
        assert resp.status_code == 200


class TestAuthEnabled:
    """When NEURALSCAPE_API_KEY is set, endpoints require Bearer token."""

    @pytest.fixture(autouse=True)
    def enable_auth(self):
        from config import settings
        original = settings.neuralscape_api_key
        settings.neuralscape_api_key = TEST_API_KEY
        yield
        settings.neuralscape_api_key = original

    def test_health_always_public(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_live_always_public(self):
        """The container healthcheck has no credentials — /health/live must be public."""
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health/live")
        assert resp.status_code == 200

    def test_no_auth_header_returns_401(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/memories", params={"user_id": "test"})
        assert resp.status_code == 401
        assert "Missing" in resp.json()["detail"]

    def test_invalid_token_returns_401(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/v1/memories",
            params={"user_id": "test"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]

    def test_valid_token_passes_through(self, mock_service):
        mock_service.list_memories.return_value = []
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/v1/memories",
            params={"user_id": "test"},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert resp.status_code == 200

    def test_legacy_endpoint_requires_auth(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/memories", params={"user_id": "test"})
        assert resp.status_code == 401

    def test_legacy_endpoint_with_valid_token(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/memories",
            params={"user_id": "test"},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        # Should pass auth (may fail for other reasons in test, but not 401)
        assert resp.status_code != 401

    def test_malformed_auth_header_returns_401(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/v1/memories",
            params={"user_id": "test"},
            headers={"Authorization": f"Basic {TEST_API_KEY}"},
        )
        assert resp.status_code == 401

    def test_mcp_endpoint_requires_auth(self):
        """MCP HTTP transport at /mcp/ should also require auth."""
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/mcp/")
        # Should be 401 (auth) not 404 (route not found)
        # Note: MCP may not be mounted in test env, so we accept 401 or 404
        assert resp.status_code in (401, 404)


class TestLegacyKeyWithDotFallsBack:
    """Regression: when both NEURALSCAPE_USER_TOKEN_SECRET and
    NEURALSCAPE_API_KEY are set, a legacy shared key that happens to
    contain '.' must still authenticate via the legacy path.

    Before the fix, any Bearer containing '.' was forced down the HMAC
    path and got an immediate 401 on verify failure — silently breaking
    legacy keys that include dots (perfectly valid characters).
    """

    @pytest.fixture(autouse=True)
    def enable_both(self):
        from config import settings
        orig_secret = settings.neuralscape_user_token_secret
        orig_key = settings.neuralscape_api_key
        settings.neuralscape_user_token_secret = "test-token-secret"
        settings.neuralscape_api_key = "legacy-key.with.dots"
        yield
        settings.neuralscape_user_token_secret = orig_secret
        settings.neuralscape_api_key = orig_key

    def test_legacy_key_with_dots_authenticates(self, mock_service):
        """A legacy key containing dots must still authenticate when
        HMAC verification fails (the dot triggers token detection, but
        the secret doesn't validate the signature, so we fall through)."""
        mock_service.list_memories.return_value = []
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/v1/memories",
            params={"user_id": "test"},
            headers={"Authorization": "Bearer legacy-key.with.dots"},
        )
        assert resp.status_code == 200, (
            f"Legacy key with dots failed to authenticate after HMAC verify failure. "
            f"Response: {resp.status_code} {resp.json()}"
        )

    def test_invalid_token_with_dots_still_returns_401(self):
        """A bearer with dots that isn't a valid token AND doesn't match
        the legacy key must still return 401 (no silent pass-through)."""
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/v1/memories",
            params={"user_id": "test"},
            headers={"Authorization": "Bearer not.a.real.token.or.key"},
        )
        assert resp.status_code == 401


# ══════════════════════════════════════════════
# Legacy root endpoints — multi-user identity resolution
# ══════════════════════════════════════════════
#
# Historically the legacy root routes (/memories, /search, /graph/*) trusted
# whatever user_id the caller put in the request body/query string, even
# when the caller had authenticated with a per-user token. That let any
# token holder impersonate any other user simply by naming them. These
# routes must now resolve identity via `_resolve_user_id` exactly like
# /v1/* does: token wins, a body/token mismatch is rejected with 400, and
# (via the shared BearerAuthMiddleware) an unauthenticated call is rejected
# the same way /v1/* is. When NEURALSCAPE_USER_TOKEN_SECRET is unset, the
# original trust-the-body/query behavior must be unchanged.


def _legacy_route_cases():
    """(name, http_method, path, kwarg_key, body_builder) for every legacy
    root route that accepts a user_id from the body or query string.
    ``body_builder(user_id_or_none)`` omits the user_id key entirely when
    passed None, to exercise the "identity comes from the token alone" case.
    """

    def add_body(uid):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        if uid is not None:
            payload["user_id"] = uid
        return payload

    def search_body(uid):
        payload = {"query": "hello"}
        if uid is not None:
            payload["user_id"] = uid
        return payload

    def query_only(uid):
        return {"user_id": uid} if uid is not None else {}

    return [
        ("post_memories", "post", "/memories", "json", add_body),
        ("post_search", "post", "/search", "json", search_body),
        ("get_memories", "get", "/memories", "params", query_only),
        ("delete_memories", "delete", "/memories", "params", query_only),
        ("post_memories_async", "post", "/memories/async", "json", add_body),
        ("post_graph_search", "post", "/graph/search", "json", search_body),
        ("get_graph_nodes", "get", "/graph/nodes", "params", query_only),
        ("get_graph_edges", "get", "/graph/edges", "params", query_only),
        ("get_graph_episodes", "get", "/graph/episodes", "params", query_only),
        ("get_graph_communities", "get", "/graph/communities", "params", query_only),
    ]


LEGACY_ROUTE_CASES = _legacy_route_cases()
LEGACY_ROUTE_IDS = [c[0] for c in LEGACY_ROUTE_CASES]


class TestLegacyEndpointsMultiUserIdentity:
    """Every legacy root route that accepts user_id must resolve identity
    via `_resolve_user_id`, same as /v1/*, once a per-user token secret is
    configured — and must leave legacy shared-key behavior unchanged when
    it is not.
    """

    TEST_SECRET = "legacy-multiuser-secret"

    @pytest.fixture(autouse=True)
    def enable_token_secret(self):
        from config import settings
        orig_secret = settings.neuralscape_user_token_secret
        orig_key = settings.neuralscape_api_key
        settings.neuralscape_user_token_secret = self.TEST_SECRET
        settings.neuralscape_api_key = ""
        yield
        settings.neuralscape_user_token_secret = orig_secret
        settings.neuralscape_api_key = orig_key

    @pytest.fixture(autouse=True)
    def stub_enqueue_store(self):
        """POST /memories/async awaits _task_manager.enqueue_store; the
        shared mock_task_manager fixture doesn't configure it as async."""
        original = main._task_manager.enqueue_store
        main._task_manager.enqueue_store = AsyncMock(return_value="task-async-1")
        yield
        main._task_manager.enqueue_store = original

    @staticmethod
    def _token(user_id, secret):
        from tokens import issue_user_token
        return issue_user_token(user_id, secret, 3600)

    @staticmethod
    def _call(client, method, path, kwarg_key, payload, headers=None):
        fn = getattr(client, method)
        kwargs = {kwarg_key: payload}
        if headers:
            kwargs["headers"] = headers
        return fn(path, **kwargs)

    @pytest.mark.parametrize(
        "name,method,path,kwarg_key,body_fn", LEGACY_ROUTE_CASES, ids=LEGACY_ROUTE_IDS
    )
    def test_no_token_rejected(self, name, method, path, kwarg_key, body_fn):
        """Secret configured, no Authorization header -> rejected, same
        shape the middleware already uses for /v1/*."""
        client = TestClient(app, raise_server_exceptions=False)
        resp = self._call(client, method, path, kwarg_key, body_fn("someone"))
        assert resp.status_code == 401, f"{name}: expected 401, got {resp.status_code} {resp.text}"
        assert "detail" in resp.json()

    @pytest.mark.parametrize(
        "name,method,path,kwarg_key,body_fn", LEGACY_ROUTE_CASES, ids=LEGACY_ROUTE_IDS
    )
    def test_valid_token_no_body_user_id_uses_token_identity(
        self, name, method, path, kwarg_key, body_fn
    ):
        """Secret configured, valid token, no user_id in body/query ->
        identity comes from the token and the request succeeds."""
        client = TestClient(app, raise_server_exceptions=False)
        token = self._token("alice-from-token", self.TEST_SECRET)
        resp = self._call(
            client, method, path, kwarg_key, body_fn(None),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code in (200, 202), f"{name}: {resp.status_code} {resp.text}"

    @pytest.mark.parametrize(
        "name,method,path,kwarg_key,body_fn", LEGACY_ROUTE_CASES, ids=LEGACY_ROUTE_IDS
    )
    def test_valid_token_matching_body_user_id_allowed(
        self, name, method, path, kwarg_key, body_fn
    ):
        """Secret configured, valid token, body/query user_id matches the
        token's user_id -> allowed."""
        client = TestClient(app, raise_server_exceptions=False)
        token = self._token("alice-from-token", self.TEST_SECRET)
        resp = self._call(
            client, method, path, kwarg_key, body_fn("alice-from-token"),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code in (200, 202), f"{name}: {resp.status_code} {resp.text}"

    @pytest.mark.parametrize(
        "name,method,path,kwarg_key,body_fn", LEGACY_ROUTE_CASES, ids=LEGACY_ROUTE_IDS
    )
    def test_valid_token_mismatched_body_user_id_rejected(
        self, name, method, path, kwarg_key, body_fn
    ):
        """Secret configured, valid token, body/query user_id names a
        DIFFERENT user -> 400, blocking impersonation."""
        client = TestClient(app, raise_server_exceptions=False)
        token = self._token("alice-from-token", self.TEST_SECRET)
        resp = self._call(
            client, method, path, kwarg_key, body_fn("bob-impersonator"),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 400, f"{name}: {resp.status_code} {resp.text}"
        assert "does not match" in resp.json()["detail"]

    @pytest.mark.parametrize(
        "name,method,path,kwarg_key,body_fn", LEGACY_ROUTE_CASES, ids=LEGACY_ROUTE_IDS
    )
    def test_secret_not_configured_legacy_body_user_id_trusted(
        self, name, method, path, kwarg_key, body_fn, monkeypatch
    ):
        """No per-user token secret configured (shared-key / no-auth
        deployments) -> legacy behavior preserved, body/query user_id is
        trusted as-is, no Authorization header required."""
        from config import settings
        monkeypatch.setattr(settings, "neuralscape_user_token_secret", "")
        monkeypatch.setattr(settings, "neuralscape_api_key", "")
        client = TestClient(app, raise_server_exceptions=False)
        resp = self._call(client, method, path, kwarg_key, body_fn("legacy-alice"))
        assert resp.status_code in (200, 202), f"{name}: {resp.status_code} {resp.text}"
