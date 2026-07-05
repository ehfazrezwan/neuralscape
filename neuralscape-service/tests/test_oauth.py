"""Tests for the built-in OAuth 2.1 Authorization Server (oauth.py).

Covers the full Cowork connector flow end to end without a browser or Redis:

- Discovery metadata (AS + Protected Resource) shape & gating
- Dynamic Client Registration (stateless signed client_id)
- /authorize consent page render + request validation
- token-paste "login" -> authorization code issuance
- /token authorization_code exchange with PKCE S256
- refresh_token grant
- authorization-code single-use (replay) rejection
- typ domain separation: codes/refresh tokens are NOT valid Bearer creds
- 401 on the protected resource carries the RFC 9728 WWW-Authenticate header
- MCP tool identity resolves from the authenticated token, not arguments
"""

import base64
import hashlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main
import oauth
from main import app
from tokens import issue_user_token, sign_payload, verify_user_token

SECRET = "oauth-test-secret"
PUBLIC_URL = "https://neuralscape.example.com"
USER = "alice"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _pkce():
    verifier = _b64url(b"verifier-bytes-must-be-43-128-chars-long-xxxxxxxx")
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


@pytest.fixture(autouse=True)
def mock_backends():
    """Avoid real mem0/Graphiti/Redis like the other suites do."""
    orig_mem, orig_g, orig_b = main._memory, main._graphiti, main._bridge
    main._memory = MagicMock()
    main._graphiti = MagicMock()
    main._bridge = MagicMock()
    main._async_memory = None
    tm = MagicMock()
    tm.connect = AsyncMock()
    tm.close = AsyncMock()
    orig_tm = main._task_manager
    main._task_manager = tm
    yield
    main._memory, main._graphiti, main._bridge = orig_mem, orig_g, orig_b
    main._task_manager = orig_tm


@pytest.fixture(autouse=True)
def enable_oauth():
    """Configure the signing secret + public URL that switch OAuth on."""
    from config import settings

    orig_secret = settings.neuralscape_user_token_secret
    orig_url = settings.neuralscape_public_url
    settings.neuralscape_user_token_secret = SECRET
    settings.neuralscape_public_url = PUBLIC_URL
    yield
    settings.neuralscape_user_token_secret = orig_secret
    settings.neuralscape_public_url = orig_url


@pytest.fixture(autouse=True)
def no_redis_replay():
    """Replace the Redis-backed code replay guard with an in-memory set."""
    seen = set()

    async def _consume(jti):
        if jti in seen:
            return False
        seen.add(jti)
        return True

    with patch.object(oauth, "_consume_code_jti", side_effect=_consume):
        yield


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def _register(client, redirect_uri="https://claude.ai/api/mcp/auth_callback"):
    resp = client.post("/oauth/register", json={"redirect_uris": [redirect_uri]})
    assert resp.status_code == 201, resp.text
    return resp.json()["client_id"], redirect_uri


def _login_to_code(client, client_id, redirect_uri, challenge, login_token, state="xyz"):
    resp = client.post(
        "/oauth/authorize",
        data={
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "token": login_token,
        },
        follow_redirects=False,
    )
    return resp


# ── discovery metadata ───────────────────────────────────────────────────


class TestDiscovery:
    def test_as_metadata(self, client):
        m = client.get("/.well-known/oauth-authorization-server").json()
        assert m["issuer"] == PUBLIC_URL
        assert m["authorization_endpoint"] == f"{PUBLIC_URL}/oauth/authorize"
        assert m["token_endpoint"] == f"{PUBLIC_URL}/oauth/token"
        assert m["registration_endpoint"] == f"{PUBLIC_URL}/oauth/register"
        assert m["code_challenge_methods_supported"] == ["S256"]
        assert "authorization_code" in m["grant_types_supported"]
        assert "refresh_token" in m["grant_types_supported"]

    def test_protected_resource_metadata(self, client):
        m = client.get("/.well-known/oauth-protected-resource").json()
        assert m["resource"] == f"{PUBLIC_URL}/mcp"
        assert m["authorization_servers"] == [PUBLIC_URL]

    def test_protected_resource_metadata_mcp_suffix(self, client):
        m = client.get("/.well-known/oauth-protected-resource/mcp").json()
        assert m["resource"] == f"{PUBLIC_URL}/mcp"

    def test_metadata_404_when_not_configured(self, client):
        from config import settings

        orig = settings.neuralscape_public_url
        settings.neuralscape_public_url = ""
        try:
            resp = client.get("/.well-known/oauth-authorization-server")
            assert resp.status_code == 404
        finally:
            settings.neuralscape_public_url = orig


# ── dynamic client registration ──────────────────────────────────────────


class TestRegistration:
    def test_register_returns_signed_client_id(self, client):
        client_id, _ = _register(client)
        # client_id is a signed token carrying typ=client + redirect_uris
        from tokens import verify_payload

        claims = verify_payload(client_id, SECRET)
        assert claims is not None and claims["typ"] == "client"

    def test_register_rejects_missing_redirect_uris(self, client):
        resp = client.post("/oauth/register", json={})
        assert resp.status_code == 400

    def test_register_rejects_insecure_redirect_uri(self, client):
        # plaintext http to a non-loopback host must be rejected — it would
        # otherwise be signed into the client_id and trusted as a 303 target.
        resp = client.post(
            "/oauth/register",
            json={"redirect_uris": ["http://evil.example.com/steal"]},
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_redirect_uri"

    def test_register_allows_loopback_http_redirect(self, client):
        # native/CLI clients may use an http loopback callback (RFC 8252).
        resp = client.post(
            "/oauth/register",
            json={"redirect_uris": ["http://127.0.0.1:7777/callback"]},
        )
        assert resp.status_code == 201


# ── authorize (consent) ──────────────────────────────────────────────────


class TestAuthorize:
    def test_consent_page_renders(self, client):
        client_id, redirect_uri = _register(client)
        _, challenge = _pkce()
        resp = client.get(
            "/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "xyz",
            },
        )
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Neuralscape" in resp.text

    def test_authorize_rejects_unregistered_redirect_uri(self, client):
        client_id, _ = _register(client)
        _, challenge = _pkce()
        resp = client.get(
            "/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "https://evil.example.com/steal",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_request"

    def test_authorize_requires_pkce(self, client):
        client_id, redirect_uri = _register(client)
        resp = client.get(
            "/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
            },
        )
        assert resp.status_code == 400

    def test_bad_login_token_rerenders_with_error(self, client):
        client_id, redirect_uri = _register(client)
        _, challenge = _pkce()
        resp = _login_to_code(client, client_id, redirect_uri, challenge, "not-a-token")
        assert resp.status_code == 400
        assert "invalid or expired" in resp.text.lower()

    def test_valid_login_issues_code_redirect(self, client):
        client_id, redirect_uri = _register(client)
        _, challenge = _pkce()
        login = issue_user_token(USER, SECRET, 3600)
        resp = _login_to_code(client, client_id, redirect_uri, challenge, login)
        assert resp.status_code == 303
        loc = resp.headers["location"]
        assert loc.startswith(redirect_uri)
        assert "code=" in loc and "state=xyz" in loc


# ── token endpoint ───────────────────────────────────────────────────────


class TestToken:
    def _full_flow_to_code(self, client):
        client_id, redirect_uri = _register(client)
        verifier, challenge = _pkce()
        login = issue_user_token(USER, SECRET, 3600)
        resp = _login_to_code(client, client_id, redirect_uri, challenge, login)
        loc = resp.headers["location"]
        code = loc.split("code=")[1].split("&")[0]
        return client_id, redirect_uri, verifier, code

    def test_code_exchange_happy_path(self, client):
        client_id, redirect_uri, verifier, code = self._full_flow_to_code(client)
        resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["token_type"] == "Bearer"
        assert body["expires_in"] > 0
        # the access token is a valid Bearer user token for USER
        payload = verify_user_token(body["access_token"], SECRET)
        assert payload is not None and payload["user_id"] == USER
        assert "refresh_token" in body

    def test_code_exchange_requires_client_id(self, client):
        # omitting the bound client_id must be rejected, not silently allowed
        _cid, redirect_uri, verifier, code = self._full_flow_to_code(client)
        resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_grant"

    def test_code_exchange_requires_redirect_uri(self, client):
        client_id, _ru, verifier, code = self._full_flow_to_code(client)
        resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": verifier,
                "client_id": client_id,
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_grant"

    def test_code_exchange_rejects_code_without_user(self, client):
        # a malformed-but-signed code (no user_id) must yield a spec-compliant
        # invalid_grant, not a 500 from a KeyError on claims["user_id"].
        client_id, redirect_uri = _register(client)
        verifier, challenge = _pkce()
        forged = sign_payload(
            {
                "typ": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": challenge,
                "jti": "no-user-jti",
                "exp": int(time.time()) + 300,
            },
            SECRET,
        )
        resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": forged,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_grant"

    def test_code_exchange_wrong_pkce_rejected(self, client):
        client_id, redirect_uri, _verifier, code = self._full_flow_to_code(client)
        resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": "wrong-verifier",
                "redirect_uri": redirect_uri,
                "client_id": client_id,
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_grant"

    def test_code_is_single_use(self, client):
        client_id, redirect_uri, verifier, code = self._full_flow_to_code(client)
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
        }
        first = client.post("/oauth/token", data=data)
        assert first.status_code == 200
        second = client.post("/oauth/token", data=data)
        assert second.status_code == 400
        assert "already used" in second.json()["error_description"]

    def test_refresh_grant(self, client):
        client_id, redirect_uri, verifier, code = self._full_flow_to_code(client)
        tok = client.post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
            },
        ).json()
        resp = client.post(
            "/oauth/token",
            data={"grant_type": "refresh_token", "refresh_token": tok["refresh_token"]},
        )
        assert resp.status_code == 200
        new = resp.json()
        assert verify_user_token(new["access_token"], SECRET)["user_id"] == USER

    def test_unsupported_grant(self, client):
        resp = client.post("/oauth/token", data={"grant_type": "password"})
        assert resp.status_code == 400
        assert resp.json()["error"] == "unsupported_grant_type"


# ── token typ domain separation ──────────────────────────────────────────


class TestTypSeparation:
    def test_refresh_token_not_accepted_as_bearer(self):
        refresh = sign_payload(
            {"typ": "refresh", "user_id": USER, "exp": int(time.time()) + 600}, SECRET
        )
        assert verify_user_token(refresh, SECRET) is None

    def test_code_not_accepted_as_bearer(self):
        code = sign_payload(
            {"typ": "code", "user_id": USER, "exp": int(time.time()) + 600}, SECRET
        )
        assert verify_user_token(code, SECRET) is None

    def test_access_and_legacy_tokens_accepted(self):
        access = issue_user_token(USER, SECRET, 3600, typ="access")
        legacy = issue_user_token(USER, SECRET, 3600)  # no typ (admin-issued)
        assert verify_user_token(access, SECRET)["user_id"] == USER
        assert verify_user_token(legacy, SECRET)["user_id"] == USER


# ── resource server 401 carries discovery pointer ────────────────────────


class TestResourceChallenge:
    def test_401_has_www_authenticate(self, client):
        resp = client.get("/v1/memories", params={"user_id": "x"})
        assert resp.status_code == 401
        www = resp.headers.get("WWW-Authenticate", "")
        assert "resource_metadata=" in www
        assert "/.well-known/oauth-protected-resource" in www

    def test_401_omits_www_authenticate_without_secret(self, client):
        # OAuth self-disables unless BOTH public_url and the signing secret are
        # set. With only public_url (auth still on via a legacy key), the 401
        # must NOT advertise discovery — the /.well-known endpoints 404.
        from config import settings

        orig_secret = settings.neuralscape_user_token_secret
        orig_key = settings.neuralscape_api_key
        settings.neuralscape_user_token_secret = ""
        settings.neuralscape_api_key = "legacy-key"  # keep auth enforced
        try:
            resp = client.get("/v1/memories", params={"user_id": "x"})
            assert resp.status_code == 401
            assert "WWW-Authenticate" not in resp.headers
        finally:
            settings.neuralscape_user_token_secret = orig_secret
            settings.neuralscape_api_key = orig_key


# ── MCP identity from token, not arguments ───────────────────────────────


class TestAcceptHeaderShim:
    """The /mcp Accept-header normalization that unblocks Cowork's SSE-only client."""

    def test_accepts_json_variants(self):
        from mcp_server import _accepts_json

        assert _accepts_json(b"application/json, text/event-stream")
        assert _accepts_json(b"application/json")
        assert _accepts_json(b"*/*")
        assert _accepts_json(b"")
        assert not _accepts_json(b"text/event-stream")

    def test_sse_only_gets_json_added(self):
        from mcp_server import _ensure_accept

        out = _ensure_accept(b"text/event-stream")
        assert b"application/json" in out and b"text/event-stream" in out

    def test_json_only_gets_sse_added(self):
        from mcp_server import _ensure_accept

        out = _ensure_accept(b"application/json")
        assert b"application/json" in out and b"text/event-stream" in out

    def test_both_present_unchanged(self):
        from mcp_server import _ensure_accept

        both = b"application/json, text/event-stream"
        assert _ensure_accept(both) == both

    def test_empty_gets_both(self):
        from mcp_server import _ensure_accept

        out = _ensure_accept(b"")
        assert b"application/json" in out and b"text/event-stream" in out

    def test_router_sends_sse_only_client_to_sse_app_with_normalized_header(self):
        import asyncio

        from mcp_server import _AcceptRouter

        seen = {}

        async def json_app(scope, receive, send):
            seen["app"] = "json"

        async def sse_app(scope, receive, send):
            seen["app"] = "sse"
            seen["headers"] = scope["headers"]

        router = _AcceptRouter(json_app, sse_app)
        scope = {"type": "http", "headers": [(b"accept", b"text/event-stream")]}
        asyncio.run(router(scope, None, None))
        assert seen["app"] == "sse"
        accept = dict(seen["headers"])[b"accept"]
        assert b"application/json" in accept and b"text/event-stream" in accept

    def test_mcp_path_without_trailing_slash_is_rewritten_not_redirected(self):
        import asyncio

        from main import _McpTrailingSlash

        seen = {}

        async def fake_app(scope, receive, send):
            seen["path"] = scope["path"]
            seen["raw_path"] = scope["raw_path"]

        mw = _McpTrailingSlash(fake_app)
        scope = {"type": "http", "path": "/mcp", "raw_path": b"/mcp"}
        asyncio.run(mw(scope, None, None))
        # Anthropic's connector POSTs to the advertised resource URL /mcp
        # (no slash) and won't follow a 307 — the path must serve directly.
        assert seen["path"] == "/mcp/"
        assert seen["raw_path"] == b"/mcp/"

        scope = {"type": "http", "path": "/mcp/other", "raw_path": b"/mcp/other"}
        asyncio.run(mw(scope, None, None))
        assert seen["path"] == "/mcp/other"

    def test_router_keeps_json_clients_on_json_app_unchanged(self):
        import asyncio

        from mcp_server import _AcceptRouter

        seen = {}

        async def json_app(scope, receive, send):
            seen["app"] = "json"
            seen["headers"] = scope["headers"]

        async def sse_app(scope, receive, send):
            seen["app"] = "sse"

        router = _AcceptRouter(json_app, sse_app)
        original = [(b"accept", b"application/json, text/event-stream")]
        scope = {"type": "http", "headers": list(original)}
        asyncio.run(router(scope, None, None))
        assert seen["app"] == "json"
        assert seen["headers"] == original


class TestMcpIdentity:
    def test_tool_uses_contextvar_identity_over_arguments(self):
        import asyncio

        import mcp_server
        from auth import current_user_id

        svc = MagicMock()
        svc.list_memories.return_value = []
        orig = mcp_server._service
        mcp_server._service = svc
        tok = current_user_id.set("alice")
        try:
            asyncio.run(
                mcp_server.call_tool("list_memories", {"user_id": "mallory"})
            )
        finally:
            current_user_id.reset(tok)
            mcp_server._service = orig
        # The authenticated identity wins; the spoofed argument is ignored.
        assert svc.list_memories.call_args.kwargs["user_id"] == "alice"
