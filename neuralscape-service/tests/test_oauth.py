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
from tokens import issue_user_token, sign_payload, verify_payload, verify_user_token

# the autouse fixture patches these module attrs; keep the originals. getattr
# with a default so a reverted oauth.py fails the tests, not collection.
_ORIG = {
    n: getattr(oauth, n, None)
    for n in ("_consume_code_jti", "_consume_refresh_jti", "_family_revoked", "_revoke_family")
}

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
    main._async_memory = None
    tm = MagicMock()
    tm.connect = AsyncMock()
    tm.close = AsyncMock()
    orig_tm = main._task_manager
    main._task_manager = tm
    yield
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

    seen_refresh = set()

    async def _consume_refresh(jti, ttl):
        if jti in seen_refresh:
            return False
        seen_refresh.add(jti)
        return True

    revoked = set()

    async def _family_revoked(fam):
        return fam in revoked

    async def _revoke_family(fam, ttl):
        revoked.add(fam)

    with patch.object(oauth, "_consume_code_jti", side_effect=_consume, create=True), patch.object(
        oauth, "_consume_refresh_jti", side_effect=_consume_refresh, create=True
    ), patch.object(oauth, "_family_revoked", side_effect=_family_revoked, create=True), patch.object(
        oauth, "_revoke_family", side_effect=_revoke_family, create=True
    ):
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

    @pytest.mark.parametrize("verifier", ["vérifier-é" * 5, "\u2603" * 43])
    def test_a_non_ascii_code_verifier_is_an_invalid_grant_not_a_server_error(self, client, verifier):
        # a verifier is unreserved ASCII (RFC 7636 4.1): anything else simply does not match
        client_id, redirect_uri, _verifier, code = self._full_flow_to_code(client)
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

    def _tokens(self, client):
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
        return client_id, tok

    @staticmethod
    def _refresh(client, refresh_token, client_id):
        return client.post(
            "/oauth/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            },
        )

    def test_refresh_grant(self, client):
        client_id, tok = self._tokens(client)
        resp = self._refresh(client, tok["refresh_token"], client_id)
        assert resp.status_code == 200
        new = resp.json()
        assert verify_user_token(new["access_token"], SECRET)["user_id"] == USER
        # rotation: a different refresh token comes back, bound to the same client
        assert new["refresh_token"] != tok["refresh_token"]
        assert verify_payload(new["refresh_token"], SECRET)["client_id"] == client_id

    def test_refresh_token_is_single_use(self, client):
        client_id, tok = self._tokens(client)
        assert self._refresh(client, tok["refresh_token"], client_id).status_code == 200
        replay = self._refresh(client, tok["refresh_token"], client_id)
        assert replay.status_code == 400
        assert replay.json()["error"] == "invalid_grant"

    def test_reuse_revokes_the_whole_family(self, client):
        client_id, tok = self._tokens(client)
        second = self._refresh(client, tok["refresh_token"], client_id).json()
        # attacker replays the old, already-rotated token
        assert self._refresh(client, tok["refresh_token"], client_id).status_code == 400
        # the current successor is now dead too
        dead = self._refresh(client, second["refresh_token"], client_id)
        assert dead.status_code == 400 and dead.json()["error"] == "invalid_grant"

    def test_retry_with_same_token_by_same_client_revokes_session_no_grace(self, client):
        # A lost response makes the client retry the SAME token. By design
        # there is no grace window: the retry is reuse, and the session dies.
        client_id, tok = self._tokens(client)
        first = self._refresh(client, tok["refresh_token"], client_id)
        assert first.status_code == 200
        retry = self._refresh(client, tok["refresh_token"], client_id)
        assert retry.status_code == 400 and retry.json()["error"] == "invalid_grant"
        successor = self._refresh(client, first.json()["refresh_token"], client_id)
        assert successor.status_code == 400 and successor.json()["error"] == "invalid_grant"

    def test_other_family_unaffected_by_revocation(self, client):
        client_id, tok = self._tokens(client)
        _, other = self._tokens(client)  # an unrelated grant
        self._refresh(client, tok["refresh_token"], client_id)
        self._refresh(client, tok["refresh_token"], client_id)  # reuse -> revoke
        other_client = verify_payload(other["refresh_token"], SECRET)["client_id"]
        assert self._refresh(client, other["refresh_token"], other_client).status_code == 200

    def test_family_survives_rotation(self, client):
        client_id, tok = self._tokens(client)
        second = self._refresh(client, tok["refresh_token"], client_id).json()
        assert (
            verify_payload(second["refresh_token"], SECRET)["fam"]
            == verify_payload(tok["refresh_token"], SECRET)["fam"]
        )

    def test_revocation_check_fails_closed_when_redis_down(self, client):
        client_id, tok = self._tokens(client)

        async def _down(fam):
            raise oauth._ReplayGuardUnavailable()

        with patch.object(oauth, "_family_revoked", side_effect=_down):
            resp = self._refresh(client, tok["refresh_token"], client_id)
        assert resp.status_code == 503 and resp.json()["error"] == "temporarily_unavailable"

    def test_non_ascii_client_id_is_invalid_grant_not_500(self, client):
        client_id, tok = self._tokens(client)
        resp = self._refresh(client, tok["refresh_token"], "clïent-id-é")
        assert resp.status_code == 400 and resp.json()["error"] == "invalid_grant"

    def test_rotated_refresh_token_keeps_working(self, client):
        client_id, tok = self._tokens(client)
        second = self._refresh(client, tok["refresh_token"], client_id).json()
        third = self._refresh(client, second["refresh_token"], client_id)
        assert third.status_code == 200

    def test_refresh_requires_matching_client_id_and_does_not_burn_token(self, client):
        client_id, tok = self._tokens(client)
        other_id, _ = _register(client, "https://other.example.com/cb")
        wrong = self._refresh(client, tok["refresh_token"], other_id)
        assert wrong.status_code == 400 and wrong.json()["error"] == "invalid_grant"
        missing = self._refresh(client, tok["refresh_token"], "")
        assert missing.status_code == 400 and missing.json()["error"] == "invalid_grant"
        # the legitimate client can still redeem it
        assert self._refresh(client, tok["refresh_token"], client_id).status_code == 200

    def test_legacy_refresh_token_without_jti_or_client_is_rejected(self, client):
        client_id, _ = self._tokens(client)
        legacy = sign_payload(
            {"typ": "refresh", "user_id": USER, "exp": int(time.time()) + 600}, SECRET
        )
        resp = self._refresh(client, legacy, client_id)
        assert resp.status_code == 400 and resp.json()["error"] == "invalid_grant"

    def test_refresh_token_without_family_is_rejected(self, client):
        client_id, _ = self._tokens(client)
        nofam = sign_payload(
            {"typ": "refresh", "user_id": USER, "client_id": client_id, "jti": "j",
             "exp": int(time.time()) + 600}, SECRET)
        resp = self._refresh(client, nofam, client_id)
        assert resp.status_code == 400 and resp.json()["error"] == "invalid_grant"

    def test_refresh_fails_closed_when_guard_unavailable(self, client):
        client_id, tok = self._tokens(client)

        async def _down(jti, ttl):
            raise oauth._ReplayGuardUnavailable()

        with patch.object(oauth, "_consume_refresh_jti", side_effect=_down):
            resp = self._refresh(client, tok["refresh_token"], client_id)
        assert resp.status_code == 503
        assert resp.json()["error"] == "temporarily_unavailable"
        assert "access_token" not in resp.json()

    def test_code_exchange_fails_closed_when_guard_unavailable(self, client):
        client_id, redirect_uri, verifier, code = self._full_flow_to_code(client)

        async def _down(jti):
            raise oauth._ReplayGuardUnavailable()

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
        }
        with patch.object(oauth, "_consume_code_jti", side_effect=_down):
            resp = client.post("/oauth/token", data=data)
        assert resp.status_code == 503
        assert resp.json()["error"] == "temporarily_unavailable"
        assert "access_token" not in resp.json()
        # nothing was consumed, so the code still works once Redis is back
        assert client.post("/oauth/token", data=data).status_code == 200


class TestSingleUseGuard:
    """The real Redis helper must raise (not allow) when Redis is unreachable."""

    def test_unreachable_redis_raises_guard_unavailable(self):
        import asyncio

        from config import settings

        orig = settings.redis_url
        settings.redis_url = "redis://127.0.0.1:1/0"  # nothing listens here
        try:
            with pytest.raises(oauth._ReplayGuardUnavailable):
                asyncio.run(_ORIG["_consume_code_jti"]("j1"))
            with pytest.raises(oauth._ReplayGuardUnavailable):
                asyncio.run(_ORIG["_consume_refresh_jti"]("j2", 60))
        finally:
            settings.redis_url = orig

    def test_first_claim_true_second_false_and_ttl_floor(self):
        import asyncio

        calls = []

        class FakeRedis:
            seen = set()

            async def set(self, key, val, nx=None, ex=None):
                calls.append((key, ex))
                if key in self.seen:
                    return None
                self.seen.add(key)
                return True

            async def aclose(self):
                pass

        fake = FakeRedis()
        with patch("redis.asyncio.from_url", return_value=fake):
            assert asyncio.run(_ORIG["_consume_refresh_jti"]("abc", 0)) is True
            assert asyncio.run(_ORIG["_consume_refresh_jti"]("abc", 0)) is False
        assert calls[0] == ("oauth:refresh:abc", 1)

    def test_family_helpers_and_client_timeouts(self):
        import asyncio

        kwargs = {}

        class FakeRedis:
            store = {}

            async def set(self, key, val, nx=None, ex=None):
                self.store[key] = ex
                return True

            async def exists(self, key):
                return 1 if key in self.store else 0

            async def aclose(self):
                pass

        def fake_from_url(url, **kw):
            kwargs.update(kw)
            return FakeRedis()

        with patch("redis.asyncio.from_url", side_effect=fake_from_url):
            assert asyncio.run(_ORIG["_family_revoked"]("f1")) is False
            asyncio.run(_ORIG["_revoke_family"]("f1", 100))
            assert asyncio.run(_ORIG["_family_revoked"]("f1")) is True
        assert FakeRedis.store["oauth:revoked_family:f1"] == 100
        assert kwargs["socket_connect_timeout"] > 0 and kwargs["socket_timeout"] > 0


# ── anti-framing headers + consent page layout / contrast ────────────────


def _lum(hexcolor):
    h = hexcolor.lstrip("#")
    chans = []
    for i in (0, 2, 4):
        c = int(h[i : i + 2], 16) / 255
        chans.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * chans[0] + 0.7152 * chans[1] + 0.0722 * chans[2]


def _contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


class TestHtmlHardening:
    def _assert_hardened(self, resp):
        assert resp.headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in resp.headers["content-security-policy"]
        assert resp.headers["referrer-policy"] == "no-referrer"
        assert resp.headers["x-content-type-options"] == "nosniff"

    def _authorize_params(self, client):
        client_id, redirect_uri = _register(client)
        _, challenge = _pkce()
        return {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }

    def test_consent_page_not_frameable(self, client):
        resp = client.get("/oauth/authorize", params=self._authorize_params(client))
        assert resp.status_code == 200
        self._assert_hardened(resp)

    def test_consent_error_page_not_frameable(self, client):
        params = self._authorize_params(client)
        resp = _login_to_code(
            client, params["client_id"], params["redirect_uri"], params["code_challenge"], "nope"
        )
        assert resp.status_code == 400
        self._assert_hardened(resp)

    def test_link_error_page_not_frameable(self, client):
        resp = client.post("/oauth/link", data={"ticket": "garbage", "action": "new"})
        assert resp.status_code == 400
        self._assert_hardened(resp)

    def test_link_page_not_frameable(self):
        resp = oauth._render_link_page("t", "a@b.c", "abc")
        self._assert_hardened(resp)

    def test_supabase_callback_page_not_frameable(self, client):
        from config import settings

        orig = (settings.auth_provider, settings.supabase_url, settings.supabase_anon_key)
        settings.auth_provider = "supabase"
        settings.supabase_url = "https://x.supabase.example"
        settings.supabase_anon_key = "anon-test"
        try:
            resp = client.get("/oauth/supabase/callback")
        finally:
            settings.auth_provider, settings.supabase_url, settings.supabase_anon_key = orig
        assert resp.status_code == 200
        self._assert_hardened(resp)

    def test_supabase_consent_page_keeps_its_script(self, client):
        # the hardening CSP must not disable the inline module the Supabase
        # button needs: it only restricts framing/base/object.
        from config import settings

        orig = (settings.auth_provider, settings.supabase_url, settings.supabase_anon_key)
        settings.auth_provider = "supabase"
        settings.supabase_url = "https://x.supabase.example"
        settings.supabase_anon_key = "anon-test"
        try:
            resp = client.get("/oauth/authorize", params=self._authorize_params(client))
        finally:
            settings.auth_provider, settings.supabase_url, settings.supabase_anon_key = orig
        assert "ns-google-btn" in resp.text
        csp = resp.headers["content-security-policy"]
        assert "script-src" not in csp and "default-src" not in csp
        self._assert_hardened(resp)


class TestConsentLayout:
    def _page(self, client):
        client_id, redirect_uri = _register(client)
        _, challenge = _pkce()
        return client.get(
            "/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
        ).text

    def test_card_is_border_box_and_wraps_long_text(self, client):
        page = self._page(client)
        # universal rule: the .card (padding + border) must not add to its 92vw width
        assert "*, *::before, *::after { box-sizing: border-box; }" in page
        assert "overflow-wrap: anywhere" in page
        assert "max-width: 100%" in page

    def test_link_and_error_pages_reflow_too(self, client):
        link = oauth._render_link_page("t", "a@b.c", "abc").body.decode()
        assert "*, *::before, *::after { box-sizing: border-box; }" in link and "overflow-wrap:anywhere" in link
        err = oauth._login_error_page(oauth.LoginError("x", 400)).body.decode()
        assert "*,*::before,*::after{box-sizing:border-box;}" in err and "overflow-wrap:anywhere" in err

    def test_primary_button_text_meets_wcag_aa(self, client):
        import re

        for page in (
            self._page(client),
            oauth._render_link_page("t", "a@b.c", "abc").body.decode(),
        ):
            m = re.search(r"(?:button|\.primary) *\{[^}]*?background: *(#[0-9a-fA-F]{6});[^}]*?color: *(#[0-9a-fA-F]{3,6})", page)
            assert m, "primary action style not found"
            fg = m.group(2) if len(m.group(2)) == 7 else "#ffffff"
            assert _contrast(m.group(1), fg) >= 4.5
        # the regression value: white on #5b7cfa is 3.68:1
        assert _contrast("#5b7cfa", "#ffffff") < 4.5

    def test_hint_text_meets_wcag_aa(self):
        assert _contrast("#7b8198", "#151821") >= 4.5


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


class TestClientIdentityIsUnique:
    """RFC 7591 §3.2.1: each registration gets its own client_id, and a code or
    refresh token issued to one client is refused for another, even when both
    registered the same redirect URI in the same second (acceptance scenario
    test_the_code_is_bound_to_its_client_redirect_and_verifier)."""

    def test_two_registrations_in_the_same_second_get_distinct_client_ids(self, client):
        with patch.object(oauth.time, "time", return_value=1_900_000_000):
            first, _ = _register(client)
            second, _ = _register(client)
        assert first != second

    def test_a_code_for_one_client_is_refused_for_another_registered_in_the_same_second(
        self, client
    ):
        verifier, challenge = _pkce()
        with patch.object(oauth.time, "time", return_value=int(time.time())):
            client_a, redirect = _register(client)
            client_b, _ = _register(client)
        login = issue_user_token(USER, SECRET, 3600)
        loc = _login_to_code(client, client_a, redirect, challenge, login).headers["location"]
        code = loc.split("code=")[1].split("&")[0]
        resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect,
                "client_id": client_b,
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_grant"
