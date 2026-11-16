"""Login CSRF: the signed login_state must be bound to the browser that started
the login (cookie), for BOTH federated providers.

Attack being closed: an attacker registers a client, starts /oauth/authorize in
their own browser (so they hold a login_state for their client and their own
PKCE challenge), and sends the victim the IdP URL carrying that state. The
victim logs in; without a binding the callback issued an authorization code for
the VICTIM's account to the attacker's redirect URI.

Everything here drives the real FastAPI app. Only the IdP verification (Google
token exchange / Supabase JWT) and the Redis single-use claim are stubbed.
"""

from __future__ import annotations

import base64
import hashlib
import html
import re
import secrets
import time
from urllib.parse import parse_qs, urlparse

import jwt
import pytest
from fastapi.testclient import TestClient

import login_providers as lp
import oauth
from config import settings
from main import app
from tokens import sign_payload

PUBLIC_URL = "https://neuralscape.example.com"
REDIRECT = "https://claude.ai/api/mcp/auth_callback"
ATTACKER_REDIRECT = "https://attacker.example.net/cb"
VICTIM = "victim-user"
COOKIE = lp.LOGIN_BINDING_COOKIE


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _pkce():
    verifier = _b64url(secrets.token_bytes(40))
    return verifier, _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


class _Resp:
    status_code = 200
    text = ""

    def json(self):
        return {"id_token": "stub.id.token"}


class _FakeGoogleClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, *a, **k):
        return _Resp()


@pytest.fixture
def env(monkeypatch):
    secret = secrets.token_hex(32)
    sb_secret = secrets.token_hex(32)
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", secret)
    monkeypatch.setattr(settings, "neuralscape_public_url", PUBLIC_URL)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    monkeypatch.setattr(settings, "auth_allow_token_paste", False)
    monkeypatch.setattr(settings, "google_oauth_client_id", "google-client-id")
    monkeypatch.setattr(settings, "google_oauth_client_secret", secrets.token_hex(8))
    monkeypatch.setattr(settings, "auth_allowed_domains", "example.com")
    monkeypatch.setattr(settings, "supabase_url", "https://proj.supabase.example")
    monkeypatch.setattr(settings, "supabase_anon_key", secrets.token_hex(8))
    monkeypatch.setattr(settings, "supabase_jwt_secret", sb_secret)

    # IdP verification stubs: every callback authenticates the victim.
    monkeypatch.setattr(lp.httpx, "AsyncClient", _FakeGoogleClient)
    monkeypatch.setattr(
        lp, "_verify_oidc_jwt",
        lambda *a, **k: {"email": "victim@example.com", "email_verified": True, "sub": "g-victim"},
    )

    async def _linked(sub, email):
        return VICTIM, True

    monkeypatch.setattr(lp, "resolve_login_identity", _linked)

    # Redis stand-ins: single-use claims in memory, with an outage switch.
    store = {"nonces": set(), "codes": set(), "down": False}

    async def _consume_nonce(nonce):
        if store["down"]:
            raise lp.LoginBindingUnavailable("redis down")
        if nonce in store["nonces"]:
            return False
        store["nonces"].add(nonce)
        return True

    async def _consume_code(jti):
        if jti in store["codes"]:
            return False
        store["codes"].add(jti)
        return True

    monkeypatch.setattr(lp, "_consume_login_nonce", _consume_nonce)
    monkeypatch.setattr(oauth, "_consume_code_jti", _consume_code)

    def sb_token():
        return jwt.encode(
            {"aud": "authenticated", "email": "victim@example.com", "email_verified": True,
             "sub": "sb-victim", "exp": int(time.time()) + 300},
            sb_secret, algorithm="HS256",
        )

    return {"secret": secret, "store": store, "sb_token": sb_token}


@pytest.fixture(params=["google", "supabase"])
def provider(request, env, monkeypatch):
    monkeypatch.setattr(settings, "auth_provider", request.param)
    return request.param


def _client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False, base_url=PUBLIC_URL)


def _start_login(provider: str, redirect_uri: str = REDIRECT):
    """A browser starts a login: returns (state, binding cookie, set-cookie header, verifier, client_id)."""
    client = _client()
    client_id = client.post("/oauth/register", json={"redirect_uris": [redirect_uri]}).json()["client_id"]
    verifier, challenge = _pkce()
    page = client.get(
        "/oauth/authorize",
        params={
            "response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri,
            "code_challenge": challenge, "code_challenge_method": "S256", "state": "clientstate",
        },
    )
    assert page.status_code == 200
    body = html.unescape(page.text)
    if provider == "google":
        state = parse_qs(urlparse(re.search(r'href="(https://accounts\.google\.com[^"]+)"', page.text)[1].replace("&amp;", "&")).query)["state"][0]
    else:
        state = re.search(r"supabase/callback\?state=([A-Za-z0-9_.\-=]+)", body)[1]
    set_cookie = page.headers["set-cookie"]
    cookie = re.match(rf"{COOKIE}=([^;]+)", set_cookie)[1]
    return state, cookie, set_cookie, verifier, client_id


def _callback(provider: str, state: str, cookie: str | None):
    """The IdP sends the browser back to us. ``cookie`` is what that browser holds."""
    headers = {"cookie": f"{COOKIE}={cookie}"} if cookie is not None else {}
    client = _client()
    if provider == "google":
        return client.get(
            "/oauth/google/callback", params={"code": "idp-code", "state": state},
            headers=headers, follow_redirects=False,
        )
    return client.post(
        "/oauth/supabase/callback", data={"access_token": _sb(), "state": state},
        headers=headers, follow_redirects=False,
    )


_sb = None


@pytest.fixture(autouse=True)
def _wire_sb(env):
    global _sb
    _sb = env["sb_token"]
    yield
    _sb = None


def _assert_refused(resp, status=403):
    assert resp.status_code == status
    assert "location" not in resp.headers
    assert "code=" not in resp.text
    assert "Sign-in failed" in resp.text


# ── the cookie is set when the login starts ───────────────────────────────


def test_login_start_sets_hardened_binding_cookie(provider):
    _, cookie, set_cookie, _, _ = _start_login(provider)
    attrs = set_cookie.lower()
    assert "httponly" in attrs
    assert "secure" in attrs
    assert "samesite=lax" in attrs
    assert "path=/oauth" in attrs
    assert f"max-age={lp.LOGIN_BINDING_MAX_AGE}" in attrs
    assert lp.verify_login_binding(cookie, settings.neuralscape_user_token_secret)


def test_cookie_not_secure_on_plain_http_deployment(provider, monkeypatch):
    monkeypatch.setattr(settings, "neuralscape_public_url", "http://localhost:8199")
    client = TestClient(app, raise_server_exceptions=False, base_url="http://localhost:8199")
    cid = client.post("/oauth/register", json={"redirect_uris": [REDIRECT]}).json()["client_id"]
    _, challenge = _pkce()
    page = client.get(
        "/oauth/authorize",
        params={"response_type": "code", "client_id": cid, "redirect_uri": REDIRECT,
                "code_challenge": challenge, "code_challenge_method": "S256"},
    )
    assert "secure" not in page.headers["set-cookie"].lower()


# ── happy path: same browser, code bound to the original PKCE challenge ──


def test_same_browser_login_issues_code_and_clears_cookie(provider):
    state, cookie, _, verifier, client_id = _start_login(provider)
    resp = _callback(provider, state, cookie)
    assert resp.status_code == 303
    loc = urlparse(resp.headers["location"])
    assert f"{loc.scheme}://{loc.netloc}{loc.path}" == REDIRECT
    q = parse_qs(loc.query)
    assert q["state"] == ["clientstate"]
    cleared = resp.headers["set-cookie"].lower()
    assert f"{COOKIE}=" in cleared and ("max-age=0" in cleared or "expires=" in cleared)
    assert "path=/oauth" in cleared

    # The code is bound to the ORIGINAL authorize request's S256 challenge.
    code = q["code"][0]
    claims = lp.verify_payload(code, settings.neuralscape_user_token_secret)
    assert claims["user_id"] == VICTIM
    assert claims["code_challenge"] == _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    http = _client()
    wrong = http.post("/oauth/token", data={
        "grant_type": "authorization_code", "code": code, "client_id": client_id,
        "redirect_uri": REDIRECT, "code_verifier": _b64url(secrets.token_bytes(40)),
    })
    assert wrong.status_code == 400 and wrong.json()["error"] == "invalid_grant"
    ok = http.post("/oauth/token", data={
        "grant_type": "authorization_code", "code": code, "client_id": client_id,
        "redirect_uri": REDIRECT, "code_verifier": verifier,
    })
    assert ok.status_code == 200 and "access_token" in ok.json()


# ── refusals: no code is ever issued ──────────────────────────────────────


def test_callback_without_cookie_refused(provider):
    state, *_ = _start_login(provider)
    _assert_refused(_callback(provider, state, None))


def test_forged_cross_browser_callback_refused(provider):
    """The attack: state minted for the attacker's client, completed by the victim."""
    attacker_state, _attacker_cookie, *_ = _start_login(provider, ATTACKER_REDIRECT)
    _victim_state, victim_cookie, *_ = _start_login(provider)  # the victim's own browser cookie
    resp = _callback(provider, attacker_state, victim_cookie)
    _assert_refused(resp)
    # and the victim's genuine login is not destroyed by the forged hit
    assert "max-age=0" not in resp.headers.get("set-cookie", "").lower()


def test_victim_without_any_cookie_cannot_complete_attacker_state(provider):
    attacker_state, *_ = _start_login(provider, ATTACKER_REDIRECT)
    _assert_refused(_callback(provider, attacker_state, None))


def test_replayed_state_refused(provider):
    state, cookie, *_ = _start_login(provider)
    assert _callback(provider, state, cookie).status_code == 303
    _assert_refused(_callback(provider, state, cookie))  # same cookie value presented again


def test_expired_cookie_refused(provider, env):
    state, *_ = _start_login(provider)
    nonce = lp.verify_login_state_nonce(state, env["secret"])[1]
    expired = sign_payload({"typ": "login_bind", "nonce": nonce, "exp": int(time.time()) - 5}, env["secret"])
    _assert_refused(_callback(provider, state, expired))


def test_tampered_cookie_refused(provider, env):
    state, cookie, *_ = _start_login(provider)
    payload, sig = cookie.split(".", 1)
    flipped = ("A" if sig[0] != "A" else "B") + sig[1:]
    _assert_refused(_callback(provider, state, f"{payload}.{flipped}"))
    # a cookie signed with some other secret
    nonce = lp.verify_login_state_nonce(state, env["secret"])[1]
    forged = lp.sign_login_binding(nonce, secrets.token_hex(32))
    _assert_refused(_callback(provider, state, forged))


def test_state_token_is_not_accepted_as_binding_cookie(provider, env):
    state, *_ = _start_login(provider)
    _assert_refused(_callback(provider, state, state))


def test_expired_state_refused(provider, env):
    state, cookie, *_ = _start_login(provider)
    claims = lp.verify_payload(state, env["secret"])
    claims["exp"] = int(time.time()) - 5
    resp = _callback(provider, sign_payload(claims, env["secret"]), cookie)
    _assert_refused(resp, status=400)


def test_redis_down_fails_closed_and_keeps_cookie(provider, env):
    state, cookie, *_ = _start_login(provider)
    env["store"]["down"] = True
    resp = _callback(provider, state, cookie)
    _assert_refused(resp, status=503)
    assert "max-age=0" not in resp.headers.get("set-cookie", "").lower()
    env["store"]["down"] = False
    assert _callback(provider, state, cookie).status_code == 303  # retry works, nothing was consumed


def test_login_state_without_nonce_refused(provider, env):
    state, cookie, *_ = _start_login(provider)
    claims = lp.verify_payload(state, env["secret"])
    claims.pop("nonce")
    _assert_refused(_callback(provider, sign_payload(claims, env["secret"]), cookie), status=400)
