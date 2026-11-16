"""Login-state browser binding against a REAL Redis (sec-15).

The single-use claim of a login nonce lives in Redis (``SET NX EX``), so the
replay guard is only proven here, not by the unit suite's in-memory stand-in.
Drives the real FastAPI app for both federated providers with the IdP
verification stubbed (no real Google / Supabase):

* the browser that started the login completes it once (code issued, PKCE-bound,
  redeemable), and a second callback with the same state and cookie is refused;
* a callback carrying another browser's state (the login CSRF attack: attacker
  starts /oauth/authorize, victim finishes the IdP login) is refused and no
  authorization code is issued.

An unreachable Redis fails the test through the ``stores`` fixture on the gate.
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
from config import settings
from main import app
from tokens import verify_payload

PUBLIC_URL = "https://neuralscape.example.com"
REDIRECT = "https://claude.ai/api/mcp/auth_callback"
ATTACKER_REDIRECT = "https://attacker.example.net/cb"
COOKIE = lp.LOGIN_BINDING_COOKIE


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


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


@pytest.fixture(params=["google", "supabase"])
def stack(request, stores, unique_ns, flags, monkeypatch):
    """OAuth on with the chosen provider, real Redis, stubbed IdP verification."""
    provider = request.param
    secret = secrets.token_hex(32)
    sb_secret = secrets.token_hex(32)
    flags(
        neuralscape_user_token_secret=secret,
        neuralscape_public_url=PUBLIC_URL,
        neuralscape_api_key="",
        auth_provider=provider,
        auth_allow_token_paste=False,
        google_oauth_client_id="google-client-id",
        google_oauth_client_secret=secrets.token_hex(8),
        auth_allowed_domains="example.com",
        supabase_url="https://proj.supabase.example",
        supabase_anon_key=secrets.token_hex(8),
        supabase_jwt_secret=sb_secret,
        redis_url=stores.redis_url,
    )
    monkeypatch.setattr(lp.httpx, "AsyncClient", _FakeGoogleClient)
    monkeypatch.setattr(
        lp, "_verify_oidc_jwt",
        lambda *a, **k: {"email": "victim@example.com", "email_verified": True, "sub": "g-victim"},
    )

    async def _linked(sub, email):
        return unique_ns.user_id, True

    monkeypatch.setattr(lp, "resolve_login_identity", _linked)

    import redis

    nonces: list[str] = []
    jtis: list[str] = []

    def _cleanup():
        client = redis.Redis.from_url(stores.redis_url)
        try:
            keys = [f"oauth:login_nonce:{n}" for n in nonces] + [f"oauth:code:{j}" for j in jtis]
            if keys:
                client.delete(*keys)
        finally:
            client.close()

    unique_ns.on_cleanup(_cleanup)

    class Stack:
        name = provider
        secret_ = secret

        @staticmethod
        def sb_token():
            return jwt.encode(
                {"aud": "authenticated", "email": "victim@example.com", "email_verified": True,
                 "sub": "sb-victim", "exp": int(time.time()) + 300},
                sb_secret, algorithm="HS256",
            )

        @staticmethod
        def track(state=None, code=None):
            if state:
                nonces.append(lp.verify_login_state_nonce(state, secret)[1])
            if code:
                jtis.append(verify_payload(code, secret)["jti"])

    return Stack


def _client():
    return TestClient(app, raise_server_exceptions=False, base_url=PUBLIC_URL)


def _start_login(stack, redirect_uri=REDIRECT):
    client = _client()
    client_id = client.post("/oauth/register", json={"redirect_uris": [redirect_uri]}).json()["client_id"]
    verifier = _b64url(secrets.token_bytes(40))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    page = client.get(
        "/oauth/authorize",
        params={"response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri,
                "code_challenge": challenge, "code_challenge_method": "S256", "state": "clientstate"},
    )
    assert page.status_code == 200
    if stack.name == "google":
        link = html.unescape(re.search(r'href="(https://accounts\.google\.com[^"]+)"', page.text)[1])
        state = parse_qs(urlparse(link).query)["state"][0]
    else:
        state = re.search(r"supabase/callback\?state=([A-Za-z0-9_.\-=]+)", html.unescape(page.text))[1]
    cookie = re.match(rf"{COOKIE}=([^;]+)", page.headers["set-cookie"])[1]
    stack.track(state=state)
    return state, cookie, verifier, client_id


def _callback(stack, state, cookie):
    headers = {"cookie": f"{COOKIE}={cookie}"} if cookie is not None else {}
    client = _client()
    if stack.name == "google":
        return client.get("/oauth/google/callback", params={"code": "idp-code", "state": state},
                          headers=headers, follow_redirects=False)
    return client.post("/oauth/supabase/callback",
                       data={"access_token": stack.sb_token(), "state": state},
                       headers=headers, follow_redirects=False)


def _assert_refused(resp):
    assert resp.status_code in (400, 403)
    assert "location" not in resp.headers
    assert "code=" not in resp.text


def test_replay_guard_on_real_redis(stack):
    state, cookie, verifier, client_id = _start_login(stack)
    first = _callback(stack, state, cookie)
    assert first.status_code == 303
    code = parse_qs(urlparse(first.headers["location"]).query)["code"][0]
    stack.track(code=code)

    # The code is bound to the original authorize request and redeems once.
    exchange = dict(grant_type="authorization_code", code=code, code_verifier=verifier,
                    redirect_uri=REDIRECT, client_id=client_id)
    assert _client().post("/oauth/token", data=exchange).status_code == 200

    # Same state, same cookie, second time: the real Redis claim is already taken.
    _assert_refused(_callback(stack, state, cookie))


def test_cross_browser_callback_refused_end_to_end(stack):
    attacker_state, _attacker_cookie, _, _ = _start_login(stack, ATTACKER_REDIRECT)
    _victim_state, victim_cookie, _, _ = _start_login(stack)

    # Victim's browser (its own cookie) finishes the IdP login for the attacker's state.
    _assert_refused(_callback(stack, attacker_state, victim_cookie))
    # And with no cookie at all.
    _assert_refused(_callback(stack, attacker_state, None))

    # Nothing was consumed by the refusals: the attacker's own browser is unaffected
    # (it is the legitimate holder of that state), and the victim's login still works.
    victim_ok = _callback(stack, _victim_state, victim_cookie)
    assert victim_ok.status_code == 303
    assert urlparse(victim_ok.headers["location"]).netloc == urlparse(REDIRECT).netloc
