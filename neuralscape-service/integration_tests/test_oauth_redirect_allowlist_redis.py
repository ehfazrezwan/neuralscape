"""OAUTH_REDIRECT_HOST_ALLOWLIST set, against a REAL Redis.

With the list set, drive register -> authorize -> paste-login -> token through the
real FastAPI app (the single-use code marker lives in the real Redis):

* a listed redirect host completes the whole flow and the access token works on a
  protected route;
* an unlisted host is refused at registration, and a client registered before the
  list was set is refused at GET and POST authorize with no code issued;
* a loopback redirect (native client) still completes while the list is set.

An unreachable Redis fails the test through the ``stores`` fixture on the gate.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from main import app
from tokens import issue_user_token, verify_payload

pytestmark = pytest.mark.integration

PUBLIC_URL = "https://neuralscape.example.com"
LISTED = "https://claude.ai/api/mcp/auth_callback"
UNLISTED = "https://attacker.example.net/cb"
LOOPBACK = "http://127.0.0.1:53682/callback"
PROTECTED = "/v1/queue/status"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


@pytest.fixture()
def oauth(stores, unique_ns, flags):
    secret = secrets.token_hex(32)
    flags(
        neuralscape_user_token_secret=secret,
        neuralscape_public_url=PUBLIC_URL,
        neuralscape_api_key="",
        auth_provider="token",
        auth_allow_token_paste=True,
        redis_url=stores.redis_url,
    )

    import redis

    jtis: list[str] = []

    def _cleanup():
        client = redis.Redis.from_url(stores.redis_url)
        try:
            if jtis:
                client.delete(*[f"oauth:code:{j}" for j in jtis])
        finally:
            client.close()

    unique_ns.on_cleanup(_cleanup)

    class Ctx:
        user = unique_ns.user_id
        web = TestClient(app, raise_server_exceptions=False, base_url=PUBLIC_URL)

        @staticmethod
        def set_list(value: str) -> None:
            flags(oauth_redirect_host_allowlist=value)

        @staticmethod
        def track(code: str) -> None:
            jtis.append(verify_payload(code, secret)["jti"])

        @staticmethod
        def login_token() -> str:
            return issue_user_token(unique_ns.user_id, secret, 600)

    return Ctx


def _pkce():
    verifier = _b64url(secrets.token_bytes(40))
    return verifier, _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def _authorize_params(client_id, redirect_uri, challenge):
    return {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": "st",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }


def _full_flow(ctx, redirect_uri):
    """register -> consent page -> paste-login -> token; returns the token response."""
    reg = ctx.web.post("/oauth/register", json={"redirect_uris": [redirect_uri]})
    assert reg.status_code == 201, reg.text
    client_id = reg.json()["client_id"]
    verifier, challenge = _pkce()
    params = _authorize_params(client_id, redirect_uri, challenge)

    page = ctx.web.get("/oauth/authorize", params={"response_type": "code", **params})
    assert page.status_code == 200 and "Authorize" in page.text

    resp = ctx.web.post(
        "/oauth/authorize", data={**params, "token": ctx.login_token()}, follow_redirects=False
    )
    assert resp.status_code == 303, resp.text
    assert resp.headers["location"].startswith(redirect_uri)
    code = parse_qs(urlparse(resp.headers["location"]).query)["code"][0]
    ctx.track(code)

    return ctx.web.post(
        "/oauth/token",
        data=dict(
            grant_type="authorization_code",
            code=code,
            code_verifier=verifier,
            redirect_uri=redirect_uri,
            client_id=client_id,
        ),
    )


def test_listed_host_completes_the_full_flow(oauth):
    oauth.set_list("claude.ai")
    tok = _full_flow(oauth, LISTED)
    assert tok.status_code == 200, tok.text
    hdr = {"Authorization": f"Bearer {tok.json()['access_token']}"}
    # The token carries the logged-in identity: asking for someone else is refused.
    mismatch = oauth.web.get(f"{PROTECTED}?user_id=someone-else", headers=hdr)
    assert mismatch.status_code == 400 and oauth.user in mismatch.text


def test_loopback_native_client_completes_the_flow_with_the_list_set(oauth):
    oauth.set_list("claude.ai")
    tok = _full_flow(oauth, LOOPBACK)
    assert tok.status_code == 200, tok.text


def test_unlisted_host_is_refused_end_to_end(oauth):
    # Registered while the list was empty (an attacker-registered or older client).
    reg = oauth.web.post("/oauth/register", json={"redirect_uris": [UNLISTED]})
    assert reg.status_code == 201
    client_id = reg.json()["client_id"]
    _verifier, challenge = _pkce()
    params = _authorize_params(client_id, UNLISTED, challenge)

    oauth.set_list("claude.ai")

    # A fresh registration for the same host is refused.
    refused = oauth.web.post("/oauth/register", json={"redirect_uris": [UNLISTED]})
    assert refused.status_code == 400 and refused.json()["error"] == "invalid_redirect_uri"

    # The earlier client cannot reach consent or log in.
    page = oauth.web.get("/oauth/authorize", params={"response_type": "code", **params})
    assert page.status_code == 400 and page.json()["error"] == "invalid_request"
    resp = oauth.web.post(
        "/oauth/authorize", data={**params, "token": oauth.login_token()}, follow_redirects=False
    )
    assert resp.status_code == 400
    assert "location" not in resp.headers  # no authorization code was issued
