"""Full OAuth flow with the flag ON against a real Redis (sec-03).

OAuth is only offered when NEURALSCAPE_PUBLIC_URL and the signing secret are
both set, so this sets both and drives the whole flow through the FastAPI app:
discovery -> register -> consent page (anti-framing headers) -> authorize with a
pasted admin token -> code -> PKCE token exchange -> the access token on a real
protected route -> refresh rotation -> reuse / wrong-client / code-replay
rejection. The single-use markers live in the real Redis the gate provides; an
unreachable Redis is a test failure (the guards fail closed, so no token would
be issued and the flow assertions would fail). No skips.
"""

import base64
import hashlib
import secrets
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from config import settings
from main import app
from tokens import issue_user_token

pytestmark = pytest.mark.integration

PUBLIC_URL = "https://neuralscape.example.com"
REDIRECT = "https://claude.ai/api/mcp/auth_callback"
USER = "oauth-int-alice"
PROTECTED = "/v1/queue/status"  # auth layer resolves identity before any backend


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


@pytest.fixture()
def oauth_on(monkeypatch):
    secret = secrets.token_hex(32)
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", secret)
    monkeypatch.setattr(settings, "neuralscape_public_url", PUBLIC_URL)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    monkeypatch.setattr(settings, "auth_provider", "token")
    return secret


def _token_post(client, **data):
    return client.post("/oauth/token", data=data)


def test_full_oauth_flow_with_refresh_rotation(oauth_on):
    secret = oauth_on
    client = TestClient(app, raise_server_exceptions=False)

    # discovery
    meta = client.get("/.well-known/oauth-authorization-server").json()
    assert meta["token_endpoint"] == f"{PUBLIC_URL}/oauth/token"
    assert "refresh_token" in meta["grant_types_supported"]

    # dynamic client registration
    client_id = client.post("/oauth/register", json={"redirect_uris": [REDIRECT]}).json()["client_id"]
    other_id = client.post(
        "/oauth/register", json={"redirect_uris": ["https://other.example.com/cb"]}
    ).json()["client_id"]

    # consent page: renders, and cannot be framed
    verifier = _b64url(secrets.token_bytes(40))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    page = client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": REDIRECT,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": "st",
        },
    )
    assert page.status_code == 200 and "Authorize" in page.text
    assert page.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert page.headers["referrer-policy"] == "no-referrer"
    assert page.headers["x-content-type-options"] == "nosniff"

    # authorize with a pasted admin token -> 303 with a code
    admin_token = issue_user_token(USER, secret, 600)
    resp = client.post(
        "/oauth/authorize",
        data={
            "client_id": client_id,
            "redirect_uri": REDIRECT,
            "state": "st",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "token": admin_token,
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    qs = parse_qs(urlparse(resp.headers["location"]).query)
    assert qs["state"] == ["st"]
    code = qs["code"][0]

    exchange = dict(
        grant_type="authorization_code",
        code=code,
        code_verifier=verifier,
        redirect_uri=REDIRECT,
        client_id=client_id,
    )
    tok = _token_post(client, **exchange)
    assert tok.status_code == 200, tok.text
    tokens = tok.json()

    # code replay is rejected (real Redis SET NX)
    replay = _token_post(client, **exchange)
    assert replay.status_code == 400 and replay.json()["error"] == "invalid_grant"

    # access token works on a real protected route and carries the identity
    assert client.get(f"{PROTECTED}?user_id={USER}").status_code == 401
    hdr = {"Authorization": f"Bearer {tokens['access_token']}"}
    mismatch = client.get(f"{PROTECTED}?user_id=someone-else", headers=hdr)
    assert mismatch.status_code == 400 and USER in mismatch.text
    # a refresh token is never a Bearer credential
    assert client.get(PROTECTED, headers={"Authorization": f"Bearer {tokens['refresh_token']}"}).status_code == 401

    # wrong client_id is refused and does NOT burn the refresh token
    wrong = _token_post(
        client, grant_type="refresh_token", refresh_token=tokens["refresh_token"], client_id=other_id
    )
    assert wrong.status_code == 400 and wrong.json()["error"] == "invalid_grant"

    # refresh rotates
    refreshed = _token_post(
        client, grant_type="refresh_token", refresh_token=tokens["refresh_token"], client_id=client_id
    )
    assert refreshed.status_code == 200, refreshed.text
    new = refreshed.json()
    assert new["refresh_token"] != tokens["refresh_token"]
    hdr2 = {"Authorization": f"Bearer {new['access_token']}"}
    assert client.get(f"{PROTECTED}?user_id=someone-else", headers=hdr2).status_code == 400

    # the consumed refresh token is dead
    reuse = _token_post(
        client, grant_type="refresh_token", refresh_token=tokens["refresh_token"], client_id=client_id
    )
    assert reuse.status_code == 400 and reuse.json()["error"] == "invalid_grant"

    # reuse revoked the family, so the successor is dead too
    dead = _token_post(
        client, grant_type="refresh_token", refresh_token=new["refresh_token"], client_id=client_id
    )
    assert dead.status_code == 400 and dead.json()["error"] == "invalid_grant"


def _login(client, secret):
    """Register + authorize + exchange; returns (client_id, tokens)."""
    client_id = client.post("/oauth/register", json={"redirect_uris": [REDIRECT]}).json()["client_id"]
    verifier = _b64url(secrets.token_bytes(40))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    resp = client.post(
        "/oauth/authorize",
        data={
            "client_id": client_id,
            "redirect_uri": REDIRECT,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "token": issue_user_token(USER, secret, 600),
        },
        follow_redirects=False,
    )
    code = parse_qs(urlparse(resp.headers["location"]).query)["code"][0]
    tok = _token_post(
        client,
        grant_type="authorization_code",
        code=code,
        code_verifier=verifier,
        redirect_uri=REDIRECT,
        client_id=client_id,
    )
    assert tok.status_code == 200, tok.text
    return client_id, tok.json()


def test_concurrent_redemption_has_one_winner_and_reuse_revokes_family(oauth_on):
    from concurrent.futures import ThreadPoolExecutor

    client = TestClient(app, raise_server_exceptions=False)
    client_id, tokens = _login(client, oauth_on)
    _, unrelated = _login(client, oauth_on)

    def redeem(_):
        return _token_post(
            TestClient(app, raise_server_exceptions=False),
            grant_type="refresh_token",
            refresh_token=tokens["refresh_token"],
            client_id=client_id,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(redeem, range(2)))
    codes = sorted(r.status_code for r in results)
    assert codes == [200, 400], [r.text for r in results]
    winner = next(r for r in results if r.status_code == 200).json()

    # Reuse (the loser, or any replay) revokes the family: even if the loser
    # lost the race before the revoke landed, a further replay does it.
    replay = _token_post(
        client, grant_type="refresh_token", refresh_token=tokens["refresh_token"], client_id=client_id
    )
    assert replay.status_code == 400
    dead = _token_post(
        client, grant_type="refresh_token", refresh_token=winner["refresh_token"], client_id=client_id
    )
    assert dead.status_code == 400 and dead.json()["error"] == "invalid_grant"

    # an unrelated grant (different family) is unaffected
    ok = _token_post(
        client, grant_type="refresh_token", refresh_token=unrelated["refresh_token"],
        client_id=_client_of(unrelated, oauth_on),
    )
    assert ok.status_code == 200


def _client_of(tokens, secret):
    from tokens import verify_payload

    return verify_payload(tokens["refresh_token"], secret)["client_id"]
