"""Two scenarios here are EXPECTED RED UNTIL PR #495 (sec-03): the consent-page framing headers and refresh-token rotation.

The built-in OAuth 2.1 authorization server (what lets a claude.ai/Cowork-style client connect).

Profile ``flags-on`` turns it on with NEURALSCAPE_PUBLIC_URL (AUTH_PROVIDER stays ``token``) and
folds scripts/oauth_login_test.py into an automatic flow: discovery, dynamic client registration,
the consent page, a login by pasting an admin-issued token, the PKCE code exchange, the access token
used as a Bearer on REST and MCP, refresh, and replay protection.

Profiles ``oauth-google`` and ``oauth-supabase`` prove everything up to the external identity
provider: the consent page offers the provider's button wired to the configured client, and the
provider callbacks refuse a login whose signed state is missing or invalid. Google stops there (it
needs a real identity provider; the redirect to Google and the identity token coming back are not
exercised), so ``auth.login-google`` is not claimed. Supabase is proven end to end against a fake
Supabase auth service in ``test_oauth_supabase_login.py``, which claims ``auth.login-supabase``.
"""

from __future__ import annotations

import base64
import hashlib
import html
import re
import secrets
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

import pytest

from nsaccept import expect_check_fails, negative_control, precondition, profiles, proves
from nsaccept.client import NSClient
from nsaccept.helpers import list_rows
from nsaccept.mcp import MCPClient

REDIRECT = "http://127.0.0.1:8765/callback"


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def pkce() -> tuple[str, str]:
    verifier = b64url(secrets.token_bytes(32))
    return verifier, b64url(hashlib.sha256(verifier.encode()).digest())


def public_url(ctx) -> str:
    url = ctx.profile_flags.get("NEURALSCAPE_PUBLIC_URL", "")
    assert url, "the profile sets no NEURALSCAPE_PUBLIC_URL"
    assert url.rstrip("/") == ctx.base_url, (
        f"the profile's NEURALSCAPE_PUBLIC_URL ({url}) is not the URL this run targets ({ctx.base_url}): "
        "run the stack on the profile's port, or edit that line for this run"
    )
    return url.rstrip("/")


# ── the real checks ──────────────────────────────────────────────


def check_as_metadata(meta: dict, base: str) -> None:
    assert meta.get("issuer") == base, f"issuer {meta.get('issuer')!r} != {base!r}"
    for key, path in (("authorization_endpoint", "/oauth/authorize"), ("token_endpoint", "/oauth/token"), ("registration_endpoint", "/oauth/register")):
        assert meta.get(key) == f"{base}{path}", f"{key} {meta.get(key)!r}"
    assert meta.get("code_challenge_methods_supported") == ["S256"], f"PKCE methods {meta.get('code_challenge_methods_supported')}"
    assert {"authorization_code", "refresh_token"} <= set(meta.get("grant_types_supported", [])), f"grants {meta.get('grant_types_supported')}"


def check_bearer_identifies(base_url: str, token: str, user_id: str, a_client: NSClient, marker: str) -> None:
    """The token authenticates REST, and the write it makes belongs to ``user_id``."""
    c = NSClient(base_url, token)
    try:
        task = c.store_raw(f"{marker} The signal lamp is trimmed by the dock crew.", "preference", visibility="private")
        assert c.poll_task(task, timeout=120)["status"] == "completed", "write with the OAuth access token did not complete"
        rows = [r for r in list_rows(a_client) if marker in r.get("memory", "")]
        assert rows and rows[0].get("owner_user_id") == user_id, f"write made with the access token is not user {user_id}'s: {rows[:1]}"
    finally:
        c.close()


def check_mcp_accepts(base_url: str, token: str) -> None:
    m = MCPClient(base_url, token)
    try:
        m.initialize()
        assert len(m.list_tools()) > 0, "MCP tools/list returned no tools for the access token"
    finally:
        m.close()


def check_refused(resp, what: str, *, status: int = 400, error: str = "invalid_grant") -> None:
    assert resp.status_code == status, f"{what} -> {resp.status_code} (want {status}): {resp.text[:160]}"
    assert resp.json().get("error") == error, f"{what}: error {resp.json().get('error')!r}"


# ── a full login, recorded once ──────────────────────────────────


@dataclass
class Login:
    client_id: str = ""
    verifier: str = ""
    state: str = ""
    code: str = ""
    tokens: dict = field(default_factory=dict)
    refreshed: dict = field(default_factory=dict)
    replay_code: object = None
    error: str = ""


def register(anon: NSClient, redirect: str = REDIRECT) -> str:
    r = anon.post("/oauth/register", json={"redirect_uris": [redirect]})
    assert r.status_code == 201, f"POST /oauth/register -> {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert body["client_id"] and body["token_endpoint_auth_method"] == "none" and body["redirect_uris"] == [redirect], f"DCR body: {body}"
    return body["client_id"]


def authorize_url_params(client_id: str, challenge: str, state: str, redirect: str = REDIRECT) -> dict:
    return {
        "response_type": "code", "client_id": client_id, "redirect_uri": redirect, "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256",
    }


def paste_login(anon: NSClient, client_id: str, challenge: str, state: str, token: str, redirect: str = REDIRECT):
    return anon.post("/oauth/authorize", data={
        "client_id": client_id, "redirect_uri": redirect, "state": state, "code_challenge": challenge,
        "code_challenge_method": "S256", "token": token,
    })


def get_code(anon: NSClient, token: str) -> tuple[str, str, str, str]:
    """(client_id, verifier, state, code) from a complete DCR -> consent -> paste-login round."""
    client_id = register(anon)
    verifier, challenge = pkce()
    state = secrets.token_urlsafe(8)
    page = anon.get("/oauth/authorize", params=authorize_url_params(client_id, challenge, state))
    assert page.status_code == 200, f"GET /oauth/authorize -> {page.status_code}: {page.text[:200]}"
    r = paste_login(anon, client_id, challenge, state, token)
    assert r.status_code == 303, f"POST /oauth/authorize with a valid token -> {r.status_code}: {r.text[:200]}"
    loc = urlparse(r.headers["location"])
    q = parse_qs(loc.query)
    assert f"{loc.scheme}://{loc.netloc}{loc.path}" == REDIRECT, f"redirected to {r.headers['location']}"
    assert q.get("state") == [state], f"state not echoed: {q.get('state')} != {state}"
    assert q.get("code"), "redirect carries no code"
    return client_id, verifier, state, q["code"][0]


def exchange(anon: NSClient, client_id: str, verifier: str, code: str, redirect: str = REDIRECT):
    return anon.post("/oauth/token", data={
        "grant_type": "authorization_code", "code": code, "code_verifier": verifier,
        "redirect_uri": redirect, "client_id": client_id,
    })


@pytest.fixture(scope="module")
def login(ctx, anon) -> Login:
    lg = Login()
    try:
        lg.client_id, lg.verifier, lg.state, lg.code = get_code(anon, ctx.token_a)
        r = exchange(anon, lg.client_id, lg.verifier, lg.code)
        assert r.status_code == 200, f"POST /oauth/token (code) -> {r.status_code}: {r.text[:200]}"
        lg.tokens = r.json()
        lg.replay_code = exchange(anon, lg.client_id, lg.verifier, lg.code)
        time.sleep(1.1)  # access tokens are stateless with 1 s granularity: a same-second refresh can return identical bytes
        rf = anon.post("/oauth/token", data={"grant_type": "refresh_token", "refresh_token": lg.tokens["refresh_token"]})
        assert rf.status_code == 200, f"POST /oauth/token (refresh) -> {rf.status_code}: {rf.text[:200]}"
        lg.refreshed = rf.json()
    except AssertionError as exc:
        lg.error = str(exc)
    return lg


def _ok(lg: Login) -> Login:
    assert not lg.error, lg.error
    return lg


# ═══ profile flags-on: the token-paste authorization server ═════════════


@profiles("flags-on")
@proves("auth.oauth-server")
def test_discovery_metadata_names_this_server_as_issuer(ctx, anon):
    base = public_url(ctx)
    meta = anon.get("/.well-known/oauth-authorization-server")
    assert meta.status_code == 200, f"AS metadata -> {meta.status_code}: {meta.text[:160]}"
    check_as_metadata(meta.json(), base)
    for path in ("/.well-known/oauth-protected-resource", "/.well-known/oauth-protected-resource/mcp"):
        r = anon.get(path)
        assert r.status_code == 200, f"GET {path} -> {r.status_code}"
        assert r.json()["resource"] == f"{base}/mcp" and r.json()["authorization_servers"] == [base], f"{path}: {r.json()}"


@profiles("flags-on")
@proves("auth.oauth-server")
@negative_control
def test_metadata_check_fails_for_another_issuer_and_missing_pkce(ctx, anon):
    meta = anon.get("/.well-known/oauth-authorization-server").json()
    precondition(check_as_metadata, meta, public_url(ctx), what="the served metadata names this server")
    expect_check_fails(check_as_metadata, meta, "https://elsewhere.example")
    expect_check_fails(check_as_metadata, {**meta, "code_challenge_methods_supported": ["plain"]}, public_url(ctx))


@profiles("flags-on")
@proves("auth.oauth-server")
def test_dynamic_client_registration_accepts_https_and_loopback_only(anon):
    assert register(anon, "https://client.example/callback")
    assert register(anon, "http://localhost:9/cb")
    for bad in ("http://client.example/callback", "ftp://127.0.0.1/cb", "https://client.example/cb#frag"):
        r = anon.post("/oauth/register", json={"redirect_uris": [bad]})
        assert r.status_code == 400 and r.json()["error"] == "invalid_redirect_uri", f"{bad} -> {r.status_code}: {r.text[:120]}"
    assert anon.post("/oauth/register", json={}).status_code == 400, "an empty registration was accepted"


@profiles("flags-on")
@proves("auth.oauth-server", "auth.consent-page")
def test_the_consent_page_offers_the_token_login_and_validates_the_request(ctx, anon):
    client_id = register(anon)
    _, challenge = pkce()
    page = anon.get("/oauth/authorize", params=authorize_url_params(client_id, challenge, "st-1"))
    assert page.status_code == 200 and "text/html" in page.headers["content-type"], f"consent page: {page.status_code} {page.headers.get('content-type')}"
    for needle in ('action="/oauth/authorize"', 'name="token"', f'value="{html.escape(client_id)}"', 'name="code_challenge"', "Connect Neuralscape"):
        assert needle in page.text, f"consent page lacks {needle!r}"
    # the stylesheet always defines .provider-btn, so look for an ELEMENT using it, or a link to a provider
    assert not re.search(r"""<(a|button)\b[^>]*class=["'][^"']*\bprovider-btn\b""", page.text), "token mode shows a provider button element"
    assert "accounts.google.com/o/oauth2" not in page.text and "ns-google-btn" not in page.text, "token mode links to an identity provider"
    bad_cases = {
        "response_type": {**authorize_url_params(client_id, challenge, "s"), "response_type": "token"},
        "unknown client": {**authorize_url_params("not-a-client", challenge, "s")},
        "unregistered redirect_uri": {**authorize_url_params(client_id, challenge, "s", redirect="https://evil.example/cb")},
        "no PKCE": {k: v for k, v in authorize_url_params(client_id, challenge, "s").items() if k != "code_challenge"},
        "plain PKCE": {**authorize_url_params(client_id, challenge, "s"), "code_challenge_method": "plain"},
    }
    for what, params in bad_cases.items():
        r = anon.get("/oauth/authorize", params=params)
        assert r.status_code == 400 and "location" not in r.headers, f"{what}: GET /oauth/authorize -> {r.status_code} (want a local 400, never a redirect)"


@profiles("flags-on")
@proves("auth.consent-page")
def test_the_consent_page_cannot_be_framed(anon):
    """EXPECTED RED UNTIL PR #495 (sec-03) merges. A login page that can be framed lets another site
    overlay it; wanted: X-Frame-Options DENY or CSP frame-ancestors 'none'. Not weakened."""
    client_id = register(anon)
    _, challenge = pkce()
    r = anon.get("/oauth/authorize", params=authorize_url_params(client_id, challenge, "st-2"))
    xfo = r.headers.get("x-frame-options", "").upper()
    csp = r.headers.get("content-security-policy", "").lower()
    assert xfo == "DENY" or "frame-ancestors 'none'" in csp, f"consent page is frameable: X-Frame-Options={xfo!r} CSP={csp!r}"


@profiles("flags-on")
@proves("auth.consent-page")
@negative_control
def test_frame_check_fails_for_a_page_without_the_headers(anon):
    r = anon.get("/health")  # an ordinary page: no anti-framing headers asked of it

    def check():
        assert r.headers.get("x-frame-options", "").upper() == "DENY" or "frame-ancestors 'none'" in r.headers.get("content-security-policy", "").lower(), "frameable"

    expect_check_fails(check)


@profiles("flags-on")
@proves("auth.oauth-server")
def test_a_pasted_admin_token_logs_in_and_the_code_exchange_issues_tokens(ctx, anon, login: Login):
    lg = _ok(login)
    t = lg.tokens
    assert t["token_type"] == "Bearer" and t["access_token"] and t["refresh_token"] and t["expires_in"] > 0, f"token response keys: {sorted(t)}"
    payload = base64.urlsafe_b64decode(t["access_token"].split(".")[0] + "==")
    assert ctx.user_a.encode() in payload, "the access token does not carry user A's identity"
    # an invalid paste is answered with the consent page (400) and an error, never a code
    client_id = register(anon)
    _, challenge = pkce()
    r = paste_login(anon, client_id, challenge, "st-3", "not-a-token")
    assert r.status_code == 400 and "invalid or expired" in r.text and "location" not in r.headers, f"bad paste -> {r.status_code}: {r.text[:160]}"
    assert paste_login(anon, client_id, challenge, "st-3", ctx.bad_token).status_code == 400, "a token signed with the wrong secret logged in"


@profiles("flags-on")
@proves("auth.oauth-server")
def test_the_code_is_bound_to_its_client_redirect_and_verifier(ctx, anon):
    client_id, verifier, _, code = get_code(anon, ctx.token_a)
    check_refused(exchange(anon, client_id, "x" * 43, code), "wrong code_verifier")
    check_refused(exchange(anon, client_id, verifier, code, redirect="http://127.0.0.1:9/other"), "wrong redirect_uri")
    check_refused(exchange(anon, register(anon), verifier, code), "another client's id")
    check_refused(exchange(anon, client_id, verifier, "garbage"), "a garbage code")
    check_refused(anon.post("/oauth/token", data={"grant_type": "password", "username": "x"}), "an unsupported grant", error="unsupported_grant_type")
    assert exchange(anon, client_id, verifier, code).status_code == 200, "the right verifier/redirect/client no longer redeems the code (a failed attempt consumed it)"


@profiles("flags-on")
@proves("auth.oauth-server")
def test_an_authorization_code_redeems_once(login: Login):
    lg = _ok(login)
    check_refused(lg.replay_code, "replaying a redeemed code")


@profiles("flags-on")
@proves("auth.oauth-server")
def test_the_access_token_is_a_bearer_for_rest_and_mcp_and_identifies_the_user(ctx, a, login: Login):
    lg = _ok(login)
    check_bearer_identifies(ctx.base_url, lg.tokens["access_token"], ctx.user_a, a, f"zq-{ctx.run_id}-bearerlogin")
    check_mcp_accepts(ctx.base_url, lg.tokens["access_token"])


@profiles("flags-on")
@proves("auth.oauth-server")
@negative_control
def test_bearer_check_fails_for_a_token_signed_with_the_wrong_secret(ctx, a, login: Login):
    precondition(check_bearer_identifies, ctx.base_url, _ok(login).tokens["access_token"], ctx.user_a, a, f"zq-{ctx.run_id}-bearerpre", what="a real access token identifies A")
    expect_check_fails(check_bearer_identifies, ctx.base_url, ctx.bad_token, ctx.user_a, a, f"zq-{ctx.run_id}-bearerwrong")


@profiles("flags-on")
@proves("auth.oauth-server")
def test_a_refresh_token_yields_a_working_access_token_for_the_same_user(ctx, a, login: Login):
    lg = _ok(login)
    new = lg.refreshed
    assert new["access_token"] and new["access_token"] != lg.tokens["access_token"], "refresh returned the same access token"
    check_bearer_identifies(ctx.base_url, new["access_token"], ctx.user_a, a, f"zq-{ctx.run_id}-bearerrefreshed")


@profiles("flags-on")
@proves("auth.oauth-server")
def test_a_refresh_token_is_consumed_when_it_rotates(anon, login: Login):
    """EXPECTED RED UNTIL PR #495 (sec-03) merges. RFC 6749 sec 10.4 / OAuth 2.1: a refresh token is
    single-use and each refresh issues a new one. The first refresh already happened in the login
    fixture. Not weakened."""
    lg = _ok(login)
    assert lg.refreshed.get("refresh_token") and lg.refreshed["refresh_token"] != lg.tokens["refresh_token"], "refresh did not rotate the refresh token"
    replay = anon.post("/oauth/token", data={"grant_type": "refresh_token", "refresh_token": lg.tokens["refresh_token"]})
    check_refused(replay, "replaying a consumed refresh token")


@profiles("flags-on")
@proves("auth.oauth-server")
@negative_control
def test_refused_check_fails_for_a_successful_redemption(ctx, anon, login: Login):
    precondition(check_refused, _ok(login).replay_code, "a replayed code", what="a replayed code is refused")
    client_id, verifier, _, code = get_code(anon, ctx.token_a)
    expect_check_fails(check_refused, exchange(anon, client_id, verifier, code), "a good redemption")


# ═══ provider profiles: everything up to the identity provider ═════════════


def consent_with_provider(anon: NSClient) -> str:
    client_id = register(anon)
    _, challenge = pkce()
    r = anon.get("/oauth/authorize", params=authorize_url_params(client_id, challenge, "st-p"))
    assert r.status_code == 200, f"GET /oauth/authorize -> {r.status_code}: {r.text[:200]}"
    return r.text


@profiles("oauth-google")
@proves("auth.consent-page", "auth.oauth-server")
def test_google_consent_page_links_to_google_with_the_configured_client_and_our_callback(ctx, anon):
    text = consent_with_provider(anon)
    m = re.search(r'href="(https://accounts\.google\.com/o/oauth2/v2/auth\?[^"]+)"', text)
    assert m, "no Sign-in-with-Google link on the consent page"
    q = parse_qs(urlparse(html.unescape(m.group(1))).query)
    assert q["client_id"] == [ctx.profile_flags["GOOGLE_OAUTH_CLIENT_ID"]], f"client_id {q.get('client_id')}"
    assert q["redirect_uri"] == [f"{public_url(ctx)}/oauth/google/callback"], f"redirect_uri {q.get('redirect_uri')}"
    assert q["response_type"] == ["code"] and "openid" in q["scope"][0] and q["state"][0].count(".") == 1, f"google params: {sorted(q)}"
    assert 'name="token"' in text, "the admin-token paste box vanished under the Google provider"  # AUTH_ALLOW_TOKEN_PASTE defaults on
    # the signed state carries the request across the redirect: it is long, signed, and not our client id
    assert len(q["state"][0]) > 100, "state looks unsigned"


@profiles("oauth-google")
@proves("auth.consent-page")
def test_google_callback_refuses_a_login_without_a_valid_signed_state(anon):
    # STOP: past this point Google itself must authenticate someone; nothing here can do that.
    for params in ({}, {"state": "garbage", "code": "x"}, {"state": "a.b", "code": "x"}, {"error": "access_denied"}):
        r = anon.get("/oauth/google/callback", params=params)
        assert r.status_code == 400 and "location" not in r.headers, f"GET /oauth/google/callback {params} -> {r.status_code}: {r.text[:120]}"
        assert "code=" not in r.headers.get("location", ""), "the callback redirected with a code"
    other = anon.get("/oauth/supabase/callback")
    assert other.status_code == 404 and "not enabled" in other.text, f"the supabase callback is mounted under the google provider: {other.status_code}"


@profiles("oauth-google")
@proves("auth.oauth-server")
def test_a_pasted_token_still_logs_in_under_the_google_provider(ctx, anon):
    get_code(anon, ctx.token_a)


@profiles("oauth-google", "oauth-supabase")
@proves("auth.consent-page")
@negative_control
def test_provider_button_check_fails_for_the_token_consent_page(anon):
    def check(page: str):
        assert "accounts.google.com/o/oauth2" in page or "ns-google-btn" in page, "no provider login control"

    precondition(lambda: check(consent_with_provider(anon)), what="this profile's consent page shows its provider control")
    # the discovery document is not a consent page and has no provider button
    expect_check_fails(check, anon.get("/.well-known/oauth-authorization-server").text)


@profiles("oauth-supabase")
@proves("auth.consent-page", "auth.oauth-server")
def test_supabase_consent_page_wires_the_sign_in_button_to_the_configured_project(ctx, anon):
    text = consent_with_provider(anon)
    assert 'id="ns-google-btn"' in text, "no Supabase sign-in button on the consent page"
    assert ctx.profile_flags["SUPABASE_URL"] in text, "the consent page does not use the configured SUPABASE_URL"
    assert f"{public_url(ctx)}/oauth/supabase/callback?state=" in html.unescape(text), "the redirectTo is not this server's Supabase callback"
    assert 'name="token"' in text, "the admin-token paste box vanished under the Supabase provider"


@profiles("oauth-supabase")
@proves("auth.consent-page")
def test_supabase_callback_refuses_a_login_without_a_valid_signed_state(anon):
    # STOP: a Supabase session JWT can only come from a real Supabase project.
    page = anon.get("/oauth/supabase/callback")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"], f"GET callback -> {page.status_code}"
    for form in ({}, {"access_token": "a.b.c", "state": "garbage"}, {"access_token": "", "state": "a.b"}):
        r = anon.post("/oauth/supabase/callback", data=form)
        assert r.status_code == 400 and "location" not in r.headers, f"POST /oauth/supabase/callback {form} -> {r.status_code}: {r.text[:120]}"
    other = anon.get("/oauth/google/callback")
    assert other.status_code == 404 and "not enabled" in other.text, f"the google callback is mounted under the supabase provider: {other.status_code}"


@profiles("oauth-supabase")
@proves("auth.oauth-server")
def test_a_pasted_token_still_logs_in_under_the_supabase_provider(ctx, anon):
    get_code(anon, ctx.token_a)


@profiles("oauth-google", "oauth-supabase")
@proves("auth.consent-page")
def test_the_authorize_response_binds_the_login_to_the_browser_that_started_it(anon):
    """Login-state binding. GET /oauth/authorize must set the login-binding cookie: HttpOnly, SameSite=Lax,
    named ns_login_bind (plain http) or __Host-ns_login_bind (https public URL), so a login state built by
    someone else cannot be completed in a victim's browser."""
    client_id = register(anon)
    _, challenge = pkce()
    r = anon.get("/oauth/authorize", params=authorize_url_params(client_id, challenge, "st-bind"))
    assert r.status_code == 200, f"GET /oauth/authorize -> {r.status_code}"
    cookies = [c for c in r.headers.get_list("set-cookie") if "ns_login_bind=" in c]
    assert len(cookies) == 1, f"the authorize response sets {len(cookies)} login-binding cookies (want 1): {r.headers.get_list('set-cookie')}"
    low = cookies[0].lower()
    assert "httponly" in low and "samesite=lax" in low, f"binding cookie attributes: {cookies[0].split('=')[0]}; {low.split(';', 1)[1] if ';' in low else ''}"
