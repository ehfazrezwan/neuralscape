"""Sign in via Supabase, END TO END, against a FAKE Supabase auth service (profile ``oauth-supabase``).

What runs, in words (one login):

  1. the OAuth client registers (``/oauth/register``) and opens ``/oauth/authorize`` in a browser (a real
     cookie jar, like a browser) -> the consent page carries the signed ``login_state`` in its
     ``redirectTo``;
  2. the browser step is SIMULATED by the harness: it signs a Supabase-shaped session JWT with a per-run RSA
     key and POSTs ``{access_token, state}`` to ``/oauth/supabase/callback``, which is exactly what the
     sign-in finisher page's form submit does;
  3. the stack fetches the JWKS from ``{SUPABASE_URL}/auth/v1/.well-known/jwks.json`` -- the
     ``nsc-acc-supabase`` sidecar, which serves only the PUBLIC half of that key -- verifies signature,
     audience and expiry, applies the email allowlist, resolves the identity (admin map -> durable link ->
     email slug), shows the first-login link page when the identity is new, and issues a PKCE-bound code;
  4. the client redeems the code at ``/oauth/token`` and uses the access token as a Bearer on ``/v1/*``.

Real: every line of the stack's code path above. Faked: only the identity provider (the JWKS host and the
signing of the session token). Not exercised (no real Supabase can be reached from a throwaway stack): the
supabase-js sign-in in a real browser, Google behind Supabase, and the Supabase Before-User-Created hook
(``auth.supabase-signup-hook``, a SQL script applied by hand in the Supabase dashboard).

Scope: the signature path proven is RS256 via the JWKS (what a modern Supabase project publishes). The
verifier's ES256 branch and its HS256 branch (``SUPABASE_JWT_SECRET``, a shared secret) are NOT exercised:
the profile deliberately sets no secret, and an HS256 token is only used as an algorithm-confusion probe.

The login state is bound to the browser that started the login by PR #536 (a cookie that the callback
requires). The harness carries that cookie the way a browser does; the scenarios that assert the binding
are marked ``ns_expected_red("#536")`` until #536 is on the stack under test.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import secrets
import uuid
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from nsaccept import expect_check_fails, needs, negative_control, profiles, proves
from nsaccept import fakeidp, sidecar
from nsaccept.client import NSClient
from nsaccept.helpers import list_rows
from nsaccept.mcp import MCPClient

pytestmark = [profiles("oauth-supabase"), needs("compose")]

REDIRECT = "http://127.0.0.1:8765/callback"
DOMAIN = "acceptance-placeholder.example"
GUEST_ADDRESS = "acc-guest@acceptance-guest.example"
MAPPED_EMAIL = "acc-mapped@acceptance-placeholder.example"
MAPPED_USER = "acc-supabase-mapped"
#: what the callback says for an unusable login state (the second phrase is the browser-binding refusal of #536)
STATE_REFUSED = ("expired or invalid", "not started in this browser")


def expected_red(reference: str):
    """The same ``ns_expected_red`` marker the coverage-closure PR defines: runs, fails honestly, labelled."""
    return pytest.mark.ns_expected_red(reference)


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


# ── the fake Supabase auth service ───────────────────────────────


@dataclass
class FakeSupabase:
    """The sidecar plus the harness-held signing key."""

    url: str
    key: fakeidp.RsaKey
    other_key: fakeidp.RsaKey
    sink: sidecar.Sink
    hs_secret: bytes = field(repr=False, default=b"")

    def claims(self, sub: str, email: str, **kw) -> dict:
        return fakeidp.supabase_claims(self.url, sub, email, **kw)

    def token(self, sub: str, email: str, **kw) -> str:
        """A session token Supabase would have issued: signed with the key the JWKS publishes."""
        return fakeidp.sign_jwt(self.key, self.claims(sub, email, **kw))

    def forged(self, sub: str, email: str, *, same_kid: bool) -> str:
        """A well-formed token signed by a key the JWKS does not publish."""
        return fakeidp.sign_jwt(self.other_key, self.claims(sub, email), kid=self.key.kid if same_kid else self.other_key.kid)

    def jwks_fetches(self) -> list[dict]:
        return self.sink.requests("GET", fakeidp.JWKS_PATH)


@pytest.fixture(scope="module")
def idp(ctx):
    if not ctx.compose_project:
        pytest.skip("the fake Supabase sidecar needs --compose-project")
    want = f"http://{fakeidp.FAKE_SUPABASE_NAME}:{sidecar.PORT}"
    got = ctx.profile_flags.get("SUPABASE_URL")
    assert got == want, f"the profile's SUPABASE_URL ({got!r}) is not the sidecar's address ({want!r})"
    key = fakeidp.RsaKey.generate(kid=f"acc-{ctx.run_id}")  # in memory only; never written anywhere
    other = fakeidp.RsaKey.generate(kid=f"acc-{ctx.run_id}-other")
    sink = sidecar.Sink(ctx.compose_project, {fakeidp.JWKS_PATH: key.jwks()}, name=fakeidp.FAKE_SUPABASE_NAME)
    sink.start()
    try:
        yield FakeSupabase(url=want, key=key, other_key=other, sink=sink, hs_secret=secrets.token_bytes(32))
    finally:
        sink.stop()


# ── a browser, and the login steps ───────────────────────────────


class Browser:
    """An HTTP client with a cookie jar and no auto-redirects: one browser profile."""

    def __init__(self, base_url: str) -> None:
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0, headers={"Accept": "text/html"})

    def close(self) -> None:
        self.http.close()

    def get(self, path: str, **kw) -> httpx.Response:
        return self.http.get(path, **kw)

    def post(self, path: str, **kw) -> httpx.Response:
        return self.http.post(path, **kw)

    def callback(self, login_state: str, access_token: str) -> httpx.Response:
        """What the sign-in finisher page does: a same-origin form POST of the session token and the state."""
        return self.post("/oauth/supabase/callback", data={"access_token": access_token, "state": login_state})


def public_url(ctx) -> str:
    url = ctx.profile_flags.get("NEURALSCAPE_PUBLIC_URL", "").rstrip("/")
    assert url == ctx.base_url, (
        f"the profile's NEURALSCAPE_PUBLIC_URL ({url!r}) is not the URL this run targets ({ctx.base_url}): "
        "run the stack on the profile's port, or edit that line for this run"
    )
    return url


def pkce() -> tuple[str, str]:
    verifier = b64url(secrets.token_bytes(32))
    return verifier, b64url(hashlib.sha256(verifier.encode()).digest())


@dataclass
class Started:
    client_id: str
    verifier: str
    app_state: str
    login_state: str
    browser: Browser
    consent: httpx.Response


def start_login(ctx, anon: NSClient, browser: Browser) -> Started:
    r = anon.post("/oauth/register", json={"redirect_uris": [REDIRECT]})
    assert r.status_code == 201, f"POST /oauth/register -> {r.status_code}: {r.text[:200]}"
    client_id = r.json()["client_id"]
    verifier, challenge = pkce()
    app_state = secrets.token_urlsafe(8)
    page = browser.get("/oauth/authorize", params={
        "response_type": "code", "client_id": client_id, "redirect_uri": REDIRECT, "state": app_state,
        "code_challenge": challenge, "code_challenge_method": "S256",
    })
    assert page.status_code == 200, f"GET /oauth/authorize -> {page.status_code}: {page.text[:200]}"
    return Started(client_id, verifier, app_state, fakeidp.consent_login_state(page.text, public_url(ctx)), browser, page)


def code_from(resp: httpx.Response, st: Started) -> str:
    assert resp.status_code == 303, f"expected a 303 back to the client, got {resp.status_code}: {resp.text[:200]}"
    loc = urlparse(resp.headers["location"])
    assert f"{loc.scheme}://{loc.netloc}{loc.path}" == REDIRECT, f"redirected to {resp.headers['location']}"
    q = parse_qs(loc.query)
    assert q.get("state") == [st.app_state], f"the client's state was not echoed: {q.get('state')}"
    assert q.get("code"), "the redirect carries no code"
    return q["code"][0]


def exchange(anon: NSClient, st: Started, code: str, verifier: str | None = None) -> httpx.Response:
    return anon.post("/oauth/token", data={
        "grant_type": "authorization_code", "code": code, "code_verifier": verifier or st.verifier,
        "redirect_uri": REDIRECT, "client_id": st.client_id,
    })


def redeem(anon: NSClient, st: Started, code: str) -> dict:
    r = exchange(anon, st, code)
    assert r.status_code == 200, f"POST /oauth/token -> {r.status_code}: {r.text[:200]}"
    tokens = r.json()
    assert tokens.get("token_type") == "Bearer" and tokens.get("access_token"), f"token response keys: {sorted(tokens)}"
    return tokens


def token_user(access_token: str) -> str:
    return json.loads(fakeidp.b64url_decode(access_token.split(".")[0]))["user_id"]


@dataclass
class Done:
    """One complete login: what the callback answered, whether the link page appeared, and the tokens."""

    st: Started
    first: httpx.Response
    saw_link_page: bool
    suggested: str
    redirect: httpx.Response
    tokens: dict

    @property
    def user_id(self) -> str:
        return token_user(self.tokens["access_token"])


def login(ctx, anon, idp: FakeSupabase, email: str, sub: str, *, link_action: str = "new", link_token: str = "") -> Done:
    browser = Browser(ctx.base_url)
    try:
        st = start_login(ctx, anon, browser)
        first = browser.callback(st.login_state, idp.token(sub, email))
        resp, saw_link, suggested = first, False, ""
        if first.status_code == 200 and 'action="/oauth/link"' in first.text:
            saw_link = True
            fields = fakeidp.link_page_fields(first.text)
            suggested = fields["suggested"]
            resp = browser.post("/oauth/link", data={"ticket": fields["ticket"], "action": link_action, "token": link_token})
        code = code_from(resp, st)
        return Done(st, first, saw_link, suggested, resp, redeem(anon, st, code))
    finally:
        browser.close()


def attempt(ctx, anon, idp: FakeSupabase, access_token: str, *, mutate_state=None) -> httpx.Response:
    """A fresh browser, a fresh login, then the callback with ``access_token`` (one attempt, one login)."""
    browser = Browser(ctx.base_url)
    try:
        st = start_login(ctx, anon, browser)
        state = mutate_state(st.login_state) if mutate_state else st.login_state
        return browser.callback(state, access_token)
    finally:
        browser.close()


# ── the real checks (each negative control reuses one) ───────────


def check_refused(resp: httpx.Response, *, status: int | tuple = 403, needle: str | tuple = "") -> None:
    """The callback refused: the right status, a sign-in page, and NO redirect and NO code.

    ``needle`` is a phrase the page must contain (any one of a tuple)."""
    statuses = (status,) if isinstance(status, int) else tuple(status)
    assert resp.status_code in statuses, f"callback -> {resp.status_code} (want {statuses}): {resp.text[:160]}"
    assert "location" not in resp.headers, f"a refused login redirected to {resp.headers.get('location')}"
    assert not re.search(r"[?&]code=", resp.text), "a refused login's page carries a code"
    needles = (needle,) if isinstance(needle, str) else tuple(needle)
    text = html.unescape(resp.text).lower()
    if any(needles):
        assert any(n.lower() in text for n in needles if n), f"the refusal does not say any of {needles}: {resp.text[:200]}"


def check_bearer_user(base_url: str, access_token: str, user_id: str, marker: str) -> None:
    """The access token authenticates REST and the write it makes belongs to ``user_id``; its payload says so too."""
    assert token_user(access_token) == user_id, f"the access token names {token_user(access_token)!r}, not {user_id!r}"
    c = NSClient(base_url, access_token)
    try:
        task = c.store_raw(f"{marker} The harbour light is oiled by the pier crew.", "preference", visibility="private")
        assert c.poll_task(task, timeout=120)["status"] == "completed", "a write with the access token did not complete"
        rows = [r for r in list_rows(c) if marker in r.get("memory", "")]
        assert rows and rows[0].get("owner_user_id") == user_id, f"the write is not user {user_id}'s: {rows[:1]}"
    finally:
        c.close()


def check_mcp_accepts(base_url: str, access_token: str) -> None:
    m = MCPClient(base_url, access_token)
    try:
        m.initialize()
        assert len(m.list_tools()) > 0, "MCP tools/list returned no tools for the access token"
    finally:
        m.close()


BINDING_REFUSED = ("not started in this browser",)  # the refusal wording #536 documents


def precondition(check, *args, **kw) -> None:
    """A negative control proves nothing unless the SAME check first passes for the right input."""
    try:
        check(*args, **kw)
    except AssertionError as exc:
        raise AssertionError(f"control precondition: the real check does not pass for the right input: {exc}") from None


def require_binding_deployed(consent: httpx.Response) -> None:
    """FAIL (not skip, not pass) when /oauth/authorize sets no login-binding cookie: #536 is not deployed."""
    assert any("ns_login_bind" in h for h in consent.headers.get_list("set-cookie")), (
        "login binding not deployed (#536): /oauth/authorize set no ns_login_bind cookie, so a login state can be "
        "completed by any browser and more than once"
    )


def check_fetched(sink: sidecar.Sink, path: str) -> None:
    got = sink.requests("GET", path)
    assert got, f"the stack never asked the fake Supabase for {path}; it saw {sorted({r['path'] for r in sink.requests()})}"


def check_redeemed(resp: httpx.Response) -> None:
    assert resp.status_code == 200 and resp.json().get("access_token"), f"/oauth/token -> {resp.status_code}: {resp.text[:160]}"


# ── logins, recorded once per module ─────────────────────────────


@dataclass
class Recorded:
    email: str = ""
    sub: str = ""
    first: Done | None = None
    again: Done | None = None  # same sub, same email
    other_email: Done | None = None  # same sub, a different allowlisted email
    other_email_value: str = ""
    error: str = ""


def _ok(rec: Recorded) -> Recorded:
    assert not rec.error, rec.error
    return rec


@pytest.fixture(scope="module")
def slug_login(ctx, anon, idp) -> Recorded:
    """A brand-new allowlisted person: first login (link page -> new account), then the same sub twice more."""
    rec = Recorded(email=f"harbor.{ctx.run_id}@{DOMAIN}", sub=f"sb-{uuid.uuid4()}")
    rec.other_email_value = f"harbor.{ctx.run_id}.alt@{DOMAIN}"
    try:
        rec.first = login(ctx, anon, idp, rec.email, rec.sub)
        rec.again = login(ctx, anon, idp, rec.email, rec.sub)
        rec.other_email = login(ctx, anon, idp, rec.other_email_value, rec.sub)
    except AssertionError as exc:
        rec.error = str(exc)
    return rec


@pytest.fixture(scope="module")
def mapped_login(ctx, anon, idp) -> Recorded:
    """The person AUTH_IDENTITY_MAP names: no link page, the admin-assigned id; then the same sub with another email."""
    rec = Recorded(email=MAPPED_EMAIL, sub=f"sb-{uuid.uuid4()}")
    rec.other_email_value = f"acc-mapped.alt.{ctx.run_id}@{DOMAIN}"
    try:
        rec.first = login(ctx, anon, idp, rec.email, rec.sub)
        rec.other_email = login(ctx, anon, idp, rec.other_email_value, rec.sub)
    except AssertionError as exc:
        rec.error = str(exc)
    return rec


@pytest.fixture(scope="module")
def claim_login(ctx, anon, idp) -> Recorded:
    """A new identity that claims an EXISTING account (user A) by pasting A's token on the link page."""
    rec = Recorded(email=f"claimer.{ctx.run_id}@{DOMAIN}", sub=f"sb-{uuid.uuid4()}")
    try:
        rec.first = login(ctx, anon, idp, rec.email, rec.sub, link_action="link", link_token=ctx.token_a)
    except AssertionError as exc:
        rec.error = str(exc)
    return rec


# ═══ a new allowlisted person: slug identity ══════════════════════════════


@proves("auth.login-supabase", "auth.email-allowlist", "auth.email-identity-slug", "auth.oauth-server")
def test_an_allowlisted_verified_email_signs_in_and_the_bearer_is_the_slug_user(ctx, slug_login: Recorded):
    rec = _ok(slug_login)
    slug = fakeidp.expected_slug(rec.email)
    d = rec.first
    assert d.saw_link_page, "a brand-new identity did not get the first-login link page"
    assert d.suggested == slug, f"the suggested account is {d.suggested!r}, the email slug rule gives {slug!r}"
    check_bearer_user(ctx.base_url, d.tokens["access_token"], slug, f"zq-{ctx.run_id}-sbslug")
    check_mcp_accepts(ctx.base_url, d.tokens["access_token"])


@proves("auth.login-supabase")
def test_the_stack_verified_the_session_token_against_the_fake_supabases_jwks(idp, slug_login: Recorded):
    _ok(slug_login)
    check_fetched(idp.sink, fakeidp.JWKS_PATH)


@proves("auth.login-supabase", "auth.email-identity-slug", "auth.oauth-server")
@negative_control
def test_the_login_checks_fail_for_the_wrong_user_and_a_path_nobody_asked_for(ctx, idp, slug_login: Recorded):
    d = _ok(slug_login).first
    precondition(check_bearer_user, ctx.base_url, d.tokens["access_token"], fakeidp.expected_slug(slug_login.email), f"zq-{ctx.run_id}-sbslug1")
    precondition(check_fetched, idp.sink, fakeidp.JWKS_PATH)
    expect_check_fails(check_bearer_user, ctx.base_url, d.tokens["access_token"], "someone-else", f"zq-{ctx.run_id}-sbslug0")
    expect_check_fails(check_fetched, idp.sink, "/auth/v1/never-asked")


@proves("auth.identity-store", "auth.login-supabase")
def test_a_second_login_skips_the_link_page_and_the_sub_wins_over_a_changed_email(slug_login: Recorded):
    rec = _ok(slug_login)
    slug = fakeidp.expected_slug(rec.email)
    assert not rec.again.saw_link_page, "the same person was shown the first-login link page again (the durable link was not kept)"
    assert rec.again.user_id == slug, f"second login is {rec.again.user_id!r}, first was {slug!r}"
    assert not rec.other_email.saw_link_page, "a known sub with a new email was treated as a new person"
    assert rec.other_email.user_id == slug, f"same sub, other email -> {rec.other_email.user_id!r} (want {slug!r}: the sub is the immutable key)"


@proves("auth.identity-store")
@negative_control
def test_identity_check_fails_for_a_first_login_and_a_different_person(slug_login: Recorded, mapped_login: Recorded):
    slug, mapped = _ok(slug_login), _ok(mapped_login)

    def check_linked(done: Done, user: str) -> None:
        assert not done.saw_link_page, "link page shown"
        assert done.user_id == user, f"user {done.user_id!r} != {user!r}"

    check_linked(slug.again, fakeidp.expected_slug(slug.email))  # the real check passes for the linked person
    expect_check_fails(check_linked, slug.first, fakeidp.expected_slug(slug.email))  # a first login DID see the page
    expect_check_fails(check_linked, mapped.first, fakeidp.expected_slug(slug.email))  # another person is another id


# ═══ an admin-assigned id (AUTH_IDENTITY_MAP) ═════════════════════════════


@proves("auth.email-identity-slug", "auth.login-supabase", "auth.identity-store")
def test_a_mapped_email_signs_straight_in_as_the_admin_assigned_user(ctx, mapped_login: Recorded):
    rec = _ok(mapped_login)
    d = rec.first
    assert not d.saw_link_page, "an email named in AUTH_IDENTITY_MAP was shown the link page"
    assert d.user_id == MAPPED_USER, f"mapped email -> {d.user_id!r}, the map says {MAPPED_USER!r}"
    assert d.user_id != fakeidp.expected_slug(rec.email), "the map made no difference to the slug"
    check_bearer_user(ctx.base_url, d.tokens["access_token"], MAPPED_USER, f"zq-{ctx.run_id}-sbmap")
    # the seed was promoted into the durable store: the same sub with another email is still that user
    assert rec.other_email.user_id == MAPPED_USER and not rec.other_email.saw_link_page, (
        f"after the seed was promoted, the same sub with another email -> {rec.other_email.user_id!r}"
    )


@proves("auth.email-identity-slug")
@negative_control
def test_user_check_fails_for_the_slug_of_a_mapped_email(ctx, mapped_login: Recorded):
    d = _ok(mapped_login).first
    precondition(check_bearer_user, ctx.base_url, d.tokens["access_token"], MAPPED_USER, f"zq-{ctx.run_id}-sbmap1")
    expect_check_fails(check_bearer_user, ctx.base_url, d.tokens["access_token"], fakeidp.expected_slug(MAPPED_EMAIL), f"zq-{ctx.run_id}-sbmap0")


# ═══ claiming an existing account on the link page ═════════════════════════


@proves("auth.identity-store", "auth.login-supabase", "auth.oauth-server")
def test_a_new_identity_can_claim_an_existing_account_with_that_accounts_token(ctx, claim_login: Recorded):
    d = _ok(claim_login).first
    assert d.saw_link_page, "a new identity did not get the link page"
    assert d.user_id == ctx.user_a, f"claiming user A's account gave {d.user_id!r}, not {ctx.user_a!r}"
    check_bearer_user(ctx.base_url, d.tokens["access_token"], ctx.user_a, f"zq-{ctx.run_id}-sbclaim")


@proves("auth.identity-store", "auth.login-supabase")
def test_the_link_page_refuses_a_bad_token_and_a_forged_ticket(ctx, anon, idp):
    browser = Browser(ctx.base_url)
    try:
        st = start_login(ctx, anon, browser)
        first = browser.callback(st.login_state, idp.token(f"sb-{uuid.uuid4()}", f"badlink.{ctx.run_id}@{DOMAIN}"))
        fields = fakeidp.link_page_fields(first.text)
        for what, token in (("a token signed with the wrong secret", ctx.bad_token), ("garbage", "not-a-token")):
            r = browser.post("/oauth/link", data={"ticket": fields["ticket"], "action": "link", "token": token})
            check_refused(r, status=400, needle="invalid or expired")
            assert fakeidp.link_page_fields(r.text)["ticket"], f"after {what} the page offers no way to retry"
        forged = fields["ticket"][:-4] + ("AAAA" if not fields["ticket"].endswith("AAAA") else "BBBB")
        check_refused(browser.post("/oauth/link", data={"ticket": forged, "action": "new"}), status=400, needle="expired")
        check_refused(browser.post("/oauth/link", data={"ticket": "garbage", "action": "new"}), status=400, needle="expired")
    finally:
        browser.close()


@proves("auth.identity-store")
@negative_control
def test_link_refusal_check_fails_for_a_good_claim(ctx, anon, idp, claim_login: Recorded):
    precondition(check_refused, attempt(ctx, anon, idp, idp.token(f"sb-{uuid.uuid4()}", MAPPED_EMAIL), mutate_state=lambda s: "garbage"), status=400)
    expect_check_fails(check_refused, _ok(claim_login).first.redirect, status=400)


# ═══ who may sign in (allowlist) ══════════════════════════════════════════


@proves("auth.email-allowlist", "auth.login-supabase")
def test_a_verified_email_outside_the_allowlist_is_refused(ctx, anon, idp):
    for email in (f"outsider.{ctx.run_id}@elsewhere.example", f"outsider.{ctx.run_id}@{DOMAIN}.evil.example", f"sub.{ctx.run_id}@mail.{DOMAIN}"):
        check_refused(attempt(ctx, anon, idp, idp.token(f"sb-{uuid.uuid4()}", email)), status=403, needle="not authorized")


@proves("auth.email-allowlist", "auth.login-supabase")
def test_the_exact_address_allowlist_admits_that_address_and_not_its_domain(ctx, anon, idp):
    ok = attempt(ctx, anon, idp, idp.token(f"sb-{uuid.uuid4()}", GUEST_ADDRESS))
    assert ok.status_code == 200 and 'action="/oauth/link"' in ok.text, f"the allowlisted guest address was not admitted: {ok.status_code} {ok.text[:160]}"
    check_refused(attempt(ctx, anon, idp, idp.token(f"sb-{uuid.uuid4()}", f"acc-other@acceptance-guest.example")), status=403, needle="not authorized")


@proves("auth.email-allowlist", "auth.login-supabase")
def test_an_unverified_email_is_refused_even_on_an_allowed_domain(ctx, anon, idp):
    email = f"unverified.{ctx.run_id}@{DOMAIN}"
    sub = f"sb-{uuid.uuid4()}"
    check_refused(attempt(ctx, anon, idp, idp.token(sub, email, verified=False)), status=403, needle="unverified")
    # user_metadata.email_verified is user-writable: claiming verification there must not count
    check_refused(attempt(ctx, anon, idp, idp.token(sub, email, verified=False, user_metadata_verified=True)), status=403, needle="unverified")
    check_refused(attempt(ctx, anon, idp, idp.token(sub, "", verified=True)), status=403, needle="unverified")


@proves("auth.email-allowlist", "auth.login-supabase")
@negative_control
def test_refusal_check_fails_for_a_login_that_succeeded(ctx, anon, idp, mapped_login: Recorded):
    precondition(check_refused, attempt(ctx, anon, idp, idp.token(f"sb-{uuid.uuid4()}", f"o.{ctx.run_id}@elsewhere.example")), status=403, needle="not authorized")
    precondition(check_refused, attempt(ctx, anon, idp, idp.token(f"sb-{uuid.uuid4()}", f"o.{ctx.run_id}@elsewhere.example")), status=403)
    ok = _ok(mapped_login).first
    expect_check_fails(check_refused, ok.first, status=403, needle="not authorized")
    expect_check_fails(check_refused, ok.redirect, status=403)  # a redirect is never a refusal


# ═══ the token itself (signature, audience, expiry) ═══════════════════════


@proves("auth.login-supabase")
def test_a_token_not_signed_by_the_published_key_is_refused(ctx, anon, idp, slug_login: Recorded):
    _ok(slug_login)  # the real key works first (the control for this family)
    email = f"forger.{ctx.run_id}@{DOMAIN}"
    sub = f"sb-{uuid.uuid4()}"
    check_refused(attempt(ctx, anon, idp, idp.forged(sub, email, same_kid=True)), status=401, needle="could not verify")
    check_refused(attempt(ctx, anon, idp, idp.forged(sub, email, same_kid=False)), status=401, needle="could not verify")
    # algorithm confusion: a symmetric token must not be accepted against an asymmetric JWKS
    hs = fakeidp.sign_hs256(idp.hs_secret, idp.claims(sub, email), kid=idp.key.kid)
    check_refused(attempt(ctx, anon, idp, hs), status=401, needle="could not verify")
    # a signature that does not match the payload
    head, payload, sig = idp.token(sub, email).split(".")
    tampered = fakeidp.b64url(json.dumps({**idp.claims(sub, f"admin.{ctx.run_id}@{DOMAIN}")}, separators=(",", ":")).encode())
    check_refused(attempt(ctx, anon, idp, f"{head}.{tampered}.{sig}"), status=401, needle="could not verify")


@proves("auth.login-supabase")
def test_a_token_for_another_audience_or_expired_or_without_expiry_is_refused(ctx, anon, idp):
    email = f"claims.{ctx.run_id}@{DOMAIN}"
    sub = f"sb-{uuid.uuid4()}"
    check_refused(attempt(ctx, anon, idp, idp.token(sub, email, aud="anon")), status=401, needle="could not verify")
    check_refused(attempt(ctx, anon, idp, idp.token(sub, email, aud="service_role")), status=401, needle="could not verify")
    check_refused(attempt(ctx, anon, idp, idp.token(sub, email, ttl=-120)), status=401, needle="could not verify")
    check_refused(attempt(ctx, anon, idp, idp.token(sub, email, with_exp=False)), status=401, needle="could not verify")


@proves("auth.login-supabase")
def test_the_callback_refuses_a_bad_state_and_a_missing_token(ctx, anon, idp):
    token = idp.token(f"sb-{uuid.uuid4()}", MAPPED_EMAIL)
    check_refused(attempt(ctx, anon, idp, token, mutate_state=lambda s: s[:-3] + ("AAA" if not s.endswith("AAA") else "BBB")), status=400, needle=STATE_REFUSED)
    check_refused(attempt(ctx, anon, idp, token, mutate_state=lambda s: "garbage"), status=400, needle=STATE_REFUSED)
    check_refused(attempt(ctx, anon, idp, ""), status=400, needle=("no supabase session",))  # no session at all


@proves("auth.login-supabase")
@negative_control
def test_token_check_fails_for_a_token_the_published_key_signed(ctx, anon, idp):
    resp = attempt(ctx, anon, idp, idp.token(f"sb-{uuid.uuid4()}", MAPPED_EMAIL))
    assert resp.status_code == 303, f"the right key did not log in: {resp.status_code} {resp.text[:160]}"
    precondition(check_refused, attempt(ctx, anon, idp, idp.forged(f"sb-{uuid.uuid4()}", MAPPED_EMAIL, same_kid=True)), status=401, needle="could not verify")
    expect_check_fails(check_refused, resp, status=401, needle="could not verify")


# ═══ the code is PKCE-bound ═══════════════════════════════════════════════


@proves("auth.oauth-server", "auth.login-supabase")
def test_the_code_a_supabase_login_issues_is_pkce_bound_and_redeems_once(ctx, anon, idp):
    browser = Browser(ctx.base_url)
    try:
        st = start_login(ctx, anon, browser)
        code = code_from(browser.callback(st.login_state, idp.token(f"sb-{uuid.uuid4()}", MAPPED_EMAIL)), st)
    finally:
        browser.close()
    bad = exchange(anon, st, code, verifier="x" * 43)
    assert bad.status_code == 400 and bad.json().get("error") == "invalid_grant", f"wrong verifier -> {bad.status_code}: {bad.text[:160]}"
    check_redeemed(exchange(anon, st, code))
    replay = exchange(anon, st, code)
    assert replay.status_code == 400 and replay.json().get("error") == "invalid_grant", f"replayed code -> {replay.status_code}: {replay.text[:160]}"


@proves("auth.oauth-server")
@negative_control
def test_redeem_check_fails_for_a_refused_exchange(ctx, anon, idp):
    browser = Browser(ctx.base_url)
    try:
        st = start_login(ctx, anon, browser)
        code = code_from(browser.callback(st.login_state, idp.token(f"sb-{uuid.uuid4()}", MAPPED_EMAIL)), st)
    finally:
        browser.close()
    precondition(check_redeemed, exchange(anon, st, code))
    expect_check_fails(check_redeemed, anon.post("/oauth/token", data={"grant_type": "authorization_code", "code": "garbage", "code_verifier": "x" * 43,
                                                                       "redirect_uri": REDIRECT, "client_id": "nobody"}))


# ═══ the login is bound to the browser that started it (PR #536) ═══════════


def _binding_cookies(browser: Browser) -> list:
    return [c for c in browser.http.cookies.jar if c.name.endswith("ns_login_bind")]


@proves("auth.login-supabase")
@expected_red("#536")
def test_the_consent_page_sets_the_login_binding_cookie(ctx, anon):
    """EXPECTED RED UNTIL PR #536 merges: ``GET /oauth/authorize`` sets an HttpOnly, SameSite=Lax binding cookie."""
    browser = Browser(ctx.base_url)
    try:
        st = start_login(ctx, anon, browser)
        raw = [h for h in st.consent.headers.get_list("set-cookie") if "ns_login_bind" in h]
        assert raw, f"no login-binding cookie was set on /oauth/authorize ({len(st.consent.headers.get_list('set-cookie'))} Set-Cookie headers)"
        flags = raw[0].lower()
        assert "httponly" in flags and "samesite=lax" in flags and "max-age=" in flags, f"binding cookie attributes: {flags.split(';', 1)[-1]}"
        assert _binding_cookies(browser), "the browser kept no binding cookie"
    finally:
        browser.close()


@proves("auth.login-supabase")
@expected_red("#536")
def test_a_callback_from_a_browser_that_did_not_start_the_login_is_refused(ctx, anon, idp):
    """EXPECTED RED UNTIL PR #536 merges. Login CSRF: an attacker starts a login and gets a victim's browser
    to finish it, so the code would be issued to the attacker's client for the VICTIM's account."""
    attacker, victim = Browser(ctx.base_url), Browser(ctx.base_url)
    try:
        st = start_login(ctx, anon, attacker)
        require_binding_deployed(st.consent)
        resp = victim.callback(st.login_state, idp.token(f"sb-{uuid.uuid4()}", MAPPED_EMAIL))  # the victim has no cookie for this state
        check_refused(resp, status=403, needle=BINDING_REFUSED)
        # nor does a cookie from the victim's own, different login unlock the attacker's state
        start_login(ctx, anon, victim)
        check_refused(victim.callback(st.login_state, idp.token(f"sb-{uuid.uuid4()}", MAPPED_EMAIL)), status=403, needle=BINDING_REFUSED)
    finally:
        attacker.close()
        victim.close()


@proves("auth.login-supabase")
@expected_red("#536")
def test_a_login_state_cannot_be_completed_twice(ctx, anon, idp):
    """EXPECTED RED UNTIL PR #536 merges: the state's nonce is claimed once; a replay of the same callback is refused."""
    browser = Browser(ctx.base_url)
    try:
        st = start_login(ctx, anon, browser)
        require_binding_deployed(st.consent)
        token = idp.token(f"sb-{uuid.uuid4()}", MAPPED_EMAIL)
        code_from(browser.callback(st.login_state, token), st)  # the first completion works
        check_refused(browser.callback(st.login_state, token), status=403, needle=BINDING_REFUSED)
    finally:
        browser.close()


@proves("auth.login-supabase")
@negative_control
def test_binding_check_fails_for_the_browser_that_did_start_the_login(ctx, anon, idp):
    browser = Browser(ctx.base_url)
    try:
        st = start_login(ctx, anon, browser)
        ok = browser.callback(st.login_state, idp.token(f"sb-{uuid.uuid4()}", MAPPED_EMAIL))
        code_from(ok, st)  # the browser that started the login is let through, with or without the binding
        precondition(check_refused, attempt(ctx, anon, idp, idp.token(f"sb-{uuid.uuid4()}", f"o.{ctx.run_id}@elsewhere.example")), status=403, needle="not authorized")
        expect_check_fails(check_refused, ok, status=403, needle="not authorized")
    finally:
        browser.close()
