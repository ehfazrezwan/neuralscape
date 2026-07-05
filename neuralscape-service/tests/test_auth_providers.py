"""Unit tests for the federated-login providers (google / supabase) and the
shared allowlist + identity-derivation logic.

No running services required — JWT verification is exercised with real HS256
tokens (Supabase) and a stubbed verify/exchange (Google).
"""

import re
import time

import jwt
import pytest

import identity_store
import login_providers as lp
from allowlist import email_domain, is_email_allowed, normalize_email
from config import Settings, settings
from identity import derive_user_id, slugify_email
from login_providers import (
    AuthorizeContext,
    GoogleProvider,
    LoginError,
    LoginNeedsLink,
    LoginResult,
    SupabaseProvider,
    get_login_provider,
    resolve_login_identity,
    sign_link_ticket,
    sign_login_state,
    verify_link_ticket,
    verify_login_state,
)
from schemas import _ID_PATTERN

_SECRET = "test-signing-secret"
_ID_RE = re.compile(_ID_PATTERN)


# ── fixtures / stubs ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _hermetic_auth_env(monkeypatch):
    """Strip auth/provider env vars before each test in this module.

    ``graphiti_core`` calls ``load_dotenv()`` at import time, leaking the repo
    ``.env`` into ``os.environ``. When the full suite runs (so something imports
    graphiti_core), a real ``GOOGLE_OAUTH_CLIENT_ID`` / ``AUTH_ALLOWED_DOMAINS``
    from that .env would defeat the "missing creds/allowlist → error" assertions
    in ``TestValidateRequired`` (which build ``Settings(**kw)`` and expect the
    unspecified fields to be empty). Clearing them keeps those tests hermetic.
    """
    for var in (
        "AUTH_PROVIDER", "AUTH_ALLOW_TOKEN_PASTE",
        "NEURALSCAPE_PUBLIC_URL", "NEURALSCAPE_USER_TOKEN_SECRET",
        "GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET",
        "AUTH_ALLOWED_DOMAINS", "AUTH_EMAIL_ALLOWLIST", "AUTH_IDENTITY_MAP",
        "SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_JWT_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def auth_settings():
    """Snapshot + restore every auth-related settings field per test."""
    fields = [
        "auth_provider", "neuralscape_public_url", "neuralscape_user_token_secret",
        "google_oauth_client_id", "google_oauth_client_secret",
        "auth_allowed_domains", "auth_email_allowlist", "auth_identity_map",
        "supabase_url", "supabase_anon_key", "supabase_jwt_secret",
    ]
    orig = {f: getattr(settings, f) for f in fields}
    settings.neuralscape_user_token_secret = _SECRET
    settings.neuralscape_public_url = "https://ns.example.com"
    yield settings
    for f, v in orig.items():
        setattr(settings, f, v)


class _Req:
    """Minimal stand-in for fastapi.Request used by provider.complete()."""

    def __init__(self, query=None, form=None):
        self.query_params = query or {}
        self._form = form or {}

    async def form(self):
        return self._form


def _make_ctx(state="mcpstate"):
    return AuthorizeContext(
        client_id="cid", redirect_uri="https://client/cb", state=state,
        code_challenge="chal", code_challenge_method="S256", resource="https://ns/mcp",
    )


# ── allowlist ─────────────────────────────────────────────────────────────


class TestAllowlist:
    def test_domain_match(self):
        assert is_email_allowed(
            "a@example.com", email_verified=True,
            allowed_domains={"example.com"}, email_allowlist=set())

    def test_exact_match(self):
        assert is_email_allowed(
            "guest@gmail.com", email_verified=True,
            allowed_domains=set(), email_allowlist={"guest@gmail.com"})

    def test_case_insensitive(self):
        assert is_email_allowed(
            "A.B@Example.COM", email_verified=True,
            allowed_domains={"example.com"}, email_allowlist=set())

    def test_unverified_rejected(self):
        assert not is_email_allowed(
            "a@example.com", email_verified=False,
            allowed_domains={"example.com"}, email_allowlist=set())

    def test_wrong_domain_rejected(self):
        assert not is_email_allowed(
            "a@evil.com", email_verified=True,
            allowed_domains={"example.com"}, email_allowlist={"x@y.com"})

    def test_empty_allowlist_fails_closed(self):
        assert not is_email_allowed(
            "a@x.com", email_verified=True,
            allowed_domains=set(), email_allowlist=set())

    def test_malformed_email_rejected(self):
        assert not is_email_allowed(
            "not-an-email", email_verified=True,
            allowed_domains={"x.com"}, email_allowlist=set())

    def test_empty_local_part_rejected(self):
        # "@example.com" has an empty local-part and must not pass the domain gate.
        assert not is_email_allowed(
            "@example.com", email_verified=True,
            allowed_domains={"example.com"}, email_allowlist=set())
        assert not is_email_allowed(
            "a@", email_verified=True,
            allowed_domains={"example.com"}, email_allowlist={"a@"})

    def test_helpers(self):
        assert normalize_email("  A@B.com ") == "a@b.com"
        assert email_domain("a@b.com") == "b.com"
        assert email_domain("garbage") == ""
        assert email_domain("@b.com") == ""   # empty local-part
        assert email_domain("a@") == ""        # empty domain


# ── identity ──────────────────────────────────────────────────────────────


class TestIdentity:
    def test_slug_basic(self):
        assert slugify_email("Alice.Smith@Example.com") == "alice.smith-example.com"

    def test_slug_is_pattern_valid(self):
        for e in ["weird+tag@sub.example.co.uk", "a..b@x.com", "x!#$@y.com", "中文@例子.com"]:
            slug = slugify_email(e)
            assert slug, e
            assert _ID_RE.match(slug), (e, slug)
            assert len(slug) <= 100

    def test_slug_all_symbol_localpart_hashes(self):
        slug = slugify_email("!!!@@@")
        assert slug.startswith("user-")
        assert _ID_RE.match(slug)

    def test_map_override_wins(self):
        m = {"alice@example.com": "alice"}
        assert derive_user_id("alice@example.com", m) == "alice"
        assert derive_user_id("Alice@Example.com", m) == "alice"  # case-insensitive

    def test_unmapped_falls_back_to_slug(self):
        assert derive_user_id("new@x.com", {}) == "new-x.com"


# ── login state ───────────────────────────────────────────────────────────


class TestLoginState:
    def test_roundtrip(self):
        ctx = _make_ctx()
        got = verify_login_state(sign_login_state(ctx, _SECRET), _SECRET)
        assert got == ctx

    def test_wrong_secret_rejected(self):
        ctx = _make_ctx()
        assert verify_login_state(sign_login_state(ctx, _SECRET), "other") is None

    def test_garbage_rejected(self):
        assert verify_login_state("garbage", _SECRET) is None

    def test_wrong_typ_rejected(self):
        # A code/access token must not be redeemable as login state.
        from tokens import sign_payload
        tok = sign_payload({"typ": "access", "user_id": "x", "client_id": "c",
                            "redirect_uri": "r", "code_challenge": "h"}, _SECRET)
        assert verify_login_state(tok, _SECRET) is None

    def test_expired_rejected(self):
        from tokens import sign_payload
        claims = {"typ": "login_state", "client_id": "c", "redirect_uri": "r",
                  "code_challenge": "h", "code_challenge_method": "S256",
                  "state": "", "resource": "", "exp": int(time.time()) - 5}
        assert verify_login_state(sign_payload(claims, _SECRET), _SECRET) is None


# ── config parsing + provider factory ─────────────────────────────────────


class TestConfigAndFactory:
    def test_allowed_domains_strips_at(self, auth_settings):
        auth_settings.auth_allowed_domains = "@example.com, Acme.COM "
        assert auth_settings.allowed_domains_set() == {"example.com", "acme.com"}

    def test_email_allowlist_parse(self, auth_settings):
        auth_settings.auth_email_allowlist = "A@x.com, b@Y.com"
        assert auth_settings.email_allowlist_set() == {"a@x.com", "b@y.com"}

    def test_identity_map_parse(self, auth_settings):
        auth_settings.auth_identity_map = "Alice@Acme.com:alice, bad-no-colon, ops@example.com:ops-bot"
        assert auth_settings.identity_map_dict() == {
            "alice@acme.com": "alice", "ops@example.com": "ops-bot"}

    def test_factory_selects_provider(self, auth_settings):
        from login_providers import GoogleProvider, SupabaseProvider, TokenProvider
        auth_settings.auth_provider = "token"
        assert isinstance(get_login_provider(), TokenProvider)
        auth_settings.auth_provider = "google"
        assert isinstance(get_login_provider(), GoogleProvider)
        auth_settings.auth_provider = "supabase"
        assert isinstance(get_login_provider(), SupabaseProvider)


# ── validate_required ──────────────────────────────────────────────────────


class TestValidateRequired:
    def _base(self, **over):
        kw = dict(google_api_key="k", neo4j_password="p")
        kw.update(over)
        return Settings(**kw)

    def test_google_missing_creds_errors(self):
        s = self._base(auth_provider="google", neuralscape_public_url="https://x",
                       neuralscape_user_token_secret="s", auth_allowed_domains="x.com")
        with pytest.raises(ValueError, match="GOOGLE_OAUTH_CLIENT_ID"):
            s.validate_required()

    def test_google_missing_allowlist_errors(self):
        s = self._base(auth_provider="google", neuralscape_public_url="https://x",
                       neuralscape_user_token_secret="s",
                       google_oauth_client_id="c", google_oauth_client_secret="sec")
        with pytest.raises(ValueError, match="AUTH_ALLOWED_DOMAINS"):
            s.validate_required()

    def test_supabase_missing_creds_errors(self):
        s = self._base(auth_provider="supabase", neuralscape_public_url="https://x",
                       neuralscape_user_token_secret="s")
        with pytest.raises(ValueError, match="SUPABASE_URL"):
            s.validate_required()

    def test_google_happy_validates(self):
        s = self._base(auth_provider="google", neuralscape_public_url="https://x",
                       neuralscape_user_token_secret="s",
                       google_oauth_client_id="c", google_oauth_client_secret="sec",
                       auth_allowed_domains="example.com")
        s.validate_required()  # no raise

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValueError, match="AUTH_PROVIDER"):
            Settings(auth_provider="github")


# ── Supabase provider (real HS256 JWT) ────────────────────────────────────


def _supabase_jwt(email, *, verified=True, aud="authenticated", secret="sbsecret"):
    return jwt.encode(
        {"aud": aud, "email": email, "email_verified": verified,
         "exp": int(time.time()) + 300, "sub": "uuid-123"},
        secret, algorithm="HS256")


class TestSupabaseProvider:
    @pytest.mark.asyncio
    async def test_happy_path_maps_identity(self, auth_settings):
        auth_settings.auth_provider = "supabase"
        auth_settings.supabase_url = "https://abc.supabase.co"
        auth_settings.supabase_anon_key = "anon"
        auth_settings.supabase_jwt_secret = "sbsecret"
        auth_settings.auth_identity_map = "alice@example.com:alice"
        ctx = _make_ctx()
        state = sign_login_state(ctx, _SECRET)
        token = _supabase_jwt("alice@example.com")
        req = _Req(form={"access_token": token, "state": state})
        res = await SupabaseProvider().complete(req)
        assert isinstance(res, LoginResult)
        assert res.user_id == "alice"
        assert res.ctx == ctx

    @pytest.mark.asyncio
    async def test_trusts_external_gate_without_env_allowlist(self, auth_settings, fake_store):
        # No env allowlist → trust Supabase's hook; a fresh (unlinked) user is
        # offered the link page with a slug suggestion.
        auth_settings.supabase_jwt_secret = "sbsecret"
        token = _supabase_jwt("anyone@gmail.com")
        req = _Req(form={"access_token": token, "state": sign_login_state(_make_ctx(), _SECRET)})
        res = await SupabaseProvider().complete(req)
        assert isinstance(res, LoginNeedsLink)
        assert res.suggested_user_id == "anyone-gmail.com"

    @pytest.mark.asyncio
    async def test_linked_user_returns_result(self, auth_settings, fake_store):
        # A previously-linked identity skips the link page.
        auth_settings.supabase_jwt_secret = "sbsecret"
        await identity_store.link("anyone", sub="uuid-123", email="anyone@gmail.com")
        token = _supabase_jwt("anyone@gmail.com")
        req = _Req(form={"access_token": token, "state": sign_login_state(_make_ctx(), _SECRET)})
        res = await SupabaseProvider().complete(req)
        assert isinstance(res, LoginResult)
        assert res.user_id == "anyone"

    @pytest.mark.asyncio
    async def test_user_metadata_cannot_assert_verification(self, auth_settings):
        # user_metadata is user-writable → must NOT satisfy the verification gate
        # even if it claims email_verified=True while the top-level flag is False.
        auth_settings.supabase_jwt_secret = "sbsecret"
        token = jwt.encode(
            {"aud": "authenticated", "email": "a@example.com", "email_verified": False,
             "user_metadata": {"email_verified": True}, "sub": "uuid-x",
             "exp": int(time.time()) + 300},
            "sbsecret", algorithm="HS256")
        req = _Req(form={"access_token": token, "state": sign_login_state(_make_ctx(), _SECRET)})
        res = await SupabaseProvider().complete(req)
        assert isinstance(res, LoginError)
        assert res.status == 403

    @pytest.mark.asyncio
    async def test_env_allowlist_as_extra_gate_denies(self, auth_settings):
        auth_settings.supabase_jwt_secret = "sbsecret"
        auth_settings.auth_allowed_domains = "example.com"
        token = _supabase_jwt("outsider@gmail.com")
        req = _Req(form={"access_token": token, "state": sign_login_state(_make_ctx(), _SECRET)})
        res = await SupabaseProvider().complete(req)
        assert isinstance(res, LoginError)
        assert res.status == 403

    @pytest.mark.asyncio
    async def test_bad_signature_rejected(self, auth_settings):
        auth_settings.supabase_jwt_secret = "sbsecret"
        token = _supabase_jwt("a@example.com", secret="WRONG")
        req = _Req(form={"access_token": token, "state": sign_login_state(_make_ctx(), _SECRET)})
        res = await SupabaseProvider().complete(req)
        assert isinstance(res, LoginError)
        assert res.status == 401

    @pytest.mark.asyncio
    async def test_wrong_audience_rejected(self, auth_settings):
        auth_settings.supabase_jwt_secret = "sbsecret"
        token = _supabase_jwt("a@example.com", aud="anon")
        req = _Req(form={"access_token": token, "state": sign_login_state(_make_ctx(), _SECRET)})
        res = await SupabaseProvider().complete(req)
        assert isinstance(res, LoginError)
        assert res.status == 401  # JWT verification failure

    @pytest.mark.asyncio
    async def test_invalid_state_rejected(self, auth_settings):
        auth_settings.supabase_jwt_secret = "sbsecret"
        token = _supabase_jwt("a@example.com")
        req = _Req(form={"access_token": token, "state": "garbage"})
        res = await SupabaseProvider().complete(req)
        assert isinstance(res, LoginError)
        assert res.status == 400


# ── Google provider (stubbed exchange + id_token verify) ──────────────────


class _Resp:
    status_code = 200
    text = ""

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class _FakeClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, *a, **k):
        return _Resp({"id_token": "fake.id.token"})


class TestGoogleProvider:
    @pytest.mark.asyncio
    async def test_happy_path_allowlisted_new_user_needs_link(self, auth_settings, monkeypatch, fake_store):
        auth_settings.auth_provider = "google"
        auth_settings.google_oauth_client_id = "cid"
        auth_settings.google_oauth_client_secret = "sec"
        auth_settings.auth_allowed_domains = "example.com"
        monkeypatch.setattr(lp.httpx, "AsyncClient", _FakeClient)
        monkeypatch.setattr(lp, "_verify_oidc_jwt",
                            lambda *a, **k: {"email": "x@example.com", "email_verified": True,
                                             "sub": "goog-x"})
        state = sign_login_state(_make_ctx(), _SECRET)
        res = await GoogleProvider().complete(_Req(query={"code": "abc", "state": state}))
        assert isinstance(res, LoginNeedsLink)
        assert res.suggested_user_id == "x-example.com"
        assert res.sub == "goog-x"

    @pytest.mark.asyncio
    async def test_happy_path_linked_returns_result(self, auth_settings, monkeypatch, fake_store):
        auth_settings.auth_provider = "google"
        auth_settings.google_oauth_client_id = "cid"
        auth_settings.google_oauth_client_secret = "sec"
        auth_settings.auth_allowed_domains = "example.com"
        await identity_store.link("ehfazrezwan", sub="goog-x", email="x@example.com")
        monkeypatch.setattr(lp.httpx, "AsyncClient", _FakeClient)
        monkeypatch.setattr(lp, "_verify_oidc_jwt",
                            lambda *a, **k: {"email": "x@example.com", "email_verified": True,
                                             "sub": "goog-x"})
        state = sign_login_state(_make_ctx(), _SECRET)
        res = await GoogleProvider().complete(_Req(query={"code": "abc", "state": state}))
        assert isinstance(res, LoginResult)
        assert res.user_id == "ehfazrezwan"

    @pytest.mark.asyncio
    async def test_not_allowlisted_denied(self, auth_settings, monkeypatch):
        auth_settings.auth_provider = "google"
        auth_settings.google_oauth_client_id = "cid"
        auth_settings.google_oauth_client_secret = "sec"
        auth_settings.auth_allowed_domains = "example.com"
        monkeypatch.setattr(lp.httpx, "AsyncClient", _FakeClient)
        monkeypatch.setattr(lp, "_verify_oidc_jwt",
                            lambda *a, **k: {"email": "x@gmail.com", "email_verified": True})
        state = sign_login_state(_make_ctx(), _SECRET)
        res = await GoogleProvider().complete(_Req(query={"code": "abc", "state": state}))
        assert isinstance(res, LoginError)
        assert res.status == 403

    @pytest.mark.asyncio
    async def test_unverified_email_denied(self, auth_settings, monkeypatch):
        auth_settings.auth_provider = "google"
        auth_settings.google_oauth_client_id = "cid"
        auth_settings.google_oauth_client_secret = "sec"
        auth_settings.auth_allowed_domains = "example.com"
        monkeypatch.setattr(lp.httpx, "AsyncClient", _FakeClient)
        monkeypatch.setattr(lp, "_verify_oidc_jwt",
                            lambda *a, **k: {"email": "x@example.com", "email_verified": False})
        state = sign_login_state(_make_ctx(), _SECRET)
        res = await GoogleProvider().complete(_Req(query={"code": "abc", "state": state}))
        assert isinstance(res, LoginError)
        assert res.status == 403  # unverified email → not authorized

    @pytest.mark.asyncio
    async def test_provider_error_param(self, auth_settings):
        auth_settings.auth_provider = "google"
        res = await GoogleProvider().complete(_Req(query={"error": "access_denied"}))
        assert isinstance(res, LoginError)
        assert res.status == 400

    @pytest.mark.asyncio
    async def test_bad_state_rejected(self, auth_settings):
        auth_settings.auth_provider = "google"
        auth_settings.google_oauth_client_id = "cid"
        auth_settings.google_oauth_client_secret = "sec"
        res = await GoogleProvider().complete(_Req(query={"code": "abc", "state": "garbage"}))
        assert isinstance(res, LoginError)
        assert res.status == 400

    def test_consent_block_has_google_link(self, auth_settings):
        auth_settings.auth_provider = "google"
        auth_settings.google_oauth_client_id = "cid"
        auth_settings.google_oauth_client_secret = "sec"
        block = GoogleProvider().consent_block(_make_ctx())
        assert '<a class="provider-btn"' in block
        assert "accounts.google.com" in block and "client_id=cid" in block
        # redirect_uri is percent-encoded in the href query string
        assert "google%2Fcallback" in block

    def test_consent_block_not_configured(self, auth_settings):
        auth_settings.auth_provider = "google"
        auth_settings.google_oauth_client_id = ""
        auth_settings.google_oauth_client_secret = ""
        block = GoogleProvider().consent_block(_make_ctx())
        assert "not configured" in block.lower()


class TestConsentComposition:
    def test_token_provider_block_empty(self, auth_settings):
        from login_providers import TokenProvider
        assert TokenProvider().consent_block(_make_ctx()) == ""

    def test_supabase_consent_block_has_button_and_url(self, auth_settings):
        from login_providers import SupabaseProvider
        auth_settings.supabase_url = "https://abc.supabase.co"
        auth_settings.supabase_anon_key = "anon"
        block = SupabaseProvider().consent_block(_make_ctx())
        assert "ns-google-btn" in block
        assert "abc.supabase.co" in block
        assert "supabase/callback" in block

    def _render(self, provider_block, show_token_form):
        from oauth import _render_consent
        resp = _render_consent(
            client_id="c", redirect_uri="https://r/cb", state="s",
            code_challenge="h", code_challenge_method="S256", resource="",
            provider_block=provider_block, show_token_form=show_token_form,
        )
        return resp.body.decode()

    def test_page_shows_both_provider_and_token(self):
        # The headline requirement: Google button AND token paste, with a divider.
        body = self._render('<a class="provider-btn" href="x">G</a>', True)
        assert '<a class="provider-btn"' in body
        assert 'name="token"' in body
        assert '<div class="divider">' in body

    def test_page_provider_only_when_paste_disabled(self):
        body = self._render('<a class="provider-btn" href="x">G</a>', False)
        assert '<a class="provider-btn"' in body
        assert 'name="token"' not in body
        assert '<div class="divider">' not in body

    def test_page_token_only_in_token_mode(self):
        body = self._render("", True)
        assert 'name="token"' in body
        assert '<a class="provider-btn"' not in body
        assert '<div class="divider">' not in body


# ── durable identity store + resolver + link flow ─────────────────────────


class _FakeRedis:
    """Tiny in-memory async stand-in for redis.asyncio (hash ops only)."""

    def __init__(self):
        self.h = {}

    async def hget(self, name, key):
        return self.h.get(name, {}).get(key)

    async def hset(self, name, key, val):
        self.h.setdefault(name, {})[key] = val

    async def hdel(self, name, key):
        self.h.get(name, {}).pop(key, None)

    async def hgetall(self, name):
        return dict(self.h.get(name, {}))

    async def aclose(self):
        pass


@pytest.fixture(autouse=True)
def fake_store(monkeypatch):
    """Isolate EVERY test from real Redis: the identity store always talks to a
    fresh in-memory fake. autouse so no test can leak links into a dev instance
    (resolver promotion writes to the store). Tests that inspect it just request
    ``fake_store`` to get the instance."""
    fake = _FakeRedis()
    monkeypatch.setattr(identity_store, "_client", lambda: fake)
    return fake


class TestIdentityStore:
    @pytest.mark.asyncio
    async def test_link_and_resolve_by_sub_and_email(self, fake_store):
        await identity_store.link("alice", sub="sub-123", email="Alice@Example.com")
        assert await identity_store.resolve("sub-123", None) == "alice"
        assert await identity_store.resolve(None, "alice@example.com") == "alice"  # normalized
        assert await identity_store.resolve("nope", "nobody@x.com") is None

    @pytest.mark.asyncio
    async def test_sub_precedence_over_email(self, fake_store):
        await identity_store.link("by-sub", sub="s1")
        await identity_store.link("by-email", email="dup@example.com")
        # sub wins when both could match
        assert await identity_store.resolve("s1", "dup@example.com") == "by-sub"

    @pytest.mark.asyncio
    async def test_unlink(self, fake_store):
        await identity_store.link("alice", sub="s", email="a@x.com")
        await identity_store.unlink(sub="s", email="a@x.com")
        assert await identity_store.resolve("s", "a@x.com") is None

    @pytest.mark.asyncio
    async def test_degrades_when_redis_down(self, monkeypatch):
        def boom():
            raise ConnectionError("no redis")
        monkeypatch.setattr(identity_store, "_client", boom)
        assert await identity_store.resolve("s", "a@x.com") is None   # no raise
        assert await identity_store.link("alice", sub="s") is False    # best-effort


class TestResolver:
    @pytest.mark.asyncio
    async def test_store_hit_is_linked(self, auth_settings, fake_store):
        await identity_store.link("ehfazrezwan", sub="goog-1", email="a@example.com")
        uid, linked = await resolve_login_identity("goog-1", "a@example.com")
        assert (uid, linked) == ("ehfazrezwan", True)

    @pytest.mark.asyncio
    async def test_env_seed_promoted_to_store(self, auth_settings, fake_store):
        auth_settings.auth_identity_map = "a@example.com:ehfazrezwan"
        uid, linked = await resolve_login_identity("goog-2", "a@example.com")
        assert (uid, linked) == ("ehfazrezwan", True)
        # seed should now be persisted durably (so the env var can be removed)
        assert await identity_store.resolve("goog-2", None) == "ehfazrezwan"

    @pytest.mark.asyncio
    async def test_new_user_slug_unlinked(self, auth_settings, fake_store):
        auth_settings.auth_identity_map = ""
        uid, linked = await resolve_login_identity("goog-3", "fresh@example.com")
        assert (uid, linked) == ("fresh-example.com", False)


class TestResolveIdentityResult:
    @pytest.mark.asyncio
    async def test_linked_returns_login_result(self, auth_settings, fake_store):
        auth_settings.auth_allowed_domains = "example.com"
        await identity_store.link("ehfazrezwan", sub="g1", email="a@example.com")
        res = await lp._resolve_identity(
            _make_ctx(), "a@example.com", True, "g1", trust_external_gate=False)
        assert isinstance(res, LoginResult) and res.user_id == "ehfazrezwan"

    @pytest.mark.asyncio
    async def test_unlinked_returns_needs_link(self, auth_settings, fake_store):
        auth_settings.auth_allowed_domains = "example.com"
        auth_settings.auth_identity_map = ""
        res = await lp._resolve_identity(
            _make_ctx(), "new@example.com", True, "g9", trust_external_gate=False)
        assert isinstance(res, LoginNeedsLink)
        assert res.suggested_user_id == "new-example.com"
        assert res.sub == "g9"

    @pytest.mark.asyncio
    async def test_denied_returns_error_before_link(self, auth_settings, fake_store):
        auth_settings.auth_allowed_domains = "example.com"
        res = await lp._resolve_identity(
            _make_ctx(), "outsider@gmail.com", True, "g8", trust_external_gate=False)
        assert isinstance(res, LoginError) and res.status == 403


class TestLinkTicket:
    def test_roundtrip(self, auth_settings):
        needs = LoginNeedsLink(_make_ctx(), "a@example.com", "g1", "a-example.com")
        claims = verify_link_ticket(sign_link_ticket(needs, _SECRET), _SECRET)
        assert claims["sub"] == "g1"
        assert claims["email"] == "a@example.com"
        assert claims["suggested_user_id"] == "a-example.com"
        assert claims["ctx"] == _make_ctx()

    def test_wrong_secret_and_typ_rejected(self, auth_settings):
        needs = LoginNeedsLink(_make_ctx(), "a@example.com", "g1", "a-example.com")
        tok = sign_link_ticket(needs, _SECRET)
        assert verify_link_ticket(tok, "other") is None
        # a login_state token must not be redeemable as a link ticket
        assert verify_link_ticket(sign_login_state(_make_ctx(), _SECRET), _SECRET) is None


class TestOAuthLinkEndpoint:
    @pytest.mark.asyncio
    async def test_action_new_provisions_and_links(self, auth_settings, fake_store):
        import oauth
        needs = LoginNeedsLink(_make_ctx(), "new@example.com", "g1", "new-example.com")
        ticket = sign_link_ticket(needs, _SECRET)
        resp = await oauth.oauth_link(None, ticket=ticket, action="new", token="")
        assert resp.status_code == 303
        assert "code=" in resp.headers["location"]
        # mapping persisted so next login skips the page
        assert await identity_store.resolve("g1", "new@example.com") == "new-example.com"

    @pytest.mark.asyncio
    async def test_action_link_with_valid_token_claims_existing(self, auth_settings, fake_store):
        import oauth
        from tokens import issue_user_token
        existing = issue_user_token("ehfazrezwan", _SECRET, 3600)
        needs = LoginNeedsLink(_make_ctx(), "a@example.com", "g2", "a-example.com")
        ticket = sign_link_ticket(needs, _SECRET)
        resp = await oauth.oauth_link(None, ticket=ticket, action="link", token=existing)
        assert resp.status_code == 303
        assert await identity_store.resolve("g2", "a@example.com") == "ehfazrezwan"

    @pytest.mark.asyncio
    async def test_action_link_bad_token_rerenders(self, auth_settings, fake_store):
        import oauth
        needs = LoginNeedsLink(_make_ctx(), "a@example.com", "g3", "a-example.com")
        ticket = sign_link_ticket(needs, _SECRET)
        resp = await oauth.oauth_link(None, ticket=ticket, action="link", token="garbage")
        assert resp.status_code == 400
        assert b"invalid or expired" in resp.body

    @pytest.mark.asyncio
    async def test_expired_ticket_errors(self, auth_settings, fake_store):
        import oauth
        resp = await oauth.oauth_link(None, ticket="garbage", action="new", token="")
        assert resp.status_code == 400
