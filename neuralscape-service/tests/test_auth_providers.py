"""Unit tests for the federated-login providers (google / supabase) and the
shared allowlist + identity-derivation logic.

No running services required — JWT verification is exercised with real HS256
tokens (Supabase) and a stubbed verify/exchange (Google).
"""

import re
import time

import jwt
import pytest

import login_providers as lp
from allowlist import email_domain, is_email_allowed, normalize_email
from config import Settings, settings
from identity import derive_user_id, slugify_email
from login_providers import (
    AuthorizeContext,
    GoogleProvider,
    LoginError,
    LoginResult,
    SupabaseProvider,
    get_login_provider,
    sign_login_state,
    verify_login_state,
)
from schemas import _ID_PATTERN

_SECRET = "test-signing-secret"
_ID_RE = re.compile(_ID_PATTERN)


# ── fixtures / stubs ──────────────────────────────────────────────────────


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
    async def test_trusts_external_gate_without_env_allowlist(self, auth_settings):
        # No env allowlist → trust Supabase's hook; slug a fresh user.
        auth_settings.supabase_jwt_secret = "sbsecret"
        token = _supabase_jwt("anyone@gmail.com")
        req = _Req(form={"access_token": token, "state": sign_login_state(_make_ctx(), _SECRET)})
        res = await SupabaseProvider().complete(req)
        assert isinstance(res, LoginResult)
        assert res.user_id == "anyone-gmail.com"

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
    async def test_happy_path_allowlisted(self, auth_settings, monkeypatch):
        auth_settings.auth_provider = "google"
        auth_settings.google_oauth_client_id = "cid"
        auth_settings.google_oauth_client_secret = "sec"
        auth_settings.auth_allowed_domains = "example.com"
        monkeypatch.setattr(lp.httpx, "AsyncClient", _FakeClient)
        monkeypatch.setattr(lp, "_verify_oidc_jwt",
                            lambda *a, **k: {"email": "x@example.com", "email_verified": True})
        state = sign_login_state(_make_ctx(), _SECRET)
        res = await GoogleProvider().complete(_Req(query={"code": "abc", "state": state}))
        assert isinstance(res, LoginResult)
        assert res.user_id == "x-example.com"

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

    @pytest.mark.asyncio
    async def test_begin_redirects_to_google(self, auth_settings):
        auth_settings.auth_provider = "google"
        auth_settings.google_oauth_client_id = "cid"
        auth_settings.google_oauth_client_secret = "sec"
        resp = await GoogleProvider().begin(_make_ctx(), lambda: None)
        assert resp.status_code == 302
        loc = resp.headers["location"]
        assert "accounts.google.com" in loc and "client_id=cid" in loc
        # redirect_uri is percent-encoded in the query string
        assert "google%2Fcallback" in loc
