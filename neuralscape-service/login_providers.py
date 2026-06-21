"""Pluggable consent-screen login providers for the OAuth Authorization Server.

The MCP-facing OAuth flow (Dynamic Client Registration → ``/oauth/authorize``
→ auth code → ``/oauth/token`` → access/refresh tokens) is identical no matter
how the human authenticates. The *only* part that varies is the consent step:

* ``token``    — paste an admin-issued HMAC token (legacy default).
* ``google``   — Sign in with Google (OIDC), gated by the env email allowlist.
* ``supabase`` — Sign in via Supabase (Google under the hood); the allowlist is
                 Supabase's Before-User-Created hook, with the env allowlist as
                 an optional extra gate.

A provider's job is to turn an inbound login into a verified, allowlisted
``user_id`` (or a rejection). The in-flight OAuth ``/authorize`` parameters are
carried across the external redirect inside a signed, short-lived *login state*
token (no server-side session needed).

``AUTH_PROVIDER`` selects the provider per deployment. Pre-issued Bearer tokens
keep working as credentials via ``BearerAuthMiddleware`` regardless of this
setting, so CLI/CI/e2e are unaffected.
"""

from __future__ import annotations

import html
import logging
import secrets
import time
from dataclasses import asdict, dataclass
from urllib.parse import urlencode

import httpx
import jwt
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from jwt import PyJWKClient

from allowlist import is_email_allowed
from config import settings
from identity import derive_user_id
from tokens import sign_payload, verify_payload, verify_user_token

logger = logging.getLogger(__name__)

# Login state lifetime — the user has this long to finish the external IdP
# round-trip. Short, single-purpose, signed with the AS secret.
_LOGIN_STATE_TTL = 600  # 10 minutes

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_JWKS_URI = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = {"https://accounts.google.com", "accounts.google.com"}

# Lazily-built, cached JWKS clients (each fetches + caches signing keys).
_jwks_clients: dict[str, PyJWKClient] = {}


def _jwks_client(uri: str) -> PyJWKClient:
    client = _jwks_clients.get(uri)
    if client is None:
        client = PyJWKClient(uri, cache_keys=True)
        _jwks_clients[uri] = client
    return client


# ── login state (carries the /authorize params across the IdP redirect) ──


@dataclass
class AuthorizeContext:
    """The MCP-client ``/oauth/authorize`` request, preserved across login."""

    client_id: str
    redirect_uri: str
    state: str
    code_challenge: str
    code_challenge_method: str
    resource: str


def sign_login_state(ctx: AuthorizeContext, secret: str) -> str:
    """Sign the authorize context into a short-lived ``login_state`` token."""
    claims = {
        "typ": "login_state",
        "nonce": secrets.token_urlsafe(8),
        "exp": int(time.time()) + _LOGIN_STATE_TTL,
        **asdict(ctx),
    }
    return sign_payload(claims, secret)


def verify_login_state(token: str, secret: str) -> AuthorizeContext | None:
    """Verify a ``login_state`` token and rebuild the authorize context."""
    claims = verify_payload(token, secret)
    if claims is None or claims.get("typ") != "login_state":
        return None
    try:
        return AuthorizeContext(
            client_id=claims["client_id"],
            redirect_uri=claims["redirect_uri"],
            state=claims.get("state", ""),
            code_challenge=claims["code_challenge"],
            code_challenge_method=claims.get("code_challenge_method", "S256"),
            resource=claims.get("resource", ""),
        )
    except KeyError:
        return None


# ── login results ────────────────────────────────────────────────────────


@dataclass
class LoginResult:
    """A completed, authorized login."""

    ctx: AuthorizeContext
    user_id: str


@dataclass
class LoginError:
    """A failed/denied login. ``status`` is the HTTP code to surface."""

    message: str
    status: int = 400


def _resolve_identity(
    ctx: AuthorizeContext,
    email: str,
    email_verified: bool,
    *,
    trust_external_gate: bool,
) -> LoginResult | LoginError:
    """Apply the allowlist + identity map to a verified email.

    ``trust_external_gate`` is True for Supabase (its Before-User-Created hook
    already enforced the allowlist) — there the env allowlist is only applied
    as an extra gate *when configured*. For Google it is False: the env
    allowlist is the sole gate and must be configured.
    """
    domains = settings.allowed_domains_set()
    emails = settings.email_allowlist_set()

    if trust_external_gate:
        if not email or not email_verified:
            return LoginError("Email is missing or unverified.", 403)
        # Only re-check when an env allowlist is also configured.
        if (domains or emails) and not is_email_allowed(
            email, email_verified=email_verified,
            allowed_domains=domains, email_allowlist=emails,
        ):
            return LoginError("This account is not authorized.", 403)
    else:
        if not is_email_allowed(
            email, email_verified=email_verified,
            allowed_domains=domains, email_allowlist=emails,
        ):
            return LoginError("This account is not authorized.", 403)

    user_id = derive_user_id(email, settings.identity_map_dict())
    return LoginResult(ctx=ctx, user_id=user_id)


# ── provider base ────────────────────────────────────────────────────────


class LoginProvider:
    """Base interface. ``begin`` renders/starts the login for ``GET
    /oauth/authorize``; redirect-based providers also implement callbacks."""

    name = "base"

    async def begin(self, ctx: AuthorizeContext, render_consent) -> Response:
        raise NotImplementedError


# ── token provider (paste an admin HMAC token) ───────────────────────────


class TokenProvider(LoginProvider):
    """Legacy default: render the token-paste consent page. The POST handler
    in ``oauth.py`` verifies the pasted token (kept there for back-compat)."""

    name = "token"

    async def begin(self, ctx: AuthorizeContext, render_consent) -> Response:
        return render_consent()


# ── google provider (OIDC) ───────────────────────────────────────────────


class GoogleProvider(LoginProvider):
    name = "google"

    def _redirect_uri(self) -> str:
        return f"{settings.neuralscape_public_url.rstrip('/')}/oauth/google/callback"

    async def begin(self, ctx: AuthorizeContext, render_consent) -> Response:
        if not settings.google_oauth_client_id or not settings.google_oauth_client_secret:
            return JSONResponse(
                status_code=500,
                content={"error": "server_error",
                         "error_description": "Google login is not configured"},
            )
        state = sign_login_state(ctx, settings.neuralscape_user_token_secret)
        params = {
            "response_type": "code",
            "client_id": settings.google_oauth_client_id,
            "redirect_uri": self._redirect_uri(),
            "scope": "openid email profile",
            "state": state,
            "access_type": "online",
            "prompt": "select_account",
        }
        return RedirectResponse(
            url=f"{GOOGLE_AUTH_ENDPOINT}?{urlencode(params)}", status_code=302
        )

    async def complete(self, request: Request) -> LoginResult | LoginError:
        q = request.query_params
        if q.get("error"):
            return LoginError(f"Google sign-in failed: {q.get('error')}", 400)
        ctx = verify_login_state(q.get("state", ""), settings.neuralscape_user_token_secret)
        if ctx is None:
            return LoginError("Login session expired or invalid. Try again.", 400)
        code = q.get("code", "")
        if not code:
            return LoginError("Missing authorization code from Google.", 400)

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    GOOGLE_TOKEN_ENDPOINT,
                    data={
                        "grant_type": "authorization_code",
                        "code": code,
                        "client_id": settings.google_oauth_client_id,
                        "client_secret": settings.google_oauth_client_secret,
                        "redirect_uri": self._redirect_uri(),
                    },
                )
            if resp.status_code != 200:
                logger.warning("Google token exchange failed: %s", resp.text[:200])
                return LoginError("Google token exchange failed.", 502)
            id_token = resp.json().get("id_token")
        except httpx.HTTPError as e:
            logger.warning("Google token exchange error: %s", e)
            return LoginError("Could not reach Google to complete sign-in.", 502)
        if not id_token:
            return LoginError("Google did not return an id_token.", 502)

        claims = _verify_oidc_jwt(
            id_token,
            jwks_uri=GOOGLE_JWKS_URI,
            audience=settings.google_oauth_client_id,
            issuers=GOOGLE_ISSUERS,
        )
        if claims is None:
            return LoginError("Could not verify Google identity token.", 401)

        email = claims.get("email", "")
        email_verified = bool(claims.get("email_verified"))
        return _resolve_identity(ctx, email, email_verified, trust_external_gate=False)


# ── supabase provider (Supabase Auth → Google) ───────────────────────────


class SupabaseProvider(LoginProvider):
    name = "supabase"

    def _callback_uri(self, state: str) -> str:
        base = settings.neuralscape_public_url.rstrip("/")
        return f"{base}/oauth/supabase/callback?state={state}"

    async def begin(self, ctx: AuthorizeContext, render_consent) -> Response:
        if not settings.supabase_url or not settings.supabase_anon_key:
            return JSONResponse(
                status_code=500,
                content={"error": "server_error",
                         "error_description": "Supabase login is not configured"},
            )
        state = sign_login_state(ctx, settings.neuralscape_user_token_secret)
        return HTMLResponse(_supabase_start_page(state))

    async def complete(self, request: Request) -> LoginResult | LoginError:
        form = await request.form()
        access_token = (form.get("access_token") or "").strip()
        state = (form.get("state") or "").strip()
        ctx = verify_login_state(state, settings.neuralscape_user_token_secret)
        if ctx is None:
            return LoginError("Login session expired or invalid. Try again.", 400)
        if not access_token:
            return LoginError("No Supabase session was provided.", 400)

        claims = _verify_supabase_jwt(access_token)
        if claims is None:
            return LoginError("Could not verify Supabase session.", 401)

        email = claims.get("email", "")
        # Supabase nests the verification flag under user_metadata; accept the
        # top-level flag too for forward-compat.
        meta = claims.get("user_metadata") or {}
        email_verified = bool(claims.get("email_verified") or meta.get("email_verified"))
        return _resolve_identity(ctx, email, email_verified, trust_external_gate=True)


# ── JWT verification helpers ─────────────────────────────────────────────


def _verify_oidc_jwt(token: str, *, jwks_uri: str, audience: str, issuers: set[str]) -> dict | None:
    """Verify an RS256 OIDC id_token against a JWKS, audience, and issuer set."""
    try:
        signing_key = _jwks_client(jwks_uri).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            options={"require": ["exp"], "verify_iss": False},
        )
    except (jwt.InvalidTokenError, Exception) as e:  # noqa: BLE001 — JWKS fetch can raise
        logger.warning("OIDC id_token verification failed: %s", e)
        return None
    if claims.get("iss") not in issuers:
        logger.warning("OIDC id_token issuer mismatch: %s", claims.get("iss"))
        return None
    return claims


def _verify_supabase_jwt(token: str) -> dict | None:
    """Verify a Supabase session JWT — HS256 with the project secret if set,
    else the project's asymmetric JWKS."""
    try:
        if settings.supabase_jwt_secret:
            return jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
                options={"require": ["exp"]},
            )
        jwks_uri = f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
        signing_key = _jwks_client(jwks_uri).get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience="authenticated",
            options={"require": ["exp"]},
        )
    except (jwt.InvalidTokenError, Exception) as e:  # noqa: BLE001 — JWKS fetch can raise
        logger.warning("Supabase JWT verification failed: %s", e)
        return None


# ── Supabase browser pages ───────────────────────────────────────────────

_SUPABASE_HEAD = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Connect Neuralscape</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin:0; min-height:100vh; display:grid; place-items:center; background:#0b0d12; color:#e7e9ee; }}
  .card {{ width:min(440px,92vw); background:#151821; border:1px solid #262b38; border-radius:16px;
          padding:28px; box-shadow:0 12px 40px rgba(0,0,0,.45); text-align:center; }}
  h1 {{ font-size:19px; margin:0 0 10px; }}
  p {{ font-size:14px; color:#aab1c2; }}
  .err {{ background:#2a1518; border:1px solid #5b2a2f; color:#ffb4b4; font-size:13px;
         border-radius:10px; padding:10px 12px; margin-top:14px; }}
</style></head><body><div class="card">"""

_SUPABASE_FOOT = "</div></body></html>"


def _supabase_start_page(state: str) -> str:
    """Page rendered at GET /oauth/authorize — kicks off Supabase Google sign-in."""
    cb = html.escape(
        f"{settings.neuralscape_public_url.rstrip('/')}/oauth/supabase/callback?state={state}"
    )
    return (
        _SUPABASE_HEAD.format()
        + "<h1>Connect Neuralscape memory</h1><p>Redirecting you to sign in with Google…</p>"
        + f"""<script type="module">
import {{ createClient }} from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm';
const supabase = createClient({_js_str(settings.supabase_url)}, {_js_str(settings.supabase_anon_key)});
const {{ error }} = await supabase.auth.signInWithOAuth({{
  provider: 'google',
  options: {{ redirectTo: {_js_str_raw(cb)} }}
}});
if (error) {{ document.querySelector('.card').innerHTML =
  '<h1>Sign-in error</h1><p class=\\'err\\'>' + error.message + '</p>'; }}
</script>"""
        + _SUPABASE_FOOT
    )


def _supabase_finish_page() -> str:
    """Page rendered at GET /oauth/supabase/callback — exchanges the code for a
    session and POSTs the access token back to the server."""
    return (
        _SUPABASE_HEAD.format()
        + "<h1>Finishing sign-in…</h1><p>One moment.</p>"
        + f"""<script type="module">
import {{ createClient }} from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm';
const supabase = createClient({_js_str(settings.supabase_url)}, {_js_str(settings.supabase_anon_key)});
const state = new URLSearchParams(location.search).get('state') || '';
function fail(msg) {{ document.querySelector('.card').innerHTML =
  '<h1>Sign-in error</h1><p class=\\'err\\'>' + msg + '</p>'; }}
try {{
  // detectSessionInUrl (default) exchanges the ?code= on load.
  let {{ data: {{ session }} }} = await supabase.auth.getSession();
  if (!session) {{ await new Promise(r => setTimeout(r, 400));
    ({{ data: {{ session }} }} = await supabase.auth.getSession()); }}
  if (!session || !session.access_token) {{ fail('No session returned.'); }}
  else {{
    const f = document.createElement('form');
    f.method = 'POST'; f.action = location.origin + location.pathname;
    for (const [k,v] of Object.entries({{access_token: session.access_token, state}})) {{
      const i = document.createElement('input'); i.type='hidden'; i.name=k; i.value=v; f.appendChild(i);
    }}
    document.body.appendChild(f); f.submit();
  }}
}} catch (e) {{ fail(String(e)); }}
</script>"""
        + _SUPABASE_FOOT
    )


def _js_str(value: str) -> str:
    """Embed a Python string as a single-quoted JS string literal (escaped)."""
    return "'" + (value or "").replace("\\", "\\\\").replace("'", "\\'") + "'"


def _js_str_raw(value: str) -> str:
    """Like _js_str but for an already-HTML-escaped value used in JS context."""
    return "'" + (value or "").replace("'", "\\'") + "'"


# ── factory ──────────────────────────────────────────────────────────────

_PROVIDERS = {
    "token": TokenProvider,
    "google": GoogleProvider,
    "supabase": SupabaseProvider,
}


def get_login_provider() -> LoginProvider:
    """The active provider per ``AUTH_PROVIDER`` (defaults to token)."""
    return _PROVIDERS.get(settings.auth_provider, TokenProvider)()


# Re-export for oauth.py's token POST path (kept there for back-compat).
__all__ = [
    "AuthorizeContext", "LoginResult", "LoginError", "LoginProvider",
    "get_login_provider", "sign_login_state", "verify_login_state",
    "verify_user_token",
]
