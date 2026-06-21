"""Built-in OAuth 2.1 Authorization Server for the Neuralscape MCP connector.

Why this exists
---------------
Claude Cowork / claude.ai connect to a remote MCP server as a *custom
connector*. That UI only supports OAuth — there is no field for a static
Bearer token or custom headers (anthropics/claude-ai-mcp#112). So to give
Cowork users a first-class "Add connector → Connect → log in" experience,
the Neuralscape service has to speak OAuth 2.1 as a Resource Server *and*
ship an Authorization Server.

Design: token-as-login, (almost) stateless
-------------------------------------------
There is no user/password database — identity has always come from
admin-issued per-user HMAC tokens (``scripts/issue_user_token.py``). So the
consent screen simply asks the user to paste that token *once*. We verify it,
read the ``user_id``, and from then on Anthropic holds short-lived OAuth
access tokens (silently refreshed). The user never pastes anything again.

Everything is signed with the existing ``NEURALSCAPE_USER_TOKEN_SECRET`` via
``tokens.sign_payload`` and distinguished by a ``typ`` claim:

* ``typ="client"``  → the Dynamic-Client-Registration ``client_id`` itself
  (encodes the registered ``redirect_uris``; no server-side client store).
* ``typ="code"``    → an authorization code (encodes user_id, client_id,
  redirect_uri, PKCE challenge; ~5 min TTL; single-use via Redis).
* ``typ="access"``  → the Bearer access token. Identical in shape to an
  admin token, so the existing ``BearerAuthMiddleware`` validates it unchanged.
* ``typ="refresh"`` → the refresh token.

``verify_user_token`` only accepts ``typ`` in {absent, "access"}, so codes /
refresh / client tokens can never be replayed as a Bearer credential.

The single piece of real state is a short-TTL Redis key per consumed auth
code (replay protection); it degrades to best-effort if Redis is down.
"""

from __future__ import annotations

import base64
import hashlib
import html
import logging
import secrets
import time
from urllib.parse import urlencode, urlparse

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from config import settings
from login_providers import (
    AuthorizeContext,
    GoogleProvider,
    LoginError,
    SupabaseProvider,
    _supabase_finish_page,
    get_login_provider,
)
from tokens import issue_user_token, sign_payload, verify_payload, verify_user_token

logger = logging.getLogger(__name__)

router = APIRouter()

SCOPE = "neuralscape"
_CODE_TTL = 300  # authorization code lifetime (seconds)
_CLIENT_TTL: int | None = None  # client_id registrations don't expire
# RFC 8252 §7.3 / OAuth 2.1: redirect URIs must be https, except loopback
# interface URIs which native clients may serve over http.
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_valid_redirect_uri(uri: str) -> bool:
    """Accept only absolute https:// URIs (or http:// to a loopback host) with
    no fragment. These are signed into the client_id and later trusted by
    /oauth/authorize as 303 redirect targets, so they must be constrained at
    registration time to avoid open-redirect / token-leak to plaintext hosts."""
    if not isinstance(uri, str) or not uri:
        return False
    parsed = urlparse(uri)
    if not parsed.netloc or parsed.fragment:
        return False
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https":
        return True
    return parsed.scheme == "http" and host in _LOOPBACK_HOSTS


# ── helpers ──────────────────────────────────────────────────────────────


def _secret() -> str:
    return settings.neuralscape_user_token_secret


def _base_url() -> str:
    """Public base URL (no trailing slash). Empty when OAuth isn't configured."""
    return settings.neuralscape_public_url.rstrip("/")


def oauth_enabled() -> bool:
    """OAuth is offered only when both a public URL and a signing secret exist.

    Without the secret there's nothing to sign tokens with; without the public
    URL the discovery metadata can't name a reachable issuer.
    """
    return bool(_base_url() and _secret())


def _not_configured() -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": "OAuth is not enabled on this deployment"},
    )


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _verify_pkce(code_verifier: str, code_challenge: str) -> bool:
    """RFC 7636 S256: BASE64URL(SHA256(verifier)) == challenge."""
    if not code_verifier or not code_challenge:
        return False
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return secrets.compare_digest(_b64url(digest), code_challenge)


async def _consume_code_jti(jti: str) -> bool:
    """Mark an auth-code id used. Returns True if this is the first use.

    Backed by Redis ``SET NX EX`` for replay protection. If Redis is
    unavailable we log and allow the exchange (best-effort) rather than
    breaking login — the code is already single-client, PKCE-bound, and
    short-lived, so the residual replay window is small.
    """
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(settings.redis_url)
        try:
            ok = await client.set(f"oauth:code:{jti}", "1", nx=True, ex=_CODE_TTL)
            return bool(ok)
        finally:
            await client.aclose()
    except Exception as e:  # noqa: BLE001 — best-effort guard
        logger.warning("OAuth code replay-guard unavailable (%s); allowing", e)
        return True


# ── discovery metadata ───────────────────────────────────────────────────


@router.get("/.well-known/oauth-authorization-server")
async def authorization_server_metadata() -> JSONResponse:
    """RFC 8414 Authorization Server Metadata."""
    if not oauth_enabled():
        return _not_configured()
    base = _base_url()
    return JSONResponse(
        {
            "issuer": base,
            "authorization_endpoint": f"{base}/oauth/authorize",
            "token_endpoint": f"{base}/oauth/token",
            "registration_endpoint": f"{base}/oauth/register",
            "scopes_supported": [SCOPE],
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
        }
    )


def _protected_resource_metadata() -> dict:
    base = _base_url()
    return {
        "resource": f"{base}/mcp",
        "authorization_servers": [base],
        "scopes_supported": [SCOPE],
        "bearer_methods_supported": ["header"],
    }


@router.get("/.well-known/oauth-protected-resource")
async def protected_resource_metadata() -> JSONResponse:
    """RFC 9728 Protected Resource Metadata (the /mcp resource)."""
    if not oauth_enabled():
        return _not_configured()
    return JSONResponse(_protected_resource_metadata())


@router.get("/.well-known/oauth-protected-resource/mcp")
async def protected_resource_metadata_mcp() -> JSONResponse:
    """Path-suffixed PRM variant — some clients derive it from the resource path."""
    if not oauth_enabled():
        return _not_configured()
    return JSONResponse(_protected_resource_metadata())


# ── dynamic client registration (RFC 7591) ───────────────────────────────


@router.post("/oauth/register")
async def register(request: Request) -> JSONResponse:
    """Stateless Dynamic Client Registration.

    The returned ``client_id`` is itself a signed token carrying the client's
    ``redirect_uris`` — so there's no client database to keep. We only support
    public clients (PKCE, ``token_endpoint_auth_method: none``), which is what
    the Anthropic connector uses.
    """
    if not oauth_enabled():
        return _not_configured()
    try:
        body = await request.json()
    except Exception:
        body = {}
    redirect_uris = body.get("redirect_uris") or []
    if (
        not isinstance(redirect_uris, list)
        or not redirect_uris
        or not all(_is_valid_redirect_uri(u) for u in redirect_uris)
    ):
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_redirect_uri",
                "error_description": (
                    "redirect_uris must be a non-empty array of absolute https:// "
                    "URLs (http:// allowed only for loopback hosts)"
                ),
            },
        )

    issued_at = int(time.time())
    client_id = sign_payload(
        {"typ": "client", "redirect_uris": redirect_uris, "iat": issued_at},
        _secret(),
    )
    return JSONResponse(
        status_code=201,
        content={
            "client_id": client_id,
            "client_id_issued_at": issued_at,
            "redirect_uris": redirect_uris,
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": SCOPE,
        },
    )


def _decode_client(client_id: str) -> dict | None:
    payload = verify_payload(client_id, _secret())
    if payload is None or payload.get("typ") != "client":
        return None
    return payload


# ── authorization endpoint (consent) ─────────────────────────────────────


_CONSENT_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Connect Neuralscape</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; min-height: 100vh; display: grid; place-items: center;
         background: #0b0d12; color: #e7e9ee; }}
  .card {{ width: min(440px, 92vw); background: #151821; border: 1px solid #262b38;
          border-radius: 16px; padding: 28px 28px 24px; box-shadow: 0 12px 40px rgba(0,0,0,.45); }}
  h1 {{ font-size: 19px; margin: 0 0 4px; }}
  p  {{ font-size: 14px; line-height: 1.5; color: #aab1c2; margin: 8px 0; }}
  .who {{ font-size: 13px; color: #8b93a7; margin-bottom: 18px; }}
  label {{ display:block; font-size: 13px; color:#cbd2e1; margin: 16px 0 6px; }}
  textarea {{ width: 100%; box-sizing: border-box; min-height: 84px; resize: vertical;
             font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
             background:#0d0f16; color:#e7e9ee; border:1px solid #2c3242; border-radius:10px;
             padding:10px; }}
  button {{ margin-top: 18px; width: 100%; padding: 12px; font-size: 15px; font-weight: 600;
           border: 0; border-radius: 10px; background: #5b7cfa; color: #fff; cursor: pointer; }}
  button:hover {{ background: #4f70ee; }}
  .err {{ background: #2a1518; border: 1px solid #5b2a2f; color: #ffb4b4; font-size: 13px;
         border-radius: 10px; padding: 10px 12px; margin: 14px 0 0; }}
  .hint {{ font-size: 12px; color: #7b8198; }}
  code {{ background:#0d0f16; padding:1px 5px; border-radius:5px; font-size:12px; }}
</style>
</head>
<body>
  <div class="card">
    <h1>Connect Neuralscape memory</h1>
    <p class="who">{client_label} is requesting access to your Neuralscape memories.</p>
    {error_html}
    <form method="post" action="/oauth/authorize">
      <input type="hidden" name="client_id" value="{client_id}">
      <input type="hidden" name="redirect_uri" value="{redirect_uri}">
      <input type="hidden" name="state" value="{state}">
      <input type="hidden" name="code_challenge" value="{code_challenge}">
      <input type="hidden" name="code_challenge_method" value="{code_challenge_method}">
      <input type="hidden" name="resource" value="{resource}">
      <label for="token">Your Neuralscape access token</label>
      <textarea id="token" name="token" placeholder="paste the token your admin issued you" autofocus></textarea>
      <p class="hint">Issued by your team admin via <code>issue_user_token.py</code>. You only paste this once.</p>
      <button type="submit">Authorize</button>
    </form>
  </div>
</body>
</html>"""


def _render_consent(
    *,
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    code_challenge_method: str,
    resource: str,
    client_label: str = "Claude",
    error: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    error_html = f'<div class="err">{html.escape(error)}</div>' if error else ""
    page = _CONSENT_PAGE.format(
        client_label=html.escape(client_label),
        client_id=html.escape(client_id),
        redirect_uri=html.escape(redirect_uri),
        state=html.escape(state or ""),
        code_challenge=html.escape(code_challenge or ""),
        code_challenge_method=html.escape(code_challenge_method or ""),
        resource=html.escape(resource or ""),
        error_html=error_html,
    )
    return HTMLResponse(content=page, status_code=status_code)


@router.get("/oauth/authorize", response_model=None)
async def authorize(request: Request) -> HTMLResponse | JSONResponse:
    """Validate the authorize request, then hand the human-login step to the
    configured provider (token-paste page, Google redirect, or Supabase
    sign-in page per AUTH_PROVIDER)."""
    if not oauth_enabled():
        return _not_configured()
    q = request.query_params
    response_type = q.get("response_type", "")
    client_id = q.get("client_id", "")
    redirect_uri = q.get("redirect_uri", "")
    code_challenge = q.get("code_challenge", "")
    code_challenge_method = q.get("code_challenge_method", "")
    state = q.get("state", "")
    resource = q.get("resource", "")

    if response_type != "code":
        return JSONResponse(
            status_code=400,
            content={"error": "unsupported_response_type"},
        )
    client = _decode_client(client_id)
    if client is None:
        return JSONResponse(status_code=400, content={"error": "invalid_client"})
    if redirect_uri not in client.get("redirect_uris", []):
        # Never redirect to an unregistered URI — surface the error locally.
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request", "error_description": "redirect_uri mismatch"},
        )
    # PKCE is mandatory (OAuth 2.1 for public clients).
    if not code_challenge or code_challenge_method != "S256":
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request", "error_description": "PKCE S256 required"},
        )

    # Delegate the human-login step to the configured provider. Everything
    # validated above (client, redirect_uri, PKCE) is preserved across any
    # external IdP redirect inside the provider's signed login-state token.
    ctx = AuthorizeContext(
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        resource=resource,
    )

    def render_consent() -> HTMLResponse:
        return _render_consent(
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            resource=resource,
        )

    return await get_login_provider().begin(ctx, render_consent)


@router.post("/oauth/authorize", response_model=None)
async def authorize_submit(
    request: Request,
    client_id: str = Form(...),
    redirect_uri: str = Form(...),
    code_challenge: str = Form(...),
    token: str = Form(""),
    state: str = Form(""),
    code_challenge_method: str = Form("S256"),
    resource: str = Form(""),
) -> HTMLResponse | RedirectResponse | JSONResponse:
    """Validate the pasted login token and issue an authorization code."""
    if not oauth_enabled():
        return _not_configured()
    # The token-paste form is only the login mechanism when AUTH_PROVIDER=token.
    # Under google/supabase the consent page never renders it; reject as a
    # defense against a hand-crafted POST bypassing the configured provider.
    if settings.auth_provider != "token":
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request",
                     "error_description": "token login is disabled; use the configured provider"},
        )

    client = _decode_client(client_id)
    if client is None:
        return JSONResponse(status_code=400, content={"error": "invalid_client"})
    if redirect_uri not in client.get("redirect_uris", []):
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request", "error_description": "redirect_uri mismatch"},
        )

    # "Login": the pasted credential is an admin-issued per-user token.
    payload = verify_user_token(token.strip(), _secret()) if token.strip() else None
    if payload is None:
        return _render_consent(
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            resource=resource,
            error="That token is invalid or expired. Paste the token your admin issued you.",
            status_code=400,
        )

    ctx = AuthorizeContext(
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        resource=resource,
    )
    return _issue_code_and_redirect(ctx, payload["user_id"])


def _issue_code_and_redirect(ctx: AuthorizeContext, user_id: str) -> RedirectResponse:
    """Mint a PKCE-bound authorization code for ``user_id`` and 303 back to the
    MCP client's redirect_uri. Shared by every login provider — the only thing
    that varies upstream is how ``user_id`` was authenticated."""
    code = sign_payload(
        {
            "typ": "code",
            "user_id": user_id,
            "client_id": ctx.client_id,
            "redirect_uri": ctx.redirect_uri,
            "code_challenge": ctx.code_challenge,
            "jti": secrets.token_urlsafe(12),
            "exp": int(time.time()) + _CODE_TTL,
        },
        _secret(),
    )
    # Build the query with proper percent-encoding: an opaque `state` may carry
    # reserved characters (&, =, #, spaces) that would otherwise corrupt the
    # client's CSRF/state check when concatenated raw.
    params = {"code": code}
    if ctx.state:
        params["state"] = ctx.state
    sep = "&" if "?" in ctx.redirect_uri else "?"
    location = f"{ctx.redirect_uri}{sep}{urlencode(params)}"
    # 303 so the browser issues a GET to the client's redirect URI.
    return RedirectResponse(url=location, status_code=303)


def _login_error_page(err: LoginError) -> HTMLResponse:
    """Render a login failure (allowlist denial, verification failure) as a
    simple page rather than a raw JSON blob the user lands on mid-browser-flow."""
    safe = html.escape(err.message)
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Sign-in failed</title><style>"
        ":root{color-scheme:light dark;}"
        "body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;"
        "margin:0;min-height:100vh;display:grid;place-items:center;background:#0b0d12;color:#e7e9ee;}"
        ".card{width:min(440px,92vw);background:#151821;border:1px solid #262b38;border-radius:16px;"
        "padding:28px;box-shadow:0 12px 40px rgba(0,0,0,.45);}"
        "h1{font-size:19px;margin:0 0 8px;}p{font-size:14px;color:#aab1c2;}"
        "</style></head><body><div class='card'><h1>Sign-in failed</h1>"
        f"<p>{safe}</p></div></body></html>"
    )
    return HTMLResponse(content=page, status_code=err.status)


# ── federated login callbacks (google / supabase) ────────────────────────


@router.get("/oauth/google/callback", response_model=None)
async def google_callback(request: Request) -> RedirectResponse | HTMLResponse | JSONResponse:
    """Google OIDC redirect target: verify the id_token, allowlist-check the
    email, then continue the MCP authorization-code flow."""
    if not oauth_enabled():
        return _not_configured()
    if settings.auth_provider != "google":
        return JSONResponse(status_code=404, content={"detail": "google login not enabled"})
    result = await GoogleProvider().complete(request)
    if isinstance(result, LoginError):
        return _login_error_page(result)
    return _issue_code_and_redirect(result.ctx, result.user_id)


@router.get("/oauth/supabase/callback", response_model=None)
async def supabase_callback_get(request: Request) -> HTMLResponse | JSONResponse:
    """Supabase redirect target (browser): render the page that exchanges the
    PKCE code for a session and POSTs the access token back to the server."""
    if not oauth_enabled():
        return _not_configured()
    if settings.auth_provider != "supabase":
        return JSONResponse(status_code=404, content={"detail": "supabase login not enabled"})
    return HTMLResponse(_supabase_finish_page())


@router.post("/oauth/supabase/callback", response_model=None)
async def supabase_callback_post(request: Request) -> RedirectResponse | HTMLResponse | JSONResponse:
    """Verify the posted Supabase session JWT, then continue the MCP flow."""
    if not oauth_enabled():
        return _not_configured()
    if settings.auth_provider != "supabase":
        return JSONResponse(status_code=404, content={"detail": "supabase login not enabled"})
    result = await SupabaseProvider().complete(request)
    if isinstance(result, LoginError):
        return _login_error_page(result)
    return _issue_code_and_redirect(result.ctx, result.user_id)


# ── token endpoint ───────────────────────────────────────────────────────


def _token_response(user_id: str) -> JSONResponse:
    access = issue_user_token(
        user_id, _secret(), settings.oauth_access_ttl, typ="access"
    )
    refresh = sign_payload(
        {
            "typ": "refresh",
            "user_id": user_id,
            "exp": int(time.time()) + settings.oauth_refresh_ttl,
        },
        _secret(),
    )
    return JSONResponse(
        {
            "access_token": access,
            "token_type": "Bearer",
            "expires_in": settings.oauth_access_ttl,
            "refresh_token": refresh,
            "scope": SCOPE,
        }
    )


def _token_error(error: str, description: str = "", status_code: int = 400) -> JSONResponse:
    body = {"error": error}
    if description:
        body["error_description"] = description
    return JSONResponse(status_code=status_code, content=body)


@router.post("/oauth/token")
async def token(
    request: Request,
    grant_type: str = Form(...),
    code: str = Form(""),
    code_verifier: str = Form(""),
    redirect_uri: str = Form(""),
    client_id: str = Form(""),
    refresh_token: str = Form(""),
) -> JSONResponse:
    """Exchange an authorization code (with PKCE) or a refresh token for tokens."""
    if not oauth_enabled():
        return _not_configured()

    if grant_type == "authorization_code":
        claims = verify_payload(code, _secret())
        if claims is None or claims.get("typ") != "code":
            return _token_error("invalid_grant", "code is invalid or expired")
        # The code is bound to the client_id and redirect_uri it was issued for.
        # Require both on exchange and demand an exact match — otherwise a caller
        # could omit them and redeem any otherwise-valid code, defeating the
        # binding.
        if not client_id or claims.get("client_id") != client_id:
            return _token_error("invalid_grant", "client mismatch")
        if not redirect_uri or claims.get("redirect_uri") != redirect_uri:
            return _token_error("invalid_grant", "redirect_uri mismatch")
        if not _verify_pkce(code_verifier, claims.get("code_challenge", "")):
            return _token_error("invalid_grant", "PKCE verification failed")
        jti = claims.get("jti", "")
        if jti and not await _consume_code_jti(jti):
            return _token_error("invalid_grant", "authorization code already used")
        user_id = claims.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            return _token_error("invalid_grant", "code missing user")
        return _token_response(user_id)

    if grant_type == "refresh_token":
        claims = verify_payload(refresh_token, _secret())
        if claims is None or claims.get("typ") != "refresh":
            return _token_error("invalid_grant", "refresh_token is invalid or expired")
        user_id = claims.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            return _token_error("invalid_grant", "refresh_token missing user")
        return _token_response(user_id)

    return _token_error("unsupported_grant_type", f"grant_type '{grant_type}' not supported")
