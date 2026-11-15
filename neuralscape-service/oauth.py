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

The only real state is a Redis key per consumed single-use credential: one per
auth code (``oauth:code:<jti>``) and one per redeemed refresh token
(``oauth:refresh:<jti>``, rotate-and-consume) plus
``oauth:revoked_family:<fam>`` set when a rotated token is replayed. Both guards fail CLOSED: when
Redis is unreachable the token endpoint answers 503 ``temporarily_unavailable``
and issues nothing.
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
import identity_store
from login_providers import (
    AuthorizeContext,
    GoogleProvider,
    LoginError,
    LoginNeedsLink,
    SupabaseProvider,
    _supabase_finish_page,
    get_login_provider,
    sign_link_ticket,
    verify_link_ticket,
)
from tokens import (
    identity_fingerprint,
    issue_user_token,
    refresh_revocations,
    sign_payload,
    token_fingerprint,
    verify_payload,
    verify_user_token,
)
import tokens as _tokens

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


class _ReplayGuardUnavailable(Exception):
    """Redis could not be reached, so single-use cannot be enforced."""


_REDIS_TIMEOUT = 2  # seconds; a hung Redis must fail closed promptly, not stall /oauth/token


async def _redis_op(op):
    """Run ``op(client)`` against Redis with short timeouts. Any failure raises
    ``_ReplayGuardUnavailable``: callers must refuse (fail closed) rather than
    let a single-use credential be redeemed more than once."""
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            settings.redis_url,
            socket_connect_timeout=_REDIS_TIMEOUT,
            socket_timeout=_REDIS_TIMEOUT,
        )
        try:
            return await op(client)
        finally:
            await client.aclose()
    except Exception as e:  # noqa: BLE001 — any store failure means fail closed
        logger.warning("OAuth single-use guard unavailable (%s); refusing", e)
        raise _ReplayGuardUnavailable() from e


async def _consume_once(key: str, ttl: int) -> bool:
    """Atomically claim ``key`` (Redis ``SET NX EX``). True on the first claim,
    False if it was already claimed."""

    async def op(client):
        return bool(await client.set(key, "1", nx=True, ex=max(1, int(ttl))))

    return await _redis_op(op)


async def _family_revoked(fam: str) -> bool:
    """True if reuse of a token in this family was detected earlier."""

    async def op(client):
        return bool(await client.exists(f"oauth:revoked_family:{fam}"))

    return await _redis_op(op)


async def _revoke_family(fam: str, ttl: int) -> None:
    """Revoke every refresh token in a family (TTL = the refresh lifetime)."""

    async def op(client):
        await client.set(f"oauth:revoked_family:{fam}", "1", ex=max(1, int(ttl)))

    await _redis_op(op)


async def _consume_code_jti(jti: str) -> bool:
    """Mark an auth-code id used. True if this is the first use."""
    return await _consume_once(f"oauth:code:{jti}", _CODE_TTL)


async def _consume_refresh_jti(jti: str, ttl: int) -> bool:
    """Mark a refresh-token id redeemed. ``ttl`` is the token's remaining life,
    so the marker outlives the token and a replay can never slip through."""
    return await _consume_once(f"oauth:refresh:{jti}", ttl)


# ── HTML responses: anti-framing + hardening headers ─────────────────────

_HTML_SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    # frame-ancestors is the modern equivalent of X-Frame-Options. The pages
    # use inline CSS/JS and the Supabase page loads a CDN module, so no
    # script/style policy is imposed here; only the directives that cannot
    # break them.
    "Content-Security-Policy": "frame-ancestors 'none'; base-uri 'none'; object-src 'none'",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
}


def _src_revoked(src) -> bool:
    """True when a session's source credential/identity digest is revoked."""
    if not isinstance(src, str) or not src:
        return False
    digests = _tokens._revoked_digests()
    return digests is None or src.lower() in digests


def _digest_revoked(token: str) -> bool:
    digests = _tokens._revoked_digests()
    return digests is None or token_fingerprint(token) in digests


def _html_response(content: str, status_code: int = 200) -> HTMLResponse:
    """Every HTML page served under /oauth/* goes through here so none can be
    framed (clickjacking of the consent button) or leak the URL via Referer."""
    return HTMLResponse(
        content=content, status_code=status_code, headers=dict(_HTML_SECURITY_HEADERS)
    )


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
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; min-height: 100vh; display: grid; place-items: center;
         background: #0b0d12; color: #e7e9ee; overflow-wrap: anywhere; }}
  .card {{ width: min(440px, 92vw); max-width: 100%; min-width: 0; background: #151821; border: 1px solid #262b38;
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
           border: 0; border-radius: 10px; background: #3b5bdb; color: #fff; cursor: pointer; }}
  button:hover {{ background: #364fc7; }}
  .err {{ background: #2a1518; border: 1px solid #5b2a2f; color: #ffb4b4; font-size: 13px;
         border-radius: 10px; padding: 10px 12px; margin: 14px 0 0; }}
  .hint {{ font-size: 12px; color: #7b8198; overflow-wrap: anywhere; }}
  code {{ background:#0d0f16; padding:1px 5px; border-radius:5px; font-size:12px;
         overflow-wrap:anywhere; word-break:break-word; }}
  .provider-btn {{ display:flex; align-items:center; justify-content:center; gap:8px;
           width:100%; box-sizing:border-box; margin-top:8px; padding:11px; font-size:14px;
           font-weight:600; text-decoration:none; border-radius:10px; background:#fff;
           color:#1f2330; border:1px solid #d6d9e0; cursor:pointer; }}
  .provider-btn:hover {{ background:#f1f3f7; }}
  .divider {{ display:flex; align-items:center; text-align:center; color:#7b8198;
           font-size:12px; margin:20px 0 4px; }}
  .divider::before, .divider::after {{ content:""; flex:1; border-top:1px solid #262b38; }}
  .divider span {{ padding:0 10px; }}
</style>
</head>
<body>
  <div class="card">
    <h1>Connect Neuralscape memory</h1>
    <p class="who">{client_label} is requesting access to your Neuralscape memories.</p>
    {error_html}
    <!--PROVIDER_BLOCK-->
    <!--DIVIDER-->
    <!--TOKEN_FORM-->
  </div>
</body>
</html>"""

_TOKEN_FORM = """<form method="post" action="/oauth/authorize">
      <input type="hidden" name="client_id" value="{client_id}">
      <input type="hidden" name="redirect_uri" value="{redirect_uri}">
      <input type="hidden" name="state" value="{state}">
      <input type="hidden" name="code_challenge" value="{code_challenge}">
      <input type="hidden" name="code_challenge_method" value="{code_challenge_method}">
      <input type="hidden" name="resource" value="{resource}">
      <label for="token">Your Neuralscape access token</label>
      <textarea id="token" name="token" placeholder="paste the token your admin issued you"></textarea>
      <p class="hint">Issued by your team admin via <code>issue_user_token.py</code>. You only paste this once.</p>
      <button type="submit">Authorize</button>
    </form>"""


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
    provider_block: str = "",
    show_token_form: bool = True,
) -> HTMLResponse:
    """Render the consent page composing an optional provider login button
    (Google/Supabase) and/or the admin-token paste form, with a divider between
    them when both are present."""
    error_html = f'<div class="err">{html.escape(error)}</div>' if error else ""
    page = _CONSENT_PAGE.format(
        client_label=html.escape(client_label),
        error_html=error_html,
    )
    token_form = ""
    if show_token_form:
        token_form = _TOKEN_FORM.format(
            client_id=html.escape(client_id),
            redirect_uri=html.escape(redirect_uri),
            state=html.escape(state or ""),
            code_challenge=html.escape(code_challenge or ""),
            code_challenge_method=html.escape(code_challenge_method or ""),
            resource=html.escape(resource or ""),
        )
    divider = (
        '<div class="divider"><span>or paste an access token</span></div>'
        if provider_block and token_form
        else ""
    )
    page = (
        page.replace("<!--PROVIDER_BLOCK-->", provider_block)
        .replace("<!--DIVIDER-->", divider)
        .replace("<!--TOKEN_FORM-->", token_form)
    )
    return _html_response(page, status_code)


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

    # Compose the consent page: the configured provider contributes a login
    # button (Google / Supabase), and — unless explicitly disabled — the
    # admin-token paste form is shown alongside it. Everything validated above
    # (client, redirect_uri, PKCE) is preserved across any external IdP redirect
    # inside the provider's signed login-state token.
    ctx = AuthorizeContext(
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        resource=resource,
    )
    provider_block = get_login_provider().consent_block(ctx)
    # The token-paste box always shows in token mode; under a federated provider
    # it shows unless the deployer turned it off.
    show_token_form = settings.auth_provider == "token" or settings.auth_allow_token_paste
    return _render_consent(
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        resource=resource,
        provider_block=provider_block,
        show_token_form=show_token_form,
    )


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
    """Validate the pasted login token and issue an authorization code.

    Available under any AUTH_PROVIDER so admin-issued tokens remain a login
    option alongside Google/Supabase — unless the deployer disabled the paste
    box via AUTH_ALLOW_TOKEN_PASTE for a provider-only login screen."""
    if not oauth_enabled():
        return _not_configured()
    if settings.auth_provider != "token" and not settings.auth_allow_token_paste:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request",
                     "error_description": "token login is disabled on this deployment"},
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
    await refresh_revocations()
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
    # ``src`` ties every session minted here to the pasted credential: revoking
    # it ends them all. (Pasting an OAuth access token keeps its original src.)
    src = payload.get("src") or token_fingerprint(token.strip())
    return _issue_code_and_redirect(ctx, payload["user_id"], src=src)


def _issue_code_and_redirect(
    ctx: AuthorizeContext, user_id: str, src: str | None = None
) -> RedirectResponse:
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
            **({"src": src} if src else {}),
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
        "*,*::before,*::after{box-sizing:border-box;}"
        "body{overflow-wrap:anywhere;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;"
        "margin:0;min-height:100vh;display:grid;place-items:center;background:#0b0d12;color:#e7e9ee;}"
        ".card{width:min(440px,92vw);max-width:100%;min-width:0;background:#151821;border:1px solid #262b38;border-radius:16px;"
        "padding:28px;box-shadow:0 12px 40px rgba(0,0,0,.45);}"
        "h1{font-size:19px;margin:0 0 8px;}p{font-size:14px;color:#aab1c2;}"
        "</style></head><body><div class='card'><h1>Sign-in failed</h1>"
        f"<p>{safe}</p></div></body></html>"
    )
    return _html_response(page, err.status)


# ── self-service link (claim an existing account) ─────────────────────────

_LINK_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Link your Neuralscape account</title><style>
  :root {{ color-scheme: light dark; }}
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin:0; min-height:100vh; display:grid; place-items:center; background:#0b0d12; color:#e7e9ee;
         overflow-wrap:anywhere; }}
  .card {{ width:min(460px,92vw); max-width:100%; min-width:0; background:#151821; border:1px solid #262b38; border-radius:16px;
          padding:28px; box-shadow:0 12px 40px rgba(0,0,0,.45); }}
  h1 {{ font-size:19px; margin:0 0 4px; }}
  p {{ font-size:14px; line-height:1.5; color:#aab1c2; margin:8px 0; }}
  .who {{ font-size:13px; color:#8b93a7; }}
  textarea {{ width:100%; box-sizing:border-box; min-height:78px; resize:vertical;
             font-family:ui-monospace,Menlo,monospace; font-size:12px; background:#0d0f16;
             color:#e7e9ee; border:1px solid #2c3242; border-radius:10px; padding:10px; }}
  button {{ margin-top:12px; width:100%; padding:11px; font-size:14px; font-weight:600; border:0;
           border-radius:10px; cursor:pointer; }}
  .primary {{ background:#3b5bdb; color:#fff; }} .primary:hover {{ background:#364fc7; }}
  .secondary {{ background:#fff; color:#1f2330; border:1px solid #d6d9e0; }}
  .divider {{ display:flex; align-items:center; color:#7b8198; font-size:12px; margin:20px 0 6px; }}
  .divider::before, .divider::after {{ content:""; flex:1; border-top:1px solid #262b38; }}
  .divider span {{ padding:0 10px; }}
  .hint {{ font-size:12px; color:#7b8198; }} code {{ background:#0d0f16; padding:1px 5px; border-radius:5px;
         overflow-wrap:anywhere; word-break:break-word; }}
  .err {{ background:#2a1518; border:1px solid #5b2a2f; color:#ffb4b4; font-size:13px;
         border-radius:10px; padding:10px 12px; margin:14px 0 0; }}
</style></head><body><div class="card">
  <h1>Welcome to Neuralscape</h1>
  <p class="who">Signed in as <b>{email}</b>. First time here — how should we set up your memory?</p>
  {error_html}
  <form method="post" action="/oauth/link">
    <input type="hidden" name="ticket" value="{ticket}">
    <input type="hidden" name="action" value="new">
    <button class="primary" type="submit">Continue as a new account (<code>{suggested}</code>)</button>
  </form>
  <div class="divider"><span>or link an existing account</span></div>
  <form method="post" action="/oauth/link">
    <input type="hidden" name="ticket" value="{ticket}">
    <input type="hidden" name="action" value="link">
    <p class="hint">Already have Neuralscape memories under another id? Paste an access
       token for it (from <code>issue_user_token.py</code>) to link this Google account to it.</p>
    <textarea name="token" placeholder="paste an existing access token to link"></textarea>
    <button class="secondary" type="submit">Link to that account</button>
  </form>
</div></body></html>"""


def _render_link_page(ticket: str, email: str, suggested: str,
                      error: str | None = None, status_code: int = 200) -> HTMLResponse:
    error_html = f'<div class="err">{html.escape(error)}</div>' if error else ""
    page = _LINK_PAGE.format(
        email=html.escape(email), ticket=html.escape(ticket),
        suggested=html.escape(suggested), error_html=error_html,
    )
    return _html_response(page, status_code)


async def _finish_login(result) -> HTMLResponse | RedirectResponse:
    """Turn a provider login result into the right response: issue the code
    (linked), render the link page (first login), or show the error."""
    if isinstance(result, LoginError):
        return _login_error_page(result)
    if isinstance(result, LoginNeedsLink):
        ticket = sign_link_ticket(result, _secret())
        return _render_link_page(ticket, result.email, result.suggested_user_id)
    return _issue_code_and_redirect(
        result.ctx,
        result.user_id,
        src=identity_fingerprint(settings.auth_provider, result.user_id),
    )


# ── federated login callbacks (google / supabase) ────────────────────────


@router.get("/oauth/google/callback", response_model=None)
async def google_callback(request: Request) -> RedirectResponse | HTMLResponse | JSONResponse:
    """Google OIDC redirect target: verify the id_token, allowlist-check the
    email, then continue the MCP flow (or show the first-login link page)."""
    if not oauth_enabled():
        return _not_configured()
    if settings.auth_provider != "google":
        return JSONResponse(status_code=404, content={"detail": "google login not enabled"})
    return await _finish_login(await GoogleProvider().complete(request))


@router.get("/oauth/supabase/callback", response_model=None)
async def supabase_callback_get(request: Request) -> HTMLResponse | JSONResponse:
    """Supabase redirect target (browser): render the page that exchanges the
    PKCE code for a session and POSTs the access token back to the server."""
    if not oauth_enabled():
        return _not_configured()
    if settings.auth_provider != "supabase":
        return JSONResponse(status_code=404, content={"detail": "supabase login not enabled"})
    return _html_response(_supabase_finish_page())


@router.post("/oauth/supabase/callback", response_model=None)
async def supabase_callback_post(request: Request) -> RedirectResponse | HTMLResponse | JSONResponse:
    """Verify the posted Supabase session JWT, then continue the MCP flow."""
    if not oauth_enabled():
        return _not_configured()
    if settings.auth_provider != "supabase":
        return JSONResponse(status_code=404, content={"detail": "supabase login not enabled"})
    return await _finish_login(await SupabaseProvider().complete(request))


@router.post("/oauth/link", response_model=None)
async def oauth_link(
    request: Request,
    ticket: str = Form(...),
    action: str = Form("new"),
    token: str = Form(""),
) -> RedirectResponse | HTMLResponse | JSONResponse:
    """Finalize a first-login link decision, then issue the auth code.

    The signed ticket binds the verified Google/Supabase ``sub``+``email`` and
    the original authorize context. ``action=link`` claims an existing id by
    proving ownership with that id's admin token; ``action=new`` provisions the
    suggested slug. Either way the (sub, email) → user_id mapping is persisted
    so subsequent logins skip this page."""
    if not oauth_enabled():
        return _not_configured()
    claims = verify_link_ticket(ticket, _secret())
    if claims is None:
        return _login_error_page(LoginError("This link request expired. Please sign in again.", 400))

    ctx = claims["ctx"]
    sub = claims.get("sub") or None
    email = claims.get("email") or None

    if action == "link":
        payload = verify_user_token(token.strip(), _secret()) if token.strip() else None
        if payload is None:
            return _render_link_page(
                ticket, email or "", claims.get("suggested_user_id", ""),
                error="That token is invalid or expired. Paste a valid access token, or continue as a new account.",
                status_code=400,
            )
        user_id = payload["user_id"]
    else:
        user_id = claims.get("suggested_user_id") or ""
        if not user_id:
            return _login_error_page(LoginError("Could not determine an account id.", 400))

    await identity_store.link(user_id, sub=sub, email=email)
    return _issue_code_and_redirect(ctx, user_id)


# ── token endpoint ───────────────────────────────────────────────────────


def _token_response(
    user_id: str, client_id: str, family: str | None = None, src: str | None = None
) -> JSONResponse:
    src_claim = {"src": src} if src else {}
    access = issue_user_token(
        user_id, _secret(), settings.oauth_access_ttl, typ="access", extra_claims=src_claim
    )
    # The refresh token is single-use (rotate-and-consume via its jti) and
    # bound to the client it was issued to (RFC 6749 §6).
    refresh = sign_payload(
        {
            "typ": "refresh",
            "user_id": user_id,
            "client_id": client_id,
            # ``fam`` ties every rotation of one grant together so reuse of any
            # old token can revoke the whole chain.
            "fam": family or secrets.token_urlsafe(12),
            "jti": secrets.token_urlsafe(16),
            **src_claim,
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


def _guard_unavailable() -> JSONResponse:
    return _token_error(
        "temporarily_unavailable",
        "single-use check is unavailable; retry shortly",
        status_code=503,
    )


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

    await refresh_revocations()

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
        jti = claims.get("jti")
        if not isinstance(jti, str) or not jti:
            return _token_error("invalid_grant", "code is invalid or expired")
        user_id = claims.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            return _token_error("invalid_grant", "code missing user")
        if _src_revoked(claims.get("src")):
            return _token_error("invalid_grant", "code is invalid or expired")
        try:
            first_use = await _consume_code_jti(jti)
        except _ReplayGuardUnavailable:
            return _guard_unavailable()
        if not first_use:
            return _token_error("invalid_grant", "authorization code already used")
        return _token_response(user_id, client_id, src=claims.get("src"))

    if grant_type == "refresh_token":
        claims = verify_payload(refresh_token, _secret())
        if claims is None or claims.get("typ") != "refresh":
            return _token_error("invalid_grant", "refresh_token is invalid or expired")
        user_id = claims.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            return _token_error("invalid_grant", "refresh_token missing user")
        # Refresh tokens minted before rotation/families shipped carry no
        # jti/client_id/fam. They cannot be made single-use or revoked, so they
        # are refused and the user reconnects once.
        jti = claims.get("jti")
        fam = claims.get("fam")
        bound_client = claims.get("client_id")
        if not all(isinstance(v, str) and v for v in (jti, fam, bound_client)):
            return _token_error("invalid_grant", "refresh_token is invalid or expired")
        # Checked before consuming so a wrong client cannot burn the real
        # client's token. Compare bytes: compare_digest rejects non-ASCII str.
        if not client_id or not secrets.compare_digest(
            bound_client.encode("utf-8"), client_id.encode("utf-8")
        ):
            return _token_error("invalid_grant", "client mismatch")
        remaining = int(claims.get("exp", 0)) - int(time.time())
        # Revoked directly (its own digest) or via the credential/identity it
        # descends from. Direct revocation also kills the whole family.
        if _digest_revoked(refresh_token):
            try:
                await _revoke_family(fam, max(1, remaining))
            except _ReplayGuardUnavailable:
                return _guard_unavailable()
            return _token_error("invalid_grant", "refresh_token is invalid or expired")
        if _src_revoked(claims.get("src")):
            return _token_error("invalid_grant", "refresh_token is invalid or expired")
        try:
            if await _family_revoked(fam):
                return _token_error("invalid_grant", "refresh_token is invalid or expired")
            first_use = await _consume_refresh_jti(jti, remaining)
        except _ReplayGuardUnavailable:
            return _guard_unavailable()
        if not first_use:
            # Strict by design, no grace window (OAuth 2.0 Security BCP reuse
            # detection): a legitimate client retrying after a lost response
            # lands here too and must reconnect.
            # A rotated token was replayed: someone holds a copy. Revoke the
            # whole family so the successor (possibly held by the thief) dies
            # too; the legitimate client reconnects.
            logger.warning("OAuth refresh token reuse: revoking family (user=%s)", user_id)
            try:
                await _revoke_family(fam, settings.oauth_refresh_ttl)
            except _ReplayGuardUnavailable:
                return _guard_unavailable()
            return _token_error("invalid_grant", "refresh_token already used")
        return _token_response(user_id, client_id, family=fam, src=claims.get("src"))

    return _token_error("unsupported_grant_type", f"grant_type '{grant_type}' not supported")
