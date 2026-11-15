"""Bearer token authentication for Neuralscape API.

Supports two token formats simultaneously:

1. **Per-user HMAC tokens** (multi-user model): two segments
   `base64url({user_id, exp})` + `.` + `hmac_sha256(secret, payload)`.
   When `NEURALSCAPE_USER_TOKEN_SECRET` is set, tokens of this shape are
   accepted; the verified user_id is attached to ``request.state.user_id``
   so routes pull identity from the token, not the request body.

2. **Legacy shared API key** (single-user / pre-multi-user installs):
   one segment, opaque string. Validated against ``NEURALSCAPE_API_KEY``.
   When this path matches, the key acts as ONE identity:
   ``NEURALSCAPE_API_KEY_USER_ID`` if set, else ``DEFAULT_USER_ID``. That user
   is attached to ``request.state.user_id`` and ``current_user_id`` just like a
   per-user token, so a body/query ``user_id`` naming anyone else gets HTTP 400
   and MCP ignores ``arguments["user_id"]``. Setting
   ``NEURALSCAPE_LEGACY_KEY_TRUST_BODY_USER_ID=true`` restores the old
   behaviour (``request.state.user_id`` not set; routes trust the body). A
   deprecation warning header is added to every legacy-key response.

Auth is fail-closed. When neither secret is configured, startup refuses
(``Settings.validate_required``) and this middleware rejects every non-public
request with 503, unless the operator explicitly set
``NEURALSCAPE_AUTH_DISABLED=true`` (trusted local machine only), which is the
one path that lets requests through unauthenticated.

Public paths (``/health``, ``/health/live``, ``/.well-known/*``, ``/oauth/*``)
always bypass auth.
"""

import asyncio
import hmac
import logging
from contextvars import ContextVar

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings
from tokens import issue_user_token, verify_user_token  # noqa: F401 (re-export)

logger = logging.getLogger(__name__)

# Identity of the caller for the current request, set after a per-user token
# is verified. Lets the MCP tool layer — which only receives a tool-arguments
# dict, not the HTTP request — scope operations to the authenticated user
# without the model having to pass `user_id`. Default None = unauthenticated
# (or the legacy-key opt-out): fall back to the request body / arguments.
current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)
# True while a request authenticated with the legacy shared key is being served, so the MCP layer (which
# only sees ``current_user_id``) can tell that identity from a per-user token's.
current_legacy_shared: ContextVar[bool] = ContextVar("current_legacy_shared", default=False)

# Paths that never require authentication
# The refresh endpoint authenticates by the single-use refresh token in its body
# (an expired access token cannot be required to obtain a new one); the handler
# is a 404 (not 401) unless CREDENTIALS_ENABLED and NS_POLICY_MODE=enforce.
PUBLIC_PATHS = {"/health", "/health/live", "/v1/credentials/refresh"}
# Path prefixes that never require auth: OAuth discovery metadata and the
# Authorization Server endpoints themselves (the consent page, DCR, token
# exchange). These are the public front door of the OAuth flow.
PUBLIC_PREFIXES = ("/.well-known/", "/oauth/")


def _public_base_url() -> str:
    """Public base URL (no trailing slash) for building metadata URLs."""
    return settings.neuralscape_public_url.rstrip("/")

# Header set on responses when a caller used a legacy shared API key, to
# nudge them toward per-user tokens. Non-fatal.
DEPRECATION_HEADER = "X-Neuralscape-Deprecation"
DEPRECATION_MSG = "shared-api-key auth is deprecated; issue per-user tokens via scripts/issue_user_token.py"

AUTH_NOT_CONFIGURED_MSG = (
    "Authentication is not configured on this server: set NEURALSCAPE_USER_TOKEN_SECRET "
    "or NEURALSCAPE_API_KEY (or NEURALSCAPE_AUTH_DISABLED=true on a trusted local machine)"
)


def resolve_request_user_id(request: Request, body_user_id: str | None) -> str:
    """Resolve the authoritative user_id for a request.

    - With a verified per-user token (``request.state.user_id`` set by the
      middleware) the token user wins; a body ``user_id`` that disagrees is
      rejected with 400 (token-vs-body identity confusion).
    - Legacy shared-key callers have no ``request.state.user_id``; the body
      ``user_id`` is trusted as before.
    - With neither, fall back to ``settings.default_user_id``.

    Lives here (not in main.py) so extension routers can use it without a
    circular import.
    """
    token_user_id = getattr(request.state, "user_id", None)
    if token_user_id:
        if body_user_id and body_user_id != token_user_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Request body user_id ({body_user_id!r}) does not match the "
                    f"user_id encoded in the auth token ({token_user_id!r})."
                ),
            )
        return token_user_id
    return body_user_id or settings.default_user_id


def is_admin_user(user_id: str | None) -> bool:
    """Admin gate for dreaming runs and project knowledge-config writes.

    Owner decision 2026-11-15: with ``DICTATOR_USER_IDS`` empty the default
    user (``settings.default_user_id``) is the admin, so single-user and
    legacy-key installs keep working; with a non-empty list only the listed
    users are admins. ``settings.is_dictator`` is deliberately unchanged
    (standards-tier writes keep their meaning). Callers must pass the
    token-resolved identity, never a body-supplied one.
    """
    if not user_id:
        return False
    listed = settings.dictator_user_ids_set()
    if listed:
        return user_id in listed
    return user_id == settings.default_user_id


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Bearer token middleware supporting per-user HMAC + legacy shared keys."""

    async def dispatch(self, request: Request, call_next):
        token_secret = settings.neuralscape_user_token_secret
        legacy_key = settings.neuralscape_api_key
        path = request.url.path
        # Scoped credentials (off unless CREDENTIALS_ENABLED and policy
        # enforcement is on) make authentication mandatory.
        import credentials

        scoped_enabled = credentials.accepts_scoped_credentials()
        if not token_secret and not legacy_key and not scoped_enabled:
            # The ONLY unauthenticated pass-through: an explicit operator
            # opt-out (startup logs a WARNING for it). Never a silent default.
            if settings.neuralscape_auth_disabled:
                return await call_next(request)
            # No credential configured and no opt-out: fail closed. Startup
            # validation normally refuses this state; this guards any path
            # that serves the app without running the lifespan.
            if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
                return await call_next(request)
            return JSONResponse(
                status_code=503,
                content={"detail": AUTH_NOT_CONFIGURED_MSG},
            )

        # Health checks and the OAuth front door are always public
        if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return self._unauthorized(request, "Missing or invalid Authorization header")
        token = auth_header[7:]  # strip "Bearer "

        # New scoped credential format, alongside the existing tokens.  A token
        # in this format that does not verify is rejected outright; it never
        # falls through to the shared-key comparison.
        if scoped_enabled and token.startswith(credentials.ACCESS_PREFIX):
            return await self._scoped_credential(request, call_next, token)

        # Try the per-user HMAC token path first (preferred). The token
        # carries the user_id we attach to request.state for downstream
        # routes to use authoritatively. If verification fails (bad
        # signature, expired, malformed) we fall through to the legacy
        # shared-key check rather than returning 401 — that protects
        # legacy shared keys that happen to contain a `.` character
        # (the only signal we use to detect token shape).
        if token_secret and "." in token:
            payload = verify_user_token(token, token_secret)
            if payload is not None:
                request.state.user_id = payload["user_id"]
                request.state.auth_mode = "user_token"
                # Expose identity to the MCP tool layer for this request.
                ctx_token = current_user_id.set(payload["user_id"])
                try:
                    return await call_next(request)
                finally:
                    current_user_id.reset(ctx_token)
            # else: HMAC verify failed — fall through to legacy check.

        # Fall back to the legacy shared API key. By default the key acts as
        # ONE identity (NEURALSCAPE_API_KEY_USER_ID, else DEFAULT_USER_ID): that
        # user is bound to request.state and the MCP ContextVar exactly like a
        # per-user token, so a body/query user_id naming anyone else is a 400.
        # NEURALSCAPE_LEGACY_KEY_TRUST_BODY_USER_ID=true restores the old
        # trust-the-body behaviour (no identity bound).
        # Compare as bytes: compare_digest on str raises TypeError for non-ASCII.
        if legacy_key and hmac.compare_digest(
            token.encode("utf-8"), legacy_key.encode("utf-8")
        ):
            request.state.auth_mode = "legacy_shared"
            shared_token = current_legacy_shared.set(True)
            try:
                if settings.neuralscape_legacy_key_trust_body_user_id:
                    response = await call_next(request)
                else:
                    bound_user_id = settings.legacy_key_identity()
                    request.state.user_id = bound_user_id
                    ctx_token = current_user_id.set(bound_user_id)
                    try:
                        response = await call_next(request)
                    finally:
                        current_user_id.reset(ctx_token)
            finally:
                current_legacy_shared.reset(shared_token)
            response.headers[DEPRECATION_HEADER] = DEPRECATION_MSG
            return response

        return self._unauthorized(request, "Invalid or expired token")

    async def _scoped_credential(self, request: Request, call_next, token: str):
        import credentials

        service = credentials.get_service()
        if service is None:
            return JSONResponse(status_code=503, content={"detail": "credential service unavailable"})
        try:
            # Synchronous SQLite/psycopg I/O: off the event loop.
            verified = await asyncio.to_thread(service.verify_access, token)
        except credentials.AuthorityUnavailable:
            return JSONResponse(status_code=503, content={"detail": "credential service unavailable"})
        if verified is None:
            return self._unauthorized(request, "Invalid or expired token")
        # Downstream code (memory reads and writes, task ownership) sees the ``user_id`` the subject's
        # rows are keyed by: the bound legacy id, or the subject id itself when there is none.  The
        # policy layer still identifies the principal by ``request.state.credential``.
        try:
            memory_user = await asyncio.to_thread(service.memory_user_id, verified.subject_id)
        except credentials.AuthorityUnavailable:
            return JSONResponse(status_code=503, content={"detail": "credential service unavailable"})
        request.state.user_id = memory_user
        request.state.auth_mode = "scoped_credential"
        request.state.credential = verified
        user_token = current_user_id.set(memory_user)
        credential_token = credentials.current_credential.set(verified)
        try:
            return await call_next(request)
        finally:
            credentials.current_credential.reset(credential_token)
            current_user_id.reset(user_token)

    def _unauthorized(self, request: Request, detail: str) -> JSONResponse:
        """401 response, carrying an RFC 9728 ``WWW-Authenticate`` header.

        When a public URL is configured, the header points MCP clients
        (Claude Cowork / claude.ai) at our Protected Resource Metadata so they
        can discover the Authorization Server and start the OAuth flow. Without
        this header, the connector UI never offers a "Connect" / login step.
        """
        headers = {}
        base = _public_base_url()
        # Only advertise OAuth discovery when the AS is actually enabled — i.e.
        # both the public URL AND the signing secret are set. The /.well-known
        # endpoints 404 without the secret, so pointing a client there would
        # send it chasing discovery against dead endpoints.
        if base and settings.neuralscape_user_token_secret:
            resource_metadata = f'{base}/.well-known/oauth-protected-resource'
            headers["WWW-Authenticate"] = (
                f'Bearer resource_metadata="{resource_metadata}"'
            )
        return JSONResponse(
            status_code=401,
            content={"detail": detail},
            headers=headers,
        )
