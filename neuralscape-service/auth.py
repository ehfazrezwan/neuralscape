"""Bearer token authentication for Neuralscape API.

Supports two token formats simultaneously:

1. **Per-user HMAC tokens** (multi-user model): two segments
   `base64url({user_id, exp})` + `.` + `hmac_sha256(secret, payload)`.
   When `NEURALSCAPE_USER_TOKEN_SECRET` is set, tokens of this shape are
   accepted; the verified user_id is attached to ``request.state.user_id``
   so routes pull identity from the token, not the request body.

2. **Legacy shared API key** (single-user / pre-multi-user installs):
   one segment, opaque string. Validated against ``NEURALSCAPE_API_KEY``.
   When this path matches, ``request.state.user_id`` is **not** set
   (routes must fall back to body ``user_id``), and a deprecation
   warning header is added to the response.

When neither secret is configured, auth is disabled (local dev).

Public paths (`/health`) always bypass auth.
"""

import hmac
import logging
from contextvars import ContextVar

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings
from tokens import issue_user_token, verify_user_token  # noqa: F401 (re-export)

logger = logging.getLogger(__name__)

# Identity of the caller for the current request, set after a per-user token
# is verified. Lets the MCP tool layer — which only receives a tool-arguments
# dict, not the HTTP request — scope operations to the authenticated user
# without the model having to pass `user_id`. Default None = unauthenticated
# or legacy shared-key (fall back to the request body / arguments).
current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)

# Paths that never require authentication
# The refresh endpoint authenticates by the single-use refresh token in its body
# (an expired access token cannot be required to obtain a new one); the handler
# is a 404 unless CREDENTIALS_ENABLED.  The console pages sign users in themselves.
PUBLIC_PATHS = {"/health", "/health/live", "/api/v1/health", "/console", "/v1/credentials/refresh"}
# Path prefixes that never require auth: OAuth discovery metadata and the
# Authorization Server endpoints themselves (the consent page, DCR, token
# exchange). These are the public front door of the OAuth flow. ``/console/``
# pages carry their own session login and call the API with the signed-in
# credential; the routes exist only when CONSOLE_ENABLED is set.
PUBLIC_PREFIXES = ("/.well-known/", "/oauth/", "/console/")


def _public_base_url() -> str:
    """Public base URL (no trailing slash) for building metadata URLs."""
    return settings.neuralscape_public_url.rstrip("/")

# Header set on responses when a caller used a legacy shared API key, to
# nudge them toward per-user tokens. Non-fatal.
DEPRECATION_HEADER = "X-Neuralscape-Deprecation"
DEPRECATION_MSG = "shared-api-key auth is deprecated; issue per-user tokens via scripts/issue_user_token.py"


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Bearer token middleware supporting per-user HMAC + legacy shared keys."""

    async def dispatch(self, request: Request, call_next):
        # Auth disabled when neither secret is set (local dev convenience)
        token_secret = settings.neuralscape_user_token_secret
        legacy_key = settings.neuralscape_api_key
        # Scoped credentials (off unless CREDENTIALS_ENABLED and policy
        # enforcement is on) make authentication mandatory.
        import credentials

        scoped_enabled = credentials.accepts_scoped_credentials()
        if not token_secret and not legacy_key and not scoped_enabled:
            return await call_next(request)

        # Health checks and the OAuth front door are always public
        path = request.url.path
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

        # Fall back to the legacy shared API key. user_id is NOT set on
        # request.state — routes must read it from the request body
        # (trust-based, pre-multi-user behavior).
        # Compared as bytes: ``compare_digest`` refuses str with non-ASCII characters (a TypeError, so a
        # server error), and a header may carry any Latin-1 character.
        if legacy_key and hmac.compare_digest(
            token.encode("utf-8", "surrogateescape"), legacy_key.encode("utf-8", "surrogateescape")
        ):
            request.state.auth_mode = "legacy_shared"
            response = await call_next(request)
            response.headers[DEPRECATION_HEADER] = DEPRECATION_MSG
            return response

        return self._unauthorized(request, "Invalid or expired token")

    async def _scoped_credential(self, request: Request, call_next, token: str):
        import credentials

        service = credentials.get_service()
        if service is None:
            return JSONResponse(status_code=503, content={"detail": "credential service unavailable"})
        try:
            verified = service.verify_access(token)
        except credentials.AuthorityUnavailable:
            return JSONResponse(status_code=503, content={"detail": "credential service unavailable"})
        if verified is None:
            return self._unauthorized(request, "Invalid or expired token")
        # The opaque subject id is the identity downstream code sees as user_id.
        request.state.user_id = verified.subject_id
        request.state.auth_mode = "scoped_credential"
        request.state.credential = verified
        user_token = current_user_id.set(verified.subject_id)
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
