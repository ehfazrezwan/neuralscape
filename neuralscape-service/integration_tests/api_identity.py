"""Per-run API identity for tests that call a live API (see README.md).

Shared by ``conftest.py`` (the ``api_user`` / ``live_api`` fixtures) and by
``tests/test_async_pipeline.py``, which lives outside this directory and so
cannot use fixtures.  In-repo tests, not black-box ones: the token is minted
with the service's own ``tokens.issue_user_token`` (two-segment HMAC format,
see AUTH.md).

Resolution order:

1. ``NS_API_TOKEN`` (override).  The user id is ``NS_API_USER`` when set,
   otherwise read from the token's payload (unverified: the server verifies).
2. ``NEURALSCAPE_USER_TOKEN_SECRET`` (the stack's own secret, passed by the
   gate): mint a short-lived token for a fresh unique user id.
3. Nothing configured: ``None`` (an API running with auth disabled).
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

TOKEN_TTL_SECONDS = 3600


@dataclass(frozen=True)
class ApiIdentity:
    user_id: str
    token: str

    def __repr__(self) -> str:  # never show the bearer
        return f"ApiIdentity(user_id={self.user_id!r})"

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}


def _user_from_token(token: str) -> str | None:
    import tokens

    try:
        payload = json.loads(tokens._b64url_decode(token.split(".", 1)[0]))
        user_id = payload.get("user_id")
        return user_id if isinstance(user_id, str) and user_id else None
    except Exception:  # noqa: BLE001 - an unreadable override is reported by the caller
        return None


def resolve_identity(env: Mapping[str, str] | None = None) -> ApiIdentity | None:
    env = os.environ if env is None else env
    token = (env.get("NS_API_TOKEN") or "").strip()
    if token:
        user_id = (env.get("NS_API_USER") or "").strip() or _user_from_token(token)
        if not user_id:
            raise ValueError("NS_API_TOKEN is set but its user id cannot be read; set NS_API_USER")
        return ApiIdentity(user_id=user_id, token=token)
    secret = (env.get("NEURALSCAPE_USER_TOKEN_SECRET") or "").strip()
    if secret:
        import tokens

        user_id = f"nsit-api-{uuid.uuid4().hex[:12]}"
        return ApiIdentity(user_id=user_id, token=tokens.issue_user_token(user_id, secret, TOKEN_TTL_SECONDS))
    return None
