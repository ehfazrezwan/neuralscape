"""Conftest for neuralscape-service tests — sys.path + auth isolation."""

import sys
from pathlib import Path

import pytest

# Ensure the service root is importable (main.py, config.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _disable_auth_for_tests():
    """Disable BearerAuthMiddleware for the duration of each test.

    Without this, the test process inherits the host's env (e.g. a
    ``NEURALSCAPE_USER_TOKEN_SECRET`` set for a running deployment),
    which makes every unauthenticated TestClient request return 401
    before reaching the route under test.

    Tests that specifically exercise auth (test_auth.py, test_tokens.py)
    construct their own credentials and don't rely on the middleware
    being on.
    """
    from config import settings
    saved_token_secret = settings.neuralscape_user_token_secret
    saved_api_key = settings.neuralscape_api_key
    saved_default_user_id = settings.default_user_id
    settings.neuralscape_user_token_secret = ""
    settings.neuralscape_api_key = ""
    # Pin to the Pydantic class default so route fallbacks are deterministic
    # across deployment envs that set DEFAULT_USER_ID for production use.
    settings.default_user_id = "default_user"
    yield
    settings.neuralscape_user_token_secret = saved_token_secret
    settings.neuralscape_api_key = saved_api_key
    settings.default_user_id = saved_default_user_id
