"""An integration test that drives the real app with its credentials cleared must opt out of auth explicitly.

Root cause of the gate's ``assert (503 == 200)`` in ``test_patch_revision_real_stores``: auth is fail-closed
(``auth.py`` BearerAuthMiddleware answers every non-public request with 503 when no credential is configured and
``NEURALSCAPE_AUTH_DISABLED`` is not set). ``tests/conftest.py`` sets that opt-out for the unit suite, but
``integration_tests/`` has no such fixture, so a test that clears both secrets and then calls the app gets 503.
"""

import re
from pathlib import Path

import pytest

INTEGRATION_DIR = Path(__file__).resolve().parents[1] / "integration_tests"
CLEARS_BOTH_SECRETS = re.compile(
    r'setattr\(\s*settings\s*,\s*"neuralscape_api_key"\s*,\s*""\s*\).*?'
    r'setattr\(\s*settings\s*,\s*"neuralscape_user_token_secret"\s*,\s*""\s*\)',
    re.S,
)


def test_a_clean_app_with_no_credentials_and_no_optout_answers_503():
    """The behaviour the integration test tripped over: documented and pinned (fail-closed auth)."""
    from fastapi.testclient import TestClient

    import main
    from config import settings

    saved = (settings.neuralscape_api_key, settings.neuralscape_user_token_secret, settings.neuralscape_auth_disabled)
    settings.neuralscape_api_key = ""
    settings.neuralscape_user_token_secret = ""
    settings.neuralscape_auth_disabled = False
    try:
        assert TestClient(main.app, raise_server_exceptions=False).get("/v1/memories/x").status_code == 503
        settings.neuralscape_auth_disabled = True
        assert TestClient(main.app, raise_server_exceptions=False).get("/v1/memories/x").status_code != 503
    finally:
        settings.neuralscape_api_key, settings.neuralscape_user_token_secret, settings.neuralscape_auth_disabled = saved


@pytest.mark.parametrize("path", sorted(INTEGRATION_DIR.glob("test_*.py")), ids=lambda p: p.name)
def test_integration_tests_that_clear_credentials_and_call_the_app_opt_out_explicitly(path):
    source = path.read_text(encoding="utf-8")
    if "TestClient(" in source and CLEARS_BOTH_SECRETS.search(source):
        assert "neuralscape_auth_disabled" in source, (
            f"{path.name} clears both credentials and drives the app but never sets neuralscape_auth_disabled: "
            "the fail-closed auth middleware will answer 503 (see this module's docstring)"
        )
