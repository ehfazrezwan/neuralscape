import pytest

from nsaccept import runctx


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("NS_TOKEN_SECRET", "NS_TOKEN_A", "NS_TOKEN_B", "NS_USER_A", "NS_USER_B",
              "NS_COMPOSE_PROJECT", "NS_RUN_ID"):
        monkeypatch.delenv(k, raising=False)


def test_no_token_source_is_config_error():
    with pytest.raises(runctx.ConfigError):
        runctx.build("http://127.0.0.1:1", "default")


def test_secret_source_makes_unique_users(monkeypatch):
    monkeypatch.setenv("NS_TOKEN_SECRET", "s")
    c = runctx.build("http://127.0.0.1:1/", "default")
    assert c.base_url == "http://127.0.0.1:1"
    assert c.user_a == f"acc-{c.run_id}-a" and c.user_b == f"acc-{c.run_id}-b"
    assert c.project_id == f"acc-{c.run_id}-proj"
    assert c.token_a != c.token_b and c.bad_token != c.token_a


def test_supplied_tokens_need_two_distinct_users(monkeypatch):
    monkeypatch.setenv("NS_TOKEN_A", "ta")
    monkeypatch.setenv("NS_TOKEN_B", "tb")
    with pytest.raises(runctx.ConfigError):
        runctx.build("http://x", "default")
    monkeypatch.setenv("NS_USER_A", "u")
    monkeypatch.setenv("NS_USER_B", "u")
    with pytest.raises(runctx.ConfigError):
        runctx.build("http://x", "default")
    monkeypatch.setenv("NS_USER_B", "v")
    assert runctx.build("http://x", "default").token_source == "NS_TOKEN_A/B"


def test_compose_project_takes_precedence(monkeypatch):
    monkeypatch.setenv("NS_TOKEN_SECRET", "s")
    calls = []

    def fake(project, user, service="neuralscape", days=1, run=None):
        calls.append((project, user, service))
        return f"tok-{user}"

    monkeypatch.setattr(runctx.tokens, "mint_via_compose", fake)
    monkeypatch.setenv("NS_API_SERVICE", "api")
    c = runctx.build("http://x", "default", compose_project="p1")
    assert c.token_a == f"tok-{c.user_a}" and c.token_source == "compose:p1"
    assert calls[0][0] == "p1" and calls[0][2] == "api"
