"""The CBM flag-on test must fail, not skip, under NS_INTEGRATION_REQUIRED."""

import pytest

from integration_tests.cbm_gate import FAIL, RUN, SKIP, cbm_decision, probe_bridge

_DOWN = lambda url: False  # noqa: E731
_UP = lambda url: True  # noqa: E731


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("NS_INTEGRATION_REQUIRED", "GITHUB_ACTIONS", "NS_INTEGRATION_WAIVE", "CBM_BRIDGE_URL"):
        monkeypatch.delenv(name, raising=False)


def test_required_without_bridge_fails_naming_the_url_variable(monkeypatch):
    monkeypatch.setenv("NS_INTEGRATION_REQUIRED", "1")
    action, reason = cbm_decision(_DOWN)
    assert action == FAIL
    assert "CBM_BRIDGE_URL" in reason


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_required_truthy_values_match_conftest(monkeypatch, value):
    from integration_tests.conftest import integration_required

    monkeypatch.setenv("NS_INTEGRATION_REQUIRED", value)
    assert integration_required() is True
    assert cbm_decision(_DOWN)[0] == FAIL


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_required_falsy_values_match_conftest(monkeypatch, value):
    from integration_tests.conftest import integration_required

    monkeypatch.setenv("NS_INTEGRATION_REQUIRED", value)
    assert integration_required() is False
    assert cbm_decision(_DOWN)[0] == SKIP


def test_github_actions_counts_as_required(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert cbm_decision(_DOWN)[0] == FAIL


def test_not_required_without_bridge_skips():
    action, reason = cbm_decision(_DOWN)
    assert action == SKIP
    assert "CBM_BRIDGE_URL" in reason


def test_reachable_bridge_runs_even_when_required(monkeypatch):
    monkeypatch.setenv("NS_INTEGRATION_REQUIRED", "1")
    monkeypatch.setenv("CBM_BRIDGE_URL", "http://b")
    assert cbm_decision(_UP) == (RUN, "")


def test_probe_receives_the_configured_url(monkeypatch):
    monkeypatch.setenv("CBM_BRIDGE_URL", "http://bridge:9")
    seen = []
    cbm_decision(lambda u: seen.append(u) or True)
    assert seen == ["http://bridge:9"]


def test_waiver_turns_required_failure_into_skip(monkeypatch):
    monkeypatch.setenv("NS_INTEGRATION_REQUIRED", "1")
    monkeypatch.setenv("NS_INTEGRATION_WAIVE", "other, CBM")
    assert cbm_decision(_DOWN)[0] == SKIP


@pytest.mark.parametrize("waive", ["other", "cbmx", "xcbm", "cbm-extra", "my_cbm"])
def test_waiver_matches_exact_tokens_not_substrings(monkeypatch, waive):
    monkeypatch.setenv("NS_INTEGRATION_REQUIRED", "1")
    monkeypatch.setenv("NS_INTEGRATION_WAIVE", waive)
    assert cbm_decision(_DOWN)[0] == FAIL


def test_waiver_does_not_hide_a_reachable_bridge(monkeypatch):
    monkeypatch.setenv("NS_INTEGRATION_REQUIRED", "1")
    monkeypatch.setenv("NS_INTEGRATION_WAIVE", "cbm")
    assert cbm_decision(_UP)[0] == RUN


def test_configured_but_unreachable_bridge_fails_despite_waiver(monkeypatch):
    monkeypatch.setenv("NS_INTEGRATION_REQUIRED", "1")
    monkeypatch.setenv("NS_INTEGRATION_WAIVE", "cbm")
    monkeypatch.setenv("CBM_BRIDGE_URL", "http://bridge:9")
    action, reason = cbm_decision(_DOWN)
    assert action == FAIL
    assert "CBM_BRIDGE_URL" in reason


def test_configured_but_unreachable_bridge_skips_when_not_required(monkeypatch):
    monkeypatch.setenv("CBM_BRIDGE_URL", "http://bridge:9")
    assert cbm_decision(_DOWN)[0] == SKIP


@pytest.mark.parametrize("url", [None, "", "http://127.0.0.1:1", "file:///etc/passwd", "ftp://x/y", "gopher://x"])
def test_real_probe_rejects_missing_unreachable_and_non_http(url):
    assert probe_bridge(url) is False
