"""The CBM flag-on test must fail, not skip, under NS_INTEGRATION_REQUIRED."""

from integration_tests.cbm_gate import FAIL, RUN, SKIP, cbm_decision

_DOWN = lambda url: False  # noqa: E731
_UP = lambda url: True  # noqa: E731


def test_required_without_bridge_fails_naming_the_url_variable():
    action, reason = cbm_decision({"NS_INTEGRATION_REQUIRED": "1"}, _DOWN)
    assert action == FAIL
    assert "CBM_BRIDGE_URL" in reason


def test_github_actions_counts_as_required():
    assert cbm_decision({"GITHUB_ACTIONS": "true"}, _DOWN)[0] == FAIL


def test_not_required_without_bridge_skips():
    action, reason = cbm_decision({}, _DOWN)
    assert action == SKIP
    assert "CBM_BRIDGE_URL" in reason


def test_reachable_bridge_runs_even_when_required():
    assert cbm_decision({"NS_INTEGRATION_REQUIRED": "1", "CBM_BRIDGE_URL": "http://b"}, _UP) == (RUN, "")


def test_probe_receives_the_configured_url():
    seen = []
    cbm_decision({"CBM_BRIDGE_URL": "http://bridge:9"}, lambda u: seen.append(u) or True)
    assert seen == ["http://bridge:9"]


def test_waiver_turns_required_failure_into_skip_for_cbm_only():
    env = {"NS_INTEGRATION_REQUIRED": "1", "NS_INTEGRATION_WAIVE": "other, CBM"}
    assert cbm_decision(env, _DOWN)[0] == SKIP
    assert cbm_decision({"NS_INTEGRATION_REQUIRED": "1", "NS_INTEGRATION_WAIVE": "other"}, _DOWN)[0] == FAIL


def test_waiver_does_not_hide_a_reachable_bridge():
    env = {"NS_INTEGRATION_REQUIRED": "1", "NS_INTEGRATION_WAIVE": "cbm"}
    assert cbm_decision(env, _UP)[0] == RUN


def test_real_probe_without_url_is_unreachable():
    from integration_tests.cbm_gate import probe_bridge

    assert probe_bridge(None) is False
    assert probe_bridge("http://127.0.0.1:1") is False
