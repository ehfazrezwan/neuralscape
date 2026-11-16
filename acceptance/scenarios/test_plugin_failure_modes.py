"""The never-block contract and the fail-loud counter, against a service that is not there.

No hook may ever stop Claude: an unreachable service gets a notice and exit 0; only malformed hook
input exits 2, and only on the hooks where exit 2 cannot block anything. Consecutive unreachable
events are counted across hooks; at FAIL_LOUD_THRESHOLD the notice names the count; a success resets
it. The service URL is a closed port on loopback; nothing is sent anywhere and nothing is stored.
"""

from __future__ import annotations

from nsaccept import expect_check_fails, needs, negative_control, proves
from nsaccept.hooks import HookResult
from plugin_kit import Session, assert_continues, context_of, option

pytestmark = needs("node")

DEAD_URL = "http://127.0.0.1:1"

# hook script -> exit code on malformed stdin (the README's taxonomy: 2 only where it cannot block)
MALFORMED = {
    "session-start": 2,
    "session-summary": 2,
    "post-tool-use": 2,
    "session-end": 0,  # Stop
    "user-prompt-submit": 0,
    "pre-tool-use": 0,  # exit 2 would DENY the Read
    "pre-compact": 0,
}


def check_exit_code(res: HookResult, want: int) -> None:
    assert res.returncode == want, f"exit code {res.returncode}, want {want}: {res.stderr[:160]}"


@proves("plugin.hook-exit-taxonomy")
def test_malformed_input_exits_2_only_where_that_cannot_block_claude(ctx):
    s = Session(ctx, project_id="acc-malformed", proxy=None, url=DEAD_URL)
    try:
        for script, want in MALFORMED.items():
            res = s.hook(script, {}, raw_stdin="{this is not json")
            check_exit_code(res, want)
            if want == 0 and script != "pre-compact":
                assert_continues(res)
    finally:
        s.close()


@proves("plugin.hook-exit-taxonomy")
def test_an_unreachable_service_never_blocks_any_hook(ctx):
    s = Session(ctx, project_id="acc-down", proxy=None, url=DEAD_URL)
    try:
        s.transcript.turn("Please look at the pier.", "Looked at it, with enough words to be a real answer.")
        results = {
            "SessionStart": s.session_start(),
            "UserPromptSubmit": s.user_prompt("hello there"),
            "PostToolUse": s.bash("make build", "ok"),
            "Stop": s.stop("Looked at it, with enough words to be a real answer."),
            "PreCompact": s.pre_compact(),
            "SessionEnd": s.session_end(),
            "PreToolUse": s.pre_read(s.sandbox.work / "missing.txt"),
        }
        for event, res in results.items():
            check_exit_code(res, 0)
            assert res.data is None or res.data.get("continue") is True, f"{event} did not answer continue"
        text = context_of(results["SessionStart"])
        assert "unreachable" in text and "[neuralscape]" in text, f"no outage notice at SessionStart: {text[:160]!r}"
        assert s.offset() is None, "the transcript cursor moved although the flush could not be delivered"
    finally:
        s.close()


@proves("plugin.hook-exit-taxonomy")
@negative_control
def test_exit_check_fails_for_the_wrong_expectation(ctx):
    s = Session(ctx, project_id="acc-malformed", proxy=None, url=DEAD_URL)
    try:
        expect_check_fails(check_exit_code, s.hook("pre-tool-use", {}, raw_stdin="{nope"), 2)  # a PreToolUse exit 2 would deny the Read
        expect_check_fails(check_exit_code, s.hook("session-start", {}, raw_stdin="{nope"), 0)
    finally:
        s.close()


# ── the counter ───────────────────────────────────────────────────


def counter_value(s: Session) -> int | None:
    path = s.sandbox.observations / "unreachable.count"
    return int(path.read_text().strip()) if path.is_file() else None


def check_notice_names_count(text: str, count: int) -> None:
    assert f"unreachable for {count} consecutive events" in text, f"the notice does not name {count} consecutive events: {text[:160]!r}"


@proves("plugin.unreachable-counter")
def test_consecutive_outages_are_counted_across_hooks_and_the_notice_gets_loud_at_the_threshold(ctx):
    s = Session(ctx, project_id="acc-counter", proxy=None, url=DEAD_URL, extra_env=option("FAIL_LOUD_THRESHOLD", "3"))
    try:
        quiet = context_of(s.session_start())  # event 1
        assert "unreachable" in quiet and "consecutive" not in quiet, f"first outage already loud: {quiet[:160]!r}"
        assert counter_value(s) == 1
        s.transcript.turn("Please look at the pier.", "Looked at it, with enough words to be a real answer.")
        s.pre_compact()  # event 2, from a different hook
        assert counter_value(s) == 2, f"the counter is {counter_value(s)} after two outages, one from another hook"
        loud = context_of(s.session_start())  # event 3
        check_notice_names_count(loud, 3)
        assert "docker compose ps" in loud, "the loud notice does not say where to look"
        assert counter_value(s) == 3
    finally:
        s.close()


@proves("plugin.unreachable-counter")
def test_a_successful_call_resets_the_outage_streak(ctx):
    s = Session(ctx, project_id="acc-counter-reset", proxy=None, url=DEAD_URL, extra_env=option("FAIL_LOUD_THRESHOLD", "2"))
    try:
        s.session_start()
        check_notice_names_count(context_of(s.session_start()), 2)
        assert counter_value(s) == 2
        up = Session(ctx, project_id=s.project_id, sandbox=s.sandbox, extra_env=option("FAIL_LOUD_THRESHOLD", "2"))  # same data, the stack is back
        res = up.session_start()
        assert "unreachable" not in context_of(res), "an outage notice survived a successful call"
        assert counter_value(s) is None, "the counter was not cleared by a successful call"
        down_again = context_of(s.session_start())
        assert "consecutive" not in down_again, "the streak was not reset: the next outage is already loud"
    finally:
        s.close()


@proves("plugin.unreachable-counter")
@negative_control
def test_notice_check_fails_when_the_count_is_not_the_streak(ctx):
    s = Session(ctx, project_id="acc-counter-neg", proxy=None, url=DEAD_URL, extra_env=option("FAIL_LOUD_THRESHOLD", "3"))
    try:
        text = context_of(s.session_start())
        expect_check_fails(check_notice_names_count, text, 3)  # one outage is not three
    finally:
        s.close()
