"""Plugin Stop / SessionEnd hooks against the live stack: privacy and the windowed compile.

Runs the plugin's BUILT ``session-end.js`` (the Stop hook) and ``session-summary.js``
(the SessionEnd hook) on a real-shaped Claude Code transcript whose turns carry
`<private>` spans that are nested, split across entries, opened in the prompt and
closed in the answer, and never closed. The stack ends up with the public
facts and none of the sentinel strings, in the turn's memories and in the
session-note memory.

Compile requests are not observable from the stack (the route enqueues a job;
the status route only reports the last compile), so the "at most once per
window" check reads the hook's own state file in the plugin data dir the
scenario owns: compiledAt must not move across the second and third Stop of a
session, must move at SessionEnd, and the dirty dates must be empty after it.

Assertions are on unique markers and sentinels, never on model wording. The
turn's memories come from the LLM extraction path, so the positive control
(the public marker IS stored) is asserted first: a sentinel check on an empty
result would prove nothing.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nsaccept import eventually, expect_check_fails, needs, negative_control, proves
from nsaccept.hooks import run_hook

CONTEXT_LOOKBACK = 200


def _line(role: str, content) -> str:
    return json.dumps({"type": role, "message": {"role": role, "content": content}}) + "\n"


def _user(text: str) -> str:
    return _line("user", text)


def _assistant(text: str) -> str:
    return _line("assistant", [{"type": "text", "text": text}])


class Session:
    def __init__(self, ctx, user_id: str, token: str, tag: str):
        self.ctx, self.user_id, self.token = ctx, user_id, token
        self.session_id = f"acc-{ctx.run_id}-{tag}"
        self.tmp = tempfile.TemporaryDirectory(prefix="ns-accept-stop-")
        root = Path(self.tmp.name)
        self.data_dir = root / "plugin-data"
        self.data_dir.mkdir()
        self.transcript = root / "transcript.jsonl"
        self.transcript.write_text("")
        self.public = f"zq-{ctx.run_id}-{tag}"
        self.sentinels = [f"zqsecret{ctx.run_id.replace('-', '')}{tag}{n}" for n in "ABCDEF"]
        self.results: list = []

    def append(self, *lines: str) -> None:
        with self.transcript.open("a") as fh:
            fh.writelines(lines)

    def _run(self, hook: str, event: str, **payload):
        res = run_hook(
            hook,
            {"session_id": self.session_id, "transcript_path": str(self.transcript), "hook_event_name": event, **payload},
            base_url=self.ctx.base_url,
            token=self.token,
            user_id=self.user_id,
            project_id=self.ctx.project_id,
            extra_env={"CLAUDE_PLUGIN_DATA": str(self.data_dir), "NEURALSCAPE_TRANSCRIPT_LAG_WAIT_MS": "0"},
        )
        assert res.returncode == 0, f"{hook} exited {res.returncode}: {res.stderr[:300]}"
        self.results.append(res)
        return res

    def stop(self):
        return self._run("session-end", "Stop")

    def session_end(self):
        return self._run("session-summary", "SessionEnd", reason="exit")

    def compile_state(self) -> dict:
        path = self.data_dir / "observations" / f"{self.session_id}.compile.lastturn"
        return json.loads(path.read_text()) if path.is_file() else {"dirtyDates": [], "compiledAt": 0}


@pytest.fixture(scope="module")
def stopped(ctx, a):
    """Three turns, one Stop each, with private spans of every awkward shape; then SessionEnd."""
    s = Session(ctx, a.user_id, ctx.token_a, "privstop")
    s1, s2, s3, s4, s5, s6 = s.sentinels
    out = {"session": s, "states": []}

    # Turn 1: a span opened in the prompt and closed in the answer, plus a public decision.
    s.append(
        _user(
            f"We need a name for the new build tool. Decision: the build tool is called {s.public}-tool. "
            f"My registry token is <private>{s1}"
        ),
        _assistant(f"Recorded: the build tool is named {s.public}-tool. The token {s2}</private> stays private."),
    )
    s.stop()
    out["states"].append(s.compile_state())

    # Turn 2: a NESTED span split across two assistant entries.
    s.append(
        _user("Next, record where the deploy config lives."),
        _assistant(f"Working on it. <private>outer {s3} <private>inner {s4}"),
        _assistant(f"{s4}</private> {s5}</private>\nDecision: deploy config lives in the {s.public}-deploy directory."),
    )
    s.stop()
    out["states"].append(s.compile_state())

    # Turn 3: an unclosed opener (fails closed: the rest of that text is hidden).
    s.append(
        _user("Finally note the rollback window."),
        _assistant(f"Decision: the rollback window is fifteen minutes ({s.public}-rollback). Backup key <private>{s6}\nmore secret"),
    )
    s.stop()
    out["states"].append(s.compile_state())

    out["end"] = s.session_end()
    out["state_after_end"] = s.compile_state()
    return out


def _project_text(a, ctx) -> str:
    rows = a.get("/v1/memories", params={"project_id": ctx.project_id, "limit": CONTEXT_LOOKBACK}).json()
    return "\n".join(m.get("memory", "") for m in rows)


def check_public_decision_stored(a, ctx, marker: str) -> None:
    assert marker in _project_text(a, ctx), f"{marker} not among the project's memories yet"


def check_no_sentinels(a, ctx, sentinels: list[str]) -> None:
    text = _project_text(a, ctx)
    leaked = [s for s in sentinels if s in text]
    assert not leaked, f"private text reached the stack: {leaked}"


def _note_text(a, ctx) -> str:
    rows = a.get("/v1/memories", params={"project_id": ctx.project_id, "limit": CONTEXT_LOOKBACK}).json()
    return "\n".join(m.get("memory", "") for m in rows if "Session note:" in m.get("memory", ""))


def check_session_note_stored(a, ctx, marker: str) -> None:
    assert marker in _note_text(a, ctx), "session note (with the public marker) not stored yet"


@needs("node")
@proves("plugin.stop-transcript-flush", "plugin.private-redaction")
def test_stop_turns_reach_the_stack_without_private_text(ctx, a, stopped):
    s = stopped["session"]
    # Positive control: the public decisions of the same turns ARE stored (LLM extraction; wait for the async writes).
    eventually(
        lambda: check_public_decision_stored(a, ctx, f"{s.public}-tool") or check_public_decision_stored(a, ctx, f"{s.public}-deploy"),
        timeout=240, interval=5, what="a public decision of the Stop-delivered turns is stored",
    )
    check_no_sentinels(a, ctx, s.sentinels)


@needs("node")
@proves("plugin.stop-transcript-flush", "plugin.private-redaction")
def test_session_end_note_has_the_public_wrap_up_and_no_private_text(ctx, a, stopped):
    s = stopped["session"]
    eventually(lambda: check_session_note_stored(a, ctx, s.public), timeout=120, interval=3, what="session note stored")
    note = _note_text(a, ctx)
    assert not [x for x in s.sentinels if x in note], "private text reached the session note"
    check_no_sentinels(a, ctx, s.sentinels)


@needs("node")
@proves("plugin.stop-transcript-flush")
def test_compile_is_requested_once_per_window_and_finally_at_session_end(stopped):
    first, second, third = stopped["states"]
    assert first["compiledAt"] > 0, f"the first delivering Stop did not compile: {first}"
    assert second["compiledAt"] == first["compiledAt"], "a second Stop inside the window compiled again"
    assert third["compiledAt"] == first["compiledAt"], "a third Stop inside the window compiled again"
    assert second["dirtyDates"] and third["dirtyDates"], "turns delivered inside the window were not marked for the final compile"
    after = stopped["state_after_end"]
    assert after["compiledAt"] > first["compiledAt"], f"SessionEnd did not compile what was left: {after}"
    assert after["dirtyDates"] == [], f"dates still dirty after SessionEnd: {after}"


@needs("node")
@proves("plugin.stop-transcript-flush")
@negative_control
def test_sentinel_check_fails_when_a_sentinel_is_present(ctx, a, stopped):
    # The same check must fail for a string that IS in the project's memories (the public marker).
    s = stopped["session"]
    eventually(lambda: check_public_decision_stored(a, ctx, f"{s.public}-tool") or check_public_decision_stored(a, ctx, f"{s.public}-deploy"),
               timeout=240, interval=5, what="public decision stored")
    text_marker = f"{s.public}-tool" if f"{s.public}-tool" in _project_text(a, ctx) else f"{s.public}-deploy"
    expect_check_fails(check_no_sentinels, a, ctx, [text_marker])
