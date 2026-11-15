"""Capture -> flush -> recall, through the plugin's BUILT hooks, plus `<private>` redaction at every egress.

One simulated Claude Code session in its own project: a session start, three
tool observations (PostToolUse), the compile prompt (UserPromptSubmit) crossing
its threshold, one real transcript turn (Stop -> POST /flush -> the fast worker's
Gemini extraction -> memories, then POST /compile). The hooks talk to the stack
through a recording proxy that forwards everything, so the scenarios see both what
the hook SENT and what the stack DID with it.

LLM cost: exactly ONE extraction flush for the whole module (one turn), shared by
every scenario below. Assertions on extracted facts use the unique marker only,
never the model's wording; the Stop's compile job is only checked to complete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pytest

from nsaccept import expect_check_fails, needs, negative_control, proves
from nsaccept.client import NSClient
from nsaccept.helpers import write_raw
from nsaccept.proxy import COMPILE, FLUSH, RecordingProxy
from plugin_kit import (
    Session,
    check_never_stored,
    check_recalled,
    context_of,
    everything_sent,
    flushes_with,
    option,
    wait_recalled,
)

pytestmark = needs("node")

THRESHOLD = 3


@dataclass
class Captured:
    session: Session
    proxy: RecordingProxy
    project: str
    m: dict[str, str]
    final_answer: str
    start: object = None
    prompt_below: object = None
    prompt_at: object = None
    prompt_again: object = None
    stop: object = None
    flush_status: dict = field(default_factory=dict)
    compile_status: dict = field(default_factory=dict)


@pytest.fixture(scope="module")
def captured(ctx, a):
    run = ctx.run_id
    project = f"{ctx.project_id}-pcap"
    m = {k: f"zq-{run}-{k}" for k in ("fact", "queue", "tool", "priv", "privout", "skip", "side", "meta", "toolres")}
    proxy = RecordingProxy(ctx.base_url, forward_llm=True).start()
    s = Session(ctx, project_id=project, proxy=proxy, extra_env=option("COMPILE_THRESHOLD", str(THRESHOLD)))
    work = s.sandbox.work
    cap = Captured(s, proxy, project, m, final_answer=f"Noted. The staging queue for this project is called {m['queue']}.")
    try:
        cap.start = s.session_start()
        cap.prompt_below = s.user_prompt("Let us start with the deploy codename.")

        # three observations the hook must keep, and four it must drop
        s.bash("make build", f"build finished: artifact {m['tool']}")
        s.post_tool(
            "Edit",
            {"file_path": f"{work}/src/{run}-app.ts", "old_string": "const a = 1", "new_string": "const a = 2"},
            {"filePath": f"{work}/src/{run}-app.ts", "success": True},
        )
        s.bash("make deploy", f"deployed <private>{m['privout']}</private> to staging")
        s.post_tool("Read", {"file_path": f"{work}/notes.txt"}, f"file text {m['skip']}")
        s.bash("ls -la", f"total 0 {m['skip']}")
        s.post_tool("mcp__plugin_neuralscape_neuralscape__remember", {"content": "x"}, f"stored {m['skip']}")

        cap.prompt_at = s.user_prompt("Please continue.")
        cap.prompt_again = s.user_prompt("And one more thing.")

        # the turn, in the shape Claude Code writes it
        t = s.transcript
        t.user(
            f"For the record: my deploy codename is {m['fact']}. <private>{m['priv']}</private> Please remember it."
        )
        t.assistant({"type": "thinking", "thinking": "synthetic reasoning"}, {
            "type": "tool_use", "id": "toolu_synthetic_cap", "name": "Bash", "input": {"command": "make build"},
        })
        t.tool_result("toolu_synthetic_cap", f"{m['toolres']} output of the tool, which is not conversation text")
        t.assistant(f"{m['side']} a sub-agent's own conversation", isSidechain=True)
        t.user(f"{m['meta']} injected context, not something the user said", isMeta=True)
        t.assistant(cap.final_answer)

        cap.stop = s.stop(cap.final_answer)
        flushes = proxy.calls(FLUSH, "POST")
        compiles = proxy.calls(COMPILE, "POST")
        if flushes and flushes[0].task_id:
            cap.flush_status = a.poll_task(flushes[0].task_id, timeout=240)
        if compiles and compiles[0].task_id:
            cap.compile_status = a.poll_task(compiles[0].task_id, timeout=240)
        yield cap
    finally:
        proxy.stop()
        s.close()


# ── PostToolUse buffer ────────────────────────────────────────────


def check_buffer(cap: Captured, *, tools: list[str], present: list[str], absent: list[str]) -> None:
    rows = cap.session.buffer_rows()
    assert [r.get("tool") for r in rows] == tools, f"buffered tools {[r.get('tool') for r in rows]}, want {tools}"
    blob = cap.session.buffer_path.read_text()
    for needle in present:
        assert needle in blob, f"{needle} not in the buffer"
    for needle in absent:
        assert needle not in blob, f"{needle} leaked into the buffer"
    for row in rows:
        assert row.get("session_id") == cap.session.session_id, f"row session {row.get('session_id')!r}"
        assert row.get("project_id") == cap.project, f"row project {row.get('project_id')!r}"
        assert row.get("user_id") == cap.session.user_id, f"row user {row.get('user_id')!r}"
        assert row.get("ts"), "row without a timestamp"


@proves("plugin.post-tool-use-buffer")
def test_post_tool_use_buffers_what_matters_and_drops_the_rest(captured):
    m = captured.m
    check_buffer(
        captured,
        tools=["Bash", "Edit", "Bash"],
        present=[m["tool"], "make build", "make deploy", "src/"],
        absent=[m["skip"], m["privout"]],
    )
    rows = captured.session.buffer_rows()
    assert rows[0]["input"].get("command") == "make build", "Bash input was not reduced to its command"
    assert "old_string" in rows[1]["input"] and "file_path" in rows[1]["input"], "Edit input lost its fields"


@proves("plugin.post-tool-use-buffer")
@negative_control
def test_buffer_check_fails_when_a_dropped_tool_is_expected(captured):
    # the Read / `ls` / Neuralscape-tool observations were (rightly) never buffered
    expect_check_fails(
        check_buffer, captured, tools=["Bash", "Edit", "Bash"], present=[captured.m["skip"]], absent=[]
    )


# ── UserPromptSubmit compile prompt ───────────────────────────────


def check_compile_prompt(res, cap: Captured) -> None:
    text = context_of(res)
    assert str(cap.session.buffer_path) in text, "the compile prompt does not name the buffer path"
    assert "compile-observations" in text, "the compile prompt does not name the skill"
    assert f"{THRESHOLD} tool observations" in text, f"the compile prompt does not state the row count ({THRESHOLD})"


@proves("plugin.compile-prompt-ups")
def test_compile_prompt_is_injected_once_when_the_threshold_is_crossed(captured):
    assert context_of(captured.prompt_below) == "", "a prompt injected the compile instruction before there was anything to compile"
    check_compile_prompt(captured.prompt_at, captured)
    assert context_of(captured.prompt_again) == "", "the compile instruction repeated on the very next prompt"


@proves("plugin.compile-prompt-ups")
@negative_control
def test_compile_prompt_check_fails_below_the_threshold(captured):
    expect_check_fails(check_compile_prompt, captured.prompt_below, captured)


# ── Stop: transcript -> /flush ────────────────────────────────────


def check_flush_body(cap: Captured, *, user_has: str, assistant_has: str, absent: list[str], user_id: str) -> None:
    calls = cap.proxy.calls(FLUSH, "POST")
    assert len(calls) == 1, f"{len(calls)} /flush requests for one finished turn (want 1)"
    call = calls[0]
    body = call.body
    assert user_has in body.get("user_message", ""), f"{user_has} not in the delivered user message"
    assert assistant_has in body.get("assistant_response", ""), f"{assistant_has} not in the delivered assistant response"
    sent = call.raw_body
    for needle in absent:
        assert needle not in sent, f"{needle} was delivered but is not conversation text"
    assert body.get("session_id") == cap.session.session_id, f"session_id {body.get('session_id')!r}"
    assert body.get("channel") == "claude-code", f"channel {body.get('channel')!r}"
    assert body.get("project_id") == cap.project, f"project_id {body.get('project_id')!r}"
    assert body.get("user_id") == user_id, f"user_id {body.get('user_id')!r}"
    assert call.bearer == cap.session.token, "the flush was not sent with the configured API key"
    assert call.status == 202, f"/flush answered {call.status}"


@proves("plugin.stop-transcript-flush")
def test_stop_delivers_the_real_turn_and_only_the_turn(captured):
    m = captured.m
    check_flush_body(
        captured,
        user_has=m["fact"],
        assistant_has=m["queue"],
        absent=[m["toolres"], m["side"], m["meta"]],
        user_id=captured.session.user_id,
    )
    assert captured.session.offset() is not None and captured.session.offset() >= 2, (
        f"the transcript cursor was not committed after delivery (offset {captured.session.offset()})"
    )


@proves("plugin.stop-transcript-flush")
@negative_control
def test_flush_check_fails_for_text_that_was_never_conversation(captured):
    # the sub-agent entry is in the transcript but must not be in the delivered turn
    expect_check_fails(
        check_flush_body, captured,
        user_has=captured.m["fact"], assistant_has=captured.m["side"], absent=[], user_id=captured.session.user_id,
    )


# ── the stack extracts, stores and serves what the hook sent ──────


@proves("ext.compiler-flush", "plugin.stop-transcript-flush")
def test_the_flushed_turn_is_extracted_and_recallable(captured, a):
    task = captured.flush_status
    assert task.get("status") == "completed", f"the flush task ended as {task.get('status')!r}: {str(task.get('error'))[:200]}"
    stored = (task.get("result") or {}).get("memories_stored", 0)
    assert stored >= 1, f"the worker completed the flush but stored {stored} memories"
    m = captured.m
    wait_recalled(a, captured.project, m["fact"], m["queue"])


@proves("ext.compiler-flush")
@negative_control
def test_recall_check_fails_for_another_user(captured, b):
    m = captured.m
    expect_check_fails(check_recalled, b, captured.project, m["fact"], m["queue"])


def check_compile_ran(cap: Captured) -> None:
    calls = cap.proxy.calls(COMPILE, "POST")
    assert calls, "the Stop hook never asked for a compile"
    sent = calls[0]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", sent.body.get("date", "")), f"compile date {sent.body.get('date')!r}"
    assert sent.body.get("user_id") == cap.session.user_id, f"compile user {sent.body.get('user_id')!r}"
    status = cap.compile_status
    assert status.get("status") == "completed", f"the compile task ended as {status.get('status')!r}: {str(status.get('error'))[:200]}"
    result = status.get("result") or {}
    assert result.get("date") == sent.body["date"], f"the worker compiled {result.get('date')!r}, not the requested {sent.body['date']}"


@proves("ext.compiler-compile")
def test_stop_triggers_a_compile_the_worker_completes(captured):
    check_compile_ran(captured)


@proves("ext.compiler-compile")
@negative_control
def test_compile_check_fails_when_no_compile_was_requested(captured):
    # the real check, fed a session that never reached a Stop
    idle = Captured(captured.session, RecordingProxy(None).start(), captured.project, captured.m, captured.final_answer)
    try:
        expect_check_fails(check_compile_ran, idle)
    finally:
        idle.proxy.stop()


# ── <private> redaction at every egress of this session ───────────


def check_private_stays_home(cap: Captured, a: NSClient, *, wire: list[str], disk: list[str], stack: list[str]) -> None:
    sent = everything_sent(cap.proxy)
    for needle in wire:
        assert needle not in sent, f"{needle} was put on the wire"
    buffer = cap.session.buffer_path.read_text() if cap.session.buffer_path.is_file() else ""
    for needle in disk:
        assert needle not in buffer, f"{needle} was written to the observation buffer"
    for needle in stack:
        check_never_stored(a, cap.project, needle)


@proves("plugin.private-redaction")
def test_private_spans_reach_neither_the_wire_nor_the_buffer_nor_the_stack(captured, a):
    m = captured.m
    assert captured.flush_status.get("status") == "completed", "precondition: the flush was never processed, so 'not stored' proves nothing"
    assert flushes_with(captured.proxy, m["fact"]), "precondition: the turn around the private span was not delivered"
    check_private_stays_home(captured, a, wire=[m["priv"], m["privout"]], disk=[m["privout"]], stack=[m["priv"]])
    delivered = flushes_with(captured.proxy, m["fact"])[0]["user_message"]
    assert "[redacted]" in delivered, "the private span was removed without a redaction marker"


@proves("plugin.private-redaction")
@negative_control
def test_private_check_fails_for_text_that_was_sent(captured, a):
    # the outer (public) text WAS delivered: the same check must notice it on the wire
    expect_check_fails(check_private_stays_home, captured, a, wire=[captured.m["fact"]], disk=[], stack=[])


@proves("plugin.private-redaction")
@negative_control
def test_never_stored_check_fails_for_a_stored_memory(ctx, a):
    marker = f"zq-{ctx.run_id}-negstore"
    project = f"{ctx.project_id}-pcap"
    w = write_raw(a, f"{marker} a public fact about the pier lamps.", "tech_stack", marker=marker,
                  scope="project", project_id=project, visibility="private")
    w.require()
    wait_recalled(a, project, marker)
    expect_check_fails(check_never_stored, a, project, marker)


@proves("plugin.private-redaction", "plugin.post-tool-use-buffer")
def test_an_unclosed_multiline_private_span_does_not_leak_its_later_lines(ctx):
    """EXPECTED RED until PR #514 (plg-07, bug/plugin-stop-flush-and-privacy): today an unmatched
    `<private>` is redacted only to the end of its line, so the second line reaches the buffer."""
    first, second = f"zq-{ctx.run_id}-unc1", f"zq-{ctx.run_id}-unc2"
    s = Session(ctx, project_id="acc-unclosed", proxy=None, url="http://127.0.0.1:1")
    try:
        # tool_response as a plain string with a real newline (some tools answer that way)
        s.post_tool("Bash", {"command": "make deploy"}, f"before <private>{first} starts here\n{second} is on the next line")
        blob = s.buffer_path.read_text()
        assert first not in blob, f"{first} (inside the span) was buffered"
        assert second not in blob, f"{second} (a later line of an unclosed span) was buffered"
    finally:
        s.close()
