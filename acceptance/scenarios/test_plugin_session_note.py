"""SessionEnd structured note -> stored session note -> "Previously" at the next SessionStart.

The SessionEnd hook builds a five-field note from the transcript and the PostToolUse buffer
(regex heuristics, no LLM) and sends it in ONE `POST /v1/checkpoint`. The next session in the project
shows it back as the "Previously" block. LLM-free; the conversation-compiler routes stay stubbed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nsaccept import eventually, expect_check_fails, needs, negative_control, proves
from nsaccept.hooks import HookResult
from nsaccept.proxy import LLM_PATHS, RecordingProxy
from plugin_kit import Session, check_never_stored, context_of, everything_sent, memories_with, task_completed

pytestmark = needs("node")

CHECKPOINT = "/v1/checkpoint"


@dataclass
class Ended:
    s: Session
    proxy: RecordingProxy
    project: str
    m: dict
    end: HookResult


@pytest.fixture(scope="module")
def ended(ctx, a):
    run = ctx.run_id
    project = f"{ctx.project_id}-pnote"
    m = {k: f"zq-{run}-{k}" for k in ("req", "next", "priv")}
    proxy = RecordingProxy(ctx.base_url).start()
    s = Session(ctx, project_id=project, proxy=proxy)
    work = s.sandbox.work
    s.post_tool(
        "Edit",
        {"file_path": f"{work}/src/{run}-note.ts", "old_string": "a", "new_string": "b"},
        {"filePath": f"{work}/src/{run}-note.ts", "success": True},
    )
    s.bash("make test", "12 passed")
    s.transcript.turn(
        f"Please fix the flaky cache test {m['req']} <private>{m['priv']}</private> before the release.",
        "All done, the fix is in place. Turns out the bug was in the cache layer, not the test.\n\n"
        f"Next steps:\n- {m['next']} verify on staging\n- rotate the keys",
    )
    end = s.session_end()
    sent = proxy.calls(CHECKPOINT, "POST")
    if sent and sent[0].task_id:
        a.poll_task(sent[0].task_id, timeout=120)
    yield Ended(s, proxy, project, m, end)
    proxy.stop()
    s.close()


@proves("plugin.session-end-note")
def test_session_end_sends_one_structured_note_and_no_memories(ended: Ended):
    assert ended.end.returncode == 0, f"SessionEnd exited {ended.end.returncode}"
    posts = ended.proxy.calls(CHECKPOINT, "POST")
    assert len(posts) == 1, f"{len(posts)} checkpoint calls from one SessionEnd (want 1)"
    body = posts[0].body
    assert body.get("memories") == [], f"the note call also carried memories: {body.get('memories')!r}"
    assert body.get("user_id") == ended.s.user_id and body.get("project_id") == ended.project, (
        f"identity {body.get('user_id')!r}/{body.get('project_id')!r}"
    )
    note = body.get("session_note") or {}
    assert ended.m["req"] in note.get("request", ""), f"request {note.get('request')!r}"
    assert ended.m["next"] in note.get("next_steps", ""), f"next_steps {note.get('next_steps')!r}"
    assert "cache layer" in note.get("learned", ""), f"learned {note.get('learned')!r}"
    assert "1 file(s)" in note.get("investigated", "") and "1 command(s)" in note.get("investigated", ""), (
        f"investigated {note.get('investigated')!r}"
    )
    assert note.get("completed"), "no 'completed' field"
    assert posts[0].bearer == ended.s.token, "the note was not sent with the configured API key"
    marker = ended.s.sandbox.observations / f"{ended.s.session_id}.jsonl.ended"
    assert marker.is_file(), "SessionEnd left no 'session ended' marker next to the buffer"


@proves("plugin.session-end-note", "plugin.private-redaction")
def test_the_private_span_in_the_request_never_leaves_the_machine(ended: Ended, a):
    assert ended.m["priv"] not in everything_sent(ended.proxy), "a <private> span was sent to the service"
    task_completed(a, ended.proxy.calls(CHECKPOINT, "POST")[0].task_id, timeout=120)
    check_never_stored(a, ended.project, ended.m["priv"])


def note_rows(client, project: str, marker: str) -> list[dict]:
    rows = [r for r in memories_with(client, project, marker) if "session_note" in (r.get("tags") or [])]
    assert rows, f"no session note of {project} carries {marker}"
    return rows


@proves("plugin.session-end-note")
def test_the_note_is_stored_as_a_session_note_memory(ended: Ended, a):
    task_completed(a, ended.proxy.calls(CHECKPOINT, "POST")[0].task_id, timeout=120)
    row = eventually(lambda: note_rows(a, ended.project, ended.m["req"]), timeout=60, interval=2, what="the stored session note")[0]
    assert row["memory"].startswith("Session note:"), f"note body starts {row['memory'][:30]!r}"
    assert ended.m["next"] in row["memory"] and row.get("category") == "task_context", f"category {row.get('category')!r}"


@proves("plugin.session-end-note")
@negative_control
def test_note_listing_check_fails_for_another_user(ended: Ended, b):
    expect_check_fails(note_rows, b, ended.project, ended.m["req"])


# ── the next session ──────────────────────────────────────────────


def check_previously(res: HookResult, ended: Ended) -> None:
    text = context_of(res)
    assert "## Previously" in text, "no 'Previously' block was injected"
    assert "**Next steps:**" in text and ended.m["next"] in text, "the note's next steps are not shown"
    assert text.count(ended.m["req"]) == 1, f"the note's request appears {text.count(ended.m['req'])} times (the note must not also be indexed)"


@proves("plugin.previously-block", "plugin.session-end-note")
def test_the_next_session_starts_with_the_previous_sessions_note(ctx, ended: Ended, a):
    task_completed(a, ended.proxy.calls(CHECKPOINT, "POST")[0].task_id, timeout=120)
    proxy = RecordingProxy(ctx.base_url).start()
    nxt = Session(ctx, project_id=ended.project, proxy=proxy)  # new session, fresh plugin data
    try:
        eventually(lambda: check_previously(nxt.session_start("startup"), ended), timeout=60, interval=3, what="the Previously block")
    finally:
        proxy.stop()
        nxt.close()


@proves("plugin.previously-block")
@negative_control
def test_previously_check_fails_for_a_user_who_does_not_own_the_note(ctx, ended: Ended):
    proxy = RecordingProxy(ctx.base_url).start()
    other = Session(ctx, project_id=ended.project, proxy=proxy, user_id=ctx.user_b, token=ctx.token_b)
    try:
        expect_check_fails(check_previously, other.session_start("startup"), ended)
    finally:
        proxy.stop()
        other.close()


# ── a private span split over two assistant entries ───────────────


@proves("plugin.session-end-note", "plugin.private-redaction")
def test_a_private_span_split_over_two_assistant_entries_stays_out_of_the_note(ctx):
    """EXPECTED RED until PR #514 (plg-07, bug/plugin-stop-flush-and-privacy): the note's `completed`
    field redacts the final assistant entry alone, so a span opened in the earlier entry loses its
    opening tag and its tail (`SECRETB`) is sent to /v1/checkpoint."""
    a_mark, b_mark = f"zq-{ctx.run_id}-splita", f"zq-{ctx.run_id}-splitb"
    proxy = RecordingProxy(ctx.base_url, stub=LLM_PATHS + (CHECKPOINT,)).start()  # record only: nothing is stored
    s = Session(ctx, project_id=f"{ctx.project_id}-psplit", proxy=proxy)
    try:
        s.transcript.user("Please wrap up the release.")
        s.transcript.assistant(f"Working on it. <private>{a_mark} stays secret\nstill secret")
        s.transcript.assistant(f"{b_mark}</private>\nAll done, shipped to staging.")
        res = s.session_end()
        assert res.returncode == 0, f"SessionEnd exited {res.returncode}"
        posts = proxy.calls(CHECKPOINT, "POST")
        assert len(posts) == 1, f"{len(posts)} checkpoint calls"
        assert "shipped to staging" in (posts[0].body.get("session_note") or {}).get("completed", ""), "the note lost its wrap-up"
        sent = everything_sent(proxy)
        assert a_mark not in sent, f"{a_mark} (inside the span) was sent"
        assert b_mark not in sent, f"{b_mark} (the span's tail, in the final entry) was sent"
    finally:
        proxy.stop()
        s.close()
