"""The stale-buffer handoff: observations left by a session that is OVER are offered to the next
session for compilation; a session that is still running is never touched.

Three sessions share one plugin data directory (as two terminals of one user would): one ended
(SessionEnd), one still live, one that ended and then kept working. A fourth session starts and
must be told about the first only, with the service up AND down. No LLM: the proxy records the
SessionEnd's checkpoint call without storing anything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from nsaccept import expect_check_fails, needs, negative_control, proves
from nsaccept.hooks import HookResult, HookSandbox
from nsaccept.proxy import LLM_PATHS, RecordingProxy
from plugin_kit import Session, context_of, option

pytestmark = needs("node")

PENDING = "# Neuralscape — Pending Observations"
DEAD_URL = "http://127.0.0.1:1"


@dataclass
class Terminals:
    sandbox: HookSandbox
    project: str
    ended: Session
    live: Session
    resumed: Session
    before: dict


@pytest.fixture(scope="module")
def terminals(ctx):
    project = f"{ctx.project_id}-pbuf"
    sandbox = HookSandbox()
    proxy = RecordingProxy(ctx.base_url, stub=LLM_PATHS + ("/v1/checkpoint",)).start()
    kw = dict(project_id=project, proxy=proxy, sandbox=sandbox)
    ended, live, resumed = Session(ctx, **kw), Session(ctx, **kw), Session(ctx, **kw)
    for s in (ended, live, resumed):
        s.bash("make build", f"built for {s.session_id}")
        s.bash("make test", f"tested for {s.session_id}")
    ended.transcript.turn("Please build it.", "Built, with enough words to be an answer for the note.")
    ended.session_end()
    resumed.transcript.turn("Please test it.", "Tested, with enough words to be an answer for the note.")
    resumed.session_end()
    time.sleep(0.2)  # the marker must be OLDER than the next write
    resumed.bash("make deploy", "this session kept working after its SessionEnd")
    before = {s.session_id: s.buffer_path.read_text() for s in (ended, live, resumed)}
    yield Terminals(sandbox, project, ended, live, resumed, before)
    proxy.stop()
    sandbox.close()


def check_pending_lists(res: HookResult, session: Session) -> None:
    text = context_of(res)
    assert PENDING in text, "no pending-observations note was injected"
    assert str(session.buffer_path) in text, f"the buffer of session {session.session_id} is not offered for compilation"


def new_session(ctx, t: Terminals, **kw) -> Session:
    return Session(ctx, project_id=t.project, sandbox=t.sandbox, **kw)


@proves("plugin.stale-buffer-handoff")
def test_next_session_is_told_to_compile_the_ended_sessions_buffer_and_only_that(ctx, terminals: Terminals):
    with RecordingProxy(ctx.base_url) as proxy:
        res = new_session(ctx, terminals, proxy=proxy).session_start("startup")
    check_pending_lists(res, terminals.ended)
    text = context_of(res)
    assert str(terminals.live.buffer_path) not in text, "a session that is still running was offered for compilation (and truncation)"
    assert str(terminals.resumed.buffer_path) not in text, "a session that kept working after its SessionEnd was offered"
    assert "compile-observations" in text, "the note does not name the skill that compiles buffers"
    for s in (terminals.ended, terminals.live, terminals.resumed):
        assert s.buffer_path.read_text() == terminals.before[s.session_id], f"the hook modified the buffer of {s.session_id}"


@proves("plugin.stale-buffer-handoff")
def test_the_handoff_still_happens_while_the_service_is_down(ctx, terminals: Terminals):
    res = new_session(ctx, terminals, url=DEAD_URL).session_start("startup")
    assert "unreachable" in context_of(res), "the outage was not announced"
    check_pending_lists(res, terminals.ended)
    assert str(terminals.live.buffer_path) not in context_of(res)


@proves("plugin.stale-buffer-handoff")
def test_a_buffer_nobody_has_written_to_for_a_long_time_counts_as_abandoned(ctx, terminals: Terminals):
    time.sleep(1.2)
    # PENDING_IDLE_HOURS: 0.0002 h = 0.72 s of silence; the default is 12 h (a crashed session)
    res = new_session(ctx, terminals, url=DEAD_URL).session_start("startup", **option("PENDING_IDLE_HOURS", "0.0002"))
    check_pending_lists(res, terminals.live)


@proves("plugin.stale-buffer-handoff")
@negative_control
def test_pending_check_fails_for_a_session_that_is_still_running(ctx, terminals: Terminals):
    with RecordingProxy(ctx.base_url) as proxy:
        res = new_session(ctx, terminals, proxy=proxy).session_start("startup")
    expect_check_fails(check_pending_lists, res, terminals.live)
    expect_check_fails(check_pending_lists, res, terminals.resumed)
