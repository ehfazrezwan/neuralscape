"""The Stop hook's delivery guarantees: a turn is sent once, the cursor only moves past what
the service accepted, and two hooks on one transcript do not both deliver it.

These are properties of what the hook PUTS ON THE WIRE, so the recording proxy answers the two
LLM-backed conversation-compiler routes itself: no extraction call is made and nothing is stored.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from nsaccept import expect_check_fails, needs, negative_control, proves
from nsaccept.proxy import FLUSH, RecordingProxy
from plugin_kit import Session, check_sent_exactly_once, flushes_with

pytestmark = needs("node")


@pytest.fixture
def lane(ctx):
    proxy = RecordingProxy(ctx.base_url).start()  # flush + compile answered locally
    s = Session(ctx, project_id=f"{ctx.project_id}-pofs", proxy=proxy)
    yield s, proxy
    proxy.stop()
    s.close()


def answer(marker: str) -> str:
    return f"Done. The result for {marker} is written down, with enough words to count as a real answer."


def add_turn(s: Session, marker: str) -> str:
    final = answer(marker)
    s.transcript.turn(f"Please look into {marker} for me.", final)
    return final


@proves("plugin.flush-offset-commit", "plugin.stop-transcript-flush")
def test_a_second_stop_with_no_new_turns_sends_nothing(ctx, lane):
    s, proxy = lane
    first, second = f"zq-{ctx.run_id}-off1", f"zq-{ctx.run_id}-off2"
    final = add_turn(s, first)
    s.stop(final)
    check_sent_exactly_once(proxy, first)
    cursor = s.offset()
    assert cursor == 2, f"the cursor after one delivered turn is {cursor}, want 2 (one prompt, one answer)"

    s.stop(final)  # nothing new in the transcript
    s.stop(final)
    check_sent_exactly_once(proxy, first)
    assert s.offset() == cursor, "the cursor moved although nothing new was delivered"

    final2 = add_turn(s, second)
    s.stop(final2)
    check_sent_exactly_once(proxy, second)
    check_sent_exactly_once(proxy, first)  # the first turn was not sent along with the new one
    assert s.offset() == 4, f"the cursor after two delivered turns is {s.offset()}, want 4"


@proves("plugin.flush-offset-commit")
@negative_control
def test_once_check_fails_when_the_cursor_is_lost(ctx, lane):
    s, proxy = lane
    marker = f"zq-{ctx.run_id}-offlost"
    final = add_turn(s, marker)
    s.stop(final)
    check_sent_exactly_once(proxy, marker)
    s.offset_path.unlink()  # a lost cursor makes the hook re-send what it already delivered
    s.stop(final)
    expect_check_fails(check_sent_exactly_once, proxy, marker)


@proves("plugin.flush-offset-commit")
def test_the_cursor_stays_put_while_the_service_refuses_and_the_turns_follow_in_order_after(ctx, lane):
    s, proxy = lane
    one, two = f"zq-{ctx.run_id}-ref1", f"zq-{ctx.run_id}-ref2"
    add_turn(s, one)
    final = add_turn(s, two)

    proxy.fail[FLUSH] = 500
    res = s.stop(final)
    assert res.returncode == 0, f"a refused flush made the hook exit {res.returncode}"
    assert s.offset() is None, f"the cursor was committed ({s.offset()}) although nothing was delivered"
    attempts = proxy.calls(FLUSH, "POST")
    assert len(attempts) == 1, f"{len(attempts)} flush attempts while the service was down: delivery must stop at the first failure"

    del proxy.fail[FLUSH]
    s.stop(final)
    delivered = [c for c in proxy.calls(FLUSH, "POST") if c.status == 202]
    assert [one in c.body["user_message"] for c in delivered] == [True, False], "first turn not delivered first"
    assert two in delivered[1].body["user_message"], "second turn not delivered second"
    assert s.offset() == 4, f"the cursor after delivery is {s.offset()}, want 4"


@proves("plugin.transcript-lock", "plugin.flush-offset-commit")
def test_four_stops_started_together_deliver_the_turn_once(ctx, lane):
    s, proxy = lane
    marker = f"zq-{ctx.run_id}-lock"
    final = add_turn(s, marker)
    proxy.delay[FLUSH] = 1.5  # long enough that the four hooks really overlap
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: s.stop(final), range(4)))
    assert [r.returncode for r in results] == [0, 0, 0, 0], f"exit codes {[r.returncode for r in results]}"
    check_sent_exactly_once(proxy, marker)
    assert s.offset() == 2, f"the cursor ended at {s.offset()}, want 2"
    assert not list(s.transcript.path.parent.glob("*.neuralscape-lock*")), "a lock file was left behind"


@proves("plugin.transcript-lock")
@negative_control
def test_once_check_fails_when_two_hooks_do_not_share_a_lock(ctx, lane):
    s, proxy = lane
    marker = f"zq-{ctx.run_id}-nolock"
    final = add_turn(s, marker)
    twin = s.transcript.path.with_name("twin.jsonl")  # another path: another lock and another cursor
    shutil.copy(s.transcript.path, twin)
    s.stop(final)
    s.hook("session-end", {**s.payload("Stop", stop_hook_active=False, last_assistant_message=final), "transcript_path": str(twin)})
    assert len(flushes_with(proxy, marker)) == 2, "precondition: both transcripts were delivered"
    expect_check_fails(check_sent_exactly_once, proxy, marker)
