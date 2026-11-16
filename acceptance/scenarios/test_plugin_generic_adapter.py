"""The generic adapter: `scripts/conversation-turn.js` fed the normalised JSON any other agent can pipe in.

No transcript, no cursor: the plugin keeps the hash of the last delivered turn instead, so a
second identical delivery is dropped. LLM cost: ONE extraction flush (the second call of the
suite's plugin scenarios; the first is in test_plugin_capture_flush).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nsaccept import expect_check_fails, needs, negative_control, proves
from nsaccept.proxy import FLUSH, RecordingProxy
from plugin_kit import Session, check_recalled, check_sent_exactly_once, flushes_with, wait_recalled

pytestmark = needs("node")


@dataclass
class Generic:
    proxy: RecordingProxy
    project: str
    marker: str
    session_id: str
    first: object
    second: object
    flush_status: dict


@pytest.fixture(scope="module")
def generic(ctx, a):
    project = f"{ctx.project_id}-pgen"
    marker = f"zq-{ctx.run_id}-gen"
    session_id = f"acc-{ctx.run_id}-generic"
    payload = {
        "user_message": f"Remember that the harbour crane operator's callsign is {marker}.",
        "assistant_response": f"Understood: the callsign {marker} is recorded for the harbour crane operator.",
        "session_id": session_id,
        "channel": "acc-generic",
        "project_id": project,
        "user_id": ctx.user_a,
    }
    proxy = RecordingProxy(ctx.base_url, forward_llm=True).start()
    s = Session(ctx, project_id=project, proxy=proxy)
    try:
        first = s.hook("conversation-turn", payload)
        second = s.hook("conversation-turn", payload)  # the same turn again: must not be delivered twice
        status: dict = {}
        sent = proxy.calls(FLUSH, "POST")
        if sent and sent[0].task_id:
            status = a.poll_task(sent[0].task_id, timeout=240)
        yield Generic(proxy, project, marker, session_id, first, second, status)
    finally:
        proxy.stop()
        s.close()


@proves("plugin.generic-adapter")
def test_generic_payload_is_delivered_once_with_its_own_identity(ctx, generic: Generic):
    assert generic.first.returncode == 0 and generic.second.returncode == 0, "the hook did not exit 0"
    check_sent_exactly_once(generic.proxy, generic.marker)
    body = flushes_with(generic.proxy, generic.marker)[0]
    assert body.get("channel") == "acc-generic", f"channel {body.get('channel')!r}"
    assert body.get("session_id") == generic.session_id, f"session_id {body.get('session_id')!r}"
    assert body.get("project_id") == generic.project, f"project_id {body.get('project_id')!r}"
    assert body.get("user_id") == ctx.user_a, f"user_id {body.get('user_id')!r}"
    assert generic.marker in body.get("user_message", "") and generic.marker in body.get("assistant_response", "")


@proves("plugin.generic-adapter", "ext.compiler-flush")
def test_generic_turn_is_extracted_and_recallable(generic: Generic, a):
    status = generic.flush_status
    assert status.get("status") == "completed", f"the flush task ended as {status.get('status')!r}: {str(status.get('error'))[:200]}"
    assert (status.get("result") or {}).get("memories_stored", 0) >= 1, "the worker stored no memory for the turn"
    wait_recalled(a, generic.project, generic.marker)


@proves("plugin.generic-adapter")
@negative_control
def test_generic_checks_fail_for_a_turn_that_was_never_sent_and_for_another_user(ctx, generic: Generic, b):
    expect_check_fails(check_sent_exactly_once, generic.proxy, f"zq-{ctx.run_id}-never-sent")
    expect_check_fails(check_recalled, b, generic.project, generic.marker)
