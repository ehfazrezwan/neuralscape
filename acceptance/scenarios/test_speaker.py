"""Speaker-aware extraction with EXTRACTION_REQUIRE_SPEAKER on (profile ``flags-on``).

Two real LLM extractions, each with unique markers: a conversation whose turns come from two named
speakers, and an ordinary single-user conversation. With the flag on, the extraction prompt must
attribute every fact to its speaker (the speaker travels as ``metadata.speaker``); a single
anonymous user is labelled ``user``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nsaccept import expect_check_fails, negative_control, precondition, profiles, proves
from nsaccept.client import NSClient

pytestmark = profiles("flags-on")

EXTRACT_WAIT = 240


def memories_of(status: dict) -> list[dict]:
    result = status.get("result") or {}
    return [m for m in (result.get("memories") if isinstance(result, dict) else None) or [] if isinstance(m, dict)]


def check_attributed(memories: list[dict], marker: str, speaker: str) -> dict:
    mine = [m for m in memories if marker in (m.get("memory") or "")]
    assert mine, f"no extracted fact carries {marker} among {len(memories)} facts: {[m.get('memory', '')[:50] for m in memories]}"
    wrong = [m for m in mine if m.get("speaker") != speaker]
    assert not wrong, f"fact(s) with {marker} attributed to {[m.get('speaker') for m in wrong]}, not {speaker!r}"
    return mine[0]


@dataclass
class Convo:
    marker_maren: str
    marker_tobias: str
    marker_solo: str
    named: list[dict] = field(default_factory=list)
    solo: list[dict] = field(default_factory=list)
    error: str = ""


def _extract(a: NSClient, messages: list[dict], project_id: str, run_id: str) -> list[dict]:
    r = a.post("/v1/memories", json={"messages": messages, "project_id": project_id, "run_id": run_id})
    assert r.status_code == 202, f"POST /v1/memories -> {r.status_code}: {r.text[:200]}"
    st = a.poll_task(r.json()["task_id"], timeout=EXTRACT_WAIT)
    assert st.get("status") == "completed", f"extraction ended as {st}"
    return memories_of(st)


@pytest.fixture(scope="module")
def convo(ctx, a) -> Convo:
    run = ctx.run_id
    c = Convo(f"zq-{run}-maren", f"zq-{run}-tobias", f"zq-{run}-solo")
    try:
        c.named = _extract(a, [
            {"role": "Maren", "content": f"I lead the {c.marker_maren} mooring crew, and my favourite knot is the bowline."},
            {"role": "Tobias", "content": f"My boat is named {c.marker_tobias} and I keep it at pier nine."},
            {"role": "Maren", "content": f"The {c.marker_maren} crew meets every Thursday at the pilot hut."},
            {"role": "Tobias", "content": f"I always refuel {c.marker_tobias} on Sunday evenings."},
        ], ctx.project_id, f"acc-{run}-speakers")
        c.solo = _extract(a, [
            {"role": "user", "content": f"Please remember that my build tool is called {c.marker_solo} and I run it before every push."},
            {"role": "assistant", "content": f"Noted: your build tool is {c.marker_solo}."},
        ], ctx.project_id, f"acc-{run}-solo")
    except AssertionError as exc:
        c.error = str(exc)
    return c


def _ok(c: Convo) -> Convo:
    assert not c.error, c.error
    return c


@proves("write.speaker-aware-extraction")
def test_facts_from_named_speakers_are_attributed_to_the_right_speaker(convo: Convo):
    c = _ok(convo)
    check_attributed(c.named, c.marker_maren, "Maren")
    check_attributed(c.named, c.marker_tobias, "Tobias")


@proves("write.speaker-aware-extraction")
def test_a_single_anonymous_user_is_labelled_user(convo: Convo):
    c = _ok(convo)
    check_attributed(c.solo, c.marker_solo, "user")


@proves("write.speaker-aware-extraction")
@negative_control
def test_attribution_check_fails_for_the_other_speaker_and_for_an_unknown_marker(convo: Convo):
    c = _ok(convo)
    precondition(check_attributed, c.named, c.marker_maren, "Maren", what="Maren's fact is attributed to Maren")
    expect_check_fails(check_attributed, c.named, c.marker_maren, "Tobias")
    expect_check_fails(check_attributed, c.named, f"{c.marker_maren}-never", "Maren")
