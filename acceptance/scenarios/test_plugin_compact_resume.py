"""PreCompact snapshot -> stored memory -> "Resuming after compaction" at the next SessionStart.

Two sessions compact in the same project (one marker per prompt). The snapshot of one must come
back for that session only, from the stack (fresh plugin data, as after a restart) and from the
plugin's local copy when the stack is down. Everything is LLM-free (`POST /v1/memories/raw`); the
conversation-compiler routes are answered by the proxy and the scenario asserts the PreCompact hook
does not deliver turns itself (the Stop hook owns delivery).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from nsaccept import eventually, expect_check_fails, needs, negative_control, proves
from nsaccept.hooks import HookResult
from nsaccept.proxy import FLUSH, RecordingProxy
from plugin_kit import Session, check_never_stored, context_of, memories_with, task_completed

pytestmark = needs("node")

RAW = "/v1/memories/raw"
DEAD_URL = "http://127.0.0.1:1"


@dataclass
class Compacted:
    s: Session
    proxy: RecordingProxy
    other: Session
    other_proxy: RecordingProxy
    project: str
    m: dict


def compact_session(ctx, a, project: str, proxy: RecordingProxy, prompts: list[str]) -> Session:
    s = Session(ctx, project_id=project, proxy=proxy)
    for i, text in enumerate(prompts):
        s.transcript.turn(text, f"Understood, working on step {i} now, with enough words to be an answer.")
    res = s.pre_compact("manual")
    assert res.returncode == 0, f"pre-compact exited {res.returncode}"
    raw = proxy.calls(RAW, "POST")
    if raw and raw[0].task_id:
        a.poll_task(raw[0].task_id, timeout=120)
    return s


@pytest.fixture(scope="module")
def compacted(ctx, a):
    run = ctx.run_id
    project = f"{ctx.project_id}-pcmp"
    m = {k: f"zq-{run}-{k}" for k in ("p0", "p1", "p2", "p3", "priv", "o1", "o2")}
    proxy, other_proxy = RecordingProxy(ctx.base_url).start(), RecordingProxy(ctx.base_url).start()
    prompts = [
        f"First we looked at {m['p0']}.",
        f"Then we moved on to {m['p1']}.",
        f"After that came {m['p2']}.",
        f"Now the open task is {m['p3']} <private>{m['priv']}</private> and it is not finished.",
    ]
    s = compact_session(ctx, a, project, proxy, prompts)
    other = compact_session(ctx, a, project, other_proxy, [f"Another terminal works on {m['o1']}.", f"And then on {m['o2']}."])
    yield Compacted(s, proxy, other, other_proxy, project, m)
    for p in (proxy, other_proxy):
        p.stop()
    s.close()
    other.close()


# ── the PreCompact hook ───────────────────────────────────────────


@proves("plugin.precompact-snapshot")
def test_precompact_posts_one_tagged_snapshot_of_the_last_prompts(compacted: Compacted):
    c = compacted
    posts = c.proxy.calls(RAW, "POST")
    assert len(posts) == 1, f"{len(posts)} snapshot writes from one PreCompact (want 1)"
    body = posts[0].body
    assert body.get("category") == "task_context", f"category {body.get('category')!r}"
    assert body.get("scope") == "project" and body.get("project_id") == c.project, f"scope/project {body.get('scope')!r}/{body.get('project_id')!r}"
    assert body.get("user_id") == c.s.user_id, f"user {body.get('user_id')!r}"
    tags = body.get("tags") or []
    assert "compact_snapshot" in tags and f"session:{c.s.session_id}" in tags, f"tags {tags}"
    expires = datetime.fromisoformat(body["expires_at"].replace("Z", "+00:00"))
    ahead = expires - datetime.now(timezone.utc)
    assert timedelta(days=13) < ahead < timedelta(days=15), f"the snapshot expires in {ahead}, not in about 14 days"
    content = body.get("content", "")
    assert content.startswith("Compact snapshot:"), f"content starts {content[:40]!r}"
    for kept in ("p1", "p2", "p3"):
        assert c.m[kept] in content, f"the last prompts are in the snapshot, {kept} is not"
    assert c.m["p0"] not in content, "the oldest prompt is in a snapshot that keeps only the last three"
    assert c.m["priv"] not in posts[0].raw_body, "a <private> span left the machine in the snapshot"
    assert c.proxy.calls(FLUSH, "POST") == [], "PreCompact delivered conversation turns (the Stop hook owns delivery)"


@proves("plugin.precompact-snapshot", "plugin.private-redaction")
def test_the_snapshot_is_stored_and_the_private_span_never_is(compacted: Compacted, a):
    c = compacted
    task_completed(a, c.proxy.calls(RAW, "POST")[0].task_id, timeout=120)
    rows = eventually(
        lambda: _rows_or_fail(a, c.project, c.m["p3"]), timeout=60, interval=2, what="the snapshot memory is listed"
    )
    row = rows[0]
    assert row.get("category") == "task_context", f"category {row.get('category')!r}"
    assert "compact_snapshot" in (row.get("tags") or []), f"tags {row.get('tags')}"
    check_never_stored(a, c.project, c.m["priv"])


def _rows_or_fail(a, project, marker):
    rows = memories_with(a, project, marker)
    assert rows, f"no memory of {project} carries {marker} yet"
    return rows


@proves("plugin.precompact-snapshot")
@negative_control
def test_snapshot_listing_check_fails_for_another_user(compacted: Compacted, b):
    expect_check_fails(_rows_or_fail, b, compacted.project, compacted.m["p3"])


# ── the SessionStart that follows ─────────────────────────────────


def check_resumes(res: HookResult, marker: str, *, absent: tuple[str, ...] = ()) -> None:
    text = context_of(res)
    assert "## Resuming after compaction" in text, "no 'Resuming after compaction' block was injected"
    assert marker in text, f"{marker} (the last prompt before the compaction) is not in the injected context"
    assert text.count(marker) == 1, f"{marker} appears {text.count(marker)} times (the snapshot must not also be indexed)"
    for other in absent:
        assert other not in text, f"{other} belongs to another session's snapshot"


@proves("plugin.resume-after-compact", "plugin.precompact-snapshot")
def test_after_a_compaction_the_stored_snapshot_comes_back_for_its_own_session_only(ctx, compacted: Compacted, a):
    c = compacted
    task_completed(a, c.other_proxy.calls(RAW, "POST")[0].task_id, timeout=120)
    proxy = RecordingProxy(ctx.base_url).start()
    # fresh plugin data and the same session id: what a restarted client sees
    fresh = Session(ctx, project_id=c.project, proxy=proxy, session_id=c.s.session_id)
    try:
        eventually(
            lambda: check_resumes(fresh.session_start("compact"), c.m["p3"], absent=(c.m["o1"], c.m["o2"], c.m["p0"])),
            timeout=60, interval=3, what="the resume block is served from the stored snapshot",
        )
    finally:
        proxy.stop()
        fresh.close()


@proves("plugin.resume-after-compact")
def test_the_local_copy_resumes_the_session_when_the_service_is_down(ctx, compacted: Compacted):
    c = compacted
    down = Session(ctx, project_id=c.project, url=DEAD_URL, session_id=c.s.session_id, sandbox=c.s.sandbox)
    res = down.session_start("compact")
    check_resumes(res, c.m["p3"], absent=(c.m["o1"], c.m["p0"]))
    assert "unreachable" in context_of(res), "the outage was not announced next to the resumed context"


@proves("plugin.resume-after-compact")
@negative_control
def test_resume_check_fails_without_a_compaction_and_for_a_session_that_never_compacted(ctx, compacted: Compacted):
    c = compacted
    proxy = RecordingProxy(ctx.base_url).start()
    try:
        same_session = Session(ctx, project_id=c.project, proxy=proxy, session_id=c.s.session_id)
        expect_check_fails(check_resumes, same_session.session_start("startup"), c.m["p3"])
        stranger = Session(ctx, project_id=c.project, proxy=proxy)
        expect_check_fails(check_resumes, stranger.session_start("compact"), c.m["p3"])
        same_session.close()
        stranger.close()
    finally:
        proxy.stop()
