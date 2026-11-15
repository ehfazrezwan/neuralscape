"""EXCLUDED_PROJECTS: a project that matches the glob is never captured from or injected into.

One full session is driven through EVERY hook (session start, prompt, tool call, Stop, PreCompact,
read, SessionEnd) for an excluded project and for its twin that is not excluded. The recording
proxy forwards to the stack, so what the twin sends is real; for the excluded project it must have
seen nothing at all, and nothing may be on disk.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nsaccept import expect_check_fails, needs, negative_control, proves
from nsaccept.proxy import LLM_PATHS, RecordingProxy
from plugin_kit import Session, context_of, option

pytestmark = needs("node")

BIG = "y = 2  # padding so that the file is larger than the read gate's size floor\n" * 40


@dataclass
class Drive:
    s: Session
    proxy: RecordingProxy
    context: str


def drive_full_session(ctx, project: str, **env) -> Drive:
    proxy = RecordingProxy(ctx.base_url, stub=LLM_PATHS).start()  # LLM routes answered locally, the rest is real
    s = Session(ctx, project_id=project, proxy=proxy, extra_env=env)
    big = s.sandbox.work / "pier.ts"
    big.write_text(BIG)
    start = s.session_start()
    s.user_prompt("Look at the pier schedule.")
    s.bash("make build", "built")
    s.transcript.turn("Look at the pier schedule.", "Looked at it, with enough words to be a real answer.")
    s.stop("Looked at it, with enough words to be a real answer.")
    s.pre_compact()
    s.pre_read(big)
    s.session_end()
    return Drive(s, proxy, context_of(start))


def check_nothing_leaves(d: Drive) -> None:
    sent = [f"{c.method} {c.path}" for c in d.proxy.requests]
    assert not sent, f"{len(sent)} requests reached the service: {sent[:6]}"
    assert d.context == "", f"memory context was injected: {d.context[:80]!r}"
    assert not d.s.buffer_path.exists(), "an observation buffer was written"
    assert d.s.offset() is None, "a transcript cursor was written"
    leftovers = [p.name for p in d.s.sandbox.observations.glob("*")] if d.s.sandbox.observations.exists() else []
    assert not leftovers, f"files were written to the plugin data directory: {leftovers}"


@pytest.fixture(scope="module")
def excluded(ctx):
    d = drive_full_session(ctx, f"{ctx.project_id}-pexcl", **option("EXCLUDED_PROJECTS", "ACC-*-PEXCL"))
    yield d
    d.proxy.stop()
    d.s.close()


@pytest.fixture(scope="module")
def twin(ctx):
    d = drive_full_session(ctx, f"{ctx.project_id}-pincl", **option("EXCLUDED_PROJECTS", "ACC-*-PEXCL"))
    yield d
    d.proxy.stop()
    d.s.close()


@proves("plugin.excluded-projects")
def test_an_excluded_project_sends_nothing_injects_nothing_and_writes_nothing(excluded: Drive):
    check_nothing_leaves(excluded)  # also: case-insensitive glob, `*` across the run id


@proves("plugin.excluded-projects")
def test_the_same_session_in_a_project_that_does_not_match_is_captured(twin: Drive):
    paths = {c.path for c in twin.proxy.requests}
    for expected in ("/v1/memories/raw", "/v1/checkpoint", "/v1/extensions/conversation-compiler/flush"):
        assert expected in paths, f"{expected} was never called: the control session did not exercise the hook it controls"
    assert any(p.startswith("/v1/context/") for p in paths), "the control SessionStart did not fetch context"
    assert twin.s.buffer_path.is_file(), "the control session buffered no observation"


@proves("plugin.excluded-projects")
@negative_control
def test_nothing_leaves_check_fails_for_the_twin(twin: Drive):
    expect_check_fails(check_nothing_leaves, twin)


@proves("plugin.excluded-projects")
def test_the_ns_prefixed_alias_excludes_too(ctx):
    project = f"{ctx.project_id}-palias"
    kw = dict(project_id=project, proxy=None, url="http://127.0.0.1:1")
    excluded = Session(ctx, extra_env={"NS_EXCLUDED_PROJECTS": "*-palias"}, **kw)
    control = Session(ctx, **kw)  # same project, no exclusion: the buffer is written
    try:
        control.bash("make build", "built")
        assert control.buffer_path.is_file(), "precondition: without the alias the observation is buffered"
        excluded.bash("make build", "built")
        assert not excluded.buffer_path.exists(), "NS_EXCLUDED_PROJECTS did not stop the observation buffer"
    finally:
        excluded.close()
        control.close()
