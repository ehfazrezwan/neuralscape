"""The PreToolUse read gate: before a large file that memories already mention is Read, the hook
slips a ranked list of those memories next to the Read (it never denies and never replaces the file).

One LLM-free memory that names a file by its `dir/basename` is written once. The hook then runs on
real files in its sandbox: the gate speaks for the file the memory names, once, and stays quiet for
a small file, a binary extension, an unrelated file, a repeat Read, a disabled gate and an excluded
project. The stack serves the index rows the gate matches against, once per session and project.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nsaccept import eventually, expect_check_fails, needs, negative_control, proves
from nsaccept.helpers import write_raw
from nsaccept.hooks import HookResult
from nsaccept.proxy import RecordingProxy
from plugin_kit import Session, context_of, memories_with, option

pytestmark = needs("node")

BIG = "x = 1  # padding so that the file is larger than the gate's size floor\n" * 40  # ~2.6 kB


@dataclass
class Gate:
    project: str
    owner: Session  # holds the sandbox whose working directory contains the files
    target: object  # the file the memory names
    other: object  # a large file nothing mentions
    small: object
    image: object
    memory_id: str

    def session(self, proxy, **kw) -> Session:
        """A new Claude Code session in the same working directory (its own gate cache)."""
        return Session(self.owner.ctx, project_id=self.project, proxy=proxy, sandbox=self.owner.sandbox, **kw)


@pytest.fixture(scope="module")
def gate(ctx, a):
    project = f"{ctx.project_id}-pgate"
    stem = f"zqmod{ctx.run_id}"
    marker = f"zq-{ctx.run_id}-gate"
    owner = Session(ctx, project_id=project, proxy=None, url="http://127.0.0.1:1")  # owns the sandbox only
    folder = owner.sandbox.work / f"zqgate{ctx.run_id}"
    folder.mkdir()
    target, other = folder / f"{stem}.ts", folder / f"zqother{ctx.run_id}.ts"
    small, image = folder / f"zqsmall{ctx.run_id}.ts", folder / f"{stem}.png"
    for path in (target, other, image):
        path.write_text(BIG)
    small.write_text("tiny\n")
    w = write_raw(
        a,
        f"Fixed the retry loop in {folder.name}/{stem}.ts so queued jobs resume after a restart ({marker}).",
        "tech_stack", marker=marker, scope="project", project_id=project, visibility="private",
        observation_type="bugfix",
    )
    w.require()
    eventually(lambda: _listed(a, project, marker), timeout=90, what="the seed memory is listed")
    yield Gate(project, owner, target, other, small, image, w.memory_id)
    owner.close()


def _listed(a, project, marker):
    assert memories_with(a, project, marker), f"{marker} not listed yet"


def check_gate_steers(res: HookResult, g: Gate) -> None:
    assert res.returncode == 0, f"the gate exited {res.returncode}"
    out = (res.data or {}).get("hookSpecificOutput") or {}
    assert "permissionDecision" not in out, "the gate made a permission decision (it must only steer)"
    text = out.get("additionalContext") or ""
    assert "[Neuralscape] 1 stored memory references" in text, f"no steering context: {text[:120]!r}"
    assert f"#{g.memory_id}" in text, "the memory that names the file is not listed by its id"
    assert "get_memories" in text and "timeline" in text, "no escalation menu"


@proves("plugin.read-gate")
def test_a_large_file_that_a_memory_names_gets_that_memory_alongside_the_read(ctx, gate: Gate):
    g = gate
    with RecordingProxy(ctx.base_url) as proxy:
        res = g.session(proxy).pre_read(g.target)
        check_gate_steers(res, g)
        fetched = proxy.calls("/v1/memories", "GET")
        assert len(fetched) == 1, f"{len(fetched)} index fetches for one Read"
        q = fetched[0].query
        assert q.get("fields") == ["index"] and q.get("project_id") == [g.project], f"index fetch query {q}"


@proves("plugin.read-gate")
def test_the_gate_speaks_once_per_file_and_fetches_once_per_session(ctx, gate: Gate):
    g = gate
    with RecordingProxy(ctx.base_url) as proxy:
        s = g.session(proxy)
        check_gate_steers(s.pre_read(g.target), g)
        again = s.pre_read(g.target)
        assert context_of(again) == "", "the same file steered twice in one session"
        unrelated = s.pre_read(g.other)
        assert context_of(unrelated) == "", "a file no memory names was steered"
        assert len(proxy.calls("/v1/memories", "GET")) == 1, "the memory index was fetched more than once in one session"


@proves("plugin.read-gate")
def test_the_gate_stays_quiet_for_small_files_binaries_a_disabled_gate_and_excluded_projects(ctx, gate: Gate):
    g = gate
    with RecordingProxy(ctx.base_url) as proxy:
        for what, path, env in (
            ("a small file", g.small, {}),
            ("an image", g.image, {}),
            ("a disabled gate", g.target, option("READ_GATE_ENABLED", "false")),
            ("an excluded project", g.target, option("EXCLUDED_PROJECTS", "acc-*-PGATE")),
        ):
            res = g.session(proxy).pre_read(path, **env)
            assert res.returncode == 0 and context_of(res) == "", f"the gate steered for {what}"
        assert proxy.calls("/v1/memories", "GET") == [], "the gate fetched memories for a Read it had no reason to gate"


@proves("plugin.read-gate")
@negative_control
def test_steer_check_fails_for_a_file_no_memory_names_and_for_a_user_without_the_memory(ctx, gate: Gate):
    g = gate
    with RecordingProxy(ctx.base_url) as proxy:
        expect_check_fails(check_gate_steers, g.session(proxy).pre_read(g.other), g)
        stranger = g.session(proxy, user_id=ctx.user_b, token=ctx.token_b)
        expect_check_fails(check_gate_steers, stranger.pre_read(g.target), g)
