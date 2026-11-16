"""SessionStart variants that need no feature flag: index vs full content, the code-graph deferral.

(The standards block and the identity cards need a flags-on stack and are proven in test_standards
and test_dreaming; the plain index and the "Previously" block are proven in test_plugin_hooks and
test_plugin_session_note.)

One LLM-free memory is written once; each scenario starts a fresh session against it through a
recording proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nsaccept import eventually, expect_check_fails, needs, negative_control, profiles, proves
from nsaccept.helpers import write_raw
from nsaccept.hooks import HookResult
from nsaccept.proxy import RecordingProxy
from plugin_kit import Session, context_of, memories_with, option

pytestmark = [needs("node"), profiles("default")]

PROBE = "/v1/code-graph/query"
DEFERRAL = "defer to the code graph"


@dataclass
class Seeded:
    project: str
    marker: str
    tail: str


@pytest.fixture(scope="module")
def seeded(ctx, a) -> Seeded:
    project = f"{ctx.project_id}-pstart"
    marker, tail = f"zq-{ctx.run_id}-ss", f"zq-{ctx.run_id}-sstail"
    w = write_raw(
        a,
        f"{marker} The harbour pilot schedule is posted every morning at the main gate and then "
        f"copied to the radio room where the duty officer reads it aloud, finally the code word is {tail}.",
        "tech_stack", marker=marker, scope="project", project_id=project, visibility="private",
    )
    w.require()
    eventually(lambda: _visible(a, project, marker), timeout=90, what="the seed memory is listed")
    return Seeded(project, marker, tail)


def _visible(a, project: str, marker: str):
    assert memories_with(a, project, marker), f"{marker} not listed yet"


def start(ctx, seeded: Seeded, proxy: RecordingProxy, **env) -> HookResult:
    s = Session(ctx, project_id=seeded.project, proxy=proxy)
    try:
        return s.session_start("startup", **env)
    finally:
        s.close()


# ── index vs full ─────────────────────────────────────────────────


def check_mode(res: HookResult, mode: str, seeded: Seeded) -> None:
    text = context_of(res)
    assert seeded.marker in text, "the seeded memory is not in the injected context at all"
    if mode == "index":
        assert "## Memory Index" in text and "## Using this index" in text, "no index table and escalation footer"
        assert "tokens vs" in text, "no savings header on the index"
        assert seeded.tail not in text, "index mode injected the memory's full content, not its title"
    else:
        assert "# Neuralscape Memory Context" in text, "no full-content header"
        assert seeded.tail in text, "full mode did not inject the memory's full content"
        assert "## Memory Index" not in text, "full mode also rendered the index table"


@proves("plugin.session-start-index")
def test_default_mode_injects_the_index_of_titles_not_the_memories(ctx, seeded):
    with RecordingProxy(ctx.base_url) as proxy:
        check_mode(start(ctx, seeded, proxy), "index", seeded)
        asked = proxy.calls(startswith=f"/v1/context/{seeded.project}", method="GET")
        assert len(asked) == 1 and asked[0].query.get("user_id") == [ctx.user_a], "the index was not built from this project's context"


@proves("plugin.session-start-full-mode")
def test_context_mode_full_injects_the_full_text_of_the_memories(ctx, seeded):
    with RecordingProxy(ctx.base_url) as proxy:
        check_mode(start(ctx, seeded, proxy, **option("CONTEXT_MODE", "full")), "full", seeded)


@proves("plugin.session-start-full-mode", "plugin.session-start-index")
@negative_control
def test_mode_check_fails_when_the_wrong_mode_is_expected(ctx, seeded):
    with RecordingProxy(ctx.base_url) as proxy:
        full = start(ctx, seeded, proxy, **option("CONTEXT_MODE", "full"))
        expect_check_fails(check_mode, full, "index", seeded)
        index = start(ctx, seeded, proxy)
        expect_check_fails(check_mode, index, "full", seeded)


# ── code-graph deferral ───────────────────────────────────────────


def check_deferral(res: HookResult, shown: bool) -> None:
    text = context_of(res)
    assert (DEFERRAL in text) == shown, f"the code-graph deferral is {'absent' if shown else 'present'}, want {'present' if shown else 'absent'}"


@proves("plugin.code-graph-deferral")
def test_code_graph_option_forces_the_deferral_on_and_off(ctx, seeded):
    with RecordingProxy(ctx.base_url) as proxy:
        check_deferral(start(ctx, seeded, proxy, **option("CODE_GRAPH", "on")), True)
        assert proxy.calls(PROBE) == [], "CODE_GRAPH=on still probed the service"
        proxy.clear()
        check_deferral(start(ctx, seeded, proxy, **option("CODE_GRAPH", "off")), False)
        assert proxy.calls(PROBE) == [], "CODE_GRAPH=off still probed the service"


@proves("plugin.code-graph-deferral")
def test_auto_probes_the_service_and_follows_its_answer(ctx, seeded):
    with RecordingProxy(ctx.base_url) as proxy:
        res = start(ctx, seeded, proxy, **option("CODE_GRAPH", "auto"))
        probes = proxy.calls(PROBE, "GET")
        assert len(probes) == 1, f"{len(probes)} code-graph probes at one session start (want 1)"
        assert probes[0].query.get("question"), "the probe carried no question"
        check_deferral(res, probes[0].status == 200)  # defers exactly when the stack says a graph resolves


@proves("plugin.code-graph-deferral")
@negative_control
def test_deferral_check_fails_when_the_option_says_the_opposite(ctx, seeded):
    with RecordingProxy(ctx.base_url) as proxy:
        expect_check_fails(check_deferral, start(ctx, seeded, proxy, **option("CODE_GRAPH", "off")), True)
        expect_check_fails(check_deferral, start(ctx, seeded, proxy, **option("CODE_GRAPH", "on")), False)
