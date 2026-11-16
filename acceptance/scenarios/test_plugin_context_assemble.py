"""``POST /v1/context/assemble`` through the plugin's BUILT SessionStart hook (CONTEXT_MODE=assemble).

The hook is run against the live stack with the setting on and a query that
matches a memory the scenario stored first (the shared ``a_fact``): the
injected text must carry the assembled bundle with that memory's row, within
the budget the bundle reports. With the setting off the hook injects the
index and no assembled section. Assertions are on a unique marker and on the
route's own accounting, never on wording.
"""

from __future__ import annotations

import re

import pytest

from conftest import Fact
from nsaccept import eventually, expect_check_fails, needs, negative_control, proves
from nsaccept.hooks import HookResult, run_hook

BUDGET = 1200
HEADING = re.compile(r"## Session context \(assembled, ~(\d+) of (\d+) tokens\)")


def _start(ctx, token: str, user_id: str, project_id: str, fact_marker: str | None, *, assemble: bool) -> HookResult:
    extra = {}
    if assemble:
        extra = {
            "CLAUDE_PLUGIN_OPTION_CONTEXT_MODE": "assemble",
            "CLAUDE_PLUGIN_OPTION_ASSEMBLE_BUDGET_TOKENS": str(BUDGET),
        }
        if fact_marker:
            extra["CLAUDE_PLUGIN_OPTION_ASSEMBLE_QUERY"] = fact_marker
    return run_hook(
        "session-start",
        {"hook_event_name": "SessionStart", "source": "startup", "session_id": f"acc-{ctx.run_id}-assemble"},
        base_url=ctx.base_url,
        token=token,
        user_id=user_id,
        project_id=project_id,
        extra_env=extra,
    )


def _context(res: HookResult) -> str:
    assert res.returncode == 0, f"hook exited {res.returncode}: {res.stderr[:200]}"
    assert res.data is not None, f"stdout was not a JSON object: {res.stdout[:200]!r}"
    context = (res.data.get("hookSpecificOutput") or {}).get("additionalContext")
    assert isinstance(context, str) and context.strip(), "no additionalContext emitted"
    assert "unreachable" not in context, f"hook degraded: {context[:160]}"
    return context


def check_assembled_bundle_has_marker(res: HookResult, marker: str) -> tuple[int, int]:
    """The assembled section is present, names the marker, and is within its own budget."""
    context = _context(res)
    m = HEADING.search(context)
    assert m, "no assembled section in the injected context (route failed or setting not honoured)"
    section = context[m.end():].split("\n---\n", 1)[0]
    assert marker in section, f"marker {marker} not in the assembled section"
    used, budget = int(m.group(1)), int(m.group(2))
    assert budget == BUDGET, f"the route clamped the budget to {budget}, not {BUDGET}"
    assert used <= budget, f"used_tokens {used} exceeds budget_tokens {budget}"
    return used, budget


@pytest.fixture(scope="module")
def assembled(ctx, a, a_fact: Fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became visible"
    holder: dict = {}

    def run():
        res = _start(ctx, ctx.token_a, ctx.user_a, a_fact.project_id, a_fact.marker, assemble=True)
        holder["res"] = res
        holder["used_budget"] = check_assembled_bundle_has_marker(res, a_fact.marker)

    eventually(run, timeout=90, interval=3, what="assembled bundle carries the stored memory")
    return holder


@needs("node")
@proves("read.context-assemble", "plugin.session-start-index")
def test_assembled_bundle_is_injected_with_the_stored_memory(assembled):
    used, budget = assembled["used_budget"]
    assert 0 < used <= budget


@needs("node")
@proves("read.context-assemble")
def test_route_accounting_matches_what_the_hook_injected(ctx, a, a_fact, assembled):
    """The route itself, called directly: same budget bound, plain bundle, the marker row, per-section accounting."""
    resp = a.post(
        "/v1/context/assemble",
        json={"budget_tokens": BUDGET, "project_id": a_fact.project_id, "query": a_fact.marker, "format": "plain"},
    )
    assert resp.status_code == 200, f"POST /v1/context/assemble -> {resp.status_code}: {resp.text[:200]}"
    body = resp.json()
    assert body["used_tokens"] <= body["budget_tokens"] == BUDGET
    assert a_fact.marker in body["bundle"]["text"] or body["sections"]["index_rows"] >= 1
    assert body["sections"]["index_rows"] >= 1, f"no relevant-memory row for the marker: {body['sections']}"


@needs("node")
@proves("read.context-assemble")
def test_setting_off_injects_the_index_and_no_assembled_section(ctx, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became visible"
    context = _context(_start(ctx, ctx.token_a, ctx.user_a, a_fact.project_id, None, assemble=False))
    assert not HEADING.search(context), "assembled section injected with the setting off"


@needs("node")
@proves("read.context-assemble")
@negative_control
def test_bundle_check_fails_for_user_b(ctx, a_fact, assembled):
    # B cannot see A's private memory: the same check, run for B, must fail.
    res = _start(ctx, ctx.token_b, ctx.user_b, a_fact.project_id, a_fact.marker, assemble=True)
    expect_check_fails(check_assembled_bundle_has_marker, res, a_fact.marker)
