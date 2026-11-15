"""The plugin's BUILT hook scripts, run with node exactly as the client runs them."""

from __future__ import annotations

import pytest

from conftest import Fact
from nsaccept import eventually, expect_check_fails, needs, negative_control, proves
from nsaccept.hooks import HookResult, run_hook

SESSION_START = {"hook_event_name": "SessionStart", "source": "startup", "session_id": "acc-session"}


def _run(ctx, token: str, user_id: str, project_id: str) -> HookResult:
    return run_hook(
        "session-start",
        {**SESSION_START, "session_id": f"acc-{ctx.run_id}"},
        base_url=ctx.base_url,
        token=token,
        user_id=user_id,
        project_id=project_id,
    )


def check_hook_output(res: HookResult) -> str:
    """Exit 0, JSON on stdout, additionalContext present and not the 'unreachable' degrade notice."""
    assert res.returncode == 0, f"hook exited {res.returncode}: {res.stderr[:200]}"
    assert res.data is not None, f"stdout was not a JSON object: {res.stdout[:200]!r}"
    assert res.data.get("continue") is True, f"continue={res.data.get('continue')!r}"
    out = res.data.get("hookSpecificOutput") or {}
    context = out.get("additionalContext")
    assert isinstance(context, str) and context.strip(), "no additionalContext emitted"
    assert "unreachable" not in context, f"hook degraded: {context[:160]}"
    return context


def check_context_has_marker(res: HookResult, marker: str) -> None:
    assert marker in check_hook_output(res), f"marker {marker} not in injected context"


@pytest.fixture(scope="module")
def hook_a(ctx, a_fact: Fact):
    return lambda: _run(ctx, ctx.token_a, ctx.user_a, a_fact.project_id)


@needs("node")
@proves("plugin.session-start-index")
def test_session_start_emits_context(hook_a):
    check_hook_output(hook_a())


@needs("node")
@proves("plugin.session-start-index")
def test_session_start_index_lists_users_memory(hook_a, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became visible"
    eventually(
        lambda: check_context_has_marker(hook_a(), a_fact.marker),
        timeout=60, interval=3, what="session-start index shows A's fact",
    )


@needs("node")
@proves("plugin.session-start-index")
@negative_control
def test_marker_check_fails_for_user_b(ctx, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became visible"
    res = _run(ctx, ctx.token_b, ctx.user_b, a_fact.project_id)
    expect_check_fails(check_context_has_marker, res, a_fact.marker)
