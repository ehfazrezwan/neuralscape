"""Conversation compiler through REST: /flush, /compile, /status (LLM-backed).

A turn sent to /flush goes through the main extraction path (one LLM call, the
core category set), lands as an ordinary memory the user can recall, and is
mirrored to the vault's daily log; /compile then turns that log into articles
and runs the post-compile dedup. Assertions are on a unique marker and on
structure (categories, task state), never on model wording.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nsaccept import eventually, expect_check_fails, negative_control, proves

PREFIX = "/v1/extensions/conversation-compiler"

# The core taxonomy (schemas.CORE_MEMORY_CATEGORIES). The pre-unification flush
# vocabulary (gotcha, pattern, research, action_item, fact, project) is not in it.
CORE_CATEGORIES = {
    "preference", "personal_fact", "technical_skill", "domain_knowledge",
    "tech_stack", "convention", "architecture", "dependency",
    "decision", "interaction", "workflow", "procedure", "task_context",
}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@pytest.fixture(scope="module")
def flushed(ctx, a):
    """User A flushes one turn that states a decision with a unique marker."""
    marker = f"zq-{ctx.run_id}-flushdecision"
    body = {
        "user_message": (
            f"We need to pick a name for the new queue service. I have decided the "
            f"queue service will be called {marker}, because it must be easy to grep for."
        ),
        "assistant_response": (
            f"Understood. The decision is recorded: the new queue service is named {marker}, "
            f"chosen because the name is easy to grep for in logs."
        ),
        "session_id": f"acc-{ctx.run_id}-flush",
        "channel": "api",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "project_id": ctx.project_id,
        "user_id": a.user_id,
    }
    resp = a.post(f"{PREFIX}/flush", json=body)
    out = {"marker": marker, "http": resp.status_code, "task": {}, "hits": []}
    if resp.status_code == 202 and resp.json().get("task_id"):
        out["task"] = a.poll_task(resp.json()["task_id"], timeout=180)
        try:
            eventually(lambda: _hits(a, marker, ctx.project_id, out), timeout=90, what="flushed fact searchable")
        except AssertionError:
            pass
    return out


def _hits(client, marker, project_id, out=None):
    results = client.search(marker, project_id=project_id)
    hits = [r for r in results if marker in r.get("memory", "")]
    if out is not None:
        out["hits"] = hits
    assert hits, f"{marker} not in {len(results)} search results"


@proves("ext.compiler-flush", "queue.task-manager")
def test_flush_is_accepted_and_the_task_completes(flushed):
    assert flushed["http"] == 202, f"POST /flush -> {flushed['http']}"
    assert flushed["task"].get("status") == "completed", f"flush task ended as {flushed['task']}"
    assert not flushed["task"].get("error"), flushed["task"].get("error")


@proves("ext.compiler-flush", "write.remember-conversation")
def test_flushed_fact_is_stored_in_the_core_taxonomy(flushed):
    assert flushed["hits"], "the flushed decision was not stored or not searchable"
    for hit in flushed["hits"]:
        assert hit.get("category") in CORE_CATEGORIES, f"category {hit.get('category')!r} is outside the core taxonomy"


@proves("ext.compiler-flush", "ext.compiler-obsidian-dual-write")
def test_flush_updates_the_status_endpoint(a, flushed):
    assert flushed["task"].get("status") == "completed", "precondition: flush did not complete"
    check_status_reports_a_flush(a)
    body = a.get(f"{PREFIX}/status").json()
    assert body["daily_log_count"] >= 1, f"no daily log after a flush: {body}"


@proves("ext.compiler-compile", "queue.dedup-cron")
def test_compile_runs_and_triggers_dedup(ctx, a, flushed):
    assert flushed["task"].get("status") == "completed", "precondition: flush did not complete"
    resp = a.post(f"{PREFIX}/compile", json={"date": _today(), "user_id": a.user_id})
    assert resp.status_code == 202, f"POST /compile -> {resp.status_code}: {resp.text[:200]}"
    task = a.poll_task(resp.json()["task_id"], timeout=300)
    assert task.get("status") == "completed", f"compile task ended as {task}"
    result = task.get("result") or {}
    assert result.get("dedup_triggered") is True, f"post-compile dedup did not run: {result}"
    assert result.get("articles"), f"compile produced no articles: {result}"
    status = a.get(f"{PREFIX}/status").json()
    assert status["last_compile"], f"last_compile not recorded: {status}"


def check_status_reports_a_flush(client):
    resp = client.get(f"{PREFIX}/status")
    assert resp.status_code == 200, f"GET /status -> {resp.status_code}"
    assert resp.json()["last_flush"], "no flush recorded"


@proves("ext.compiler-flush", "ext.compiler-obsidian-dual-write")
@negative_control
def test_flush_check_fails_when_nothing_was_flushed(anon):
    # An unauthenticated caller cannot read the status at all: the check cannot pass.
    expect_check_fails(check_status_reports_a_flush, anon)
