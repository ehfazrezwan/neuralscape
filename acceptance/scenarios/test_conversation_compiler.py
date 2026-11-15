"""Conversation compiler through REST: /flush, /compile, /status (LLM-backed).

A turn sent to /flush goes through the main extraction path (one LLM call, the
core category set), lands as an ordinary memory the user can recall, and is
mirrored to the vault's daily log; /compile then turns that log into articles
and runs the post-compile dedup. Assertions are on a unique marker and on
structure (categories, task state), never on model wording.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from nsaccept import eventually, expect_check_fails, negative_control, proves, profiles

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


def _marker_hits(client, marker, project_id):
    results = client.search(marker, project_id=project_id, limit=20)
    return [r for r in results if marker in r.get("memory", "")]


def check_both_copies_recalled(client, marker, project_id):
    assert len(_marker_hits(client, marker, project_id)) == 2, "both copies not searchable yet"


def check_pair_collapsed_to_the_newer(client, marker, project_id):
    """Exactly one of the near-duplicate pair is recalled, and it is the newer one."""
    hits = _marker_hits(client, marker, project_id)
    assert len(hits) == 1, f"expected one recalled copy, got {len(hits)}"
    assert hits[0]["memory"].endswith("."), "the survivor should be the newer (second) write"


def _compile_with_a_near_duplicate_pair(ctx, a, suffix):
    """Store two near-identical facts (older, newer), compile, return (marker, older, newer)."""
    marker = f"zq-{ctx.run_id}-{suffix}"
    older = f"{marker} The nightly report job runs at 02:00 UTC on weekdays"
    newer = f"{older}."
    for text in (older, newer):
        task = a.store_raw(text, "convention", scope="project", project_id=ctx.project_id, visibility="private")
        assert a.poll_task(task, timeout=120)["status"] == "completed"
        time.sleep(1.1)  # distinct created_at, so "older" is well defined
    eventually(lambda: check_both_copies_recalled(a, marker, ctx.project_id), timeout=90, what="both copies recalled")

    resp = a.post(f"{PREFIX}/compile", json={"date": _today(), "user_id": a.user_id})
    assert resp.status_code == 202, f"POST /compile -> {resp.status_code}"
    task = a.poll_task(resp.json()["task_id"], timeout=300)
    assert task.get("status") == "completed", f"compile task ended as {task}"
    assert (task.get("result") or {}).get("dedup_triggered") is True
    eventually(lambda: check_pair_collapsed_to_the_newer(a, marker, ctx.project_id), timeout=60, what="older copy hidden")
    return marker, older, newer


@proves("ext.compiler-compile", "queue.dedup-cron")
def test_compile_dedup_hides_the_older_near_duplicate(ctx, a, flushed):
    """Post-compile dedup leaves only the newer of two near-identical facts recalled."""
    assert flushed["task"].get("status") == "completed", "precondition: flush did not complete"
    _compile_with_a_near_duplicate_pair(ctx, a, "nightlyreport")


@proves("ext.compiler-compile", "queue.dedup-cron")
@profiles("dedup-tombstone")
def test_compile_dedup_tombstone_is_reversible(ctx, a, flushed):
    """With the tombstone flag on, the older copy is hidden, not deleted: the audit listing keeps it."""
    assert flushed["task"].get("status") == "completed", "precondition: flush did not complete"
    marker, older, newer = _compile_with_a_near_duplicate_pair(ctx, a, "nightlytomb")
    plain = a.get("/v1/memories", params={"project_id": ctx.project_id, "limit": 200})
    audit = a.get("/v1/memories", params={"project_id": ctx.project_id, "limit": 200, "include_tombstoned": "true"})
    assert plain.status_code == 200 and audit.status_code == 200
    plain_n = sum(marker in m.get("memory", "") for m in plain.json())
    audit_n = sum(marker in m.get("memory", "") for m in audit.json())
    assert audit_n == 2, f"the older copy was deleted, not tombstoned: {audit_n} rows"
    assert plain_n == 1, f"the plain listing should show one copy, got {plain_n}"


@proves("ext.compiler-compile", "queue.dedup-cron")
@negative_control
def test_dedup_check_fails_for_a_user_without_the_pair(ctx, b, flushed):
    expect_check_fails(check_pair_collapsed_to_the_newer, b, f"zq-{ctx.run_id}-nightlyreport", ctx.project_id)


def check_status_reports_a_flush(client):
    resp = client.get(f"{PREFIX}/status")
    assert resp.status_code == 200, f"GET /status -> {resp.status_code}"
    assert resp.json()["last_flush"], "no flush recorded"


@proves("ext.compiler-flush", "ext.compiler-obsidian-dual-write")
@negative_control
def test_flush_check_fails_when_nothing_was_flushed(anon):
    # An unauthenticated caller cannot read the status at all: the check cannot pass.
    expect_check_fails(check_status_reports_a_flush, anon)
