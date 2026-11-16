"""Write path (REST -> queue -> worker) and read path, with two-user isolation.

Assertions are on a unique marker only, never on LLM wording (this path makes
no LLM call: /v1/memories/raw stores a pre-categorised fact).
"""

from __future__ import annotations

from conftest import check_search_finds_marker
from nsaccept import eventually, expect_check_fails, negative_control, proves


@proves("write.remember-raw", "queue.task-manager")
def test_raw_write_is_queued_and_completes(a_fact):
    assert a_fact.task_id, "no task id from POST /v1/memories/raw"
    assert a_fact.status.get("status") == "completed", f"task ended as {a_fact.status}"
    assert not a_fact.status.get("error"), f"task error: {a_fact.status.get('error')}"


@proves("write.remember-raw", "read.hybrid-search-core")
def test_user_a_recalls_own_fact(a, a_fact):
    assert a_fact.status.get("status") == "completed", "write did not complete"
    eventually(lambda: check_search_finds_marker(a, a_fact), timeout=90, what="A recalls own fact")


@proves("auth.request-identity-resolution", "read.hybrid-search-core")
def test_user_b_cannot_recall_a_fact(b, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became searchable, isolation unproven"
    results = b.search(a_fact.content, project_id=a_fact.project_id)
    leaked = [r.get("id") for r in results if a_fact.marker in r.get("memory", "")]
    assert not leaked, f"user B can see A's private memory: {leaked}"


@proves("auth.request-identity-resolution", "read.hybrid-search-core")
def test_user_b_cannot_list_a_fact(b, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became searchable, isolation unproven"
    r = b.get("/v1/memories", params={"project_id": a_fact.project_id, "limit": 100})
    assert r.status_code == 200, f"GET /v1/memories -> {r.status_code}"
    assert a_fact.marker not in r.text, "user B's memory listing contains A's marker"


@proves("read.hybrid-search-core", "auth.request-identity-resolution")
@negative_control
def test_recall_check_fails_for_user_b(b, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became searchable"
    expect_check_fails(check_search_finds_marker, b, a_fact)
