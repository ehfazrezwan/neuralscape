"""Dreaming with the flag on, through REST only (profile ``dreaming``).

The admin user (the only identity allowed to POST /run) is a fixed id the
profile lists in DICTATOR_USER_IDS; its token is minted here from the same
source the run's other tokens use. Data is seeded via /v1/memories/raw, a
forced sweep is enqueued and polled, then the results are read back through
status, card, search. LLM wording is never asserted; a missing LLM decision
fails with a message saying so.
"""

from __future__ import annotations

import time

import pytest

import os

from nsaccept import eventually, expect_check_fails, negative_control, profiles, proves, tokens
from nsaccept.client import NSClient

ADMIN = "acc-dreaming-admin"
POOL = f"user--{ADMIN}"
PAIR_QUERY = "Python 3.12 backend development"


@pytest.fixture(scope="module")
def admin(ctx):
    src = ctx.token_source
    if src.startswith("compose:"):
        token = tokens.mint_via_compose(src.split(":", 1)[1], ADMIN, os.environ.get("NS_API_SERVICE", "neuralscape"))
    elif src == "NS_TOKEN_SECRET":
        token = tokens.mint_local(ADMIN, os.environ["NS_TOKEN_SECRET"])
    else:
        pytest.fail("dreaming scenario needs --compose-project or NS_TOKEN_SECRET to mint the admin token")
    c = NSClient(ctx.base_url, token, ADMIN, timeout=60.0)
    yield c
    c.close()


@pytest.fixture(scope="module")
def swept(ctx, admin):
    token = f"zq{ctx.run_id}kw"
    rows = [
        ("preference", f"User prefers Python 3.12 for backend development, especially with FastAPI and the {token} toolchain"),
        ("preference", "The user likes to use Python 3.12 for backend work"),
        ("task_context", "ok got it"),
        ("decision", "The demo service is deployed as a single-node setup only"),
        ("decision", "As of late June 2026, the demo service is deployed on a managed orchestration cluster via IaC tooling"),
        ("workflow", "After restarting the service, the user runs the full MCP tool suite as a smoke test"),
        ("workflow", "The user verifies async memory writes by recalling them later instead of blocking on the write"),
        ("workflow", "Before merging a PR, the user re-runs the full unit test suite locally"),
    ]
    task_ids = [admin.store_raw(text, cat, scope="global") for cat, text in rows]
    statuses = [admin.poll_task(t, timeout=120) for t in task_ids]
    out = {"token": token, "write_ok": all(s.get("status") == "completed" for s in statuses)}
    before = (admin.get("/v1/extensions/dreaming/status").json().get("last_run") or {}).get("run_id")
    r = admin.post("/v1/extensions/dreaming/run", json={"force": True, "pool": POOL})
    out["run_status_code"], out["run_body"] = r.status_code, r.text[:200]
    if r.status_code == 202:
        deadline = time.monotonic() + 420
        while time.monotonic() < deadline:
            last = admin.get("/v1/extensions/dreaming/status").json().get("last_run") or {}
            if last.get("run_id") and last.get("run_id") != before and last.get("finished_at"):
                out["last_run"] = last
                break
            time.sleep(3)
    return out


def _pool_report(swept) -> dict:
    last = swept.get("last_run") or {}
    return next((p for p in last.get("pools", []) if p["pool"] == POOL), {})


@profiles("dreaming")
@proves("ext.dreaming-sweep")
def test_non_admin_cannot_trigger_a_sweep(b):
    r = b.post("/v1/extensions/dreaming/run", json={"dry_run": True})
    assert r.status_code == 403, f"non-admin POST /run -> {r.status_code}: {r.text[:150]}"


@profiles("dreaming")
@proves("ext.dreaming-sweep", "ext.dreaming-deep-consolidate")
def test_admin_sweep_runs_and_applies(swept):
    assert swept["write_ok"], "seed writes did not all complete"
    assert swept["run_status_code"] == 202, f"POST /run -> {swept['run_status_code']}: {swept['run_body']}"
    rep = _pool_report(swept)
    assert rep, f"no DreamRun report for {POOL}: {swept.get('last_run')}"
    assert rep["status"] == "dreamt", f"{rep['status']}: {rep.get('reason')} {rep.get('errors')}"
    assert rep["applied"] > 0, "the LLM applied no consolidation action on a pathological seed"
    assert not rep["errors"], rep["errors"]


@profiles("dreaming")
@proves("ext.dreaming-deep-consolidate")
def test_recall_has_no_tombstoned_duplicate(admin, swept):
    assert _pool_report(swept).get("applied", 0) > 0, "precondition: sweep applied nothing"
    results = admin.search(PAIR_QUERY, limit=20)
    pair = [r for r in results if "python 3.12" in r.get("memory", "").lower() and "backend" in r.get("memory", "").lower()]
    assert len(pair) == 1, (
        f"expected the duplicate pair merged to one live row, recall has {len(pair)} "
        "(LLM produced no merge, or tombstones leak into recall)"
    )


@profiles("dreaming")
@proves("ext.dreaming-deep-consolidate")
def test_keyword_search_still_finds_the_merged_row(admin, swept):
    token = swept["token"]
    eventually(
        lambda: _assert_found(admin, token),
        timeout=60, what="keyword-only search finds the merged/rewritten row",
    )


def _assert_found(admin, token):
    results = admin.search(token, limit=3)
    assert any(token in r.get("memory", "") for r in results), (
        f"rare keyword {token} not in top-3: rewritten row lost its lexical leg?"
    )


@profiles("dreaming")
@proves("ext.dreaming-identity-card")
def test_identity_card_readable_by_owner_and_refused_to_others(admin, b, swept):
    assert _pool_report(swept).get("status") == "dreamt", "precondition: sweep did not complete"
    r = admin.get("/v1/extensions/dreaming/card", params={"pool": POOL})
    assert r.status_code == 200, f"GET /card -> {r.status_code}: {r.text[:150]}"
    assert r.json()["lines"], "empty card"
    other = b.get("/v1/extensions/dreaming/card", params={"pool": POOL})
    assert other.status_code == 403, f"another user's private card -> {other.status_code}"


@profiles("dreaming")
@proves("ext.dreaming-rem-reflections", "ext.dreaming-librarian-vault")
def test_reflection_and_vault_pages_reported(swept):
    rep = _pool_report(swept)
    assert rep.get("insights", 0) > 0, "LLM produced no reflection insight"
    assert rep.get("pages_written", 0) > 0, "librarian wrote no vault page"
    assert rep.get("diary_path"), "no dream diary written"


@profiles("dreaming")
@proves("ext.dreaming-deep-consolidate")
@negative_control
def test_keyword_check_fails_for_another_user(b, swept):
    assert _pool_report(swept).get("status") == "dreamt", "precondition: sweep did not complete"
    expect_check_fails(_assert_found, b, swept["token"])
