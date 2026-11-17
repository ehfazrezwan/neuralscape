"""Two principals, one stack, enforced route policy (profile ``phase2``).

User A owns a private record and a queued write. The checks are the same four boundaries every time: the
request (read by id, search, change), the job status of A's write, and the export. They are run for

* an outsider: a valid token for an identity the authority store has never been told about, which proves
  enforcement is live (with policy off it would be served);
* user B: a bound member who owns nothing of A's;
* scoped credentials with a narrower scope than their owner (a later slice issues them, see below).

Credential issuance needs the ``manage_credentials`` grant, which the migration that binds the acceptance
users never gives and which only the membership slice hands out, so the credential scenarios carry
``@expected_red`` until then. They are written, not skipped, and not loosened.

Surfaces: ``GET /v1/memories``, ``GET|PATCH|DELETE /v1/memories/{id}``, ``POST /v1/search``,
``POST /v1/memories/raw``, ``GET /v1/memories/status/{id}``, ``GET /v1/export/okf``,
``POST /v1/credentials``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pytest

from nsaccept import eventually, expect_check_fails, negative_control, profiles, proves
from nsaccept.client import NSClient
from nsaccept.redmark import expected_red

from phase2_kit import (
    DENIED, RED_MEMBERSHIP, Step, bundle_text, call, check_blind_by_id, check_export_carries,
    check_recalled, client_for, memory_id_of, okf_export,
)

pytestmark = profiles("phase2")


@dataclass
class Pair:
    marker: str
    query: str
    memory_id: str = ""
    task_id: str = ""
    write: Step = field(default_factory=lambda: Step("write"))
    outsider: NSClient | None = None
    outsider_error: str = ""
    # one step per boundary and principal, recorded in the fixture
    steps: dict = field(default_factory=dict)
    b_export_text: str = ""
    b_export_error: str = ""
    credentials: dict = field(default_factory=dict)  # name -> (issue Step, client or None)


def _issue(owner: NSClient, ctx, name: str, scope: list[str]) -> tuple[Step, NSClient | None]:
    """Ask for a scoped credential; the secrets are kept out of the recorded step."""
    step = call(f"issue {name}", owner, "POST", "/v1/credentials", poll=False, json={"scope": scope, "with_refresh": False})
    body = step.body if isinstance(step.body, dict) else {}
    token = body.get("access_token") if step.status == 200 else None
    step.body = {k: v for k, v in body.items() if k not in ("access_token", "refresh_token")}
    return step, (NSClient(ctx.base_url, token, owner.user_id) if token else None)


@pytest.fixture(scope="module")
def pair(ctx, a, b) -> Pair:
    run = ctx.run_id
    nonce = uuid.uuid4().hex[:6]  # a rerun with the same NS_RUN_ID must not see leftover text
    p = Pair(marker=f"zq-{run}{nonce}-p2priv", query=f"zq-{run}{nonce}-p2priv quay crane rota")
    p.write = call("write", a, "POST", "/v1/memories/raw", json={
        "content": f"{p.marker} The quay crane rota is posted on the east gate.",
        "category": "preference", "scope": "global", "visibility": "private",
    })
    p.task_id, p.memory_id = p.write.task_id, memory_id_of(p.write.final)
    try:
        p.outsider = client_for(ctx, f"acc-{run}-outsider")
    except Exception as exc:  # noqa: BLE001 - recorded, the scenario reports it
        p.outsider_error = f"{type(exc).__name__}: {exc}"[:200]
    if not p.memory_id:
        return p  # every scenario then reports "the write did not complete"
    eventually(lambda: check_recalled(a, p.marker, p.query), timeout=90, what="A recalls its private record")

    s = p.steps
    for who, client in (("b", b), ("outsider", p.outsider)):
        if client is None:
            continue
        s[f"{who} list"] = call(f"{who} list", client, "GET", "/v1/memories", poll=False)
        s[f"{who} search"] = call(f"{who} search", client, "POST", "/v1/search", poll=False, json={"query": p.query, "limit": 20})
        s[f"{who} by id"] = call(f"{who} by id", client, "GET", f"/v1/memories/{p.memory_id}", poll=False)
        s[f"{who} status"] = call(f"{who} status", client, "GET", f"/v1/memories/status/{p.task_id}", poll=False)
        if who == "outsider":  # B is a member and may write its own pool; only the outsider's write must be refused
            s["outsider write"] = call("outsider write", client, "POST", "/v1/memories/raw", poll=False, json={
                "content": f"zq-{run}-p2probe-outsider unrelated boundary probe.", "category": "preference",
                "scope": "global", "visibility": "private"})
        s[f"{who} export"] = call(f"{who} export", client, "GET", "/v1/export/okf", poll=False)
    try:
        p.b_export_text = bundle_text(okf_export(b))
    except AssertionError as exc:
        p.b_export_error = str(exc)
    s["b patch"] = call("b patch", b, "PATCH", f"/v1/memories/{p.memory_id}", poll=False, json={"content": f"{p.marker} hijacked"})
    s["b delete"] = call("b delete", b, "DELETE", f"/v1/memories/{p.memory_id}", poll=False)

    # scoped credentials, narrower than their owners'
    p.credentials["a read"] = _issue(a, ctx, "a read", ["search", "read"])
    p.credentials["a narrow"] = _issue(a, ctx, "a narrow", ["search"])
    p.credentials["b read"] = _issue(b, ctx, "b read", ["search", "read"])
    for name, path_calls in {
        "a read": [("search", "POST", "/v1/search", {"json": {"query": p.query, "limit": 20}}),
                   ("by id", "GET", f"/v1/memories/{p.memory_id}", {}),
                   ("write", "POST", "/v1/memories/raw", {"json": {"content": f"zq-{run}-p2probe-cred unrelated boundary probe.",
                                                                    "category": "preference", "scope": "global", "visibility": "private"}}),
                   ("list", "GET", "/v1/memories", {}),
                   ("export", "GET", "/v1/export/okf", {}),
                   ("patch", "PATCH", f"/v1/memories/{p.memory_id}", {"json": {"tags": ["acc-cred"]}})],
        "a narrow": [("status", "GET", f"/v1/memories/status/{p.task_id}", {}),
                     ("by id", "GET", f"/v1/memories/{p.memory_id}", {})],
        "b read": [("by id", "GET", f"/v1/memories/{p.memory_id}", {}),
                   ("search", "POST", "/v1/search", {"json": {"query": p.query, "limit": 20}}),
                   ("status", "GET", f"/v1/memories/status/{p.task_id}", {})],
    }.items():
        client = p.credentials[name][1]
        if client is None:
            continue
        for label, method, path, kw in path_calls:
            s[f"cred {name} {label}"] = call(f"cred {name} {label}", client, method, path, poll=False, **kw)
    # retire the read credential, then use it again
    issue, client = p.credentials["a read"]
    credential_id = (issue.body or {}).get("credential_id") if isinstance(issue.body, dict) else None
    if client is not None and credential_id:
        s["revoke a read"] = call("revoke a read", a, "POST", "/v1/credentials/revoke", poll=False, json={"credential_id": credential_id})
        s["cred a read search after revoke"] = call("cred a read search after revoke", client, "POST", "/v1/search", poll=False,
                                                    json={"query": p.query, "limit": 20})
    return p


def steps(pair: Pair) -> dict:
    assert pair.memory_id, f"the setup write did not complete: {pair.write.describe()}"
    return pair.steps


def _denied(step: Step | None) -> bool:
    return step is not None and not step.error and step.status in DENIED


def _no_marker(step: Step | None, marker: str) -> bool:
    return step is not None and marker not in str(step.body)


# ── enforcement is live ──────────────────────────────────────────


def check_member_may_list(client: NSClient) -> None:
    r = client.get("/v1/memories", params={"limit": 1})
    assert r.status_code == 200, f"GET /v1/memories -> {r.status_code}"


@proves("contracts.policy", "archive.phase2-pol")
def test_an_identity_the_authority_store_does_not_know_is_denied_everywhere(pair):
    assert pair.outsider is not None, pair.outsider_error
    for boundary in ("list", "search", "by id", "status", "write", "export"):
        step = steps(pair)[f"outsider {boundary}"]
        assert _denied(step), f"outsider {boundary}: {step.describe()} (enforcement is not live, or the identity is bound)"


@proves("contracts.policy", "archive.phase2-pol")
@negative_control
def test_member_check_fails_for_the_identity_nobody_bound(a, pair):
    assert pair.outsider is not None, pair.outsider_error
    check_member_may_list(a)  # the same check passes for a bound member first
    expect_check_fails(check_member_may_list, pair.outsider)


# ── user B at the request boundary ───────────────────────────────


@proves("model.visibility-tiers")
def test_b_cannot_read_a_private_record_by_id_or_find_it_by_search_or_list(pair):
    assert pair.memory_id, pair.write.describe()
    assert _denied(steps(pair)["b by id"]), steps(pair)["b by id"].describe()
    for name in ("b search", "b list"):
        step = steps(pair)[name]
        assert step.ok and _no_marker(step, pair.marker), f"{name} exposed the marker: {step.describe()}"


@proves("model.visibility-tiers")
@negative_control
def test_blindness_check_fails_for_the_owner(a, pair):
    assert pair.memory_id, pair.write.describe()
    expect_check_fails(check_blind_by_id, a, pair.memory_id)


@proves("edit.patch-memory")
def test_b_cannot_change_or_delete_a_private_record(a, pair):
    for name in ("b patch", "b delete"):
        assert _denied(steps(pair)[name]), steps(pair)[name].describe()
    row = a.get(f"/v1/memories/{pair.memory_id}")
    assert row.status_code == 200 and pair.marker in row.json()["memory"] and "hijacked" not in row.json()["memory"], (
        f"A's record after B's attempts: {row.status_code} {row.text[:120]}"
    )


@proves("edit.patch-memory")
@negative_control
def test_change_check_fails_for_the_owner(a, pair):
    def check_change_denied(client):
        r = client.request("PATCH", f"/v1/memories/{pair.memory_id}", json={"tags": ["acc-owner-edit"]})
        assert r.status_code in DENIED, f"PATCH -> {r.status_code}"

    expect_check_fails(check_change_denied, a)


# ── user B at the job-status boundary ────────────────────────────


def check_can_poll(client: NSClient, task_id: str) -> None:
    r = client.get(f"/v1/memories/status/{task_id}")
    assert r.status_code == 200 and r.json().get("status") == "completed", f"status poll -> {r.status_code}: {r.text[:120]}"


@proves("auth.task-status-isolation", "queue.task-owner-isolation")
def test_b_cannot_poll_the_job_that_stored_a_record_and_a_can(a, pair):
    assert pair.task_id, pair.write.describe()
    assert _denied(steps(pair)["b status"]), steps(pair)["b status"].describe()
    check_can_poll(a, pair.task_id)


@proves("auth.task-status-isolation", "queue.task-owner-isolation")
@negative_control
def test_poll_check_fails_for_b(a, b, pair):
    steps(pair)
    check_can_poll(a, pair.task_id)  # the same check passes for the owner first
    expect_check_fails(check_can_poll, b, pair.task_id)


# ── user B at the export boundary ────────────────────────────────


@proves("ingest.okf-export")
def test_b_exports_only_its_own_pool_and_a_exports_the_record(a, pair):
    step = steps(pair)["b export"]
    assert step.status == 200 and not pair.b_export_error, f"B's own export should be served: {step.describe()} {pair.b_export_error}"
    check_export_carries(a, pair.marker)
    assert pair.marker not in pair.b_export_text, "B's export carries A's record"


@proves("ingest.okf-export")
@negative_control
def test_export_check_fails_for_b(a, b, pair):
    check_export_carries(a, pair.marker)  # the same check passes for the owner first
    expect_check_fails(check_export_carries, b, pair.marker)


# ── scoped credentials with a narrower scope ─────────────────────


def check_credential_issued(client: NSClient) -> None:
    r = client.post("/v1/credentials", json={"scope": ["search"], "with_refresh": False})
    assert r.status_code == 200 and str(r.json().get("access_token", "")).startswith("nsc_"), (
        f"POST /v1/credentials -> {r.status_code}: {r.text[:120]}"
    )


def _issued(pair: Pair, name: str) -> None:
    steps(pair)
    step, client = pair.credentials[name]
    assert client is not None, f"no credential: {step.describe()}"


@proves("contracts.policy", "archive.phase2-pol")
@expected_red(RED_MEMBERSHIP)
def test_a_credential_scoped_to_search_and_read_works_inside_its_scope_and_is_refused_outside_it(pair):
    _issued(pair, "a read")
    s = steps(pair)
    assert s["cred a read search"].ok and pair.marker in str(s["cred a read search"].body), s["cred a read search"].describe()
    assert s["cred a read by id"].ok, s["cred a read by id"].describe()
    for outside in ("write", "list", "patch"):
        assert _denied(s[f"cred a read {outside}"]), f"{outside} outside the scope: {s[f'cred a read {outside}'].describe()}"


@proves("contracts.policy", "archive.phase2-pol", "auth.task-status-isolation")
@expected_red(RED_MEMBERSHIP)
def test_a_credential_without_the_read_action_cannot_poll_a_job_or_read_a_record(pair):
    _issued(pair, "a narrow")
    s = steps(pair)
    assert _denied(s["cred a narrow status"]), s["cred a narrow status"].describe()
    assert _denied(s["cred a narrow by id"]), s["cred a narrow by id"].describe()


@proves("contracts.policy", "archive.phase2-pol", "ingest.okf-export")
@expected_red(RED_MEMBERSHIP)
def test_a_credential_without_the_export_action_cannot_export(pair):
    _issued(pair, "a read")
    assert _denied(steps(pair)["cred a read export"]), steps(pair)["cred a read export"].describe()


@proves("contracts.policy", "archive.phase2-pol", "model.visibility-tiers")
@expected_red(RED_MEMBERSHIP)
def test_a_credential_issued_to_b_cannot_reach_a_record_job_or_search_hit_of_a(pair):
    _issued(pair, "b read")
    s = steps(pair)
    assert _denied(s["cred b read by id"]), s["cred b read by id"].describe()
    assert _denied(s["cred b read status"]), s["cred b read status"].describe()
    assert s["cred b read search"].ok and _no_marker(s["cred b read search"], pair.marker), s["cred b read search"].describe()


@proves("contracts.policy", "archive.phase2-pol")
@expected_red(RED_MEMBERSHIP)
def test_a_revoked_credential_stops_working(pair):
    _issued(pair, "a read")
    s = steps(pair)
    assert s["revoke a read"].status == 200 and (s["revoke a read"].body or {}).get("revoked") is True, s["revoke a read"].describe()
    after = s["cred a read search after revoke"]
    assert after.status in (401, 403, 404), f"the revoked credential still works: {after.describe()}"


@proves("contracts.policy", "archive.phase2-pol")
@negative_control
@expected_red(RED_MEMBERSHIP)
def test_credential_check_fails_for_the_identity_nobody_bound(a, pair):
    assert pair.outsider is not None, pair.outsider_error
    check_credential_issued(a)  # the same check passes for a member first
    expect_check_fails(check_credential_issued, pair.outsider)
