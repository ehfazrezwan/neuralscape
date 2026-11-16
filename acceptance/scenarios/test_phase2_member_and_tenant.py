"""The member journey and two real tenants (profile ``phase2``; gaps O-18 and E-06).

Both are written against behaviour that later Phase 2 slices deliver, so every positive scenario here
carries ``@expected_red`` with its slice. They run and fail honestly today; nothing is skipped.

Member journey (gap O-18). The tenant's first administrator is created by an operator (the gate binds
acceptance user A as that administrator before the run). A invites a new person, the person joins with the
single-use invite, A sees them in the member list, and the new member's ordinary capture stays private to
them: either refused or stored private, never team-visible.

Two real tenants (gap E-06). Two bound people of DIFFERENT tenants, each with a real client (REST and MCP),
see nothing of each other's records, jobs or exports, and a record shared inside one tenant is not shared
into the other. The second tenant's token comes from the gate as ``NS_TOKEN_T2`` / ``NS_USER_T2`` once
tenant cells exist; without them the scenarios fail with that message.

Surfaces: ``POST /v1/members/invites``, ``POST /v1/members/join``, ``GET /v1/members``,
``POST /v1/memories/raw``, ``POST /v1/search``, ``GET /v1/memories/status/{id}``, ``GET /v1/export/okf``,
MCP ``recall_memories``.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field

import pytest

from nsaccept import eventually, expect_check_fails, negative_control, profiles, proves
from nsaccept.client import NSClient
from nsaccept.helpers import mcp_json
from nsaccept.mcp import MCPClient
from nsaccept.redmark import expected_red

from phase2_kit import (
    DENIED, RED_MEMBERSHIP, RED_TENANT_CELLS, Step, bundle_text, call, check_recalled,
    client_for, memory_id_of, okf_export,
)

pytestmark = profiles("phase2")


# ── member journey (O-18) ────────────────────────────────────────


@dataclass
class Journey:
    marker: str
    query: str
    invite: Step = field(default_factory=lambda: Step("invite"))
    member: NSClient | None = None
    member_error: str = ""
    join: Step = field(default_factory=lambda: Step("join"))
    listing: Step = field(default_factory=lambda: Step("members"))
    capture: Step = field(default_factory=lambda: Step("capture"))
    capture_id: str = ""
    member_recalls_capture: str = ""  # "" = the member recalls their own capture, else why not
    admin_recalls_capture: bool = False


def check_join_accepted(client: NSClient, invite: str) -> None:
    r = client.post("/v1/members/join", json={"invite": invite})
    assert r.status_code == 200 and r.json().get("tenant_id"), f"POST /v1/members/join -> {r.status_code}: {r.text[:120]}"


@pytest.fixture(scope="module")
def journey(ctx, a) -> Journey:
    run = ctx.run_id
    nonce = uuid.uuid4().hex[:6]  # a rerun with the same NS_RUN_ID must not see leftover text
    j = Journey(marker=f"zq-{run}{nonce}-o18", query=f"zq-{run}{nonce}-o18 sluice gate logbook")
    j.invite = call("invite", a, "POST", "/v1/members/invites", poll=False, json={})
    body = j.invite.body if isinstance(j.invite.body, dict) else {}
    token = str(body.get("invite", "")) if j.invite.status == 200 else ""
    j.invite.body = {k: v for k, v in body.items() if k != "invite"}  # the secret stays out of the record
    try:
        j.member = client_for(ctx, f"acc-{run}-member")
    except Exception as exc:  # noqa: BLE001
        j.member_error = f"{type(exc).__name__}: {exc}"[:200]
    if not token or j.member is None:
        return j
    j.join = call("join", j.member, "POST", "/v1/members/join", poll=False, json={"invite": token})
    j.listing = call("members", a, "GET", "/v1/members", poll=False)
    # the member's ordinary capture: what the plugin sends for a team-default category, with no visibility field
    j.capture = call("capture", j.member, "POST", "/v1/memories/raw", json={
        "content": f"{j.marker} The sluice gate logbook is kept in the harbour office.",
        "category": "tech_stack", "scope": "global",
    })
    j.capture_id = memory_id_of(j.capture.final)
    if j.capture.ok and j.capture_id:
        j.member_recalls_capture = _held(lambda: eventually(lambda: check_recalled(j.member, j.marker, j.query), timeout=90, what="member recalls own capture"))
    j.admin_recalls_capture = _holds(lambda: check_recalled(a, j.marker, j.query))
    return j


def _holds(check) -> bool:
    try:
        check()
        return True
    except AssertionError:
        return False


def _held(check) -> str:
    """'' when ``check`` holds, else why not."""
    try:
        check()
        return ""
    except AssertionError as exc:
        return str(exc)[:200]


@proves("archive.phase2-onb")
@expected_red(RED_MEMBERSHIP)
def test_an_administrator_invites_a_person_who_joins_once_and_is_listed(a, journey):
    assert journey.invite.status == 200, journey.invite.describe()
    assert journey.invite.body.get("invite_id") and journey.invite.body.get("expires_at"), journey.invite.describe()
    assert not journey.member_error, journey.member_error
    assert journey.join.status == 200 and isinstance(journey.join.body, dict) and journey.join.body.get("tenant_id"), journey.join.describe()
    assert journey.listing.status == 200, journey.listing.describe()
    members = json.dumps(journey.listing.body)
    assert journey.member.user_id and len(journey.listing.body.get("members", [])) >= 2, f"member list: {members[:200]}"


@proves("archive.phase2-onb")
@negative_control
@expected_red(RED_MEMBERSHIP)
def test_join_check_fails_for_an_invite_nobody_issued(journey):
    assert journey.member is not None, journey.member_error
    assert journey.join.status == 200, f"a real invite does not join first: {journey.join.describe()}"
    expect_check_fails(check_join_accepted, journey.member, "nsi_" + "0" * 24 + ".bogus")


@proves("archive.phase2-onb", "model.visibility-tiers")
@expected_red(RED_MEMBERSHIP)  # first failing step today is the missing join; the private-capture rule itself belongs to the onboarding policy slice, reached after it
def test_a_members_default_capture_is_private_to_them_and_never_team_visible(journey):
    assert journey.join.status == 200, f"the member never joined: {journey.join.describe()}"
    if journey.capture.status != 202 and not journey.capture.ok:
        err = ((journey.capture.body or {}).get("detail") or {}).get("error") if isinstance(journey.capture.body, dict) else None
        assert journey.capture.status == 403 and isinstance(err, dict) and err.get("code") == "permission_denied", (
            f"the capture was refused, but not by a permission denial: {journey.capture.describe()}"
        )
        return  # refused by policy: nothing was stored, so nothing can be team-visible
    assert journey.capture.ok and journey.capture_id, f"the capture was neither refused nor stored: {journey.capture.describe()}"
    assert not journey.member_recalls_capture, f"the member cannot recall their own capture: {journey.member_recalls_capture}"
    assert not journey.admin_recalls_capture, "the administrator recalled the member's default capture"


def _must_not_recall(client: NSClient, marker: str, words: str = "harbour kiosk six") -> None:
    hits = client.search(f"{marker} {words}", limit=20)
    assert not any(marker in h.get("memory", "") for h in hits), f"{marker} was recalled"


@proves("model.visibility-tiers")
@negative_control
def test_not_recalled_check_fails_for_a_fact_shared_with_the_team(a, b, ctx):
    """The 'nobody else recalls it' check must be able to fail: a SHARED fact is recalled by the other member."""
    from nsaccept.helpers import write_raw

    marker = f"zq-{ctx.run_id}-o18ctl"
    w = write_raw(a, f"{marker} The harbour kiosk opens at six.", "domain_knowledge", marker=marker, scope="global", visibility="shared")
    assert w.completed, f"control write ended as {w.status}"
    eventually(lambda: check_recalled(b, marker, f"{marker} harbour kiosk six"), timeout=90, what="B recalls the shared control")
    expect_check_fails(_must_not_recall, b, marker)


# ── two real tenants (E-06) ──────────────────────────────────────


@dataclass
class Tenants:
    private_marker: str
    shared_marker: str
    t2_marker: str
    t2: NSClient | None = None
    t2_mcp: MCPClient | None = None
    error: str = ""
    private_id: str = ""
    shared_id: str = ""
    task_id: str = ""
    steps: dict = field(default_factory=dict)
    t2_export_text: str = ""
    a_blind_to_t2: str = ""
    mcp_t2_hits: str = ""
    mcp_error: str = ""  # any MCP failure is a failure of the story, never "not recalled"
    t2_own_rest: str = ""  # positive controls: tenant two recalls its OWN record over REST and over MCP
    t2_own_mcp: str = ""


@pytest.fixture(scope="module")
def tenants(ctx, a) -> Tenants:
    run = ctx.run_id
    nonce = uuid.uuid4().hex[:6]  # a rerun with the same NS_RUN_ID must not see leftover text
    t = Tenants(private_marker=f"zq-{run}{nonce}-t1p", shared_marker=f"zq-{run}{nonce}-t1s", t2_marker=f"zq-{run}{nonce}-t2p")
    token, user = os.environ.get("NS_TOKEN_T2", "").strip(), os.environ.get("NS_USER_T2", "").strip()
    if not token or not user:
        t.error = "no second tenant client: the gate provides NS_TOKEN_T2 and NS_USER_T2 (a member of another tenant) once tenant cells exist"
        return t
    t.t2 = NSClient(ctx.base_url, token, user)
    t.t2_mcp = MCPClient(ctx.base_url, token)
    priv = call("t1 private", a, "POST", "/v1/memories/raw", json={
        "content": f"{t.private_marker} Tenant one keeps its berth plan in the red folder.",
        "category": "preference", "scope": "global", "visibility": "private"})
    shared = call("t1 shared", a, "POST", "/v1/memories/raw", json={
        "content": f"{t.shared_marker} Tenant one posts the berth plan on the harbour board.",
        "category": "domain_knowledge", "scope": "global", "visibility": "shared"})
    t.private_id, t.shared_id, t.task_id = memory_id_of(priv.final), memory_id_of(shared.final), priv.task_id
    own = call("t2 own", t.t2, "POST", "/v1/memories/raw", json={
        "content": f"{t.t2_marker} Tenant two keeps its crane log in the blue folder.",
        "category": "preference", "scope": "global", "visibility": "private"})
    s = t.steps
    s["t2 own"] = own
    s["t2 private by id"] = call("t2 private by id", t.t2, "GET", f"/v1/memories/{t.private_id}", poll=False)
    s["t2 shared by id"] = call("t2 shared by id", t.t2, "GET", f"/v1/memories/{t.shared_id}", poll=False)
    s["t2 search"] = call("t2 search", t.t2, "POST", "/v1/search", poll=False, json={"query": f"{t.private_marker} {t.shared_marker} berth plan", "limit": 20})
    s["t2 status"] = call("t2 status", t.t2, "GET", f"/v1/memories/status/{t.task_id}", poll=False)
    try:
        t.t2_export_text = bundle_text(okf_export(t.t2))
    except AssertionError as exc:
        t.error = str(exc)
    own_q = f"{t.t2_marker} tenant two crane log blue folder"
    t.t2_own_rest = _held(lambda: eventually(lambda: check_recalled(t.t2, t.t2_marker, own_q), timeout=90, what="tenant two recalls its own record over REST"))
    if not t.t2_own_rest:  # only meaningful once tenant two's own record is known to be recallable
        t.a_blind_to_t2 = _held(lambda: _must_not_recall(a, t.t2_marker))
    else:
        t.a_blind_to_t2 = "not checked: tenant two's own record was not recallable"
    try:
        t.t2_mcp.initialize()
        t.mcp_t2_hits = json.dumps(mcp_json(t.t2_mcp, "recall_memories", {"query": f"{t.shared_marker} berth plan harbour board", "limit": 20}))
        own = json.dumps(mcp_json(t.t2_mcp, "recall_memories", {"query": own_q, "limit": 20}))
        t.t2_own_mcp = "" if t.t2_marker in own else "tenant two's MCP recall did not return its own record"
    except Exception as exc:  # noqa: BLE001
        t.mcp_error = f"{type(exc).__name__}: {exc}"[:200]
    return t


def _need(t: Tenants) -> None:
    assert not t.error, t.error
    assert not t.mcp_error, f"MCP failed: {t.mcp_error}"
    assert not t.t2_own_rest, f"tenant two cannot recall its own record over REST: {t.t2_own_rest}"
    assert not t.t2_own_mcp, t.t2_own_mcp
    assert t.private_id and t.shared_id, "tenant one's setup writes did not complete"


@proves("archive.phase2-ten", "model.visibility-tiers")
@expected_red(RED_TENANT_CELLS)
def test_a_person_of_another_tenant_cannot_read_search_poll_or_export_tenant_ones_records(tenants):
    _need(tenants)
    s = tenants.steps
    for name in ("t2 private by id", "t2 shared by id", "t2 status"):
        assert s[name].status in DENIED, s[name].describe()
    assert s["t2 search"].ok, s["t2 search"].describe()
    blob = json.dumps(s["t2 search"].body)
    assert tenants.private_marker not in blob and tenants.shared_marker not in blob, "tenant two's search returned tenant one's records"
    assert tenants.private_marker not in tenants.t2_export_text and tenants.shared_marker not in tenants.t2_export_text, (
        "tenant two's export carries tenant one's records"
    )


@proves("archive.phase2-ten", "model.visibility-tiers")
@expected_red(RED_TENANT_CELLS)
def test_a_record_shared_inside_one_tenant_is_not_shared_into_the_other_over_rest_or_mcp(b, tenants):
    _need(tenants)
    eventually(lambda: check_recalled(b, tenants.shared_marker, "berth plan harbour board"), timeout=90, what="same-tenant B recalls the shared fact")
    assert tenants.shared_marker not in tenants.mcp_t2_hits, f"tenant two's MCP recall returned it: {tenants.mcp_t2_hits[:200]}"
    assert not tenants.a_blind_to_t2, f"tenant one recalled tenant two's private record: {tenants.a_blind_to_t2}"


@proves("model.visibility-tiers")
@negative_control
def test_recall_check_fails_for_a_same_tenant_user_reading_a_private_record(a, b, ctx):
    from nsaccept.helpers import write_raw

    marker = f"zq-{ctx.run_id}-t1ctl"
    w = write_raw(a, f"{marker} The lock keeper's lamp is amber.", "preference", marker=marker, scope="global", visibility="private")
    assert w.completed, f"control write ended as {w.status}"
    eventually(lambda: check_recalled(a, marker, f"{marker} lock keeper lamp amber"), timeout=90, what="A recalls its private record")
    expect_check_fails(check_recalled, b, marker, f"{marker} lock keeper lamp amber")
