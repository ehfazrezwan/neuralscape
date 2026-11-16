"""Dictator role, the authoritative `standard` tier, the process registry, and the
dictator-only admin paths, with the flags ON (profile ``flags-on``).

Documented behaviour under test: only a configured dictator may write the
`standard` tier (403 for everyone else, answered synchronously); standards
tagged `critical`/`always` are injected into every user's context (global,
project, and the session-start hook) and the rest surface through recall; a
process is a set of standards sharing a `process:<slug>` tag, enumerated by
list_processes and assembled by get_process (steps in numeric order); the
admin tools and project-wide settings that are dictator-only are open to the
dictator and closed to everyone else.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nsaccept import eventually, expect_check_fails, needs, negative_control, precondition, profiles, proves
from nsaccept.client import NSClient
from nsaccept.helpers import Written, mcp_json, write_raw
from nsaccept.hooks import run_hook

pytestmark = [profiles("flags-on"), needs("dictator")]


# ── the real checks ──────────────────────────────────────────────


def standards_in(client: NSClient, path: str, **params) -> list[dict]:
    r = client.get(path, params=params)
    assert r.status_code == 200, f"GET {path} -> {r.status_code}: {r.text[:160]}"
    return r.json().get("standards", [])


def check_standard_injected(client: NSClient, marker: str, project_id: str) -> None:
    """The marker is in the always-injected standards of every context surface."""
    for path in ("/v1/context/global", f"/v1/context/{project_id}"):
        got = standards_in(client, path)
        assert any(marker in (s.get("memory") or "") for s in got), (
            f"{marker} not among the {len(got)} standards served by GET {path}"
        )
    r = client.get("/v1/context/inject", params={"project_id": project_id})
    assert r.status_code == 200, f"GET /v1/context/inject -> {r.status_code}: {r.text[:160]}"
    text = r.json().get("additionalContext", "")
    assert marker in text and "AUTHORITATIVE Standards" in text, f"{marker} not in the injected standards block"


def check_standard_write_refused(client: NSClient, ctx_run_id: str) -> None:
    r = client.post(
        "/v1/memories/raw",
        json={
            "content": f"zq-{ctx_run_id}-forged A forged standard that must never be stored.",
            "category": "convention", "scope": "global", "visibility": "standard",
        },
    )
    assert r.status_code == 403, f"standard-tier write by {client.user_id} -> {r.status_code} (want 403): {r.text[:160]}"


def check_process_bundle(body: dict, slug: str, step_markers: list[str], guideline_marker: str, title_marker: str) -> None:
    assert body.get("slug") == slug, f"slug: {body.get('slug')!r}"
    assert title_marker in (body.get("title") or ""), f"title {body.get('title')!r} lacks {title_marker}"
    steps = body.get("steps") or []
    assert len(steps) == len(step_markers), f"{len(steps)} steps, want {len(step_markers)}"
    for got, marker in zip(steps, step_markers):
        assert marker in got, f"steps out of numeric order: {marker} not at its place in {[s[:40] for s in steps]}"
    assert any(guideline_marker in g for g in body.get("guidelines") or []), "guideline missing from the bundle"


# ── module fixture: the dictator's writes ────────────────────────


@dataclass
class Std:
    slug: str
    critical: Written
    plain: Written
    proc_def: Written
    step_late: Written
    step_early: Written
    guideline: Written
    project: str
    markers: dict


@pytest.fixture(scope="module")
def std(ctx, dictator) -> Std:
    run = ctx.run_id
    slug = f"acc-{run}-proc"
    m = {k: f"zq-{run}-{k}" for k in ("crit", "plain", "title", "stepa", "stepb", "guide")}
    kw = dict(scope="global", visibility="standard")
    critical = write_raw(
        dictator, f"{m['crit']} Always sign off a deployment in the harbour log before leaving the quay.",
        "convention", marker=m["crit"], tags=["critical"], **kw,
    )
    plain = write_raw(
        dictator, f"{m['plain']} Lanterns on the east pier are inspected by the second shift.",
        "convention", marker=m["plain"], **kw,
    )
    proc_def = write_raw(
        dictator, f"Pier closing procedure {m['title']}\nHow the harbour pier is closed at night.",
        "procedure", marker=m["title"], tags=[f"process:{slug}", "process-def"], **kw,
    )
    # Written in the "wrong" order on purpose: step 10 first, then step 2.
    step_late = write_raw(
        dictator, f"{m['stepb']} Step ten: log the lamp count.", "procedure", marker=m["stepb"],
        tags=[f"process:{slug}", "process-step:10"], **kw,
    )
    step_early = write_raw(
        dictator, f"{m['stepa']} Step two: lower the gate.", "procedure", marker=m["stepa"],
        tags=[f"process:{slug}", "process-step:2"], **kw,
    )
    guideline = write_raw(
        dictator, f"{m['guide']} Never close the pier while a vessel is moored on it.", "convention",
        marker=m["guide"], tags=[f"process:{slug}"], **kw,
    )
    return Std(slug, critical, plain, proc_def, step_late, step_early, guideline, f"{ctx.project_id}-std", m)


def _written(std: Std) -> None:
    for w in (std.critical, std.plain, std.proc_def, std.step_late, std.step_early, std.guideline):
        w.require()


# ── the standard tier ────────────────────────────────────────────


@proves("read.standards-always-inject", "ext.standards-tier", "model.standard-tier")
def test_a_dictator_standard_is_injected_into_every_users_context(ctx, a, b, dictator, std: Std):
    _written(std)
    for client in (a, b, dictator):
        eventually(lambda c=client: check_standard_injected(c, std.markers["crit"], std.project),
                   timeout=60, interval=3, what=f"critical standard in {client.user_id}'s context")


@proves("read.standards-always-inject", "ext.standards-tier", "model.standard-tier")
@negative_control
def test_injection_check_fails_for_a_marker_nobody_wrote(ctx, a, std: Std):
    _written(std)
    precondition(check_standard_injected, a, std.markers["crit"], std.project, what="the critical standard is injected for A")
    expect_check_fails(check_standard_injected, a, f"zq-{ctx.run_id}-never-written", std.project)


@proves("read.standards-always-inject", "model.standard-tier")
def test_an_untagged_standard_is_not_always_injected_but_is_recallable(ctx, a, std: Std):
    _written(std)
    marker = std.markers["plain"]
    for path in ("/v1/context/global", f"/v1/context/{std.project}"):
        assert not any(marker in (s.get("memory") or "") for s in standards_in(a, path)), (
            f"untagged standard {marker} was always-injected by GET {path}"
        )

    def recalled():
        hits = [h for h in a.search(f"{marker} lanterns east pier inspected", limit=10) if marker in h.get("memory", "")]
        assert hits, f"{marker} not recalled by user A"
        assert hits[0].get("visibility") == "standard", f"recalled row visibility {hits[0].get('visibility')!r}"

    eventually(recalled, timeout=60, interval=3, what="untagged standard recallable via search")


@proves("auth.dictator-role", "ext.standards-tier", "model.standard-tier")
def test_only_the_dictator_may_write_the_standard_tier(ctx, a, b, dictator, mcp_a):
    check_standard_write_refused(a, ctx.run_id)
    check_standard_write_refused(b, ctx.run_id)
    mcp_a.initialize()
    out = mcp_json(mcp_a, "remember", {
        "content": f"zq-{ctx.run_id}-mcp-forged Forged through MCP.", "category": "convention",
        "scope": "global", "visibility": "standard",
    })
    assert "not authorized" in str(out.get("error", "")), f"MCP remember(standard) by A answered {out}"
    # the refused writes left nothing behind
    assert not [h for h in a.search(f"zq-{ctx.run_id}-forged forged standard", limit=10) if f"zq-{ctx.run_id}-forged" in h.get("memory", "")]


@proves("auth.dictator-role", "model.standard-tier")
@negative_control
def test_write_refusal_check_fails_for_the_dictator(ctx, a, dictator):
    precondition(check_standard_write_refused, a, ctx.run_id, what="a non-dictator's standard write is refused")
    # the dictator's identical request is accepted (202), so the 403 check must trip
    expect_check_fails(check_standard_write_refused, dictator, ctx.run_id)


# ── plugin: the session-start hook shows the standards block ─────


@needs("node")
@proves("plugin.standards-block", "read.standards-always-inject")
def test_session_start_hook_injects_the_standards_block(ctx, a, std: Std):
    _written(std)

    def hook_has_block():
        res = run_hook(
            "session-start", {"hook_event_name": "SessionStart", "source": "startup", "session_id": f"acc-{ctx.run_id}-std"},
            base_url=ctx.base_url, token=ctx.token_a, user_id=ctx.user_a, project_id=std.project,
        )
        assert res.returncode == 0 and res.data, f"hook exited {res.returncode}: {res.stderr[:160]}"
        text = (res.data.get("hookSpecificOutput") or {}).get("additionalContext", "")
        assert "AUTHORITATIVE Standards" in text, "no standards block in the hook output"
        assert std.markers["crit"] in text, f"{std.markers['crit']} not in the hook's standards block"

    eventually(hook_has_block, timeout=60, interval=3, what="session-start hook standards block")


@needs("node")
@proves("plugin.standards-block")
@negative_control
def test_hook_block_check_fails_without_a_token(ctx, std: Std):
    res = run_hook(
        "session-start", {"hook_event_name": "SessionStart", "source": "startup", "session_id": f"acc-{ctx.run_id}-std0"},
        base_url=ctx.base_url, token=ctx.bad_token, user_id=ctx.user_a, project_id=std.project,
    )

    own = run_hook(
        "session-start", {"hook_event_name": "SessionStart", "source": "startup", "session_id": f"acc-{ctx.run_id}-std1"},
        base_url=ctx.base_url, token=ctx.token_a, user_id=ctx.user_a, project_id=std.project,
    )

    def check(r):
        text = ((r.data or {}).get("hookSpecificOutput") or {}).get("additionalContext", "")
        assert std.markers["crit"] in text, "the standards block is absent from the hook output"

    precondition(check, own, what="an authenticated hook shows the standards block")
    expect_check_fails(check, res)


# ── processes ────────────────────────────────────────────────────


@proves("ext.processes-registry", "read.processes")
def test_the_process_registry_lists_and_assembles_a_dictator_process(ctx, a, b, mcp_a, std: Std):
    _written(std)
    mcp_a.initialize()
    for client in (a, b):
        def listed(c=client):
            r = c.get("/v1/processes")
            assert r.status_code == 200, f"GET /v1/processes -> {r.status_code}"
            procs = r.json()["processes"]
            row = next((p for p in procs if p["slug"] == std.slug), None)
            assert row is not None, f"{std.slug} not among {[p['slug'] for p in procs]}"
            assert std.markers["title"] in row["title"], f"title {row['title']!r}"

        eventually(listed, timeout=60, interval=3, what="process listed")
        r = client.get(f"/v1/processes/{std.slug}")
        assert r.status_code == 200, f"GET /v1/processes/{std.slug} -> {r.status_code}: {r.text[:160]}"
        check_process_bundle(r.json(), std.slug, [std.markers["stepa"], std.markers["stepb"]], std.markers["guide"], std.markers["title"])
    listing = mcp_json(mcp_a, "list_processes", {})
    assert any(p["slug"] == std.slug for p in listing["processes"]), f"MCP list_processes: {listing}"
    bundle = mcp_json(mcp_a, "get_process", {"slug": std.slug})
    check_process_bundle(bundle, std.slug, [std.markers["stepa"], std.markers["stepb"]], std.markers["guide"], std.markers["title"])
    assert a.get("/v1/processes/acc-no-such-process").status_code == 404
    assert "not found" in str(mcp_json(mcp_a, "get_process", {"slug": "acc-no-such-process"}).get("error", "")).lower()


@proves("ext.processes-registry", "read.processes")
@negative_control
def test_process_bundle_check_fails_for_a_different_order_and_slug(std: Std):
    good = {"slug": std.slug, "title": std.markers["title"], "steps": [std.markers["stepa"], std.markers["stepb"]], "guidelines": [std.markers["guide"]]}
    precondition(check_process_bundle, good, std.slug, [std.markers["stepa"], std.markers["stepb"]], std.markers["guide"], std.markers["title"], what="the well-formed bundle passes")
    swapped = {**good, "steps": list(reversed(good["steps"]))}
    expect_check_fails(check_process_bundle, swapped, std.slug, [std.markers["stepa"], std.markers["stepb"]], std.markers["guide"], std.markers["title"])
    expect_check_fails(check_process_bundle, {**good, "slug": "other"}, std.slug, [std.markers["stepa"], std.markers["stepb"]], std.markers["guide"], std.markers["title"])


# ── dictator-only admin paths: allowed for the dictator ──────────


def check_admin_open(client: NSClient, user: str) -> None:
    audit = client.post("/v1/admin/audit-private-leakage", json={"user_id": user})
    assert audit.status_code == 200 and isinstance(audit.json(), dict), f"audit -> {audit.status_code}: {audit.text[:160]}"
    dry = client.post("/v1/admin/rescope-private-derivatives", json={"user_id": user, "dry_run": True})
    assert dry.status_code == 200 and "graph_jobs_enqueued" in dry.json(), f"rescope(dry-run) -> {dry.status_code}: {dry.text[:160]}"


@proves("hygiene.private-leak-remediation", "auth.dictator-role")
def test_the_leak_audit_and_dry_run_rescope_are_open_to_the_dictator_only(ctx, a, dictator):
    check_admin_open(dictator, ctx.user_a)
    for path, body in (("/v1/admin/audit-private-leakage", {}), ("/v1/admin/rescope-private-derivatives", {"dry_run": True})):
        assert a.post(path, json=body).status_code == 403, f"A reached {path}"


@proves("hygiene.private-leak-remediation", "auth.dictator-role")
@negative_control
def test_admin_open_check_fails_for_a_non_dictator(ctx, a, dictator):
    precondition(check_admin_open, dictator, ctx.user_a, what="the dictator reaches the admin routes")
    expect_check_fails(check_admin_open, a, ctx.user_a)


def check_config(body: dict, project: str, systems: list[str], fuse: bool, engine: str | None) -> None:
    assert body == {"project_id": project, "code_systems": systems, "fuse_code_into_recall": fuse, "default_engine": engine}, f"config: {body}"


@proves("knowledge.router", "auth.dictator-role")
def test_the_dictator_may_set_a_projects_knowledge_config_and_others_may_not(ctx, a, dictator, mcp_dictator, mcp_a):
    project = f"{ctx.project_id}-kr"
    path = f"/v1/projects/{project}/knowledge-config"
    body = {"project_id": project, "code_systems": ["code-native"], "fuse_code_into_recall": False, "default_engine": "code-native"}
    assert a.request("PUT", path, json=body).status_code == 403, "non-admin PUT knowledge-config was accepted"
    check_config(a.get(path).json(), project, [], True, None)  # refused write changed nothing
    r = dictator.request("PUT", path, json=body)
    assert r.status_code == 200, f"dictator PUT knowledge-config -> {r.status_code}: {r.text[:160]}"
    check_config(r.json(), project, ["code-native"], False, "code-native")
    check_config(a.get(path).json(), project, ["code-native"], False, "code-native")  # the write is what everyone reads
    # and through MCP
    mcp_dictator.initialize()
    mcp_a.initialize()
    other = f"{project}2"
    out = mcp_json(mcp_dictator, "set_project_knowledge_config", {"project_id": other, "code_systems": ["code-native"], "fuse_code_into_recall": True})
    assert "error" not in out, f"dictator MCP set_project_knowledge_config answered {out}"
    got = mcp_json(mcp_a, "get_project_knowledge_config", {"project_id": other})
    assert got.get("code_systems") == ["code-native"], f"A reads {got} after the dictator set {other}"


@proves("knowledge.router")
@negative_control
def test_config_check_fails_for_the_default_config(ctx):
    precondition(check_config, {"project_id": "p", "code_systems": ["code-native"], "fuse_code_into_recall": False, "default_engine": "code-native"}, "p", ["code-native"], False, "code-native", what="a matching config passes")
    expect_check_fails(check_config, {"project_id": "p", "code_systems": [], "fuse_code_into_recall": True, "default_engine": None}, "p", ["code-native"], False, "code-native")


def check_instructions(client: NSClient, project: str, text: str | None) -> None:
    r = client.get("/v1/settings/extraction-instructions", params={"project_id": project})
    assert r.status_code == 200, f"GET extraction-instructions -> {r.status_code}"
    assert r.json().get("instructions") == text, f"instructions for {project}: {r.json().get('instructions')!r}"


@proves("write.operator-extraction-instructions", "auth.dictator-role")
def test_project_wide_extraction_instructions_are_a_dictator_only_write(ctx, a, b, dictator):
    project = f"{ctx.project_id}-ei"
    text = f"zq-{ctx.run_id}-guidance Prefer to phrase decisions as 'we chose X because Y'."
    for client in (a, b):
        r = client.request("PUT", "/v1/settings/extraction-instructions", json={"instructions": text, "project_id": project})
        assert r.status_code == 403, f"{client.user_id} PUT project instructions -> {r.status_code} (want 403)"
    check_instructions(a, project, None)
    r = dictator.request("PUT", "/v1/settings/extraction-instructions", json={"instructions": text, "project_id": project})
    assert r.status_code == 200, f"dictator PUT -> {r.status_code}: {r.text[:160]}"
    assert r.json()["scope"] == "project" and r.json()["target_id"] == project and r.json()["updated_by"] == ctx.dictator_user, f"PUT body: {r.json()}"
    check_instructions(a, project, text)  # members read what the dictator set
    check_instructions(b, project, text)
    cleared = dictator.request("PUT", "/v1/settings/extraction-instructions", json={"instructions": "", "project_id": project})
    assert cleared.status_code == 200
    check_instructions(a, project, None)


@proves("write.operator-extraction-instructions")
@negative_control
def test_instructions_check_fails_for_an_unset_project(ctx, a):
    precondition(check_instructions, a, f"{ctx.project_id}-never-set", None, what="an unset project reads as no instructions")
    expect_check_fails(check_instructions, a, f"{ctx.project_id}-never-set", "anything")
