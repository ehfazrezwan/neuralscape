"""Dreaming (profile ``flags-on``): the nightly consolidation sweep, run on demand.

Folds the assertions of scripts/dreaming_e2e.py, bridges_card_e2e.py and the
settling part of salience_e2e.py into black-box checks: the dictator enqueues a
sweep over REST (and MCP), the graph worker runs it with the real LLM, and the
effects are read back through recall, listings, the card routes/tool, the
plugin hook and the vault on the shared volume.

Seeds (all unique-marker raw writes, one LLM-bound sweep per pool, shared by
every test below):  user A's private pool (an identity, a contradiction pair,
a three-fact workflow pattern), a shared project pool, and user B's private pool.

Not asserted on purpose: the wording of any LLM output. A marker, an id, a
count or a grammar is asserted; a paraphrase never is.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

import pytest

from nsaccept import eventually, expect_check_fails, needs, negative_control, precondition, profiles, proves
from nsaccept.client import NSClient
from nsaccept.helpers import list_rows, mcp_json, write_raw
from nsaccept.hooks import run_hook

pytestmark = [profiles("flags-on"), needs("dictator")]

SWEEP_WAIT = 900
CARD_LINE_RE = re.compile(r"^(IDENTITY|ATTRIBUTE|RELATIONSHIP|INSTRUCTION): .+$")
API_SERVICE = os.environ.get("NS_API_SERVICE", "neuralscape")


# ── the real checks ──────────────────────────────────────────────


def pool_report(result: dict, pool: str) -> dict:
    rows = [p for p in result.get("pools", []) if p.get("pool") == pool]
    assert len(rows) == 1, f"sweep result has {len(rows)} reports for {pool}: {[p.get('pool') for p in result.get('pools', [])]}"
    return rows[0]


def check_dreamt(result: dict, pool: str, min_staged: int = 1) -> dict:
    rep = pool_report(result, pool)
    assert rep["status"] == "dreamt", f"{pool}: status {rep['status']!r} ({rep.get('reason')}) errors={rep.get('errors')}"
    assert rep["staged"] >= min_staged, f"{pool}: staged {rep['staged']} < {min_staged}"
    return rep


def check_superseded(client: NSClient, old_id: str, new_id: str) -> None:
    """The older fact is gone from the live listing (but kept, tombstoned) and the newer one stays."""
    live = {r["id"] for r in list_rows(client)}
    assert new_id in live, "the newer fact is not live"
    assert old_id not in live, "the older, contradicted fact is still live after the sweep"
    audit = {r["id"] for r in list_rows(client, include_tombstoned="true")}
    assert old_id in audit, "the older fact is not even in the audit listing (hard-deleted, not superseded)"
    recalled = {h.get("id") for h in client.search("harbour-monitor depot bare-metal single host failover", limit=20)}
    assert old_id not in recalled, "recall still returns the superseded fact"


def check_card(client: NSClient, pool: str) -> dict:
    r = client.get("/v1/extensions/dreaming/card", params={"pool": pool})
    assert r.status_code == 200, f"GET card?pool={pool} as {client.user_id} -> {r.status_code}: {r.text[:160]}"
    body = r.json()
    lines = body.get("lines") or []
    assert 0 < len(lines) <= 40, f"{len(lines)} card lines (want 1..40)"
    bad = [ln for ln in lines if not CARD_LINE_RE.match(ln)]
    assert not bad, f"card lines outside the grammar: {bad[:3]}"
    assert body.get("card") == "\n".join(lines), "card text is not the joined lines"
    return body


def vault_grep(ctx, needle: str, root: str = "/data/vault") -> list[str]:
    proc = ctx.compose("exec", "-T", API_SERVICE, "grep", "-rl", "--", needle, root, check=False)
    if proc.returncode == 2 and "No such file" in (proc.stderr or ""):
        return []  # that part of the vault was never written
    assert proc.returncode in (0, 1), f"grep in the vault failed ({proc.returncode}): {(proc.stderr or '')[:160]}"
    return [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]


# ── seeds + sweeps ───────────────────────────────────────────────


@dataclass
class Dream:
    pool_a: str
    pool_b: str
    pool_p: str
    project: str
    name_a: str
    name_b: str
    ids: dict = field(default_factory=dict)
    written: dict = field(default_factory=dict)
    settling: dict = field(default_factory=dict)  # result of the non-forced run
    results: dict = field(default_factory=dict)  # pool -> sweep result
    job_ids: dict = field(default_factory=dict)
    error: str = ""


def _w(a_client, dream: Dream, key: str, content: str, category: str, **kw) -> None:
    w = write_raw(a_client, content, category, marker=key, **kw)
    dream.written[key] = w
    dream.ids[key] = w.memory_id


@pytest.fixture(scope="module")
def dream(ctx, a, b, dictator) -> Dream:
    run = ctx.run_id
    cap = run.capitalize()
    d = Dream(
        pool_a=f"user--{ctx.user_a}", pool_b=f"user--{ctx.user_b}",
        pool_p=f"shared--project--{ctx.project_id}-dream", project=f"{ctx.project_id}-dream",
        name_a=f"Zq{cap}Quillen", name_b=f"Zq{cap}Marlow",
    )
    depot = f"Zq{cap}Depot"
    mk = lambda tag: f"zq-{run}-{tag}"  # noqa: E731
    priv = dict(scope="global", visibility="private")
    # A: identity
    _w(a, d, "id1", f"{mk('id1')} The operator is called {d.name_a} and works as a quay supervisor.", "personal_fact", **priv)
    _w(a, d, "id2", f"{mk('id2')} {d.name_a} prefers tea over coffee and answers messages in the morning.", "preference", **priv)
    _w(a, d, "id3", f"{mk('id3')} {d.name_a} wants release notes written as a short bulleted list.", "preference", **priv)
    # A: a contradiction pair, older first
    _w(a, d, "old", f"{mk('old')} The harbour-monitor for the {depot} depot runs on one bare-metal host and has no failover.", "decision", **priv)
    time.sleep(1.2)
    _w(
        a, d, "new",
        f"{mk('new')} As of November 2026 the harbour-monitor for the {depot} depot no longer runs on a single "
        f"bare-metal host: it moved to a three-node managed cluster with automatic failover, so the single-host "
        f"statement is obsolete.",
        "decision", **priv,
    )
    # A: a three-fact pattern for the reflection pass
    for i, text in enumerate((
        "After each tide survey the surveyor files the readings in the logbook before leaving the quay",
        "Before every night shift the surveyor re-reads the logbook entries of the previous survey",
        "The surveyor never closes a survey day without cross-checking the logbook against the gauge readings",
    ), start=1):
        _w(a, d, f"pat{i}", f"{mk(f'pat{i}')} {text}.", "workflow", **priv)
    # shared project pool
    shared = dict(scope="project", project_id=d.project, visibility="shared")
    for i, text in enumerate((
        f"The {depot} depot dashboard is built with React and Vite",
        f"The {depot} depot dashboard polls the tide gauge API every thirty seconds",
        f"Deploys of the {depot} depot dashboard go out through the pier pipeline on Fridays",
        f"The {depot} depot dashboard keeps its alerts in a small SQLite file on the pilot hut host",
    ), start=1):
        _w(a, d, f"sh{i}", f"{mk(f'sh{i}')} {text}.", ["architecture", "decision", "workflow", "tech_stack"][i - 1], **shared)
    # B: its own private pool
    _w(b, d, "b1", f"{mk('b1')} The operator is called {d.name_b} and works as a harbour pilot.", "personal_fact", **priv)
    _w(b, d, "b2", f"{mk('b2')} {d.name_b} prefers coffee over tea and answers messages in the evening.", "preference", **priv)
    try:
        for key, w in d.written.items():
            w.require()
        # 1. a NON-forced run right after the writes: the settling guard defers the pool (no LLM spend)
        jid = _enqueue(dictator, d.pool_p, force=False)
        d.settling = _finish(dictator, jid)
        # 2. forced sweeps of the three pools, concurrently on the graph worker
        for pool in (d.pool_a, d.pool_b, d.pool_p):
            d.job_ids[pool] = _enqueue(dictator, pool, force=True)
        for pool, jid in d.job_ids.items():
            d.results[pool] = _finish(dictator, jid)
    except AssertionError as exc:
        d.error = str(exc)
    return d


def _enqueue(dictator: NSClient, pool: str, *, force: bool, dry_run: bool = False) -> str:
    r = dictator.post("/v1/extensions/dreaming/run", json={"pool": pool, "force": force, "dry_run": dry_run})
    assert r.status_code == 202, f"POST /v1/extensions/dreaming/run -> {r.status_code}: {r.text[:200]}"
    job_id = r.json().get("job_id")
    assert job_id, f"/run answered 202 without a job_id: {r.text[:200]}"
    return job_id


def _finish(dictator: NSClient, job_id: str) -> dict:
    st = dictator.poll_task(job_id, timeout=SWEEP_WAIT)
    assert st.get("status") == "completed", f"sweep job {job_id} ended as {st.get('status')}: {st.get('error') or st.get('last_status')}"
    return st["result"]


def _ready(d: Dream) -> Dream:
    assert not d.error, d.error
    return d


# ── the sweep ────────────────────────────────────────────────────


@proves("ext.dreaming-sweep")
def test_the_sweep_dreams_every_pool_and_reports_it(ctx, dictator, a, dream: Dream):
    _ready(dream)
    for pool, floor in ((dream.pool_a, 8), (dream.pool_b, 2), (dream.pool_p, 4)):
        check_dreamt(dream.results[pool], pool, min_staged=floor)
    totals = dream.results[dream.pool_a]["totals"]
    assert totals["pools_dreamt"] == 1 and totals["pools_failed"] == 0, f"totals: {totals}"
    status = a.get("/v1/extensions/dreaming/status")
    assert status.status_code == 200, f"GET /status -> {status.status_code}"
    body = status.json()
    assert body["enabled"] is True, "status says dreaming is disabled although the flag is on"
    run_ids = {r["run_id"] for r in dream.results.values()}
    assert (body.get("last_run") or {}).get("run_id") in run_ids, f"last_run {body.get('last_run', {}).get('run_id')} is none of our sweeps {run_ids}"


@proves("ext.dreaming-sweep")
@negative_control
def test_dreamt_check_fails_for_another_pool_and_a_failed_report(dream: Dream):
    _ready(dream)
    precondition(check_dreamt, dream.results[dream.pool_a], dream.pool_a, what="the sweep of A's pool is dreamt")
    expect_check_fails(check_dreamt, dream.results[dream.pool_a], dream.pool_b)  # a report that is not about that pool
    failed = {"pools": [{"pool": dream.pool_a, "status": "sweep_failed", "reason": "x", "staged": 0}]}
    expect_check_fails(check_dreamt, failed, dream.pool_a)


@proves("ext.dreaming-sweep")
def test_a_fresh_pool_is_deferred_by_the_settling_guard_until_forced(dream: Dream):
    _ready(dream)
    rep = pool_report(dream.settling, dream.pool_p)
    assert rep["status"] == "settling", f"non-forced sweep of a pool written seconds ago ended {rep['status']!r} ({rep.get('reason')})"
    assert rep["applied"] == 0 and rep["insights"] == 0, f"a deferred pool was still consolidated: {rep}"
    check_dreamt(dream.results[dream.pool_p], dream.pool_p)  # force=true bypassed it


@proves("ext.dreaming-sweep")
def test_only_the_dictator_may_trigger_a_sweep(ctx, a, mcp_a, mcp_dictator, dream: Dream):
    _ready(dream)
    r = a.post("/v1/extensions/dreaming/run", json={"pool": dream.pool_a, "force": True, "dry_run": True})
    assert r.status_code == 403, f"non-dictator POST /run -> {r.status_code} (want 403)"
    mcp_a.initialize()
    refused = mcp_json(mcp_a, "schedule_dream", {"pool": dream.pool_a, "dry_run": True, "force": True})
    assert "not authorized" in str(refused.get("error", "")), f"MCP schedule_dream by A answered {refused}"


@proves("ext.dreaming-sweep")
def test_schedule_dream_over_mcp_runs_a_dry_run_that_writes_nothing(ctx, b, mcp_dictator, dictator, dream: Dream):
    """The fixture's forced sweep already stamped B's pool, and staging is incremental (only rows new or
    changed since that stamp), so a fresh private row is written first. The MCP dry run must then stage it,
    write nothing, and leave the pool unstamped: a second dry run stages the same row again."""
    _ready(dream)
    marker = f"zq-{ctx.run_id}-dryrun"
    write_raw(
        b, f"{marker} B keeps a laminated tide table pinned inside the boat shed door.", "preference",
        marker=marker, scope="global", visibility="private",
    ).require()
    before = sorted(r["id"] for r in list_rows(b))
    mcp_dictator.initialize()
    staged = []
    for attempt in ("first", "second"):
        out = mcp_json(mcp_dictator, "schedule_dream", {"pool": dream.pool_b, "dry_run": True, "force": True})
        assert out.get("status") == "enqueued" and out.get("job_id"), f"{attempt} schedule_dream answered {out}"
        result = _finish(dictator, out["job_id"])
        assert result["dry_run"] is True, f"{attempt}: dry_run flag lost: {result.get('dry_run')}"
        staged.append(check_dreamt(result, dream.pool_b)["staged"])  # >= 1: the fresh row is new since the last stamp
        assert sorted(r["id"] for r in list_rows(b)) == before, f"the {attempt} dry-run sweep changed B's memories"
    assert staged[1] >= 1, f"the second dry run staged {staged[1]}: the first dry run stamped the pool (staged {staged})"


# ── deep consolidation: contradiction -> supersession ────────────


@proves("ext.dreaming-deep-consolidate", "model.contradiction-supersession")
def test_the_older_of_two_contradicting_facts_is_superseded(a, dream: Dream):
    _ready(dream)
    check_superseded(a, dream.ids["old"], dream.ids["new"])


@proves("ext.dreaming-deep-consolidate", "model.contradiction-supersession")
@negative_control
def test_supersession_check_fails_when_the_roles_are_swapped_or_another_user_asks(b, a, dream: Dream):
    _ready(dream)
    precondition(check_superseded, a, dream.ids["old"], dream.ids["new"], what="A's older fact is superseded")
    expect_check_fails(check_superseded, a, dream.ids["new"], dream.ids["old"])  # the newer one was not superseded
    expect_check_fails(check_superseded, b, dream.ids["old"], dream.ids["new"])  # B never held either row


# ── REM reflections ──────────────────────────────────────────────


def check_insight_from(client: NSClient, premise_ids: set[str]) -> dict:
    rows = [r for r in list_rows(client) if r.get("source_type") == "dream"]
    mine = [r for r in rows if premise_ids & set(r.get("derived_from") or [])]
    assert mine, f"no dream insight derived from the seeded memories among {len(rows)} dream rows"
    row = mine[0]
    assert row.get("epistemic_level") in ("deductive", "inductive", "reflection"), f"epistemic_level {row.get('epistemic_level')!r}"
    return row


@proves("ext.dreaming-rem-reflections")
def test_the_reflection_pass_stores_an_insight_derived_from_the_seeded_memories(a, dream: Dream):
    _ready(dream)
    rep = pool_report(dream.results[dream.pool_a], dream.pool_a)
    assert rep["insights"] >= 1, f"the sweep stored {rep['insights']} insights (errors: {rep.get('errors')})"
    row = check_insight_from(a, {dream.ids[k] for k in dream.ids if k in ("id1", "id2", "id3", "old", "new", "pat1", "pat2", "pat3")})
    chain = a.get(f"/v1/memories/{row['id']}/reasoning_chain")
    assert chain.status_code == 200, f"GET reasoning_chain -> {chain.status_code}: {chain.text[:160]}"


@proves("ext.dreaming-rem-reflections")
@negative_control
def test_insight_check_fails_for_user_b(a, b, dream: Dream):
    _ready(dream)
    precondition(check_insight_from, a, {dream.ids[k] for k in ("pat1", "pat2", "pat3")}, what="A holds an insight derived from the pattern")
    expect_check_fails(check_insight_from, b, {dream.ids["pat1"], dream.ids["pat2"], dream.ids["pat3"]})


# ── identity cards ───────────────────────────────────────────────


@proves("ext.dreaming-identity-card", "read.identity-card-read")
def test_the_user_card_is_grammar_valid_built_from_the_users_memories_and_private(a, b, dream: Dream):
    _ready(dream)
    body = check_card(a, dream.pool_a)
    text = body["card"].lower()
    assert dream.name_a.lower() in text, f"A's card does not carry A's name {dream.name_a}: {body['lines'][:4]}"
    assert dream.name_b.lower() not in text, "A's card carries B's name"
    own_b = check_card(b, dream.pool_b)
    assert dream.name_a.lower() not in own_b["card"].lower(), "B's card carries A's name"
    r = b.get("/v1/extensions/dreaming/card", params={"pool": dream.pool_a})
    assert r.status_code == 403, f"B reading A's private card -> {r.status_code} (want 403)"


@proves("ext.dreaming-identity-card", "read.identity-card-read")
def test_the_project_card_is_a_team_artifact_and_get_card_serves_both_over_mcp(a, b, mcp_a, mcp_b, dream: Dream):
    _ready(dream)
    for client in (a, b):
        check_card(client, dream.pool_p)
    mcp_a.initialize()
    mcp_b.initialize()
    mine = mcp_json(mcp_a, "get_card", {})  # defaults to the caller's own user pool
    assert mine.get("pool") == dream.pool_a and mine["card"] == check_card(a, dream.pool_a)["card"], f"get_card(): {str(mine)[:200]}"
    proj = mcp_json(mcp_b, "get_card", {"project_id": dream.project})
    assert proj.get("pool") == dream.pool_p and proj.get("lines"), f"get_card(project) as B: {str(proj)[:200]}"
    refused = mcp_json(mcp_b, "get_card", {"pool": dream.pool_a})
    assert "not readable" in str(refused.get("error", "")), f"B's get_card(A's pool) answered {refused}"


@proves("read.identity-card-read")
@negative_control
def test_card_check_fails_for_a_pool_without_a_card_and_for_another_users_pool(ctx, a, b, dream: Dream):
    _ready(dream)
    precondition(check_card, a, dream.pool_a, what="A reads its own card")
    expect_check_fails(check_card, a, f"user--{ctx.user_a}-nobody")
    expect_check_fails(check_card, b, dream.pool_a)


@needs("node")
@proves("plugin.identity-cards", "ext.dreaming-identity-card")
def test_the_session_start_hook_shows_the_users_and_the_projects_card(ctx, dream: Dream):
    _ready(dream)
    res = run_hook(
        "session-start", {"hook_event_name": "SessionStart", "source": "startup", "session_id": f"acc-{ctx.run_id}-card"},
        base_url=ctx.base_url, token=ctx.token_a, user_id=ctx.user_a, project_id=dream.project,
    )
    assert res.returncode == 0 and res.data, f"hook exited {res.returncode}: {res.stderr[:160]}"
    text = (res.data.get("hookSpecificOutput") or {}).get("additionalContext", "")
    assert "## Identity Card (user)" in text, "no user identity card in the session-start context"
    assert f"## Identity Card (project: {dream.project})" in text, "no project identity card in the session-start context"
    assert dream.name_a.lower() in text.lower(), "the user card in the hook output does not carry A's name"
    assert dream.name_b.lower() not in text.lower(), "B's name leaked into A's session-start context"


@needs("node")
@proves("plugin.identity-cards")
@negative_control
def test_hook_card_check_fails_for_user_b(ctx, dream: Dream):
    _ready(dream)
    res = run_hook(
        "session-start", {"hook_event_name": "SessionStart", "source": "startup", "session_id": f"acc-{ctx.run_id}-card0"},
        base_url=ctx.base_url, token=ctx.token_b, user_id=ctx.user_b, project_id=dream.project,
    )

    own = run_hook(
        "session-start", {"hook_event_name": "SessionStart", "source": "startup", "session_id": f"acc-{ctx.run_id}-card1"},
        base_url=ctx.base_url, token=ctx.token_a, user_id=ctx.user_a, project_id=dream.project,
    )

    def check(r):
        text = ((r.data or {}).get("hookSpecificOutput") or {}).get("additionalContext", "")
        assert dream.name_a.lower() in text.lower(), "A's identity is not in the context"

    precondition(check, own, what="A's own session-start context carries A's identity")
    expect_check_fails(check, res)


# ── the vault on the shared volume ───────────────────────────────


@needs("compose")
@proves("ext.dreaming-librarian-vault")
def test_the_shared_pool_is_rendered_as_vault_pages_and_the_private_pool_is_not(ctx, dream: Dream):
    _ready(dream)
    safe = re.sub(r"[^\w-]", "-", dream.project).strip("-")
    page_dir = f"/data/vault/Projects/{safe}"
    sources = [vault_grep(ctx, dream.ids[k], page_dir) for k in ("sh1", "sh2", "sh3", "sh4")]
    assert any(sources), f"no topic page under {page_dir} lists any of the shared memories as a source"
    pages = {p for hits in sources for p in hits}
    assert all(p.endswith(".md") for p in pages), f"non-markdown vault files: {sorted(pages)[:3]}"
    cards = [p for p in vault_grep(ctx, "Maintained by the dreaming sweep", page_dir) if p.endswith("/Card.md")]
    assert cards, f"no Card.md rendered under {page_dir}"
    # pool isolation (spec 4.2): user A's PRIVATE pool is never rendered into the operator's topic
    # pages. (The per-pool dream diary under Dreams/ is a separate review surface and is not part of this.)
    for key in ("id1", "old", "pat1"):
        for area in ("Projects", "Me", "Knowledge", "Home.md"):
            leaked = vault_grep(ctx, dream.written[key].marker, f"/data/vault/{area}") + vault_grep(ctx, dream.ids[key], f"/data/vault/{area}")
            assert not leaked, f"A's private memory {key} appears in the vault under {area}: {leaked[:2]}"


@needs("compose")
@proves("ext.dreaming-librarian-vault")
@negative_control
def test_vault_check_fails_in_a_directory_that_holds_no_pages(ctx, dream: Dream):
    _ready(dream)

    safe = re.sub(r"[^\w-]", "-", dream.project).strip("-")

    def check(directory):
        assert vault_grep(ctx, dream.ids["sh1"], directory), "no page lists the memory"

    precondition(check, f"/data/vault/Projects/{safe}", what="the project's own pages list the shared memory")
    expect_check_fails(check, "/data/vault/Projects/acc-no-such-project")


@needs("compose")
@proves("ext.compiler-obsidian-dual-write")
def test_a_shared_write_is_mirrored_into_the_vault_raw_log_and_a_private_one_is_not(ctx, dream: Dream):
    _ready(dream)
    marker = dream.written["sh1"].marker

    def mirrored():
        hits = vault_grep(ctx, marker)
        assert hits, f"{marker} is in no vault file"
        assert any("/_raw/" in h for h in hits), f"{marker} found only outside _raw/: {hits[:3]}"

    eventually(mirrored, timeout=60, interval=3, what="shared write mirrored into the vault")
    assert not vault_grep(ctx, dream.written["id1"].marker, "/data/vault/_raw"), "a private memory was mirrored into the vault raw log"


@needs("compose")
@proves("ext.compiler-obsidian-dual-write")
@negative_control
def test_mirror_check_fails_for_the_private_marker(ctx, dream: Dream):
    _ready(dream)

    def check(key):
        assert vault_grep(ctx, dream.written[key].marker, "/data/vault/_raw"), "marker not mirrored"

    precondition(check, "sh1", what="a shared write is mirrored into the vault raw log")
    expect_check_fails(check, "id1")


# ── rewrites keep their lexical (BM25) leg ───────────────────────


def check_keyword_finds(client: NSClient, rows: list[dict], marker_of: dict[str, str]) -> int:
    """Every live row that still carries its rare marker is found by a marker-only search (top 3).

    A dream rewrite once upserted a dense-only point, dropping the bm25 sparse vector, so a
    merged/rewritten row stopped matching by keyword. Returns how many rows were checked.
    """
    checked, missed = 0, []
    for row in rows:
        marker = marker_of.get(row["id"])
        if not marker or marker not in (row.get("memory") or row.get("content") or ""):
            continue  # the sweep rewrote the text without the marker: nothing lexical to check
        checked += 1
        hits = {h.get("id") for h in client.search(marker, limit=3)}
        if row["id"] not in hits:
            missed.append(row["id"])
    assert checked >= 3, f"only {checked} live rows still carry their marker; keyword check is not meaningful"
    assert not missed, f"marker-only search misses {len(missed)} of {checked} live rows (lost bm25 vector?): {missed[:3]}"
    return checked


@proves("ext.dreaming-deep-consolidate")
def test_keyword_only_search_still_finds_swept_rows(ctx, a, dream: Dream):
    _ready(dream)
    marker_of = {w.memory_id: f"zq-{ctx.run_id}-{k}" for k, w in dream.written.items() if w.memory_id}
    rows = [r for r in list_rows(a) if r["id"] in marker_of]
    eventually(lambda: check_keyword_finds(a, rows, marker_of), timeout=60, interval=3, what="keyword search over swept rows")


@proves("ext.dreaming-deep-consolidate")
@negative_control
def test_keyword_check_fails_for_user_b(ctx, a, b, dream: Dream):
    _ready(dream)
    marker_of = {w.memory_id: f"zq-{ctx.run_id}-{k}" for k, w in dream.written.items() if w.memory_id}
    rows = [r for r in list_rows(a) if r["id"] in marker_of]
    precondition(check_keyword_finds, a, rows, marker_of, what="A's keyword search finds A's swept rows")
    expect_check_fails(check_keyword_finds, b, rows, marker_of)
