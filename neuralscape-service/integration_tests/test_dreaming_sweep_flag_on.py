"""Full dreaming sweep, ``DREAMING_ENABLED=true``, real Neo4j/Qdrant/Redis + real LLM.

Seeds an unambiguous pathological pool (near-duplicates, a contradiction,
a passed future plan, a 3-memory pattern, noise, a fake secret, a code-
liveness-stale row) plus a shared project pool, then runs one dry-run and one
real sweep and asserts structure and provenance, never exact LLM wording.
Where the LLM must decide something (merge/invalidate/prune/reframe/...), a
missing decision FAILS with a message that says the LLM produced no action.
Exact apply semantics live in test_dreaming_apply_flag_on.py.

Run (from neuralscape-service/):
    uv run pytest integration_tests/test_dreaming_sweep_flag_on.py -q
Env: NEO4J_URI NEO4J_USER NEO4J_PASSWORD QDRANT_URL REDIS_URL GOOGLE_API_KEY
"""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest

from extensions.dreaming import gate
from extensions.dreaming.card import build_card_view, load_card, resolve_card_pool
from extensions.dreaming.reflect import diary_page_path
from extensions.dreaming.sweep import _STAGED_IDS_KEY, dream_all, get_last_run
from integration_tests._dreaming_support import dream_env, marker_hits, timed  # noqa: F401  (fixture)

pytestmark = pytest.mark.integration

SEED = {
    "dup_old": ("preference", "User prefers Python 3.12 for backend development, especially with FastAPI"),
    "dup_new": ("preference", "The user likes to use Python 3.12 for backend work"),
    "contra_old": ("decision", "The demo service is deployed as a single-node setup only"),
    "contra_new": ("decision", "As of late June 2026, the demo service is deployed on a managed orchestration cluster via IaC tooling"),
    "future": ("task_context", "User is planning to write the dreaming-mode spec in early July 2026"),
    "pat1": ("workflow", "After restarting the Neuralscape service, the user runs the full MCP tool suite as a smoke test"),
    "pat2": ("workflow", "The user verifies async memory writes by recalling them later instead of blocking on the write"),
    "pat3": ("workflow", "Before merging a PR, the user re-runs the full unit test suite locally"),
    "noise": ("task_context", "ok got it"),
    "secret": ("task_context", "The staging API key is sk-e2e-FAKE-abc123456789 for testing"),
    "live": ("task_context", "The payment module validates cards through validate_card() in billing.py"),
}
# A second pool that must reach the reflection pass AND still hold at least two
# live rows after the sweep's own consolidation. The consolidation stage may merge
# near-duplicates, fold a pattern into one row, or prune a row it judges noise or
# credential-like (a gate-cleared secret prune hard-deletes), so these rows are
# deliberately unrelated to each other (no shared topic, no pattern of three) and
# each states a distinct, substantive fact. One carries the leak marker.
OTHER_SEED = [
    ("preference", "User prefers Rust for systems work and keeps a {tok} cheat sheet"),
    ("personal_fact", "The user is a native Portuguese speaker who lives in Lisbon"),
    ("preference", "The user drinks green tea instead of coffee in the morning"),
    ("workflow", "The user profiles every release build before shipping"),
    ("decision", "The user chose a standing desk with a walnut top for the home office"),
]
# The shared project pool. The first four facts are close relatives (one stack, one
# pipeline), so the consolidation stage may legitimately fold them into one row. The
# librarian writes a topic page only from two or more live rows, so the pool also holds
# three facts on clearly different subjects of the same project (a nightly import, who
# may use the admin pages, how long logs are kept): a merge of the first four still
# leaves at least three live rows for a topic page and the hub.
PROJECT_SEED = [
    ("tech_stack", "The project backend is a FastAPI service using Python 3.12"),
    ("tech_stack", "The project stores vectors in Qdrant and graph facts in Neo4j"),
    ("convention", "The project convention is to run the full unit suite before every merge"),
    ("workflow", "The project deploys through a container image built in CI"),
    ("workflow", "The project imports the partner price list every night at two o'clock from an SFTP drop"),
    ("decision", "Access to the project admin pages is limited to members of the finance operations group"),
    ("convention", "The project keeps audit logs for ninety days and then archives them to cold storage"),
]


def _diary(env) -> str:
    from extensions.dreaming.config import dreaming_settings

    path = diary_page_path(dreaming_settings.dreams_dir, env.pool)
    return path.read_text(encoding="utf-8") if path.exists() else ""


@pytest.fixture(scope="module")
def swept(dream_env):
    env = dream_env
    env.clear_pool_state()
    ids = {name: env.store(text, cat) for name, (cat, text) in SEED.items()}
    pid = f"nsit-proj-{uuid.uuid4().hex[:6]}"
    proj_ids = [env.store_project(t, pid, cat) for cat, t in PROJECT_SEED]
    # A second private user in the same collection: isolation fixtures.
    other_user = env.user + "-other"
    other_tok = "zq" + uuid.uuid4().hex[:10]
    other_pool = f"user--{other_user}"
    env.extra_pools.append(other_pool)
    other_ids = [env.store(t.format(tok=other_tok), c, user=other_user) for c, t in OTHER_SEED]
    env.clear_pool_state()

    client = env.service._memory.vector_store.client
    with timed(env, "set_payload (liveness flag)"):
        client.set_payload(
            collection_name=env.collection,
            payload={"code_liveness_stale": True, "code_liveness_reason": "anchor validate_card was removed",
                     "code_liveness_anchor": "billing.py::validate_card"},
            points=[ids["live"]], key="metadata",
        )
    settings = env.sweep_settings()
    before_noise = env.meta(ids["noise"]).get("dream_tombstoned")

    with timed(env, "dream_all dry run (enumerate + hydrate + LLM)"):
        dry = asyncio.run(dream_all(service=env.service, settings=settings, dry_run=True, force=True))
    with timed(env, "read back seeded rows"):
        after_dry = {n: env.payload(i) for n, i in ids.items()}
    with timed(env, "dream_all real sweep (apply + reflect + vault)"):
        real = asyncio.run(dream_all(service=env.service, settings=settings, dry_run=False, force=True))
    reports = {p.pool: p for p in real.pools}
    return SimpleNamespace(
        ids=ids, proj_ids=proj_ids, other_user=other_user, other_tok=other_tok,
        other_pool=other_pool, other_ids=other_ids, pid=pid, project_pool=f"shared--project--{pid}",
        settings=settings, dry=dry, real=real, reports=reports,
        before_noise=before_noise, after_dry=after_dry,
    )


def _live_project_rows(env, swept) -> int:
    """Seeded rows of the shared project pool that are neither deleted nor tombstoned."""
    return sum(1 for i in swept.proj_ids if env.payload(i) and not _tombstoned(env, i))


def _tombstoned(env, mid) -> bool:
    return bool(env.meta(mid).get("dream_tombstoned"))


def _handled_or_proposed(env, mid) -> bool:
    return _tombstoned(env, mid) or mid in _diary(env)


# ── sweep orchestration ─────────────────────────────────────────────


def test_dry_run_plans_actions_and_writes_nothing(dream_env, swept):
    rep = next(p for p in swept.dry.pools if p.pool == dream_env.pool)
    assert rep.status == "dreamt", f"dry run status {rep.status}: {rep.reason} {rep.errors}"
    assert rep.applied + rep.reported > 0, "LLM planned no action at all on a pathological seed"
    assert not swept.before_noise
    for name, payload in swept.after_dry.items():
        md = payload.get("metadata") or {}
        assert not md.get("dream_tombstoned"), f"dry run tombstoned {name}"
        assert payload.get("data") == (SEED[name][1] if name != "live" else payload["data"]), name


def test_real_sweep_completes_applies_and_persists_dreamrun(dream_env, swept):
    rep = swept.reports[dream_env.pool]
    assert rep.status == "dreamt", f"{rep.status}: {rep.reason} errors={rep.errors}"
    assert rep.applied > 0, "real sweep applied no action"
    assert not rep.errors, rep.errors
    last = get_last_run()
    assert last and last["run_id"] == swept.real.run_id
    assert any(p["pool"] == dream_env.pool for p in last["pools"])
    assert gate.get_gate_state(dream_env.redis, dream_env.pool)["last_dreamt_at"] > 0


def test_status_route_reports_flag_on_and_last_run(dream_env, swept):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from extensions.dreaming import DreamingExtension

    app = FastAPI()
    app.include_router(DreamingExtension().get_routes())
    body = TestClient(app).get("/status").json()
    assert body["enabled"] is True
    assert body["last_run"]["run_id"] == swept.real.run_id


# ── deep consolidate (LLM decisions, tolerant but never vacuous) ────


def test_duplicate_pair_merged_with_provenance_and_keyword_findable(dream_env, swept):
    env, ids = dream_env, swept.ids
    tomb = [n for n in ("dup_old", "dup_new") if _tombstoned(env, ids[n])]
    assert len(tomb) == 1, (
        f"LLM produced no merge for the duplicate pair (tombstoned={tomb}); "
        f"diary: {_diary(env)[:400]}"
    )
    live = "dup_new" if tomb == ["dup_old"] else "dup_old"
    p = env.payload(ids[live])
    meta = p["metadata"]
    assert ids[tomb[0]] in (meta.get("derived_from") or []), "survivor lacks loser in derived_from"
    assert env.meta(ids[tomb[0]]).get("superseded_by") == ids[live]
    assert "fastapi" in p["data"].lower(), "merge dropped the unique detail"
    assert meta.get("dream_prev_content"), "merge rewrite is not reversible"
    # BM25 leg survives the rewrite: keyword-only query finds the survivor
    pt = env.point(ids[live], vectors=True)
    assert isinstance(pt.vector, dict) and pt.vector.get("bm25") is not None, "survivor lost bm25"
    assert ids[live] in env.lexical_ids("FastAPI"), "keyword-only query misses the merged row"


def test_contradiction_invalidated_or_proposed(dream_env, swept):
    env = dream_env
    old = swept.ids["contra_old"]
    assert _handled_or_proposed(env, old), "LLM produced no action for the contradiction pair"
    if _tombstoned(env, old):
        assert env.meta(old).get("superseded_at")


def test_noise_pruned_or_proposed(dream_env, swept):
    assert _handled_or_proposed(dream_env, swept.ids["noise"]), "LLM produced no action for the noise row"


def test_passed_future_plan_reframed(dream_env, swept):
    env = dream_env
    p = env.payload(swept.ids["future"])
    md = p.get("metadata") or {}
    assert md.get("dream_temporal_reframed") or p["data"] != SEED["future"][1] or _tombstoned(
        env, swept.ids["future"]
    ), "LLM produced no temporal reframe for the passed future plan"
    if md.get("dream_temporal_reframed"):
        assert md["dream_prev_content"] == SEED["future"][1]


def test_secret_row_hard_deleted_or_claim_recorded(dream_env, swept):
    env = dream_env
    sid = swept.ids["secret"]
    if env.payload(sid) == {}:
        return  # hard-deleted at/above the confidence gate
    md = env.meta(sid)
    assert md.get("dream_tombstoned") and md.get("dream_secret_claim"), (
        "LLM produced no secret action: row neither deleted nor tombstoned with a claim"
    )


def test_recall_excludes_every_tombstone(dream_env, swept):
    env = dream_env
    tombs = {i for i in swept.ids.values() if env.payload(i) and _tombstoned(env, i)}
    got: set[str] = set()
    for q in ("how the user deploys the demo service", "python backend", "user workflow smoke test",
              "plans and specs", "payment validation", "acknowledgement"):
        got |= {h.id for h in env.service.search(query=q, user_id=env.user, limit=20)}
    assert tombs, "nothing was tombstoned"
    assert not (tombs & got), f"tombstoned rows surfaced in recall: {tombs & got}"


# ── REM reflections ─────────────────────────────────────────────────


def test_reflections_stored_with_provenance_and_diary_written(dream_env, swept):
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    env = dream_env
    rep = swept.reports[env.pool]
    assert rep.insights > 0, f"LLM produced no reflection insight (errors={rep.errors})"
    client = env.service._memory.vector_store.client
    recs, _ = client.scroll(
        collection_name=env.collection, limit=50, with_payload=True, with_vectors=False,
        scroll_filter=Filter(must=[FieldCondition(key="metadata.source_type", match=MatchValue(value="dream"))]),
    )
    assert recs, "insights reported stored but none carries source_type=dream"
    # The collection also holds the second user's insights (the sweep covers
    # every pool), which legitimately cite that user's rows. Each insight must
    # cite only rows of its own pool: pick the allowed set by owner, so a
    # cross-pool citation still fails (the owner-keyed isolation test covers
    # the reverse direction).
    own_pool_ids = set(swept.ids.values()) | set(swept.proj_ids)
    other_pool_ids = set(swept.other_ids)
    assert any(r.payload["metadata"].get("owner_user_id") != swept.other_user for r in recs), (
        "no insight of the operator pools was stored"
    )
    for r in recs:
        md = r.payload["metadata"]
        assert md.get("epistemic_level") in ("deductive", "inductive", "reflection"), md.get("epistemic_level")
        derived = md.get("derived_from") or []
        allowed = other_pool_ids if md.get("owner_user_id") == swept.other_user else own_pool_ids
        assert derived and set(derived) <= allowed, f"bad derived_from {derived} (owner {md.get('owner_user_id')})"
        env.seeded_ids.add(str(r.id))
    assert diary_page_path(env.sweep_settings().dreams_dir, env.pool).exists()
    assert "Reflections" in _diary(env)


# ── identity card ───────────────────────────────────────────────────


def test_identity_card_stored_readable_and_private_to_owner(dream_env, swept):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from extensions.dreaming import DreamingExtension

    env = dream_env
    rep = swept.reports[env.pool]
    assert rep.card_status in ("updated", "stable"), f"card not produced: {rep.card_status!r} {rep.errors}"
    rec = load_card(env.redis, env.pool)
    assert rec and rec["lines"], "no card record in Redis"
    # get_card path (shared contract of the MCP tool and the REST route)
    assert resolve_card_pool(user_id=env.user) == env.pool
    view = build_card_view(env.pool, env.user, is_dictator=False, redis=env.redis)
    assert view["status"] == "ok" and view["lines"] == rec["lines"] and view["card"]
    other = build_card_view(env.pool, "someone-else", is_dictator=False, redis=env.redis)
    assert other["status"] == "forbidden"
    assert build_card_view(env.pool, "someone-else", is_dictator=True, redis=env.redis)["status"] == "ok"
    # REST surface
    app = FastAPI()
    app.include_router(DreamingExtension().get_routes())
    c = TestClient(app)
    ok = c.get("/card", params={"pool": env.pool, "user_id": env.user})
    assert ok.status_code == 200 and ok.json()["lines"] == rec["lines"]
    assert c.get("/card", params={"pool": env.pool, "user_id": "someone-else"}).status_code == 403
    # a card is a pinned artifact, never a searchable memory
    hits = env.service.search(query=rec["lines"][0], user_id=env.user, limit=10)
    assert all(getattr(h, "source_type", None) != "card" for h in hits)


# ── librarian vault ─────────────────────────────────────────────────


def test_vault_topic_pages_home_and_hub_written(dream_env, swept):
    env = dream_env
    vault = env.vault
    home = vault / "Home.md"
    me_pages = [p for p in (vault / "Me").glob("*.md") if p.stem != "Card"] if (vault / "Me").exists() else []
    assert swept.reports[env.pool].pages_written > 0, "librarian wrote no topic page for the private pool"
    assert home.exists(), "Home.md not written"
    assert me_pages, "no Me/ topic pages"
    assert "## Essential Story" in home.read_text(encoding="utf-8")
    prep = swept.reports.get(swept.project_pool)
    assert prep is not None, f"project pool not swept: {list(swept.reports)}"
    assert prep.status == "dreamt", f"{prep.status}: {prep.reason} {prep.errors}"
    hub = vault / "Projects" / swept.pid / f"{swept.pid}.md"
    # The librarian needs two live rows for a topic; say what the sweep did to the pool
    # so a miss can be told apart (rows folded away by consolidation vs a writer defect).
    assert prep.pages_written > 0 and hub.exists(), (
        f"project hub missing: {hub}; sweep report for {swept.project_pool}: staged={prep.staged} "
        f"applied={prep.applied} reported={prep.reported} pages_written={prep.pages_written} "
        f"pages_skipped={prep.pages_skipped} errors={prep.errors}; live seeded rows in the pool: "
        f"{_live_project_rows(env, swept)} of {len(swept.proj_ids)} (the librarian needs at least two; "
        f"fewer means consolidation folded the pool, more means the page writer missed)"
    )
    # index.md / log.md are OKF bundle files (standard markdown links, written
    # by refresh_bundle_indexes), not librarian topic pages: exclude them.
    from okf.translate import is_reserved_filename

    topics = [
        p for p in (vault / "Projects" / swept.pid).glob("*.md")
        if p.stem not in (swept.pid, "Card") and not is_reserved_filename(p.name)
    ]
    assert topics, "no project topic pages"
    for page in topics:
        assert f"[[{swept.pid}]]" in page.read_text(encoding="utf-8"), f"topic page {page.name} does not link its hub"


# ── recall traces ───────────────────────────────────────────────────


def test_search_with_dreaming_on_writes_recall_traces(dream_env, swept):
    from extensions.dreaming import traces

    env = dream_env
    ids = swept.ids
    survivor = next(ids[n] for n in ("dup_old", "dup_new") if not _tombstoned(env, ids[n]))
    q = "which python version does the user prefer for backend work"
    hits = [h.id for h in env.service.search(query=q, user_id=env.user, limit=10)]
    assert survivor in hits or hits, "search returned nothing to trace"
    traced = hits[:1]
    import time

    deadline = time.time() + 10
    agg = {}
    while time.time() < deadline:
        agg = traces.read_aggregates(env.redis, traced)
        if agg[traced[0]]["recall_count"] >= 1:
            break
        time.sleep(0.2)
    assert agg[traced[0]]["recall_count"] >= 1, "recall trace counter not written to Redis"
    assert agg[traced[0]]["unique_query_count"] >= 1
    assert agg[traced[0]]["last_recalled_at"] > 0
    assert traces.read_dynamics(env.redis, traced), "salience dynamics state not written"


# ── code liveness ───────────────────────────────────────────────────


def test_code_liveness_flag_reframed_and_cleared_on_sweep(dream_env, swept):
    env = dream_env
    mid = swept.ids["live"]
    p = env.payload(mid)
    assert p, "liveness row vanished (LLM pruned a code-anchored fact?)"
    md = p["metadata"]
    assert p["data"].startswith("[stale: anchor validate_card was removed]"), p["data"]
    assert md.get("dream_temporal_reframed") is True
    assert md.get("code_liveness_stale") is False, "flag not cleared after the reframe applied"
    assert md.get("dream_prev_content"), "reframe is not reversible (no undo stash)"
    pt = env.point(mid, vectors=True)
    assert isinstance(pt.vector, dict) and pt.vector.get("bm25") is not None, "reframed row lost bm25"


# ── gates ───────────────────────────────────────────────────────────


def test_second_unforced_sweep_is_gated(dream_env, swept):
    env = dream_env
    s = env.sweep_settings(min_hours=24.0)
    run = asyncio.run(dream_all(service=env.service, settings=s, dry_run=False, only_pool=env.pool, force=False))
    assert run.pools[0].status == "gated" and run.pools[0].reason.startswith("time"), run.pools[0]


def test_volume_gate_blocks_a_quiet_pool(dream_env, swept):
    env = dream_env
    s = env.sweep_settings(min_new_memories=500)
    run = asyncio.run(dream_all(service=env.service, settings=s, dry_run=False, only_pool=env.pool, force=False))
    assert run.pools[0].status == "gated" and run.pools[0].reason.startswith("volume"), run.pools[0]


def test_lock_blocks_even_a_forced_sweep(dream_env, swept):
    env = dream_env
    token = gate.acquire_lock(env.redis, env.pool)
    assert token
    try:
        run = asyncio.run(dream_all(service=env.service, settings=env.sweep_settings(), dry_run=False,
                                    only_pool=env.pool, force=True))
        assert run.pools[0].status == "locked"
    finally:
        gate.release_lock(env.redis, env.pool, token)
    assert not gate.is_locked(env.redis, env.pool)


def test_unchanged_staged_id_set_is_skipped_without_llm_work(dream_env, swept):
    """Record the pool's CURRENT staged-id set as the last-swept set, reopen the
    time gate, and the next unforced sweep must skip as unchanged (and so apply
    nothing) instead of paying for another consolidation pass."""
    from extensions.dreaming import consolidate

    env = dream_env
    s = env.sweep_settings()
    pools = consolidate.enumerate_pools(env.service)
    batch = consolidate.hydrate_pool(env.service, pools[env.pool])
    batch = consolidate.stage_pool(
        batch, env.redis, last_dreamt_at=0.0, max_memories=s.max_memories_per_pool,
        strength_half_life_days=s.strength_half_life_days,
        prune_strength_threshold=s.prune_strength_threshold, dynamics_enabled=s.dynamics_enabled,
    )
    staged = sorted(m["memory_id"] for m in batch.memories)
    env.redis.set(_STAGED_IDS_KEY.format(pool=env.pool), json.dumps(staged))
    env.redis.delete(f"dreaming:gate:{env.pool}")  # last_dreamt_at back to 0: gates open
    run = asyncio.run(dream_all(service=env.service, settings=s, dry_run=False, only_pool=env.pool, force=False))
    rep = run.pools[0]
    assert rep.status == "skipped_unchanged", f"{rep.status}: {rep.reason}"
    assert rep.applied == 0 and rep.insights == 0


# ── multi-user isolation (privacy) ──────────────────────────────────


def _is_pool_diary(env, path) -> bool:
    """Exactly the per-pool diary pages ``Dreams/<pool>.md`` of the pools this module swept."""
    from extensions.dreaming.reflect import diary_page_path

    dreams = env.vault / "Dreams"
    names = {diary_page_path(dreams, p).name for p in (env.pool, *env.extra_pools)}
    return path.parent == dreams and path.name in names


# Pool-IDENTIFYING surfaces: the sweep names every pool it dreamt (user id in the
# pool key) in the bundle log and the Dreams index entry for its diary. They carry
# a pool NAME, never the pool's memory text, so they are scanned for content
# markers but not for the other user's id.
_POOL_NAME_SURFACES = ("log.md", "Dreams/index.md")


def _vault_text(env, *, include_diaries: bool = False) -> str:
    """Every vault file except the per-pool diary pages ``Dreams/<pool>.md``.

    Those are written for EVERY pool, other users' private pools included, and
    cite their insight text: the owner accepted that as single-operator by
    design. Everything else, ``Dreams/index.md`` included, must never carry
    another user's private content.
    """
    return "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in sorted(env.vault.rglob("*"))
        if p.is_file() and (include_diaries or not _is_pool_diary(env, p))
    )


def _content_hits(env, markers) -> list[str]:
    """``path:line: text`` of every non-diary vault file holding a content marker."""
    return marker_hits(
        env.vault, markers, skip=lambda rel: _is_pool_diary(env, env.vault / rel),
    )


def _pool_id_hits(env, marker) -> list[str]:
    """Where the other user's id appears outside the diaries and the pool-name surfaces."""
    return marker_hits(
        env.vault, [marker],
        skip=lambda rel: _is_pool_diary(env, env.vault / rel) or rel in _POOL_NAME_SURFACES,
    )


def test_other_users_private_pool_swept_but_never_leaks(dream_env, swept):
    env = dream_env
    rep = swept.reports.get(swept.other_pool)
    assert rep is not None and rep.status == "dreamt", f"other pool not swept: {rep}"
    # never rendered into the operator's vault (pool isolation)
    hits = _content_hits(env, [swept.other_tok, "Rust"])
    assert not hits, "other user's private content in the operator vault:\n" + "\n".join(hits)
    # their card exists but is refused to this user, and unreadable via REST identity
    assert load_card(env.redis, swept.other_pool), "other user's card was not produced"
    view = build_card_view(swept.other_pool, env.user, is_dictator=False, redis=env.redis)
    assert view["status"] == "forbidden"
    # recall is user-scoped both ways. Match by id and by whole word: the shared
    # project pool is team-wide BY DESIGN (it mentions FastAPI and is visible to
    # every user), and "rust" as a substring also matches "trust"/"robust".
    import re

    other_ids, mine_ids = set(swept.other_ids), set(swept.ids.values())
    mine = env.service.search(query=f"Rust systems {swept.other_tok}", user_id=env.user, limit=20)
    leaked = [h for h in mine if str(h.id) in other_ids or swept.other_tok in h.memory
              or re.search(r"\brust\b", h.memory, re.I)]
    assert not leaked, [(str(h.id), h.memory[:80]) for h in leaked]
    theirs = env.service.search(query="Python FastAPI backend", user_id=swept.other_user, limit=20)
    leaked = [h for h in theirs if str(h.id) in mine_ids]
    assert not leaked, [(str(h.id), h.memory[:80]) for h in leaked]


def test_leak_scan_covers_the_dreams_index_but_not_the_pool_diaries(dream_env, swept):
    env = dream_env
    diary = env.vault / "Dreams" / "index.md"
    assert diary.exists(), "Dreams/index.md missing: the scan's coverage cannot be shown"
    assert not _is_pool_diary(env, diary)
    others = [p for p in (env.vault / "Dreams").glob("*.md") if p.name != "index.md"]
    assert others and all(_is_pool_diary(env, p) for p in others)
    assert diary.read_text(encoding="utf-8") in _vault_text(env), "Dreams/index.md is not scanned"


def test_dream_diary_is_written_for_every_pool_by_design(dream_env, swept):
    """Documented contract (single-operator vault): a ``Dreams/<pool>.md`` exists for EVERY
    swept pool, another user's private pool included, and cites that pool's content.

    Pinned so a policy change is a visible test change.
    """
    from extensions.dreaming.reflect import diary_page_path

    env = dream_env
    dreams = env.sweep_settings().dreams_dir
    for pool in (env.pool, swept.other_pool, swept.project_pool):
        page = diary_page_path(dreams, pool)
        assert page.exists(), f"no dream diary page for {pool}"
        assert pool in page.read_text(encoding="utf-8"), f"diary for {pool} does not name its pool"


def test_insights_only_cite_their_own_pool(dream_env, swept, monkeypatch):
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from extensions.dreaming import reflect as reflect_mod

    env = dream_env

    # The reflection LLM is the only nondeterministic step this test depends on
    # (it may legitimately return no insight). Pin that single seam: for each pool
    # the stub cites the pool's OWN staged memories, exactly the shape a model
    # reply would be validated into. Sweep, hydrate, store_insights, Qdrant and the
    # provenance fields stay real, so the isolation property is still what is tested.
    async def _stub_reflect(batch, llm_call, *, max_insights, surprisal_top_k=0):
        ids = [m["memory_id"] for m in batch.memories
               if m.get("source_type") != "dream" and not m.get("dream_tombstoned")]
        assert len(ids) >= 2, f"pool {batch.pool} has too few live memories to cite: {ids}"
        return [{
            "content": f"Deterministic isolation probe insight for {batch.pool} {uuid.uuid4().hex[:8]}",
            "source_memory_ids": ids[:3], "category": "domain_knowledge", "lens": "pattern",
            "epistemic_level": "inductive", "confidence": 0.8,
        }]

    monkeypatch.setattr(reflect_mod, "reflect", _stub_reflect)
    for pool in (env.pool, swept.other_pool):
        # The `swept` fixture already dreamt every pool, which recorded each pool's
        # last_dreamt_at. A second sweep stages only rows that changed since then (or
        # have decayed), and a pool's fresh, strong rows are neither, so it would stage
        # nothing and there would be nothing for the stub to cite. Forget the dreamt
        # state (not the rows) so this sweep stages the pool's rows again.
        env.clear_pool_state()
        run = asyncio.run(dream_all(service=env.service, settings=swept.settings, dry_run=False,
                                    only_pool=pool, force=True))
        assert run.pools and run.pools[0].status == "dreamt", run.pools
        assert run.pools[0].insights >= 1, f"stubbed reflection stored nothing for {pool}: {run.pools[0]}"

    client = env.service._memory.vector_store.client
    recs, _ = client.scroll(
        collection_name=env.collection, limit=100, with_payload=True, with_vectors=False,
        scroll_filter=Filter(must=[FieldCondition(key="metadata.source_type", match=MatchValue(value="dream"))]),
    )
    mine, theirs = set(swept.ids.values()), set(swept.other_ids)
    owners = {r.payload["metadata"].get("owner_user_id") for r in recs}
    for want in (env.user, swept.other_user):
        assert want in owners, (
            f"no insight produced for {want} (insight owners: {sorted(o for o in owners if o)}); "
            "the stubbed reflection stored nothing for this pool"
        )
    for r in recs:
        md = r.payload["metadata"]
        env.seeded_ids.add(str(r.id))
        derived = set(md.get("derived_from") or [])
        owner = md.get("owner_user_id")
        if owner == env.user:
            assert not (derived & theirs), f"insight of {owner} cites the other user's memories"
        elif owner == swept.other_user:
            assert not (derived & (mine | set(swept.proj_ids))), "other user's insight cites foreign memories"
            assert md.get("visibility") == "private"


def test_bridges_pass_runs_with_flag_on(dream_env, swept):
    env = dream_env
    s = env.sweep_settings(bridges_enabled=True)
    run = asyncio.run(dream_all(service=env.service, settings=s, dry_run=False,
                                only_pool=swept.project_pool, force=True))
    assert run.pools and run.pools[0].status == "dreamt", run.pools
    assert "error" not in run.bridges, run.bridges
    assert set(run.bridges) >= {"pages_bridged", "pages_unchanged", "links"}, run.bridges
    # Isolation after the bridges pass: it only reads the operator's vault,
    # which never holds the other user's private pool, so nothing of theirs
    # can be linked or rendered. (Proving a *bridged* page needs graph entities
    # shared across hubs, which the graph LLM extracts nondeterministically:
    # out of scope here.)
    hits = _content_hits(env, [swept.other_tok, "Rust"])
    assert not hits, "other user's private content after the bridges pass:\n" + "\n".join(hits)
    hits = _pool_id_hits(env, swept.other_pool.split("--", 1)[1])
    assert not hits, "other user's id outside the diaries, log.md and Dreams/index.md:\n" + "\n".join(hits)
