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
from integration_tests._dreaming_support import dream_env  # noqa: F401  (fixture)

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
OTHER_SEED = [
    ("preference", "User prefers Rust for systems work and keeps a {tok} cheat sheet"),
    ("preference", "The user likes using Rust for low-level systems programming"),
    ("workflow", "The user profiles every release build before shipping"),
    ("workflow", "The user benchmarks the hot path after each refactor"),
    ("workflow", "The user reviews flamegraphs before merging performance changes"),
]
PROJECT_SEED = [
    "The project backend is a FastAPI service using Python 3.12",
    "The project stores vectors in Qdrant and graph facts in Neo4j",
    "The project convention is to run the full unit suite before every merge",
    "The project deploys through a container image built in CI",
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
    proj_ids = [env.store_project(t, pid) for t in PROJECT_SEED]
    # A second private user in the same collection: isolation fixtures.
    other_user = env.user + "-other"
    other_tok = "zq" + uuid.uuid4().hex[:10]
    other_pool = f"user--{other_user}"
    env.extra_pools.append(other_pool)
    other_ids = [env.store(t.format(tok=other_tok), c, user=other_user) for c, t in OTHER_SEED]
    env.clear_pool_state()

    client = env.service._memory.vector_store.client
    client.set_payload(
        collection_name=env.collection,
        payload={"code_liveness_stale": True, "code_liveness_reason": "anchor validate_card was removed",
                 "code_liveness_anchor": "billing.py::validate_card"},
        points=[ids["live"]], key="metadata",
    )
    settings = env.sweep_settings()
    before_noise = env.meta(ids["noise"]).get("dream_tombstoned")

    dry = asyncio.run(dream_all(service=env.service, settings=settings, dry_run=True, force=True))
    after_dry = {n: env.payload(i) for n, i in ids.items()}
    real = asyncio.run(dream_all(service=env.service, settings=settings, dry_run=False, force=True))
    reports = {p.pool: p for p in real.pools}
    return SimpleNamespace(
        ids=ids, proj_ids=proj_ids, other_user=other_user, other_tok=other_tok,
        other_pool=other_pool, other_ids=other_ids, pid=pid, project_pool=f"shared--project--{pid}",
        settings=settings, dry=dry, real=real, reports=reports,
        before_noise=before_noise, after_dry=after_dry,
    )


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
    seeded = set(swept.ids.values())
    for r in recs:
        md = r.payload["metadata"]
        assert md.get("epistemic_level") in ("deductive", "inductive", "reflection"), md.get("epistemic_level")
        derived = md.get("derived_from") or []
        assert derived and set(derived) <= seeded | set(swept.proj_ids), f"bad derived_from {derived}"
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
    assert prep.pages_written > 0 and hub.exists(), f"project hub missing: {hub}"
    topics = [p for p in (vault / "Projects" / swept.pid).glob("*.md") if p.stem not in (swept.pid, "Card")]
    assert topics, "no project topic pages"
    assert f"[[{swept.pid}]]" in topics[0].read_text(encoding="utf-8"), "topic page does not link its hub"


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


def _vault_text(env) -> str:
    return "\n".join(
        p.read_text(encoding="utf-8", errors="ignore") for p in env.vault.rglob("*") if p.is_file()
    )


def test_other_users_private_pool_swept_but_never_leaks(dream_env, swept):
    env = dream_env
    rep = swept.reports.get(swept.other_pool)
    assert rep is not None and rep.status == "dreamt", f"other pool not swept: {rep}"
    # never rendered into the operator's vault (pool isolation)
    vault = _vault_text(env)
    assert swept.other_tok not in vault, "other user's private content leaked into the operator vault"
    assert "Rust" not in vault, "other user's private topic leaked into the operator vault"
    # their card exists but is refused to this user, and unreadable via REST identity
    assert load_card(env.redis, swept.other_pool), "other user's card was not produced"
    view = build_card_view(swept.other_pool, env.user, is_dictator=False, redis=env.redis)
    assert view["status"] == "forbidden"
    # recall is user-scoped both ways
    mine = env.service.search(query=f"Rust systems {swept.other_tok}", user_id=env.user, limit=20)
    assert not [h for h in mine if swept.other_tok in h.memory or "rust" in h.memory.lower()]
    theirs = env.service.search(query="Python FastAPI backend", user_id=swept.other_user, limit=20)
    assert not [h for h in theirs if "fastapi" in h.memory.lower()]


def test_insights_only_cite_their_own_pool(dream_env, swept):
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    env = dream_env
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
            "the LLM produced no reflection, so citation isolation is unproven"
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
    vault = _vault_text(env)
    assert swept.other_tok not in vault and "Rust" not in vault
    assert swept.other_pool.split("--", 1)[1] not in vault
