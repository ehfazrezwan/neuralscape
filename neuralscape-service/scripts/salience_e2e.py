"""End-to-end exercise of A4 salience dynamics + the A3-lite settling guard.

SAFE BY CONSTRUCTION: runs against a dedicated Qdrant collection
(``QDRANT_COLLECTION`` must be overridden — the script refuses the default
collection), a dedicated test user pool, and a temp vault. Cleans up after
itself, including its rows in the global ``dreaming:dyn`` / trace hashes.

Usage (from neuralscape-service/, with Neo4j/Redis/Qdrant up):

    set -a; source ../.env; set +a
    QDRANT_COLLECTION=salience_e2e \
    DREAMING_E2E_USER=salience-e2e \
    uv run python scripts/salience_e2e.py

Asserts the §A4/§A3 contract end to end:

1. a real ``service.search()`` writes salience-dynamics state through the
   fire-and-forget trace thread (hot path integration);
2. a frequently co-recalled OLD pair resists PRUNE nomination while its
   never-recalled, equally old sibling decays below the threshold —
   nomination only, nothing is deleted (guardrail 2);
3. the faded sibling is still RETURNED by recall at the k=0 default —
   salience never gates retrieval (guardrail 1);
4. a pool with a fresh write defers with status ``"settling"``;
5. ``force=true`` bypasses the settling guard and the pool dreams.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Keep the sweep lean (no vault/reflection LLM work) and force the dreaming
# feature on for the settling assertions. Env vars beat .env values, and
# these run BEFORE the config modules are imported below.
os.environ.setdefault("DREAMING_ENABLED", "true")
os.environ.setdefault("DREAMING_REFLECTION_ENABLED", "false")
os.environ.setdefault("DREAMING_VAULT_PAGES_ENABLED", "false")
os.environ.setdefault("DREAMING_OBSIDIAN_VAULT_PATH", "/tmp/salience-e2e-vault")

# Overridable so parallel E2E runs (e.g. two feature branches sharing the
# same backing Redis) don't contend on one pool's gate/lock keys.
USER = os.environ.get("DREAMING_E2E_USER", "salience-e2e")
POOL = f"user--{USER}"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))


def seed(service) -> dict[str, str]:
    """Seed the pool. Returns name → memory_id. All rows except ``fresh``
    are back-dated 200 days so the retention math is deterministic."""
    rows = {
        # the co-recalled pair (old, but recall keeps them salient)
        "pair_a": ("architecture", "The websocket relay terminates TLS at the edge proxy before fanout"),
        "pair_b": ("architecture", "Relay fanout workers pull from the edge proxy over plain TCP internally"),
        # equally old, never recalled → should decay into PRUNE candidacy
        "control": ("domain_knowledge", "The legacy invoice PDF generator uses wkhtmltopdf 0.12 under Xvfb"),
    }
    ids: dict[str, str] = {}
    for name, (category, content) in rows.items():
        res = service.store_raw(
            content=content, user_id=USER, category=category, scope="global",
            source_type="explicit", confidence=0.9, visibility="private",
            add_to_graph=False,
        )
        if isinstance(res, tuple):
            res = res[0]
        ids[name] = res[0].id
        time.sleep(0.05)
    return ids


def backdate(service, memory_ids: list[str], days: int) -> None:
    """Rewrite created_at/updated_at so decay math sees genuinely old rows."""
    from config import settings

    old_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    client = service._memory.vector_store.client
    for mid in memory_ids:
        client.set_payload(
            collection_name=settings.qdrant_collection,
            payload={"created_at": old_iso, "updated_at": old_iso},
            points=[mid],
        )


async def main() -> int:
    from config import settings

    if settings.qdrant_collection == "neuralscape_memories":
        print("REFUSING to run against the default collection. Set QDRANT_COLLECTION.")
        return 2

    import redis as redis_lib

    from extensions.dreaming import consolidate
    from extensions.dreaming.config import dreaming_settings
    from extensions.dreaming.sweep import dream_all
    from extensions.dreaming.traces import log_recall, read_dynamics
    from memory_service import MemoryService

    print(f"collection={settings.qdrant_collection} pool={POOL} "
          f"settling={dreaming_settings.settling_minutes}m "
          f"dynamics_enabled={dreaming_settings.dynamics_enabled}")
    service = MemoryService()
    service._get_memory()
    r = redis_lib.Redis.from_url(settings.redis_url)

    # start clean: this pool's dreaming keys from any prior run
    for pat in (f"dreaming:gate:{POOL}*", f"dreaming:lock:{POOL}*",
                f"dreaming:staged_ids:{POOL}*", f"dreaming:essential:{POOL}*"):
        for k in r.scan_iter(pat):
            r.delete(k)

    print("\n[1/6] seeding + back-dating …")
    ids = seed(service)
    backdate(service, [ids["pair_a"], ids["pair_b"], ids["control"]], days=200)
    print(f"  seeded {len(ids)} memories, back-dated 200 days")

    print("\n[2/6] hot path writes dynamics state (fire-and-forget) …")
    # limit=2 keeps the semantically-distant control row out of this recall:
    # its whole point is to stay disused. (search() may still interleave live
    # graph-edge rows — those ids just collect harmless, TTL'd trace state.)
    hits = service.search(query="websocket relay TLS termination", user_id=USER, limit=2)
    hit_ids = [h.id for h in hits if h.id]
    check("control row was not recalled (stays disused)",
          ids["control"] not in hit_ids, f"hits={hit_ids}")
    deadline = time.time() + 5
    states = {}
    while time.time() < deadline and not states:
        time.sleep(0.5)  # trace writes ride a daemon thread — give it a beat
        states = read_dynamics(r, hit_ids)
    check("search() persisted dynamics state via the trace thread",
          bool(states), f"{len(states)}/{len(hit_ids)} returned ids have state")

    print("\n[3/6] co-recall reinforcement of the pair …")
    pair = [ids["pair_a"], ids["pair_b"]]
    for q in ("how does the relay terminate TLS",
              "edge proxy fanout topology",
              "internal transport between proxy and workers"):
        log_recall(pair, q)  # the exact call service.search() makes
    deadline = time.time() + 5
    pair_states = {}
    while time.time() < deadline and len(pair_states) < 2:
        time.sleep(0.5)
        pair_states = read_dynamics(r, pair)
    check("pair carries co-recall dynamics",
          len(pair_states) == 2
          and all(s.co_recall_count >= 3 and s.strength > 1.2 for s in pair_states.values()),
          ", ".join(f"strength={s.strength:.2f} co={s.co_recall_count}"
                    for s in pair_states.values()) or "no states")

    print("\n[4/6] staging: co-recalled pair resists pruning, disused twin is nominated …")
    pools = await asyncio.to_thread(consolidate.enumerate_pools, service)
    batch = pools.get(POOL)
    if batch is None:
        check("pool enumerated", False, f"{POOL} missing from {sorted(pools)}")
    else:
        staged = await asyncio.to_thread(
            consolidate.stage_pool, batch, r,
            last_dreamt_at=time.time() - 86400,   # everything back-dated is "old"
            max_memories=200,
            strength_half_life_days=dreaming_settings.strength_half_life_days,
            prune_strength_threshold=dreaming_settings.prune_strength_threshold,
            dynamics_enabled=dreaming_settings.dynamics_enabled,
        )
        by_id = {m["memory_id"]: m for m in staged.memories}
        control = by_id.get(ids["control"])
        check("disused twin nominated for PRUNE (retention below threshold)",
              control is not None
              and control["retention_strength"] < dreaming_settings.prune_strength_threshold,
              f"retention={control['retention_strength']:.3f}" if control else "not staged")
        pair_staged = [n for n in ("pair_a", "pair_b") if ids[n] in by_id]
        check("co-recalled pair resists PRUNE nomination", not pair_staged,
              f"staged={pair_staged}" if pair_staged else "neither pair row nominated")
        check("staged rows expose salience for the vault",
              control is not None and "salience" in control
              and control["salience"] == control["retention_strength"])
        # guardrail 2 sanity: staging wrote nothing — the nominee is untombstoned
        check("nomination wrote nothing (no tombstone on the nominee)",
              control is not None and not control.get("dream_tombstoned"))

    print("\n[5/6] guardrail 1: the faded twin is still retrievable (k=0 default) …")
    faded_hits = service.search(query="invoice PDF generator wkhtmltopdf", user_id=USER, limit=5)
    check("faded-but-relevant memory still returned by recall",
          ids["control"] in {h.id for h in faded_hits},
          f"{len(faded_hits)} hits")

    print("\n[6/6] settling guard + force bypass …")
    fresh = service.store_raw(
        content="Fresh mid-conversation note: still drafting the relay migration plan",
        user_id=USER, category="task_context", scope="global",
        source_type="explicit", confidence=0.8, visibility="private",
        add_to_graph=False,
    )
    fresh_id = (fresh[0] if isinstance(fresh, tuple) else fresh)[0].id
    run = await dream_all(service=service, dry_run=True, only_pool=POOL, force=False)
    rep = run.pools[0] if run.pools else None
    check("fresh write defers the pool with status 'settling'",
          rep is not None and rep.status == "settling",
          f"{rep.status}: {rep.reason}" if rep else "no pool report")

    run2 = await dream_all(service=service, dry_run=True, only_pool=POOL, force=True)
    rep2 = run2.pools[0] if run2.pools else None
    check("force=true bypasses the settling guard",
          rep2 is not None and rep2.status == "dreamt",
          f"{rep2.status}: {rep2.reason}" if rep2 else "no pool report")

    # ── cleanup ── (include recalled graph-edge ids that picked up trace state)
    all_ids = list({*ids.values(), fresh_id, *hit_ids,
                    *[h.id for h in faded_hits if h.id]})
    try:
        for pat in (f"dreaming:gate:{POOL}*", f"dreaming:lock:{POOL}*",
                    f"dreaming:staged_ids:{POOL}*", f"dreaming:essential:{POOL}*"):
            for k in r.scan_iter(pat):
                r.delete(k)
        r.hdel("dreaming:dyn", *all_ids)
        r.hdel("dreaming:tr:count", *all_ids)
        r.hdel("dreaming:tr:last", *all_ids)
        for mid in all_ids:
            r.delete(f"dreaming:tr:q:{mid}")
        print(f"\ncleaned up redis keys for {POOL}")
    except Exception as exc:
        print(f"\ncleanup warning (redis): {exc}")
    try:
        service._memory.vector_store.client.delete_collection(settings.qdrant_collection)
        print(f"cleaned up collection {settings.qdrant_collection}")
    except Exception as exc:
        print(f"cleanup warning (collection): {exc}")

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'='*60}\nE2E: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
