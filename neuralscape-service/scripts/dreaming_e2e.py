"""End-to-end exercise of the dreaming extension against live services.

SAFE BY CONSTRUCTION: runs against a dedicated Qdrant collection
(``QDRANT_COLLECTION`` must be overridden to a throwaway name before
invoking — the script refuses to run against the default collection),
a temp vault dir, and a dedicated test user pool. Cleans up after itself.

Usage (from neuralscape-service/, with Neo4j/Redis/Qdrant up):

    set -a; source ../.env; set +a
    QDRANT_COLLECTION=dreaming_e2e \
    DREAMING_ENABLED=true \
    DREAMING_OBSIDIAN_VAULT_PATH=/tmp/dreaming-e2e-vault \
    uv run python scripts/dreaming_e2e.py

Seeds one private pool with: a near-duplicate pair (older row holds a
unique detail), a contradiction pair, a passed future-dated plan, a
3-memory pattern, an acknowledgment noise row, and a fake secret. Then:

1. dry-run sweep → actions planned, nothing written
2. real sweep    → merge/invalidate/prune/reframe applied, reflection stored
3. verification  → recall excludes tombstones, includes the reflection;
                   secret hard-deleted; diary written; DreamRun in Redis;
                   A1 provenance: merge survivor stamped with derived_from,
                   insight self-labeled with an epistemic_level, and
                   get_reasoning_chain resolves the insight back to the
                   seeded premise memories
4. second sweep (no force) → gated (volume) — the gate economy holds
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

USER = "dreamer-e2e"
POOL = f"user--{USER}"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))


def seed(service) -> dict[str, str]:
    """Store the pathological seed set. Returns name → memory_id."""
    rows = {
        # near-duplicate pair — the OLDER one holds the unique detail
        # (exactly the case the old semantic dedup cron got wrong)
        "dup_old": ("preference", "User prefers Python 3.12 for backend development, especially with FastAPI"),
        "dup_new": ("preference", "The user likes to use Python 3.12 for backend work"),
        # contradiction pair
        "contra_old": ("decision", "The demo service is deployed as a single-node setup only"),
        "contra_new": ("decision", "As of late June 2026, the demo service is deployed on a managed orchestration cluster via IaC tooling"),
        # passed future date → temporal reframe
        "future": ("task_context", "User is planning to write the dreaming-mode spec in early July 2026"),
        # 3-memory pattern → reflection substrate
        "pat1": ("workflow", "After restarting the Neuralscape service, the user runs the full MCP tool suite as a smoke test"),
        "pat2": ("workflow", "The user verifies async memory writes by recalling them later instead of blocking on the write"),
        "pat3": ("workflow", "Before merging a PR, the user re-runs the full unit test suite locally"),
        # noise → prune
        "noise": ("task_context", "ok got it"),
        # fake secret → hard delete
        "secret": ("task_context", "The staging API key is sk-e2e-FAKE-abc123456789 for testing"),
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
        time.sleep(0.05)  # distinct created_at ordering (older → newer)
    return ids


def fetch_payload(service, mid: str) -> dict:
    from config import settings

    client = service._memory.vector_store.client
    pts = client.retrieve(
        collection_name=settings.qdrant_collection, ids=[mid],
        with_payload=True, with_vectors=False,
    )
    return dict(pts[0].payload) if pts else {}


async def main() -> int:
    from config import settings

    if settings.qdrant_collection == "neuralscape_memories":
        print("REFUSING to run against the default collection. Set QDRANT_COLLECTION.")
        return 2

    from memory_service import MemoryService
    from extensions.dreaming.config import dreaming_settings
    from extensions.dreaming.sweep import dream_all, get_last_run

    print(f"collection={settings.qdrant_collection} vault={dreaming_settings.dreams_dir}")
    service = MemoryService()
    service._get_memory()

    # start clean: drop any prior e2e collection + redis keys
    import redis as redis_lib

    r = redis_lib.Redis.from_url(settings.redis_url)
    for pat in ("dreaming:gate:user--dreamer-e2e*", "dreaming:lock:user--dreamer-e2e*",
                "dreaming:staged_ids:user--dreamer-e2e*"):
        for k in r.scan_iter(pat):
            r.delete(k)

    print("\n[1/5] seeding pathological memories …")
    ids = seed(service)
    print(f"  seeded {len(ids)} memories into pool {POOL}")

    print("\n[2/5] dry-run sweep …")
    run = await dream_all(service=service, dry_run=True, only_pool=POOL, force=True)
    pool_report = run.pools[0] if run.pools else None
    planned = pool_report.applied + pool_report.reported if pool_report else 0
    check("dry-run planned actions", planned > 0, f"{planned} actions")
    check("dry-run wrote nothing",
          not fetch_payload(service, ids["noise"]).get("metadata", {}).get("dream_tombstoned"))

    print("\n[3/5] real sweep …")
    run = await dream_all(service=service, dry_run=False, only_pool=POOL, force=True)
    report = run.pools[0]
    print(f"  status={report.status} staged={report.staged} applied={report.applied} "
          f"reported={report.reported} insights={report.insights} errors={report.errors}")
    check("sweep completed", report.status == "dreamt", report.status)
    check("actions applied", report.applied > 0, str(report.applied))

    print("\n[4/5] verifying store state …")
    # merge: one of the dup pair tombstoned, survivor holds the unique detail
    dup_payloads = {n: fetch_payload(service, ids[n]) for n in ("dup_old", "dup_new")}
    tomb = [n for n, p in dup_payloads.items()
            if (p.get("metadata") or {}).get("dream_tombstoned")]
    live = [n for n in ("dup_old", "dup_new") if n not in tomb]
    merged_ok = len(tomb) == 1 and "fastapi" in (
        dup_payloads[live[0]].get("data", "").lower() if live else ""
    )
    check("duplicate merged, unique detail preserved", merged_ok,
          f"tombstoned={tomb}, survivor text keeps FastAPI detail")

    # A1 provenance: the merge survivor's derived_from records the loser id(s)
    surv_meta = (dup_payloads[live[0]].get("metadata") or {}) if live else {}
    surv_derived = surv_meta.get("derived_from") or []
    check("merge survivor stamped with derived_from",
          bool(tomb) and {ids[n] for n in tomb} <= set(surv_derived),
          f"derived_from={surv_derived}")

    contra_meta = fetch_payload(service, ids["contra_old"]).get("metadata") or {}
    check("contradiction invalidated (old row tombstoned, superseded_by set)",
          bool(contra_meta.get("dream_tombstoned")),
          f"superseded_by={contra_meta.get('superseded_by', '—')!r}")

    future_p = fetch_payload(service, ids["future"])
    future_meta = future_p.get("metadata") or {}
    reframed = bool(future_meta.get("dream_temporal_reframed")) or (
        "plan" not in future_p.get("data", "").lower()
    )
    check("temporal reframe", reframed, future_p.get("data", "")[:80])

    noise_meta = fetch_payload(service, ids["noise"]).get("metadata") or {}
    check("noise pruned (tombstoned)", bool(noise_meta.get("dream_tombstoned")))

    check("secret hard-deleted", fetch_payload(service, ids["secret"]) == {})

    # recall path: tombstones excluded, reflection retrievable
    hits = service.search(query="how the user deploys neuralscape", user_id=USER, limit=10)
    hit_ids = {h.id for h in hits}
    check("recall excludes tombstoned rows", ids["contra_old"] not in hit_ids)
    refl = service.search(query="how does the user verify their work", user_id=USER, limit=10)
    dream_hits = [h for h in refl if getattr(h, "source_type", None) == "dream"]
    check("reflection recallable via normal search", len(dream_hits) > 0,
          dream_hits[0].memory[:80] if dream_hits else "none returned")

    # A1 provenance: the insight self-labels its epistemic level and its
    # reasoning chain resolves back to the seeded premise memories.
    if dream_hits:
        # Prefer a vector hit: its id is a real memory id the chain can walk
        # (a graph edge uuid enriched with source_type="dream" is not).
        insight = next(
            (h for h in dream_hits if getattr(h, "source", None) == "vector"),
            dream_hits[0],
        )
        level = getattr(insight, "epistemic_level", None)
        check("insight carries an epistemic_level",
              level in ("deductive", "inductive", "reflection"), str(level))
        chain = service.get_reasoning_chain(insight.id)
        kids = (chain or {}).get("children") or []
        seeded_ids = set(ids.values())
        resolved = [k for k in kids
                    if k.get("content") and k["memory_id"] in seeded_ids]
        check("reasoning chain resolves insight → seeded premises",
              chain is not None and len(resolved) >= 2,
              f"{len(resolved)} of {len(kids)} premises resolved to seeds")
    else:
        check("insight carries an epistemic_level", False, "no dream insight returned")
        check("reasoning chain resolves insight → seeded premises", False,
              "no dream insight returned")

    diary = dreaming_settings.dreams_dir / "user-dreamer-e2e.md"
    check("diary written", diary.exists(), str(diary))
    check("DreamRun in Redis", (get_last_run() or {}).get("run_id") == run.run_id)

    print("\n[5/5] second sweep without force → gate economy …")
    run2 = await dream_all(service=service, dry_run=False, only_pool=POOL, force=False)
    r2 = run2.pools[0] if run2.pools else None
    check("second sweep gated", r2 is not None and r2.status in ("gated", "skipped_unchanged"),
          f"{r2.status}: {r2.reason}" if r2 else "no pool report")

    # cleanup
    try:
        service._memory.vector_store.client.delete_collection(settings.qdrant_collection)
        print(f"\ncleaned up collection {settings.qdrant_collection}")
    except Exception as exc:
        print(f"\ncleanup warning: {exc}")

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'='*60}\nE2E: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
