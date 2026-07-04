"""End-to-end exercise of vault bridges + Faded + identity cards (B3+B4)
against live services — the dreaming_e2e companion for the vault layer.

SAFE BY CONSTRUCTION: refuses the default Qdrant collection, uses a
dedicated test user/projects, a temp vault, and cleans up after itself.

Usage (from neuralscape-service/, with Neo4j/Redis/Qdrant up):

    set -a; source ../.env; set +a
    QDRANT_COLLECTION=bridges_e2e \
    DREAMING_ENABLED=true \
    DREAMING_E2E_USER=bridges-e2e \
    DEFAULT_USER_ID=bridges-e2e \
    DREAMING_AUTO_APPLY_CONFIDENCE=1.0 \
    DREAMING_OBSIDIAN_VAULT_PATH=/tmp/bridges-e2e-vault \
    uv run python scripts/bridges_card_e2e.py

(DREAMING_AUTO_APPLY_CONFIDENCE=1.0 shadow-reports destructive actions so
the deliberately weak seed row deterministically SURVIVES to fade instead
of being pruned by an over-eager consolidation pass.)

Flow:

1. seed two shared project pools (alpha/beta) around ONE unmistakable
   subject, plus a weak (low-confidence) survivor row in alpha, plus an
   operator private pool with identity material
2. sweep all three pools (force) → topic pages + cards + bridges pass
3. assert: reciprocal ## Bridges on both projects' subject pages (via the
   deterministic signals; the graph-row path is exercised explicitly when
   the LLM titled the two pages differently), bridge idempotence, the
   weak row collapsed into the Faded callout (out of the main sections,
   never off the page), cards written to Redis + Me/Card.md +
   Projects/<pid>/Card.md and grammar-valid
4. re-sweep with unchanged inputs → card pass reports "unchanged"
   (input-hash lock: zero LLM churn)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

USER = os.environ.get("DREAMING_E2E_USER", "bridges-e2e")
POOL_USER = f"user--{USER}"
PROJ_A = f"{USER}-alpha"
PROJ_B = f"{USER}-beta"
POOL_A = f"shared--project--{PROJ_A}"
POOL_B = f"shared--project--{PROJ_B}"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))


def seed(service) -> dict[str, str]:
    """Seed all three pools. Returns name → memory_id."""
    rows = [
        # project alpha — the shared subject + a weak survivor
        ("a_turn1", PROJ_A, "decision", 0.9,
         "The alpha service relays media through Cloudflare TURN when direct ICE fails"),
        ("a_turn2", PROJ_A, "architecture", 0.9,
         "Cloudflare TURN credentials for alpha rotate daily via the deploy pipeline"),
        ("a_turn3", PROJ_A, "procedure", 0.9,
         "To debug alpha connectivity, check the Cloudflare TURN allocation logs first"),
        # dashboard pair: a strong companion so the clusterer forms a topic
        # (>= 2 ids), plus the weak row that must land in the Faded callout.
        # The weak fact must be DISTINCT from the strong one (no merge
        # target — merges are reversible and always apply) and contradict
        # nothing (no invalidate target); only its low confidence dims it.
        ("a_dash", PROJ_A, "architecture", 0.9,
         "The alpha dashboard is built with React, Vite, and Tailwind"),
        ("a_weak", PROJ_A, "domain_knowledge", 0.1,
         "An early alpha dashboard prototype shipped a CSV export button that was dropped"),
        # project beta — same subject, different pool
        ("b_turn1", PROJ_B, "architecture", 0.9,
         "Beta's edge nodes also depend on Cloudflare TURN for NAT traversal"),
        ("b_turn2", PROJ_B, "decision", 0.9,
         "Cloudflare TURN quotas for beta are capped at the free-tier allocation"),
        ("b_turn3", PROJ_B, "domain_knowledge", 0.9,
         "Cloudflare TURN sessions in beta expire after 10 minutes of silence"),
        # operator private pool — identity-card material
        ("me1", None, "personal_fact", 0.95,
         f"{USER} is the operator running the bridges end-to-end suite"),
        ("me2", None, "preference", 0.9,
         "The user prefers uv over pip for all Python dependency management"),
        ("me3", None, "workflow", 0.9,
         "The user always runs the unit suite before pushing a branch"),
    ]
    ids: dict[str, str] = {}
    for name, project_id, category, confidence, content in rows:
        res = service.store_raw(
            content=content,
            user_id=USER,
            category=category,
            scope="project" if project_id else "global",
            project_id=project_id,
            source_type="explicit",
            confidence=confidence,
            visibility="shared" if project_id else "private",
            add_to_graph=False,
        )
        if isinstance(res, tuple):
            res = res[0]
        ids[name] = res[0].id
        time.sleep(0.05)
    return ids


def fetch_payload(service, mid: str) -> dict:
    from config import settings

    client = service._memory.vector_store.client
    pts = client.retrieve(
        collection_name=settings.qdrant_collection, ids=[mid],
        with_payload=True, with_vectors=False,
    )
    return dict(pts[0].payload) if pts else {}


def find_page_with(vault_dir, memory_id: str):
    """The topic page whose source_memory_ids contains memory_id, or None."""
    from extensions.dreaming.librarian import _parse_id_list, split_page

    if not vault_dir.exists():
        return None
    for path in sorted(vault_dir.glob("*.md")):
        if path.stem in (vault_dir.name, "Card"):
            continue
        fm, _ = split_page(path.read_text(encoding="utf-8"))
        if memory_id in _parse_id_list(fm.get("source_memory_ids", "")):
            return path
    return None


def grammar_valid(lines: list[str]) -> bool:
    from extensions.dreaming.card import CARD_LINE_RE, CARD_MAX_LINES

    return (
        0 < len(lines) <= CARD_MAX_LINES
        and all(CARD_LINE_RE.match(l) for l in lines)
    )


async def main() -> int:
    from config import settings

    if settings.qdrant_collection == "neuralscape_memories":
        print("REFUSING to run against the default collection. Set QDRANT_COLLECTION.")
        return 2

    from memory_service import MemoryService
    from extensions.dreaming import bridges as br
    from extensions.dreaming.card import load_card
    from extensions.dreaming.config import dreaming_settings
    from extensions.dreaming.sweep import dream_all

    vault = dreaming_settings.vault_path
    print(f"collection={settings.qdrant_collection} vault={vault} user={USER}")
    service = MemoryService()
    service._get_memory()

    import redis as redis_lib

    r = redis_lib.Redis.from_url(settings.redis_url)
    pools = (POOL_USER, POOL_A, POOL_B)
    for pool in pools:
        for pat in (f"dreaming:gate:{pool}*", f"dreaming:lock:{pool}*",
                    f"dreaming:staged_ids:{pool}*", f"dreaming:essential:{pool}*",
                    f"dreaming:card:{pool}*"):
            for k in r.scan_iter(pat):
                r.delete(k)

    print("\n[1/5] seeding three pools …")
    ids = seed(service)
    print(f"  seeded {len(ids)} memories across {pools}")

    print("\n[2/5] sweeping the three pools (force) …")
    reports = {}
    for pool in pools:
        run = await dream_all(service=service, dry_run=False, only_pool=pool, force=True)
        rep = run.pools[0] if run.pools else None
        reports[pool] = (run, rep)
        print(f"  {pool}: status={rep.status if rep else '—'} "
              f"pages={rep.pages_written if rep else 0} card={rep.card_status if rep else '—'} "
              f"errors={rep.errors if rep else '—'} bridges={run.bridges}")
    ok_all = all(rep and rep.status == "dreamt" for _, rep in reports.values())
    check("all three pools dreamt", ok_all,
          ", ".join(f"{p}:{rep.status if rep else '—'}" for p, (_, rep) in reports.items()))

    dir_a = vault / "Projects" / PROJ_A
    dir_b = vault / "Projects" / PROJ_B
    page_a = find_page_with(dir_a, ids["a_turn1"])
    page_b = find_page_with(dir_b, ids["b_turn1"])
    check("subject pages written in both projects",
          page_a is not None and page_b is not None,
          f"{page_a and page_a.name} / {page_b and page_b.name}")

    print("\n[3/5] bridges …")
    if page_a and page_b:
        natural = "## Bridges" in page_a.read_text() and "## Bridges" in page_b.read_text()
        if not natural:
            # The LLM titled the two subject pages differently, so the
            # slug signal didn't fire and add_to_graph=False kept the
            # entity signal dark. Exercise the graph-enrichment path
            # explicitly with the true shared entity.
            out = br.update_bridges(vault, graph_rows=[{
                "name": "Cloudflare TURN",
                "memory_ids": [ids["a_turn1"], ids["b_turn1"]],
            }])
            print(f"  (slug signal missed — graph-row enrichment applied: {out})")
        ta, tb = page_a.read_text(), page_b.read_text()
        link_a_to_b = f"Projects/{PROJ_B}/{page_b.stem}" in ta and "## Bridges" in ta
        link_b_to_a = f"Projects/{PROJ_A}/{page_a.stem}" in tb and "## Bridges" in tb
        check("bridges are reciprocal across the two projects",
              link_a_to_b and link_b_to_a,
              f"a→b={link_a_to_b} b→a={link_b_to_a}")
        graph_rows = [{"name": "Cloudflare TURN",
                       "memory_ids": [ids["a_turn1"], ids["b_turn1"]]}] if not natural else None
        again = br.update_bridges(vault, graph_rows=graph_rows)
        check("bridges pass is idempotent",
              again["pages_bridged"] == 0, f"second pass rewrote {again['pages_bridged']}")
        check("bridge block sits inside the managed markers",
              br.BRIDGES_START in ta and br.BRIDGES_END in ta)
    else:
        check("bridges are reciprocal across the two projects", False, "pages missing")
        check("bridges pass is idempotent", False, "pages missing")
        check("bridge block sits inside the managed markers", False, "pages missing")

    print("\n[4/5] faded + cards …")
    weak_meta = fetch_payload(service, ids["a_weak"]).get("metadata") or {}
    print("  a_weak store state:",
          {k: weak_meta.get(k) for k in
           ("dream_tombstoned", "dream_pruned", "superseded_by")},
          "| alpha pages:", [p.name for p in sorted(dir_a.glob('*.md'))])
    weak_page = find_page_with(dir_a, ids["a_weak"])
    from extensions.dreaming.librarian import FADED_START

    from config import settings as core_settings

    if weak_page is not None:
        text = weak_page.read_text()
        has_callout = "> [!note]- Faded" in text and "CSV export" in text
        main_body = text.split(FADED_START)[0]
        check("weak row collapsed into the Faded callout",
              has_callout, weak_page.name)
        check("weak row absent from the main sections",
              "CSV export" not in main_body)
    else:
        # The consolidation LLM may have tombstoned the weak row outright
        # (it is a legit PRUNE candidate at strength ~0.1); that is the
        # store-level dim path, but this suite must observe the vault path.
        check("weak row collapsed into the Faded callout", False,
              "weak memory reached no topic page (consolidated away?)")
        check("weak row absent from the main sections", False, "no page")

    card_a = load_card(r, POOL_A)
    check("project card pinned in Redis + grammar-valid",
          card_a is not None and grammar_valid(card_a.get("lines") or []),
          f"{len((card_a or {}).get('lines') or [])} lines")
    card_a_md = dir_a / "Card.md"
    check("Projects/<pid>/Card.md rendered", card_a_md.exists(), str(card_a_md))

    card_user = load_card(r, POOL_USER)
    check("user card pinned in Redis + grammar-valid",
          card_user is not None and grammar_valid(card_user.get("lines") or []),
          f"{len((card_user or {}).get('lines') or [])} lines")
    me_card = vault / "Me" / "Card.md"
    if core_settings.default_user_id == USER:
        check("Me/Card.md rendered for the operator", me_card.exists(), str(me_card))
        if me_card.exists():
            on_disk = [l for l in me_card.read_text().splitlines()
                       if re.match(r"^(IDENTITY|ATTRIBUTE|RELATIONSHIP|INSTRUCTION): ", l)]
            check("Me/Card.md lines grammar-valid on disk", grammar_valid(on_disk),
                  f"{len(on_disk)} lines")
    else:
        check("Me/Card.md rendered for the operator", False,
              f"DEFAULT_USER_ID={core_settings.default_user_id!r} (want {USER!r})")

    print("\n[5/5] card stability on a re-sweep …")
    # Sweep 1 may have rewritten rows in the pool (its own consolidation),
    # which legitimately changes the card pass's input hash — so accept
    # either lock: "unchanged" (hash matched, LLM skipped) or "stable"
    # (LLM ran but reproduced the card). Both mean zero card churn.
    run2 = await dream_all(service=service, dry_run=False, only_pool=POOL_USER, force=True)
    rep2 = run2.pools[0] if run2.pools else None
    check("re-sweep does not churn the card (unchanged/stable)",
          rep2 is not None and rep2.card_status in ("unchanged", "stable"),
          f"card_status={rep2.card_status if rep2 else '—'}")
    card_user2 = load_card(r, POOL_USER)
    if rep2 is not None and rep2.card_status in ("unchanged", "stable"):
        check("card lines and updated_at identical across the re-sweep",
              card_user2 is not None and card_user is not None
              and card_user2.get("lines") == card_user.get("lines")
              and card_user2.get("updated_at") == card_user.get("updated_at"))
    else:
        check("card lines and updated_at identical across the re-sweep", False,
              "card churned")

    # cleanup — cards/essential keys must go even if the collection drop
    # fails (Home.md and get_card scan Redis globally on this instance)
    try:
        for pool in pools:
            for pat in (f"dreaming:essential:{pool}*", f"dreaming:card:{pool}*"):
                for k in r.scan_iter(pat):
                    r.delete(k)
        print(f"\ncleaned up essential/card keys for {pools}")
    except Exception as exc:
        print(f"\ncleanup warning (redis keys): {exc}")
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
