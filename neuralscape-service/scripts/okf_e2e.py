"""End-to-end exercise of OKF edge interop (G1) against live services.

SAFE BY CONSTRUCTION: refuses the default Qdrant collection, uses
dedicated e2e collections/users, a temp vault, and cleans up after
itself (set OKF_E2E_KEEP=1 to keep the collections for inspection).

Usage (from neuralscape-service/, with Neo4j/Redis/Qdrant up):

    set -a; source ../.env; set +a
    QDRANT_COLLECTION=okf_e2e \
    DREAMING_ENABLED=true \
    DREAMING_E2E_USER=okf-e2e \
    DEFAULT_USER_ID=okf-e2e \
    DREAMING_AUTO_APPLY_CONFIDENCE=1.0 \
    DREAMING_REFLECTION_ENABLED=false \
    DREAMING_OBSIDIAN_VAULT_PATH=/tmp/okf-e2e-vault \
    uv run python scripts/okf_e2e.py

Flow:

1. seed a shared project pool + the operator's private pool (including a
   marker "secret" private row)
2. dream sweep (force) → the vault renders as an OKF bundle: §9
   conformance over the whole vault, root version marker, §7 log.md
3. export the default bundle (everything the identity reads) and a
   shared-only bundle → conformance on both; the private marker row must
   be absent from the shared bundle BY CONSTRUCTION
4. re-ingest the exported bundle into a SECOND isolated collection via
   the bundle walker → recall parity on the distilled facts, source_ref
   resolving to {bundle path, concept id}
5. if the knowledge-catalog clone is present, ingest its GA4 sample
   bundle and assert concepts are queryable
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

USER = os.environ.get("DREAMING_E2E_USER", "okf-e2e")
PROJ = f"{USER}-alpha"
POOL_USER = f"user--{USER}"
POOL_PROJ = f"shared--project--{PROJ}"
REIMPORT_USER = f"{USER}-reimport"
REIMPORT_COLLECTION = os.environ.get("OKF_E2E_REIMPORT_COLLECTION", "okf_e2e_reimport")
SECRET = "ZANZIBAR-OCELOT private launch codes for the okf e2e suite"
SAMPLE_BUNDLE = Path("/tmp/knowledge-catalog/okf/bundles/ga4")

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))


def seed(service) -> dict[str, str]:
    rows = [
        ("turn1", PROJ, "architecture", "shared",
         "The alpha service relays media through Cloudflare TURN when direct ICE fails"),
        ("turn2", PROJ, "decision", "shared",
         "Cloudflare TURN quotas for alpha are capped at the free-tier allocation"),
        ("turn3", PROJ, "procedure", "shared",
         "To debug alpha connectivity, inspect the Cloudflare TURN allocation logs first"),
        ("pref", None, "preference", "private",
         "The user prefers uv over pip for all Python dependency management"),
        ("fact", None, "personal_fact", "private",
         f"{USER} operates the OKF end-to-end suite"),
    ]
    ids: dict[str, str] = {}
    for name, project_id, category, visibility, content in rows:
        res = service.store_raw(
            content=content,
            user_id=USER,
            category=category,
            scope="project" if project_id else "global",
            project_id=project_id,
            source_type="explicit",
            confidence=0.9,
            visibility=visibility,
            add_to_graph=False,
        )
        if isinstance(res, tuple):
            res = res[0]
        ids[name] = res[0].id
        time.sleep(0.05)
    return ids


def unzip_to(data: bytes, target: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(target)


async def main() -> int:
    from config import settings

    if settings.qdrant_collection == "neuralscape_memories":
        print("REFUSING to run against the default collection. Set QDRANT_COLLECTION.")
        return 2
    if REIMPORT_COLLECTION == "neuralscape_memories":
        print("REFUSING the default collection as reimport target.")
        return 2

    from okf import conformance, translate
    from okf.export import export_bundle
    from extensions.dreaming.config import dreaming_settings
    from extensions.dreaming.sweep import dream_all
    from memory_service import MemoryService

    vault = dreaming_settings.vault_path
    if str(vault).startswith("/tmp") and vault.exists():
        shutil.rmtree(vault)
    print(f"collection={settings.qdrant_collection} reimport={REIMPORT_COLLECTION} "
          f"vault={vault} user={USER}")

    # Drop stale e2e collections from prior runs.
    from qdrant_client import QdrantClient

    qc = QdrantClient(url=settings.qdrant_url)
    for coll in (settings.qdrant_collection, REIMPORT_COLLECTION):
        if qc.collection_exists(coll):
            qc.delete_collection(coll)

    import redis as redis_lib

    r = redis_lib.Redis.from_url(settings.redis_url)
    for pool in (POOL_USER, POOL_PROJ):
        for pat in (f"dreaming:gate:{pool}*", f"dreaming:lock:{pool}*",
                    f"dreaming:staged_ids:{pool}*", f"dreaming:essential:{pool}*",
                    f"dreaming:card:{pool}*"):
            for k in r.scan_iter(pat):
                r.delete(k)

    service = MemoryService()
    service._get_memory()

    print("\n[1/5] seeding …")
    ids = seed(service)
    print(f"  seeded {len(ids)} memories")

    print("\n[2/5] dream sweep → vault as OKF bundle …")
    for pool in (POOL_PROJ, POOL_USER):
        run = await dream_all(service=service, dry_run=False, only_pool=pool, force=True)
        rep = run.pools[0] if run.pools else None
        print(f"  {pool}: status={rep.status if rep else '—'} "
              f"pages={rep.pages_written if rep else 0} errors={rep.errors if rep else '—'}")
        check(f"pool {pool} dreamt", bool(rep and rep.status == "dreamt"))

    problems = conformance.check_bundle(vault, ignore=("_raw",))
    check("vault is §9-conformant", problems == [], "; ".join(problems[:3]))
    root_index = vault / "index.md"
    if root_index.exists():
        fm, _ = translate.parse_document(root_index.read_text())
        check("root index carries the okf_version marker",
              translate.has_version_marker(fm), translate.declared_version(fm) or "—")
    else:
        check("root index carries the okf_version marker", False, "index.md missing")
    log_path = vault / "log.md"
    log_ok = log_path.exists() and conformance.check_files(
        {"log.md": log_path.read_text()}
    ) == []
    check("vault log.md follows §7", bool(log_ok))

    print("\n[3/5] export bundles …")
    # Seed the private marker AFTER the sweep: it exists purely to prove
    # shared-bundle exclusion, and seeding it earlier lets the
    # consolidation pass (correctly) prune it as a secret.
    res = service.store_raw(
        content=SECRET, user_id=USER, category="personal_fact", scope="global",
        source_type="explicit", visibility="private", add_to_graph=False,
    )
    ids["secret"] = (res[0] if isinstance(res, tuple) else res)[0].id

    full_zip, full_stats = export_bundle(service, user_id=USER)
    shared_zip, shared_stats = export_bundle(service, user_id=USER, visibility="shared")
    print(f"  full: {full_stats}  shared: {shared_stats}")
    # The sweep may legitimately consolidate rows (merge/rewrite), so exact
    # counts aren't stable — the invariant is private rows ride ONLY in the
    # full bundle.
    check("full export strictly larger than shared (private rows ride)",
          full_stats["concepts"] > shared_stats["concepts"])

    tmp = Path(tempfile.mkdtemp(prefix="okf-e2e-"))
    full_dir, shared_dir = tmp / "full", tmp / "shared"
    unzip_to(full_zip, full_dir)
    unzip_to(shared_zip, shared_dir)
    check("full bundle §9-conformant", conformance.check_bundle(full_dir) == [])
    check("shared bundle §9-conformant", conformance.check_bundle(shared_dir) == [])
    shared_text = "\n".join(
        p.read_text() for p in shared_dir.rglob("*.md")
    )
    check("private memory ABSENT from shared bundle", "ZANZIBAR-OCELOT" not in shared_text)
    check("shared bundle still carries team facts", "Cloudflare TURN" in shared_text)
    full_text = "\n".join(p.read_text() for p in full_dir.rglob("*.md"))
    check("full bundle carries the caller's private memories",
          "ZANZIBAR-OCELOT" in full_text and "uv" in full_text.lower())
    check("envelope extension keys ride in concept frontmatter",
          "epistemic_level" in full_text or "confidence" in full_text or "memory_id" in full_text)

    print("\n[4/5] re-ingest the exported bundle into a second collection …")
    settings.qdrant_collection = REIMPORT_COLLECTION  # all service2 calls read this
    service2 = MemoryService()
    service2._get_memory()

    from ingest.okf_bundle import default_type_llm, ingest_okf_bundle, is_okf_bundle, load_bundle_dir

    files = load_bundle_dir(full_dir)
    check("exported bundle detected as OKF", is_okf_bundle(files))
    summary = ingest_okf_bundle(
        service2,
        files=files,
        bundle_uri=str(full_dir),
        user_id=REIMPORT_USER,
        llm_call=default_type_llm(service2),
    )
    print(f"  reimport: concepts={summary['concepts']} passages={summary['passages']} "
          f"facts={summary['facts']} links={summary['links']}")
    check("all exported concepts re-ingested", summary["concepts"] == full_stats["concepts"])
    check("distilled facts produced on re-ingest", summary["facts"] > 0)

    parity_queries = [
        ("Cloudflare TURN relay fallback for alpha", "TURN"),
        ("preferred Python dependency manager", "uv"),
    ]
    for query, token in parity_queries:
        hits = service2.search(query, user_id=REIMPORT_USER, limit=8)
        hit = next((h for h in hits if token.lower() in (h.memory or "").lower()), None)
        check(f"recall parity: {query!r}", hit is not None,
              (hit.memory[:70] + "…") if hit else f"{len(hits)} hits, none mention {token!r}")
    # source_ref resolves back to {bundle path, concept id}
    hits = service2.search(parity_queries[0][0], user_id=REIMPORT_USER, limit=8)
    ref_hit = next((h for h in hits if h.source_ref), None)
    ref_ok = bool(
        ref_hit
        and ref_hit.source_ref.get("connector_type") == "okf_bundle"
        and ref_hit.source_ref.get("parent_id") == str(full_dir)
        and (full_dir / (ref_hit.source_ref.get("external_id", "") + ".md")).exists()
    )
    check("source_ref resolves to {bundle path, concept id}", ref_ok,
          str((ref_hit.source_ref or {}).get("external_id")) if ref_hit else "no source_ref hit")

    print("\n[5/5] knowledge-catalog GA4 sample bundle …")
    if SAMPLE_BUNDLE.exists():
        ga4_files = load_bundle_dir(SAMPLE_BUNDLE)
        check("GA4 sample detected as OKF", is_okf_bundle(ga4_files))
        ga4 = ingest_okf_bundle(
            service2,
            files=ga4_files,
            bundle_uri=str(SAMPLE_BUNDLE),
            user_id=f"{USER}-ga4",
            extract_facts=False,  # passages suffice for queryability; keeps LLM spend flat
            llm_call=default_type_llm(service2),
        )
        print(f"  ga4: concepts={ga4['concepts']} passages={ga4['passages']} links={ga4['links']}")
        check("GA4 concepts ingested", ga4["concepts"] >= 10)
        hits = service2.search(
            "average pageviews per user metric", user_id=f"{USER}-ga4", limit=8,
        )
        hit = next((h for h in hits if "pageview" in (h.memory or "").lower()), None)
        check("GA4 concepts queryable", hit is not None,
              (hit.memory[:70] + "…") if hit else f"{len(hits)} hits")
        concept_ids = {rel[:-3] for rel in ga4_files if rel.endswith(".md")}
        ref = (hit.source_ref or {}) if hit else {}
        check("GA4 hit's source_ref carries a real concept id",
              ref.get("external_id") in concept_ids, str(ref.get("external_id")))
    else:
        print("  (skipped — clone /tmp/knowledge-catalog to exercise this)")

    # ── cleanup ──
    if os.environ.get("OKF_E2E_KEEP") != "1":
        for coll in (os.environ.get("QDRANT_COLLECTION", ""), REIMPORT_COLLECTION):
            if coll and qc.collection_exists(coll):
                qc.delete_collection(coll)
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'ALL CHECKS PASSED' if not failed else 'FAILURES:'} "
          f"({len(CHECKS) - len(failed)}/{len(CHECKS)})")
    for name, ok, detail in failed:
        print(f"  ✗ {name} — {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    import asyncio

    sys.exit(asyncio.run(main()))
