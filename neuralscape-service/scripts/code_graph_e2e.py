"""End-to-end exercise of the code-graph adapter (F1) against live services.

SAFE BY CONSTRUCTION: runs against a dedicated Qdrant collection
(``QDRANT_COLLECTION`` must be overridden — the script refuses the default),
an isolated ingest-storage dir, and a dedicated test user. Cleans up after
itself. Requires the ``code-graph`` extra (graphifyy) installed.

Usage (from neuralscape-service/, with Qdrant up + GOOGLE_API_KEY set):

    set -a; source ../.env; set +a
    QDRANT_COLLECTION=codegraph_e2e \
    INGEST_STORAGE_DIR=/tmp/codegraph-e2e-ingest \
    uv run python scripts/code_graph_e2e.py

Flow:
1. store the fixture graphify-out/graph.json as an owner-scoped artifact
   (its id becomes the bundle's graph_id) and run the semantic ingest;
2. ingest the fixture GRAPH_REPORT.md through the normal pipeline under the
   code_graph adapter (section chunker + LLM report extractor);
3. verify memories are recallable (vector search), carry the epistemic
   mapping, and their source_refs resolve through NS's OWN surface
   (query_code_graph over the artifact graph_id);
4. verify the three delegation queries answer from the ingested fixture;
5. cleanup (collection + artifacts).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

USER = os.environ.get("CODEGRAPH_E2E_USER", "codegraph-e2e")
FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "graphify-out"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    from config import settings

    if settings.qdrant_collection == "neuralscape_memories":
        print("REFUSING to run against the default collection. Set QDRANT_COLLECTION.")
        return 2

    from adapters.code_graph import code_graph_available

    if not code_graph_available():
        print("graphifyy not installed — run `uv sync` (dev group) or the code-graph extra.")
        return 2

    from adapters.code_graph.query import query_code_graph, resolve_graph_path
    from ingest.code_graph import detect_graphify_member, ingest_code_graph_json
    from ingest.pipeline import IngestDoc, ingest_document
    from ingest.storage import artifact_source_ref, store_artifact
    from memory_service import MemoryService

    print(f"collection={settings.qdrant_collection} user={USER} "
          f"storage={settings.ingest_storage_dir}")
    service = MemoryService()
    service._get_memory()

    # ── [1/5] graph.json → artifact + semantic ingest ──
    print("\n[1/5] ingesting graphify-out/graph.json …")
    graph_bytes = (FIXTURES / "graph.json").read_bytes()
    check("bundle detection", detect_graphify_member("graphify-out/graph.json", graph_bytes) == "graph")

    art = store_artifact(graph_bytes, "graph.json", USER, "codegraph-e2e-proj", "module", settings)
    payload = {
        "user_id": USER,
        "source_ref": artifact_source_ref(art, connector_type="file_upload"),
        "options": {"project_id": "codegraph-e2e-proj", "tags": ["codegraph-e2e"]},
    }
    result = ingest_code_graph_json(service, graph_bytes, payload)
    graph_id = result["graph_id"]
    print(f"  graph_id={graph_id} facts={result['facts']} "
          f"(graph {result['graph_nodes']}n/{result['graph_edges']}e)")
    check("semantic facts stored", result["facts"] >= 5, f"{result['facts']} facts")
    check("no raw-graph passages", result["passages"] == 0)
    check("graph jobs deferred (not inline)", len(result["graph_jobs"]) == result["facts"])

    # ── [2/5] GRAPH_REPORT.md through the pipeline under the adapter ──
    print("\n[2/5] ingesting GRAPH_REPORT.md (code_graph adapter) …")
    report_text = (FIXTURES / "GRAPH_REPORT.md").read_text()
    report_art = store_artifact(
        report_text.encode(), "GRAPH_REPORT.md", USER, "codegraph-e2e-proj", "module", settings
    )
    doc = IngestDoc(
        content=report_text,
        source=artifact_source_ref(report_art, connector_type="file_upload"),
        user_id=USER,
        project_id="codegraph-e2e-proj",
        tags=["codegraph-e2e"],
        adapter="code_graph",
    )
    rep = ingest_document(service, doc)
    print(f"  report → {rep['passages']} passages + {rep['facts']} facts (adapter={rep['adapter']})")
    check("report used code_graph adapter", rep["adapter"] == "code_graph")
    check("report passages chunked", rep["passages"] >= 3, f"{rep['passages']} sections")
    # Facts need a live Gemini key; treat 0 as a soft warning, not failure.
    if rep["facts"] == 0:
        print("  (report facts = 0 — LLM extraction unavailable? passages still verify)")

    # ── [3/5] memories recallable + provenance resolves through NS ──
    print("\n[3/5] verifying recall + source_refs …")
    hits = service.search(
        query="why does the code rebuild the whole index", user_id=USER,
        project_id="codegraph-e2e-proj", limit=10,
    )
    rationale_hits = [h for h in hits if (h.category or "") == "rationale"]
    check("rationale memory recallable", bool(rationale_hits),
          rationale_hits[0].memory[:70] + "…" if rationale_hits else "no hit")

    hits = service.search(
        query="which class is the core abstraction hotspot of the codebase",
        user_id=USER, project_id="codegraph-e2e-proj", limit=10,
    )
    hotspot_hits = [h for h in hits if (h.category or "") == "hotspot"]
    check("hotspot memory recallable", bool(hotspot_hits),
          hotspot_hits[0].memory[:70] + "…" if hotspot_hits else "no hit")

    # Pull a stored payload to inspect the envelope.
    client = service._memory.vector_store.client
    mid = result["memory_ids"][0]
    pts = client.retrieve(collection_name=settings.qdrant_collection, ids=[mid],
                          with_payload=True, with_vectors=False)
    meta = (dict(pts[0].payload).get("metadata") or {}) if pts else {}
    ref = meta.get("source_ref") or {}
    check("epistemic_level stamped", meta.get("epistemic_level") in
          ("explicit", "deductive", "inductive"), str(meta.get("epistemic_level")))
    check("source_ref → NS surface", ref.get("connector_type") == "code_graph"
          and (ref.get("retrieval") or {}).get("tool") == "query_code_graph"
          and (ref.get("retrieval") or {}).get("mcp_server") == "neuralscape",
          str((ref.get("retrieval") or {}).get("args")))
    check("source_ref carries graph_id", (ref.get("retrieval") or {}).get("args", {})
          .get("graph_id") == graph_id, str(graph_id))

    # The retrieval handle must actually resolve: graph_id → the stored artifact.
    try:
        resolved = resolve_graph_path(graph_id, USER, settings)
        check("graph_id resolves to artifact", resolved.endswith(".json"), resolved)
    except Exception as exc:  # noqa: BLE001
        check("graph_id resolves to artifact", False, str(exc))
    # …and owner-scoping holds: another user cannot resolve it.
    try:
        resolve_graph_path(graph_id, "someone-else", settings)
        check("graph_id owner-scoped", False, "foreign user resolved the graph!")
    except Exception:  # noqa: BLE001
        check("graph_id owner-scoped", True)

    # ── [4/5] delegation queries answer from the fixture ──
    print("\n[4/5] query_code_graph / neighbors / path over graph_id …")
    from adapters.code_graph.query import code_path, get_code_neighbors

    out = query_code_graph("who calls MemoryEngine", user_id=USER,
                           settings=settings, graph_id=graph_id)
    check("query_code_graph answers", "MemoryEngine" in out, out.splitlines()[0][:80])
    out = get_code_neighbors("MemoryEngine", user_id=USER, settings=settings, graph_id=graph_id)
    check("get_code_neighbors shows confidence tags",
          "[INFERRED]" in out and "[EXTRACTED]" in out)
    out = code_path("audit_log", "MemoryEngine", user_id=USER, settings=settings, graph_id=graph_id)
    check("code_path traces hops", "2 hops" in out and "rebuild_index()" in out, out.splitlines()[-1][:80])

    # Re-ingest idempotency: same bytes → dedup hits, no new graph jobs.
    print("\n[5/5] re-ingest idempotency + cleanup …")
    again = ingest_code_graph_json(service, graph_bytes, payload)
    check("re-ingest is idempotent (no new graph jobs)", again["graph_jobs"] == [],
          f"facts echoed={again['facts']}")

    # cleanup
    try:
        client.delete_collection(settings.qdrant_collection)
        print(f"  dropped collection {settings.qdrant_collection}")
    except Exception as exc:  # noqa: BLE001
        print(f"cleanup warning (collection): {exc}")
    try:
        import shutil

        user_dir = Path(os.path.expanduser(settings.ingest_storage_dir)) / USER
        if user_dir.is_dir():
            shutil.rmtree(user_dir)
            print(f"  removed artifacts under {user_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"cleanup warning (artifacts): {exc}")

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'ALL CHECKS PASSED' if not failed else 'FAILURES:'} "
          f"({len(CHECKS) - len(failed)}/{len(CHECKS)})")
    for name, ok, detail in failed:
        print(f"  ✗ {name} — {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
