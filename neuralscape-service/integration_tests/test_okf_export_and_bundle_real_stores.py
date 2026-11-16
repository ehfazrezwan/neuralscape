"""Real-store proof: OKF export is never silently cut, bundle import reports partial failures.

Real Qdrant, a throwaway collection (``stores``, ``llm_key``, ``flags``,
``unique_ns``; every row is dropped afterwards), the real embedder and the real
ingest path (fact extraction off, so no extraction LLM spend).

* A pool larger than the old 5000-row cap is exported whole; with a smaller
  explicit cap the export says it is incomplete with the exact counts (stats and
  the bundle's ``log.md``). Rows beyond the few real ones are clones of a real
  stored point (payload and vector), so no embedding is paid per row.
* A bundle with one deliberately broken concept imports the rest and reports
  ``partial`` with the broken concept's id; a bundle of only broken concepts
  reports ``failed``.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import copy
import io
import uuid
import zipfile

import pytest


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns):
    from qdrant_client import QdrantClient

    flags(qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url)
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    return service


def _clone(svc, source_id, n):
    from qdrant_client.models import PointStruct

    from config import settings

    client = svc._memory.vector_store.client
    (src,) = client.retrieve(
        collection_name=settings.qdrant_collection, ids=[source_id], with_payload=True, with_vectors=True,
    )
    for start in range(0, n, 500):
        points = []
        for i in range(start, min(n, start + 500)):
            payload = copy.deepcopy(src.payload)
            payload["data"] = f"{src.payload['data']} clone {i}"
            points.append(PointStruct(id=str(uuid.uuid4()), vector=src.vector, payload=payload))
        client.upsert(collection_name=settings.qdrant_collection, points=points)


def test_export_beyond_the_old_cap_is_complete_and_a_smaller_cap_is_flagged(svc, unique_ns):
    from okf.export import export_bundle

    uid = unique_ns.user_id
    out = svc.store_raw(
        content=f"The zq{uuid.uuid4().hex[:6]} billing service deploys on Fridays.", user_id=uid,
        category="convention", scope="global", visibility="private", add_to_graph=False,
    )
    _clone(svc, out[0].id, 5100)

    data, stats = export_bundle(svc, user_id=uid, visibility="private")
    assert stats["concepts"] == 5101 and stats["truncated"] is False  # past the old 5000 cut

    data, stats = export_bundle(svc, user_id=uid, visibility="private", limit=1000)
    assert stats["concepts"] == 1000 and stats["truncated"] is True
    assert stats["truncated_pools"] == {"personal": {"exported": 1000, "total": 5101}}
    log = zipfile.ZipFile(io.BytesIO(data)).read("log.md").decode()
    assert "INCOMPLETE" in log and "personal pool 1000 of 5101" in log


def _bundle(*concepts):
    files = {"index.md": "---\nokf_version: 0.1\n---\n"}
    for name, body in concepts:
        files[f"{name}.md"] = f"---\ntype: Topic\ntitle: {name}\n---\n\n{body}\n"
    return files


def _break_on(svc, marker):
    """Make the real store reject content carrying ``marker``; everything else is untouched."""
    for attr in ("store_raw", "store_raw_batch"):
        original = getattr(svc, attr)

        def wrapped(*args, _orig=original, **kwargs):
            blob = repr((args, kwargs))
            if marker in blob:
                raise RuntimeError("store rejected this concept")
            return _orig(*args, **kwargs)

        setattr(svc, attr, wrapped)


def test_bundle_with_one_broken_concept_is_partial_on_the_real_ingest_path(svc, unique_ns):
    from ingest.okf_bundle import ingest_okf_bundle

    tag = uuid.uuid4().hex[:6]
    files = _bundle(
        ("alpha", f"Alpha knowledge zq{tag} about queues."),
        ("broken", f"BROKENCONCEPT{tag} knowledge."),
        ("gamma", f"Gamma knowledge zq{tag} about caches."),
    )
    _break_on(svc, f"BROKENCONCEPT{tag}")

    out = ingest_okf_bundle(
        svc, files=files, bundle_uri="it.zip", user_id=unique_ns.user_id,
        extract_facts=False, llm_call=lambda p: "{}",
    )

    assert out["status"] == "partial", out
    assert out["concepts"] == 2 and out["failed"] == 1
    assert [f["concept_id"] for f in out["failed_concepts"]] == ["broken"]
    assert out["memory_ids"], "the healthy concepts were stored"


def test_bundle_of_only_broken_concepts_is_failed(svc, unique_ns):
    from ingest.okf_bundle import ingest_okf_bundle

    tag = uuid.uuid4().hex[:6]
    files = _bundle(("one", f"BROKENCONCEPT{tag} one."), ("two", f"BROKENCONCEPT{tag} two."))
    _break_on(svc, f"BROKENCONCEPT{tag}")

    out = ingest_okf_bundle(
        svc, files=files, bundle_uri="it.zip", user_id=unique_ns.user_id,
        extract_facts=False, llm_call=lambda p: "{}",
    )
    assert out["status"] == "failed" and out["concepts"] == 0 and out["failed"] == 2
