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
* Cross-link memories that fail to store are counted (``link_failures`` /
  ``failed_links``) in the import status, on the real ingest path and through the
  worker job that produces the task result, without failing the import or changing
  its ``status``; the same bundle with a healthy store reports no link failures.
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


LINK_MARKER = "references and relates to"  # the sentence every cross-link memory carries


def _linked_bundle(tag):
    """a -> b and b -> c cross-links; c links nowhere. Two concepts therefore get a link memory."""
    files = {"index.md": "---\nokf_version: 0.1\n---\n"}
    bodies = {
        "alpha": f"Alpha knowledge zq{tag} about queues. See [beta](beta.md).",
        "beta": f"Beta knowledge zq{tag} about caches. See [gamma](gamma.md).",
        "gamma": f"Gamma knowledge zq{tag} about locks.",
    }
    for name, body in bodies.items():
        files[f"{name}.md"] = f"---\ntype: Topic\ntitle: {name}\n---\n\n{body}\n"
    return files


def test_bundle_cross_links_that_fail_to_store_are_counted_without_failing_the_import(svc, unique_ns):
    from ingest.okf_bundle import ingest_okf_bundle

    tag = uuid.uuid4().hex[:6]
    out = ingest_okf_bundle(
        svc, files=_linked_bundle(tag), bundle_uri="it.zip", user_id=unique_ns.user_id,
        extract_facts=False, llm_call=lambda p: "{}",
    )
    # Control: with a healthy store the same bundle stores its cross-links and reports no failure.
    assert out["status"] == "completed" and out["concepts"] == 3, out
    assert out["link_failures"] == 0 and out["failed_links"] == [] and out["links"] == 2, out

    _break_on(svc, LINK_MARKER)
    tag2 = uuid.uuid4().hex[:6]
    out = ingest_okf_bundle(
        svc, files=_linked_bundle(tag2), bundle_uri="it2.zip", user_id=unique_ns.user_id,
        extract_facts=False, llm_call=lambda p: "{}",
    )
    assert out["status"] == "completed", out  # a link is a hint on top of ingested concepts
    assert out["concepts"] == 3 and out["failed"] == 0, out
    assert out["link_failures"] == 2, out
    assert sorted(f["concept_id"] for f in out["failed_links"]) == ["alpha", "beta"], out["failed_links"]
    assert out["links"] == 0, out
    assert all("store rejected" not in f["error"] for f in out["failed_links"]), "raw exception text must not be reported"


def test_the_import_job_result_carries_the_link_failures(svc, unique_ns, monkeypatch):
    import asyncio
    import base64

    import worker

    async def no_graph_jobs(*args, **kwargs):
        return 0

    monkeypatch.setattr(worker, "_enqueue_graph_jobs", no_graph_jobs)
    _break_on(svc, LINK_MARKER)
    tag = uuid.uuid4().hex[:6]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in _linked_bundle(tag).items():
            zf.writestr(name, text)
    payload = {
        "filename": "linked.zip",
        "data_b64": base64.b64encode(buf.getvalue()).decode(),
        "user_id": unique_ns.user_id,
        "options": {"extract_facts": False},
    }
    result = asyncio.run(worker.process_ingest_okf_bundle({"service": svc}, payload))

    assert result["status"] == "completed" and result["concepts"] == 3, result
    assert result["link_failures"] == 2, result
    assert sorted(f["concept_id"] for f in result["failed_links"]) == ["alpha", "beta"], result["failed_links"]
