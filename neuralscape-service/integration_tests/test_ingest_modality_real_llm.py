"""Issue #444 on real stores.

``test_stored_label_follows_the_modality_check`` is the deterministic proof: the fact
extractor is stubbed to return a fact that upgrades the source's "should" to "must" and a
faithful twin, ``ingest_document`` runs against a real Qdrant collection with the real
embedder, and the STORED ``epistemic_level`` is read back (``inductive`` vs ``explicit``).
It fails if the post-check is removed or not wired in.

``test_real_llm_smoke`` is only a smoke test: the model is nondeterministic, so it asserts
that the real extraction path stores facts and that none is both flagged by the check and
labelled explicit. It cannot prove the check on its own (the check decides the label).

Needs Neo4j (service init), Qdrant, GOOGLE_API_KEY (embeddings; the smoke test also extraction).
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import pytest

pytestmark = pytest.mark.integration

WORKSHOP = (
    "The design review requires compatibility evidence before release.\n"
    "Workshop transcript: Mina says that session leads should capture an observation before suggesting a "
    "follow-up experiment. Jules replies that each follow-up should have a named owner. Ivo notes that the "
    "whiteboard is erased after the workshop, so key observations need durable text. Mina asks that unanswered "
    "questions stay labelled as open instead of becoming claims. Jules confirms that a later session may revise "
    "a tentative finding; keep the original source attached. Ivo says all participants should review the "
    "written summary before treating it as agreed."
)
UPGRADED = "Session leads must capture observations before suggesting follow-up experiments."
FAITHFUL = "Each follow-up should have a named owner."


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns):
    from qdrant_client import QdrantClient

    flags(qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url, extraction_require_speaker=False)
    unique_ns.on_cleanup(lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection))
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    return service


def _facts(svc, uid):
    rows = [r["payload"] for r in svc._scroll_all_user_memories(uid)]

    def meta(row, key):
        return (row.get("metadata") or {}).get(key) or row.get(key)

    return [(r.get("data") or "", meta(r, "epistemic_level")) for r in rows if meta(r, "memory_kind") == "fact"]


def _doc(uid):
    from ingest.pipeline import IngestDoc

    return IngestDoc(content=WORKSHOP, source={"connector_id": "manual", "connector_type": "manual", "external_id": uid[-8:]},
                     user_id=uid, scope="global", visibility="private", index_passages=False)


def test_stored_label_follows_the_modality_check(svc, unique_ns, monkeypatch):
    from ingest.pipeline import ingest_document

    uid = unique_ns.user_id
    monkeypatch.setattr(svc, "extract_facts_only", lambda *a, **k: [("procedure", UPGRADED), ("procedure", FAITHFUL)])
    ingest_document(svc, _doc(uid))
    assert dict(_facts(svc, uid)) == {UPGRADED: "inductive", FAITHFUL: "explicit"}


def test_real_llm_smoke(svc, unique_ns):
    """SMOKE ONLY (nondeterministic): the real extraction path stores facts and none is flagged-and-explicit."""
    from ingest.modality import ModalitySource, modality_upgraded
    from ingest.pipeline import ingest_document

    uid = unique_ns.user_id
    result = ingest_document(svc, _doc(uid))
    assert result["facts"] >= 1, result
    source = ModalitySource(WORKSHOP)
    offenders = [t for t, level in _facts(svc, uid) if modality_upgraded(t, source) and level == "explicit"]
    assert not offenders, offenders
