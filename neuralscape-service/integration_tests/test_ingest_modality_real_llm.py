"""Issue #444 end to end: the workshop text through the real ingest path and the real LLM.

The model is nondeterministic, so the assertion is on the post-check's outcome, not on its
wording: whatever facts come back, no stored fact may BOTH carry an obligation word the
source sentence lacks (should -> must) AND be labelled ``explicit``. If the post-check in
``ingest.pipeline`` is removed, every upgraded fact is labelled explicit and this fails
(whenever the model does upgrade; the unit tests pin the check deterministically).

Needs Neo4j (service init), Qdrant, GOOGLE_API_KEY.
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


def test_no_stored_fact_both_upgrades_should_to_must_and_claims_explicit(flags, stores, llm_key, unique_ns):
    from qdrant_client import QdrantClient

    from ingest.pipeline import IngestDoc, ingest_document, modality_upgraded
    from memory_service import MemoryService

    flags(qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url, extraction_require_speaker=False)
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    service = MemoryService()
    service._get_memory()
    uid = unique_ns.user_id
    result = ingest_document(
        service,
        IngestDoc(content=WORKSHOP, source={"connector_id": "manual", "connector_type": "manual", "external_id": uid[-8:]},
                  user_id=uid, scope="global", visibility="private", index_passages=False),
    )
    assert result["facts"] >= 1, result

    rows = [r["payload"] for r in service._scroll_all_user_memories(uid)]
    def meta(row, key):
        return (row.get("metadata") or {}).get(key) or row.get(key)

    facts = [r for r in rows if meta(r, "memory_kind") == "fact"]
    assert facts, rows
    offenders = [
        r.get("data") for r in facts
        if modality_upgraded(r.get("data") or "", WORKSHOP) and meta(r, "epistemic_level") == "explicit"
    ]
    assert not offenders, f"upgraded facts labelled explicit: {offenders}"
