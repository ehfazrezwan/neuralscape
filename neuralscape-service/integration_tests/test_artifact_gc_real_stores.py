"""Flag-on proof: the artifact sweep against real Qdrant and a real volume path.

``ARTIFACT_GC_ENABLED`` on, ``ARTIFACT_GC_DRY_RUN`` off, grace 1 day. Two
artifacts older than the grace period sit on the volume under a unique storage
directory; one is referenced by real memory rows (ingested through the real
pipeline with the artifact's ``source_ref``), one is not. The cron must delete
only the unreferenced one; after the referencing rows are deleted the next run
removes the other. A dry run deletes nothing.

Uses ``stores``, ``llm_key``, ``flags``, ``unique_ns`` (throwaway Qdrant
collection); fact extraction is off, so no extraction LLM spend.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import os
import time

import pytest

DAY = 86400


@pytest.fixture
def env(flags, stores, llm_key, unique_ns, tmp_path):
    from qdrant_client import QdrantClient

    storage = str(tmp_path / "vol")
    flags(
        qdrant_collection=unique_ns.qdrant_collection,
        qdrant_url=stores.qdrant_url,
        ingest_storage_dir=storage,
        artifact_gc_enabled=True,
        artifact_gc_dry_run=False,
        artifact_gc_grace_days=1,
    )
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    return service, unique_ns


def _artifact(uid, data):
    from config import settings
    from ingest.storage import store_artifact

    art = store_artifact(data, "doc.txt", uid, None, "domain_knowledge", settings)
    old = time.time() - 30 * DAY
    os.utime(art.abs_path, (old, old))
    return art


def _ingest_referencing(service, uid, art, text):
    from ingest.pipeline import IngestDoc, ingest_document
    from ingest.storage import artifact_source_ref

    return ingest_document(service, IngestDoc(
        content=text, source=artifact_source_ref(art), user_id=uid, category="domain_knowledge",
        scope="global", extract_facts=False, index_passages=True,
    ))


def _gc(service):
    import worker

    return asyncio.run(worker.artifact_gc_cron({"service": service}))


def test_only_the_unreferenced_artifact_is_removed_and_the_other_follows_its_rows(env):
    service, ns = env
    uid = ns.user_id
    kept = _artifact(uid, f"kept artifact {ns.group_prefix} with some content".encode())
    orphan = _artifact(uid, f"orphan artifact {ns.group_prefix} with other content".encode())
    result = _ingest_referencing(service, uid, kept, f"A passage ingested from the kept artifact {ns.group_prefix}.")
    assert result["memory_ids"], "the referencing rows were stored"

    out = _gc(service)
    assert out["deleted"] == 1 and out["referenced"] == 1, out
    assert os.path.exists(kept.abs_path) and not os.path.exists(orphan.abs_path)

    # delete every row that referenced the artifact; the next sweep removes it
    service.delete_memories(user_id=uid, include_shared=True)
    out = _gc(service)
    assert out["deleted"] == 1, out
    assert not os.path.exists(kept.abs_path)


def test_dry_run_deletes_nothing_even_with_the_flag_on(env, flags):
    service, ns = env
    flags(artifact_gc_dry_run=True)
    orphan = _artifact(ns.user_id, f"orphan {ns.group_prefix}".encode())
    out = _gc(service)
    assert out["orphans"] == 1 and out["deleted"] == 0 and out["dry_run"] is True
    assert os.path.exists(orphan.abs_path)
