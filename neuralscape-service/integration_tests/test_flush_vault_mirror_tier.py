"""Real-store proof of the vault mirror's tier rule for conversation-extracted rows.

Real Qdrant + the real extraction LLM + real embeddings, a throwaway collection
and a temporary vault. Uses the integration fixtures (``stores``, ``llm_key``,
``flags``, ``unique_ns``).

1. A turn that yields a decision fact reaches the daily log while the stored row
   reads back ``visibility=private`` owned by its author (recall is unchanged).
2. ``store_raw(X, decision, private)`` followed by a turn that extracts the same
   text X must NOT put X in the vault: the dedup survivor is the explicit
   private row, not a row this extraction created.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import uuid

import pytest


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns):
    from qdrant_client import QdrantClient

    flags(qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url,
          extraction_require_speaker=False)
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    return service


@pytest.fixture
def vault(tmp_path):
    from extensions.conversation_compiler.obsidian_writer import ObsidianWriter

    return ObsidianWriter(vault_path=tmp_path / "vault")


def _vault_text(vault) -> str:
    return "\n".join(vault.read_file(f) for f in vault.list_all_files())


def _turn(svc, vault, uid, project, user_text, assistant_text):
    from extensions.conversation_compiler.flush import flush_conversation_turn

    return asyncio.run(flush_conversation_turn(
        user_message=user_text, assistant_response=assistant_text, session_id=f"it-{uuid.uuid4().hex[:6]}",
        channel="api", timestamp=None, project_id=project, user_id=uid, service=svc, writer=vault,
    ))


def test_extracted_decision_reaches_the_daily_log_while_the_row_stays_private(svc, vault, unique_ns):
    from schemas import MemoryVisibility, default_visibility_for_category

    uid, project = unique_ns.user_id, unique_ns.project_id
    tag = uuid.uuid4().hex[:8]
    name = f"zq{tag}"
    result = _turn(
        svc, vault, uid, project,
        f"We decided the new queue service will be called {name}, because the name is easy to grep.",
        f"Recorded: the decision is to name the new queue service {name} so it is easy to grep for.",
    )
    mine = [f for f in result.facts if name in f.content]
    assert mine, f"the LLM extracted nothing mentioning {name}: {[f.content for f in result.facts]}"

    rows = [r for r in svc._scroll_all_user_memories(uid) if name in r["payload"].get("data", "")]
    assert rows
    for row in rows:
        meta = row["payload"].get("metadata") or {}
        assert meta.get("visibility") == "private" and meta.get("owner_user_id") == uid  # recall unchanged

    shared_tier = [f for f in mine if default_visibility_for_category(f.category) == MemoryVisibility.SHARED]
    if shared_tier:  # the mirror decides by the category the model chose
        assert name in _vault_text(vault)
        assert vault.list_daily_logs()
    else:
        assert name not in _vault_text(vault)


def test_a_turn_that_dedups_onto_an_explicit_private_row_does_not_mirror_it(svc, vault, unique_ns):
    uid, project = unique_ns.user_id, unique_ns.project_id
    tag = uuid.uuid4().hex[:8]
    name = f"zq{tag}"
    fact = f"We chose {name} as the queue service name because it is easy to grep"
    svc.store_raw(content=fact, user_id=uid, category="decision", scope="project", project_id=project,
                  visibility="private", add_to_graph=False)

    _turn(
        svc, vault, uid, project,
        f"Remember: {fact}.", f"Noted: {fact}.",
    )

    assert name not in _vault_text(vault), "a private explicit write was published to the shared vault"
