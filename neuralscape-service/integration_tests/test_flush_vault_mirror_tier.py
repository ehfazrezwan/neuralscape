"""Real-store proof of the vault mirror's tier rule for conversation-extracted rows.

Real Qdrant, the real embedder and the real ``store_raw`` / ``extract_and_store``
write path over a throwaway collection and a temporary vault (``stores``,
``llm_key``, ``flags``, ``unique_ns``). Only the extraction LLM's output is
stubbed, so categories and texts are chosen by the test and every assertion is
unconditional. Both paths that feed the vault are driven: ``/flush``
(``flush_conversation_turn``) and the worker/event path
(``process_memory_store`` -> ``memory_stored`` handler).

1. A created decision fact reaches the daily log while its row reads back private
   and owned by its author; a private-category fact does not.
2. ``store_raw(X, decision, private)`` followed by a turn extracting the same text
   dedups onto the explicit row (asserted: same id, still one row) and X is NOT in
   the vault, on both paths.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

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
    service._graphiti = None  # no graph episode: this test is about vectors and the vault
    service._bridge = None
    return service


@pytest.fixture
def vault(tmp_path):
    from extensions.conversation_compiler.obsidian_writer import ObsidianWriter

    return ObsidianWriter(vault_path=tmp_path / "vault")


def _stub_extraction(svc, lines):
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text=json.dumps({"facts": lines}))
    svc._genai_model = client


def _vault_text(vault) -> str:
    return "\n".join(vault.read_file(f) for f in vault.list_all_files())


def _rows_with(svc, uid, text):
    return [r for r in svc._scroll_all_user_memories(uid) if text in r["payload"].get("data", "")]


def _flush(svc, vault, uid, project):
    from extensions.conversation_compiler.flush import flush_conversation_turn

    return asyncio.run(flush_conversation_turn(
        user_message="u", assistant_response="a long enough assistant reply", session_id=f"it-{uuid.uuid4().hex[:6]}",
        channel="api", timestamp=None, project_id=project, user_id=uid, service=svc, writer=vault,
    ))


def _worker_path(svc, vault, uid, project):
    """process_memory_store, with the registry delivering events to the real extension handler."""
    import worker
    from extensions.conversation_compiler import ConversationCompilerExtension

    ext = ConversationCompilerExtension()
    ext._writer, ext._service = vault, MagicMock()

    class Registry:
        _extensions = [SimpleNamespace(_writer=vault)]

        async def emit_event(self, name, payload):
            if name == "memory_stored":
                await ext._handle_memory_stored(payload)

    return asyncio.run(worker.process_memory_store(
        {"service": svc, "extension_registry": Registry()},
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a long enough assistant reply"}],
        uid, project,
    ))


@pytest.mark.parametrize("path", ["flush", "worker"])
def test_created_decision_reaches_the_vault_while_the_row_stays_private(svc, vault, unique_ns, path):
    uid, project = unique_ns.user_id, unique_ns.project_id
    name = f"zq{uuid.uuid4().hex[:8]}"
    decision = f"We decided the queue service is called {name} because it is easy to grep"
    private_fact = f"Alice prefers dark roast coffee {name}"
    _stub_extraction(svc, [f"[decision] {decision}", f"[preference] {private_fact}"])

    (_flush if path == "flush" else _worker_path)(svc, vault, uid, project)

    text = _vault_text(vault)
    assert decision in text, "a created shared-category fact must be mirrored"
    assert private_fact not in text, "a private-category fact must not be mirrored"
    for row in _rows_with(svc, uid, name):
        meta = row["payload"].get("metadata") or {}
        assert meta.get("visibility") == "private" and meta.get("owner_user_id") == uid  # recall unchanged


@pytest.mark.parametrize("path", ["flush", "worker"])
def test_a_turn_that_dedups_onto_an_explicit_private_row_does_not_publish_it(svc, vault, unique_ns, path):
    uid, project = unique_ns.user_id, unique_ns.project_id
    name = f"zq{uuid.uuid4().hex[:8]}"
    fact = f"We chose {name} as the queue service name because it is easy to grep"
    explicit = svc.store_raw(content=fact, user_id=uid, category="decision", scope="project",
                             project_id=project, visibility="private", add_to_graph=False)[0]
    _stub_extraction(svc, [f"[decision] {fact}"])

    result = (_flush if path == "flush" else _worker_path)(svc, vault, uid, project)

    # the turn really deduped onto the explicit row: still exactly one row, same id
    rows = _rows_with(svc, uid, name)
    assert len(rows) == 1 and rows[0]["id"] == explicit.id, [r["id"] for r in rows]
    if path == "worker":
        assert [m["id"] for m in result["memories"]] == [explicit.id]
    else:
        assert result.memories_stored == 1
    assert name not in _vault_text(vault), "a private explicit write was published to the shared vault"
