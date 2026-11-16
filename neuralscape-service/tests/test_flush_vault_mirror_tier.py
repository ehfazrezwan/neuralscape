"""The vault mirror of conversation-extracted rows decides by the category's tier.

#517 stamps every extracted row ``visibility=private`` (the row stays private in
recall). The vault mirror then read that stamp and skipped every extracted
fact: /flush -> daily log -> compile went dead. Extracted rows are mirrored by
their CATEGORY's default tier; a row the sensitivity gate forced private is
never mirrored; explicit writes behave exactly as before.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fcntl")

from extensions.conversation_compiler import ConversationCompilerExtension
from extensions.conversation_compiler.flush import flush_conversation_turn, mirror_to_vault
from extensions.conversation_compiler.obsidian_writer import ObsidianWriter
from memory_service import MemoryService
from schemas import MemoryResponse


def _row(category, text, **kw):
    """A row the way ``_batch_store_facts`` returns it: private stamp, explicit epistemic level."""
    base = dict(id=f"m-{abs(hash(text)) % 10_000}", memory=text, category=category,
                visibility="private", owner_user_id="alice", epistemic_level="explicit")
    base.update(kw)
    return MemoryResponse(**base)


@pytest.fixture
def writer(tmp_path):
    return ObsidianWriter(vault_path=tmp_path / "vault")


async def _flush(writer, rows, created=None):
    """Flush a turn whose extraction returned ``rows``; ``created`` = rows this extraction created
    (default: all of them; the rest are dedup survivors)."""
    created = rows if created is None else created
    svc = MagicMock()
    svc.extract_and_store.return_value = (rows, {"created_ids": [r.id for r in created]})
    return await flush_conversation_turn(
        user_message="u", assistant_response="a", session_id="s", channel="api",
        timestamp="2026-05-15T12:00:00", project_id=None, user_id="alice", service=svc, writer=writer,
    )


@pytest.mark.asyncio
async def test_shared_category_extracted_fact_is_mirrored_although_stamped_private(writer, tmp_path):
    result = await _flush(writer, [_row("decision", "Chose Qdrant")])
    assert len(result.category_paths) == 1
    daily = (tmp_path / "vault" / "_raw" / "Daily" / "2026-05-15.md").read_text()
    assert "Chose Qdrant" in daily


@pytest.mark.asyncio
async def test_private_category_extracted_fact_is_not_mirrored(writer, tmp_path):
    result = await _flush(writer, [_row("preference", "Likes tea")])
    assert result.category_paths == []
    assert not (tmp_path / "vault" / "_raw" / "Daily" / "2026-05-15.md").exists()


@pytest.mark.asyncio
async def test_sensitivity_forced_private_extracted_fact_is_never_mirrored(writer, tmp_path):
    forced = _row("decision", "Salary is 200k", sensitivity="financial", sensitivity_source="regex")
    result = await _flush(writer, [forced])
    assert result.category_paths == []
    assert not (tmp_path / "vault" / "_raw" / "Daily" / "2026-05-15.md").exists()


@pytest.mark.asyncio
async def test_mixed_turn_mirrors_only_the_shared_unforced_fact(writer, tmp_path):
    rows = [
        _row("decision", "Chose Qdrant"),
        _row("preference", "Likes tea"),
        _row("convention", "Secret rate is 5", sensitivity="client_commercial", sensitivity_source="llm"),
    ]
    await _flush(writer, rows)
    daily = (tmp_path / "vault" / "_raw" / "Daily" / "2026-05-15.md").read_text()
    assert "Chose Qdrant" in daily and "Likes tea" not in daily and "Secret rate" not in daily


class TestExplicitWritesUnchanged:
    def test_private_raw_write_is_not_mirrored_even_in_a_shared_category(self):
        assert mirror_to_vault("decision", "private", extracted=False) is False

    def test_shared_raw_write_is_mirrored_even_in_a_private_category(self):
        assert mirror_to_vault("preference", "shared", extracted=False) is True

    def test_unstamped_row_falls_back_to_the_category_tier(self):
        assert mirror_to_vault("decision", None, extracted=False) is True
        assert mirror_to_vault("preference", None, extracted=False) is False

    @pytest.mark.asyncio
    async def test_flush_dedup_onto_an_explicit_private_row_is_not_mirrored(self, writer):
        # the turn re-derived text an explicit private write already holds: the survivor is
        # returned but was not created by this extraction, so it keeps its own (private) visibility
        survivor = _row("decision", "Chose Qdrant")
        result = await _flush(writer, [survivor], created=[])
        assert result.category_paths == []

    @pytest.mark.asyncio
    async def test_flush_dedup_onto_an_explicit_shared_row_is_mirrored(self, writer):
        survivor = _row("preference", "Likes tea", visibility="shared")
        result = await _flush(writer, [survivor], created=[])
        assert len(result.category_paths) == 1

    @pytest.mark.asyncio
    async def test_epistemic_level_is_not_an_extraction_marker(self, writer):
        # a caller-supplied (or ingest-stamped) epistemic level on a private row is no licence to mirror it
        row = _row("decision", "Chose Qdrant", epistemic_level="explicit")
        result = await _flush(writer, [row], created=[])
        assert result.category_paths == []


class TestEventPath:
    @pytest.fixture
    def extension(self, writer):
        ext = ConversationCompilerExtension()
        ext._writer = writer
        ext._service = MagicMock()
        return ext

    def _payload(self, category, **kw):
        base = {"content": "Chose Qdrant", "category": category, "visibility": "private",
                "created_at": "2026-05-15T12:00:00", "user_id": "alice", "extracted": True}
        base.update(kw)
        return base

    @pytest.mark.asyncio
    async def test_extracted_shared_category_is_mirrored(self, extension):
        assert await extension._handle_memory_stored(self._payload("decision")) is not None

    @pytest.mark.asyncio
    async def test_extracted_private_category_is_not(self, extension):
        assert await extension._handle_memory_stored(self._payload("preference")) is None

    @pytest.mark.asyncio
    async def test_extracted_sensitivity_forced_is_not(self, extension):
        assert await extension._handle_memory_stored(self._payload("decision", sensitivity="financial")) is None

    @pytest.mark.asyncio
    async def test_explicit_private_write_is_not(self, extension):
        assert await extension._handle_memory_stored(self._payload("decision", extracted=False)) is None

    @pytest.mark.asyncio
    async def test_explicit_shared_write_is(self, extension):
        assert await extension._handle_memory_stored(
            self._payload("decision", extracted=False, visibility="shared")
        ) is not None


@pytest.mark.asyncio
async def test_extracted_row_stays_private_and_invisible_to_another_user():
    """The mirror changes nothing about recall: the stored row is private to its owner."""
    svc = MemoryService()
    svc._memory = MagicMock()
    svc._memory.embedding_model.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 8 for _ in texts]
    svc._find_by_content_hash = MagicMock(return_value=None)
    svc._get_memory = MagicMock(return_value=svc._memory)
    stored = svc._batch_store_facts(facts=[("decision", "Chose Qdrant")], user_id="alice")
    assert stored[0].visibility == "private" and stored[0].owner_user_id == "alice"
    payload = svc._memory.vector_store.insert.call_args.kwargs["payloads"][0]
    assert payload["metadata"]["visibility"] == "private"
    assert payload["metadata"]["owner_user_id"] == "alice"
    assert payload["user_id"] == "alice"  # another user's search filters on their own user_id


# ── the worker path over rows from the real _batch_store_facts ─────────


class _Registry:
    def __init__(self, writer):
        self.events = []
        self._extensions = [SimpleNamespace(_writer=writer)]

    async def emit_event(self, name, payload):
        self.events.append((name, payload))


def _real_service(survivors=None):
    """A real MemoryService (real _batch_store_facts) over a fake store.

    ``survivors`` maps content -> pre-existing row returned by the content-hash lookup.
    """
    survivors = survivors or {}
    svc = MemoryService()
    svc._memory = MagicMock()
    svc._memory.embedding_model.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 8 for _ in texts]
    svc._graphiti = None
    svc._bridge = None
    svc._get_memory = MagicMock(return_value=svc._memory)

    def find(user_id, content_hash, scope, **kw):
        from memory.hashing import content_hash as ch

        for text, row in survivors.items():
            if ch(text) == content_hash:
                return row
        return None

    svc._find_by_content_hash = find
    svc._bump_times_derived = MagicMock()
    svc._revive_if_tombstoned = MagicMock(return_value=False)
    return svc


def _llm(svc, lines):
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text=json.dumps({"facts": lines}))
    svc._genai_model = client


async def _run_worker(svc, writer, monkeypatch):
    import worker
    from config import settings

    monkeypatch.setattr(settings, "sensitivity_gate_enabled", True)
    monkeypatch.setattr(settings, "extraction_require_speaker", False)
    registry = _Registry(writer)
    out = await worker.process_memory_store(
        {"service": svc, "extension_registry": registry},
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        "alice",
    )
    return out, registry


def _vault_text(tmp_path):
    daily = list((tmp_path / "vault" / "_raw" / "Daily").glob("*.md")) if (tmp_path / "vault" / "_raw" / "Daily").exists() else []
    return "\n".join(f.read_text() for f in daily)


@pytest.mark.asyncio
async def test_worker_mirrors_only_clean_shared_category_created_rows(writer, tmp_path, monkeypatch):
    svc = _real_service()
    _llm(svc, [
        "[decision] We chose Qdrant for vector storage",
        "[preference] Alice likes strong tea",
        "[decision] The contractor rate is $150,000 per year, salary band confidential",
    ])
    ext = ConversationCompilerExtension()
    ext._writer, ext._service = writer, MagicMock()
    _, registry = await _run_worker(svc, writer, monkeypatch)

    payloads = {p["content"]: p for _, p in registry.events}
    assert payloads["We chose Qdrant for vector storage"]["extracted"] is True
    assert payloads["We chose Qdrant for vector storage"]["visibility"] == "private"  # stays private
    sensitive = next(p for c, p in payloads.items() if "salary" in c)
    assert sensitive["sensitivity"] and sensitive["extracted"] is True

    for _, payload in registry.events:
        await ext._handle_memory_stored(payload)
    vault = _vault_text(tmp_path)
    assert "We chose Qdrant" in vault
    assert "strong tea" not in vault  # private category
    assert "salary band" not in vault  # sensitivity gate forced it private


@pytest.mark.asyncio
async def test_worker_dedup_onto_an_explicit_private_row_is_not_mirrored(writer, tmp_path, monkeypatch):
    text = "We chose Qdrant for vector storage"
    survivor = MemoryResponse(id="exp-1", memory=text, category="decision", visibility="private",
                              owner_user_id="alice")
    svc = _real_service({text: survivor})
    _llm(svc, [f"[decision] {text}"])
    ext = ConversationCompilerExtension()
    ext._writer, ext._service = writer, MagicMock()
    _, registry = await _run_worker(svc, writer, monkeypatch)

    (payload,) = [p for _, p in registry.events]
    assert payload["memory_id"] == "exp-1" and payload["extracted"] is False
    await ext._handle_memory_stored(payload)
    assert text not in _vault_text(tmp_path)
