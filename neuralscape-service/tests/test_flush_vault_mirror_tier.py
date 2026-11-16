"""The vault mirror of conversation-extracted rows decides by the category's tier.

#517 stamps every extracted row ``visibility=private`` (the row stays private in
recall). The vault mirror then read that stamp and skipped every extracted
fact: /flush -> daily log -> compile went dead. Extracted rows are mirrored by
their CATEGORY's default tier; a row the sensitivity gate forced private is
never mirrored; explicit writes behave exactly as before.
"""

import json
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


async def _flush(writer, rows):
    svc = MagicMock()
    svc.extract_and_store.return_value = rows
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
        # a turn that dedups onto an explicit raw private write returns THAT row (no epistemic level)
        explicit = _row("decision", "Chose Qdrant", epistemic_level=None)
        result = await _flush(writer, [explicit])
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
