"""Tests for the wiki_synthesizer extension.

Focused unit tests with all I/O mocked. The container-level end-to-end
verification (real Neo4j, real Gemini) lives in
``tests/test_async_pipeline.py`` territory and is intentionally not
duplicated here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fcntl")  # writer primitives are POSIX-only

from extensions.wiki_synthesizer.community_loader import Community
from extensions.wiki_synthesizer.config import SynthesizerSettings
from extensions.wiki_synthesizer.graph_patcher import (
    attach_memory_id,
    patch_wiki_path,
)
from extensions.wiki_synthesizer.prompts import (
    INCREMENTAL_MERGE_PROMPT,
    render_memories_block,
)
from extensions.wiki_synthesizer.synthesizer import synthesize_all
from extensions.wiki_synthesizer.wiki_renderer import (
    community_filename,
    render_page,
    split_existing_page,
    wiki_page_path,
    wikilink_path,
)


# ──────────────────────────────────────────────
# wiki_renderer
# ──────────────────────────────────────────────


class TestWikiRenderer:
    def test_community_filename_includes_slug_and_short_id(self):
        name = community_filename("01928fab-1234-5678-9abc-def012345678", "Async-First Python")
        assert name.startswith("community-01928fab-")
        assert "async-first-python" in name
        assert name.endswith(".md")

    def test_community_filename_handles_missing_id(self):
        assert community_filename("", "topic").startswith("community-noid-")

    def test_wiki_page_path_uses_category_folder(self, tmp_path):
        path = wiki_page_path(tmp_path / "Wiki", "preference", "community-x.md")
        assert path == tmp_path / "Wiki" / "Semantic" / "Preferences" / "community-x.md"

    def test_wikilink_path_includes_wiki_prefix(self):
        assert wikilink_path("decision", "community-x.md") == "Wiki/Episodic/Decisions/community-x.md"

    def test_split_existing_page_with_frontmatter(self):
        content = "---\ntitle: T\ncommunity_id: 12\n---\n\n# T\n\nBody text"
        fm, body = split_existing_page(content)
        assert fm["title"] == "T"
        assert fm["community_id"] == "12"
        assert "Body text" in body

    def test_split_existing_page_empty(self):
        assert split_existing_page("") == ({}, "")

    def test_split_existing_page_no_frontmatter(self):
        fm, body = split_existing_page("just a body")
        assert fm == {}
        assert body == "just a body"

    def test_render_page_includes_all_metadata(self):
        page = render_page(
            title="Async-First Python",
            category="preference",
            community_id="01928fab",
            community_name="Async Style",
            group_id="shared",
            visibility="shared",
            body="Some body content.",
            source_memory_ids=["mem-1", "mem-2"],
            graph_node_uuids=["u-1", "u-2"],
            synthesis_count=3,
            source_count=5,
            now=datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc),
        )
        assert page.startswith("---\n")
        assert "title: Async-First Python" in page
        assert "synthesis_count: 3" in page
        assert "source_count: 5" in page
        assert "source_memory_ids: [mem-1, mem-2]" in page
        assert "graph_node_uuids: [u-1, u-2]" in page
        assert "# Async-First Python" in page
        assert "Some body content." in page

    def test_render_page_handles_empty_id_lists(self):
        page = render_page(
            title="T",
            category="decision",
            community_id="c1",
            community_name="C",
            group_id="shared",
            visibility="shared",
            body="b",
            source_memory_ids=[],
            graph_node_uuids=[],
            synthesis_count=1,
            source_count=0,
        )
        assert "source_memory_ids: []" in page
        assert "graph_node_uuids: []" in page


# ──────────────────────────────────────────────
# prompts
# ──────────────────────────────────────────────


class TestPrompts:
    def test_incremental_prompt_has_required_placeholders(self):
        # Every placeholder must be present and orthogonal so .format()
        # works without KeyError surprises.
        for token in ("{topic_title}", "{category}", "{existing_body}", "{memories_block}"):
            assert token in INCREMENTAL_MERGE_PROMPT

    def test_render_memories_block_includes_metadata(self):
        block = render_memories_block([
            {
                "content": "Prefer async/await everywhere.",
                "observation_type": "pattern",
                "domain": "coding",
                "confidence": 0.9,
                "created_at": "2026-05-15T10:00:00",
            },
        ])
        assert "Prefer async/await everywhere." in block
        assert "observation_type=pattern" in block
        assert "confidence=0.9" in block

    def test_render_memories_block_skips_empty(self):
        block = render_memories_block([{"content": ""}, {"content": "x"}])
        # Only the non-empty entry survives, numbered from 1.
        assert block.startswith("1. x")
        assert "\n2." not in block


# ──────────────────────────────────────────────
# graph_patcher
# ──────────────────────────────────────────────


def _fake_driver(record_value):
    """Build a fake Neo4j async driver returning a single record."""
    record = {"patched": record_value} if record_value is not None else None

    class _Result:
        async def single(self):
            return record

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def run(self, *args, **kwargs):
            return _Result()

    class _Driver:
        def session(self):
            return _Session()

    return _Driver()


class TestGraphPatcher:
    @pytest.mark.asyncio
    async def test_attach_memory_id_returns_patched_count(self):
        driver = _fake_driver(record_value=3)
        n = await attach_memory_id(
            driver,
            group_id="shared",
            memory_id="mem-1",
            visibility="shared",
            owner_user_id="ehfaz",
            write_started_at=datetime.now(timezone.utc),
        )
        assert n == 3

    @pytest.mark.asyncio
    async def test_attach_memory_id_skips_empty_inputs(self):
        driver = _fake_driver(record_value=0)
        assert await attach_memory_id(
            driver,
            group_id="",
            memory_id="",
            visibility=None,
            owner_user_id=None,
            write_started_at=datetime.now(timezone.utc),
        ) == 0

    @pytest.mark.asyncio
    async def test_attach_memory_id_swallows_driver_errors(self):
        # Driver that raises on session() — patcher must not propagate.
        broken = MagicMock()
        broken.session.side_effect = RuntimeError("connection refused")
        assert await attach_memory_id(
            broken,
            group_id="shared",
            memory_id="mem-1",
            visibility="shared",
            owner_user_id="ehfaz",
            write_started_at=datetime.now(timezone.utc),
        ) == 0

    @pytest.mark.asyncio
    async def test_patch_wiki_path_returns_patched_count(self):
        driver = _fake_driver(record_value=2)
        n = await patch_wiki_path(
            driver,
            node_uuids=["u1", "u2"],
            wiki_path="Wiki/Semantic/Preferences/community-x.md",
        )
        assert n == 2

    @pytest.mark.asyncio
    async def test_patch_wiki_path_empty_uuids_short_circuits(self):
        driver = _fake_driver(record_value=99)  # would lie about count if called
        assert await patch_wiki_path(
            driver,
            node_uuids=[],
            wiki_path="Wiki/x.md",
        ) == 0


# ──────────────────────────────────────────────
# synthesize_all — top-level orchestration smoke
# ──────────────────────────────────────────────


class TestSynthesizeAll:
    @pytest.mark.asyncio
    async def test_disabled_returns_empty_result(self):
        settings = SynthesizerSettings(enabled=False)
        result = await synthesize_all(service=MagicMock(), settings=settings)
        assert result.pages_created == 0
        assert result.pages_updated == 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_missing_driver_records_error(self):
        settings = SynthesizerSettings(enabled=True)
        service = MagicMock()
        service._graphiti = None
        result = await synthesize_all(service=service, settings=settings)
        assert result.pages_created == 0
        assert any("Neo4j driver" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_skips_empty_communities(self, tmp_path):
        # Driver returns one community with no member memory IDs.
        # The synthesizer should count it as skipped and produce no pages.
        empty_community = Community(uuid="c1", name="Empty topic", member_memory_ids=[])

        settings = SynthesizerSettings(
            enabled=True,
            obsidian_vault_path=str(tmp_path),
        )
        service = MagicMock()
        service._graphiti = MagicMock()
        service._graphiti.driver = MagicMock()

        with patch(
            "extensions.wiki_synthesizer.synthesizer.load_communities",
            new=AsyncMock(return_value=[empty_community]),
        ):
            result = await synthesize_all(
                service=service,
                settings=settings,
                only_category="preference",
            )

        assert result.pages_created == 0
        assert result.pages_updated == 0
        assert result.communities_skipped_empty == 1
