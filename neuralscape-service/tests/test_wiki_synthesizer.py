"""Tests for the wiki_synthesizer extension (category-based).

Focused unit tests with all I/O mocked. The container-level end-to-end
verification (real Neo4j, real Gemini) lives in
``tests/test_async_pipeline.py`` territory and is intentionally not
duplicated here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fcntl")  # writer primitives are POSIX-only

from extensions.wiki_synthesizer.config import SynthesizerSettings
from extensions.wiki_synthesizer.graph_patcher import (
    attach_memory_id,
    patch_wiki_path,
    patch_wiki_path_by_memory_ids,
)
from extensions.wiki_synthesizer.prompts import (
    INCREMENTAL_MERGE_PROMPT,
    render_memories_block,
)
from extensions.wiki_synthesizer.synthesizer import (
    _parse_shared_group_id,
    synthesize_all,
)
from extensions.wiki_synthesizer.wiki_renderer import (
    category_filename,
    category_page_title,
    render_page,
    split_existing_page,
    wiki_page_path,
    wikilink_path,
)


# ──────────────────────────────────────────────
# wiki_renderer
# ──────────────────────────────────────────────


class TestWikiRenderer:
    def test_category_filename_team_wide(self):
        assert category_filename("shared") == "shared.md"

    def test_category_filename_per_project(self):
        assert category_filename("shared--project--neuralscape") == "neuralscape.md"

    def test_category_filename_slugifies_dirty_project_id(self):
        # `_slugify` should normalize whitespace/symbols to a clean filename.
        assert category_filename("shared--project--My Big Project!") == "my-big-project.md"

    def test_category_filename_handles_empty(self):
        assert category_filename("") == "shared.md"

    def test_wiki_page_path_uses_category_folder(self, tmp_path):
        path = wiki_page_path(tmp_path / "Wiki", "preference", "shared.md")
        assert path == tmp_path / "Wiki" / "Semantic" / "Preferences" / "shared.md"

    def test_wikilink_path_includes_wiki_prefix(self):
        assert wikilink_path("decision", "shared.md") == "Wiki/Episodic/Decisions/shared.md"

    def test_category_page_title_team_wide(self):
        assert category_page_title("convention", "shared") == "Conventions"

    def test_category_page_title_per_project(self):
        assert (
            category_page_title("convention", "shared--project--neuralscape")
            == "Conventions — neuralscape"
        )

    def test_split_existing_page_with_frontmatter(self):
        content = "---\ntitle: T\nsynthesis_count: 2\n---\n\n# T\n\nBody text"
        fm, body = split_existing_page(content)
        assert fm["title"] == "T"
        assert fm["synthesis_count"] == "2"
        assert "Body text" in body

    def test_split_existing_page_empty(self):
        assert split_existing_page("") == ({}, "")

    def test_split_existing_page_no_frontmatter(self):
        fm, body = split_existing_page("just a body")
        assert fm == {}
        assert body == "just a body"

    def test_render_page_includes_all_metadata(self):
        page = render_page(
            title="Conventions — neuralscape",
            category="convention",
            group_id="shared--project--neuralscape",
            visibility="shared",
            body="Some body content.",
            source_memory_ids=["mem-1", "mem-2"],
            synthesis_count=3,
            source_count=5,
            now=datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc),
        )
        assert page.startswith("---\n")
        assert "title: Conventions — neuralscape" in page
        assert "category: convention" in page
        assert "group_id: shared--project--neuralscape" in page
        assert "synthesis_count: 3" in page
        assert "source_count: 5" in page
        assert "source_memory_ids: [mem-1, mem-2]" in page
        assert "# Conventions — neuralscape" in page
        assert "Some body content." in page

    def test_render_page_handles_empty_source_list(self):
        page = render_page(
            title="T",
            category="decision",
            group_id="shared",
            visibility="shared",
            body="b",
            source_memory_ids=[],
            synthesis_count=1,
            source_count=0,
        )
        assert "source_memory_ids: []" in page


# ──────────────────────────────────────────────
# prompts
# ──────────────────────────────────────────────


class TestPrompts:
    def test_incremental_prompt_has_required_placeholders(self):
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
        assert block.startswith("1. x")
        assert "\n2." not in block


# ──────────────────────────────────────────────
# parser
# ──────────────────────────────────────────────


class TestParseSharedGroupId:
    def test_team_wide(self):
        assert _parse_shared_group_id("shared") == ("shared", None)

    def test_per_project(self):
        assert _parse_shared_group_id("shared--project--neuralscape") == (
            "shared",
            "neuralscape",
        )

    def test_unknown_returns_none(self):
        assert _parse_shared_group_id("user--ehfaz") == (None, None)
        assert _parse_shared_group_id("global") == (None, None)


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


def _fake_service_with(driver):
    """Build a minimal fake MemoryService wrapping ``driver``.

    The graph patchers take a service and dispatch via
    ``service._run_on_bridge_async``. For unit tests we just await the
    inner coroutine on the calling loop — the bridge indirection only
    matters when the real Graphiti driver is bound to a separate loop.
    """

    class _FakeGraphiti:
        pass

    class _FakeService:
        def __init__(self):
            self._graphiti = _FakeGraphiti()
            self._graphiti.driver = driver

        async def _run_on_bridge_async(self, coro, timeout=30.0):
            return await coro

    return _FakeService()


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
        service = _fake_service_with(driver)
        n = await patch_wiki_path(
            service,
            node_uuids=["u1", "u2"],
            wiki_path="Wiki/Semantic/Preferences/shared.md",
        )
        assert n == 2

    @pytest.mark.asyncio
    async def test_patch_wiki_path_empty_uuids_short_circuits(self):
        driver = _fake_driver(record_value=99)
        service = _fake_service_with(driver)
        assert await patch_wiki_path(
            service,
            node_uuids=[],
            wiki_path="Wiki/x.md",
        ) == 0

    @pytest.mark.asyncio
    async def test_patch_wiki_path_by_memory_ids_returns_patched_count(self):
        driver = _fake_driver(record_value=5)
        service = _fake_service_with(driver)
        n = await patch_wiki_path_by_memory_ids(
            service,
            memory_ids=["mem-1", "mem-2", "mem-3"],
            wiki_path="Wiki/Project/Conventions/neuralscape.md",
            group_id="shared--project--neuralscape",
        )
        assert n == 5

    @pytest.mark.asyncio
    async def test_patch_wiki_path_by_memory_ids_empty_short_circuits(self):
        driver = _fake_driver(record_value=99)
        service = _fake_service_with(driver)
        assert await patch_wiki_path_by_memory_ids(
            service,
            memory_ids=[],
            wiki_path="Wiki/x.md",
        ) == 0


# ──────────────────────────────────────────────
# synthesize_all — top-level orchestration smoke
# ──────────────────────────────────────────────


def _qdrant_point(memory_id: str, content: str, category: str) -> SimpleNamespace:
    """Shape a Qdrant scroll point the way ``client.scroll()`` returns it."""
    return SimpleNamespace(
        id=memory_id,
        payload={
            "data": content,
            "metadata": {
                "category": category,
                "visibility": "shared",
                "domain": "coding",
            },
        },
    )


def _service_with_qdrant_points(
    points_by_filter: dict[tuple[str, str], list[SimpleNamespace]],
):
    """Build a MemoryService mock whose Qdrant client returns the
    supplied points based on ``(group_id, category)``.

    The scroll mock inspects the must-filter list for
    ``metadata.category`` and ``metadata.project_id`` and returns the
    matching pre-staged list — empty if no entry matches.
    """
    from qdrant_client.models import FieldCondition

    def _scroll(collection_name, scroll_filter, limit, with_payload, with_vectors):
        cat: str | None = None
        project: str | None = None
        scope: str | None = None
        for cond in getattr(scroll_filter, "must", []) or []:
            if not isinstance(cond, FieldCondition):
                continue
            key = cond.key
            val = getattr(cond.match, "value", None)
            if key == "metadata.category":
                cat = val
            elif key == "metadata.project_id":
                project = val
            elif key == "metadata.scope":
                scope = val
        gid = f"shared--project--{project}" if project else "shared"
        # If scope is missing the scroll is malformed for our purposes.
        assert scope in ("global", "project")
        return (points_by_filter.get((gid, cat or ""), []), None)

    client = MagicMock()
    client.scroll = _scroll

    service = MagicMock()
    service._memory = MagicMock()
    service._memory.vector_store = MagicMock()
    service._memory.vector_store.client = client
    service._graphiti = MagicMock()
    service._graphiti.driver = MagicMock()
    return service


class TestSynthesizeAll:
    @pytest.mark.asyncio
    async def test_disabled_returns_empty_result(self):
        settings = SynthesizerSettings(enabled=False)
        result = await synthesize_all(service=MagicMock(), settings=settings)
        assert result.pages_created == 0
        assert result.pages_updated == 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_no_shared_group_ids_short_circuits(self, tmp_path):
        settings = SynthesizerSettings(
            enabled=True, obsidian_vault_path=str(tmp_path)
        )
        service = MagicMock()
        with patch(
            "extensions.wiki_synthesizer.synthesizer._list_shared_group_ids",
            new=AsyncMock(return_value=[]),
        ):
            result = await synthesize_all(service=service, settings=settings)
        assert result.pages_created == 0
        assert result.pages_updated == 0
        assert result.pages_skipped_empty == 0

    @pytest.mark.asyncio
    async def test_buckets_with_no_memories_are_skipped(self, tmp_path):
        # All Qdrant scrolls return empty. Every (group × category) pair
        # should be counted as skipped, no pages created.
        settings = SynthesizerSettings(
            enabled=True, obsidian_vault_path=str(tmp_path)
        )
        service = _service_with_qdrant_points({})
        with patch(
            "extensions.wiki_synthesizer.synthesizer._list_shared_group_ids",
            new=AsyncMock(return_value=["shared"]),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer._call_gemini",
            new=AsyncMock(return_value="body"),  # never invoked
        ):
            result = await synthesize_all(service=service, settings=settings)
        # 13 categories × 1 group, all empty
        assert result.pages_skipped_empty == 13
        assert result.pages_created == 0

    @pytest.mark.asyncio
    async def test_creates_one_page_per_nonempty_bucket(self, tmp_path):
        # One non-empty bucket: shared/convention. Synthesizer should
        # produce exactly one created page and skip the other 12.
        settings = SynthesizerSettings(
            enabled=True, obsidian_vault_path=str(tmp_path)
        )
        service = _service_with_qdrant_points({
            ("shared", "convention"): [
                _qdrant_point("mem-1", "use PRs targeting dev", "convention"),
                _qdrant_point("mem-2", "no direct commits to main", "convention"),
            ],
        })
        with patch(
            "extensions.wiki_synthesizer.synthesizer._list_shared_group_ids",
            new=AsyncMock(return_value=["shared"]),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer._call_gemini",
            new=AsyncMock(return_value="wiki body about conventions"),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer."
            "patch_wiki_path_by_memory_ids",
            new=AsyncMock(return_value=2),
        ):
            result = await synthesize_all(service=service, settings=settings)
        assert result.pages_created == 1
        assert result.pages_updated == 0
        assert result.memories_processed == 2
        assert result.pages_skipped_empty == 12
        page = result.pages[0]
        assert page.category == "convention"
        assert page.group_id == "shared"
        assert page.wiki_path == "Wiki/Project/Conventions/shared.md"
        # The actual file was written (not dry_run).
        assert (tmp_path / "Wiki" / "Project" / "Conventions" / "shared.md").exists()

    @pytest.mark.asyncio
    async def test_only_category_restricts_scope(self, tmp_path):
        # Stage memories in two categories. With only_category="convention"
        # we should produce exactly one page, even though both buckets
        # have content.
        settings = SynthesizerSettings(
            enabled=True, obsidian_vault_path=str(tmp_path)
        )
        service = _service_with_qdrant_points({
            ("shared", "convention"): [_qdrant_point("m1", "x", "convention")],
            ("shared", "decision"): [_qdrant_point("m2", "y", "decision")],
        })
        with patch(
            "extensions.wiki_synthesizer.synthesizer._list_shared_group_ids",
            new=AsyncMock(return_value=["shared"]),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer._call_gemini",
            new=AsyncMock(return_value="body"),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer."
            "patch_wiki_path_by_memory_ids",
            new=AsyncMock(return_value=1),
        ):
            result = await synthesize_all(
                service=service,
                settings=settings,
                only_category="convention",
            )
        assert result.pages_created == 1
        assert result.pages[0].category == "convention"
        # only_category="convention" → never even scrolls decision bucket.
        assert result.pages_skipped_empty == 0

    @pytest.mark.asyncio
    async def test_dry_run_does_not_write_or_patch(self, tmp_path):
        settings = SynthesizerSettings(
            enabled=True, obsidian_vault_path=str(tmp_path)
        )
        service = _service_with_qdrant_points({
            ("shared", "convention"): [_qdrant_point("m1", "x", "convention")],
        })
        patch_mock = AsyncMock(return_value=0)
        with patch(
            "extensions.wiki_synthesizer.synthesizer._list_shared_group_ids",
            new=AsyncMock(return_value=["shared"]),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer._call_gemini",
            new=AsyncMock(return_value="body"),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer."
            "patch_wiki_path_by_memory_ids",
            new=patch_mock,
        ):
            result = await synthesize_all(
                service=service, settings=settings, dry_run=True
            )
        assert result.pages_created == 1
        # dry_run → no file written, no graph patch
        assert not (tmp_path / "Wiki" / "Project" / "Conventions" / "shared.md").exists()
        patch_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_walks_all_shared_group_ids(self, tmp_path):
        # Verify both team-wide and per-project shared groups are visited.
        settings = SynthesizerSettings(
            enabled=True, obsidian_vault_path=str(tmp_path)
        )
        service = _service_with_qdrant_points({
            ("shared", "convention"): [_qdrant_point("m1", "x", "convention")],
            ("shared--project--neuralscape", "convention"): [
                _qdrant_point("m2", "y", "convention"),
            ],
        })
        with patch(
            "extensions.wiki_synthesizer.synthesizer._list_shared_group_ids",
            new=AsyncMock(return_value=[
                "shared",
                "shared--project--neuralscape",
            ]),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer._call_gemini",
            new=AsyncMock(return_value="body"),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer."
            "patch_wiki_path_by_memory_ids",
            new=AsyncMock(return_value=1),
        ):
            result = await synthesize_all(
                service=service,
                settings=settings,
                only_category="convention",
            )
        assert result.pages_created == 2
        paths = {p.wiki_path for p in result.pages}
        assert "Wiki/Project/Conventions/shared.md" in paths
        assert "Wiki/Project/Conventions/neuralscape.md" in paths
