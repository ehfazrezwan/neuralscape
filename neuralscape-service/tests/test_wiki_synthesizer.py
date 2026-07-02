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
    attach_source_ref,
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
    def test_wiki_page_path_global_scope(self, tmp_path):
        path = wiki_page_path(tmp_path / "Wiki", "preference", "shared")
        assert path == tmp_path / "Wiki" / "global" / "Semantic" / "Preferences.md"

    def test_wiki_page_path_per_project_scope(self, tmp_path):
        path = wiki_page_path(
            tmp_path / "Wiki", "decision", "shared--project--neuralscape"
        )
        assert (
            path
            == tmp_path / "Wiki" / "neuralscape" / "Episodic" / "Decisions.md"
        )

    def test_wikilink_path_global_scope(self):
        assert (
            wikilink_path("decision", "shared")
            == "Wiki/global/Episodic/Decisions.md"
        )

    def test_wikilink_path_project_scope_renames_project_to_general(self):
        # 'Project' type-group renames to 'General' inside per-project trees
        # because 'Wiki/<project>/Project/...' would be redundant.
        assert (
            wikilink_path("architecture", "shared--project--neuralscape")
            == "Wiki/neuralscape/General/Architecture.md"
        )

    def test_wikilink_path_project_scope_non_project_typegroup_unchanged(self):
        # Episodic / Procedural / Semantic / Working stay named the same
        # under per-project trees — only the 'Project' type-group renames.
        assert (
            wikilink_path("decision", "shared--project--neuralscape")
            == "Wiki/neuralscape/Episodic/Decisions.md"
        )

    def test_wikilink_path_global_scope_with_project_category_also_renames(self):
        # The 'Project' rename to 'General' is consistent across scopes —
        # not asymmetric. A shared 'architecture' memory (rare but legal)
        # lands at Wiki/global/General/Architecture.md, not Wiki/global/Project/.
        assert (
            wikilink_path("architecture", "shared")
            == "Wiki/global/General/Architecture.md"
        )

    def test_wikilink_path_slugifies_dirty_project_id(self):
        assert (
            wikilink_path("decision", "shared--project--My Big Project!")
            == "Wiki/my-big-project/Episodic/Decisions.md"
        )

    def test_wikilink_path_empty_group_id_treated_as_global(self):
        # Empty / falsy group_id is defensively routed to the global tree
        # rather than crashing — matches the legacy fallback at the call site.
        assert wikilink_path("decision", "") == "Wiki/global/Episodic/Decisions.md"

    def test_wikilink_path_reserved_project_id_global_returns_none(self):
        # A project literally named "global" would collide with the
        # global-scope dir — skip the bucket.
        assert wikilink_path("decision", "shared--project--global") is None

    def test_wikilink_path_reserved_project_id_shared_returns_none(self):
        # A project named "shared" would collide with reserved layout
        # vocabulary — skip.
        assert wikilink_path("decision", "shared--project--shared") is None

    def test_wikilink_path_reserved_project_id_typegroup_returns_none(self):
        # A project slugged to "episodic" would create
        # Wiki/episodic/General/Architecture.md, which on
        # case-insensitive filesystems collides with
        # Wiki/<other-project>/Episodic/Decisions.md — skip.
        assert wikilink_path("decision", "shared--project--Episodic") is None

    def test_wikilink_path_empty_project_id_returns_none(self):
        # 'shared--project--' alone (no pid) must not silently slip
        # through to 'Wiki/<scope>/Episodic/Decisions.md' with an
        # unknown scope. Returns None instead.
        assert wikilink_path("decision", "shared--project--") is None

    def test_wikilink_path_unknown_category_falls_back_to_uncategorized(self):
        # Categories not in CATEGORY_VAULT_PATHS land under an
        # 'Uncategorized' type-group with a slugged leaf so the page is
        # still discoverable rather than dropped.
        assert (
            wikilink_path("nonsense_category", "shared")
            == "Wiki/global/Uncategorized/nonsense-category.md"
        )

    def test_wiki_page_path_returns_none_for_reserved_project_id(self, tmp_path):
        # wiki_page_path must propagate the same skip signal as
        # wikilink_path — they're a pair and the call site checks both.
        assert (
            wiki_page_path(tmp_path / "Wiki", "decision", "shared--project--global")
            is None
        )

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
    async def test_attach_source_ref_retries_deadlocks_then_succeeds(self, monkeypatch):
        """Concurrent graph jobs MERGEing the same (:Source) node deadlock in
        Neo4j (TransientError); the attach must retry, not drop the backlink."""
        from neo4j.exceptions import TransientError

        import extensions.wiki_synthesizer.graph_patcher as gp

        # Stub the backoff — the retry loop is what's under test, not the wait.
        monkeypatch.setattr(gp, "_backoff_sleep", AsyncMock())

        record = {"patched": 4}

        class _Result:
            async def single(self):
                return record

        attempts = {"n": 0}

        class _Session:
            async def __aenter__(self):
                attempts["n"] += 1
                if attempts["n"] <= 2:
                    raise TransientError("deadlock detected")
                return self

            async def __aexit__(self, *a):
                return False

            async def run(self, *args, **kwargs):
                return _Result()

        class _Driver:
            def session(self):
                return _Session()

        n = await attach_source_ref(
            _Driver(),
            group_id="shared",
            memory_id="mem-1",
            source_ref={"connector_id": "file_upload", "external_id": "abc123"},
            write_started_at=datetime.now(timezone.utc),
        )
        assert n == 4
        assert attempts["n"] == 3  # two deadlocks + one success

    @pytest.mark.asyncio
    async def test_attach_source_ref_exhausted_deadlocks_swallowed(self, monkeypatch):
        from neo4j.exceptions import TransientError

        import extensions.wiki_synthesizer.graph_patcher as gp

        monkeypatch.setattr(gp, "_SOURCE_ATTACH_RETRIES", 1)
        monkeypatch.setattr(gp, "_backoff_sleep", AsyncMock())

        class _Session:
            async def __aenter__(self):
                raise TransientError("deadlock detected")

            async def __aexit__(self, *a):
                return False

        class _Driver:
            def session(self):
                return _Session()

        # Best-effort contract holds: exhausted retries log + return 0.
        n = await attach_source_ref(
            _Driver(),
            group_id="shared",
            memory_id="mem-1",
            source_ref={"connector_id": "file_upload", "external_id": "abc123"},
            write_started_at=datetime.now(timezone.utc),
        )
        assert n == 0

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
            wiki_path="Wiki/neuralscape/General/Conventions.md",
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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("with_group_id", [False, True])
    async def test_wiki_synthesized_at_stored_as_string_not_native_datetime(
        self, with_group_id
    ):
        """Regression: ``wiki_synthesized_at`` must be written as a plain ISO
        string, never a native neo4j ``datetime()``.

        Graphiti hydrates unknown node properties into ``node.attributes`` on
        load and ``json.dumps`` them during entity resolution in
        ``add_episode``. A ``neo4j.time.DateTime`` is not JSON serializable, so
        a native-datetime ``wiki_synthesized_at`` broke graph storage on every
        write that resolved against a wiki-synthesized entity.
        """
        captured: list[tuple[str, dict]] = []

        class _CapSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def run(self, query, **params):
                captured.append((query, params))

                class _R:
                    async def single(self_inner):
                        return {"patched": 1}

                return _R()

        class _CapDriver:
            def session(self):
                return _CapSession()

        service = _fake_service_with(_CapDriver())
        when = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
        group_id = "shared--project--neuralscape" if with_group_id else None

        await patch_wiki_path(
            service,
            node_uuids=["u1"],
            wiki_path="Wiki/x.md",
            group_id=group_id,
            synthesized_at=when,
        )
        await patch_wiki_path_by_memory_ids(
            service,
            memory_ids=["mem-1"],
            wiki_path="Wiki/x.md",
            group_id=group_id,
            synthesized_at=when,
        )

        assert captured, "expected the patchers to issue a Cypher write"
        for query, params in captured:
            assert "n.wiki_synthesized_at = $synthesized_at" in query
            assert "datetime($synthesized_at)" not in query
            # Bound value must be a string (isoformat), not a datetime object.
            assert isinstance(params["synthesized_at"], str)
            assert params["synthesized_at"] == when.isoformat()


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
        # New layout: group_id=='shared' → scope='global'; 'Project'
        # type-group renames to 'General' here too (consistent rename).
        assert page.wiki_path == "Wiki/global/General/Conventions.md"
        # The actual file was written (not dry_run).
        assert (
            tmp_path / "Wiki" / "global" / "General" / "Conventions.md"
        ).exists()

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
        assert not (
            tmp_path / "Wiki" / "global" / "General" / "Conventions.md"
        ).exists()
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
        # 'convention' is in the 'Project' type-group → renamed to 'General'.
        # group_id 'shared' → scope 'global'; 'shared--project--neuralscape'
        # → scope 'neuralscape'.
        assert "Wiki/global/General/Conventions.md" in paths
        assert "Wiki/neuralscape/General/Conventions.md" in paths

    @pytest.mark.asyncio
    async def test_synthesis_calls_patch_wiki_path_with_new_format_path(
        self, tmp_path
    ):
        # Regression guard against drift between what gets written to
        # disk and what gets stamped onto Neo4j nodes. The graph patcher
        # must receive the same new-format path string that's reflected
        # in result.pages[].wiki_path.
        settings = SynthesizerSettings(
            enabled=True, obsidian_vault_path=str(tmp_path)
        )
        service = _service_with_qdrant_points({
            ("shared--project--neuralscape", "architecture"): [
                _qdrant_point("m1", "dual-backend design", "architecture"),
            ],
        })
        patch_mock = AsyncMock(return_value=1)
        with patch(
            "extensions.wiki_synthesizer.synthesizer._list_shared_group_ids",
            new=AsyncMock(return_value=["shared--project--neuralscape"]),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer._call_gemini",
            new=AsyncMock(return_value="body"),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer."
            "patch_wiki_path_by_memory_ids",
            new=patch_mock,
        ):
            result = await synthesize_all(
                service=service,
                settings=settings,
                only_category="architecture",
            )
        assert result.pages_created == 1
        # One patch call per successful page write
        patch_mock.assert_awaited_once()
        call_kwargs = patch_mock.await_args.kwargs
        assert (
            call_kwargs["wiki_path"]
            == "Wiki/neuralscape/General/Architecture.md"
        )
        assert call_kwargs["group_id"] == "shared--project--neuralscape"

    @pytest.mark.asyncio
    async def test_existing_page_at_new_path_increments_synthesis_count(
        self, tmp_path
    ):
        # First synthesis writes synthesis_count: 1 at the new path.
        # Second run at the same path reads existing frontmatter, bumps
        # synthesis_count to 2, and reports pages_updated.
        settings = SynthesizerSettings(
            enabled=True, obsidian_vault_path=str(tmp_path)
        )
        service = _service_with_qdrant_points({
            ("shared", "convention"): [
                _qdrant_point("m1", "use PRs targeting dev", "convention"),
            ],
        })
        with patch(
            "extensions.wiki_synthesizer.synthesizer._list_shared_group_ids",
            new=AsyncMock(return_value=["shared"]),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer._call_gemini",
            new=AsyncMock(return_value="first body"),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer."
            "patch_wiki_path_by_memory_ids",
            new=AsyncMock(return_value=1),
        ):
            first = await synthesize_all(
                service=service,
                settings=settings,
                only_category="convention",
            )
        assert first.pages_created == 1
        page_path = tmp_path / "Wiki" / "global" / "General" / "Conventions.md"
        assert page_path.exists()
        assert "synthesis_count: 1" in page_path.read_text()

        # Second run — bucket now has an ADDED memory, so the source set
        # changed and the page is re-synthesized (incremental skip only fires
        # when the source-id set is identical; see the skip test below).
        service2 = _service_with_qdrant_points({
            ("shared", "convention"): [
                _qdrant_point("m1", "use PRs targeting dev", "convention"),
                _qdrant_point("m2", "squash-merge feature branches", "convention"),
            ],
        })
        with patch(
            "extensions.wiki_synthesizer.synthesizer._list_shared_group_ids",
            new=AsyncMock(return_value=["shared"]),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer._call_gemini",
            new=AsyncMock(return_value="second body"),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer."
            "patch_wiki_path_by_memory_ids",
            new=AsyncMock(return_value=1),
        ):
            second = await synthesize_all(
                service=service2,
                settings=settings,
                only_category="convention",
            )
        # Existing file, changed source set → updated, not created.
        assert second.pages_created == 0
        assert second.pages_updated == 1
        assert "synthesis_count: 2" in page_path.read_text()

    @pytest.mark.asyncio
    async def test_reserved_project_id_bucket_is_skipped_during_synthesis(
        self, tmp_path
    ):
        # If Neo4j somehow surfaces a group_id like
        # 'shared--project--global' (a project literally named "global"),
        # the synthesizer must skip it rather than overwrite the global
        # wiki tree. No page created, no Gemini call, no graph patch.
        settings = SynthesizerSettings(
            enabled=True, obsidian_vault_path=str(tmp_path)
        )
        service = _service_with_qdrant_points({
            ("shared--project--global", "decision"): [
                _qdrant_point("m1", "x", "decision"),
            ],
        })
        gemini_mock = AsyncMock(return_value="body")
        patch_mock = AsyncMock(return_value=1)
        with patch(
            "extensions.wiki_synthesizer.synthesizer._list_shared_group_ids",
            new=AsyncMock(return_value=["shared--project--global"]),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer._call_gemini",
            new=gemini_mock,
        ), patch(
            "extensions.wiki_synthesizer.synthesizer."
            "patch_wiki_path_by_memory_ids",
            new=patch_mock,
        ):
            result = await synthesize_all(
                service=service,
                settings=settings,
                only_category="decision",
            )
        assert result.pages_created == 0
        assert result.pages_updated == 0
        # Bucket skipped before any expensive work
        gemini_mock.assert_not_awaited()
        patch_mock.assert_not_awaited()


class TestIncrementalSynthesis:
    """Re-synthesis is skipped when a page's source memory-id set is unchanged."""

    def test_parse_id_list(self):
        from extensions.wiki_synthesizer.synthesizer import _parse_id_list
        assert _parse_id_list("[a, b, c]") == {"a", "b", "c"}
        assert _parse_id_list("[only]") == {"only"}
        assert _parse_id_list("[]") == set()
        assert _parse_id_list("") == set()

    @pytest.mark.asyncio
    async def test_unchanged_bucket_is_skipped_on_second_run(self, tmp_path):
        settings = SynthesizerSettings(enabled=True, obsidian_vault_path=str(tmp_path))
        points = {("shared", "convention"): [_qdrant_point("m1", "use PRs", "convention")]}
        service = _service_with_qdrant_points(points)
        with patch(
            "extensions.wiki_synthesizer.synthesizer._list_shared_group_ids",
            new=AsyncMock(return_value=["shared"]),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer._call_gemini",
            new=AsyncMock(return_value="body one"),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer.patch_wiki_path_by_memory_ids",
            new=AsyncMock(return_value=1),
        ):
            first = await synthesize_all(service=service, settings=settings, only_category="convention")
        assert first.pages_created == 1

        # Second run, identical source set → skipped, no LLM merge.
        service2 = _service_with_qdrant_points(points)
        gemini2 = AsyncMock(return_value="body two")
        with patch(
            "extensions.wiki_synthesizer.synthesizer._list_shared_group_ids",
            new=AsyncMock(return_value=["shared"]),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer._call_gemini", new=gemini2,
        ), patch(
            "extensions.wiki_synthesizer.synthesizer.patch_wiki_path_by_memory_ids",
            new=AsyncMock(return_value=1),
        ):
            second = await synthesize_all(service=service2, settings=settings, only_category="convention")
        assert second.pages_updated == 0
        assert second.pages_skipped_unchanged == 1
        gemini2.assert_not_awaited()  # the expensive merge was skipped

    @pytest.mark.asyncio
    async def test_force_resynthesizes_unchanged_bucket(self, tmp_path):
        settings = SynthesizerSettings(enabled=True, obsidian_vault_path=str(tmp_path))
        points = {("shared", "convention"): [_qdrant_point("m1", "use PRs", "convention")]}
        service = _service_with_qdrant_points(points)
        with patch(
            "extensions.wiki_synthesizer.synthesizer._list_shared_group_ids",
            new=AsyncMock(return_value=["shared"]),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer._call_gemini",
            new=AsyncMock(return_value="body one"),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer.patch_wiki_path_by_memory_ids",
            new=AsyncMock(return_value=1),
        ):
            await synthesize_all(service=service, settings=settings, only_category="convention")

        service2 = _service_with_qdrant_points(points)
        gemini2 = AsyncMock(return_value="body two")
        with patch(
            "extensions.wiki_synthesizer.synthesizer._list_shared_group_ids",
            new=AsyncMock(return_value=["shared"]),
        ), patch(
            "extensions.wiki_synthesizer.synthesizer._call_gemini", new=gemini2,
        ), patch(
            "extensions.wiki_synthesizer.synthesizer.patch_wiki_path_by_memory_ids",
            new=AsyncMock(return_value=1),
        ):
            forced = await synthesize_all(
                service=service2, settings=settings, only_category="convention", force=True,
            )
        assert forced.pages_skipped_unchanged == 0
        assert forced.pages_updated == 1
        gemini2.assert_awaited_once()  # force bypasses the skip
