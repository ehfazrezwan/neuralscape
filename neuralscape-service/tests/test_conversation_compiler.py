"""Tests for the conversation-compiler extension.

Tests cover:
- ObsidianWriter (vault I/O, atomicity, frontmatter)
- Flush engine (extraction prompt parsing)
- Compile (grouping, idempotency)
- Lint checks (structural checks)
- Schemas (validation)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from extensions.conversation_compiler.config import CompilerSettings
from extensions.conversation_compiler.flush import (
    _map_category,
    _parse_extraction_response,
)
from extensions.conversation_compiler.obsidian_writer import (
    ObsidianWriter,
    _build_frontmatter,
    _slugify,
    _update_frontmatter_field,
)
from extensions.conversation_compiler.schemas import (
    CompileRequest,
    ExtractedFact,
    FlushRequest,
    FlushResult,
    LintFinding,
    QueryRequest,
    StatusResponse,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def tmp_vault(tmp_path):
    """Create a temporary vault directory."""
    vault = tmp_path / "test-vault"
    vault.mkdir()
    return vault


@pytest.fixture
def writer(tmp_vault):
    """Create an ObsidianWriter with a temp vault."""
    return ObsidianWriter(vault_path=tmp_vault)


# ──────────────────────────────────────────────
# ObsidianWriter tests
# ──────────────────────────────────────────────


class TestSlugify:
    def test_basic(self):
        assert _slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        assert _slugify("Use PostgreSQL vs MySQL?") == "use-postgresql-vs-mysql"

    def test_multiple_spaces(self):
        assert _slugify("too   many   spaces") == "too-many-spaces"

    def test_truncation(self):
        long = "a" * 100
        assert len(_slugify(long)) <= 80


class TestFrontmatter:
    def test_basic_frontmatter(self):
        fm = _build_frontmatter(title="Test Page")
        assert "---" in fm
        assert "title: Test Page" in fm

    def test_frontmatter_with_tags(self):
        fm = _build_frontmatter(title="Test", tags=["foo", "bar"])
        assert "tags: [foo, bar]" in fm

    def test_frontmatter_compiled_flag(self):
        fm = _build_frontmatter(title="Test", compiled=False)
        assert "compiled: false" in fm

    def test_update_frontmatter_field(self):
        content = "---\ntitle: Test\ncompiled: false\n---\n\nBody"
        updated = _update_frontmatter_field(content, "compiled", "true")
        assert "compiled: true" in updated
        assert "Body" in updated

    def test_update_frontmatter_adds_field(self):
        content = "---\ntitle: Test\n---\n\nBody"
        updated = _update_frontmatter_field(content, "compiled", "true")
        assert "compiled: true" in updated


class TestObsidianWriter:
    def test_append_daily_log_creates_file(self, writer, tmp_vault):
        entries = [
            {"time": "10:30", "category": "decision", "content": "Chose FastAPI", "session_id": "s1"}
        ]
        rel_path = writer.append_daily_log("2026-04-07", entries)
        assert rel_path == "_raw/Daily/2026-04-07.md"
        assert (tmp_vault / "_raw" / "Daily" / "2026-04-07.md").exists()

    def test_append_daily_log_content(self, writer, tmp_vault):
        entries = [
            {"time": "10:30", "category": "decision", "content": "Chose FastAPI", "session_id": "s1"}
        ]
        writer.append_daily_log("2026-04-07", entries)
        content = (tmp_vault / "_raw" / "Daily" / "2026-04-07.md").read_text()
        assert "Chose FastAPI" in content
        assert "`decision`" in content
        assert "session: s1" in content

    def test_append_daily_log_appends(self, writer, tmp_vault):
        writer.append_daily_log("2026-04-07", [{"time": "10:00", "category": "fact", "content": "First"}])
        writer.append_daily_log("2026-04-07", [{"time": "11:00", "category": "fact", "content": "Second"}])
        content = (tmp_vault / "_raw" / "Daily" / "2026-04-07.md").read_text()
        assert "First" in content
        assert "Second" in content

    def test_is_daily_log_compiled_false(self, writer):
        writer.append_daily_log("2026-04-07", [{"time": "10:00", "category": "fact", "content": "Test"}])
        assert not writer.is_daily_log_compiled("2026-04-07")

    def test_mark_daily_log_compiled(self, writer):
        writer.append_daily_log("2026-04-07", [{"time": "10:00", "category": "fact", "content": "Test"}])
        writer.mark_daily_log_compiled("2026-04-07")
        assert writer.is_daily_log_compiled("2026-04-07")

    def test_get_daily_log_entries(self, writer):
        entries = [
            {"time": "10:30", "category": "decision", "content": "Chose FastAPI", "session_id": "s1"},
            {"time": "11:00", "category": "preference", "content": "Prefers tabs", "session_id": "s2"},
        ]
        writer.append_daily_log("2026-04-07", entries)
        parsed = writer.get_daily_log_entries("2026-04-07")
        assert len(parsed) == 2
        assert parsed[0]["category"] == "decision"
        assert parsed[1]["content"] == "Prefers tabs"

    def test_list_daily_logs(self, writer):
        writer.append_daily_log("2026-04-06", [{"time": "10:00", "category": "fact", "content": "A"}])
        writer.append_daily_log("2026-04-07", [{"time": "10:00", "category": "fact", "content": "B"}])
        dates = writer.list_daily_logs()
        assert dates == ["2026-04-06", "2026-04-07"]

    def test_list_uncompiled_dates(self, writer):
        writer.append_daily_log("2026-04-06", [{"time": "10:00", "category": "fact", "content": "A"}])
        writer.append_daily_log("2026-04-07", [{"time": "10:00", "category": "fact", "content": "B"}])
        writer.mark_daily_log_compiled("2026-04-06")
        assert writer.list_uncompiled_dates() == ["2026-04-07"]

    def test_write_session_summary(self, writer, tmp_vault):
        rel_path = writer.write_session_summary("2026-04-07", "Today was productive.")
        assert rel_path == "_raw/Sessions/2026-04-07.md"
        content = (tmp_vault / "_raw" / "Sessions" / "2026-04-07.md").read_text()
        assert "Today was productive." in content

    def test_update_project_page_create(self, writer, tmp_vault):
        rel_path = writer.update_project_page("NeuralScape", "A memory service.")
        assert rel_path.startswith("_raw/Projects/")
        assert (tmp_vault / rel_path).exists()

    def test_write_decision(self, writer, tmp_vault):
        rel_path = writer.write_decision("use-fastapi", "Chose FastAPI because...")
        assert rel_path == "_raw/Decisions/use-fastapi.md"
        content = (tmp_vault / rel_path).read_text()
        assert "Chose FastAPI because..." in content

    def test_write_research(self, writer, tmp_vault):
        rel_path = writer.write_research("graphiti-vs-neo4j", "Comparison notes...")
        assert rel_path == "_raw/Research/graphiti-vs-neo4j.md"

    def test_update_index(self, writer, tmp_vault):
        entries = [
            {"path": "_raw/Sessions/2026-04-07.md", "title": "Session", "type": "session"},
            {"path": "_raw/Decisions/use-fastapi.md", "title": "Use FastAPI", "type": "decision"},
        ]
        writer.update_index(entries)
        content = (tmp_vault / "_raw" / "index.md").read_text()
        assert "[[_raw/Sessions/2026-04-07.md]]" in content
        assert "[[_raw/Decisions/use-fastapi.md]]" in content

    def test_append_log(self, writer, tmp_vault):
        writer.append_log("Something happened")
        content = (tmp_vault / "_raw" / "log.md").read_text()
        assert "Something happened" in content

    def test_list_all_files(self, writer, tmp_vault):
        (tmp_vault / "test.md").write_text("hello")
        (tmp_vault / "sub").mkdir()
        (tmp_vault / "sub" / "nested.md").write_text("world")
        files = writer.list_all_files()
        assert "test.md" in files
        assert "sub/nested.md" in files

    def test_find_wikilinks(self, writer):
        content = "See [[Page One]] and [[another-page]] for details."
        links = writer.find_wikilinks(content)
        assert "Page One" in links
        assert "another-page" in links


# ──────────────────────────────────────────────
# Flush engine tests
# ──────────────────────────────────────────────


class TestParseExtractionResponse:
    def test_valid_json(self):
        response = json.dumps({
            "facts": [
                {"type": "decision", "content": "Chose FastAPI", "project": "neuralscape", "tags": ["web"]},
                {"type": "preference", "content": "Prefers dark mode", "project": None, "tags": []},
            ]
        })
        facts = _parse_extraction_response(response)
        assert len(facts) == 2
        assert facts[0].category == "decision"
        assert facts[0].content == "Chose FastAPI"
        assert facts[0].project_id == "neuralscape"
        assert facts[1].category == "preference"

    def test_markdown_wrapped_json(self):
        response = "```json\n" + json.dumps({"facts": [{"type": "fact", "content": "Test"}]}) + "\n```"
        facts = _parse_extraction_response(response)
        assert len(facts) == 1

    def test_empty_facts(self):
        response = json.dumps({"facts": []})
        facts = _parse_extraction_response(response)
        assert len(facts) == 0

    def test_invalid_json(self):
        facts = _parse_extraction_response("this is not json")
        assert len(facts) == 0

    def test_missing_content(self):
        response = json.dumps({"facts": [{"type": "fact", "content": ""}]})
        facts = _parse_extraction_response(response)
        assert len(facts) == 0


class TestMapCategory:
    def test_decision(self):
        assert _map_category("decision") == "decision"

    def test_preference(self):
        assert _map_category("preference") == "preference"

    def test_unknown(self):
        assert _map_category("unknown_type") == "personal_fact"

    def test_pattern(self):
        assert _map_category("pattern") == "convention"


# ──────────────────────────────────────────────
# Schema tests
# ──────────────────────────────────────────────


class TestSchemas:
    def test_flush_request_validation(self):
        req = FlushRequest(
            user_message="Hello",
            assistant_response="Hi there",
            session_id="s1",
            user_id="ehfaz",
        )
        assert req.channel == "api"

    def test_compile_request_optional_date(self):
        req = CompileRequest(user_id="ehfaz")
        assert req.date is None

    def test_query_request(self):
        req = QueryRequest(question="How does X work?", user_id="ehfaz")
        assert req.file_back is False

    def test_extracted_fact(self):
        fact = ExtractedFact(category="decision", content="Chose X over Y")
        assert fact.project_id is None
        assert fact.tags == []

    def test_flush_result(self):
        result = FlushResult(session_id="s1", timestamp="2026-04-07T10:00:00")
        assert result.facts_extracted == 0

    def test_status_response(self):
        status = StatusResponse()
        assert status.extension == "conversation-compiler"


# ──────────────────────────────────────────────
# Lint tests (structural checks)
# ──────────────────────────────────────────────


class TestLintChecks:
    def test_broken_links(self, writer, tmp_vault):
        from extensions.conversation_compiler.lint import check_broken_links

        (tmp_vault / "page.md").write_text("See [[nonexistent-page]] for details.")
        findings = check_broken_links(writer)
        assert len(findings) == 1
        assert findings[0].check == "broken_links"

    def test_no_broken_links(self, writer, tmp_vault):
        from extensions.conversation_compiler.lint import check_broken_links

        (tmp_vault / "page.md").write_text("See [[other]] for details.")
        (tmp_vault / "other.md").write_text("Content here.")
        findings = check_broken_links(writer)
        assert len(findings) == 0

    def test_orphan_pages(self, writer, tmp_vault):
        from extensions.conversation_compiler.lint import check_orphan_pages

        (tmp_vault / "linked.md").write_text("See [[linked]] here.")
        (tmp_vault / "orphan.md").write_text("No links to me.")
        findings = check_orphan_pages(writer)
        orphan_files = [f.file for f in findings]
        assert "orphan.md" in orphan_files

    def test_index_drift(self, writer, tmp_vault):
        from extensions.conversation_compiler.lint import check_index_drift

        (tmp_vault / "page.md").write_text("Content")
        (tmp_vault / "index.md").write_text("# Index\n")
        findings = check_index_drift(writer)
        assert len(findings) == 1
        assert findings[0].check == "index_drift"

    def test_data_gaps(self, writer, tmp_vault):
        from extensions.conversation_compiler.lint import check_data_gaps

        (tmp_vault / "a.md").write_text("See [[missing-topic]] here.")
        (tmp_vault / "b.md").write_text("Also see [[missing-topic]].")
        findings = check_data_gaps(writer)
        assert any(f.check == "data_gaps" for f in findings)


# ──────────────────────────────────────────────
# Config tests
# ──────────────────────────────────────────────


class TestConfig:
    def test_default_settings(self):
        s = CompilerSettings()
        assert s.compile_after_hour == 18
        assert s.auto_compile is True

    def test_get_llm_model_default(self):
        s = CompilerSettings()
        assert s.get_llm_model("fallback-model") == "fallback-model"

    def test_get_llm_model_override(self):
        s = CompilerSettings(compiler_llm_model="custom-model")
        assert s.get_llm_model("fallback-model") == "custom-model"

    def test_vault_path(self):
        s = CompilerSettings(obsidian_vault_path="/tmp/test-vault")
        assert s.vault_path == Path("/tmp/test-vault").resolve()


# ──────────────────────────────────────────────
# Extension class tests
# ──────────────────────────────────────────────


class TestExtensionClass:
    def test_manifest_loads(self):
        from extensions.conversation_compiler import ConversationCompilerExtension

        ext = ConversationCompilerExtension()
        assert ext.manifest.name == "conversation-compiler"
        assert ext.manifest.version == "0.1.0"
        assert "conversation_turn" in ext.manifest.hooks
        assert "session_end" in ext.manifest.hooks
        assert "compile_requested" in ext.manifest.hooks

    def test_get_routes_returns_router(self):
        from extensions.conversation_compiler import ConversationCompilerExtension

        ext = ConversationCompilerExtension()
        router = ext.get_routes()
        assert router is not None


# ──────────────────────────────────────────────
# Path traversal protection
# ──────────────────────────────────────────────


class TestPathTraversal:
    def test_read_file_traversal_blocked(self, writer):
        with pytest.raises(ValueError, match="Path traversal"):
            writer.read_file("../../etc/passwd")

    def test_file_exists_traversal_blocked(self, writer):
        with pytest.raises(ValueError, match="Path traversal"):
            writer.file_exists("../../etc/passwd")

    def test_read_file_absolute_traversal_blocked(self, writer):
        with pytest.raises(ValueError, match="Path traversal"):
            writer.read_file("../../../tmp/secret")

    def test_read_file_valid_nested_path(self, writer, tmp_vault):
        sub = tmp_vault / "sub"
        sub.mkdir()
        (sub / "test.md").write_text("content")
        assert writer.read_file("sub/test.md") == "content"

    def test_file_exists_valid_path(self, writer, tmp_vault):
        (tmp_vault / "test.md").write_text("content")
        assert writer.file_exists("test.md") is True
        assert writer.file_exists("nonexistent.md") is False


# ──────────────────────────────────────────────
# Created flag correctness
# ──────────────────────────────────────────────


class TestCreatedFlag:
    @pytest.mark.asyncio
    async def test_created_true_for_new_session(self, writer, tmp_vault):
        """created should be True when session summary doesn't exist yet."""
        from extensions.conversation_compiler.compile import compile_date

        # Set up a daily log
        writer.append_daily_log("2026-04-07", [
            {"time": "10:00", "category": "fact", "content": "Test fact", "session_id": "s1"}
        ])

        mock_service = MagicMock()
        mock_service.dedup_memories = MagicMock()

        with patch(
            "extensions.conversation_compiler.compile._async_call_gemini",
            new_callable=AsyncMock,
            return_value="Summary text",
        ):
            result = await compile_date("2026-04-07", mock_service, writer, user_id="test")

        session_articles = [a for a in result.articles if a.article_type == "session"]
        assert len(session_articles) == 1
        assert session_articles[0].created is True

    @pytest.mark.asyncio
    async def test_created_false_for_existing_session(self, writer, tmp_vault):
        """created should be False when session summary already exists."""
        from extensions.conversation_compiler.compile import compile_date

        # Pre-create the session summary
        writer.write_session_summary("2026-04-07", "Old summary")
        writer.append_daily_log("2026-04-07", [
            {"time": "10:00", "category": "fact", "content": "Test fact", "session_id": "s1"}
        ])

        mock_service = MagicMock()
        mock_service.dedup_memories = MagicMock()

        with patch(
            "extensions.conversation_compiler.compile._async_call_gemini",
            new_callable=AsyncMock,
            return_value="New summary text",
        ):
            result = await compile_date("2026-04-07", mock_service, writer, user_id="test")

        session_articles = [a for a in result.articles if a.article_type == "session"]
        assert len(session_articles) == 1
        assert session_articles[0].created is False


# ──────────────────────────────────────────────
# User ID threading
# ──────────────────────────────────────────────


class TestUserIdThreading:
    @pytest.mark.asyncio
    async def test_compile_date_passes_user_id_to_dedup(self, writer, tmp_vault):
        """compile_date should pass user_id to service.dedup_memories."""
        from extensions.conversation_compiler.compile import compile_date

        writer.append_daily_log("2026-04-07", [
            {"time": "10:00", "category": "fact", "content": "Test", "session_id": "s1"}
        ])
        mock_service = MagicMock()
        mock_service.dedup_memories = MagicMock()

        with patch(
            "extensions.conversation_compiler.compile._async_call_gemini",
            new_callable=AsyncMock,
            return_value="Summary",
        ):
            result = await compile_date("2026-04-07", mock_service, writer, user_id="alice")

        mock_service.dedup_memories.assert_called_once_with("alice")
        assert result.dedup_triggered is True

    @pytest.mark.asyncio
    async def test_compile_date_default_user_id(self, writer, tmp_vault):
        """compile_date should use default user_id when not specified."""
        from extensions.conversation_compiler.compile import compile_date

        writer.append_daily_log("2026-04-07", [
            {"time": "10:00", "category": "fact", "content": "Test", "session_id": "s1"}
        ])
        mock_service = MagicMock()
        mock_service.dedup_memories = MagicMock()

        with patch(
            "extensions.conversation_compiler.compile._async_call_gemini",
            new_callable=AsyncMock,
            return_value="Summary",
        ):
            result = await compile_date("2026-04-07", mock_service, writer)

        mock_service.dedup_memories.assert_called_once_with("ehfaz")


# ──────────────────────────────────────────────
# Async Gemini wrapper
# ──────────────────────────────────────────────


class TestAsyncGeminiWrapper:
    @pytest.mark.asyncio
    async def test_returns_text(self):
        from extensions.conversation_compiler.compile import _async_call_gemini

        mock_response = MagicMock()
        mock_response.text = "Generated text"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("extensions.conversation_compiler.compile.genai.Client", return_value=mock_client):
            result = await _async_call_gemini("Test prompt")

        assert result == "Generated text"

    @pytest.mark.asyncio
    async def test_handles_none_text(self):
        from extensions.conversation_compiler.compile import _async_call_gemini

        mock_response = MagicMock()
        mock_response.text = None
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("extensions.conversation_compiler.compile.genai.Client", return_value=mock_client):
            result = await _async_call_gemini("Test prompt")

        assert result == ""


# ──────────────────────────────────────────────
# Category entry writer
# ──────────────────────────────────────────────


class TestCategoryEntryWriter:
    def test_creates_file_with_frontmatter(self, writer, tmp_vault):
        """First write to a category creates the file with frontmatter."""
        path = writer.append_category_entry(
            category="preference",
            content="Prefers dark mode",
            project_id=None,
            session_id="s1",
            timestamp="2026-04-29T10:30:00",
        )
        assert path == "_raw/Semantic/Preferences/entries.md"
        full_path = tmp_vault / path
        assert full_path.exists()
        content = full_path.read_text()
        assert "title: Preferences" in content
        assert "preference" in content
        assert "Prefers dark mode" in content

    def test_appends_to_existing_file(self, writer, tmp_vault):
        """Subsequent writes append to the existing file."""
        writer.append_category_entry(
            category="preference", content="Fact one",
            project_id=None, session_id="s1", timestamp="2026-04-29T10:00:00",
        )
        writer.append_category_entry(
            category="preference", content="Fact two",
            project_id=None, session_id="s2", timestamp="2026-04-29T11:00:00",
        )
        content = (tmp_vault / "_raw" / "Semantic" / "Preferences" / "entries.md").read_text()
        assert "Fact one" in content
        assert "Fact two" in content

    def test_project_scoped_uses_project_filename(self, writer, tmp_vault):
        """Project-scoped categories use project slug as filename."""
        path = writer.append_category_entry(
            category="tech_stack",
            content="Uses FastAPI for REST endpoints",
            project_id="neuralscape",
            session_id="s1",
            timestamp="2026-04-29T10:00:00",
        )
        assert path == "_raw/Project/Tech-Stack/neuralscape.md"
        assert (tmp_vault / path).exists()

    def test_global_scoped_uses_entries_file(self, writer, tmp_vault):
        """Global categories always use entries.md regardless of project_id."""
        path = writer.append_category_entry(
            category="personal_fact",
            content="Based in Dhaka",
            project_id=None,
            session_id="s1",
            timestamp="2026-04-29T10:00:00",
        )
        assert path == "_raw/Semantic/Personal-Facts/entries.md"

    def test_unknown_category_fallback(self, writer, tmp_vault):
        """Unknown categories fall back to Uncategorized/ folder."""
        path = writer.append_category_entry(
            category="invented_category",
            content="Some fact",
            project_id=None,
            session_id="s1",
            timestamp="2026-04-29T10:00:00",
        )
        assert path.startswith("_raw/Uncategorized/")
        assert (tmp_vault / path).exists()


# ──────────────────────────────────────────────
# Category index
# ──────────────────────────────────────────────


class TestCategoryIndex:
    def test_empty_vault(self, writer, tmp_vault):
        """Empty vault produces a minimal index."""
        path = writer.update_category_index()
        assert path == "_raw/category-index.md"
        content = (tmp_vault / path).read_text()
        assert "Category Index" in content

    def test_with_entries(self, writer, tmp_vault):
        """Index correctly counts entries in category files."""
        writer.append_category_entry(
            category="preference", content="Fact A",
            project_id=None, session_id="s1", timestamp="2026-04-29T10:00:00",
        )
        writer.append_category_entry(
            category="preference", content="Fact B",
            project_id=None, session_id="s2", timestamp="2026-04-29T11:00:00",
        )
        writer.update_category_index()
        content = (tmp_vault / "_raw" / "category-index.md").read_text()
        assert "2 entries" in content
        assert "_raw/Semantic/Preferences/entries.md" in content

    def test_multiple_projects(self, writer, tmp_vault):
        """Project-scoped categories list multiple project files."""
        writer.append_category_entry(
            category="tech_stack", content="Uses FastAPI",
            project_id="neuralscape", session_id="s1", timestamp="2026-04-29T10:00:00",
        )
        writer.append_category_entry(
            category="tech_stack", content="Uses grammY",
            project_id="demo-gamma", session_id="s2", timestamp="2026-04-29T11:00:00",
        )
        writer.update_category_index()
        content = (tmp_vault / "_raw" / "category-index.md").read_text()
        assert "neuralscape.md" in content
        assert "demo-gamma.md" in content


# ──────────────────────────────────────────────
# Vault path coverage
# ──────────────────────────────────────────────


class TestVaultPaths:
    def test_all_categories_have_vault_paths(self):
        """Every MemoryCategory enum value has a CATEGORY_VAULT_PATHS entry."""
        from schemas import CATEGORY_VAULT_PATHS, MemoryCategory

        for cat in MemoryCategory:
            assert cat.value in CATEGORY_VAULT_PATHS, f"Missing vault path for {cat.value}"


# ──────────────────────────────────────────────
# _handle_memory_stored tests
# ──────────────────────────────────────────────


class TestHandleMemoryStored:
    """Tests for the ConversationCompilerExtension._handle_memory_stored handler."""

    @pytest.fixture
    def extension(self, tmp_vault):
        from extensions.conversation_compiler import ConversationCompilerExtension
        from extensions.conversation_compiler.obsidian_writer import ObsidianWriter

        ext = ConversationCompilerExtension()
        ext._writer = ObsidianWriter(vault_path=tmp_vault)
        ext._service = MagicMock()
        return ext

    @pytest.mark.asyncio
    async def test_skips_conversation_compiler_source(self, extension):
        """Events from the conversation-compiler flush path should be skipped."""
        result = await extension._handle_memory_stored({
            "source": "conversation-compiler",
            "content": "some fact",
            "category": "preference",
            "user_id": "ehfaz",
        })
        assert result is None

    @pytest.mark.asyncio
    async def test_skips_empty_content(self, extension):
        """Events with empty content should be skipped."""
        result = await extension._handle_memory_stored({
            "source": "worker",
            "content": "",
            "category": "preference",
            "user_id": "ehfaz",
        })
        assert result is None

    @pytest.mark.asyncio
    async def test_skips_empty_category(self, extension):
        """Events with empty category should be skipped."""
        result = await extension._handle_memory_stored({
            "source": "worker",
            "content": "some fact",
            "category": "",
            "user_id": "ehfaz",
        })
        assert result is None

    @pytest.mark.asyncio
    async def test_writes_to_vault(self, extension, tmp_vault):
        """Worker-sourced shared events should write to category folder and daily log."""
        # 'decision' defaults to SHARED visibility, so the privacy gate
        # in _handle_memory_stored passes this through to the vault.
        result = await extension._handle_memory_stored({
            "source": "worker",
            "content": "Chose dark mode as the default theme",
            "category": "decision",
            "visibility": "shared",
            "user_id": "ehfaz",
            "project_id": None,
            "run_id": "sess-123",
        })
        assert result is not None
        assert "vault_path" in result

        # Verify category file was created
        cat_file = tmp_vault / "_raw" / "Episodic" / "Decisions" / "entries.md"
        assert cat_file.exists()
        content = cat_file.read_text()
        assert "Chose dark mode as the default theme" in content
        assert "sess-123" in content

        # Verify daily log was created
        daily_files = list((tmp_vault / "_raw" / "Daily").glob("*.md"))
        assert len(daily_files) == 1
        daily_content = daily_files[0].read_text()
        assert "Chose dark mode as the default theme" in daily_content

    @pytest.mark.asyncio
    async def test_writes_project_scoped(self, extension, tmp_vault):
        """Project-scoped categories should use project slug as filename."""
        result = await extension._handle_memory_stored({
            "source": "worker",
            "content": "Uses FastAPI for API layer",
            "category": "tech_stack",
            "user_id": "ehfaz",
            "project_id": "neuralscape",
            "run_id": "sess-456",
        })
        assert result is not None

        cat_file = tmp_vault / "_raw" / "Project" / "Tech-Stack" / "neuralscape.md"
        assert cat_file.exists()
        content = cat_file.read_text()
        assert "Uses FastAPI for API layer" in content

    @pytest.mark.asyncio
    async def test_handles_writer_exception(self, extension):
        """Writer exceptions should be caught and return None."""
        extension._writer = MagicMock()
        extension._writer.append_category_entry.side_effect = OSError("disk full")

        # SHARED category — the privacy gate lets this through, so the
        # writer is actually invoked and its exception is what we test.
        result = await extension._handle_memory_stored({
            "source": "worker",
            "content": "Convention update",
            "category": "convention",
            "visibility": "shared",
            "user_id": "ehfaz",
        })
        assert result is None

    @pytest.mark.asyncio
    async def test_worker_source_passes_through(self, extension, tmp_vault):
        """Events with source='worker' (and a shared category) should NOT be skipped."""
        # Use a SHARED-default category so the privacy gate doesn't drop it;
        # the source-skip logic only fires for source='conversation-compiler'.
        result = await extension._handle_memory_stored({
            "source": "worker",
            "content": "Team uses Conventional Commits",
            "category": "convention",
            "user_id": "ehfaz",
        })
        assert result is not None
        cat_file = tmp_vault / "_raw" / "Project" / "Conventions" / "entries.md"
        assert cat_file.exists()
