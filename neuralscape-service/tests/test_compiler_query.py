"""Direct tests for extensions/conversation_compiler/query.py and the vault readers.

The writers put every audit-trail page (index, log, articles) under ``_raw/``.
The query, lint and status readers once looked for ``index.md`` / ``log.md`` at
the vault root, so against a real vault they saw no index, no log and counted
the daily logs as articles; these tests build the vault with the real writer.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("fcntl")

from extensions.conversation_compiler.lint import check_index_drift, check_orphan_pages
from extensions.conversation_compiler.obsidian_writer import ObsidianWriter
from extensions.conversation_compiler.query import query_knowledge_base
from extensions.conversation_compiler.routes import create_router

GEMINI = "extensions.conversation_compiler.compile._async_call_gemini"


@pytest.fixture
def writer(tmp_path):
    return ObsidianWriter(vault_path=tmp_path / "vault")


@pytest.fixture
def vault(writer):
    """A vault as the compiler leaves it: articles, index and log under ``_raw/``."""
    session = writer.write_session_summary("2026-05-15", "Worked on the Qdrant migration plan.")
    decision = writer.write_decision("use-qdrant", "We chose Qdrant for vector storage.")
    writer.write_research("graph-databases", "Neo4j versus Memgraph comparison notes.")
    writer.update_index(
        [
            {"path": session, "title": "Session Summary", "type": "session"},
            {"path": decision, "title": "Vector storage choice", "type": "decision"},
        ]
    )
    writer.append_log("Compiled 2026-05-15: 2 articles from 3 entries")
    return writer


@pytest.mark.asyncio
async def test_empty_vault_has_nothing_to_answer_with(writer):
    with patch(GEMINI, new=AsyncMock(return_value="unused")) as llm:
        result = await query_knowledge_base("anything at all", writer)
    assert result.sources == []
    assert "No vault pages" in result.answer
    llm.assert_not_called()


@pytest.mark.asyncio
async def test_keyword_overlap_selects_the_relevant_pages(vault):
    with patch(GEMINI, new=AsyncMock(return_value="Qdrant, per [[use-qdrant]].")) as llm:
        result = await query_knowledge_base("Which vector storage did we choose?", vault)

    assert result.answer == "Qdrant, per [[use-qdrant]]."
    # best match first: the decision page matches the path and the index description
    assert result.sources[0] == "_raw/Decisions/use-qdrant.md"
    prompt = llm.await_args.args[0]
    assert "Which vector storage did we choose?" in prompt
    assert "--- _raw/Decisions/use-qdrant.md ---" in prompt
    assert "We chose Qdrant for vector storage." in prompt


@pytest.mark.asyncio
async def test_index_descriptions_score_pages_whose_path_does_not_match(vault):
    # "Vector" appears only in the index description of the decision page, not in its path.
    with patch(GEMINI, new=AsyncMock(return_value="ok")):
        result = await query_knowledge_base("vector", vault)
    assert "_raw/Decisions/use-qdrant.md" in result.sources


@pytest.mark.asyncio
async def test_index_log_and_category_index_are_never_sources(vault):
    vault.append_category_entry(
        category="decision", content="Use Qdrant", project_id=None, session_id="s", timestamp="2026-05-15T10:00:00"
    )
    vault.update_category_index()
    with patch(GEMINI, new=AsyncMock(return_value="ok")):
        result = await query_knowledge_base("zzz-no-keyword-overlap", vault)  # falls back to "all pages"

    for meta in ("_raw/index.md", "_raw/log.md", "_raw/category-index.md"):
        assert meta not in result.sources
    assert result.sources  # the fallback still read real pages


@pytest.mark.asyncio
async def test_fallback_reads_at_most_ten_pages(writer):
    for i in range(14):
        writer.write_research(f"topic-{i}", f"Notes number {i}.")
    with patch(GEMINI, new=AsyncMock(return_value="ok")):
        result = await query_knowledge_base("zzzz", writer)
    assert len(result.sources) == 10


@pytest.mark.asyncio
async def test_matched_pages_are_capped_at_eight_and_truncated(writer):
    for i in range(12):
        writer.write_research(f"qdrant-{i}", "x" * 6000)
    with patch(GEMINI, new=AsyncMock(return_value="ok")) as llm:
        result = await query_knowledge_base("qdrant", writer)
    assert len(result.sources) == 8
    # each page is cut to its first 4000 characters (frontmatter included)
    prompt = llm.await_args.args[0]
    assert "x" * 3000 in prompt
    assert "x" * 4000 not in prompt


@pytest.mark.asyncio
async def test_llm_failure_reports_sources_without_an_answer(vault):
    with patch(GEMINI, new=AsyncMock(side_effect=RuntimeError("quota"))):
        result = await query_knowledge_base("vector storage", vault)
    assert "Failed to generate an answer" in result.answer
    assert result.sources


@pytest.mark.asyncio
async def test_empty_llm_answer_gets_a_placeholder(vault):
    with patch(GEMINI, new=AsyncMock(return_value="")):
        result = await query_knowledge_base("vector storage", vault)
    assert result.answer == "No answer generated."


@pytest.mark.asyncio
async def test_file_back_writes_a_research_page_and_indexes_it(vault):
    with patch(GEMINI, new=AsyncMock(return_value="Because of payload filters.")):
        result = await query_knowledge_base("Why Qdrant over Chroma?", vault, file_back=True)

    assert result.filed_back == "_raw/Research/why-qdrant-over-chroma.md"
    page = vault.read_file(result.filed_back)
    assert "Why Qdrant over Chroma?" in page and "Because of payload filters." in page
    assert f"[[{result.filed_back}]]" in vault.read_index()
    # a filed-back page is therefore not flagged by the index-drift lint
    assert not [f for f in check_index_drift(vault) if f.file == result.filed_back]


@pytest.mark.asyncio
async def test_file_back_with_a_wordless_question_still_files(vault):
    with patch(GEMINI, new=AsyncMock(return_value="42")):
        result = await query_knowledge_base("???", vault, file_back=True)
    assert result.filed_back == "_raw/Research/query.md"


@pytest.mark.asyncio
async def test_file_back_is_off_by_default(vault):
    before = set(vault.list_all_files())
    with patch(GEMINI, new=AsyncMock(return_value="ok")):
        result = await query_knowledge_base("vector storage", vault)
    assert result.filed_back is None
    assert set(vault.list_all_files()) == before


@pytest.mark.asyncio
async def test_regex_characters_in_a_page_path_do_not_break_scoring(writer):
    writer.write_research("c-plus-plus", "Notes.")
    writer.update_index([{"path": "_raw/Research/c-plus-plus.md", "title": "C++ (notes)", "type": "research"}])
    with patch(GEMINI, new=AsyncMock(return_value="ok")):
        result = await query_knowledge_base("plus", writer)
    assert result.sources == ["_raw/Research/c-plus-plus.md"]


# ── readers that share the same vault layout ──────────────────────────


def test_index_drift_sees_the_real_index_and_reports_only_missing_pages(vault):
    findings = check_index_drift(vault)
    assert [f.file for f in findings] == ["_raw/Research/graph-databases.md"]  # unindexed page only
    assert not any("index" in (f.file or "") or "log" in (f.file or "") for f in findings)


def test_index_drift_still_flags_a_vault_with_no_index(writer):
    writer.write_research("lonely", "Notes.")
    findings = check_index_drift(writer)
    assert [f.message for f in findings] == ["No index.md exists but vault has files"]


def test_orphan_check_does_not_report_the_meta_pages(vault):
    vault.append_category_entry(
        category="decision", content="Use Qdrant", project_id=None, session_id="s", timestamp="2026-05-15T10:00:00"
    )
    vault.update_category_index()
    orphans = {f.file for f in check_orphan_pages(vault)}
    assert not orphans & ObsidianWriter.META_FILES


def _client(vault):
    app = FastAPI()
    app.include_router(create_router(service=None, writer=vault, task_manager=None))
    return TestClient(app)


def test_status_reads_the_real_log_and_counts_articles_not_daily_logs(vault):
    vault.append_daily_log("2026-05-15", [{"time": "10:00", "category": "decision", "content": "x", "session_id": "s"}])
    vault.append_log("Flush sess-1: 1 memories stored, 1 to the vault")

    body = _client(vault).get("/status").json()

    assert body["last_compile"] is not None
    assert body["last_flush"] is not None
    assert body["daily_log_count"] == 1
    # 3 articles + the category-less daily log is excluded + index/log excluded
    assert body["article_count"] == 3


def test_status_on_an_empty_vault(writer):
    body = _client(writer).get("/status").json()
    assert body["last_flush"] is None and body["last_compile"] is None
    assert body["article_count"] == 0
