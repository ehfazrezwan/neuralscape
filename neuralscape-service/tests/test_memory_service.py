"""Tests for MemoryService business logic."""

from unittest.mock import MagicMock, patch

import pytest

from memory_service import (
    MemoryService,
    _build_group_id,
    _clean_conversation_for_graph,
    _get_group_ids,
    _infer_project_id,
    _is_junk_fact,
)
from schemas import MemoryResponse, MemoryScope


# ──────────────────────────────────────────────
# Helper function tests
# ──────────────────────────────────────────────


class TestBuildGroupId:
    def test_global_scope(self):
        assert _build_group_id("global") == "global"

    def test_project_scope_with_id(self):
        assert _build_group_id("project", "my-project") == "project--my-project"

    def test_project_scope_without_id_falls_back_to_global(self):
        assert _build_group_id("project", None) == "global"


class TestGetGroupIds:
    def test_no_project_returns_global_only(self):
        assert _get_group_ids() == ["global"]

    def test_with_project_returns_both(self):
        ids = _get_group_ids("my-project")
        assert "global" in ids
        assert "project--my-project" in ids
        assert len(ids) == 2


# ──────────────────────────────────────────────
# MemoryService unit tests (with mocked mem0)
# ──────────────────────────────────────────────


@pytest.fixture
def service():
    """Create a MemoryService with mocked internals."""
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    # Mock the graph attribute on memory
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    return svc


class TestStoreRaw:
    def test_stores_with_metadata(self, service):
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        result = service.store_raw(
            content="Prefers tabs",
            user_id="ehfaz",
            category="preference",
        )
        assert len(result) == 1
        assert result[0].memory == "Prefers tabs"
        assert result[0].category == "preference"
        assert result[0].scope == "global"

        # Should bypass m.add and call vector_store.insert directly
        service._memory.add.assert_not_called()
        service._memory.vector_store.insert.assert_called_once()
        call_kwargs = service._memory.vector_store.insert.call_args[1]
        payload = call_kwargs["payloads"][0]
        assert payload["data"] == "Prefers tabs"
        assert payload["metadata"]["category"] == "preference"
        assert payload["metadata"]["scope"] == "global"
        assert payload["metadata"]["source"] == "explicit"

    def test_rejects_invalid_category(self, service):
        with pytest.raises(ValueError, match="Invalid category"):
            service.store_raw(content="test", user_id="u1", category="bogus")

    def test_requires_project_id_for_project_scope(self, service):
        with pytest.raises(ValueError, match="project_id is required"):
            service.store_raw(
                content="test", user_id="u1", category="tech_stack", scope="project"
            )

    def test_includes_tags_in_metadata(self, service):
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service.store_raw(
            content="Uses Python",
            user_id="u1",
            category="technical_skill",
            tags=["python", "backend"],
        )
        call_kwargs = service._memory.vector_store.insert.call_args[1]
        payload = call_kwargs["payloads"][0]
        assert payload["metadata"]["tags"] == ["python", "backend"]


class TestSearch:
    def test_basic_search(self, service):
        service._memory.search.return_value = {
            "results": [
                {"id": "m1", "memory": "Prefers tabs", "score": 0.95, "metadata": {"category": "preference"}}
            ]
        }
        results = service.search(query="indentation", user_id="ehfaz")
        assert len(results) == 1
        assert results[0].memory == "Prefers tabs"

    def test_search_with_project_merges_scopes(self, service):
        service._memory.search.return_value = {
            "results": [
                {"id": "m1", "memory": "Uses FastAPI", "score": 0.9, "metadata": {}}
            ]
        }
        results = service.search(
            query="tech stack",
            user_id="ehfaz",
            project_id="my-project",
        )
        # Should call search twice (project + global)
        assert service._memory.search.call_count == 2

    def test_search_with_explicit_scope_single_call(self, service):
        service._memory.search.return_value = {"results": []}
        service.search(
            query="preferences",
            user_id="ehfaz",
            scope="global",
        )
        assert service._memory.search.call_count == 1

    def test_search_with_categories(self, service):
        service._memory.search.return_value = {"results": []}
        service.search(
            query="coding style",
            user_id="ehfaz",
            categories=["preference", "convention"],
        )
        call_kwargs = service._memory.search.call_args[1]
        assert call_kwargs["filters"]["category"] == {"in": ["preference", "convention"]}


class TestGetContext:
    def test_global_context_organizes_by_category(self, service):
        service._memory.get_all.return_value = {
            "results": [
                {"id": "m1", "memory": "Prefers tabs", "metadata": {"category": "preference", "scope": "global"}},
                {"id": "m2", "memory": "Knows Python", "metadata": {"category": "technical_skill", "scope": "global"}},
            ]
        }
        ctx = service.get_global_context(user_id="ehfaz")
        assert ctx.user_id == "ehfaz"
        assert "preference" in ctx.categories
        assert "technical_skill" in ctx.categories

    def test_project_context_includes_both_scopes(self, service):
        # Mock returns different results for each call
        service._memory.get_all.side_effect = [
            {"results": [{"id": "m1", "memory": "Prefers tabs", "metadata": {"category": "preference"}}]},
            {"results": [{"id": "m2", "memory": "Uses FastAPI", "metadata": {"category": "tech_stack"}}]},
        ]
        ctx = service.get_project_context(user_id="ehfaz", project_id="my-project")
        assert ctx.project_id == "my-project"
        assert service._memory.get_all.call_count == 2


class TestCRUD:
    def test_get_memory_found(self, service):
        service._memory.get.return_value = {
            "id": "m1", "memory": "Prefers tabs", "metadata": {"category": "preference"}
        }
        result = service.get_memory("m1")
        assert result is not None
        assert result.id == "m1"

    def test_get_memory_not_found(self, service):
        service._memory.get.return_value = None
        result = service.get_memory("nonexistent")
        assert result is None

    def test_list_memories(self, service):
        service._memory.get_all.return_value = {
            "results": [
                {"id": "m1", "memory": "fact1", "metadata": {}},
                {"id": "m2", "memory": "fact2", "metadata": {}},
            ]
        }
        results = service.list_memories(user_id="ehfaz")
        assert len(results) == 2

    def test_list_memories_with_filters(self, service):
        service._memory.get_all.return_value = {"results": []}
        service.list_memories(
            user_id="ehfaz",
            scope="global",
            category="preference",
            project_id="my-project",
        )
        call_kwargs = service._memory.get_all.call_args[1]
        assert call_kwargs["filters"]["scope"] == "global"
        assert call_kwargs["filters"]["category"] == "preference"
        assert call_kwargs["filters"]["project_id"] == "my-project"

    def test_update_memory(self, service):
        service._memory.get.return_value = {
            "id": "m1",
            "memory": "Old content",
            "user_id": "ehfaz",
            "metadata": {"scope": "global", "category": "preference"},
        }
        service._memory.update.return_value = {"message": "Memory updated successfully!"}
        result = service.update_memory(memory_id="m1", content="Updated content")
        service._memory.update.assert_called_once_with("m1", "Updated content")

    def test_update_memory_reingests_into_graph(self, service):
        """When content is updated, the new content should be re-ingested into
        the knowledge graph so Graphiti can expire contradicting edges."""
        service._memory.get.return_value = {
            "id": "m1",
            "memory": "User prefers dark mode",
            "user_id": "ehfaz",
            "metadata": {"scope": "project", "category": "preference", "project_id": "p1"},
        }
        service._memory.update.return_value = {"message": "Memory updated successfully!"}

        service.update_memory(memory_id="m1", content="User prefers light mode")

        service._memory.graph.add.assert_called_once_with(
            data="User prefers light mode",
            filters={"user_id": "ehfaz", "group_id": "project--p1"},
        )

    def test_update_memory_skips_graph_for_metadata_only(self, service):
        """Metadata-only updates (no content) should not trigger graph re-ingestion."""
        service.update_memory(memory_id="m1", category="preference")
        service._memory.graph.add.assert_not_called()

    def test_update_memory_graph_failure_noncritical(self, service):
        """Graph re-ingestion failure should not prevent the update from succeeding."""
        service._memory.get.return_value = {
            "id": "m1",
            "memory": "Old content",
            "user_id": "ehfaz",
            "metadata": {"scope": "global"},
        }
        service._memory.update.return_value = {"message": "Memory updated successfully!"}
        service._memory.graph.add.side_effect = Exception("Neo4j connection refused")

        result = service.update_memory(memory_id="m1", content="New content")
        assert result["message"] == "Memory updated successfully"

    def test_update_memory_rejects_invalid_category(self, service):
        with pytest.raises(ValueError, match="Invalid category"):
            service.update_memory(memory_id="m1", category="bogus")

    def test_delete_memory(self, service):
        service._memory.delete.return_value = {"message": "Memory deleted successfully!"}
        result = service.delete_memory("m1")
        service._memory.delete.assert_called_once_with("m1")

    def test_delete_memories_all(self, service):
        result = service.delete_memories(user_id="ehfaz")
        service._memory.delete_all.assert_called_once_with(user_id="ehfaz")

    def test_delete_memories_with_filters(self, service):
        service._memory.get_all.return_value = {
            "results": [
                {"id": "m1", "memory": "fact1", "metadata": {"scope": "global"}},
                {"id": "m2", "memory": "fact2", "metadata": {"scope": "global"}},
            ]
        }
        service._memory.delete.return_value = {"message": "deleted"}
        result = service.delete_memories(user_id="ehfaz", scope="global")
        assert service._memory.delete.call_count == 2


class TestExtractAndStore:
    def test_extraction_batch_stores_categorized_facts(self, service):
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.return_value = MagicMock(
            text='{"facts": ["[preference] Prefers tabs over spaces", "[technical_skill] Expert in Python 3.12"]}'
        )

        # Mock batch embed returning one vector per fact
        service._memory.embedding_model.embed_batch.return_value = [
            [0.1] * 768,
            [0.2] * 768,
        ]

        results = service.extract_and_store(
            messages=[{"role": "user", "content": "I prefer tabs and I'm an expert in Python 3.12"}],
            user_id="ehfaz",
        )

        # Should NOT call m.add — uses batch path instead
        service._memory.add.assert_not_called()

        # Single embed_batch call with both facts
        service._memory.embedding_model.embed_batch.assert_called_once()
        embed_texts = service._memory.embedding_model.embed_batch.call_args[0][0]
        assert len(embed_texts) == 2

        # Single Qdrant upsert with both facts
        service._memory.vector_store.insert.assert_called_once()
        insert_kwargs = service._memory.vector_store.insert.call_args[1]
        assert len(insert_kwargs["vectors"]) == 2
        assert len(insert_kwargs["ids"]) == 2
        assert len(insert_kwargs["payloads"]) == 2

        # Returns 2 MemoryResponse objects
        assert len(results) == 2

    def test_extraction_returns_empty_on_llm_error(self, service):
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.side_effect = Exception("API error")

        results = service.extract_and_store(
            messages=[{"role": "user", "content": "hello"}],
            user_id="ehfaz",
        )

        # Should return empty list instead of falling back to m.add()
        assert results == []
        service._memory.add.assert_not_called()

    def test_extraction_still_calls_graph_add(self, service):
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.return_value = MagicMock(
            text='{"facts": ["[preference] Prefers dark mode"]}'
        )
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        service.extract_and_store(
            messages=[{"role": "user", "content": "I prefer dark mode"}],
            user_id="ehfaz",
        )

        # Graph add should still be called with the raw conversation
        service._memory.graph.add.assert_called_once()


class TestBatchStoreFacts:
    def test_batch_stores_multiple_facts(self, service):
        service._memory.embedding_model.embed_batch.return_value = [
            [0.1] * 768,
            [0.2] * 768,
        ]

        facts = [
            ("preference", "Prefers dark mode"),
            ("technical_skill", "Expert in Python"),
        ]
        results = service._batch_store_facts(facts=facts, user_id="ehfaz")

        assert len(results) == 2
        assert results[0].category == "preference"
        assert results[0].scope == "global"
        assert results[1].category == "technical_skill"
        assert results[1].scope == "global"

        # Single embed_batch call
        service._memory.embedding_model.embed_batch.assert_called_once()

        # Single Qdrant insert
        service._memory.vector_store.insert.assert_called_once()
        insert_kwargs = service._memory.vector_store.insert.call_args[1]
        assert len(insert_kwargs["vectors"]) == 2

        # History recorded for each fact
        assert service._memory.db.add_history.call_count == 2

    def test_empty_facts_returns_empty(self, service):
        results = service._batch_store_facts(facts=[], user_id="ehfaz")
        assert results == []
        service._memory.embedding_model.embed_batch.assert_not_called()
        service._memory.vector_store.insert.assert_not_called()

    def test_project_category_gets_project_scope(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        facts = [("tech_stack", "Uses FastAPI")]
        results = service._batch_store_facts(
            facts=facts, user_id="ehfaz", project_id="my-project"
        )

        assert results[0].scope == "project"
        assert results[0].project_id == "my-project"

    def test_project_category_without_project_id_falls_back_to_global(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        facts = [("tech_stack", "Uses FastAPI")]
        results = service._batch_store_facts(facts=facts, user_id="ehfaz")

        # tech_stack normally requires project_id, should fall back to global
        assert results[0].scope == "global"

    def test_global_category_stays_global_even_with_project_id(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        facts = [("preference", "Prefers dark mode")]
        results = service._batch_store_facts(
            facts=facts, user_id="ehfaz", project_id="my-project"
        )

        # preference is a GLOBAL_CATEGORIES member, should stay global
        assert results[0].scope == "global"

    def test_payload_structure(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        facts = [("preference", "Prefers tabs")]
        service._batch_store_facts(facts=facts, user_id="ehfaz", agent_id="agent-1")

        insert_kwargs = service._memory.vector_store.insert.call_args[1]
        payload = insert_kwargs["payloads"][0]

        assert payload["data"] == "Prefers tabs"
        assert "hash" in payload
        assert "created_at" in payload
        assert payload["user_id"] == "ehfaz"
        assert payload["agent_id"] == "agent-1"
        assert payload["metadata"]["category"] == "preference"
        assert payload["metadata"]["source"] == "conversation"


class TestMergeResults:
    def test_deduplicates_by_id(self):
        svc = MemoryService()
        result1 = {"results": [{"id": "m1", "memory": "fact1", "score": 0.9}]}
        result2 = {"results": [{"id": "m1", "memory": "fact1", "score": 0.8}]}
        merged = svc._merge_results(result1, result2)
        assert len(merged) == 1

    def test_sorts_by_score(self):
        svc = MemoryService()
        result1 = {"results": [{"id": "m1", "memory": "low", "score": 0.3}]}
        result2 = {"results": [{"id": "m2", "memory": "high", "score": 0.9}]}
        merged = svc._merge_results(result1, result2)
        assert merged[0]["id"] == "m2"


# ──────────────────────────────────────────────
# Junk filter tests
# ──────────────────────────────────────────────


class TestIsJunkFact:
    def test_short_content_is_junk(self):
        assert _is_junk_fact("hi") is True
        assert _is_junk_fact("") is True
        assert _is_junk_fact("   ab   ") is True

    def test_ran_command_is_junk(self):
        assert _is_junk_fact("Ran command: git status") is True

    def test_edited_file_is_junk(self):
        assert _is_junk_fact("Edited file: src/main.py") is True
        assert _is_junk_fact("Edited file src/main.py line 42") is True

    def test_wrote_file_is_junk(self):
        assert _is_junk_fact("Wrote file: /tmp/output.txt") is True

    def test_read_file_is_junk(self):
        assert _is_junk_fact("Read file: config.json") is True

    def test_tool_result_is_junk(self):
        assert _is_junk_fact("Tool result: success, 3 files changed") is True

    def test_command_output_is_junk(self):
        assert _is_junk_fact("Command output: npm install completed") is True

    def test_launched_task_is_junk(self):
        assert _is_junk_fact("Launched background task: test-runner") is True

    def test_real_fact_is_not_junk(self):
        assert _is_junk_fact("Ehfaz prefers tabs over spaces") is False
        assert _is_junk_fact("The neuralscape project uses FastAPI with Qdrant") is False
        assert _is_junk_fact("User prefers dark mode in all editors") is False

    def test_case_insensitive(self):
        assert _is_junk_fact("RAN COMMAND: ls -la") is True
        assert _is_junk_fact("edited FILE: foo.py") is True


class TestCleanConversationForGraph:
    def test_removes_junk_lines_from_content(self):
        messages = [
            {"role": "assistant", "content": "Ran command: git status\nGot it, here's the status."},
        ]
        cleaned = _clean_conversation_for_graph(messages)
        assert len(cleaned) == 1
        assert "Ran command:" not in cleaned[0]["content"]
        assert "Got it" in cleaned[0]["content"]

    def test_drops_message_that_becomes_empty(self):
        messages = [
            {"role": "user", "content": "I prefer dark mode"},
            {"role": "assistant", "content": "Ran command: echo ok"},
        ]
        cleaned = _clean_conversation_for_graph(messages)
        assert len(cleaned) == 1
        assert cleaned[0]["content"] == "I prefer dark mode"

    def test_preserves_clean_messages(self):
        messages = [
            {"role": "user", "content": "I prefer tabs over spaces"},
            {"role": "assistant", "content": "Noted, storing that preference."},
        ]
        cleaned = _clean_conversation_for_graph(messages)
        assert len(cleaned) == 2
        assert cleaned[0]["content"] == "I prefer tabs over spaces"
        assert cleaned[1]["content"] == "Noted, storing that preference."

    def test_preserves_empty_content_messages(self):
        messages = [{"role": "system", "content": ""}]
        cleaned = _clean_conversation_for_graph(messages)
        assert len(cleaned) == 1

    def test_filters_multiple_junk_patterns(self):
        messages = [
            {"role": "assistant", "content": "Edited file: src/main.py\nWrote file: /tmp/out.txt\nTool result: success\nDone with the changes."},
        ]
        cleaned = _clean_conversation_for_graph(messages)
        assert len(cleaned) == 1
        assert cleaned[0]["content"] == "Done with the changes."

    def test_empty_messages_list(self):
        assert _clean_conversation_for_graph([]) == []

    def test_preserves_role_and_other_keys(self):
        messages = [
            {"role": "user", "content": "hello", "name": "ehfaz"},
        ]
        cleaned = _clean_conversation_for_graph(messages)
        assert cleaned[0]["role"] == "user"
        assert cleaned[0]["name"] == "ehfaz"

    def test_case_insensitive_junk_detection(self):
        messages = [
            {"role": "assistant", "content": "RAN COMMAND: ls -la\nHere are the files."},
        ]
        cleaned = _clean_conversation_for_graph(messages)
        assert len(cleaned) == 1
        assert "RAN COMMAND:" not in cleaned[0]["content"]
        assert "Here are the files." in cleaned[0]["content"]


class TestExtractAndStoreJunkFilter:
    def test_junk_facts_filtered_from_extraction(self, service):
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.return_value = MagicMock(
            text='{"facts": ["[preference] Prefers tabs over spaces", "[interaction] Ran command: git status"]}'
        )

        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        results = service.extract_and_store(
            messages=[{"role": "user", "content": "I prefer tabs"}],
            user_id="ehfaz",
        )

        # Only 1 fact should remain after filtering
        assert len(results) == 1
        assert results[0].memory == "Prefers tabs over spaces"

    def test_graph_text_has_junk_lines_stripped(self, service):
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.return_value = MagicMock(
            text='{"facts": ["[preference] Prefers dark mode"]}'
        )
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        service.extract_and_store(
            messages=[
                {"role": "user", "content": "I prefer dark mode"},
                {"role": "assistant", "content": "Ran command: echo ok\nGot it, storing preference."},
            ],
            user_id="ehfaz",
        )

        # Graph add should be called with text that doesn't contain the junk line
        call_args = service._memory.graph.add.call_args
        graph_text = call_args[1]["data"] if "data" in call_args[1] else call_args[0][0]
        assert "Ran command:" not in graph_text
        assert "dark mode" in graph_text

    def test_graph_add_skipped_when_all_messages_are_junk(self, service):
        """If _clean_conversation_for_graph removes all content, graph.add() should not be called."""
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.return_value = MagicMock(
            text='{"facts": ["[preference] Prefers dark mode"]}'
        )
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        service.extract_and_store(
            messages=[
                {"role": "assistant", "content": "Ran command: git status"},
                {"role": "assistant", "content": "Tool result: success"},
            ],
            user_id="ehfaz",
        )

        service._memory.graph.add.assert_not_called()

    def test_store_raw_does_not_filter_graph_content(self, service):
        """store_raw() graph path should NOT apply conversation junk filter."""
        service._memory.embedding_model.embed.return_value = [0.1] * 768

        service.store_raw(
            content="Ran command: git status\nImportant context.",
            user_id="ehfaz",
            category="task_context",
        )

        # store_raw passes content directly to graph.add without filtering
        call_args = service._memory.graph.add.call_args
        graph_text = call_args[1]["data"] if "data" in call_args[1] else call_args[0][0]
        assert "Ran command:" in graph_text


# ──────────────────────────────────────────────
# Null-category bulk delete tests
# ──────────────────────────────────────────────


class TestBulkDeleteNullCategory:
    def test_null_category_does_not_trigger_delete_all(self, service):
        """Passing category=None should NOT delete all memories."""
        service._memory.get_all.return_value = {"results": []}
        service._memory.delete.return_value = {"message": "deleted"}

        # Without filter_null_category, category=None + no other filters = delete all
        service.delete_memories(user_id="ehfaz")
        service._memory.delete_all.assert_called_once()

        service._memory.reset_mock()

        # With filter_null_category=True, should NOT call delete_all
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)
        service.delete_memories(user_id="ehfaz", filter_null_category=True)
        service._memory.delete_all.assert_not_called()

    def test_filter_null_category_uses_qdrant_scroll(self, service):
        """filter_null_category should use IsNullCondition scroll."""
        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client

        mock_point = MagicMock()
        mock_point.id = "point-1"
        mock_point.payload = {"data": "some uncategorized memory", "metadata": {}}
        mock_client.scroll.return_value = ([mock_point], None)

        result = service.delete_memories(user_id="ehfaz", filter_null_category=True)

        # Should have called scroll with IsNullCondition
        mock_client.scroll.assert_called_once()
        # Should have deleted the found point
        service._memory.vector_store.delete.assert_called_once_with("point-1")
        assert "1 null-category" in result["message"]


# ──────────────────────────────────────────────
# project_id inference tests
# ──────────────────────────────────────────────


class TestInferProjectId:
    def test_infers_known_slug(self):
        assert _infer_project_id("The neuralscape project uses FastAPI") == "neuralscape"
        assert _infer_project_id("Lightpath uses Three.js") == "lightpath"
        assert _infer_project_id("OpenClaw agent framework") == "openclaw"
        assert _infer_project_id("svc-utility-belt deploys on GKE") == "svc-utility-belt"

    def test_returns_none_for_unknown(self):
        assert _infer_project_id("User prefers dark mode") is None
        assert _infer_project_id("Some random project") is None

    def test_case_insensitive(self):
        assert _infer_project_id("NEURALSCAPE uses Qdrant") == "neuralscape"

    def test_batch_store_infers_project_id(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        facts = [("tech_stack", "Neuralscape uses FastAPI with Qdrant for vector search")]
        results = service._batch_store_facts(facts=facts, user_id="ehfaz")

        assert results[0].scope == "project"
        assert results[0].project_id == "neuralscape"


# ──────────────────────────────────────────────
# Graph episode deletion tests
# ──────────────────────────────────────────────


class TestDeleteEpisode:
    def test_delete_episode_calls_cypher(self, service):
        service._bridge.run = MagicMock(return_value=None)
        # Mock run_coroutine_threadsafe + future
        import asyncio
        import concurrent.futures

        mock_future = MagicMock()
        mock_future.result.return_value = None
        with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future):
            result = service.delete_episode("some-uuid")

        assert result["message"] == "Episode some-uuid deleted"

    def test_delete_episode_passes_uuid_as_kwarg(self, service):
        """Regression: uuid must be a direct kwarg, not inside parameters_.

        The graphiti driver wrapper already passes parameters_= to the
        Neo4j driver, so passing parameters_={"uuid": ...} from the
        caller causes 'multiple values for keyword argument parameters_'.
        """
        import asyncio

        mock_future = MagicMock()
        mock_future.result.return_value = None
        with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future) as mock_rcts:
            service.delete_episode("test-uuid-123")

        # execute_query should have been called with uuid as a direct kwarg
        service._graphiti.driver.execute_query.assert_called_once_with(
            "MATCH (e:Episodic {uuid: $uuid}) DETACH DELETE e",
            uuid="test-uuid-123",
        )

    def test_delete_episode_handles_error(self, service):
        import asyncio

        mock_future = MagicMock()
        mock_future.result.side_effect = Exception("Neo4j down")
        with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future):
            result = service.delete_episode("bad-uuid")

        assert "error" in result


class TestDeleteJunkEpisodes:
    def _mock_episodes_by_project(self, user_id=None, project_id=None, limit=500):
        """Return test episodes keyed by project_id (None = global)."""
        data = {
            None: [
                {"uuid": "g-1", "content": "Ehfaz prefers dark mode", "group_id": "global"},
                {"uuid": "g-2", "content": "assistant: Got it, I'll fix that bug now.", "group_id": "global"},
                {"uuid": "g-3", "content": "Ran command: git status", "group_id": "global"},
            ],
            "svc-utility-belt": [
                {"uuid": "su-1", "content": "assistant: Sure, deploying now.", "group_id": "project--svc-utility-belt"},
                {"uuid": "su-2", "content": "Uses FastAPI for microservices", "group_id": "project--svc-utility-belt"},
            ],
            "lightpath": [],
            "neuralscape": [
                {"uuid": "ns-1", "content": "Wrote file: main.py", "group_id": "project--neuralscape"},
                {"uuid": "ns-2", "content": "Neo4j is the graph backend", "group_id": "project--neuralscape"},
            ],
            "openclaw": [
                {"uuid": "oc-1", "content": "Tool result: success", "group_id": "project--openclaw"},
            ],
        }
        return data.get(project_id, [])

    def test_dry_run_single_project(self, service):
        """dry_run with explicit project_id only scans that group."""
        service.get_graph_episodes = MagicMock(return_value=[
            {"uuid": "ep-1", "content": "Ehfaz prefers dark mode", "group_id": "global"},
            {"uuid": "ep-2", "content": "assistant: Got it, fixing.", "group_id": "global"},
            {"uuid": "ep-3", "content": "Ran command: git status", "group_id": "global"},
            {"uuid": "ep-4", "content": "User uses Python 3.12", "group_id": "global"},
        ])

        result = service.delete_junk_episodes(user_id="ehfaz", project_id="neuralscape", dry_run=True)

        assert result["dry_run"] is True
        assert result["junk_count"] == 2
        assert "breakdown" in result
        assert "neuralscape" in result["breakdown"]
        assert len(result["breakdown"]) == 1
        service.get_graph_episodes.assert_called_once_with(user_id="ehfaz", project_id="neuralscape", limit=500)

    def test_dry_run_all_groups(self, service):
        """dry_run without project_id scans all known groups."""
        service.get_graph_episodes = MagicMock(side_effect=self._mock_episodes_by_project)

        result = service.delete_junk_episodes(user_id="ehfaz", dry_run=True)

        assert result["dry_run"] is True
        # g-2, g-3 (global) + su-1 (svc-utility-belt) + ns-1 (neuralscape) + oc-1 (openclaw) = 5
        assert result["junk_count"] == 5
        assert "breakdown" in result
        assert result["breakdown"]["global"]["junk_count"] == 2
        assert result["breakdown"]["svc-utility-belt"]["junk_count"] == 1
        assert result["breakdown"]["lightpath"]["junk_count"] == 0
        assert result["breakdown"]["neuralscape"]["junk_count"] == 1
        assert result["breakdown"]["openclaw"]["junk_count"] == 1
        # Should have called get_graph_episodes 5 times (global + 4 projects)
        assert service.get_graph_episodes.call_count == 5

    def test_delete_all_groups(self, service):
        """Actual delete without project_id cleans all groups."""
        service.get_graph_episodes = MagicMock(side_effect=self._mock_episodes_by_project)
        service.delete_episode = MagicMock(return_value={"message": "deleted"})

        result = service.delete_junk_episodes(user_id="ehfaz", dry_run=False)

        assert "dry_run" not in result
        assert result["deleted_count"] == 5
        assert "breakdown" in result
        assert result["breakdown"]["global"]["deleted_count"] == 2
        assert result["breakdown"]["svc-utility-belt"]["deleted_count"] == 1
        assert result["breakdown"]["neuralscape"]["deleted_count"] == 1
        assert result["breakdown"]["openclaw"]["deleted_count"] == 1
        # Verify delete_episode was called for each junk episode
        deleted_uuids = [call.args[0] for call in service.delete_episode.call_args_list]
        assert "g-2" in deleted_uuids
        assert "g-3" in deleted_uuids
        assert "su-1" in deleted_uuids
        assert "ns-1" in deleted_uuids
        assert "oc-1" in deleted_uuids
        # Non-junk should NOT be deleted
        assert "g-1" not in deleted_uuids
        assert "su-2" not in deleted_uuids
        assert "ns-2" not in deleted_uuids

    def test_delete_single_project(self, service):
        """Actual delete with explicit project_id only cleans that group."""
        service.get_graph_episodes = MagicMock(return_value=[
            {"uuid": "ns-1", "content": "Wrote file: main.py", "group_id": "project--neuralscape"},
            {"uuid": "ns-2", "content": "Neo4j is the graph backend", "group_id": "project--neuralscape"},
        ])
        service.delete_episode = MagicMock(return_value={"message": "deleted"})

        result = service.delete_junk_episodes(user_id="ehfaz", project_id="neuralscape", dry_run=False)

        assert result["deleted_count"] == 1
        assert len(result["breakdown"]) == 1
        assert result["breakdown"]["neuralscape"]["deleted_count"] == 1
        service.delete_episode.assert_called_once_with("ns-1")

    def test_delete_handles_partial_failures(self, service):
        """If some deletes fail, only successful ones are counted."""
        service.get_graph_episodes = MagicMock(return_value=[
            {"uuid": "ep-1", "content": "assistant: hello", "group_id": "global"},
            {"uuid": "ep-2", "content": "Ran command: ls", "group_id": "global"},
        ])
        service.delete_episode = MagicMock(side_effect=[
            {"message": "deleted"},
            {"error": "Neo4j timeout"},
        ])

        result = service.delete_junk_episodes(user_id="ehfaz", project_id="global-only", dry_run=False)

        assert result["deleted_count"] == 1
        assert len(result["breakdown"]["global-only"]["deleted_uuids"]) == 1
