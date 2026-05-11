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
    _parse_expires_at,
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
        assert call_kwargs["filters"]["metadata.category"] == {"in": ["preference", "convention"]}


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
        assert call_kwargs["filters"]["metadata.scope"] == "global"
        assert call_kwargs["filters"]["metadata.category"] == "preference"
        assert call_kwargs["filters"]["metadata.project_id"] == "my-project"

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


# ──────────────────────────────────────────────
# Response shaping (mem0 metadata unwrap)
# ──────────────────────────────────────────────


class TestMemToResponse:
    """Regression coverage for the mem0 metadata-double-wrap fix.

    mem0's `_search_vector_store` and `_get_all_from_vector_store` lift every
    payload field that isn't on a hardcoded promoted-keys list into a
    top-level `metadata` dict. Because our Qdrant payload nests our domain
    fields under a literal `metadata` key, the result that reaches
    `_mem_to_response` is shaped like
    `{"metadata": {"metadata": {"scope": ..., "category": ...}}}`.

    Without the unwrap, `metadata.get("category")` resolves to None and
    every search/list response loses category, scope, project_id, and tags.
    """

    def test_unwraps_double_nested_metadata(self, service):
        """The shape mem0 actually produces — must unwrap once."""
        mem = {
            "id": "abc-123",
            "memory": "Prefers TypeScript over JavaScript",
            "score": 0.85,
            "created_at": "2026-05-08T19:38:47Z",
            "updated_at": None,
            "metadata": {
                "metadata": {
                    "scope": "global",
                    "category": "preference",
                    "project_id": None,
                    "tags": ["editor"],
                    "source": "explicit",
                }
            },
        }

        resp = service._mem_to_response(mem)

        assert resp.id == "abc-123"
        assert resp.memory == "Prefers TypeScript over JavaScript"
        assert resp.category == "preference"
        assert resp.scope == "global"
        assert resp.project_id is None
        assert resp.tags == ["editor"]
        assert resp.score == 0.85
        assert resp.created_at == "2026-05-08T19:38:47Z"
        assert resp.source == "vector"

    def test_handles_flat_metadata(self, service):
        """Defensive: if mem0 ever flattens, our code must still resolve."""
        mem = {
            "id": "def-456",
            "memory": "Uses FastAPI",
            "score": 0.72,
            "metadata": {
                "scope": "project",
                "category": "tech_stack",
                "project_id": "neuralscape",
                "tags": None,
            },
        }

        resp = service._mem_to_response(mem)

        assert resp.category == "tech_stack"
        assert resp.scope == "project"
        assert resp.project_id == "neuralscape"

    def test_missing_metadata_returns_null_fields(self, service):
        """No metadata key at all — fields stay None, response still valid."""
        mem = {"id": "ghi-789", "memory": "Bare memory", "score": 0.5}

        resp = service._mem_to_response(mem)

        assert resp.id == "ghi-789"
        assert resp.category is None
        assert resp.scope is None
        assert resp.project_id is None
        assert resp.tags is None
        assert resp.source == "vector"

    def test_empty_metadata_returns_null_fields(self, service):
        """Metadata is an empty dict — same behavior as missing."""
        mem = {"id": "jkl-012", "memory": "Empty md", "metadata": {}}

        resp = service._mem_to_response(mem)

        assert resp.category is None
        assert resp.scope is None

    def test_inner_metadata_dict_takes_precedence_over_outer(self, service):
        """When both layers present, the inner (real) one wins."""
        mem = {
            "id": "mno-345",
            "memory": "Layered",
            "metadata": {
                "category": "should-be-ignored",  # outer wrapper level
                "metadata": {
                    "category": "preference",  # real, inner level
                    "scope": "global",
                },
            },
        }

        resp = service._mem_to_response(mem)

        assert resp.category == "preference"
        assert resp.scope == "global"


# ──────────────────────────────────────────────
# Memory model v2: store_raw v2 fields
# ──────────────────────────────────────────────


class TestStoreRawV2:
    """Memory-model v2 — store_raw accepts and persists the new optional fields."""

    def test_v2_fields_persisted_to_metadata(self, service):
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        # _find_by_content_hash uses client.scroll; mock empty result to avoid dedup hit
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)

        from datetime import datetime, timezone

        result = service.store_raw(
            content="Adopted feature flags via GrowthBook for the checkout flow",
            user_id="ehfaz",
            category="decision",
            domain="coding",
            observation_type="decision",
            concepts=["why-it-exists", "trade-off"],
            source_type="tool_extraction",
            related_memory_ids=["mem-1", "mem-2"],
            confidence=0.85,
            expires_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
        )

        assert len(result) == 1
        resp = result[0]
        assert resp.domain == "coding"
        assert resp.observation_type == "decision"
        assert resp.concepts == ["why-it-exists", "trade-off"]
        assert resp.source_type == "tool_extraction"
        assert resp.related_memory_ids == ["mem-1", "mem-2"]
        assert resp.confidence == 0.85
        assert resp.expires_at is not None

        # Payload metadata should reflect every v2 field
        insert_kwargs = service._memory.vector_store.insert.call_args[1]
        metadata = insert_kwargs["payloads"][0]["metadata"]
        assert metadata["domain"] == "coding"
        assert metadata["observation_type"] == "decision"
        assert metadata["concepts"] == ["why-it-exists", "trade-off"]
        assert metadata["source_type"] == "tool_extraction"
        assert metadata["related_memory_ids"] == ["mem-1", "mem-2"]
        assert metadata["confidence"] == 0.85
        assert "expires_at" in metadata

    def test_v2_fields_optional(self, service):
        """Calling store_raw with no v2 fields produces a v1-compatible memory."""
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)

        result = service.store_raw(
            content="Prefers tabs over spaces",
            user_id="ehfaz",
            category="preference",
        )
        assert len(result) == 1
        # All v2 fields stay null when not supplied
        assert result[0].domain is None
        assert result[0].observation_type is None
        assert result[0].concepts is None
        assert result[0].confidence is None

        # Metadata should not contain any v2 keys when fields weren't supplied
        insert_kwargs = service._memory.vector_store.insert.call_args[1]
        metadata = insert_kwargs["payloads"][0]["metadata"]
        for v2_key in ("domain", "observation_type", "concepts", "source_type",
                       "related_memory_ids", "confidence", "expires_at"):
            assert v2_key not in metadata, f"Unexpected v2 key '{v2_key}' in v1-style payload"


# ──────────────────────────────────────────────
# Memory model v2: content-hash dedup
# ──────────────────────────────────────────────


class TestContentHashDedup:
    """Memory-model v2 — store_raw is idempotent via content-hash dedup."""

    def test_dedup_hit_returns_existing(self, service):
        """When the same (user_id, scope, hash) is found, return existing without insert."""
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client

        # Configure scroll to return one matching point
        existing_point = MagicMock()
        existing_point.id = "existing-id"
        existing_point.payload = {
            "data": "Prefers tabs over spaces",
            "created_at": "2026-01-01T00:00:00Z",
            "metadata": {
                "scope": "global",
                "category": "preference",
                "project_id": None,
                "domain": "coding",
            },
        }
        mock_client.scroll.return_value = ([existing_point], None)

        result = service.store_raw(
            content="Prefers tabs over spaces",
            user_id="ehfaz",
            category="preference",
        )

        # Should NOT have called insert (dedup hit)
        service._memory.vector_store.insert.assert_not_called()
        assert len(result) == 1
        assert result[0].id == "existing-id"
        assert result[0].domain == "coding"

    def test_dedup_miss_inserts(self, service):
        """When no matching hash found, proceed with normal insert."""
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        mock_client.scroll.return_value = ([], None)

        result = service.store_raw(
            content="A novel fact never seen before",
            user_id="ehfaz",
            category="personal_fact",
        )

        # Should have called insert (dedup miss)
        service._memory.vector_store.insert.assert_called_once()
        assert len(result) == 1
        assert result[0].id != "existing-id"

    def test_dedup_lookup_failure_does_not_block_insert(self, service):
        """Defensive: if the dedup lookup raises, store_raw should still insert."""
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        mock_client.scroll.side_effect = Exception("Qdrant transient error")

        result = service.store_raw(
            content="A fact",
            user_id="ehfaz",
            category="personal_fact",
        )

        # Insert proceeds even when dedup query fails — safer to risk a dup.
        service._memory.vector_store.insert.assert_called_once()
        assert len(result) == 1


# ──────────────────────────────────────────────
# Memory model v2: store_raw_batch
# ──────────────────────────────────────────────


class TestStoreRawBatch:
    """Memory-model v2 — batch storage of pre-categorized facts."""

    def test_stores_each_item(self, service):
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)

        items = [
            {"content": "Fact one", "user_id": "ehfaz", "category": "personal_fact",
             "domain": "personal"},
            {"content": "Fact two", "user_id": "ehfaz", "category": "preference",
             "concepts": ["how-it-works"]},
        ]
        results = service.store_raw_batch(items)

        assert len(results) == 2
        assert results[0].domain == "personal"
        assert results[1].concepts == ["how-it-works"]
        # Two inserts, one per item
        assert service._memory.vector_store.insert.call_count == 2

    def test_continues_on_per_item_error(self, service):
        """A bad item must not block the rest of the batch."""
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)

        items = [
            {"content": "Good", "user_id": "ehfaz", "category": "preference"},
            {"content": "Bad", "user_id": "ehfaz", "category": "INVALID-CATEGORY"},
            {"content": "Also good", "user_id": "ehfaz", "category": "personal_fact"},
        ]
        results = service.store_raw_batch(items)

        # Two stored, one skipped due to bad category
        assert len(results) == 2

    def test_handles_iso_string_expires_at(self, service):
        """expires_at can arrive as ISO string after JSON enqueue — should round-trip."""
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)

        items = [
            {"content": "Time-bound fact", "user_id": "ehfaz", "category": "task_context",
             "expires_at": "2026-12-01T00:00:00+00:00"},
        ]
        results = service.store_raw_batch(items)

        assert len(results) == 1
        assert results[0].expires_at is not None


# ──────────────────────────────────────────────
# Memory model v2: search filters
# ──────────────────────────────────────────────


class TestSearchV2Filters:
    """Memory-model v2 — search honors domain/observation_type/concepts filters."""

    def test_domain_filter_applied(self, service):
        service._memory.search.return_value = {"results": []}
        service.search(
            query="anything",
            user_id="ehfaz",
            scope="global",  # avoid the dual-scope merge path
            domain="research",
        )
        call_kwargs = service._memory.search.call_args[1]
        assert call_kwargs["filters"]["metadata.domain"] == "research"

    def test_observation_type_filter_applied(self, service):
        service._memory.search.return_value = {"results": []}
        service.search(
            query="anything",
            user_id="ehfaz",
            scope="global",
            observation_type="bugfix",
        )
        call_kwargs = service._memory.search.call_args[1]
        assert call_kwargs["filters"]["metadata.observation_type"] == "bugfix"

    def test_concepts_filter_applied_as_in(self, service):
        service._memory.search.return_value = {"results": []}
        service.search(
            query="anything",
            user_id="ehfaz",
            scope="global",
            concepts=["gotcha", "trade-off"],
        )
        call_kwargs = service._memory.search.call_args[1]
        assert call_kwargs["filters"]["metadata.concepts"] == {"in": ["gotcha", "trade-off"]}


# ──────────────────────────────────────────────
# Memory model v2: _mem_to_response surfaces new fields
# ──────────────────────────────────────────────


class TestMemToResponseV2:
    def test_surfaces_v2_fields(self, service):
        mem = {
            "id": "v2-001",
            "memory": "A v2 memory",
            "metadata": {
                "metadata": {
                    "category": "decision",
                    "scope": "project",
                    "project_id": "neuralscape",
                    "domain": "coding",
                    "observation_type": "decision",
                    "concepts": ["why-it-exists", "trade-off"],
                    "source_type": "tool_extraction",
                    "related_memory_ids": ["mem-1"],
                    "confidence": 0.9,
                    "expires_at": "2026-12-01T00:00:00+00:00",
                }
            },
        }
        resp = service._mem_to_response(mem)
        assert resp.domain == "coding"
        assert resp.observation_type == "decision"
        assert resp.concepts == ["why-it-exists", "trade-off"]
        assert resp.source_type == "tool_extraction"
        assert resp.related_memory_ids == ["mem-1"]
        assert resp.confidence == 0.9
        assert resp.expires_at == "2026-12-01T00:00:00+00:00"

    def test_legacy_memory_has_null_v2_fields(self, service):
        """A v1-era memory without v2 metadata renders v2 fields as None."""
        mem = {
            "id": "v1-001",
            "memory": "Legacy memory",
            "metadata": {"metadata": {"category": "preference", "scope": "global"}},
        }
        resp = service._mem_to_response(mem)
        assert resp.category == "preference"
        assert resp.domain is None
        assert resp.observation_type is None
        assert resp.concepts is None
        assert resp.confidence is None


# ──────────────────────────────────────────────
# Memory model v2: schema validation
# ──────────────────────────────────────────────


class TestSchemaV2Validators:
    def test_invalid_domain_rejected(self):
        from pydantic import ValidationError
        from schemas import RawMemoryRequest

        with pytest.raises(ValidationError):
            RawMemoryRequest(
                content="x", user_id="u", category="preference", domain="not-a-domain"
            )

    def test_invalid_observation_type_rejected(self):
        from pydantic import ValidationError
        from schemas import RawMemoryRequest

        with pytest.raises(ValidationError):
            RawMemoryRequest(
                content="x", user_id="u", category="preference", observation_type="bogus"
            )

    def test_unknown_concept_rejected(self):
        from pydantic import ValidationError
        from schemas import RawMemoryRequest

        with pytest.raises(ValidationError):
            RawMemoryRequest(
                content="x", user_id="u", category="preference",
                concepts=["how-it-works", "definitely-not-a-concept"],
            )

    def test_concepts_capped_at_5(self):
        from pydantic import ValidationError
        from schemas import RawMemoryRequest

        with pytest.raises(ValidationError):
            RawMemoryRequest(
                content="x", user_id="u", category="preference",
                concepts=["how-it-works", "why-it-exists", "what-changed",
                          "problem-solution", "gotcha", "pattern"],  # 6 > 5
            )

    def test_confidence_range_enforced(self):
        from pydantic import ValidationError
        from schemas import RawMemoryRequest

        with pytest.raises(ValidationError):
            RawMemoryRequest(
                content="x", user_id="u", category="preference", confidence=1.5
            )

    def test_valid_v2_memory_passes(self):
        from datetime import datetime, timezone
        from schemas import RawMemoryRequest

        req = RawMemoryRequest(
            content="x",
            user_id="u",
            category="decision",
            scope="project",
            project_id="proj1",
            domain="coding",
            observation_type="decision",
            concepts=["why-it-exists", "trade-off"],
            source_type="tool_extraction",
            confidence=0.7,
            expires_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
        )
        assert req.domain == "coding"
        assert req.observation_type == "decision"
        assert req.confidence == 0.7

    def test_batch_request_caps_at_50(self):
        from pydantic import ValidationError
        from schemas import RawMemoryBatchRequest, RawMemoryRequest

        small = RawMemoryRequest(content="x", user_id="u", category="preference")
        with pytest.raises(ValidationError):
            RawMemoryBatchRequest(memories=[small] * 51)

    def test_batch_request_min_one(self):
        from pydantic import ValidationError
        from schemas import RawMemoryBatchRequest

        with pytest.raises(ValidationError):
            RawMemoryBatchRequest(memories=[])

    def test_raw_invalid_source_type(self):
        from pydantic import ValidationError
        from schemas import RawMemoryRequest

        with pytest.raises(ValidationError):
            RawMemoryRequest(
                content="x", user_id="u", category="preference", source_type="bogus"
            )

    def test_raw_concepts_none_passes(self):
        """concepts=None must short-circuit validation, not iterate."""
        from schemas import RawMemoryRequest

        req = RawMemoryRequest(content="x", user_id="u", category="preference", concepts=None)
        assert req.concepts is None

    def test_search_invalid_domain(self):
        from pydantic import ValidationError
        from schemas import SearchMemoryRequest

        with pytest.raises(ValidationError):
            SearchMemoryRequest(query="hi", user_id="u", domain="not-a-domain")

    def test_search_invalid_observation_type(self):
        from pydantic import ValidationError
        from schemas import SearchMemoryRequest

        with pytest.raises(ValidationError):
            SearchMemoryRequest(query="hi", user_id="u", observation_type="bogus")

    def test_search_unknown_concept_rejected(self):
        """Mirror RawMemoryRequest so typos surface as 422, not silent misses.
        Regression for CR-12."""
        from pydantic import ValidationError
        from schemas import SearchMemoryRequest

        with pytest.raises(ValidationError):
            SearchMemoryRequest(query="hi", user_id="u", concepts=["definitely-not-a-concept"])

    def test_search_known_concept_passes(self):
        from schemas import SearchMemoryRequest

        req = SearchMemoryRequest(query="hi", user_id="u", concepts=["gotcha", "trade-off"])
        assert req.concepts == ["gotcha", "trade-off"]

    def test_search_concepts_capped_at_5(self):
        from pydantic import ValidationError
        from schemas import SearchMemoryRequest

        with pytest.raises(ValidationError):
            SearchMemoryRequest(
                query="hi", user_id="u",
                concepts=["how-it-works", "why-it-exists", "what-changed",
                          "problem-solution", "gotcha", "pattern"],  # 6 > 5
            )

    def test_search_concepts_none_allowed(self):
        from schemas import SearchMemoryRequest

        req = SearchMemoryRequest(query="hi", user_id="u", concepts=None)
        assert req.concepts is None

    def test_store_request_invalid_domain(self):
        from pydantic import ValidationError
        from schemas import StoreMemoryRequest

        with pytest.raises(ValidationError):
            StoreMemoryRequest(
                messages=[{"role": "user", "content": "x"}],
                user_id="u",
                domain="not-a-domain",
            )

    def test_store_request_valid_domain(self):
        from schemas import StoreMemoryRequest

        req = StoreMemoryRequest(
            messages=[{"role": "user", "content": "x"}],
            user_id="u",
            domain="research",
        )
        assert req.domain == "research"


# ──────────────────────────────────────────────
# Memory model v2: _find_by_content_hash project-scope branch
# ──────────────────────────────────────────────


class TestFindByContentHashProjectScope:
    """Memory-model v2 — _find_by_content_hash adds project_id filter when scope='project'."""

    def test_project_scope_appends_project_filter(self, service):
        """When scope='project' AND project_id supplied, the Qdrant filter must include project_id."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        mock_client.scroll.return_value = ([], None)

        service._find_by_content_hash(
            user_id="ehfaz",
            content_hash="abc123",
            scope="project",
            project_id="neuralscape",
        )

        mock_client.scroll.assert_called_once()
        call_kwargs = mock_client.scroll.call_args[1]
        scroll_filter = call_kwargs["scroll_filter"]
        # 4 conditions when project scope: user_id, hash, scope, project_id
        assert len(scroll_filter.must) == 4

    def test_project_scope_without_project_id_skips_filter(self, service):
        """scope='project' but project_id=None: no project_id filter appended."""
        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        mock_client.scroll.return_value = ([], None)

        service._find_by_content_hash(
            user_id="ehfaz",
            content_hash="abc123",
            scope="project",
            project_id=None,
        )

        scroll_filter = mock_client.scroll.call_args[1]["scroll_filter"]
        # 3 conditions when no project_id: user_id, hash, scope
        assert len(scroll_filter.must) == 3


# ──────────────────────────────────────────────
# Memory model v2: expire_old_memories
# ──────────────────────────────────────────────


class TestExpireOldMemories:
    """Memory-model v2 — nightly purge of memories with expired expires_at."""

    def _make_point(self, pt_id, expires_at, user_id="ehfaz"):
        pt = MagicMock()
        pt.id = pt_id
        pt.payload = {
            "data": f"memory {pt_id}",
            "user_id": user_id,
            "metadata": {
                "scope": "global",
                "category": "task_context",
                "expires_at": expires_at,
            },
        }
        return pt

    def test_deletes_expired_skips_future_and_null(self, service):
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()

        expired = self._make_point("expired-1", past, user_id="alice")
        future_pt = self._make_point("future-1", future, user_id="alice")
        no_expiry = MagicMock()
        no_expiry.id = "no-expiry-1"
        no_expiry.payload = {
            "data": "no expiry", "user_id": "alice",
            "metadata": {"scope": "global"},  # no expires_at
        }

        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        # Single page with all three; second call returns empty to terminate
        mock_client.scroll.side_effect = [
            ([expired, future_pt, no_expiry], None),  # next_offset=None terminates
        ]
        # Mock the delete + graph cleanup helper
        service._delete_qdrant_memory_with_graph_cleanup = MagicMock()

        result = service.expire_old_memories(batch_size=100)

        assert result["deleted_count"] == 1
        assert result["per_user"] == {"alice": 1}
        # Only the expired one was deleted
        service._delete_qdrant_memory_with_graph_cleanup.assert_called_once()
        deleted_id = service._delete_qdrant_memory_with_graph_cleanup.call_args[0][0]
        assert deleted_id == "expired-1"

    def test_handles_per_point_delete_failure(self, service):
        """A failed delete on one point doesn't abort the run."""
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        p1 = self._make_point("p1", past, user_id="bob")
        p2 = self._make_point("p2", past, user_id="bob")

        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        mock_client.scroll.side_effect = [([p1, p2], None)]

        # First delete fails, second succeeds
        service._delete_qdrant_memory_with_graph_cleanup = MagicMock(
            side_effect=[Exception("Qdrant transient"), None]
        )

        result = service.expire_old_memories(batch_size=100)
        assert result["deleted_count"] == 1
        assert result["per_user"] == {"bob": 1}

    def test_paginates_through_multiple_pages(self, service):
        """When Qdrant returns next_offset, the cron continues to the next page."""
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        p1 = self._make_point("p1", past, user_id="alice")
        p2 = self._make_point("p2", past, user_id="bob")

        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        # First page has p1 with next_offset='cursor', second page has p2 with None
        mock_client.scroll.side_effect = [
            ([p1], "cursor"),
            ([p2], None),
        ]
        service._delete_qdrant_memory_with_graph_cleanup = MagicMock()

        result = service.expire_old_memories(batch_size=1)
        assert result["deleted_count"] == 2
        assert result["per_user"] == {"alice": 1, "bob": 1}
        # Two scroll calls = paginated
        assert mock_client.scroll.call_count == 2

    def test_empty_collection(self, service):
        """No points returned: terminates cleanly with zero deletions."""
        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        mock_client.scroll.return_value = ([], None)

        result = service.expire_old_memories()
        assert result["deleted_count"] == 0
        assert result["per_user"] == {}

    def test_skips_unparseable_expires_at(self, service):
        """A memory whose expires_at is malformed must be skipped, not deleted."""
        pt = MagicMock()
        pt.id = "bad-1"
        pt.payload = {
            "data": "x", "user_id": "alice",
            "metadata": {"expires_at": "not-a-timestamp"},
        }
        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        mock_client.scroll.side_effect = [([pt], None)]
        service._delete_qdrant_memory_with_graph_cleanup = MagicMock()

        result = service.expire_old_memories()
        assert result["deleted_count"] == 0
        service._delete_qdrant_memory_with_graph_cleanup.assert_not_called()

    def test_cold_start_initializes_memory(self, service):
        """expire_old_memories must call _get_memory() before touching client.

        Regression for the cold-start AttributeError CodeRabbit flagged: the
        cron can fire on a worker that hasn't served any request yet.
        """
        # Pretend memory hasn't been initialized — _get_memory should be invoked
        with patch.object(service, "_get_memory", return_value=service._memory) as mock_get:
            mock_client = MagicMock()
            service._memory.vector_store.client = mock_client
            mock_client.scroll.return_value = ([], None)
            service.expire_old_memories()
        mock_get.assert_called_once()


class TestParseExpiresAt:
    """Memory-model v2 — robust ISO-8601 parsing used by the expiry cron."""

    def test_parses_z_suffix_as_utc(self):
        from datetime import timezone
        dt = _parse_expires_at("2026-12-01T00:00:00Z")
        assert dt is not None
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2026

    def test_parses_offset(self):
        dt = _parse_expires_at("2026-12-01T00:00:00-05:00")
        assert dt is not None
        # Should be tz-aware regardless of offset
        assert dt.tzinfo is not None

    def test_naive_string_treated_as_utc(self):
        from datetime import timezone
        dt = _parse_expires_at("2026-12-01T00:00:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_returns_none_for_malformed(self):
        assert _parse_expires_at("not-a-date") is None
        assert _parse_expires_at("") is None
        assert _parse_expires_at("   ") is None

    def test_returns_none_for_none(self):
        assert _parse_expires_at(None) is None

    def test_returns_none_for_non_string_non_datetime(self):
        assert _parse_expires_at(42) is None
        assert _parse_expires_at(["x"]) is None

    def test_datetime_input_passes_through(self):
        from datetime import datetime, timezone
        aware = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert _parse_expires_at(aware) is aware

    def test_naive_datetime_treated_as_utc(self):
        from datetime import datetime, timezone
        naive = datetime(2026, 1, 1)
        dt = _parse_expires_at(naive)
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_mixed_offset_comparison_ordering(self):
        """Strings that sort wrong lexicographically still compare correctly
        once parsed — regression for CR-10 / CP-02.
        """
        # "Z" sorts before "-" but Z is UTC noon, -05:00 is later actual time.
        earlier = _parse_expires_at("2026-01-01T12:00:00Z")
        later = _parse_expires_at("2026-01-01T08:00:00-05:00")  # = 13:00 UTC
        assert earlier is not None and later is not None
        assert earlier < later


# ──────────────────────────────────────────────
# Memory model v2: graph result enrichment + filtering
# ──────────────────────────────────────────────


class TestGraphEnrichment:
    """Memory-model v2 — _enrich_graph_with_v2 and _enrich_and_filter_graph.

    Graphiti edges don't carry v2 fields natively; we recover them by top-1
    semantic search against Qdrant, gated by a similarity threshold so we
    never propagate metadata from an unrelated nearest neighbor.
    """

    def _hit(self, score: float, metadata: dict, data: str = "x"):
        """Build a fake qdrant ScoredPoint-like object."""
        h = MagicMock()
        h.score = score
        h.payload = {"data": data, "metadata": metadata}
        return h

    def test_high_similarity_match_copies_v2_fields(self, service):
        from schemas import MemoryResponse
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.search.return_value = [
            self._hit(0.95, {
                "category": "decision", "scope": "global",
                "domain": "meeting", "observation_type": "meeting_outcome",
                "concepts": ["blocker"], "source_type": "tool_extraction",
                "confidence": 0.8,
            })
        ]
        graph_responses = [MemoryResponse(id="g1", memory="OKR was shifted", source="graph")]
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        assert result[0].domain == "meeting"
        assert result[0].observation_type == "meeting_outcome"
        assert result[0].concepts == ["blocker"]
        assert result[0].source_type == "tool_extraction"
        assert result[0].confidence == 0.8
        assert result[0].category == "decision"
        assert result[0].scope == "global"

    def test_below_threshold_does_not_enrich(self, service):
        from schemas import MemoryResponse
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        # Score 0.5 is below default 0.7 threshold
        service._memory.vector_store.search.return_value = [
            self._hit(0.5, {"domain": "coding", "observation_type": "decision"})
        ]
        graph_responses = [MemoryResponse(id="g1", memory="unrelated graph fact", source="graph")]
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        # Below threshold → fields stay None
        assert result[0].domain is None
        assert result[0].observation_type is None

    def test_no_hits_skips_enrichment(self, service):
        from schemas import MemoryResponse
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.search.return_value = []
        graph_responses = [MemoryResponse(id="g1", memory="lonely graph fact", source="graph")]
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        assert result[0].domain is None

    def test_does_not_overwrite_existing_v2_fields(self, service):
        from schemas import MemoryResponse
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.search.return_value = [
            self._hit(0.95, {"domain": "research"})
        ]
        graph_responses = [
            MemoryResponse(id="g1", memory="x", source="graph", domain="coding"),
        ]
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        # Existing domain="coding" preserved, not overwritten by enrichment "research"
        assert result[0].domain == "coding"

    def test_handles_double_wrapped_metadata(self, service):
        """mem0 sometimes nests metadata under metadata.metadata — unwrap it."""
        from schemas import MemoryResponse
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.search.return_value = [
            self._hit(0.95, {"metadata": {"domain": "ops", "observation_type": "feature"}})
        ]
        graph_responses = [MemoryResponse(id="g1", memory="x", source="graph")]
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        assert result[0].domain == "ops"
        assert result[0].observation_type == "feature"

    def test_skips_empty_memory_text(self, service):
        from schemas import MemoryResponse
        service._memory.vector_store.search.return_value = []
        graph_responses = [MemoryResponse(id="g1", memory="", source="graph")]
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        # Empty memory: never hits the search, fields stay None
        assert result[0].domain is None
        service._memory.embedding_model.embed.assert_not_called()

    def test_swallows_per_row_errors(self, service):
        """A failure on one row doesn't abort the whole enrichment pass."""
        from schemas import MemoryResponse
        service._memory.embedding_model.embed.side_effect = Exception("embed fail")
        graph_responses = [MemoryResponse(id="g1", memory="x", source="graph")]
        # Should not raise
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        assert result[0].domain is None

    def test_dict_hit_format_supported(self, service):
        """Some Qdrant client versions return dicts instead of ScoredPoint."""
        from schemas import MemoryResponse
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.search.return_value = [
            {"score": 0.9, "payload": {"metadata": {"domain": "writing"}}}
        ]
        graph_responses = [MemoryResponse(id="g1", memory="x", source="graph")]
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        assert result[0].domain == "writing"

    def test_project_scope_added_to_lookup_filter(self, service):
        """When project_id is supplied, the enrichment lookup must constrain
        to that project so a graph edge can't inherit metadata from a
        semantically similar memory in another project — regression for
        CR-11 / CP-05.
        """
        from schemas import MemoryResponse
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.search.return_value = []
        graph_responses = [MemoryResponse(id="g1", memory="x", source="graph")]
        service._enrich_graph_with_v2(
            graph_responses, user_id="ehfaz", project_id="neuralscape",
        )
        # Inspect the filter the lookup actually used
        call_kwargs = service._memory.vector_store.search.call_args[1]
        filters = call_kwargs["filters"]
        assert filters["user_id"] == "ehfaz"
        assert filters["metadata.project_id"] == "neuralscape"

    def test_global_scope_uses_user_filter_only(self, service):
        """Without project_id, the lookup filter only has user_id."""
        from schemas import MemoryResponse
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.search.return_value = []
        graph_responses = [MemoryResponse(id="g1", memory="x", source="graph")]
        service._enrich_graph_with_v2(
            graph_responses, user_id="ehfaz", project_id=None,
        )
        filters = service._memory.vector_store.search.call_args[1]["filters"]
        assert filters == {"user_id": "ehfaz"}


class TestGraphFilterByV2:
    """Memory-model v2 — _enrich_and_filter_graph drops rows that don't match the filter."""

    def _hit(self, score: float, metadata: dict):
        h = MagicMock()
        h.score = score
        h.payload = {"data": "x", "metadata": metadata}
        return h

    def test_domain_filter_drops_non_match(self, service):
        from schemas import MemoryResponse
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        # Two graph rows, one source has domain=coding, the other meeting
        service._memory.vector_store.search.side_effect = [
            [self._hit(0.9, {"domain": "coding", "observation_type": "decision"})],
            [self._hit(0.9, {"domain": "meeting", "observation_type": "meeting_outcome"})],
        ]
        graph_responses = [
            MemoryResponse(id="g1", memory="fact1", source="graph"),
            MemoryResponse(id="g2", memory="fact2", source="graph"),
        ]
        result = service._enrich_and_filter_graph(
            graph_responses, user_id="u", project_id=None,
            domain="meeting", observation_type=None, concepts=None,
        )
        assert len(result) == 1
        assert result[0].id == "g2"

    def test_observation_type_filter_drops_non_match(self, service):
        from schemas import MemoryResponse
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.search.side_effect = [
            [self._hit(0.9, {"observation_type": "bugfix"})],
            [self._hit(0.9, {"observation_type": "feature"})],
        ]
        graph_responses = [
            MemoryResponse(id="g1", memory="fact1", source="graph"),
            MemoryResponse(id="g2", memory="fact2", source="graph"),
        ]
        result = service._enrich_and_filter_graph(
            graph_responses, user_id="u", project_id=None,
            domain=None, observation_type="bugfix", concepts=None,
        )
        assert len(result) == 1
        assert result[0].id == "g1"

    def test_concepts_filter_keeps_overlap(self, service):
        from schemas import MemoryResponse
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.search.side_effect = [
            [self._hit(0.9, {"concepts": ["gotcha", "pattern"]})],
            [self._hit(0.9, {"concepts": ["how-it-works"]})],
            [self._hit(0.9, {})],  # no concepts at all
        ]
        graph_responses = [
            MemoryResponse(id="g1", memory="a", source="graph"),
            MemoryResponse(id="g2", memory="b", source="graph"),
            MemoryResponse(id="g3", memory="c", source="graph"),
        ]
        result = service._enrich_and_filter_graph(
            graph_responses, user_id="u", project_id=None,
            domain=None, observation_type=None, concepts=["gotcha"],
        )
        # Only g1 has overlap with concepts=[gotcha]
        assert [r.id for r in result] == ["g1"]

    def test_below_threshold_falls_off_when_filtering(self, service):
        """Rows whose source match is below threshold get None'd, then filter drops them."""
        from schemas import MemoryResponse
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.search.side_effect = [
            [self._hit(0.95, {"domain": "coding"})],  # passes threshold
            [self._hit(0.4, {"domain": "coding"})],   # below threshold → not enriched
        ]
        graph_responses = [
            MemoryResponse(id="g1", memory="related", source="graph"),
            MemoryResponse(id="g2", memory="unrelated", source="graph"),
        ]
        result = service._enrich_and_filter_graph(
            graph_responses, user_id="u", project_id=None,
            domain="coding", observation_type=None, concepts=None,
        )
        assert [r.id for r in result] == ["g1"]

    def test_combined_filters_all_must_match(self, service):
        from schemas import MemoryResponse
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.search.side_effect = [
            [self._hit(0.9, {"domain": "coding", "observation_type": "decision",
                             "concepts": ["why-it-exists"]})],
            [self._hit(0.9, {"domain": "coding", "observation_type": "bugfix",
                             "concepts": ["why-it-exists"]})],
        ]
        graph_responses = [
            MemoryResponse(id="g1", memory="a", source="graph"),
            MemoryResponse(id="g2", memory="b", source="graph"),
        ]
        # Domain matches both; obs_type only matches g1
        result = service._enrich_and_filter_graph(
            graph_responses, user_id="u", project_id=None,
            domain="coding", observation_type="decision", concepts=["why-it-exists"],
        )
        assert [r.id for r in result] == ["g1"]

    def test_empty_graph_responses_returns_empty(self, service):
        result = service._enrich_and_filter_graph(
            [], user_id="u", project_id=None,
            domain="coding", observation_type=None, concepts=None,
        )
        assert result == []
