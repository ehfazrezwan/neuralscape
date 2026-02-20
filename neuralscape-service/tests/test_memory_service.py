"""Tests for MemoryService business logic."""

from unittest.mock import MagicMock, patch

import pytest

from memory_service import MemoryService, _build_group_id, _get_group_ids
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

    def test_extraction_falls_back_on_llm_error(self, service):
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.side_effect = Exception("API error")

        service._memory.add.return_value = {"results": []}

        service.extract_and_store(
            messages=[{"role": "user", "content": "hello"}],
            user_id="ehfaz",
        )

        # Should fall back to basic mem0 add
        service._memory.add.assert_called_once()

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
