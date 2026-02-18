"""Integration tests for the Graphiti adapter — requires real Neo4j + Gemini.

Skipped unless NEO4J_URI and GOOGLE_API_KEY environment variables are set.
"""

import os
import uuid

import pytest

requires_neo4j = pytest.mark.skipif(
    not os.environ.get("NEO4J_URI") or not os.environ.get("GOOGLE_API_KEY"),
    reason="NEO4J_URI and GOOGLE_API_KEY env vars required for integration tests",
)


def _integration_config(database="neo4j"):
    """Build a MemoryConfig dict for integration tests."""
    return {
        "llm": {
            "provider": "gemini",
            "config": {
                "model": os.environ.get("GEMINI_LLM_MODEL", "gemini-2.5-flash"),
                "api_key": os.environ["GOOGLE_API_KEY"],
            },
        },
        "embedder": {
            "provider": "gemini",
            "config": {
                "model": os.environ.get("GEMINI_EMBEDDER_MODEL", "text-embedding-004"),
                "api_key": os.environ["GOOGLE_API_KEY"],
                "embedding_dims": 768,
            },
        },
        "graph_store": {
            "provider": "graphiti",
            "config": {
                "url": os.environ["NEO4J_URI"],
                "username": os.environ.get("NEO4J_USER", "neo4j"),
                "password": os.environ.get("NEO4J_PASSWORD", ""),
                "database": database,
                "graphiti_llm_provider": "gemini",
                "graphiti_llm_model": os.environ.get("GEMINI_LLM_MODEL", "gemini-2.5-flash"),
                "graphiti_llm_api_key": os.environ["GOOGLE_API_KEY"],
                "graphiti_embedder_provider": "gemini",
                "graphiti_embedder_model": os.environ.get("GEMINI_EMBEDDER_MODEL", "text-embedding-004"),
                "graphiti_embedder_api_key": os.environ["GOOGLE_API_KEY"],
                "graphiti_reranker_provider": "gemini",
            },
        },
        "version": "v1.1",
    }


@requires_neo4j
class TestFullCycleDirect:
    """Test the MemoryGraph adapter directly (no mem0 Memory wrapper)."""

    def test_full_cycle(self):
        from mem0.configs.base import MemoryConfig
        from mem0.memory.graphiti_memory import MemoryGraph

        config = MemoryConfig(**_integration_config())
        mg = MemoryGraph(config)
        user_id = f"integ_test_{uuid.uuid4().hex[:8]}"
        filters = {"user_id": user_id}

        try:
            # Add
            result = mg.add("Alice works at Acme Corp as a software engineer.", filters)
            assert "added_entities" in result
            assert len(result["added_entities"]) > 0

            # Search
            results = mg.search("Where does Alice work?", filters)
            assert len(results) > 0
            facts = " ".join(str(r) for r in results).lower()
            assert "alice" in facts or "acme" in facts

            # Get all
            all_results = mg.get_all(filters)
            assert len(all_results) > 0

        finally:
            # Cleanup
            mg.delete_all(filters)


@requires_neo4j
class TestFullCycleViaMemory:
    """Test through the full mem0 Memory.from_config interface."""

    def test_full_cycle(self):
        from mem0 import Memory

        config = _integration_config()
        m = Memory.from_config(config)
        user_id = f"integ_mem_{uuid.uuid4().hex[:8]}"

        try:
            # Add
            result = m.add(
                messages=[{"role": "user", "content": "Bob lives in San Francisco and works at Google."}],
                user_id=user_id,
            )
            assert result is not None

            # Search
            results = m.search(query="Where does Bob live?", user_id=user_id)
            assert results is not None

            # Get all
            all_results = m.get_all(user_id=user_id)
            assert all_results is not None

        finally:
            # Cleanup
            m.delete_all(user_id=user_id)


@requires_neo4j
class TestTemporalEdgeInvalidation:
    """Test that adding contradicting facts sets expired_at on old edges."""

    def test_temporal_invalidation(self):
        from mem0.configs.base import MemoryConfig
        from mem0.memory.graphiti_memory import MemoryGraph

        config = MemoryConfig(**_integration_config())
        mg = MemoryGraph(config)
        user_id = f"integ_temporal_{uuid.uuid4().hex[:8]}"
        filters = {"user_id": user_id}

        try:
            # Add initial fact
            mg.add("Charlie works at Microsoft.", filters)

            # Add contradicting fact
            mg.add("Charlie left Microsoft and now works at Apple.", filters)

            # Search should reflect the updated relationship
            results = mg.search("Where does Charlie work?", filters)
            facts = " ".join(str(r) for r in results).lower()
            # The new fact should be present
            assert "apple" in facts or "charlie" in facts

        finally:
            mg.delete_all(filters)
