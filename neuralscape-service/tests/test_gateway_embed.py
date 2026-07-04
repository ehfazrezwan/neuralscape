"""Gateway batch-embed fix — regression tests.

The LLM gateway's Vertex embedding endpoint (google-vertex/gemini-embedding-001)
accepts only ONE input per embeddings request. mem0's OpenAIEmbedding.embed_batch
sent up to 100 inputs in a single call, so with LLM_GATEWAY_ENABLED=true the
extraction path (POST /v1/memories → extract_and_store → _batch_store_facts)
failed on every conversation — and the old except-and-continue in
extract_and_store swallowed the error, silently storing ZERO facts while the
task reported success.

Covered here:
- OpenAIEmbedding honors ``embedding_batch_size`` (client-side chunking) and
  the batch-size-1 path returns identical vectors to the batched path.
- A batched call rejected with a single-input error falls back to per-item
  embeds; other errors propagate.
- get_mem0_config wires embedding_batch_size=1 on the gateway path (overridable
  via EMBEDDER_MAX_BATCH_SIZE) and leaves the AI Studio path untouched.
- Embed/store failures inside extract_and_store PROPAGATE (fail the ARQ job →
  task status "failed") instead of returning [].
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.openai import OpenAIEmbedding

from config import Settings
from memory_service import MemoryService

SINGLE_INPUT_ERROR = (
    "Error code: 400 - google-vertex/gemini-embedding-001 accepts only one input per request"
)


def _vec(text: str) -> list[float]:
    """Deterministic per-text fake embedding."""
    return [float(sum(text.encode()) % 97)] * 8


def _embedder(**config_kwargs) -> OpenAIEmbedding:
    """OpenAIEmbedding with a deterministic mocked client."""
    config = BaseEmbedderConfig(
        model="google-vertex/gemini-embedding-001",
        api_key="test-key",
        openai_base_url="http://localhost:9/v1",
        embedding_dims=768,
        **config_kwargs,
    )
    emb = OpenAIEmbedding(config)
    emb.client = MagicMock(name="OpenAIClient")

    def _create(**kwargs):
        inputs = kwargs["input"]
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=_vec(t))
                for i, t in enumerate(inputs)
            ]
        )

    emb.client.embeddings.create.side_effect = _create
    return emb


def _single_input_only(emb: OpenAIEmbedding) -> None:
    """Make the mocked backend reject any request with more than one input."""
    ok = emb.client.embeddings.create.side_effect

    def _create(**kwargs):
        if len(kwargs["input"]) > 1:
            raise Exception(SINGLE_INPUT_ERROR)
        return ok(**kwargs)

    emb.client.embeddings.create.side_effect = _create


TEXTS = ["Prefers dark mode", "Expert in Python 3.12", "Lives in Dhaka"]


class TestEmbeddingBatchSize:
    def test_default_sends_one_batched_call(self):
        emb = _embedder()
        result = emb.embed_batch(TEXTS)
        assert result == [_vec(t) for t in TEXTS]
        assert emb.client.embeddings.create.call_count == 1
        assert emb.client.embeddings.create.call_args.kwargs["input"] == TEXTS

    def test_batch_size_one_embeds_per_item(self):
        emb = _embedder(embedding_batch_size=1)
        result = emb.embed_batch(TEXTS)
        assert result == [_vec(t) for t in TEXTS]
        assert emb.client.embeddings.create.call_count == len(TEXTS)
        for call, text in zip(
            emb.client.embeddings.create.call_args_list, TEXTS, strict=True
        ):
            assert call.kwargs["input"] == [text]

    def test_batch_size_one_matches_batched_output(self):
        batched = _embedder().embed_batch(TEXTS)
        per_item = _embedder(embedding_batch_size=1).embed_batch(TEXTS)
        assert per_item == batched

    def test_batch_size_one_works_against_single_input_only_backend(self):
        emb = _embedder(embedding_batch_size=1)
        _single_input_only(emb)
        assert emb.embed_batch(TEXTS) == [_vec(t) for t in TEXTS]

    def test_batch_size_clamped_to_valid_range(self):
        emb = _embedder(embedding_batch_size=0)  # falsy → provider default
        emb.embed_batch(TEXTS)
        assert emb.client.embeddings.create.call_count == 1

    def test_string_batch_size_coerced_to_int(self):
        """Env/JSON-driven configs may carry '1' — coerced, not TypeError'd."""
        emb = _embedder(embedding_batch_size="1")
        result = emb.embed_batch(TEXTS)
        assert result == [_vec(t) for t in TEXTS]
        assert emb.client.embeddings.create.call_count == len(TEXTS)

    def test_non_numeric_batch_size_raises_clear_error(self):
        emb = _embedder(embedding_batch_size="lots")
        with pytest.raises(ValueError, match="embedding_batch_size"):
            emb.embed_batch(TEXTS)

    def test_config_without_field_uses_default(self):
        """Defensive getattr: a config object predating the NS field still batches."""
        emb = _embedder()
        del emb.config.embedding_batch_size
        emb.embed_batch(TEXTS)
        assert emb.client.embeddings.create.call_count == 1


class TestSingleInputRejectionFallback:
    def test_batched_rejection_falls_back_to_per_item(self):
        emb = _embedder()  # default batching → first call is batched
        _single_input_only(emb)
        result = emb.embed_batch(TEXTS)
        assert result == [_vec(t) for t in TEXTS]
        # 1 failed batched call + 3 per-item retries
        assert emb.client.embeddings.create.call_count == 1 + len(TEXTS)

    def test_unrelated_errors_propagate(self):
        emb = _embedder()
        emb.client.embeddings.create.side_effect = Exception("Error code: 401 - invalid api key")
        with pytest.raises(Exception, match="401"):
            emb.embed_batch(TEXTS)

    def test_per_item_failure_during_fallback_propagates(self):
        emb = _embedder()
        calls = {"n": 0}

        def _create(**kwargs):
            calls["n"] += 1
            if len(kwargs["input"]) > 1:
                raise Exception(SINGLE_INPUT_ERROR)
            raise Exception("Error code: 500 - backend exploded")

        emb.client.embeddings.create.side_effect = _create
        with pytest.raises(Exception, match="500"):
            emb.embed_batch(TEXTS)


class TestConfigWiring:
    """EMBEDDER_MAX_BATCH_SIZE → mem0 embedder config (hermetic Settings)."""

    @pytest.fixture(autouse=True)
    def _hermetic_env(self, monkeypatch):
        for var in (
            "OPENAI_BASE_URL", "OPENAI_API_KEY",
            "LLM_GATEWAY_ENABLED", "LLM_GATEWAY_BASE_URL", "LLM_GATEWAY_API_KEY",
            "LLM_GATEWAY_EMBEDDER_MODEL", "EMBEDDER_MAX_BATCH_SIZE",
            "GOOGLE_API_KEY", "NEO4J_PASSWORD",
        ):
            monkeypatch.delenv(var, raising=False)

    @staticmethod
    def _settings(**kw) -> Settings:
        return Settings(_env_file=None, **kw)

    def test_gateway_defaults_to_single_input_embeds(self):
        cfg = self._settings(
            llm_gateway_enabled=True,
            llm_gateway_base_url="https://gw.example.com",
            llm_gateway_api_key="k",
        ).get_mem0_config()
        assert cfg["embedder"]["provider"] == "openai"
        assert cfg["embedder"]["config"]["embedding_batch_size"] == 1

    def test_gateway_batch_size_overridable(self):
        cfg = self._settings(
            llm_gateway_enabled=True,
            llm_gateway_base_url="https://gw.example.com",
            llm_gateway_api_key="k",
            embedder_max_batch_size=8,
        ).get_mem0_config()
        assert cfg["embedder"]["config"]["embedding_batch_size"] == 8

    def test_ai_studio_path_unchanged(self):
        cfg = self._settings(llm_gateway_enabled=False).get_mem0_config()
        assert cfg["embedder"]["provider"] == "gemini"
        assert "embedding_batch_size" not in cfg["embedder"]["config"]


# ── Failure propagation through extract_and_store ──


@pytest.fixture
def service():
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    return svc


def _mock_extraction(svc, facts: list[str]) -> None:
    client = MagicMock()
    svc._genai_model = client
    import json

    client.models.generate_content.return_value = MagicMock(
        text=json.dumps({"facts": facts})
    )


class TestFailurePropagation:
    def test_embed_failure_fails_extract_and_store(self, service):
        """The old except-and-continue returned [] with task status success —
        the gateway rejection must now surface as a failed task."""
        _mock_extraction(service, ["[preference] Prefers dark mode"])
        service._memory.embedding_model.embed_batch.side_effect = Exception(
            SINGLE_INPUT_ERROR
        )
        with pytest.raises(Exception, match="one input per request"):
            service.extract_and_store(
                messages=[{"role": "user", "content": "I prefer dark mode"}],
                user_id="ehfaz",
            )
        service._memory.vector_store.insert.assert_not_called()
        # Graph add never runs on a failed store (no phantom half-writes)
        service._memory.graph.add.assert_not_called()

    def test_qdrant_failure_fails_extract_and_store(self, service):
        _mock_extraction(service, ["[preference] Prefers dark mode"])
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]
        service._memory.vector_store.insert.side_effect = Exception("qdrant down")
        with pytest.raises(Exception, match="qdrant down"):
            service.extract_and_store(
                messages=[{"role": "user", "content": "I prefer dark mode"}],
                user_id="ehfaz",
            )

    def test_batch_store_facts_propagates_embed_failure(self, service):
        service._memory.embedding_model.embed_batch.side_effect = Exception(
            SINGLE_INPUT_ERROR
        )
        with pytest.raises(Exception, match="one input per request"):
            service._batch_store_facts(
                facts=[("preference", "Prefers dark mode")], user_id="ehfaz"
            )

    def test_batch_size_one_embedder_produces_identical_stores(self, service):
        """End-to-end through _batch_store_facts: a real OpenAIEmbedding with
        embedding_batch_size=1 stores the exact same vectors/payload data as
        the batched embedder."""
        facts = [("preference", t) for t in TEXTS]

        service._memory.embedding_model = _embedder()
        service._batch_store_facts(facts=facts, user_id="ehfaz")
        batched_kwargs = service._memory.vector_store.insert.call_args.kwargs

        service._memory.vector_store.insert.reset_mock()
        service._memory.embedding_model = _embedder(embedding_batch_size=1)
        service._batch_store_facts(facts=facts, user_id="ehfaz")
        per_item_kwargs = service._memory.vector_store.insert.call_args.kwargs

        assert per_item_kwargs["vectors"] == batched_kwargs["vectors"]
        assert [p["data"] for p in per_item_kwargs["payloads"]] == [
            p["data"] for p in batched_kwargs["payloads"]
        ]
