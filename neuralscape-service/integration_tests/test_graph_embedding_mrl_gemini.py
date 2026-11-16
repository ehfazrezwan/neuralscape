"""The premise behind prefix scoring, proved on the real Gemini embedder.

Recall compares the 768-d query vector with a longer stored graph vector on
their common prefix (memory/ranking.py ``_graph_edge_cosine``). That is valid
only if ``gemini-embedding-001`` at ``output_dimensionality=768`` is, up to
scale, the first 768 values of the same text embedded at 1024. This test embeds
real text both ways through the Graphiti embedder the service builds.

Spends a few embedding calls; needs a real GOOGLE_API_KEY (``llm_key``).
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import math

import pytest

TEXTS = [
    "The deployment pipeline publishes the runtime image after the container gate passes.",
    "Alice prefers dark mode and keeps her notes in the sandbox project.",
    "gemini embeddings can be truncated to a smaller output dimensionality",
]


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def _embed(llm_key, model, dim, text):
    from mem0.memory.graphiti_memory import _create_embedder

    embedder = _create_embedder("gemini", model, llm_key, embedding_dim=dim)
    return asyncio.run(embedder.create(input_data=[text]))


@pytest.fixture(scope="module")
def model():
    from config import settings

    return settings.gemini_embedder_model


@pytest.mark.parametrize("text", TEXTS)
def test_768d_output_is_the_prefix_of_the_1024d_output(llm_key, model, text):
    v768 = _embed(llm_key, model, 768, text)
    v1024 = _embed(llm_key, model, 1024, text)
    assert len(v768) == 768 and len(v1024) == 1024
    assert _cos(v768, v1024[:768]) >= 0.99


def test_service_scoring_helper_on_real_vectors(llm_key, model):
    """The helper search() uses, fed real vectors: same text scores ~1, an
    unrelated text scores clearly lower, and the value equals the direct
    prefix cosine."""
    from memory.ranking import _graph_edge_cosine, _graph_vectors_mrl_comparable

    assert _graph_vectors_mrl_comparable()
    query = _embed(llm_key, model, 768, TEXTS[0])  # what search() embeds
    same = _embed(llm_key, model, 1024, TEXTS[0])  # what Graphiti stored
    other = _embed(llm_key, model, 1024, TEXTS[2])
    s_same = _graph_edge_cosine(query, same)
    s_other = _graph_edge_cosine(query, other)
    assert s_same == pytest.approx(max(0.0, min(1.0, _cos(query, same[:768]))), abs=1e-9)
    assert s_same >= 0.99
    assert s_other is not None and s_other < s_same - 0.05
