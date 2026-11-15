"""NeuralScape patches to the vendored graphiti Gemini clients.

* every ``genai.Client`` graphiti builds carries a bounded HTTP timeout
  (``HttpOptions.timeout`` is in MILLISECONDS) so a stalled call cannot hang a
  graph worker slot;
* the 503 fallback passes the fallback model per call and never mutates the
  shared client (Graphiti runs many ``generate_response`` calls concurrently on
  one client).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from graphiti_core.cross_encoder import gemini_reranker_client
from graphiti_core.embedder import gemini as gemini_embedder
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client import gemini_client as gc
from graphiti_core.llm_client.config import GEMINI_HTTP_TIMEOUT_MS
from graphiti_core.prompts.models import Message


def test_timeout_constant_is_sixty_seconds_in_milliseconds():
    assert GEMINI_HTTP_TIMEOUT_MS == 60_000


def _assert_timeout(mock_client_cls):
    mock_client_cls.assert_called_once()
    http_options = mock_client_cls.call_args.kwargs["http_options"]
    assert http_options.timeout == GEMINI_HTTP_TIMEOUT_MS


def test_llm_client_sets_http_timeout():
    with patch.object(gc.genai, "Client") as client_cls:
        gc.GeminiClient(config=LLMConfig(api_key="k", model="m"))
    _assert_timeout(client_cls)


def test_embedder_sets_http_timeout():
    with patch.object(gemini_embedder.genai, "Client") as client_cls:
        gemini_embedder.GeminiEmbedder(
            config=gemini_embedder.GeminiEmbedderConfig(api_key="k")
        )
    _assert_timeout(client_cls)


def test_reranker_sets_http_timeout():
    with patch.object(gemini_reranker_client.genai, "Client") as client_cls:
        gemini_reranker_client.GeminiRerankerClient(config=LLMConfig(api_key="k"))
    _assert_timeout(client_cls)


def test_injected_client_is_used_untouched():
    sentinel = object()
    with patch.object(gc.genai, "Client") as client_cls:
        c = gc.GeminiClient(config=LLMConfig(api_key="k", model="m"), client=sentinel)
    client_cls.assert_not_called()
    assert c.client is sentinel


def _response(text="ok"):
    return SimpleNamespace(text=text, candidates=None, prompt_feedback=None, usage_metadata=None)


def test_fallback_does_not_mutate_shared_model_under_concurrency():
    primary, fallback = "primary-model", "fallback-model"
    calls: list[tuple[str, str]] = []  # (prompt tag, model)
    fallback_started = asyncio.Event()
    release_fallback = asyncio.Event()
    model_seen_mid_fallback: list[str] = []

    async def generate_content(model, contents, config):
        tag = contents[0].parts[0].text
        calls.append((tag, model))
        if tag.startswith("A"):
            if model == primary:
                raise Exception("503 UNAVAILABLE high demand")
            # Fallback call for A is now in flight: let B start, hold A open.
            model_seen_mid_fallback.append(client.model)
            fallback_started.set()
            await release_fallback.wait()
        return _response(tag)

    fake = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    client = gc.GeminiClient(
        config=LLMConfig(api_key="k", model=primary, fallback_model=fallback), client=fake
    )

    async def run():
        task_a = asyncio.create_task(
            client.generate_response([Message(role="user", content="A")], prompt_name="a")
        )
        await fallback_started.wait()
        # B starts while A's fallback call is mid-flight.
        out_b = await client.generate_response([Message(role="user", content="B")], prompt_name="b")
        model_while_a_in_flight = client.model
        release_fallback.set()
        out_a = await task_a
        return out_a, out_b, model_while_a_in_flight

    out_a, out_b, model_while_a_in_flight = asyncio.run(run())

    assert out_a["content"].startswith("A") and out_b["content"].startswith("B")
    b_models = [m for t, m in calls if t.startswith("B")]
    assert b_models == [primary], f"B must use the primary model, saw {b_models}"
    assert (
        next(m for t, m in calls if t.startswith("A") and m == fallback) == fallback
    ), "A's last attempt must go to the fallback model"
    assert model_seen_mid_fallback == [primary]
    assert model_while_a_in_flight == primary
    assert client.model == primary
