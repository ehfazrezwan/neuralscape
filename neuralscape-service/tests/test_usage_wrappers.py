"""mem0/Graphiti client wrappers: one event per provider call, default-off."""

import asyncio
from types import SimpleNamespace

import pytest

import usage_capture as uc
import usage_wrappers as uw

CTX = uc.AttemptContext(tenant_id="t1", task_id="task1", attempt_id="a1")


class FakeGeminiModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, model=None, contents=None, config=None):
        self.calls += 1
        return SimpleNamespace(
            text="hi",
            model_version="v1",
            usage_metadata=SimpleNamespace(
                prompt_token_count=5,
                candidates_token_count=2,
                cached_content_token_count=None,
                thoughts_token_count=None,
            ),
        )

    def embed_content(self, model=None, contents=None, config=None):
        self.calls += 1
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1])])


class FakeGeminiClient:
    def __init__(self):
        self.models = FakeGeminiModels()
        self.api_key = "k"


class FakeAsyncModels(FakeGeminiModels):
    async def generate_content(self, model=None, contents=None, config=None):
        return FakeGeminiModels.generate_content(self, model, contents, config)


class FakeOpenAIClient:
    def __init__(self):
        async def create(**kw):
            return SimpleNamespace(
                model="gw", usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3)
            )

        async def emb(**kw):
            return SimpleNamespace(model="gw-e", usage=SimpleNamespace(prompt_tokens=4))

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.embeddings = SimpleNamespace(create=emb)


@pytest.fixture()
def recorder():
    rec = uc.InMemoryUsageRecorder()
    previous = uc.set_usage_recorder(rec)
    yield rec
    uc.set_usage_recorder(previous)


def test_sync_gemini_one_event_per_call_and_delegation(recorder):
    inner = FakeGeminiClient()
    proxy = uw.wrap_client(inner, "mem0_llm")
    with uc.attempt_context(CTX):
        assert proxy.models.generate_content(model="m", contents="x").text == "hi"
        proxy.models.embed_content(model="e", contents="x")
    ops = [(e.operation, e.status) for e in recorder.events]
    assert ops == [("mem0_llm", "final"), ("mem0_llm", "unavailable")]
    assert proxy.api_key == "k" and inner.models.calls == 2


def test_async_gemini_and_openai(recorder):
    inner = SimpleNamespace(aio=SimpleNamespace(models=FakeAsyncModels()))
    gem = uw.wrap_client(inner, "graphiti_llm")
    oa = uw.wrap_client(FakeOpenAIClient(), "graphiti_embedding")

    async def run():
        with uc.attempt_context(CTX):
            await gem.aio.models.generate_content(model="m", contents="x")
            await oa.chat.completions.create(model="gw", messages=[])
            await oa.embeddings.create(model="gw-e", input="x")

    asyncio.run(run())
    assert [e.operation for e in recorder.events] == [
        "graphiti_llm",
        "graphiti_embedding",
        "graphiti_embedding",
    ]
    assert recorder.events[2].usage.output_tokens.missing_reason == "not_applicable"


def test_wrap_is_idempotent_and_none_safe():
    proxy = uw.wrap_client(FakeGeminiClient(), "x")
    assert uw.wrap_client(proxy, "y") is proxy
    assert uw.wrap_client(None, "x") is None


def test_provider_error_recorded_as_failed_and_propagates(recorder):
    class Boom:
        def generate_content(self, **kw):
            raise ValueError("bad")

    proxy = uw.wrap_client(SimpleNamespace(models=Boom()), "mem0_llm")
    with uc.attempt_context(CTX), pytest.raises(ValueError):
        proxy.models.generate_content(model="m")
    (event,) = recorder.events
    assert (event.attempt_outcome, event.status) == ("failed", "unavailable")
    assert event.attribution.model == "m"


def test_async_failure_and_cancellation_recorded(recorder):
    class Boom:
        async def create(self, **kw):
            raise kw["exc"]

    proxy = uw.wrap_client(SimpleNamespace(chat=SimpleNamespace(completions=Boom())), "graphiti_llm")

    async def run():
        with uc.attempt_context(CTX):
            for exc in (RuntimeError("x"), asyncio.CancelledError()):
                try:
                    await proxy.chat.completions.create(model="gw", exc=exc)
                except BaseException:
                    pass

    asyncio.run(run())
    assert [e.attempt_outcome for e in recorder.events] == ["failed", "cancelled"]


def test_recorder_failure_never_reaches_caller():
    class Boom:
        def record(self, event):
            raise RuntimeError("x")

    previous = uc.set_usage_recorder(Boom())
    try:
        proxy = uw.wrap_client(FakeGeminiClient(), "mem0_llm")
        with uc.attempt_context(CTX):
            assert proxy.models.generate_content(model="m").text == "hi"
    finally:
        uc.set_usage_recorder(previous)


def test_install_wrappers_on_real_graphiti_openai_embedder(recorder):
    from graphiti_core.embedder.openai import OpenAIEmbedder

    class FakeEmbeddings:
        async def create(self, **kw):
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.5] * 4)],
                usage=SimpleNamespace(prompt_tokens=6),
            )

    embedder = OpenAIEmbedder(client=SimpleNamespace(embeddings=FakeEmbeddings()))
    graphiti = SimpleNamespace(llm_client=None, embedder=embedder, cross_encoder=None)
    uw.install_graphiti_wrappers(graphiti)

    async def run():
        with uc.attempt_context(CTX):
            return await embedder.create("hello")

    assert len(asyncio.run(run())) == 4
    (event,) = recorder.events
    assert event.operation == "graphiti_embedding"
    assert event.usage.input_tokens.value == 6


def test_install_mem0_wrappers_labels(recorder):
    memory = SimpleNamespace(
        llm=SimpleNamespace(client=FakeGeminiClient()),
        embedding_model=SimpleNamespace(client=FakeGeminiClient()),
    )
    uw.install_mem0_wrappers(memory)
    with uc.attempt_context(CTX):
        memory.llm.client.models.generate_content(model="m")
        memory.embedding_model.client.models.embed_content(model="e")
    assert [e.operation for e in recorder.events] == ["mem0_llm", "mem0_embedding"]


def test_install_never_raises_on_odd_objects():
    uw.install_mem0_wrappers(object())
    uw.install_graphiti_wrappers(object())


def test_async_wrappers_record_off_the_event_loop_thread(recorder):
    """Before: ``record`` (a synchronous store write that may sleep through retries) ran on the loop thread
    inside every awaited provider call.  The attempt context must still reach the recording thread."""
    import threading

    seen = []
    real = recorder.record

    def spying_record(event):
        seen.append((threading.get_ident(), uc.current_attempt_context()))
        return real(event)

    recorder.record = spying_record
    inner = SimpleNamespace(aio=SimpleNamespace(models=FakeAsyncModels()))
    gem = uw.wrap_client(inner, "graphiti_llm")

    class Boom:
        async def create(self, **kw):
            raise RuntimeError("down")

    boom = uw.wrap_client(SimpleNamespace(chat=SimpleNamespace(completions=Boom())), "graphiti_llm")
    loop_thread = []

    async def run():
        loop_thread.append(threading.get_ident())
        with uc.attempt_context(CTX):
            await gem.aio.models.generate_content(model="m", contents="x")
            with pytest.raises(RuntimeError):
                await boom.chat.completions.create(model="gw", messages=[])

    asyncio.run(run())
    assert len(seen) == 2  # a success and a failure were both recorded
    assert all(thread != loop_thread[0] for thread, _ in seen)
    assert all(context is CTX for _, context in seen)
