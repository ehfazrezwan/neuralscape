"""Producer seams (compile/dreaming/ask) and core install gating."""

import asyncio
from types import SimpleNamespace

import pytest

import usage_capture as uc

CTX = uc.AttemptContext(tenant_id="t1", task_id="task1", attempt_id="a1")


def _response():
    return SimpleNamespace(
        text="answer",
        model_version="v",
        usage_metadata=SimpleNamespace(
            prompt_token_count=4,
            candidates_token_count=1,
            cached_content_token_count=None,
            thoughts_token_count=None,
        ),
    )


@pytest.fixture()
def recorder():
    rec = uc.InMemoryUsageRecorder()
    previous = uc.set_usage_recorder(rec)
    yield rec
    uc.set_usage_recorder(previous)


@pytest.fixture()
def fake_genai(monkeypatch):
    from extensions.conversation_compiler import compile as comp

    calls = []

    class Models:
        def generate_content(self, model=None, contents=None):
            calls.append(model)
            return _response()

    class Client:
        def __init__(self, api_key=None):
            self.models = Models()

    monkeypatch.setattr(comp.genai, "Client", Client)
    return comp, calls


def test_compile_seam_default_label(fake_genai, recorder):
    comp, calls = fake_genai
    with uc.attempt_context(CTX):
        assert asyncio.run(comp._async_call_gemini("p")) == "answer"
    (event,) = recorder.events
    assert event.operation == "llm_compile" and len(calls) == 1


def test_dreaming_and_ask_labels_one_event_each(fake_genai, recorder):
    from ask import _make_llm_call
    from extensions.dreaming.sweep import _make_llm_call as dream_call

    async def run():
        dreamer = await dream_call(SimpleNamespace(llm_max_retries=0, llm_timeout_seconds=5))
        asker = _make_llm_call(SimpleNamespace(name="t", llm_timeout_s=5))
        with uc.attempt_context(CTX):
            await dreamer("p")
            await asker("p")

    asyncio.run(run())
    assert [e.operation for e in recorder.events] == ["llm_dreaming", "llm_ask"]


def test_seams_silent_without_recorder(fake_genai):
    comp, _ = fake_genai
    previous = uc.set_usage_recorder(None)
    try:
        with uc.attempt_context(CTX):
            assert asyncio.run(comp._async_call_gemini("p")) == "answer"
    finally:
        uc.set_usage_recorder(previous)


def _service(monkeypatch):
    import config
    import mem0
    from memory_service import MemoryService

    fake = SimpleNamespace(
        llm=SimpleNamespace(client=SimpleNamespace(models=SimpleNamespace())),
        embedding_model=SimpleNamespace(client=SimpleNamespace(models=SimpleNamespace())),
    )
    monkeypatch.setattr(type(config.settings), "get_mem0_config", lambda self: {})
    monkeypatch.setattr(mem0.Memory, "from_config", classmethod(lambda cls, c: fake))
    return MemoryService(), fake


def test_core_installs_wrappers_only_with_recorder(monkeypatch, recorder):
    import usage_wrappers as uw

    svc, fake = _service(monkeypatch)
    inner_llm, inner_emb = fake.llm.client, fake.embedding_model.client
    svc._get_memory()
    assert isinstance(fake.llm.client, uw._CapturingProxy)
    assert fake.llm.client is not inner_llm and fake.embedding_model.client is not inner_emb


def test_core_leaves_clients_untouched_without_recorder(monkeypatch):
    previous = uc.set_usage_recorder(None)
    try:
        svc, fake = _service(monkeypatch)
        inner_llm, inner_emb = fake.llm.client, fake.embedding_model.client
        svc._get_memory()
        assert fake.llm.client is inner_llm and fake.embedding_model.client is inner_emb
    finally:
        uc.set_usage_recorder(previous)
