"""The two extraction call sites in memory/write.py feed the usage seam."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import usage_capture as uc

CTX = uc.AttemptContext(tenant_id="t1", task_id="task1", attempt_id="a1")


def _service(response):
    from memory_service import MemoryService

    svc = MemoryService.__new__(MemoryService)
    client = MagicMock()
    client.models.generate_content = MagicMock(return_value=response)
    svc._genai_model = client
    return svc


def _response(text='{"facts": []}'):
    meta = SimpleNamespace(
        prompt_token_count=11,
        candidates_token_count=4,
        cached_content_token_count=None,
        thoughts_token_count=None,
    )
    return SimpleNamespace(text=text, usage_metadata=meta, model_version="m-1")


@pytest.fixture()
def recorder():
    rec = uc.InMemoryUsageRecorder()
    previous = uc.set_usage_recorder(rec)
    yield rec
    uc.set_usage_recorder(previous)


def test_ingest_extraction_records_usage(recorder):
    svc = _service(_response())
    with uc.attempt_context(CTX):
        assert svc.extract_facts_only("doc", user_id="u1") == []
    (event,) = recorder.events
    assert event.operation == "llm_extraction_ingest"
    assert event.usage.input_tokens.value == 11


def test_conversation_extraction_records_usage(recorder):
    svc = _service(_response())
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    with uc.attempt_context(CTX):
        svc.extract_and_store([{"role": "user", "content": "hi"}], user_id="u1")
    ops = [e.operation for e in recorder.events]
    assert ops == ["llm_extraction"]


def test_no_recorder_leaves_results_unchanged():
    previous = uc.set_usage_recorder(None)
    try:
        svc = _service(_response())
        with uc.attempt_context(CTX):
            assert svc.extract_facts_only("doc", user_id="u1") == []
    finally:
        uc.set_usage_recorder(previous)


def test_failing_recorder_does_not_break_extraction():
    class Boom:
        def record(self, event):
            raise RuntimeError("boom")

    previous = uc.set_usage_recorder(Boom())
    try:
        svc = _service(_response())
        with uc.attempt_context(CTX):
            assert svc.extract_facts_only("doc", user_id="u1") == []
    finally:
        uc.set_usage_recorder(previous)


def test_failed_generation_records_nothing(recorder):
    from memory_service import MemoryService

    svc = MemoryService.__new__(MemoryService)
    client = MagicMock()
    client.models.generate_content = MagicMock(side_effect=ValueError("bad request"))
    svc._genai_model = client
    with uc.attempt_context(CTX):
        assert svc.extract_facts_only("doc", user_id="u1") == []
    assert recorder.events == []
