"""Per-attempt capture in retry_transient: failed, retried, fallback, cancelled."""

import asyncio
from types import SimpleNamespace

import pytest

import usage_capture as uc
from contracts_usage_reconcile import reconcile_usage_events
from memory.retry import retry_transient

CTX = uc.AttemptContext(tenant_id="t1", task_id="task1", attempt_id="a1")


def _resp(model_version=None, prompt=5):
    return SimpleNamespace(
        text="ok",
        model_version=model_version,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=1,
            cached_content_token_count=None,
            thoughts_token_count=None,
        ),
    )


def make_call(outcomes):
    """A provider-shaped callable returning/raising ``outcomes`` in order."""
    seen = []

    def generate_content(model=None, contents=None):
        seen.append(model)
        item = outcomes[len(seen) - 1]
        if isinstance(item, BaseException):
            raise item
        return item

    generate_content.seen = seen
    return generate_content


@pytest.fixture()
def recorder():
    rec = uc.InMemoryUsageRecorder()
    previous = uc.set_usage_recorder(rec)
    yield rec
    uc.set_usage_recorder(previous)


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr("memory.retry.time.sleep", lambda s: None)


def run(fn, **kw):
    return retry_transient(
        fn, model="primary", contents="x", max_retries=1, base_delay=0, max_delay=0,
        operation="LLM extraction", fallback_model="fallback", **kw
    )


def test_success_records_one_final_event(recorder):
    fn = make_call([_resp()])
    with uc.attempt_context(CTX):
        assert run(fn).text == "ok"
    (event,) = recorder.events
    assert (event.operation, event.status, event.attempt_outcome) == (
        "llm_extraction", "final", "completed")
    assert event.attribution.model == "primary" and event.attribution.producer == "llm_extraction"


def test_transient_failures_then_success_each_recorded(recorder):
    fn = make_call([RuntimeError("503 UNAVAILABLE"), _resp()])
    with uc.attempt_context(CTX):
        run(fn)
    failed, ok = recorder.events
    assert (failed.attempt_outcome, failed.status) == ("failed", "unavailable")
    assert failed.usage_missing_reason == "provider_did_not_report"
    assert ok.status == "final"
    assert failed.attempt_id != ok.attempt_id
    assert reconcile_usage_events(recorder.events) is not None


def test_fallback_attribution(recorder):
    fn = make_call([RuntimeError("503"), RuntimeError("503"), _resp()])
    with uc.attempt_context(CTX):
        run(fn)
    assert fn.seen == ["primary", "primary", "fallback"]
    primary1, primary2, fb = recorder.events
    assert primary1.attribution.producer == "llm_extraction"
    assert fb.attribution.producer == "llm_extraction.fallback"
    assert fb.attribution.model == "fallback" and fb.status == "final"
    assert [e.attempt_outcome for e in recorder.events] == ["failed", "failed", "completed"]


def test_non_transient_failure_recorded_and_raised(recorder):
    fn = make_call([ValueError("bad request")])
    with uc.attempt_context(CTX), pytest.raises(ValueError):
        run(fn)
    (event,) = recorder.events
    assert event.attempt_outcome == "failed"


def test_cancelled_attempt_recorded(recorder):
    fn = make_call([asyncio.CancelledError()])
    with uc.attempt_context(CTX), pytest.raises(asyncio.CancelledError):
        run(fn)
    (event,) = recorder.events
    assert event.attempt_outcome == "cancelled"
    assert event.usage_missing_reason == "not_yet_reported"
    assert event.late_after_cancellation is False


def test_caller_hook_does_not_double_record_the_final_response(recorder):
    response = _resp()
    fn = make_call([response])
    with uc.attempt_context(CTX):
        run(fn)
        uc.record_generation_usage(response, operation="llm_extraction", model="primary")
    assert len(recorder.events) == 1


def test_separate_call_without_retry_seam_still_recorded(recorder):
    with uc.attempt_context(CTX):
        uc.record_generation_usage(_resp(), operation="llm_extraction", model="primary")
    assert len(recorder.events) == 1


def test_non_provider_callable_not_reported(recorder):
    def graph_write(**kw):
        return "ok"

    with uc.attempt_context(CTX):
        assert retry_transient(graph_write, operation="graph", max_retries=0) == "ok"
    assert recorder.events == []


def test_default_off_identical_behaviour():
    previous = uc.set_usage_recorder(None)
    try:
        fn = make_call([RuntimeError("503"), RuntimeError("503"), _resp()])
        with uc.attempt_context(CTX):
            assert run(fn).text == "ok"
        assert fn.seen == ["primary", "primary", "fallback"]
        fn = make_call([ValueError("x")])
        with pytest.raises(ValueError):
            run(fn)
    finally:
        uc.set_usage_recorder(previous)


def test_no_attempt_context_records_nothing(recorder):
    run(make_call([_resp()]))
    assert recorder.events == []


def test_recorder_failure_invisible_to_retry(recorder):
    class Boom:
        def record(self, event):
            raise RuntimeError("boom")

    previous = uc.set_usage_recorder(Boom())
    try:
        fn = make_call([RuntimeError("503"), _resp()])
        with uc.attempt_context(CTX):
            assert run(fn).text == "ok"
        fn = make_call([ValueError("v")])
        with uc.attempt_context(CTX), pytest.raises(ValueError):
            run(fn)
    finally:
        uc.set_usage_recorder(previous)


def test_unreported_usage_on_success_is_unavailable_not_zero(recorder):
    resp = SimpleNamespace(text="t", usage_metadata=None)
    with uc.attempt_context(CTX):
        run(make_call([resp]))
    (event,) = recorder.events
    assert event.status == "unavailable" and event.usage is None


def test_write_extraction_path_records_exactly_one_event_per_provider_call(recorder):
    """Real retry seam + caller hook in memory/write.py: no double counting."""
    from memory_service import MemoryService

    outcomes = [RuntimeError("503 UNAVAILABLE"), _resp(prompt=9)]
    seen = []

    def generate_content(model=None, contents=None, config=None):
        seen.append(model)
        item = outcomes[len(seen) - 1]
        if isinstance(item, BaseException):
            raise item
        item.text = '{"facts": []}'
        return item

    svc = MemoryService.__new__(MemoryService)
    svc._genai_model = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    with uc.attempt_context(CTX):
        assert svc.extract_facts_only("doc", user_id="u1") == []
    assert [e.attempt_outcome for e in recorder.events] == ["failed", "completed"]
    assert {e.operation for e in recorder.events} == {"llm_extraction_ingest"}
    assert recorder.events[1].usage.input_tokens.value == 9
