"""The retry seam remembers the last response it recorded only until the caller-side hook has seen it."""

from types import SimpleNamespace

import pytest

import usage_capture as uc


@pytest.fixture()
def recorder():
    rec = uc.InMemoryUsageRecorder()
    previous = uc.set_usage_recorder(rec)
    yield rec
    uc.set_usage_recorder(previous)


def _response():
    return SimpleNamespace(
        text="ok",
        usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=1,
                                       cached_content_token_count=None, thoughts_token_count=None),
    )


def test_the_remembered_response_is_released_once_the_duplicate_was_skipped(recorder):
    response = _response()
    with uc.attempt_context(uc.AttemptContext(tenant_id="t", task_id="task", attempt_id="a")):
        attempt = uc.ProviderAttempt("op", "m", False)
        attempt.succeeded(response)
        assert uc._attempt_recorded.get() is response
        uc.record_generation_usage(response, operation="op", model="m")  # the caller-side hook: a duplicate
        assert len(recorder.events) == 1
        assert uc._attempt_recorded.get() is None  # nothing keeps the response alive
        uc.record_generation_usage(response, operation="op", model="m")  # a later, separate call records
        assert len(recorder.events) == 2
