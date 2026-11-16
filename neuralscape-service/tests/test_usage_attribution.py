"""Who a provider call is billed to: attempt contexts bound at the service's own entry points.

Capture records nothing without a bound attempt context.  Until these entry points bound one, a service
running with USAGE_STORE_ENABLED recorded no event from any real path (only tests, the pilot and the
facade bound a context), and Ask labelled its calls but never reached the capture seam.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import usage_capture as uc
from schemas import MemoryResponse


@pytest.fixture()
def recorder():
    rec = uc.InMemoryUsageRecorder()
    previous = uc.set_usage_recorder(rec)
    yield rec
    uc.set_usage_recorder(previous)


def _response(prompt=10, output=5):
    return SimpleNamespace(
        text="ok",
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt, candidates_token_count=output,
            cached_content_token_count=None, thoughts_token_count=None,
        ),
    )


# -- ensure_attempt / attributed ------------------------------------------------------------------


def test_ensure_attempt_binds_the_person_only_while_a_recorder_is_installed(recorder):
    assert uc.current_attempt_context() is None
    with uc.ensure_attempt("alice", "search"):
        context = uc.current_attempt_context()
        assert context.tenant_id == "alice" and context.task_id and context.attempt_id.endswith(":search")
    assert uc.current_attempt_context() is None
    uc.set_usage_recorder(None)
    with uc.ensure_attempt("alice", "search"):
        assert uc.current_attempt_context() is None  # no recorder: nothing is bound


def test_ensure_attempt_keeps_a_context_the_caller_bound(recorder):
    outer = uc.AttemptContext(tenant_id="t", task_id="ledger-task", attempt_id="a", ledger_attempt=3)
    with uc.attempt_context(outer):
        with uc.ensure_attempt("alice", "search"):
            assert uc.current_attempt_context() is outer


def test_a_missing_person_is_recorded_as_unattributed_not_dropped(recorder):
    with uc.ensure_attempt(None, "ask"):
        assert uc.current_attempt_context().tenant_id == uc.UNATTRIBUTED_TENANT


def test_attributed_decorator_binds_the_user_argument(recorder):
    seen = []

    @uc.attributed("search")
    def search(query, user_id, limit=3):
        seen.append(uc.current_attempt_context())
        return limit

    assert search("q", "bob", limit=4) == 4 and search("q", user_id="carol") == 3
    assert [c.tenant_id for c in seen] == ["bob", "carol"]
    uc.set_usage_recorder(None)
    assert search("q", "dave") == 3 and seen[-1] is None


def test_the_service_search_methods_bind_the_caller(recorder):
    from memory_service import MemoryService

    for name in ("search", "search_graph"):
        fn = getattr(MemoryService, name)
        assert getattr(fn, "__wrapped__", None) is not None, f"{name} is not attributed"


def test_job_attempt_is_bound_for_the_life_of_the_task_and_unbound(recorder):
    async def job():
        assert uc.bind_job_attempt("alice", "job-1", 2) is True
        context = uc.current_attempt_context()
        await asyncio.sleep(0)
        assert uc.current_attempt_context() is context
        uc.unbind_job_attempt()
        return context

    context = asyncio.run(job())
    assert (context.tenant_id, context.task_id, context.attempt_id) == ("alice", "job-1", "job-1:try2")
    uc.set_usage_recorder(None)
    assert uc.bind_job_attempt("alice", "job-2") is False


# -- the worker's job hooks ---------------------------------------------------------------------------


class _Redis:
    def __init__(self, mapping):
        self.mapping = mapping

    async def get(self, key):
        return self.mapping.get(key)


def test_worker_binds_each_job_to_the_user_who_enqueued_it_and_unbinds_after(recorder):
    import worker

    ctx = {"job_id": "job-7", "job_try": 1, "redis": _Redis({"ns:task-user:job-7": b"alice"})}

    async def run():
        await worker._make_on_job_start("q")(ctx)
        during = uc.current_attempt_context()
        await worker._make_after_job_end("q")(ctx)
        return during, uc.current_attempt_context()

    during, after = asyncio.run(run())
    assert (during.tenant_id, during.task_id) == ("alice", "job-7")
    assert after is None


def test_worker_records_a_job_without_an_owner_as_unattributed_and_does_nothing_when_off(recorder):
    import worker

    async def run(ctx):
        await worker._bind_usage_attempt(ctx)
        return uc.current_attempt_context()

    context = asyncio.run(run({"job_id": "cron-1", "redis": _Redis({})}))
    assert context.tenant_id == uc.UNATTRIBUTED_TENANT
    uc.set_usage_recorder(None)
    assert asyncio.run(run({"job_id": "cron-2", "redis": _Redis({})})) is None


def test_a_redis_failure_never_fails_the_job(recorder):
    import worker

    class Broken:
        async def get(self, key):
            raise RuntimeError("redis down")

    asyncio.run(worker._bind_usage_attempt({"job_id": "j", "redis": Broken()}))  # no raise


# -- Ask ------------------------------------------------------------------------------------------------


def _service(results):
    svc = MagicMock(name="MemoryService")
    svc.search.return_value = results
    svc.keyword_search.return_value = ([], False)
    return svc


def _evidence():
    return [MemoryResponse(id="m1", memory="the sky is blue", category="decision", source="vector",
                           created_at="2026-07-01T00:00:00+00:00", score=0.9)]


def test_an_ask_emits_content_free_usage_events_for_the_person(recorder, monkeypatch):
    """Before: Ask never reached the capture seam (its client was not wrapped and no attempt was bound)."""
    from extensions.conversation_compiler import compile as compiler
    import ask

    answer = json.dumps({"action": "answer", "answer": "blue [m1]", "citations": ["m1"], "abstained": False})
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(
        text=answer,
        usage_metadata=SimpleNamespace(prompt_token_count=120, candidates_token_count=9,
                                       cached_content_token_count=None, thoughts_token_count=None),
    )
    monkeypatch.setattr(compiler.genai, "Client", lambda **kw: client)

    out = asyncio.run(ask.ask_memory(_service(_evidence()), question="what colour is the sky?",
                                     user_id="alice", reasoning_level="minimal"))
    assert out["answer"].startswith("blue")

    assert recorder.events, "the Ask call recorded no usage event"
    event = recorder.events[-1]
    assert event.tenant_id == "alice" and event.operation == "llm_ask"
    assert event.usage.input_tokens.value == 120 and event.usage.output_tokens.value == 9
    blob = event.model_dump_json()
    assert "the sky is blue" not in blob and "what colour" not in blob  # content-free


def test_an_ask_records_nothing_when_usage_capture_is_off(monkeypatch):
    from extensions.conversation_compiler import compile as compiler
    import ask

    assert uc.get_usage_recorder() is None
    client = MagicMock()
    client.models.generate_content.return_value = _response()
    client.models.generate_content.return_value.text = json.dumps(
        {"action": "answer", "answer": "x", "citations": [], "abstained": True})
    monkeypatch.setattr(compiler.genai, "Client", lambda **kw: client)
    asyncio.run(ask.ask_memory(_service(_evidence()), question="q", user_id="alice", reasoning_level="minimal"))
    assert uc.current_attempt_context() is None


def test_the_shared_gemini_path_labels_calls_by_operation(recorder, monkeypatch):
    from extensions.conversation_compiler import compile as compiler

    client = MagicMock()
    client.models.generate_content.return_value = _response()
    monkeypatch.setattr(compiler.genai, "Client", lambda **kw: client)
    with uc.ensure_attempt("alice", "compile"):
        compiler._call_gemini("prompt")  # outside Ask: its own label
        with uc.operation_scope("llm_ask"):
            compiler._call_gemini("prompt")
    assert [e.operation for e in recorder.events] == ["llm_compile", "llm_ask"]


# -- opening the store -----------------------------------------------------------------------------------


def test_install_usage_capture_installs_a_durable_recorder_and_uninstall_removes_it(tmp_path):
    from config import Settings
    from usage_store import install_usage_capture, uninstall_usage_capture

    off = Settings(usage_store_enabled=False)
    assert install_usage_capture(off) is None and uc.get_usage_recorder() is None

    on = Settings(usage_store_enabled=True, usage_store_backend="sqlite",
                  usage_store_sqlite_path=str(tmp_path / "usage.sqlite3"))
    store = install_usage_capture(on)
    try:
        assert isinstance(uc.get_usage_recorder(), uc.DurableUsageRecorder)
        with uc.ensure_attempt("alice", "search"):
            uc.record_generation_usage(_response(), operation="x", model="m")
        assert [e.tenant_id for e in store.events("alice")] == ["alice"]
    finally:
        uninstall_usage_capture(store)
    assert uc.get_usage_recorder() is None


def test_defaults_record_nothing_and_open_no_store():
    from config import Settings

    s = Settings()
    assert s.usage_store_enabled is False and s.inference_facade_enabled is False
    assert s.graph_support_aware_removal_enabled is False and s.ns_gemini_base_url == ""
