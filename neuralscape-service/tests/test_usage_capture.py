"""Tests for the provider-usage normalizer and recorder seam."""

from types import SimpleNamespace

import pytest

import usage_capture as uc
from contracts_common import MAX_SAFE_INTEGER
from contracts_usage_reconcile import reconcile_usage_events

CTX = uc.AttemptContext(tenant_id="t1", task_id="task1", attempt_id="a1")


def _meta(**kw):
    base = dict(
        prompt_token_count=None,
        candidates_token_count=None,
        cached_content_token_count=None,
        thoughts_token_count=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _response(meta, model_version=None):
    return SimpleNamespace(usage_metadata=meta, model_version=model_version, text="x")


@pytest.fixture()
def recorder():
    rec = uc.InMemoryUsageRecorder()
    previous = uc.set_usage_recorder(rec)
    yield rec
    uc.set_usage_recorder(previous)


class TestNormalizer:
    def test_absent_metadata_is_none(self):
        assert uc.normalize_gemini_usage(None) is None

    def test_values_and_conventions(self):
        u = uc.normalize_gemini_usage(
            _meta(
                prompt_token_count=100,
                candidates_token_count=20,
                cached_content_token_count=40,
                thoughts_token_count=7,
            )
        )
        assert u.measurement == "actual"
        assert (u.input_tokens.value, u.output_tokens.value) == (100, 20)
        assert u.cache_read_tokens.value == 40
        assert u.reasoning_tokens.value == 7
        assert u.cache_read_inclusion == "included_in_input"
        assert u.reasoning_inclusion == "additional_to_output"
        assert u.cache_write_tokens.missing_reason == "not_applicable"

    def test_missing_is_unreported_not_zero(self):
        u = uc.normalize_gemini_usage(_meta(prompt_token_count=5))
        assert u.output_tokens.value is None
        assert u.output_tokens.missing_reason == "provider_did_not_report"
        assert u.reasoning_tokens.value is None

    def test_explicit_zero_is_measured_zero(self):
        u = uc.normalize_gemini_usage(_meta(prompt_token_count=0, thoughts_token_count=0))
        assert u.input_tokens.value == 0 and u.input_tokens.missing_reason is None
        assert u.reasoning_tokens.value == 0

    @pytest.mark.parametrize(
        "bad", [-1, True, 1.5, "3", MAX_SAFE_INTEGER + 1, object()]
    )
    def test_malformed_counter_is_missing(self, bad):
        u = uc.normalize_gemini_usage(_meta(prompt_token_count=bad))
        assert u.input_tokens.value is None
        assert u.input_tokens.missing_reason == "provider_did_not_report"

    def test_max_safe_counter_accepted(self):
        u = uc.normalize_gemini_usage(_meta(prompt_token_count=MAX_SAFE_INTEGER))
        assert u.input_tokens.value == MAX_SAFE_INTEGER


class TestRecorder:
    def test_default_is_noop(self):
        uc.set_usage_recorder(None)
        with uc.attempt_context(CTX):
            uc.record_generation_usage(_response(_meta()), operation="op", model="m")

    def test_no_attempt_context_records_nothing(self, recorder):
        uc.record_generation_usage(_response(_meta(prompt_token_count=1)), operation="op", model="m")
        assert recorder.events == []

    def test_records_final_event_and_reconciles(self, recorder):
        resp = _response(
            _meta(prompt_token_count=10, candidates_token_count=3, thoughts_token_count=2),
            model_version="gemini-x-001",
        )
        with uc.attempt_context(CTX):
            uc.record_generation_usage(resp, operation="llm_extraction", model="gemini-x")
        (event,) = recorder.events
        assert event.status == "final" and event.ledger == "service"
        assert event.attribution.model == "gemini-x-001"
        assert event.attribution.recipients is None
        assert event.attribution.recipients_missing_reason == "dependency_unavailable"
        assert event.attribution.bill_owner_missing_reason == "dependency_unavailable"
        assert event.task_id == "task1" and event.attempt_id == "a1.1"
        result = reconcile_usage_events(recorder.events)
        assert result is not None

    def test_absent_metadata_records_unavailable(self, recorder):
        with uc.attempt_context(CTX):
            uc.record_generation_usage(_response(None), operation="op", model=None)
        (event,) = recorder.events
        assert event.status == "unavailable" and event.usage is None
        assert event.usage_missing_reason == "provider_did_not_report"
        assert event.attribution.model is None
        assert event.attribution.model_missing_reason == "provider_did_not_report"

    def test_recorder_failure_never_raises(self):
        class Boom:
            def record(self, event):
                raise RuntimeError("boom")

        previous = uc.set_usage_recorder(Boom())
        try:
            with uc.attempt_context(CTX):
                uc.record_generation_usage(
                    _response(_meta(prompt_token_count=1)), operation="op", model="m"
                )
        finally:
            uc.set_usage_recorder(previous)

    def test_invalid_context_never_raises(self, recorder):
        with uc.attempt_context(uc.AttemptContext(tenant_id="", task_id="t", attempt_id="a")):
            uc.record_generation_usage(_response(_meta()), operation="op", model="m")
        assert recorder.events == []

    def test_context_is_restored(self):
        with uc.attempt_context(CTX):
            assert uc._attempt.get().context is CTX
        assert uc._attempt.get() is None


class TestOpenAIShape:
    def test_chat_shape_inclusion_rules(self):
        usage = {
            "prompt_tokens": 50,
            "completion_tokens": 9,
            "total_tokens": 59,
            "prompt_tokens_details": {"cached_tokens": 30},
            "completion_tokens_details": SimpleNamespace(reasoning_tokens=4),
        }
        u = uc.normalize_openai_usage(usage)
        assert (u.input_tokens.value, u.output_tokens.value) == (50, 9)
        assert u.cache_read_tokens.value == 30 and u.reasoning_tokens.value == 4
        assert u.cache_read_inclusion == "included_in_input"
        assert u.reasoning_inclusion == "included_in_output"

    def test_responses_shape(self):
        u = uc.normalize_openai_usage(SimpleNamespace(input_tokens=7, output_tokens=2))
        assert (u.input_tokens.value, u.output_tokens.value) == (7, 2)
        assert u.cache_read_tokens.missing_reason == "provider_did_not_report"

    def test_missing_details_unreported_and_zero_measured(self):
        u = uc.normalize_openai_usage({"prompt_tokens": 0, "completion_tokens": 1})
        assert u.input_tokens.value == 0
        assert u.reasoning_tokens.missing_reason == "provider_did_not_report"

    def test_absent_usage_is_none(self):
        assert uc.normalize_openai_usage(None) is None

    def test_embedding_has_only_input(self):
        u = uc.normalize_openai_usage({"prompt_tokens": 12, "total_tokens": 12}, "embedding")
        assert u.input_tokens.value == 12
        assert u.output_tokens.missing_reason == "not_applicable"
        assert u.reasoning_inclusion == "not_applicable"


class TestResponseShapes:
    def test_openai_response_event(self, recorder):
        resp = SimpleNamespace(model="gw-model", usage={"prompt_tokens": 3, "completion_tokens": 1})
        with uc.attempt_context(CTX):
            uc.record_provider_usage(resp, operation="op", model="requested")
        (event,) = recorder.events
        assert event.attribution.provider == "openai_compatible"
        assert event.attribution.model == "gw-model"
        assert event.usage.input_tokens.value == 3

    def test_gemini_embedding_without_counts_is_unavailable_not_zero(self, recorder):
        resp = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1])])
        with uc.attempt_context(CTX):
            uc.record_provider_usage(resp, operation="op", model="m", kind="embedding")
        (event,) = recorder.events
        assert event.status == "unavailable" and event.usage is None
        assert event.attribution.provider == "google"

    def test_operation_scope_nests_and_restores(self):
        assert uc.current_operation("d") == "d"
        with uc.operation_scope("x"):
            assert uc.current_operation("d") == "x"
        assert uc.current_operation("d") == "d"


class TestMultipleCallsOneContext:
    """Several provider calls under one bound context must reconcile."""

    def test_two_windows_reconcile(self, recorder):
        resp = _response(_meta(prompt_token_count=10, candidates_token_count=3))
        with uc.attempt_context(CTX):
            uc.record_generation_usage(resp, operation="llm_extraction", model="m")
            uc.record_generation_usage(
                _response(_meta(prompt_token_count=7, candidates_token_count=2)),
                operation="llm_extraction",
                model="m",
            )
        first, second = recorder.events
        assert first.attempt_id != second.attempt_id
        result = reconcile_usage_events(recorder.events)
        assert result is not None

    def test_ids_are_deterministic_per_context(self, recorder):
        for _ in range(2):
            with uc.attempt_context(CTX):
                uc.record_generation_usage(_response(_meta(prompt_token_count=1)), operation="o", model="m")
        assert [e.attempt_id for e in recorder.events] == ["a1.1", "a1.1"]

    def test_counter_shared_with_threads(self, recorder):
        import asyncio

        async def run():
            with uc.attempt_context(CTX):
                await asyncio.gather(
                    *[
                        asyncio.to_thread(
                            uc.record_generation_usage,
                            _response(_meta(prompt_token_count=1)),
                            operation="o",
                            model="m",
                        )
                        for _ in range(8)
                    ]
                )

        asyncio.run(run())
        assert len({e.attempt_id for e in recorder.events}) == 8

    def test_long_context_attempt_id_is_bounded(self, recorder):
        ctx = uc.AttemptContext(tenant_id="t", task_id="k", attempt_id="x" * 256)
        with uc.attempt_context(ctx):
            uc.record_generation_usage(_response(_meta(prompt_token_count=1)), operation="o", model="m")
        (event,) = recorder.events
        assert len(event.attempt_id) <= 256 and event.attempt_id.startswith("x" * 200)


class TestOverlongProviderStrings:
    def test_long_model_version_bounded_deterministic_not_dropped(self, recorder):
        long = "m" * 400
        with uc.attempt_context(CTX):
            uc.record_generation_usage(
                _response(_meta(prompt_token_count=2), model_version=long), operation="o", model="m"
            )
            uc.record_generation_usage(
                _response(_meta(prompt_token_count=2), model_version=long + "z"), operation="o", model="m"
            )
        a, b = recorder.events
        assert a.attribution.model != b.attribution.model  # distinct long values stay distinct
        assert len(a.attribution.model) <= 256 and a.attribution.model.startswith("m" * 200)
        assert a.status == "final" and a.usage.input_tokens.value == 2

    def test_long_operation_bounded(self, recorder):
        with uc.attempt_context(CTX):
            uc.record_generation_usage(_response(_meta(prompt_token_count=2)), operation="o" * 500, model="m")
        (event,) = recorder.events
        assert len(event.operation) <= 256

    def test_hostile_response_still_records_degraded_event(self, recorder):
        class Hostile:
            @property
            def usage_metadata(self):
                raise RuntimeError("no")

        with uc.attempt_context(CTX):
            uc.record_generation_usage(Hostile(), operation="o", model="m")
        (event,) = recorder.events
        assert event.status == "unavailable" and event.usage is None


class TestReportedCounterQuality:
    def test_all_counters_missing_is_unavailable(self, recorder):
        with uc.attempt_context(CTX):
            uc.record_generation_usage(_response(_meta()), operation="o", model="m")
        (event,) = recorder.events
        assert event.status == "unavailable" and event.usage is None
        assert event.usage_missing_reason == "provider_did_not_report"

    def test_cached_above_input_marked_unreported(self, recorder):
        with uc.attempt_context(CTX):
            uc.record_generation_usage(
                _response(_meta(prompt_token_count=1, candidates_token_count=1, cached_content_token_count=9)),
                operation="o",
                model="m",
            )
        (event,) = recorder.events
        assert event.usage.cache_read_tokens.value is None
        assert event.usage.cache_read_tokens.missing_reason == "provider_did_not_report"
        assert event.usage.input_tokens.value == 1

    def test_reasoning_above_output_openai_marked_unreported(self, recorder):
        resp = SimpleNamespace(
            model="gw",
            usage={
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        )
        with uc.attempt_context(CTX):
            uc.record_provider_usage(resp, operation="o", model="m")
        (event,) = recorder.events
        assert event.usage.reasoning_tokens.value is None

    def test_consistent_cached_kept(self, recorder):
        with uc.attempt_context(CTX):
            uc.record_generation_usage(
                _response(_meta(prompt_token_count=10, candidates_token_count=1, cached_content_token_count=4)),
                operation="o",
                model="m",
            )
        assert recorder.events[0].usage.cache_read_tokens.value == 4


def test_concurrent_gap_flushes_persist_each_unrecorded_event_once():
    import threading
    import time

    from usage_capture import DurableUsageRecorder

    class Store:
        def __init__(self):
            self.gaps = {}
            self.lock = threading.Lock()

        def record_gap(self, tenant_id, task_id, count):
            time.sleep(0.02)
            with self.lock:
                self.gaps[(tenant_id, task_id)] = self.gaps.get((tenant_id, task_id), 0) + count

    store = Store()
    recorder = DurableUsageRecorder(store)
    recorder._unrecorded[("t", "k")] = 3
    threads = [threading.Thread(target=recorder.flush_gaps) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.gaps == {("t", "k"): 3}
    assert recorder.unrecorded == {}


def test_failed_gap_flush_keeps_the_count_for_a_later_flush():
    from usage_capture import DurableUsageRecorder

    class Store:
        def __init__(self):
            self.fail = True
            self.gaps = 0

        def record_gap(self, tenant_id, task_id, count):
            if self.fail:
                raise RuntimeError("down")
            self.gaps += count

    store = Store()
    recorder = DurableUsageRecorder(store)
    recorder._unrecorded[("t", "k")] = 2
    recorder.flush_gaps()
    assert recorder.unrecorded == {("t", "k"): 2}
    recorder._unrecorded[("t", "k")] += 1  # a new failure while it was down
    store.fail = False
    recorder.flush_gaps()
    assert store.gaps == 3 and recorder.unrecorded == {}


def test_gap_recovery_on_the_provider_call_path_is_bounded_by_the_record_budget():
    """Before: one successful append flushed every pending task gap synchronously, so 1,000 pending tasks at
    100 ms a write held the provider-call thread for 100 s.  Counts the budget does not reach are kept."""
    from usage_capture import DurableUsageRecorder
    from tests.test_usage_store import event

    clock = {"now": 0.0}

    class Store:
        def __init__(self):
            self.gaps = {}

        def append(self, e, *, ledger_attempt=None):
            return True

        def record_gap(self, tenant_id, task_id, count):
            clock["now"] += 0.1  # each gap write costs 100 ms
            self.gaps[(tenant_id, task_id)] = count

    store = Store()
    recorder = DurableUsageRecorder(store, record_budget_s=1.0, monotonic=lambda: clock["now"])
    for i in range(1000):
        recorder._unrecorded[("t", f"task-{i}")] = 1
    assert recorder.record(event("e1")) is True
    assert 1 <= len(store.gaps) <= 11  # about one second of recovery, not a hundred
    assert len(store.gaps) + len(recorder.unrecorded) == 1000  # nothing dropped
    first = len(store.gaps)
    recorder.record(event("e2"))  # the next call carries on where this one stopped
    assert len(store.gaps) > first
    recorder.flush_gaps()  # an unbounded flush (shutdown) still drains the rest
    assert len(store.gaps) == 1000 and recorder.unrecorded == {}


def test_one_deadline_bounds_every_retry_and_the_gap_flush_of_a_record_call():
    """Before: each retry started a store write with the store's own fresh budget, so a store that waited its
    full 2 s on a held lock made ``record`` take a multiple of its 2.5 s budget."""
    from usage_capture import DurableUsageRecorder
    from tests.test_usage_store import event

    clock = {"now": 0.0}
    budgets = []

    class Store:
        append_budget = 2.0

        def append(self, e, *, ledger_attempt=None, budget_s=None):
            budgets.append(budget_s)
            clock["now"] += min(self.append_budget, budget_s)  # the held lock is waited out to the budget
            raise RuntimeError("database is locked")

        def record_gap(self, *a, **kw):
            raise AssertionError("no gap is flushed while the store is failing")

    def sleep(seconds):
        clock["now"] += seconds

    recorder = DurableUsageRecorder(
        Store(), retries=5, retry_delay=0.05, sleep=sleep, record_budget_s=2.5, monotonic=lambda: clock["now"]
    )
    assert recorder.record(event("e1")) is False
    assert clock["now"] <= 2.5 + 1e-9  # one deadline for the whole call
    assert budgets[0] == 2.5 and all(b <= 0.5 + 1e-9 for b in budgets[1:])


def test_a_gap_write_started_late_in_a_record_call_gets_only_the_remaining_budget():
    from usage_capture import DurableUsageRecorder
    from tests.test_usage_store import event

    clock = {"now": 0.0}
    gap_budgets = []

    class Store:
        def append(self, e, *, ledger_attempt=None, budget_s=None):
            clock["now"] += 2.0  # the append itself took 2 s
            return True

        def record_gap(self, tenant_id, task_id, count, *, budget_s=None):
            gap_budgets.append(budget_s)
            clock["now"] += min(2.0, budget_s)

    recorder = DurableUsageRecorder(Store(), record_budget_s=2.5, monotonic=lambda: clock["now"])
    recorder._unrecorded[("t", "a")] = 1
    recorder._unrecorded[("t", "b")] = 1
    assert recorder.record(event("e1")) is True
    assert gap_budgets == [0.5]  # the second gap is not started: the deadline passed
    assert clock["now"] <= 2.5 + 1e-9
    assert recorder.unrecorded == {("t", "b"): 1}  # kept for the next call


def test_the_sqlite_store_caps_a_write_at_the_callers_remaining_budget(tmp_path):
    import threading
    import time

    from usage_store import SqliteUsageStore, UsageStoreError
    from tests.test_usage_store import event

    store = SqliteUsageStore(str(tmp_path / "usage.db"), append_budget_s=2.0)
    held, release = threading.Event(), threading.Event()

    def hold():
        with store._lock:
            held.set()
            release.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    held.wait(2)
    try:
        started = time.monotonic()
        with pytest.raises(UsageStoreError, match="busy"):
            store.append(event("e1"), budget_s=0.1)
        with pytest.raises(UsageStoreError, match="busy"):
            store.record_gap("t", "k", 1, budget_s=0.1)
        assert time.monotonic() - started < 1.0  # not the store's own 2 s each
    finally:
        release.set()
        thread.join()


def test_claimed_gap_counts_are_never_dropped_when_concurrent_failures_fill_the_map():
    """Before: counts claimed by a flush that the store did not take were dropped (``gaps_overflowed``) when
    new failures had filled the bounded map meanwhile, losing their task attribution."""
    from usage_capture import DurableUsageRecorder

    class Store:
        def __init__(self):
            self.recorder = None

        def record_gap(self, tenant_id, task_id, count, **kw):
            # while the flush is writing, new failures fill the (cap = 1) map with another key
            self.recorder._unrecorded[("t", "new")] = 1
            raise RuntimeError("down")

    store = Store()
    recorder = DurableUsageRecorder(store, max_unrecorded_keys=1)
    store.recorder = recorder
    recorder._unrecorded[("t", "claimed")] = 4
    recorder.flush_gaps()
    assert recorder.unrecorded[("t", "claimed")] == 4
    assert recorder.gaps_overflowed == 0


def test_the_store_signature_is_read_once_per_method_not_on_every_write(monkeypatch):
    """Before: ``inspect.signature`` ran on every append and gap write on the provider-call path."""
    import inspect

    class Store:
        def append(self, event, *, budget_s=None, ledger_attempt=None):
            return budget_s

        def legacy(self, event):
            return "legacy"

    store = Store()
    reads = []
    real = inspect.signature
    monkeypatch.setattr(inspect, "signature", lambda m, *a, **k: reads.append(m) or real(m, *a, **k))
    call = uc.DurableUsageRecorder._store_call
    assert [call(store.append, "e", budget_s=0.5) for _ in range(5)] == [0.5] * 5
    assert call(store.legacy, "e", budget_s=0.5) == "legacy"  # a store without budget_s is not given one
    assert call(store.legacy, "e", budget_s=0.5) == "legacy"
    assert call(store.append, "e") is None  # no budget asked for: none passed
    assert len(reads) == 2  # one read per method, however many writes
