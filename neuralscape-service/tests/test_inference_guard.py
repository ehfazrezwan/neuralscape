"""Event-controlled bounded latency, circuit recovery and late-result safety."""
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from inference_guard import InferenceGuard


def test_timeout_and_saturation_do_not_enqueue_unbounded_work():
    release = threading.Event()
    events = []
    guard = InferenceGuard(capacity=1)
    try:
        def blocked():
            release.wait(2)
            return 'late', False
        assert guard.run(blocked, timeout=.01, emit=events.append) is None
        other = Mock()
        assert guard.run(other, timeout=.01, emit=events.append) is None
        other.assert_not_called()
        assert [e['fallback_reason'] for e in events] == ['deadline', 'capacity']
        assert events[0]['usage_pending'] is True
    finally:
        release.set()
        guard.close()


def test_circuit_opens_only_on_provider_failure_and_recovers():
    now = [0]
    guard = InferenceGuard(failures=2, cooldown=10, clock=lambda: now[0])
    emit = Mock()
    try:
        # A semantic abstention is a healthy provider response.
        assert guard.run(lambda: (None, False), timeout=1, emit=emit) is None
        for _ in range(2):
            guard.run(lambda: (None, True), timeout=1, emit=emit)
        unused = Mock()
        assert guard.run(unused, timeout=1, emit=emit) is None
        unused.assert_not_called()
        assert emit.call_args.args[0]['fallback_reason'] == 'circuit_open'
        now[0] = 11
        assert guard.run(lambda: ('healthy', False), timeout=1, emit=emit) == 'healthy'
        assert guard.run(lambda: ('healthy', False), timeout=1, emit=emit) == 'healthy'
    finally:
        guard.close()


def test_half_open_admits_only_one_probe():
    now, entered, release = [0], threading.Event(), threading.Event()
    guard = InferenceGuard(failures=1, cooldown=1, clock=lambda: now[0])
    try:
        guard.run(lambda: (None, True), timeout=1, emit=lambda e: None)
        now[0] = 2
        def probe():
            entered.set()
            release.wait(2)
            return 'ok', False
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(guard.run, probe, timeout=2, emit=lambda e: None)
            assert entered.wait(1)
            emit, unused = Mock(), Mock()
            assert guard.run(unused, timeout=1, emit=emit) is None
            unused.assert_not_called()
            assert emit.call_args.args[0]['fallback_reason'] == 'circuit_open'
            release.set()
            assert first.result() == 'ok'
    finally:
        release.set()
        guard.close()


def test_late_extraction_cannot_publish_evidence(monkeypatch):
    import hybrid_inference as hybrid
    from config import settings
    from graphiti_core.llm_client.hybrid_decision_client import HybridDecisions
    release = threading.Event()
    guard = InferenceGuard(capacity=1)
    monkeypatch.setattr(settings, 'hybrid_extraction_enabled', True)
    monkeypatch.setattr(settings, 'hybrid_inference_api_key', 'test')
    monkeypatch.setattr(settings, 'hybrid_extraction_deadline_s', .01)
    monkeypatch.setattr(hybrid, '_extraction_guard', lambda *a: guard)
    monkeypatch.setattr(hybrid, 'decisions', lambda **kw: HybridDecisions('test', endpoint='https://local.test/decide', model='test-model', **kw))
    def delayed(messages, decision, *, category_evidence):
        release.wait(2)
        category_evidence['late'] = 'must not publish'
        return [('preference', 'late')]
    monkeypatch.setattr('source_extraction.select_facts', delayed)
    output = {}
    try:
        assert hybrid.extract_source_facts([], category_evidence=output) is None
        assert output == {}
    finally:
        release.set()
        guard.close()
    assert output == {}


def test_warmup_is_disabled_by_default_and_local_only(monkeypatch):
    import hybrid_inference as hybrid
    from config import settings
    tokenizer, client = Mock(), Mock()
    monkeypatch.setattr('source_extraction._segmenter', tokenizer)
    monkeypatch.setattr('graphiti_core.llm_client.hybrid_decision_client._http_client', client)
    monkeypatch.setattr(settings, 'hybrid_extraction_enabled', False)
    hybrid.warmup_extraction()
    tokenizer.assert_not_called()
    monkeypatch.setattr(settings, 'hybrid_extraction_enabled', True)
    hybrid.warmup_extraction()
    tokenizer.assert_called_once_with()
    client.assert_called_once_with()
    client.return_value.post.assert_not_called()


def test_concurrent_callers_never_exceed_capacity_and_every_caller_gets_a_clean_outcome():
    """Hammer one guard from many threads: bounded in-flight work, no exceptions."""
    capacity = 3
    guard = InferenceGuard(capacity=capacity, failures=10 ** 6)
    lock, running, peak, outcomes = threading.Lock(), [0], [0], []

    def op():
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        threading.Event().wait(.01)
        with lock:
            running[0] -= 1
        return 'ok', False

    try:
        with ThreadPoolExecutor(max_workers=24) as pool:
            for result in pool.map(lambda _: guard.run(op, timeout=2, emit=lambda e: None), range(200)):
                outcomes.append(result)
    finally:
        guard.close()
    assert peak[0] <= capacity
    assert set(outcomes) <= {'ok', None} and 'ok' in outcomes


def test_stale_epoch_success_cannot_close_a_circuit_a_probe_is_testing():
    """A slow call admitted before the circuit opened must not reset it later."""
    now, slow_started, release = [0], threading.Event(), threading.Event()
    guard = InferenceGuard(capacity=4, failures=2, cooldown=10, clock=lambda: now[0])
    try:
        def slow():
            slow_started.set()
            release.wait(2)
            return 'late-ok', False
        with ThreadPoolExecutor(max_workers=1) as pool:
            slow_call = pool.submit(guard.run, slow, timeout=3, emit=lambda e: None)
            assert slow_started.wait(1)
            for _ in range(2):
                guard.run(lambda: (None, True), timeout=1, emit=lambda e: None)
            emit, unused = Mock(), Mock()
            assert guard.run(unused, timeout=1, emit=emit) is None  # circuit is open
            release.set()
            assert slow_call.result() == 'late-ok'
            # The old success must not have closed the circuit.
            assert guard.run(unused, timeout=1, emit=emit) is None
            unused.assert_not_called()
            assert emit.call_args.args[0]['fallback_reason'] == 'circuit_open'
    finally:
        release.set()
        guard.close()


def test_failed_probe_reopens_for_a_full_cooldown():
    now = [0]
    guard = InferenceGuard(failures=1, cooldown=10, clock=lambda: now[0])
    try:
        guard.run(lambda: (None, True), timeout=1, emit=lambda e: None)
        now[0] = 11
        guard.run(lambda: (None, True), timeout=1, emit=lambda e: None)  # failed probe
        now[0] = 15  # inside the NEW cooldown (opened at 11)
        unused = Mock()
        assert guard.run(unused, timeout=1, emit=lambda e: None) is None
        unused.assert_not_called()
        now[0] = 22
        assert guard.run(lambda: ('ok', False), timeout=1, emit=lambda e: None) == 'ok'
    finally:
        guard.close()


def test_operation_exceptions_count_as_failures_and_release_the_probe():
    now = [0]
    guard = InferenceGuard(failures=1, cooldown=1, clock=lambda: now[0])
    try:
        with pytest.raises(RuntimeError):
            guard.run(Mock(side_effect=RuntimeError('boom')), timeout=1, emit=lambda e: None)
        now[0] = 2
        with pytest.raises(RuntimeError):  # probe raises: probe slot must be released
            guard.run(Mock(side_effect=RuntimeError('boom')), timeout=1, emit=lambda e: None)
        now[0] = 4
        assert guard.run(lambda: ('ok', False), timeout=1, emit=lambda e: None) == 'ok'
    finally:
        guard.close()
