"""Durable usage correlated to ledger attempts, end to end (no Redis needed).

A real SQLite lifecycle ledger, a real ``LedgerWriteWorker`` and a real SQLite
usage store, with a mocked provider that returns Gemini-shaped usage metadata
(cached and reasoning counters included).  The extraction call goes through the
real retry/fallback seam, so every provider attempt is captured with the ledger
claim's tenant, intent, stage and attempt.  The API -> Redis -> ARQ leg is in
``test_usage_end_to_end_redis.py`` (skipped without ``V1_TEST_REDIS_URL``).

Consumer metering is not collected (owner consent pending); only the service
ledger is itemised and the evaluation ledger stays separate.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import usage_capture as uc
from contracts_usage_reconcile import reconcile_usage_events
from ledger.sqlite import SqliteLedger
from memory.retry import retry_transient
from usage_store import SqliteUsageStore, task_rollup
from write_producer import (
    DEFAULT_TENANT,
    DurableWriteProducer,
    LedgerWriteWorker,
    SqlitePayloadStore,
)

RAW = dict(
    content="prefers dark mode", scope="global", project_id=None, category="preference",
    tags=None, agent_id=None, run_id=None, v2_extras={},
)


def _response(prompt=100, output=20, cached=30, reasoning=10):
    return SimpleNamespace(
        text="ok",
        model_version="gemini-test",
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=output,
            cached_content_token_count=cached,
            thoughts_token_count=reasoning,
            total_token_count=prompt + output + reasoning,
        ),
    )


class Mem:
    def __init__(self, memory_id: str) -> None:
        self.id, self.visibility, self.occurred_at = memory_id, "private", None


class ExtractingService:
    """Fake memory service whose vector stage calls a scripted provider."""

    def __init__(self, script, on_call=None) -> None:
        self.script = list(script)  # per ledger attempt: list of outcomes
        self.calls = 0
        self.on_call = on_call

    def generate_content(self, model=None, contents=None):
        self.calls += 1
        if self.on_call is not None:
            self.on_call()
        item = self.outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def store_raw(self, **kw):
        self.outcomes = list(self.script.pop(0))
        retry_transient(
            self.generate_content, model="primary", contents="x", max_retries=1, base_delay=0,
            max_delay=0, operation="LLM extraction", fallback_model="fallback",
        )
        return [Mem("mem-1")], True

    def get_memory(self, memory_id):
        return Mem(memory_id)

    def enrich_graph(self, **kw):
        return True


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr("memory.retry.time.sleep", lambda s: None)


@pytest.fixture
def env(tmp_path):
    ledger = SqliteLedger(str(tmp_path / "ledger.db"))
    payloads = SqlitePayloadStore(str(tmp_path / "ledger.db.payloads"))
    store = SqliteUsageStore(str(tmp_path / "usage.db"))
    previous = uc.set_usage_recorder(uc.DurableUsageRecorder(store))
    producer = DurableWriteProducer(ledger, payloads)
    yield SimpleNamespace(ledger=ledger, payloads=payloads, store=store, producer=producer, path=tmp_path)
    uc.set_usage_recorder(previous)
    store.close()
    payloads.close()
    ledger.close()


def accept(env) -> str:
    return env.producer.accept_raw(user_id="user-1", payload=dict(RAW, user_id="user-1")).intent_id


def drain(env, service, **kw):
    kw.setdefault("retry_delay", lambda attempt: 0.0)
    kw.setdefault("lease_seconds", 30.0)
    worker = LedgerWriteWorker(env.ledger, env.payloads, service, owner="w1", **kw)
    return asyncio.run(worker.drain())


def test_retry_fallback_and_a_failed_ledger_attempt_are_each_correlated(env) -> None:
    intent_id = accept(env)
    fail = RuntimeError("503 UNAVAILABLE")
    service = ExtractingService(
        [
            [fail, fail, fail],  # ledger attempt 1: primary x2 + fallback all fail
            [fail, fail, _response()],  # ledger attempt 2: primary x2 fail, fallback ok
        ]
    )
    drain(env, service)

    events = env.store.events(DEFAULT_TENANT, intent_id)
    assert {e.tenant_id for e in events} == {DEFAULT_TENANT}
    assert {e.intent_id for e in events} == {intent_id}
    by_attempt = env.store.ledger_attempts(DEFAULT_TENANT, intent_id)
    assert set(by_attempt.values()) == {1, 2}

    rollup = task_rollup(env.store, DEFAULT_TENANT, intent_id)
    attempts = rollup["attempts"]
    # 3 + 3 provider attempts, each its own stream, none lost.
    assert len(attempts) == 6
    assert len({a["attempt_id"] for a in attempts}) == 6
    assert sorted(a["ledger_attempt"] for a in attempts) == [1, 1, 1, 2, 2, 2]
    failed = [a for a in attempts if a["attempt_outcome"] == "failed"]
    assert len(failed) == 5 and all(a["total_tokens"] is None for a in failed)
    assert all(a["tokens"] is None and a["usage_missing_reason"] for a in failed)  # unknown, not zero

    (done,) = [a for a in attempts if a["attempt_outcome"] == "completed"]
    # The third provider call of the ledger attempt is the fallback; the event
    # carries the model the provider reports having served.
    assert done["ledger_attempt"] == 2 and done["attempt_id"].endswith(":vector:2.3")
    assert done["model"] == "gemini-test"
    tokens = done["tokens"]
    assert tokens["input_tokens"]["value"] == 100 and tokens["output_tokens"]["value"] == 20
    assert tokens["cache_read_tokens"]["value"] == 30 and tokens["reasoning_tokens"]["value"] == 10
    # Cached tokens are inside input; reasoning is additional to output: 100 + 20 + 10.
    assert done["total_tokens"] == rollup["service"]["known_token_subtotal"] == 130

    # Failed attempts are unknown, so the ledger total is honest about it.
    assert rollup["service"]["total_tokens"] is None
    assert len(rollup["service"]["incomplete_attempt_ids"]) == 5


def test_rollup_equals_reconcile_over_the_stored_rows(env) -> None:
    intent_id = accept(env)
    service = ExtractingService([[_response()]])
    drain(env, service)

    stored = env.store.reconcile(DEFAULT_TENANT, intent_id)
    assert stored == reconcile_usage_events(env.store.events(DEFAULT_TENANT, intent_id))
    rollup = task_rollup(env.store, DEFAULT_TENANT, intent_id)
    service_ledger = {ledger.ledger: ledger for ledger in stored.ledgers}["service"]
    assert rollup["service"]["total_tokens"] == service_ledger.total_tokens == 130
    assert rollup["service"]["coverage"] == service_ledger.coverage
    assert [a["head_event_id"] for a in rollup["attempts"]] == [
        s.head_event_id for s in stored.streams if s.ledger == "service"
    ]
    # Consumer metering is not collected; the evaluation ledger is not summed in.
    assert rollup["other_ledgers"]["consuming_agent"]["coverage"] == "not_collected"
    assert rollup["other_ledgers"]["evaluation"]["coverage"] != "complete"


def test_duplicate_and_late_events_are_counted_once(env) -> None:
    intent_id = accept(env)
    drain(env, ExtractingService([[_response()]]))
    before = task_rollup(env.store, DEFAULT_TENANT, intent_id)

    for event in env.store.events(DEFAULT_TENANT, intent_id):  # an exact replay
        assert env.store.append(event) is False
    assert task_rollup(env.store, DEFAULT_TENANT, intent_id) == before
    assert before["service"]["total_tokens"] == 130


def test_unknown_task_has_no_rollup(env) -> None:
    assert task_rollup(env.store, DEFAULT_TENANT, "no-such-intent") is None
    assert task_rollup(env.store, "other-tenant", accept(env)) is None


def test_default_off_binds_nothing(env, monkeypatch) -> None:
    """With no recorder installed the worker binds no attempt context."""
    uc.set_usage_recorder(None)
    seen = []
    service = ExtractingService([[_response()]], on_call=lambda: seen.append(uc.current_attempt_context()))
    intent_id = accept(env)
    drain(env, service)
    assert seen == [None]
    assert env.store.events(DEFAULT_TENANT, intent_id) == ()


def _child(root: str, flag: str) -> None:
    ledger = SqliteLedger(root + "/ledger.db")
    payloads = SqlitePayloadStore(root + "/ledger.db.payloads")
    store = SqliteUsageStore(root + "/usage.db")
    uc.set_usage_recorder(uc.DurableUsageRecorder(store))

    def block():
        Path(flag).write_text("in provider call")
        time.sleep(120)

    service = ExtractingService([[_response()]], on_call=block)
    worker = LedgerWriteWorker(ledger, payloads, service, owner="child", lease_seconds=1.0)
    asyncio.run(worker.drain())


def test_a_worker_killed_mid_call_does_not_double_count(env) -> None:
    intent_id = accept(env)
    flag = str(env.path / "blocked")
    child = multiprocessing.get_context("spawn").Process(target=_child, args=(str(env.path), flag))
    child.start()
    deadline = time.monotonic() + 30
    while not Path(flag).exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert Path(flag).exists(), "child never reached the provider call"
    os.kill(child.pid, signal.SIGKILL)
    child.join(10)
    assert child.exitcode == -signal.SIGKILL

    # The killed call left only its pending event: visible, usage unknown.
    killed = task_rollup(env.store, DEFAULT_TENANT, intent_id)
    (orphan,) = killed["attempts"]
    assert orphan["status"] == "pending" and orphan["total_tokens"] is None
    assert orphan["ledger_attempt"] == 1

    survivor = ExtractingService([[_response()]])
    deadline = time.monotonic() + 15  # the dead worker's 1s lease must expire first
    while drain(env, survivor) == 0 and time.monotonic() < deadline:
        time.sleep(0.3)
    assert survivor.calls == 1

    rollup = task_rollup(env.store, DEFAULT_TENANT, intent_id)
    assert rollup["service"]["known_token_subtotal"] == 130  # one completed call, counted once
    completed = [a for a in rollup["attempts"] if a["attempt_outcome"] == "completed"]
    assert len(completed) == 1 and completed[0]["ledger_attempt"] == 2
    assert orphan["attempt_id"] != completed[0]["attempt_id"]  # the retry is its own attempt
    assert rollup["service"]["total_tokens"] is None  # the orphan stays unknown, never zero
