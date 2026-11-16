"""Retention seen from the producers: replay inside the window, fresh work after it, payload deletion."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from contracts_lifecycle import IntentStatus
from ledger.port import IdempotencyConflict
from ledger.retention import DAY, RetentionPolicy, RetentionSweeper
from ledger.sqlite import SqliteLedger
from tests.test_v1_durable_write import PAYLOAD, FakeService, run, worker
from write_producer import DEFAULT_TENANT, DurableWriteProducer, SqlitePayloadStore, is_intent_id


@pytest.fixture
def env(tmp_path):
    ledger = SqliteLedger(str(tmp_path / "ledger.db"))
    payloads = SqlitePayloadStore(str(tmp_path / "ledger.db.payloads"))
    producer = DurableWriteProducer(ledger, payloads)
    ledger.test_path = str(tmp_path / "ledger.db")  # type: ignore[attr-defined]

    def age(seconds):
        raw = sqlite3.connect(str(tmp_path / "ledger.db"), isolation_level=None)
        for table, column in (("intents", "updated_at"), ("stage_receipts", "recorded_at")):
            raw.execute(f"UPDATE {table} SET {column} = {column} - ?", (seconds,))
        raw.close()

    yield ledger, payloads, producer, age
    payloads.close()
    ledger.close()


def applied(env, key):
    ledger, payloads, producer, _ = env
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD, idempotency_key=key)
    run(worker((ledger, payloads, producer, None), FakeService()).drain())
    assert ledger.get_intent(DEFAULT_TENANT, accepted.intent_id).status is IntentStatus.APPLIED
    return accepted


def test_a_replay_inside_the_window_is_the_original_and_after_it_is_a_new_intent(env) -> None:
    ledger, payloads, producer, age = env
    first = applied(env, "client-key")
    age(6 * DAY)
    sweeper = RetentionSweeper(ledger, RetentionPolicy(), payloads=payloads)
    assert sweeper.sweep(DEFAULT_TENANT).idempotency_released == 0
    replay = producer.accept_raw(user_id="user-1", payload=PAYLOAD, idempotency_key="client-key")
    assert (replay.intent_id, replay.created) == (first.intent_id, False)
    with pytest.raises(IdempotencyConflict):
        producer.accept_raw(user_id="user-1", payload={**PAYLOAD, "content": "different"}, idempotency_key="client-key")
    age(2 * DAY)
    assert sweeper.sweep(DEFAULT_TENANT).idempotency_released == 1
    # a changed request under the freed key is simply new work, not a conflict
    changed = {**PAYLOAD, "content": "different"}
    again = producer.accept_raw(user_id="user-1", payload=changed, idempotency_key="client-key")
    assert again.created is True and again.intent_id != first.intent_id and is_intent_id(again.intent_id)
    # the new generation replays like any other key, and conflicts with a different request
    replay = producer.accept_raw(user_id="user-1", payload=changed, idempotency_key="client-key")
    assert (replay.intent_id, replay.created) == (again.intent_id, False)
    with pytest.raises(IdempotencyConflict):
        producer.accept_raw(user_id="user-1", payload=PAYLOAD, idempotency_key="client-key")
    run(worker((ledger, payloads, producer, None), FakeService()).drain())
    assert ledger.get_intent(DEFAULT_TENANT, again.intent_id).status is IntentStatus.APPLIED


def test_the_payload_content_is_deleted_with_the_intents_blobs_after_the_content_window(env) -> None:
    ledger, payloads, producer, age = env
    first = applied(env, "k1")
    payloads.put(f"{first.intent_id}:delete:1", {"result": "blob"})
    assert payloads.get(first.intent_id)["content"] == PAYLOAD["content"]
    age(40 * DAY)
    report = RetentionSweeper(ledger, RetentionPolicy(), payloads=payloads).sweep(DEFAULT_TENANT)
    assert report.content_compacted == 1 and report.payloads_deleted >= 2  # the payload, its attempt blobs and markers
    assert payloads.get(first.intent_id) is None and payloads.get(f"{first.intent_id}:delete:1") is None
    assert ledger.get_intent(DEFAULT_TENANT, first.intent_id).status is IntentStatus.APPLIED


def test_other_intents_payloads_survive_a_prefix_like_id(env) -> None:
    ledger, payloads, producer, age = env
    payloads.put("intent-abc", {"x": 1})
    payloads.put("intent-abcdef", {"x": 2})
    payloads.put("intent-abc:vector-write-started:1", {})
    assert payloads.delete_intent("intent-abc") == 2
    assert payloads.get("intent-abcdef") == {"x": 2}


def test_concurrent_producer_replays_during_pruning_never_fail_and_converge(env) -> None:
    ledger, payloads, producer, age = env
    first = applied(env, "hot-key")
    age(8 * DAY)
    errors, ids = [], []
    gate = threading.Barrier(5)

    def replay():
        try:
            gate.wait()
            for _ in range(15):
                # one connection per thread, like separate API workers
                other = DurableWriteProducer(SqliteLedger(ledger.test_path), payloads)
                ids.append(other.accept_raw(user_id="user-1", payload=PAYLOAD, idempotency_key="hot-key").intent_id)
                other.ledger.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def prune():
        try:
            gate.wait()
            RetentionSweeper(ledger, RetentionPolicy(), payloads=payloads).sweep(DEFAULT_TENANT)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=replay) for _ in range(4)] + [threading.Thread(target=prune)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    new_generations = set(ids) - {first.intent_id}
    assert len(new_generations) <= 1  # the original id, then at most one fresh generation: no fan-out of duplicates
    assert ledger.get_intent(DEFAULT_TENANT, first.intent_id).intent.idempotency_key.startswith("retired:")
