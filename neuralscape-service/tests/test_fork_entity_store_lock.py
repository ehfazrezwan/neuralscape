"""mem0's lazy ``<collection>_entities`` store: one create, tolerant of a lost race.

``Memory.entity_store`` / ``AsyncMemory.entity_store`` build the side collection
on first access. Upstream had no lock, so N threads touching it together each
ran Qdrant's list-then-create (N creates, the losers get 409). The fork patch
(``# NEURALSCAPE PATCH`` in mem0/mem0/memory/main.py) serialises the creation
and treats Qdrant's "already exists" 409 (and the not-yet-readable 500) from
another process as success, like ``memory.core._memory_from_config``.

The integration counterpart, against real Qdrant, is
integration_tests/test_fork_smoke.py.
"""

import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

import mem0.memory.main as mem0_main

THREADS = 12
_real_sleep = time.sleep


def _unexpected(status, body):
    return UnexpectedResponse(
        status_code=status, reason_phrase="x", content=body, headers=httpx.Headers()
    )


class FakeFactory:
    """``VectorStoreFactory`` stand-in.  Each ``create`` is a slow list-then-create."""

    def __init__(self, fail_first=()):
        self.lock = threading.Lock()
        self.creates = 0
        self.calls = 0
        self.fail_first = list(fail_first)

    def create(self, provider, config):
        with self.lock:
            self.calls += 1
            failure = self.fail_first.pop(0) if self.fail_first else None
        if failure is not None:
            raise failure
        _real_sleep(0.05)  # widen the window between "not there" and "created"
        with self.lock:
            self.creates += 1
        return SimpleNamespace(collection=config["collection_name"])


def _memory(cls, factory, monkeypatch):
    monkeypatch.setattr(mem0_main, "VectorStoreFactory", factory)
    monkeypatch.setattr(time, "sleep", lambda s: None)  # the retry backoff, not FakeFactory's
    m = cls.__new__(cls)
    m.config = SimpleNamespace(
        vector_store=SimpleNamespace(provider="fake", config={"collection_name": "base"})
    )
    m.collection_name = "base"
    m.vector_store = SimpleNamespace()
    m._entity_store = None
    return m


def _touch_together(m, threads=THREADS):
    barrier = threading.Barrier(threads)
    stores, errors = [], []

    def touch():
        try:
            barrier.wait(timeout=10)
            stores.append(m.entity_store)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=touch) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    return stores, errors


@pytest.mark.parametrize("cls_name", ["Memory", "AsyncMemory"])
def test_concurrent_first_touch_creates_the_store_exactly_once(monkeypatch, cls_name):
    factory = FakeFactory()
    m = _memory(getattr(mem0_main, cls_name), factory, monkeypatch)
    stores, errors = _touch_together(m)
    assert errors == []
    assert len(stores) == THREADS
    assert factory.creates == 1
    assert all(s is stores[0] for s in stores)
    assert stores[0].collection == "base_entities"


@pytest.mark.parametrize("cls_name", ["Memory", "AsyncMemory"])
@pytest.mark.parametrize(
    "failure",
    [
        _unexpected(409, b'{"status":{"error":"Wrong input: Collection `base_entities` already exists!"}}'),
        _unexpected(500, b'{"status":{"error":"Service internal error: 0 of 0 read operations failed"}}'),
    ],
    ids=["409-already-exists", "500-not-yet-readable"],
)
def test_a_lost_create_race_in_another_process_is_success(monkeypatch, cls_name, failure):
    factory = FakeFactory(fail_first=[failure])
    m = _memory(getattr(mem0_main, cls_name), factory, monkeypatch)
    store = m.entity_store
    assert store.collection == "base_entities"
    assert factory.calls == 2  # the lost create, then building it again on the existing collection
    assert m.entity_store is store


def test_other_errors_are_raised_and_not_cached(monkeypatch):
    for failure in (
        _unexpected(409, b"Collection is being updated"),
        _unexpected(500, b"boom"),
        RuntimeError("Collection already exists"),
    ):
        factory = FakeFactory(fail_first=[failure])
        m = _memory(mem0_main.Memory, factory, monkeypatch)
        with pytest.raises(type(failure)):
            m.entity_store  # noqa: B018
        assert factory.calls == 1
        assert m._entity_store is None


def test_a_conflict_that_never_clears_is_bounded(monkeypatch):
    conflict = _unexpected(409, b"Collection `x` already exists!")
    factory = FakeFactory(fail_first=[conflict] * 100)
    m = _memory(mem0_main.Memory, factory, monkeypatch)
    with pytest.raises(UnexpectedResponse):
        m.entity_store  # noqa: B018
    assert factory.calls == mem0_main._ENTITY_STORE_ATTEMPTS
