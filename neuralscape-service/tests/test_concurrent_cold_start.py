"""Concurrent cold start: processes racing to create the same Qdrant collection.

The API and the three workers initialise mem0 at the same time against an
empty Qdrant. mem0's Qdrant store lists the collections, sees its own missing
and creates it; a process that loses the race gets ``409 Collection already
exists``. That must count as a successful initialisation, not kill the worker.

The race is reproduced against mem0's real ``Qdrant.create_col`` with a fake
client: every process lists before any of them creates.
"""

import threading
from types import SimpleNamespace

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

import mem0
from config import settings
from memory.core import CoreMixin
from mem0.vector_stores.qdrant import Qdrant

COLLECTIONS = ("neuralscape_memories", "mem0migrations")


def _conflict(name):
    return UnexpectedResponse(
        status_code=409,
        reason_phrase="Conflict",
        content=f'{{"status":{{"error":"Wrong input: Collection `{name}` already exists!"}}}}'.encode(),
        headers=httpx.Headers(),
    )


class RacingQdrantServer:
    """Collections shared by all processes; each process's first listing is
    held until every process has listed, so they all see an empty server."""

    def __init__(self, processes):
        self.existing: set[str] = set()
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(processes)
        self.local = threading.local()
        self.creates = 0
        self.conflicts = 0

    def get_collections(self):
        with self.lock:
            names = list(self.existing)
        if not getattr(self.local, "listed", False):
            self.local.listed = True
            self.barrier.wait(timeout=10)
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in names])

    def create_collection(self, collection_name, **_):
        with self.lock:
            if collection_name in self.existing:
                self.conflicts += 1
                raise _conflict(collection_name)
            self.existing.add(collection_name)
            self.creates += 1

    def get_collection(self, name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(sparse_vectors={"bm25": {}})))

    def create_payload_index(self, **_):
        return None


class Service(CoreMixin):
    pass


def _fake_from_config(server):
    def from_config(config):
        # What Memory.from_config does to Qdrant at start: the memories
        # collection, then the migrations collection.
        stores = [Qdrant(collection_name=name, embedding_model_dims=8, client=server) for name in COLLECTIONS]
        return SimpleNamespace(vector_store=stores[0], _telemetry_vector_store=stores[1])

    return from_config


def _cold_start(monkeypatch, server, processes):
    monkeypatch.setattr(mem0.Memory, "from_config", _fake_from_config(server))
    monkeypatch.setattr(type(settings), "get_mem0_config", lambda self: {})
    errors, ready = [], []

    def boot():
        try:
            ready.append(Service()._get_memory())
        except Exception as exc:  # noqa: BLE001 — a dead worker
            errors.append(exc)

    threads = [threading.Thread(target=boot) for _ in range(processes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return errors, ready


@pytest.mark.parametrize("processes", [2, 4, 8])
def test_every_process_starts_when_they_race_on_an_empty_qdrant(monkeypatch, processes):
    server = RacingQdrantServer(processes)
    errors, ready = _cold_start(monkeypatch, server, processes)
    assert errors == []
    assert len(ready) == processes
    assert server.existing == set(COLLECTIONS)
    assert server.creates == len(COLLECTIONS)
    assert server.conflicts >= processes - 1  # the race really happened


def test_repeated_cold_starts(monkeypatch):
    for _ in range(5):
        server = RacingQdrantServer(4)
        errors, ready = _cold_start(monkeypatch, server, 4)
        assert errors == [] and len(ready) == 4


def test_other_qdrant_errors_still_fail_start(monkeypatch):
    calls = []

    def from_config(config):
        calls.append(1)
        raise UnexpectedResponse(status_code=500, reason_phrase="Server Error", content=b"boom", headers=httpx.Headers())

    monkeypatch.setattr(mem0.Memory, "from_config", from_config)
    monkeypatch.setattr(type(settings), "get_mem0_config", lambda self: {})
    with pytest.raises(UnexpectedResponse):
        Service()._get_memory()
    assert len(calls) == 1


def test_a_conflict_that_never_clears_is_not_retried_forever(monkeypatch):
    calls = []

    def from_config(config):
        calls.append(1)
        raise _conflict("mem0migrations")

    monkeypatch.setattr(mem0.Memory, "from_config", from_config)
    monkeypatch.setattr(type(settings), "get_mem0_config", lambda self: {})
    monkeypatch.setattr("memory.core.time.sleep", lambda s: None, raising=False)
    with pytest.raises(UnexpectedResponse):
        Service()._get_memory()
    assert 1 < len(calls) <= 10
