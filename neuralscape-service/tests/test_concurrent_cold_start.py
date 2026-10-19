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


# ── Follow-up: only "already exists" counts; legacy lazy clients too ────────


def _unrelated_409():
    return UnexpectedResponse(
        status_code=409,
        reason_phrase="Conflict",
        content=b'{"status":{"error":"Wrong input: Collection `x` is being updated, try later"}}',
        headers=httpx.Headers(),
    )


def test_a_409_that_is_not_already_exists_is_not_retried(monkeypatch):
    calls = []

    def from_config(config):
        calls.append(1)
        raise _unrelated_409()

    monkeypatch.setattr(mem0.Memory, "from_config", from_config)
    monkeypatch.setattr(type(settings), "get_mem0_config", lambda self: {})
    monkeypatch.setattr("memory.core.time.sleep", lambda s: None, raising=False)
    with pytest.raises(UnexpectedResponse):
        Service()._get_memory()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "exc,expected",
    [
        (_conflict("neuralscape_memories"), True),
        (_unrelated_409(), False),
        (UnexpectedResponse(status_code=409, reason_phrase="Conflict", content=b"", headers=httpx.Headers()), False),
        (UnexpectedResponse(status_code=500, reason_phrase="x", content=b"already exists", headers=httpx.Headers()), False),
        (RuntimeError("Collection already exists"), False),
    ],
)
def test_only_qdrant_already_exists_409_counts_as_concurrent_creation(exc, expected):
    from memory.core import _is_collection_exists_conflict

    assert _is_collection_exists_conflict(exc) is expected


class LostRaceQdrantServer(RacingQdrantServer):
    """This process always loses: its first listing is empty, but another
    process creates every collection right after, so its create gets 409."""

    def __init__(self):
        super().__init__(1)
        self.existing = set(COLLECTIONS)
        self.listings = 0

    def get_collections(self):
        self.listings += 1
        if self.listings == 1:
            return SimpleNamespace(collections=[])
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.existing])


@pytest.fixture()
def legacy_main(monkeypatch):
    import main

    monkeypatch.setattr(main, "_memory", None)
    monkeypatch.setattr(main, "_graphiti", None)
    monkeypatch.setattr(main, "_bridge", None)
    monkeypatch.setattr(main, "_async_memory", None)
    monkeypatch.setattr(type(settings), "get_mem0_config", lambda self: {})
    return main


def test_legacy_lazy_memory_survives_the_race(monkeypatch, legacy_main):
    monkeypatch.setattr("memory.core.time.sleep", lambda s: None, raising=False)
    """main._get_memory (legacy routes) builds the client once; when that first
    build races another process it must still come up."""
    server = LostRaceQdrantServer()
    monkeypatch.setattr(mem0.Memory, "from_config", _fake_from_config(server))
    memory = legacy_main._get_memory()
    assert memory is not None and legacy_main._memory is memory
    assert server.conflicts == 1
    assert server.creates == 0


def test_legacy_lazy_memory_other_errors_fail_at_once(monkeypatch, legacy_main):
    calls = []

    def from_config(config):
        calls.append(1)
        raise _unrelated_409()

    monkeypatch.setattr(mem0.Memory, "from_config", from_config)
    with pytest.raises(UnexpectedResponse):
        legacy_main._get_memory()
    assert len(calls) == 1 and legacy_main._memory is None


def test_legacy_async_memory_survives_the_race(monkeypatch, legacy_main):
    monkeypatch.setattr("memory.core.time.sleep", lambda s: None, raising=False)
    """AsyncMemory.from_config is synchronous (it builds the same sync Qdrant
    store), so the same bounded handling applies to main._get_async_memory."""
    server = LostRaceQdrantServer()
    monkeypatch.setattr(mem0.AsyncMemory, "from_config", _fake_from_config(server))
    memory = legacy_main._get_async_memory()
    assert memory is not None and legacy_main._async_memory is memory
    assert server.conflicts == 1


# ── Follow-up: a collection listed while another process is still creating it ──


def _not_yet_readable():
    return UnexpectedResponse(
        status_code=500,
        reason_phrase="Internal Server Error",
        content=b'{"status":{"error":"Service internal error: 0 of 0 read operations failed"},"time":0.0063}',
        headers=httpx.Headers(),
    )


class StillCreatingQdrantServer(RacingQdrantServer):
    """Another process has started creating every collection: they are already
    listed, but the first ``unreadable_reads`` reads of them fail the way Qdrant
    answers before the new collection's shard is up."""

    def __init__(self, unreadable_reads):
        super().__init__(1)
        self.existing = set(COLLECTIONS)
        self.unreadable_reads = unreadable_reads
        self.reads = 0

    def get_collection(self, name):
        with self.lock:
            self.reads += 1
            if self.reads <= self.unreadable_reads:
                raise _not_yet_readable()
        return super().get_collection(name)


@pytest.mark.parametrize("unreadable_reads", [1, 2])
def test_a_process_that_reads_a_collection_still_being_created_starts(monkeypatch, unreadable_reads):
    server = StillCreatingQdrantServer(unreadable_reads)
    monkeypatch.setattr(mem0.Memory, "from_config", _fake_from_config(server))
    monkeypatch.setattr(type(settings), "get_mem0_config", lambda self: {})
    monkeypatch.setattr("memory.core.time.sleep", lambda s: None, raising=False)
    assert Service()._get_memory() is not None
    assert server.reads == unreadable_reads + len(COLLECTIONS)
    assert server.creates == 0


def test_a_collection_that_never_becomes_readable_is_not_retried_forever(monkeypatch):
    server = StillCreatingQdrantServer(unreadable_reads=10**6)
    monkeypatch.setattr(mem0.Memory, "from_config", _fake_from_config(server))
    monkeypatch.setattr(type(settings), "get_mem0_config", lambda self: {})
    monkeypatch.setattr("memory.core.time.sleep", lambda s: None, raising=False)
    with pytest.raises(UnexpectedResponse) as err:
        Service()._get_memory()
    assert err.value.status_code == 500
    assert server.reads == 5


def test_other_qdrant_500s_on_read_still_fail_start(monkeypatch):
    class BrokenRead(StillCreatingQdrantServer):
        def get_collection(self, name):
            self.reads += 1
            raise UnexpectedResponse(
                status_code=500,
                reason_phrase="Internal Server Error",
                content=b'{"status":{"error":"Service internal error: 1 of 1 read operations failed"}}',
                headers=httpx.Headers(),
            )

    server = BrokenRead(0)
    monkeypatch.setattr(mem0.Memory, "from_config", _fake_from_config(server))
    monkeypatch.setattr(type(settings), "get_mem0_config", lambda self: {})
    monkeypatch.setattr("memory.core.time.sleep", lambda s: None, raising=False)
    with pytest.raises(UnexpectedResponse):
        Service()._get_memory()
    assert server.reads == 1


@pytest.mark.parametrize(
    "exc,expected",
    [
        (_not_yet_readable(), True),
        (_conflict("mem0migrations"), True),
        (
            UnexpectedResponse(
                status_code=500,
                reason_phrase="x",
                content=b'{"status":{"error":"Service internal error: 1 of 1 read operations failed"}}',
                headers=httpx.Headers(),
            ),
            False,
        ),
        (UnexpectedResponse(status_code=503, reason_phrase="x", content=b"0 of 0 read operations failed", headers=httpx.Headers()), False),
        (UnexpectedResponse(status_code=500, reason_phrase="x", content=b"", headers=httpx.Headers()), False),
        (RuntimeError("0 of 0 read operations failed"), False),
    ],
)
def test_only_the_two_concurrent_creation_answers_are_retried(exc, expected):
    from memory.core import _is_concurrent_creation

    assert _is_concurrent_creation(exc) is expected


def test_legacy_lazy_memory_survives_a_collection_still_being_created(monkeypatch, legacy_main):
    monkeypatch.setattr("memory.core.time.sleep", lambda s: None, raising=False)
    server = StillCreatingQdrantServer(unreadable_reads=1)
    monkeypatch.setattr(mem0.Memory, "from_config", _fake_from_config(server))
    memory = legacy_main._get_memory()
    assert memory is not None and legacy_main._memory is memory
