"""The engine switch is built at startup (ENGINE_ROUTING_ENABLED) and the admin routes drive it.

Every switch here is built by the service's own startup path (``main._start_engine_switch`` or the
app lifespan) over a real SQLite ledger and a local on-disk Qdrant collection; nothing installs a
prepared switch into the module.  The embedder is a deterministic stand-in (no model is called).
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import engine_adapters_switch as sw
import main
from config import settings
from ledger.sqlite import SqliteLedger

DIMS = 8


def _embed(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in digest[:DIMS]]


class _FakeService:
    """Only what startup reads: the embedder (and no shared Qdrant client)."""

    def __init__(self) -> None:
        self.memory = SimpleNamespace(embedding_model=SimpleNamespace(embed=_embed), vector_store=None)

    def _get_memory(self):
        return self.memory

    def get_memory(self, memory_id, user_id=None):
        return None


class _TaskManager:
    """The task manager's ledger and producer as startup sees them (connect opens a real ledger)."""

    def __init__(self, ledger_path=None) -> None:
        self._ledger_path = ledger_path
        self.ledger = None
        self.producer = None

    async def connect(self) -> None:
        if self._ledger_path is not None:
            self.ledger = SqliteLedger(str(self._ledger_path))
            self.producer = SimpleNamespace(tenant_for=lambda user_id: "tenant-default" if user_id else None)

    async def close(self) -> None:
        self.ledger = None


@pytest.fixture
def routed(monkeypatch, tmp_path):
    for name, value in (
        ("engine_routing_enabled", True),
        ("engine_routing_recall_enabled", False),
        ("engine_routing_dimensions", DIMS),
        ("engine_routing_qdrant_collection", "engine_projection_test"),
        ("qdrant_url", ""),
        ("qdrant_path", str(tmp_path / "qdrant")),
        ("engine_pgvector_dsn", ""),
        ("dictator_user_ids", settings.default_user_id),
    ):
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(main, "_service", _FakeService())
    yield tmp_path
    sw.stop()


def _start(monkeypatch, manager):
    monkeypatch.setattr(main, "_task_manager", manager)
    asyncio.run(manager.connect())
    asyncio.run(main._start_engine_switch())


def test_startup_builds_the_switch_and_the_admin_routes_drive_it(routed, monkeypatch):
    _start(monkeypatch, _TaskManager(routed / "ledger.db"))
    switch = sw.get_switch()
    assert isinstance(switch, sw.EngineSwitch) and sw.unavailable_reason() is None
    client = TestClient(main.app)

    status = client.get("/v1/admin/engine-switch/status")
    assert status.status_code == 200, status.text
    assert status.json()["generation"] == 1 and status.json()["phase"] == "steady" and status.json()["active"] == "qdrant"

    caught_up = client.post("/v1/admin/engine-switch/catch-up")  # no window open: backfills the serving engine
    assert caught_up.status_code == 200, caught_up.text
    assert caught_up.json()["backfilled"] is True and caught_up.json()["active_synced"] is True

    no_shadow = client.post("/v1/admin/engine-switch/start-dual-write")  # no ENGINE_PGVECTOR_DSN configured
    assert no_shadow.status_code == 409 and "no shadow backend" in no_shadow.json()["detail"]


def test_the_routes_stay_privileged_with_a_built_switch(routed, monkeypatch):
    _start(monkeypatch, _TaskManager(routed / "ledger.db"))
    monkeypatch.setattr(settings, "dictator_user_ids", "someone-else")
    assert sw.get_switch() is not None
    response = TestClient(main.app).get("/v1/admin/engine-switch/status")
    assert response.status_code == 403


def test_with_routing_off_nothing_is_built_and_the_routes_are_not_found(routed, monkeypatch):
    monkeypatch.setattr(settings, "engine_routing_enabled", False)
    _start(monkeypatch, _TaskManager(routed / "ledger.db"))
    assert sw.get_switch() is None
    for method, path in (("get", "status"), ("post", "catch-up"), ("post", "flip")):
        response = getattr(TestClient(main.app), method)(f"/v1/admin/engine-switch/{path}")
        assert response.status_code == 404, (path, response.text)


@pytest.mark.parametrize(
    "case, reason",
    [("no-ledger", "ledger_not_open"), ("no-dimensions", "dimensions_not_set"), ("engine-unbuildable", "build_failed")],
)
def test_an_unavailable_switch_answers_503_with_its_reason(routed, monkeypatch, case, reason):
    manager = _TaskManager(None if case == "no-ledger" else routed / "ledger.db")
    if case == "no-dimensions":
        monkeypatch.setattr(settings, "engine_routing_dimensions", 0)
    if case == "engine-unbuildable":
        blocker = routed / "not-a-directory"
        blocker.write_text("x")
        monkeypatch.setattr(settings, "qdrant_path", str(blocker))  # a local Qdrant cannot open a file as storage
    _start(monkeypatch, manager)
    assert sw.get_switch() is None and sw.unavailable_reason() == reason
    response = TestClient(main.app).post("/v1/admin/engine-switch/flip")
    assert response.status_code == 503
    error = response.json()["detail"]["error"]
    assert error["code"] == "engine_switch_unavailable" and error["reason"] == reason


def test_shutdown_uninstalls_the_switch_and_closes_the_client_it_opened(routed, monkeypatch):
    _start(monkeypatch, _TaskManager(routed / "ledger.db"))
    switch = sw.get_switch()
    closed = []
    switch._closers = [lambda: closed.append(True), *switch._closers]
    asyncio.run(main._stop_engine_switch())
    assert sw.get_switch() is None and sw.unavailable_reason() == "stopped" and closed == [True]
    response = TestClient(main.app).get("/v1/admin/engine-switch/status")
    assert response.status_code == 503 and response.json()["detail"]["error"]["reason"] == "stopped"
    asyncio.run(main._stop_engine_switch())  # twice is harmless


def test_a_shared_service_client_is_reused_and_never_closed(routed, monkeypatch):
    from qdrant_client import QdrantClient

    shared = QdrantClient(path=str(routed / "shared-qdrant"))
    service = _FakeService()
    service.memory.vector_store = SimpleNamespace(client=shared)
    monkeypatch.setattr(main, "_service", service)
    _start(monkeypatch, _TaskManager(routed / "ledger.db"))
    switch = sw.get_switch()
    assert switch is not None and switch._closers == []
    asyncio.run(main._stop_engine_switch())
    assert shared.collection_exists("engine_projection_test")  # still open: the service owns it
    try:
        shared.close()
    except sqlite3.ProgrammingError:
        pass  # a local on-disk Qdrant is thread-affine: the collection was made on the startup thread (cleanup only)


def test_the_app_lifespan_starts_the_switch_after_connecting_and_stops_it_before_closing():
    """The lifespan wiring itself (its other steps differ per deployment, so its order is read, not run)."""

    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(main.lifespan)))
    awaited = [
        ast.unparse(node.value.func) for node in ast.walk(tree)
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call)
    ]
    for name in ("_task_manager.connect", "_start_engine_switch", "_stop_engine_switch", "_task_manager.close"):
        assert awaited.count(name) == 1, (name, awaited)
    assert awaited.index("_task_manager.connect") < awaited.index("_start_engine_switch")  # the ledger is open
    assert awaited.index("_stop_engine_switch") < awaited.index("_task_manager.close")  # before the ledger closes


# ---------------------------------------------------------------- the ledger worker's own switch
# The vector stage runs in the ledger worker process, so projection needs a switch there too.


def _worker_ctx(tmp_path):
    from write_producer import SqlitePayloadStore

    return {"ledger": SqliteLedger(str(tmp_path / "ledger.db")), "ledger_payloads": SqlitePayloadStore(str(tmp_path / "kv.db"))}


def test_the_ledger_worker_builds_the_switch_the_vector_stage_projects_into(routed):
    import worker
    from write_producer import _project_remove, _project_row  # noqa: F401  (the stage's own entry points)

    ctx = _worker_ctx(routed)
    try:
        assert sw.projection_switch(settings) is None  # before start-up: nothing to project into
        worker._start_worker_engine_switch(ctx, _FakeService())
        switch = sw.projection_switch(settings)
        assert isinstance(switch, sw.EngineSwitch)
        assert switch.router.generation == 1
    finally:
        ctx["ledger"].close()
        ctx["ledger_payloads"].close()


def test_the_ledger_worker_builds_nothing_with_routing_off_or_no_ledger(routed, monkeypatch):
    import worker

    ctx = _worker_ctx(routed)
    try:
        monkeypatch.setattr(settings, "engine_routing_enabled", False)
        worker._start_worker_engine_switch(ctx, _FakeService())
        assert sw.get_switch() is None
        monkeypatch.setattr(settings, "engine_routing_enabled", True)
        worker._start_worker_engine_switch({}, _FakeService())
        assert sw.get_switch() is None
    finally:
        ctx["ledger"].close()
        ctx["ledger_payloads"].close()


def test_a_worker_vector_stage_row_reaches_the_routed_engine(routed):
    """The stage's ``_project_row`` finds the worker's switch and writes the record into generation 1."""
    import worker
    from contracts_lifecycle import ProcessingStage  # noqa: F401
    from ledger.conformance import ref
    from write_producer import _project_row

    ctx = _worker_ctx(routed)
    try:
        ctx["ledger"].put_object_ref(ref("mem-1"))
        service = _FakeService()
        service.get_memory = lambda memory_id, user_id=None: SimpleNamespace(
            memory="green tea", owner_user_id="alice", visibility="private", project_id=None)
        worker._start_worker_engine_switch(ctx, service)
        tenant = "tenant-default"
        switch = sw.get_switch()
        switch._tenant_for = lambda user_id: tenant
        _project_row(tenant, "mem-1")
        hits = switch.recall("alice", "green tea", 3)
        assert [hit.record_id for hit in hits] == ["mem-1"]
    finally:
        ctx["ledger"].close()
        ctx["ledger_payloads"].close()
