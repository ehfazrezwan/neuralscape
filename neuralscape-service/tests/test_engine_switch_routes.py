"""The engine switch's admin routes and start-up, with and without ``ENGINE_ROUTING_ENABLED``.

No external service: the engines are the deterministic reference stores and the ledger is a real
SQLite ledger in a temporary directory.  The same routes are driven against real Qdrant and
PostgreSQL by ``integration_tests/test_engine_switch_rest_integration.py``.

* flag off (the default): every route answers 404, start-up builds nothing and never imports the
  switch module, and the settings default to off;
* flag on: dictator-only, 503 with the reason while the switch is unavailable, 409 for a refused
  step, and the full start-dual-write -> catch-up -> flip -> rollback -> abandon cycle.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import engine_adapters_switch as switch_module
import main
from config import Settings, settings
from engine_adapters_switch import EngineSwitch
from engine_ports import ReferenceVectorStore
from engine_routing import Backend, EngineRouter
from ledger.port import ObjectRef
from ledger.sqlite import SqliteLedger

SERVICE = Path(__file__).resolve().parent.parent
ROUTES = (
    ("get", "/v1/admin/engine-switch/status"),
    ("post", "/v1/admin/engine-switch/start-dual-write"),
    ("post", "/v1/admin/engine-switch/catch-up"),
    ("post", "/v1/admin/engine-switch/flip"),
    ("post", "/v1/admin/engine-switch/rollback"),
    ("post", "/v1/admin/engine-switch/abandon"),
)
ADMIN = "default_user"  # the identity the suite's auth opt-out resolves to
TENANT = "tenant-1"


class _Rows:
    """The memory service's rows, as the canonical source joins them."""

    def __init__(self) -> None:
        self.rows: dict[str, SimpleNamespace] = {}

    def put(self, memory_id: str, text: str, user: str = ADMIN) -> None:
        self.rows[memory_id] = SimpleNamespace(memory=text, owner_user_id=user, visibility="private", project_id=None)

    def get_memory(self, memory_id: str):
        return self.rows.get(memory_id)


@pytest.fixture()
def ledger(tmp_path):
    opened = SqliteLedger(str(tmp_path / "ledger.db"))
    yield opened
    opened.close()


@pytest.fixture()
def installed(ledger):
    """An installed switch over two reference engines (generation 1 = ``a``, shadow = ``b``)."""

    rows = _Rows()
    engines = {"a": ReferenceVectorStore("a"), "b": ReferenceVectorStore("b")}
    switch = EngineSwitch(
        EngineRouter(Backend("a", engines["a"])),
        ledger=ledger,
        service=rows,
        tenant_for=lambda user: TENANT,
        shadow_factory=lambda: Backend("b", engines["b"]),
    )
    switch_module.configure(switch)
    yield switch, rows, engines
    switch_module.stop()


@pytest.fixture()
def routing_on(monkeypatch):
    monkeypatch.setattr(settings, "engine_routing_enabled", True)
    monkeypatch.setattr(settings, "dictator_user_ids", ADMIN)


@pytest.fixture()
def stub_service(monkeypatch):
    """The memory service the lifespan has already built; start-up only reads its store and embedder."""

    memory = SimpleNamespace(vector_store=None, embedding_model=SimpleNamespace(embed=lambda text: [1.0] * 8))
    stub = SimpleNamespace(_get_memory=lambda: memory)
    monkeypatch.setattr(main, "_service", stub)
    return stub


def _call(client, method, path):
    return getattr(client, method)(path)


def _register(ledger, rows, memory_id, text, revision=1):
    rows.put(memory_id, text)
    ledger.put_object_ref(ObjectRef(TENANT, memory_id, revision, 1))


# -- the flag is off by default ------------------------------------------------


def test_the_settings_default_to_off():
    cfg = Settings(_env_file=None, google_api_key="k", neo4j_password="p")
    assert cfg.engine_routing_enabled is False
    assert cfg.engine_routing_recall_enabled is False
    assert cfg.engine_pgvector_enabled is False


@pytest.mark.parametrize("method,path", ROUTES)
def test_every_route_is_404_while_routing_is_off(method, path, monkeypatch):
    monkeypatch.setattr(settings, "engine_routing_enabled", False)
    monkeypatch.setattr(settings, "dictator_user_ids", ADMIN)
    assert _call(TestClient(main.app), method, path).status_code == 404


def test_a_switch_that_happens_to_exist_is_not_reachable_while_routing_is_off(installed, monkeypatch):
    monkeypatch.setattr(settings, "engine_routing_enabled", False)
    monkeypatch.setattr(settings, "dictator_user_ids", ADMIN)
    assert TestClient(main.app).get("/v1/admin/engine-switch/status").status_code == 404
    assert switch_module.projection_switch(settings) is None  # and nothing is projected


def test_startup_with_routing_off_builds_nothing_and_never_imports_the_switch():
    code = (
        "import asyncio, json, sys, main; "
        "asyncio.run(main._start_engine_switch()); asyncio.run(main._stop_engine_switch()); "
        "print(json.dumps({'imported': 'engine_adapters_switch' in sys.modules, "
        "'engine_modules': sorted(m for m in sys.modules if m.startswith('engine_'))}))"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("ENGINE_")}
    env.update({"PYTHONPATH": str(SERVICE), "GOOGLE_API_KEY": "k", "NEO4J_PASSWORD": "p"})
    proc = subprocess.run([sys.executable, "-c", code], cwd=SERVICE, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    seen = json.loads(proc.stdout.strip().splitlines()[-1])
    assert seen["imported"] is False, seen
    assert seen["engine_modules"] == [], seen


@pytest.mark.parametrize("identity", ["dictator", "non_dictator"])
def test_flag_off_answers_404_for_every_caller_and_imports_no_engine_module(identity):
    """The flag is checked before auth and before any engine import (a subprocess, so nothing is pre-imported)."""

    code = (
        "import json, sys; from fastapi.testclient import TestClient; import main; "
        "from config import settings; "
        "settings.neuralscape_auth_disabled = True; settings.neuralscape_user_token_secret = ''; "
        "settings.neuralscape_api_key = ''; settings.engine_routing_enabled = False; "
        f"settings.dictator_user_ids = {'default_user' if identity == 'dictator' else 'someone-else'!r}; "
        "c = TestClient(main.app); "
        "codes = [getattr(c, m)(p).status_code for m, p in " + repr(ROUTES) + "]; "
        "print(json.dumps({'codes': codes, 'engine_modules': sorted(m for m in sys.modules if m.startswith('engine_'))}))"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("ENGINE_")}
    env.update({"PYTHONPATH": str(SERVICE), "GOOGLE_API_KEY": "k", "NEO4J_PASSWORD": "p"})
    proc = subprocess.run([sys.executable, "-c", code], cwd=SERVICE, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    seen = json.loads(proc.stdout.strip().splitlines()[-1])
    assert seen["codes"] == [404] * len(ROUTES), seen
    assert seen["engine_modules"] == [], seen


def test_flag_off_with_auth_enforced_is_indistinguishable_from_an_unknown_route(monkeypatch):
    monkeypatch.setattr(settings, "engine_routing_enabled", False)
    monkeypatch.setattr(settings, "neuralscape_auth_disabled", False)
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", "s" * 32)
    client = TestClient(main.app)
    unknown = client.get("/v1/admin/no-such-route").status_code
    for method, path in ROUTES:
        assert _call(client, method, path).status_code == unknown, path


# -- the flag is on --------------------------------------------------------------


def test_a_non_dictator_is_refused_before_anything_else(routing_on, installed, monkeypatch):
    monkeypatch.setattr(settings, "dictator_user_ids", "someone-else")
    client = TestClient(main.app)
    for method, path in ROUTES:
        assert _call(client, method, path).status_code == 403, path


def test_503_names_the_reason_while_the_switch_is_unavailable(routing_on):
    switch_module.stop()
    response = TestClient(main.app).get("/v1/admin/engine-switch/status")
    assert response.status_code == 503
    error = response.json()["detail"]["error"]
    assert error["code"] == "engine_switch_unavailable"
    assert error["reason"] == "stopped"


def test_startup_without_an_open_ledger_leaves_the_switch_unavailable_with_its_reason(
    routing_on, stub_service, monkeypatch
):
    monkeypatch.setattr(settings, "engine_routing_dimensions", 8)
    monkeypatch.setattr(main._task_manager, "ledger", None, raising=False)
    asyncio.run(main._start_engine_switch())
    try:
        response = TestClient(main.app).get("/v1/admin/engine-switch/status")
        assert response.status_code == 503
        assert response.json()["detail"]["error"]["reason"] == "ledger_not_open"
    finally:
        switch_module.stop()


def test_startup_without_dimensions_leaves_the_switch_unavailable(routing_on, stub_service, ledger, monkeypatch):
    monkeypatch.setattr(settings, "engine_routing_dimensions", 0)
    monkeypatch.setattr(main._task_manager, "ledger", ledger, raising=False)
    asyncio.run(main._start_engine_switch())
    try:
        assert TestClient(main.app).get("/v1/admin/engine-switch/status").json()["detail"]["error"]["reason"] == (
            "dimensions_not_set"
        )
    finally:
        switch_module.stop()


def test_a_refused_step_is_a_409_not_a_500(routing_on, installed):
    client = TestClient(main.app)
    assert client.post("/v1/admin/engine-switch/flip").status_code == 409  # no dual-write window open
    assert client.post("/v1/admin/engine-switch/rollback").status_code == 409  # nothing to roll back to


def test_an_unexpected_failure_is_a_generic_500(routing_on, installed, monkeypatch):
    switch, _rows, _engines = installed

    def boom():
        raise RuntimeError("secret detail that must not leak")

    monkeypatch.setattr(switch, "status", boom)
    response = TestClient(main.app).get("/v1/admin/engine-switch/status")
    assert response.status_code == 500
    assert "secret detail" not in response.text


def test_the_full_cycle_through_the_routes(routing_on, installed, ledger):
    switch, rows, engines = installed
    client = TestClient(main.app)

    status = client.get("/v1/admin/engine-switch/status")
    assert status.status_code == 200
    assert (status.json()["generation"], status.json()["phase"], status.json()["active"]) == (1, "steady", "a")

    # A memory written before any window: the backfill (catch-up with no window) brings generation 1 to canonical state.
    _register(ledger, rows, "m-old", "an older memory about turbines")
    backfill = client.post("/v1/admin/engine-switch/catch-up").json()
    assert backfill["backfilled"] is True and backfill["report"]["copied"] == 1 and backfill["active_synced"] is True

    started = client.post("/v1/admin/engine-switch/start-dual-write")
    assert started.status_code == 200 and started.json()["phase"] == "dual_write"

    # A write accepted during the window reaches both engines (the ledger's vector stage calls project_apply).
    _register(ledger, rows, "m-new", "a newer memory about harbours")
    switch.project_apply(TENANT, "m-new", text="a newer memory about harbours", user_id=ADMIN, scope="global")
    ctx = switch_module._ctx(TENANT)
    assert engines["a"].get(ctx, "m-new") is not None and engines["b"].get(ctx, "m-new") is not None

    # The shadow does not yet hold the memory that predates the window; catch-up copies it from canonical state.
    assert engines["b"].get(ctx, "m-old") is None
    report = client.post("/v1/admin/engine-switch/catch-up").json()["report"]
    assert report["copied"] >= 1 and report["conflicts"] == 0
    assert engines["b"].get(ctx, "m-old") is not None
    flipped = client.post("/v1/admin/engine-switch/flip")
    assert flipped.status_code == 200 and flipped.json()["generation"] == 2 and flipped.json()["active"] == "b"

    # Reads are served by the flipped engine; a post-flip write is on it and not yet on the old one.
    _register(ledger, rows, "m-after", "a memory written after the flip about glaciers")
    switch.project_apply(TENANT, "m-after", text="a memory written after the flip about glaciers", user_id=ADMIN, scope="global")
    hits = switch.recall(ADMIN, "glaciers", 5)
    assert [h.record_id for h in hits][:1] == ["m-after"]
    assert engines["a"].get(ctx, "m-after") is None and engines["b"].get(ctx, "m-after") is not None

    # Rollback replays what was accepted since the flip onto the old engine and serves from it again.
    rolled = client.post("/v1/admin/engine-switch/rollback")
    assert rolled.status_code == 200 and rolled.json()["generation"] == 3 and rolled.json()["active"] == "a"
    assert engines["a"].get(ctx, "m-after") is not None
    assert [h.record_id for h in switch.recall(ADMIN, "glaciers", 5)][:1] == ["m-after"]

    # A second window can be abandoned without moving anything.
    assert client.post("/v1/admin/engine-switch/start-dual-write").json()["phase"] == "dual_write"
    abandoned = client.post("/v1/admin/engine-switch/abandon")
    assert abandoned.status_code == 200 and abandoned.json()["phase"] == "steady"
    assert abandoned.json()["generation"] == 3


def test_a_followed_switch_refuses_its_own_admin_operations(routing_on, installed):
    switch, _rows, _engines = installed
    switch.follow(lambda: switch._router)  # a migration controller owns the window
    assert TestClient(main.app).post("/v1/admin/engine-switch/flip").status_code == 409
    assert TestClient(main.app).get("/v1/admin/engine-switch/status").status_code == 200


def test_a_late_projection_of_an_erased_record_leaves_the_engine_empty(installed, ledger):
    switch, rows, engines = installed
    ctx = switch_module._ctx(TENANT)
    _register(ledger, rows, "m-erased", "text that must not come back")
    switch.project_apply(TENANT, "m-erased", text="text that must not come back", user_id=ADMIN, scope="global")
    assert engines["a"].get(ctx, "m-erased") is not None
    ledger.tombstone(TENANT, "m-erased", 1, "test_erasure")
    switch.project_remove(TENANT, "m-erased")
    assert engines["a"].get(ctx, "m-erased") is None
    switch.project_apply(TENANT, "m-erased", text="text that must not come back", user_id=ADMIN, scope="global", revision=1)
    assert engines["a"].get(ctx, "m-erased") is None


def test_the_tombstone_guard_holds_even_when_the_engine_never_saw_the_record(installed, ledger):
    switch, _rows, engines = installed
    ctx = switch_module._ctx(TENANT)
    ledger.tombstone(TENANT, "m-never", 3, "test_erasure")  # erased before it was ever projected
    switch.project_apply(TENANT, "m-never", text="late text", user_id=ADMIN, scope="global", revision=2)
    switch.project_apply(TENANT, "m-never", text="late text", user_id=ADMIN, scope="global", revision=3)
    assert engines["a"].get(ctx, "m-never") is None
