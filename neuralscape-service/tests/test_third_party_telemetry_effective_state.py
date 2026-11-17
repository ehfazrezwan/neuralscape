"""The effective third-party telemetry state is observable: function, /health, startup log."""

from __future__ import annotations

import logging
import sys
import types

import pytest
from fastapi.testclient import TestClient

import third_party_telemetry as tpt


# ---- the function ---------------------------------------------------------------------


def test_off_when_both_set_false():
    env = {"MEM0_TELEMETRY": "false", "GRAPHITI_TELEMETRY_ENABLED": "false"}
    assert tpt.effective_state(env) == {"mem0": False, "graphiti": False}


def test_on_when_both_opted_in():
    env = {"MEM0_TELEMETRY": "True", "GRAPHITI_TELEMETRY_ENABLED": "ON"}
    assert tpt.effective_state(env) == {"mem0": True, "graphiti": True}


@pytest.mark.parametrize("value", ["1", "yes", "TRUE"])
def test_truthy_spellings_match_the_libraries(value):
    assert tpt.effective_state({"MEM0_TELEMETRY": value})["mem0"] is True
    assert tpt.effective_state({"GRAPHITI_TELEMETRY_ENABLED": value})["graphiti"] is True


def test_mem0_does_not_accept_on_but_graphiti_does():
    # The two libraries disagree about "on"; report each the way it decides.
    assert tpt.effective_state({"MEM0_TELEMETRY": "on"})["mem0"] is False
    assert tpt.effective_state({"GRAPHITI_TELEMETRY_ENABLED": "on"})["graphiti"] is True


def test_explicit_empty_is_off_for_both():
    env = {"MEM0_TELEMETRY": "", "GRAPHITI_TELEMETRY_ENABLED": ""}
    assert tpt.effective_state(env) == {"mem0": False, "graphiti": False}


def test_unset_means_the_libraries_own_default_which_is_on():
    # What the library would do if the guard had not run (the guard is what turns it off).
    assert tpt.effective_state({}) == {"mem0": True, "graphiti": True}


def test_graphiti_reported_on_even_though_pytest_is_loaded():
    # Graphiti's own check returns False whenever 'pytest' is imported; we report the real-process answer.
    assert "pytest" in sys.modules
    assert tpt.effective_state({"GRAPHITI_TELEMETRY_ENABLED": "true"})["graphiti"] is True


def test_live_mem0_value_is_the_latched_import_time_value(monkeypatch):
    stub = types.SimpleNamespace(MEM0_TELEMETRY=True)
    monkeypatch.setitem(sys.modules, "mem0.memory.telemetry", stub)
    monkeypatch.setenv("MEM0_TELEMETRY", "false")  # changed after import: does not matter to mem0
    assert tpt.effective_state()["mem0"] is True
    stub.MEM0_TELEMETRY = False
    monkeypatch.setenv("MEM0_TELEMETRY", "true")
    assert tpt.effective_state()["mem0"] is False


def test_payload_is_booleans_and_labels_only():
    state = tpt.effective_state({})
    assert set(state) == {"mem0", "graphiti"}
    assert all(isinstance(v, bool) for v in state.values())
    assert set(tpt.effective_labels({}).values()) == {"on"}
    assert tpt.startup_log_line({}) == "third_party_telemetry mem0=on graphiti=on"


def test_default_service_process_reports_off_for_both():
    # conftest imports the guard first, so the real libraries are latched/set off here.
    assert tpt.effective_labels() == {"mem0": "off", "graphiti": "off"}
    assert tpt.startup_log_line() == "third_party_telemetry mem0=off graphiti=off"


# ---- /health --------------------------------------------------------------------------


def _health(monkeypatch, mem0: bool, graphiti: str):
    from main import app

    stub = types.SimpleNamespace(MEM0_TELEMETRY=mem0)
    monkeypatch.setitem(sys.modules, "mem0.memory.telemetry", stub)
    monkeypatch.setenv("GRAPHITI_TELEMETRY_ENABLED", graphiti)
    return TestClient(app, raise_server_exceptions=False).get("/health")


def test_health_reports_off_by_default_and_flips_with_env(monkeypatch):
    resp = _health(monkeypatch, mem0=False, graphiti="false")
    assert resp.json()["checks"]["third_party_telemetry"] == {"mem0": "off", "graphiti": "off"}
    resp = _health(monkeypatch, mem0=True, graphiti="true")
    assert resp.json()["checks"]["third_party_telemetry"] == {"mem0": "on", "graphiti": "on"}
    resp = _health(monkeypatch, mem0=False, graphiti="1")
    assert resp.json()["checks"]["third_party_telemetry"] == {"mem0": "off", "graphiti": "on"}


def test_health_field_does_not_degrade_status_or_change_code(monkeypatch):
    off = _health(monkeypatch, mem0=False, graphiti="false")
    on = _health(monkeypatch, mem0=True, graphiti="true")
    assert off.status_code == on.status_code
    assert off.json()["status"] == on.json()["status"]


def test_health_field_carries_no_endpoint_or_key(monkeypatch):
    resp = _health(monkeypatch, mem0=True, graphiti="true")
    field = resp.json()["checks"]["third_party_telemetry"]
    assert set(field) == {"mem0", "graphiti"}
    assert set(field.values()) <= {"on", "off"}
    blob = str(field).lower()
    for needle in ("http", "posthog", "phc_", "key", "id"):
        assert needle not in blob


# ---- startup log ----------------------------------------------------------------------


def test_api_lifespan_logs_the_effective_state(monkeypatch, caplog, capsys):
    import main

    def _stop():
        raise RuntimeError("stop after the first lines")

    monkeypatch.setattr(main, "settings", types.SimpleNamespace(validate_required=_stop))
    caplog.set_level(logging.INFO)

    import asyncio

    async def run():
        async with main.lifespan(main.app):
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(run())
    assert "third_party_telemetry mem0=off graphiti=off" in caplog.text + capsys.readouterr().out


def test_worker_startup_logs_the_effective_state(monkeypatch, caplog, capsys):
    import asyncio

    import worker

    class _Stop(Exception):
        pass

    def _stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(worker, "MemoryService", _stop)
    caplog.set_level(logging.INFO)
    with pytest.raises(_Stop):
        asyncio.run(worker.startup({}))
    assert "third_party_telemetry mem0=off graphiti=off" in caplog.text + capsys.readouterr().out


def test_telemetry_field_alone_does_not_turn_a_healthy_service_degraded(monkeypatch):
    import main
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setattr(main._task_manager, "pool", MagicMock(ping=AsyncMock(return_value=True)))
    monkeypatch.setattr(main._service, "_memory", object(), raising=False)
    monkeypatch.setattr(main._service, "_graphiti", object(), raising=False)
    monkeypatch.setattr(main, "_probe_vector_store", AsyncMock(return_value=None))
    monkeypatch.setattr(main, "_probe_graph_store", AsyncMock(return_value=None))
    resp = _health(monkeypatch, mem0=True, graphiti="true")
    body = resp.json()
    assert resp.status_code == 200
    assert body["checks"]["redis"] == body["checks"]["vector_store"] == body["checks"]["graph_store"] == "ok"
    assert body["status"] == "ok"
