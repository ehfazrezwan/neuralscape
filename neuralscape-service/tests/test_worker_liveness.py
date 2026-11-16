"""Worker event-loop heartbeat + the healthcheck command that reads it."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

import worker
import worker_liveness as wl

REPO_ROOT = Path(__file__).resolve().parents[2]


def _age(path: Path, seconds: float) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


@pytest.mark.asyncio
async def test_heartbeat_task_refreshes_the_file_on_an_interval(tmp_path):
    hb = tmp_path / "hb"
    ctx: dict = {}
    wl.start_heartbeat(ctx, str(hb), interval=0.05)
    assert hb.exists()
    _age(hb, 100)
    stale = hb.stat().st_mtime
    await asyncio.sleep(0.2)
    assert hb.stat().st_mtime > stale + 50
    await wl.stop_heartbeat(ctx)


@pytest.mark.asyncio
async def test_heartbeat_stops_while_the_loop_is_blocked(tmp_path):
    hb = tmp_path / "hb"
    ctx: dict = {}
    wl.start_heartbeat(ctx, str(hb), interval=0.05)
    await asyncio.sleep(0.1)
    _age(hb, 100)
    time.sleep(0.3)  # sync work on the loop: no beat can run
    assert time.time() - hb.stat().st_mtime > 90
    await wl.stop_heartbeat(ctx)


@pytest.mark.asyncio
async def test_heartbeat_continues_during_a_long_threaded_job(tmp_path):
    hb = tmp_path / "hb"
    ctx: dict = {}
    wl.start_heartbeat(ctx, str(hb), interval=0.05)
    await asyncio.sleep(0.1)
    _age(hb, 100)
    await asyncio.to_thread(time.sleep, 0.3)
    assert wl.is_fresh(str(hb), max_age=5)
    await wl.stop_heartbeat(ctx)


def test_is_fresh_fresh_stale_and_missing(tmp_path):
    hb = tmp_path / "hb"
    assert not wl.is_fresh(str(hb))
    wl.beat(str(hb))
    assert wl.is_fresh(str(hb))
    _age(hb, wl.HEARTBEAT_MAX_AGE_S + 5)
    assert not wl.is_fresh(str(hb))


def _compose_check_command() -> list[str] | None:
    """None in the packaged test image (no repo-level compose files): callers return early."""
    path = REPO_ROOT / "docker-compose.yml"
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    test = data["services"]["neuralscape-worker"]["healthcheck"]["test"]
    assert test[0] == "CMD"
    return [sys.executable, "-I", *test[2:]]


def test_compose_check_command_passes_fresh_and_fails_stale_or_missing():
    cmd = _compose_check_command()
    if cmd is None:
        return
    hb = Path(re.search(r"'(/tmp/[^']+)'", cmd[-1]).group(1))
    backup = hb.read_bytes() if hb.exists() else None
    try:
        hb.unlink(missing_ok=True)
        assert subprocess.run(cmd, capture_output=True).returncode != 0  # missing
        hb.touch()
        assert subprocess.run(cmd, capture_output=True).returncode == 0  # fresh
        _age(hb, 600)
        assert subprocess.run(cmd, capture_output=True).returncode != 0  # stale
    finally:
        hb.unlink(missing_ok=True)
        if backup is not None:
            hb.write_bytes(backup)


def test_check_command_bound_matches_module_constant():
    cmd = _compose_check_command()
    if cmd is None:
        return
    assert f"<{wl.HEARTBEAT_MAX_AGE_S} " in cmd[-1]
    assert wl.HEARTBEAT_FILE.endswith("ns-worker.heartbeat") and "ns-worker.heartbeat" in cmd[-1]


@pytest.mark.asyncio
async def test_startup_starts_the_heartbeat_and_shutdown_cancels_it(tmp_path, monkeypatch):
    hb = tmp_path / "hb"
    monkeypatch.setattr(wl, "HEARTBEAT_FILE", str(hb))
    started: dict = {}

    class FakeService:
        def _get_memory(self):
            started["warm"] = hb.exists()  # beat is already running before warm-up

        def close(self):
            pass

    monkeypatch.setattr(worker, "MemoryService", FakeService)
    monkeypatch.setattr(worker.settings, "connectors_enabled", False, raising=False)
    import hybrid_inference
    import extensions

    monkeypatch.setattr(hybrid_inference, "warmup_extraction", lambda: None)

    class FakeRegistry:
        async def discover(self): ...
        async def startup_all(self): ...
        async def shutdown_all(self): ...

    monkeypatch.setattr(extensions, "ExtensionRegistry", FakeRegistry)
    ctx: dict = {}
    await worker.startup(ctx)
    task = ctx["_liveness_task"]
    assert started["warm"] and not task.done()
    await worker.shutdown(ctx)
    assert task.cancelled() and "_liveness_task" not in ctx


def test_all_three_worker_settings_use_the_heartbeat_hooks():
    for cls in (worker.WorkerSettings, worker.GraphWorkerSettings, worker.IngestWorkerSettings):
        assert cls.on_startup is worker.startup and cls.on_shutdown is worker.shutdown
