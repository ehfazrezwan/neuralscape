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


def test_compose_check_command_passes_fresh_and_fails_stale_or_missing(tmp_path):
    cmd = _compose_check_command()
    if cmd is None:
        return
    hb = tmp_path / "hb"
    env = {**os.environ, "NS_WORKER_HEARTBEAT_FILE": str(hb)}  # never touch the real /tmp file
    assert subprocess.run(cmd, capture_output=True, env=env).returncode != 0  # missing
    hb.touch()
    assert subprocess.run(cmd, capture_output=True, env=env).returncode == 0  # fresh
    _age(hb, 600)
    assert subprocess.run(cmd, capture_output=True, env=env).returncode != 0  # stale


def test_check_command_bound_matches_module_constant():
    cmd = _compose_check_command()
    if cmd is None:
        return
    assert f"<{wl.HEARTBEAT_MAX_AGE_S} " in cmd[-1]
    # Every worker in compose and every Helm probe carries the same literal bound.
    data = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    for name in ("neuralscape-worker", "neuralscape-graph-worker", "neuralscape-ingest-worker"):
        assert f"<{wl.HEARTBEAT_MAX_AGE_S} " in data["services"][name]["healthcheck"]["test"][-1], name
    tmpl = REPO_ROOT / "deploy" / "charts" / "neuralscape" / "templates"
    for f in ("worker", "graph-worker", "ingest-worker"):
        text = (tmpl / f"{f}-deployment.yaml").read_text(encoding="utf-8")
        assert text.count(f"<{wl.HEARTBEAT_MAX_AGE_S} ") == 2, f"{f}: startup+liveness probes must use the bound"
    assert wl.HEARTBEAT_FILE.endswith("ns-worker.heartbeat") and "ns-worker.heartbeat" in cmd[-1]
    assert "NS_WORKER_HEARTBEAT_FILE" in cmd[-1]


def test_heartbeat_file_path_is_configurable_by_env(tmp_path):
    code = "import worker_liveness as w; print(w.HEARTBEAT_FILE)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env={**os.environ, "NS_WORKER_HEARTBEAT_FILE": str(tmp_path / "x")},
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert out.stdout.strip() == str(tmp_path / "x")


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


# ── the beat survives a slow sync call inside each wrapped task ──────


async def _beats_during(tmp_path, coro_factory, slow_s=0.4):
    """Run ``coro_factory()`` and count heartbeat mtime changes while it runs."""
    hb = tmp_path / "hb"
    ctx: dict = {}
    wl.start_heartbeat(ctx, str(hb), interval=0.02)
    mtimes = set()

    async def watch():
        while True:
            mtimes.add(hb.stat().st_mtime_ns)
            await asyncio.sleep(0.01)

    watcher = asyncio.get_running_loop().create_task(watch())
    try:
        await coro_factory()
    finally:
        watcher.cancel()
        await wl.stop_heartbeat(ctx)
    return len(mtimes)


@pytest.mark.asyncio
async def test_heartbeat_continues_while_startup_warmup_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(wl, "HEARTBEAT_FILE", str(tmp_path / "hb"))
    monkeypatch.setattr(wl, "HEARTBEAT_INTERVAL_S", 0.02)
    monkeypatch.setattr(wl.start_heartbeat, "__defaults__", (None, 0.02))

    class Slow:
        def _get_memory(self):
            time.sleep(0.4)

        def close(self):
            pass

    import extensions
    import hybrid_inference

    monkeypatch.setattr(worker, "MemoryService", Slow)
    monkeypatch.setattr(worker.settings, "connectors_enabled", False, raising=False)
    monkeypatch.setattr(hybrid_inference, "warmup_extraction", lambda: None)

    class Reg:
        async def discover(self): ...
        async def startup_all(self): ...
        async def shutdown_all(self): ...

    monkeypatch.setattr(extensions, "ExtensionRegistry", Reg)
    ctx: dict = {}
    hb = tmp_path / "hb"
    seen = set()

    async def watch():
        while True:
            if hb.exists():
                seen.add(hb.stat().st_mtime_ns)
            await asyncio.sleep(0.01)

    watcher = asyncio.get_running_loop().create_task(watch())
    await worker.startup(ctx)
    watcher.cancel()
    await worker.shutdown(ctx)
    assert len(seen) >= 5, "heartbeat stalled during the sync warm-up"


@pytest.mark.asyncio
async def test_heartbeat_continues_during_slow_raw_write_dedup_search(tmp_path):
    from unittest.mock import AsyncMock, MagicMock

    svc = MagicMock()
    svc.search.side_effect = lambda **kw: (time.sleep(0.4), [])[1]  # retry_transient-style sleep
    svc.store_raw.return_value = ([], False)
    ctx = {"service": svc, "redis": MagicMock(enqueue_job=AsyncMock())}
    n = await _beats_during(
        tmp_path, lambda: worker.process_memory_raw(ctx, content="x", user_id="u", category="task_context")
    )
    assert n >= 5, "heartbeat stalled while service.search ran"


@pytest.mark.asyncio
async def test_heartbeat_continues_during_graph_enrichment_row_lookup(tmp_path, monkeypatch):
    from types import SimpleNamespace

    class Svc:
        def get_memory(self, memory_id):
            return SimpleNamespace(id=memory_id, visibility="private", occurred_at=None)

        def enrich_graph(self, **kw):
            return True

    def slow_lookup(service, memory_id):
        time.sleep(0.4)

    monkeypatch.setattr(worker, "_row_episode_uuid", slow_lookup)
    n = await _beats_during(
        tmp_path,
        lambda: worker.process_graph_enrichment({"service": Svc()}, "m1", "c", "alice", None, "private"),
    )
    assert n >= 5, "heartbeat stalled during the Qdrant row lookup"


@pytest.mark.asyncio
async def test_beat_task_survives_unexpected_errors_and_logs(tmp_path, monkeypatch, caplog):
    caplog.set_level("WARNING", logger="worker_liveness")
    calls = {"n": 0}

    def flaky(path=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("boom")

    monkeypatch.setattr(wl, "beat", flaky)
    ctx: dict = {}
    wl.start_heartbeat(ctx, str(tmp_path / "hb"), interval=0.02)
    await asyncio.sleep(0.15)
    assert calls["n"] >= 3 and not ctx["_liveness_task"].done()
    assert "worker heartbeat write failed" in caplog.text
    await wl.stop_heartbeat(ctx)


@pytest.mark.asyncio
async def test_heartbeat_continues_during_strategy_synthesizer_scroll(tmp_path, monkeypatch):
    from extensions.strategy_synthesizer import synthesizer as syn
    from extensions.strategy_synthesizer.config import StrategySynthesizerSettings

    def slow_scroll(service):
        time.sleep(0.4)  # the sync Qdrant scroll
        return []

    monkeypatch.setattr(syn, "_scroll_trading_memories", slow_scroll)
    cfg = StrategySynthesizerSettings(enabled=True)
    n = await _beats_during(tmp_path, lambda: syn.synthesize_all(service=object(), settings=cfg))
    assert n >= 5, "heartbeat stalled during the synthesizer's Qdrant scroll"


# ── test-only heartbeat stall hook ───────────────────────────────────


@pytest.mark.asyncio
async def test_stall_file_pauses_the_beat_and_removing_it_resumes(tmp_path, monkeypatch):
    stall = tmp_path / "stall"
    monkeypatch.setenv(wl.STALL_ENV, str(stall))
    monkeypatch.setattr(worker.settings, "neuralscape_public_url", "")
    hb = tmp_path / "hb"
    ctx: dict = {}
    wl.start_heartbeat(ctx, str(hb), interval=0.02)
    await asyncio.sleep(0.1)
    assert wl.is_fresh(str(hb), max_age=1)
    stall.touch()
    await asyncio.sleep(0.05)
    _age(hb, 100)
    await asyncio.sleep(0.15)
    assert not wl.is_fresh(str(hb), max_age=50), "beat must stop while the stall file exists"
    stall.unlink()
    await asyncio.sleep(0.15)
    assert wl.is_fresh(str(hb), max_age=5), "beat must resume once the stall file is removed"
    await wl.stop_heartbeat(ctx)


@pytest.mark.asyncio
async def test_stall_hook_is_refused_and_warned_when_a_public_url_is_set(tmp_path, monkeypatch, caplog):
    stall = tmp_path / "stall"
    stall.touch()
    monkeypatch.setenv(wl.STALL_ENV, str(stall))
    monkeypatch.setattr(worker.settings, "neuralscape_public_url", "https://ns.example.com")
    monkeypatch.setattr(wl, "_stall_refused_logged", False)
    caplog.set_level("WARNING", logger="worker_liveness")
    hb = tmp_path / "hb"
    ctx: dict = {}
    wl.start_heartbeat(ctx, str(hb), interval=0.02)
    await asyncio.sleep(0.1)
    _age(hb, 100)
    await asyncio.sleep(0.1)
    assert wl.is_fresh(str(hb), max_age=5), "stall must be ignored in a public deployment"
    assert wl.STALL_ENV in caplog.text and "NEURALSCAPE_PUBLIC_URL" in caplog.text
    await wl.stop_heartbeat(ctx)


def test_stall_hook_is_inert_without_the_env_var(monkeypatch):
    monkeypatch.delenv(wl.STALL_ENV, raising=False)
    assert wl._stall_requested() is False


@pytest.mark.asyncio
async def test_second_opt_in_allows_the_stall_with_a_public_url(tmp_path, monkeypatch):
    stall = tmp_path / "stall"
    stall.touch()
    monkeypatch.setenv(wl.STALL_ENV, str(stall))
    monkeypatch.setenv(wl.ALLOW_PUBLIC_ENV, "1")
    monkeypatch.setattr(worker.settings, "neuralscape_public_url", "https://ns.example.com")
    assert wl._stall_requested() is True
    monkeypatch.setenv(wl.ALLOW_PUBLIC_ENV, "0")
    assert wl._stall_requested() is False
