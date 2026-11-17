"""Self-tests for the autoheal scenario's plumbing: the two destructive stall `exec` forms (and every refusal),
the read-only `docker inspect` helper, the pure recovery/wiring checks (with a mutation proof that the negative
control bites), the stall driver's cleanup discipline, and the profile file. No stack, no docker."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nsaccept import guard, profile, restore, runctx

ACCEPTANCE = Path(__file__).resolve().parents[1]
STALL = restore.STALL_PATH
CID = "0123456789ab"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("NS_ALLOW_DESTRUCTIVE", "NS_APP_SERVICES", "NS_COMPOSE_FILES", "NS_COMPOSE_ALLOWLIST", "NS_REDIS_SERVICE"):
        monkeypatch.delenv(k, raising=False)


def fake(calls, rc=0):
    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, rc, "out", "err")
    return run


APP = ("neuralscape", "neuralscape-worker", "neuralscape-graph-worker", "neuralscape-ingest-worker")


# ── the new allowlist forms ──────────────────────────────────────


@pytest.mark.parametrize("service", APP)
def test_touch_and_rm_f_of_the_stall_file_run_in_an_app_service_with_the_destructive_flag(monkeypatch, service):
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    calls = []
    runctx.run_compose("nstest-acceptance", "exec", "-T", service, "touch", STALL, run=fake(calls))
    runctx.run_compose("nstest-acceptance", "exec", "-T", service, "rm", "-f", STALL, run=fake(calls))
    assert calls == [
        ["docker", "compose", "-p", "nstest-acceptance", "exec", "-T", service, "touch", STALL],
        ["docker", "compose", "-p", "nstest-acceptance", "exec", "-T", service, "rm", "-f", STALL],
    ]


def test_the_stall_forms_are_refused_without_the_destructive_flag():
    calls = []
    for args in (("exec", "-T", "neuralscape-worker", "touch", STALL), ("exec", "-T", "neuralscape-worker", "rm", "-f", STALL)):
        with pytest.raises(runctx.ConfigError):
            runctx.run_compose("nstest-acceptance", *args, run=fake(calls))
    assert calls == []


def test_the_stall_forms_are_refused_when_the_flag_is_anything_but_1(monkeypatch):
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "true")
    with pytest.raises(runctx.ConfigError):
        runctx.run_compose("nstest-acceptance", "exec", "-T", "neuralscape-worker", "touch", STALL, run=fake([]))


@pytest.mark.parametrize("service", ["redis", "neo4j", "qdrant", "autoheal", "docker-socket-proxy", "neuralscape-worker2", "", " neuralscape-worker"])
def test_the_stall_forms_are_refused_for_any_other_service(monkeypatch, service):
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    for tail in (("touch", STALL), ("rm", "-f", STALL)):
        with pytest.raises(runctx.ConfigError):
            runctx.run_compose("nstest-acceptance", "exec", "-T", service, *tail, run=fake([]))


@pytest.mark.parametrize("path", [
    "/tmp/ns-worker.heartbeat", "/tmp/ns-test-stall-heartbeat", "/dev/shm/ns-worker.heartbeat", "/dev/shm/other",
    "/dev/shm/ns-test-stall-", "/dev/shm/ns-test-stall-a/../../etc/passwd", "/dev/shm/ns-test-stall-../x", "/dev/shm/../etc/passwd",
    "/dev/shm/ns-test-stall-heartbeat/", "/dev/shm/ns-test-stall-HEARTBEAT", "/dev/shm/ns-test-stall-" + "a" * 33,
    "dev/shm/ns-test-stall-heartbeat", "/dev/shm//ns-test-stall-heartbeat", "/dev/shm/ns-test-stall-x y", "/dev/shm/ns-test-stall-x;rm",
    "/dev/shm/ns-test-stall-x\n", "-f", "", "/",
])
def test_the_stall_forms_are_refused_for_any_other_path(monkeypatch, path):
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    for tail in (("touch", path), ("rm", "-f", path)):
        with pytest.raises(runctx.ConfigError):
            runctx.run_compose("nstest-acceptance", "exec", "-T", "neuralscape-worker", *tail, run=fake([]))


@pytest.mark.parametrize("args", [
    ("exec", "neuralscape-worker", "touch", STALL),  # no -T
    ("exec", "-T", "neuralscape-worker", "touch", "-t", "197001010000", STALL),
    ("exec", "-T", "neuralscape-worker", "touch", STALL, "/tmp/x"),
    ("exec", "-T", "neuralscape-worker", "touch"),
    ("exec", "-T", "neuralscape-worker", "rm", STALL),
    ("exec", "-T", "neuralscape-worker", "rm", "-rf", STALL),
    ("exec", "-T", "neuralscape-worker", "rm", "-f", "-r", STALL),
    ("exec", "-T", "neuralscape-worker", "rm", "-f", STALL, "/data"),
    ("exec", "-T", "neuralscape-worker", "rm", "-f"),
    ("exec", "-T", "neuralscape-worker", "sh", "-c", f"touch {STALL}"),
    ("exec", "-T", "neuralscape-worker", "cat", STALL),
    ("exec", "-T", "-u", "root", "neuralscape-worker", "touch", STALL),
    ("exec", "-T", "neuralscape-worker", "unlink", STALL),
])
def test_the_stall_forms_refuse_every_other_shape(monkeypatch, args):
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    with pytest.raises(runctx.ConfigError):
        runctx.run_compose("nstest-acceptance", *args, run=fake([]))


@pytest.mark.parametrize("destructive", [True, False])
@pytest.mark.parametrize("tail", [("touch", STALL), ("rm", "-f", STALL)])
@pytest.mark.parametrize("head", [
    ("exec", "neuralscape-worker"),  # no -T
    ("exec", "-T", "--privileged", "neuralscape-worker"),
    ("exec", "--privileged", "-T", "neuralscape-worker"),
    ("exec", "-d", "-T", "neuralscape-worker"),
    ("exec", "-T", "-d", "neuralscape-worker"),
    ("exec", "-T", "-u", "root", "neuralscape-worker"),
    ("exec", "-u", "root", "-T", "neuralscape-worker"),
    ("exec", "-T", "-e", "X=1", "neuralscape-worker"),
    ("exec", "-T", "-T", "neuralscape-worker"),
    ("exec", "-T", "--user=root", "neuralscape-worker"),
    ("exec", "-i", "neuralscape-worker"),
    ("exec", "-t", "neuralscape-worker"),
])
def test_the_stall_forms_need_exactly_exec_dash_T_and_no_other_flag(monkeypatch, head, tail, destructive):
    if destructive:
        monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    calls = []
    with pytest.raises(runctx.ConfigError):
        runctx.run_compose("nstest-acceptance", *head, *tail, run=fake(calls))
    assert calls == []
    assert not restore.stall_exec(head + tail)


@pytest.mark.parametrize("project", ["neuralscape", " NeuralScape ", "NEURALSCAPE", "prod", "nsother-x"])
def test_the_stall_forms_are_refused_for_the_production_project_and_foreign_projects(monkeypatch, project):
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    calls = []
    with pytest.raises(guard.GuardError):
        runctx.run_compose(project, "exec", "-T", "neuralscape-worker", "touch", STALL, run=fake(calls))
    assert calls == []


def test_the_stall_forms_follow_ns_app_services(monkeypatch):
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    monkeypatch.setenv("NS_APP_SERVICES", "api,w1")
    runctx.run_compose("nstest-x", "exec", "-T", "w1", "touch", STALL, run=fake([]))
    with pytest.raises(runctx.ConfigError):
        runctx.run_compose("nstest-x", "exec", "-T", "neuralscape-worker", "touch", STALL, run=fake([]))


def test_the_existing_destructive_and_read_only_forms_are_unchanged(monkeypatch):
    with pytest.raises(runctx.ConfigError):
        runctx.run_compose("nstest-x", "exec", "-T", "redis", "redis-cli", "FLUSHALL", run=fake([]))
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    runctx.run_compose("nstest-x", "exec", "-T", "redis", "redis-cli", "FLUSHALL", run=fake([]))
    for args in (("exec", "-T", "redis", "redis-cli", "FLUSHDB"), ("exec", "-T", "neuralscape", "sh", "-c", "id"), ("exec", "-T", "neuralscape", "ls")):
        with pytest.raises(runctx.ConfigError):
            runctx.run_compose("nstest-x", *args, run=fake([]))


def test_the_stall_path_is_inside_the_regex_and_the_profile_uses_it():
    assert restore.STALL_PATH_RE.fullmatch(STALL)
    p = profile.load(ACCEPTANCE / "profiles" / "autoheal.env")
    assert p.flags["NS_TEST_STALL_HEARTBEAT_FILE"] == STALL
    assert p.headers["requires"] == ["compose"] and p.headers["services"] == ["autoheal"]
    assert not p.placeholders and profile.REDACTED not in p.flags.values()


# ── docker inspect (read-only) ───────────────────────────────────

INSPECT_OUT = '2026-01-01T00:00:00.123456789Z|healthy|0|running|{"com.docker.compose.project":"nstest-x","com.neuralscape.autoheal.nstest-x":"true"}\n'


def test_container_inspect_reads_health_started_at_restarts_and_labels():
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, INSPECT_OUT, "")

    info = runctx.container_inspect(CID, run=run)
    assert info == {
        "started_at": "2026-01-01T00:00:00.123456789Z", "health": "healthy", "restart_count": 0, "status": "running",
        "labels": {"com.docker.compose.project": "nstest-x", "com.neuralscape.autoheal.nstest-x": "true"},
    }
    cmd = calls[0]
    assert cmd[:3] == ["docker", "inspect", "--format"] and cmd[-1] == CID and len(cmd) == 5
    assert "Config.Env" not in cmd[3] and ".Config." not in cmd[3].replace("Config.Labels", "")  # never the environment (secrets)


def test_container_inspect_with_no_healthcheck_or_labels_and_failures():
    assert runctx.parse_inspect("2026-01-01T00:00:00Z|none|3|running|null")["health"] == "none"
    assert runctx.parse_inspect("2026-01-01T00:00:00Z|none|3|running|null")["labels"] == {}
    with pytest.raises(RuntimeError):
        runctx.parse_inspect("garbage")
    with pytest.raises(RuntimeError):
        runctx.container_inspect(CID, run=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "no such container"))


@pytest.mark.parametrize("bad", ["", "nstest-worker", "neuralscape-worker", "ABCDEF123456", "0123456789a", "--help", "0123456789ab; rm -rf /", "$(id)" * 4, "a" * 65])
def test_container_inspect_only_takes_a_container_id(bad):
    calls = []
    with pytest.raises(runctx.ConfigError):
        runctx.container_inspect(bad, run=fake(calls))
    assert calls == []


# ── the scenario module: pure checks, negative control, driver ───


def _scenario():
    path = ACCEPTANCE / "scenarios" / "test_fault_injection_autoheal.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("autoheal_scenario_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


def test_the_negative_controls_pass_on_the_real_checks():
    mod = _scenario()
    mod.test_recovery_check_fails_without_a_restart_or_a_return_to_healthy()
    mod.test_wiring_check_fails_without_the_sidecar_or_the_project_scoped_label()


def test_the_negative_control_bites_when_the_check_accepts_anything(monkeypatch):
    """Mutation: a recovery/wiring check that accepts everything must make the controls fail."""
    mod = _scenario()
    monkeypatch.setattr(mod, "check_autoheal_recovery", lambda *a, **k: None)
    with pytest.raises(AssertionError, match="negative control"):
        mod.test_recovery_check_fails_without_a_restart_or_a_return_to_healthy()
    monkeypatch.setattr(mod, "check_autoheal_wiring", lambda *a, **k: None)
    with pytest.raises(AssertionError, match="negative control"):
        mod.test_wiring_check_fails_without_the_sidecar_or_the_project_scoped_label()


def test_the_negative_control_fails_its_precondition_when_the_good_data_is_rejected(monkeypatch):
    mod = _scenario()

    def reject(*a, **k):
        raise AssertionError("always no")

    monkeypatch.setattr(mod, "check_autoheal_recovery", reject)
    with pytest.raises(AssertionError, match="PRECONDITION FAILED"):
        mod.test_recovery_check_fails_without_a_restart_or_a_return_to_healthy()


class FakeCtx:
    """Records compose calls; ``rm_rc`` scripts the return code of ``rm -f`` calls."""

    def __init__(self, rm_rc=(0,)):
        self.calls, self.rm_rc, self.project_id = [], list(rm_rc), "acc-r-proj"

    def compose(self, *args, check=True, **kw):
        self.calls.append(args)
        rc = 0
        if args[:5] == ("exec", "-T", "neuralscape-worker", "rm", "-f"):
            rc = self.rm_rc.pop(0) if len(self.rm_rc) > 1 else self.rm_rc[0]
        return subprocess.CompletedProcess(args, rc, "autoheal log line", "")


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _inspector(script, default):
    seq = list(script)

    def inspect(cid):
        return seq.pop(0) if seq else default

    return inspect


def _states(mod):
    b = mod._BEFORE
    r = mod._RESTARTED
    return b, r


def _drive(mod, ctx, inspect, **kw):
    clock = Clock()
    return mod.drive_autoheal_cycle(ctx, "neuralscape-worker", CID, mod._BEFORE, inspect=inspect, sleep=clock.sleep, clock=clock.now, **kw)


def test_the_driver_stalls_waits_for_the_restart_releases_and_checks_recovery(monkeypatch):
    mod = _scenario()
    b, r = _states(mod)
    ctx = FakeCtx()
    inspect = _inspector([b, {**b, "health": "unhealthy"}, {**b, "health": "unhealthy"}, {**r, "health": "starting"}], {**r, "health": "healthy"})
    samples = _drive(mod, ctx, inspect)
    assert samples[-1]["health"] == "healthy" and samples[-1]["started_at"] == r["started_at"]
    assert ctx.calls[0] == ("exec", "-T", "neuralscape-worker", "touch", STALL)
    assert ("exec", "-T", "neuralscape-worker", "rm", "-f", STALL) in ctx.calls
    assert not any(c[0] == "restart" for c in ctx.calls), "a healed worker must not be restarted by the harness"
    assert sum(1 for c in ctx.calls if c[3:5] == ("rm", "-f")) == 1


def test_when_autoheal_never_acts_the_stall_is_removed_the_worker_restarted_and_the_scenario_fails(monkeypatch):
    mod = _scenario()
    b, _ = _states(mod)
    waited = []
    monkeypatch.setattr(mod.stack, "wait_services", lambda ctx, services, what, **k: waited.append((services, what)))
    ctx = FakeCtx()
    with pytest.raises(AssertionError, match="autoheal did not restart neuralscape-worker"):
        _drive(mod, ctx, _inspector([], {**b, "health": "unhealthy"}), restart_wait=20)
    assert ctx.calls[0] == ("exec", "-T", "neuralscape-worker", "touch", STALL)
    removed = [i for i, c in enumerate(ctx.calls) if c[3:5] == ("rm", "-f")]
    restarted = [i for i, c in enumerate(ctx.calls) if c[0] == "restart"]
    assert removed and restarted and removed[0] < restarted[0], "the stall file must go before the harness restarts the worker"
    assert ("logs", "--tail", "30", mod.AUTOHEAL) in ctx.calls
    assert waited and waited[0][0] == ["neuralscape-worker"]


def test_a_restart_that_is_never_healthy_again_fails_and_still_restores_the_worker(monkeypatch):
    mod = _scenario()
    b, r = _states(mod)
    monkeypatch.setattr(mod.stack, "wait_services", lambda *a, **k: None)
    ctx = FakeCtx()
    inspect = _inspector([b, {**b, "health": "unhealthy"}], {**r, "health": "starting"})
    with pytest.raises(AssertionError, match="not healthy again"):
        _drive(mod, ctx, inspect, healthy_wait=30)
    assert any(c[0] == "restart" for c in ctx.calls)


def test_a_failed_cleanup_does_not_mask_the_scenarios_own_failure(monkeypatch, capsys):
    mod = _scenario()
    b, _ = _states(mod)

    def broken(*a, **k):
        raise RuntimeError("compose is gone")

    monkeypatch.setattr(mod.stack, "wait_services", broken)
    ctx = FakeCtx(rm_rc=(1,))
    with pytest.raises(AssertionError, match="autoheal did not restart"):
        _drive(mod, ctx, _inspector([], {**b, "health": "unhealthy"}), restart_wait=10)
    assert "WARNING" in capsys.readouterr().out


def test_cleanup_failure_after_a_clean_run_fails_loudly(monkeypatch):
    """The happy sequence, but the stall file cannot be removed: the scenario must not pass quietly."""
    mod = _scenario()
    b, r = _states(mod)
    ctx = FakeCtx(rm_rc=(1,))
    inspect = _inspector([b, {**b, "health": "unhealthy"}], {**r, "health": "healthy"})
    with pytest.raises(BaseException) as exc:  # pytest.fail raises Failed (a BaseException)
        _drive(mod, ctx, inspect)
    assert "could not remove" in str(exc.value)


def test_the_scenario_fails_not_skips_when_its_preconditions_are_missing(monkeypatch):
    mod = _scenario()
    ctx = SimpleNamespace(compose_project="nstest-x", profile_flags={}, run_id="r", project_id="p")
    with pytest.raises(AssertionError, match="NS_ALLOW_DESTRUCTIVE"):
        mod.test_autoheal_restarts_a_wedged_worker_and_it_takes_work_again(ctx, object())
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    with pytest.raises(AssertionError, match="NS_TEST_STALL_HEARTBEAT_FILE"):
        mod.test_autoheal_restarts_a_wedged_worker_and_it_takes_work_again(ctx, object())
    ctx.compose_project = ""
    with pytest.raises(AssertionError, match="--compose-project"):
        mod.test_autoheal_restarts_a_wedged_worker_and_it_takes_work_again(ctx, object())
    ctx.compose_project = "neuralscape"
    with pytest.raises(guard.GuardError):
        mod.test_autoheal_restarts_a_wedged_worker_and_it_takes_work_again(ctx, object())


def test_the_scenario_fails_when_the_sidecar_is_not_in_the_stack(monkeypatch):
    mod = _scenario()
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    monkeypatch.setattr(mod.stack, "ps_rows", lambda ctx, services: [{"Service": mod.WORKER, "ID": CID, "State": "running"}])
    monkeypatch.setattr(mod, "container_inspect", lambda cid: {"labels": {mod.label_key("nstest-x"): "true"}, "health": "healthy"})
    ctx = SimpleNamespace(compose_project="nstest-x", profile_flags={"NS_TEST_STALL_HEARTBEAT_FILE": STALL}, run_id="r", project_id="p")
    with pytest.raises(AssertionError, match="no `autoheal` service"):
        mod.test_autoheal_restarts_a_wedged_worker_and_it_takes_work_again(ctx, object())
