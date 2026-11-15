"""Safety guards of the service baseline harness: production project refusal, loopback-only stacks, degradation.

Every test stubs the container tool: a refusal must happen before any docker or subprocess call.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from neuralscape_bench.service_baseline import journey_safety, runner
from neuralscape_bench.service_baseline.journey_safety import JourneySafetyError

_WHEEL_STACK = Path(runner.__file__).resolve().parent / "stack"
STACK = _WHEEL_STACK if _WHEEL_STACK.is_dir() else Path(runner.COMPOSE_FILE).parent


def load_script(name: str, monkeypatch):
    spec = importlib.util.spec_from_file_location(f"_ns_bench_{name}", STACK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "path", list(sys.path))  # the scripts prepend to sys.path
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _reset_prefixes():
    reset = getattr(journey_safety, "reset_extra_prefixes", lambda: None)
    reset()
    yield
    reset()


@pytest.fixture
def no_docker(monkeypatch):
    """Any subprocess start is recorded; none ever runs."""

    calls: list[object] = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError(f"a docker/subprocess call was made: {args!r}")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    return calls


PRODUCTION_SPELLINGS = ["neuralscape", "Neuralscape", "NeuralScape", "NEURALSCAPE", " neuralscape ", "neuralscape\n"]
NOT_DISPOSABLE = ["", "   ", "prod", "ns-benchmark", "mystack-1", "ns-bench", "ns-bench-UP", "ns-bench-a b", "my-neuralscape"]


@pytest.mark.parametrize("name", PRODUCTION_SPELLINGS + NOT_DISPOSABLE + [None, 7])
def test_only_a_disposable_project_name_is_accepted(name):
    with pytest.raises(JourneySafetyError):
        journey_safety.require_disposable_project(name)


@pytest.mark.parametrize("name", ["nscampaign", "nscampaign-x", "nstest-a", "ns-bench-x", "ns-bench-baseline"])
def test_the_default_allowlist_accepts_disposable_prefixes(name):
    assert journey_safety.require_disposable_project(name) == name


def test_a_prefix_added_by_option_is_accepted_and_unsafe_prefixes_are_not():
    with pytest.raises(JourneySafetyError):
        journey_safety.require_disposable_project("mystack-1")
    journey_safety.allow_prefix("mystack-")
    assert journey_safety.require_disposable_project("mystack-1") == "mystack-1"
    for bad in ("", "ne", "neuralscape", "neuralscape-x", "Mystack-", None):
        with pytest.raises(JourneySafetyError):
            journey_safety.allow_prefix(bad)
    with pytest.raises(JourneySafetyError, match="production"):
        journey_safety.require_disposable_project("NeuralScape")


def test_the_runner_takes_the_project_as_a_parameter_with_the_allowlist(monkeypatch, no_docker, tmp_path):
    base = ["runner", "--image", "x", "--tree", "0" * 40, "--lock-sha256", "0" * 64, "--label", "l", "--out-dir", str(tmp_path / "out")]
    monkeypatch.setattr(runner, "PROJECT", runner.PROJECT)  # restored after the test
    monkeypatch.setattr(sys, "argv", [*base, "--project", "mystack-1"])
    with pytest.raises(JourneySafetyError, match="not a disposable"):
        runner.main()
    monkeypatch.setattr(sys, "argv", [*base, "--project", "NeuralScape", "--allow-project-prefix", "mystack-"])
    with pytest.raises(JourneySafetyError, match="production"):
        runner.main()

    class Reached(Exception):
        pass

    def stop(image):
        raise Reached

    monkeypatch.setattr(runner, "image_facts", stop)
    monkeypatch.setattr(runner, "COMPOSE_FILE", STACK / "compose.yml")
    monkeypatch.setattr(sys, "argv", [*base, "--project", "mystack-1", "--allow-project-prefix", "mystack-"])
    with pytest.raises(Reached):
        runner.main()
    assert runner.PROJECT == "mystack-1" and no_docker == []


def test_the_harness_own_project_names_are_accepted():
    assert journey_safety.require_disposable_project(runner.PROJECT) == runner.PROJECT
    assert journey_safety.require_disposable_project("ns-bench-journey") == "ns-bench-journey"


@pytest.mark.parametrize("name", PRODUCTION_SPELLINGS + ["", "other", "neuralscape_state"])
def test_the_runner_builds_no_compose_command_for_the_production_project(monkeypatch, no_docker, name):
    monkeypatch.setattr(runner, "PROJECT", name)
    with pytest.raises(JourneySafetyError):
        runner.compose_argv("down", "-v")
    with pytest.raises(JourneySafetyError):
        runner.compose("down", "-v", env={})
    with pytest.raises(JourneySafetyError):
        runner.assert_network_internal()
    assert no_docker == []


def test_the_runner_refuses_the_production_project_before_parsing_a_single_stack_file(monkeypatch, no_docker, tmp_path):
    monkeypatch.setattr(runner, "PROJECT", runner.PROJECT)  # restored after the test
    monkeypatch.setattr(sys, "argv", ["runner", "--image", "x", "--tree", "0" * 40, "--lock-sha256", "0" * 64,
                                      "--label", "l", "--out-dir", str(tmp_path / "out"), "--project", "NeuralScape"])
    with pytest.raises(JourneySafetyError, match="production"):
        runner.main()
    assert no_docker == [] and not (tmp_path / "out").exists()


class RecordingDocker:
    def __init__(self):
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        raise AssertionError(f"docker was called: {args!r}")


@pytest.mark.parametrize("name", PRODUCTION_SPELLINGS + [""])
def test_preflight_and_report_refuse_the_production_project_without_calling_docker(name):
    docker = RecordingDocker()
    with pytest.raises(JourneySafetyError):
        journey_safety.preflight(docker, name, "ns-bench-sentinel")
    with pytest.raises(JourneySafetyError):
        journey_safety.uninstall_report([], [], [], name, "ns-bench-sentinel", [])
    assert docker.calls == []


@pytest.mark.parametrize("sentinel", ["", "data", "neuralscape_state", "ns-bench-journey_"])
def test_the_sentinel_volume_must_be_a_benchmark_volume(sentinel):
    docker = RecordingDocker()
    with pytest.raises(JourneySafetyError):
        journey_safety.preflight(docker, "ns-bench-journey", sentinel)
    assert docker.calls == []


def _scripted_sh(monkeypatch, project_ids, all_ids):
    calls: list[tuple[str, ...]] = []

    def fake(*args, **kwargs):
        calls.append(args)
        if args[:2] == ("docker", "ps") and any(a.startswith("label=com.docker.compose.project=") for a in args):
            return "\n".join(project_ids)
        if args[:2] == ("docker", "ps"):
            return "\n".join(all_ids)
        return ""

    monkeypatch.setattr(runner, "sh", fake)
    return calls


def test_listing_and_stats_are_scoped_to_the_run_project(monkeypatch):
    calls = _scripted_sh(monkeypatch, ["p1", "p2"], ["p1", "p2", "x1"])
    sampler = runner.StatsSampler()
    sampler.stop.set()
    sampler.stop = type("OneShot", (), {"is_set": lambda self, n=[0]: n.append(1) or len(n) > 2, "set": lambda self: None})()
    sampler.run()
    listings = [c for c in calls if c[:2] == ("docker", "ps")]
    assert any(f"label=com.docker.compose.project={runner.PROJECT}" in c for c in listings)
    stats = [c for c in calls if c[:2] == ("docker", "stats")]
    assert stats and all(c[-2:] == ("p1", "p2") for c in stats)  # ids of the project only
    assert all("{{.Names}}" not in " ".join(c) for c in calls)  # no listing that returns container names


def test_stats_is_never_run_unscoped_when_the_project_has_no_container(monkeypatch):
    calls = _scripted_sh(monkeypatch, [], ["x1", "x2"])
    sampler = runner.StatsSampler()
    sampler.stop = type("OneShot", (), {"is_set": lambda self, n=[0]: n.append(1) or len(n) > 2, "set": lambda self: None})()
    sampler.run()
    assert not [c for c in calls if c[:2] == ("docker", "stats")]  # a bare `docker stats` samples the whole host


def test_the_other_container_check_records_a_count_and_no_foreign_names(monkeypatch):
    calls = _scripted_sh(monkeypatch, ["p1"], ["p1", "foreign-secret-id", "other-id"])
    assert runner.foreign_container_count() == 2
    sampler = runner.StatsSampler()
    sampler.stop = type("OneShot", (), {"is_set": lambda self, n=[0]: n.append(1) or len(n) > 2, "set": lambda self: None})()
    sampler.run()
    assert sampler.other_containers == 2 and isinstance(sampler.other_containers, int)
    assert all("{{.Names}}" not in " ".join(c) for c in calls)
    assert "foreign" not in repr(sampler.__dict__)
    from neuralscape_bench.service_baseline import report

    assert report.is_clean({"host_before": {"other_running_containers": 0}, "host_after": {"other_running_containers": 0},
                            "other_containers_during_run": 0})
    assert not report.is_clean({"host_before": {"other_running_containers": 0}, "other_containers_during_run": 2})


def test_the_journey_never_starts_docker_for_the_production_project(monkeypatch, no_docker):
    journey = load_script("journey", monkeypatch)
    journey_safety.require_disposable_volume(journey.SENTINEL, journey.PROJECT)
    monkeypatch.setattr(journey, "PROJECT", "neuralscape")
    walker = journey.Journey("old", "new", "/nonexistent/out.json")
    try:
        with pytest.raises(JourneySafetyError):
            walker.compose("down", "-v", "--remove-orphans")
        with pytest.raises(JourneySafetyError):
            journey.project_mounts()
        with pytest.raises(SystemExit, match="production"):
            journey.begin(walker)
    finally:
        import shutil

        shutil.rmtree(walker.scratch, ignore_errors=True)
    assert no_docker == []


def test_the_journey_down_v_command_names_its_own_disposable_project(monkeypatch):
    journey = load_script("journey", monkeypatch)
    seen = []
    monkeypatch.setattr(journey, "run", lambda *args, **kwargs: seen.append(args) or None)
    walker = journey.Journey("old", "new", "/nonexistent/out.json")
    try:
        walker.compose("down", "-v", "--remove-orphans")
    finally:
        import shutil

        shutil.rmtree(walker.scratch, ignore_errors=True)
    argv = seen[0]
    assert argv[:4] == ("docker", "compose", "-p", "ns-bench-journey") and "down" in argv


# ---- loopback-only publishing


def test_the_shipped_stack_files_publish_nothing_and_are_loopback_only():
    for name in ("compose.yml", "journey.override.yml", "compose.no-egress.yml"):
        text = (STACK / name).read_text()
        assert journey_safety.local_only_violations(text) == [], name
        assert "ports:" not in text, f"{name} publishes host ports"
        journey_safety.require_local_only(text, name)


@pytest.mark.parametrize("text", [
    "services:\n  api:\n    ports:\n      - \"8199:8199\"\n",
    "services:\n  api:\n    ports:\n      - 0.0.0.0:8199:8199\n",
    "services:\n  api:\n    ports: [\"8199:8199\"]\n",
    "services:\n  api:\n    ports:\n      - target: 8199\n        published: 8199\n",
    "services:\n  api:\n    ports:\n      - 127.0.0.1:1:1\n      - 9000:9000\n",
    "services:\n  api:\n    network_mode: host\n",
    "services:\n  api:\n    privileged: true\n",
    "services:\n  api:\n    volumes:\n      - /var/run/docker.sock:/var/run/docker.sock\n",
])
def test_a_stack_that_publishes_beyond_loopback_or_reaches_the_host_is_refused(text):
    assert journey_safety.local_only_violations(text)
    with pytest.raises(JourneySafetyError, match="loopback-only"):
        journey_safety.require_local_only(text)


@pytest.mark.parametrize("text", [
    "services:\n  api:\n    ports:\n      - \"127.0.0.1:8199:8199\"\n      - '[::1]:9000:9000'\n    image: x\n",
    "services:\n  api:\n    ports: []\n    image: x\n",
])
def test_loopback_publishing_is_allowed(text):
    assert journey_safety.local_only_violations(text) == []


def test_the_runner_refuses_a_non_loopback_override_before_any_docker_call(monkeypatch, no_docker, tmp_path):
    override = tmp_path / "o.yml"
    override.write_text("services:\n  api:\n    ports:\n      - \"8199:8199\"\n")
    monkeypatch.setattr(runner, "COMPOSE_FILE", STACK / "compose.yml")  # the wheel has no checkout beside it
    monkeypatch.setattr(sys, "argv", ["runner", "--image", "x", "--tree", "0" * 40, "--lock-sha256", "0" * 64,
                                      "--label", "l", "--out-dir", str(tmp_path / "out"), "--compose-override", str(override)])
    with pytest.raises(JourneySafetyError, match="loopback-only"):
        runner.main()
    assert no_docker == [] and not (tmp_path / "out").exists()


# ---- the in-container scripts degrade cleanly when the governed service modules are absent


def _absent(monkeypatch, *names):
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)  # an import of it raises ImportError


def test_seed_authority_refuses_a_governed_arm_when_the_authority_modules_are_missing(monkeypatch):
    monkeypatch.setenv("AUTHORITY_ENABLED", "true")
    fake_tokens = type(sys)("tokens")
    fake_tokens.issue_user_token = lambda user, secret, ttl: f"token-for-{user}"
    monkeypatch.setitem(sys.modules, "tokens", fake_tokens)
    _absent(monkeypatch, "authority_store_sql", "contracts_policy", "identity_binding", "policy_enforcement")
    with pytest.raises(SystemExit) as refused:
        load_script("seed_authority", monkeypatch)
    assert "refusing to seed" in str(refused.value.code) and "authority_store_sql" in str(refused.value.code)


def test_seed_authority_without_the_flag_never_imports_the_authority_modules(monkeypatch):
    monkeypatch.delenv("AUTHORITY_ENABLED", raising=False)
    fake_tokens = type(sys)("tokens")
    fake_tokens.issue_user_token = lambda *a: "t"
    monkeypatch.setitem(sys.modules, "tokens", fake_tokens)
    _absent(monkeypatch, "authority_store_sql")
    assert load_script("seed_authority", monkeypatch).AUTHORITY is False


def test_the_api_launcher_refuses_the_enforcer_arm_when_the_modules_are_missing(monkeypatch):
    monkeypatch.setenv("HARNESS_WIRE_ENFORCER", "1")
    monkeypatch.setattr("os.chdir", lambda path: None)  # the service directory only exists in the image
    _absent(monkeypatch, "policy_enforcement", "authority_store_sql")
    with pytest.raises(SystemExit) as refused:
        load_script("harness_api", monkeypatch)
    assert "refusing to start" in str(refused.value.code)


def test_the_api_launcher_without_the_enforcer_flag_starts_without_the_modules(monkeypatch):
    monkeypatch.delenv("HARNESS_WIRE_ENFORCER", raising=False)
    monkeypatch.setattr("os.chdir", lambda path: None)
    _absent(monkeypatch, "policy_enforcement", "authority_store_sql")
    assert callable(load_script("harness_api", monkeypatch).main)


@pytest.mark.parametrize("argv", [["--help"], ["-h"]])
def test_help_and_a_bad_command_line_start_nothing(monkeypatch, no_docker, argv, capsys):
    journey = load_script("journey", monkeypatch)
    monkeypatch.setattr(sys, "argv", ["journey.py", *argv])
    with pytest.raises(SystemExit) as exit_info:
        journey.main()
    assert exit_info.value.code == 0 and "journeys" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["journey.py", "default"])
    with pytest.raises(SystemExit) as bad:
        journey.main()
    assert bad.value.code == 2
    assert no_docker == []
