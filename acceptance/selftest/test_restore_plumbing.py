"""Self-tests for the backup/restore command shapes: every new compose form is fixed,
every path is confined to the scenario's own temp directory, and each refusal is tested."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from nsaccept import flow as flow_mod, guard, restore, runctx, stack

STAMP = "ns-backup-20260101T000000Z"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("NS_COMPOSE_FILES", "NS_ALLOW_DESTRUCTIVE", "NS_API_SERVICE", "NS_REDIS_SERVICE", "NS_NEO4J_SERVICE",
              "NEO4J_DATABASE", "NS_REPO_ROOT", "NS_COMPOSE_ALLOWLIST"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def root():
    """A real nsc-backup-* directory under the temp dir holding a complete-looking backup."""
    r = Path(tempfile.mkdtemp(prefix=restore.BACKUP_ROOT_PREFIX)).resolve()
    b = r / STAMP
    for part in ("neo4j", "qdrant", "redis", "files"):
        (b / part).mkdir(parents=True)
    (b / "redis" / "dump.rdb").write_bytes(b"r")
    (b / "files" / restore.FILES_TAR).write_bytes(b"t")
    yield r
    import shutil

    shutil.rmtree(r, ignore_errors=True)


def fake(calls, rc=0):
    def run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, rc, "out", "err")
    return run


def neo4j_load(b, db="memory"):
    return ("run", "--rm", "--no-deps", "-T", "-v", f"{b}/neo4j:/backups", "neo4j", "neo4j-admin", "database", "load", db,
            "--from-path=/backups", "--overwrite-destination=true")


def qdrant_up(b):
    return ("run", "--rm", "--no-deps", "-T", "-v", f"{b}/qdrant:/backups:ro", "neuralscape", "python", "-c", restore.QDRANT_UPLOAD_SCRIPT)


FILES_EXTRACT = ("run", "--rm", "--no-deps", "-T", "-i", "--entrypoint", "tar", "neuralscape", "-C", "/data", "-xzf", "-")
WIPE = ("run", "--rm", "--no-deps", "-T", "--entrypoint", "sh", "neuralscape", "-c", restore.WIPE_INGEST_SCRIPT)


def cp(b):
    return ("cp", f"{b}/redis/dump.rdb", "redis:/data/dump.rdb")


# ── the accepted shapes ──────────────────────────────────────────


def test_every_restore_shape_is_accepted_and_runs_against_the_project(root, monkeypatch):
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    b = root / STAMP
    calls = []
    run = fake(calls)
    tar = str(b / "files" / restore.FILES_TAR)
    runctx.run_compose("nstest-x", *neo4j_load(b), backup_root=str(root), run=run)
    runctx.run_compose("nstest-x", *qdrant_up(b), backup_root=str(root), run=run)
    runctx.run_compose("nstest-x", *FILES_EXTRACT, backup_root=str(root), stdin_path=tar, run=run)
    runctx.run_compose("nstest-x", *WIPE, backup_root=str(root), run=run)
    runctx.run_compose("nstest-x", *cp(b), backup_root=str(root), run=run)
    runctx.run_compose("nstest-x", "exec", "-T", "redis", "redis-cli", "FLUSHALL", run=run)
    assert len(calls) == 6
    assert all(c[0][:4] == ["docker", "compose", "-p", "nstest-x"] for c in calls)
    assert calls[2][1]["stdin"].name == tar  # the tar is fed on stdin (file object, closed afterwards)
    assert calls[2][1]["stdin"].closed


def test_the_database_name_follows_neo4j_database_like_the_script(root, monkeypatch):
    b = root / STAMP
    monkeypatch.setenv("NEO4J_DATABASE", "graphdb")
    assert restore.neo4j_database() == "graphdb"
    runctx.run_compose("nstest-x", *neo4j_load(b, "graphdb"), backup_root=str(root), run=fake([]))


# ── refusals ─────────────────────────────────────────────────────


@pytest.mark.parametrize("project", ["neuralscape", " NeuralScape ", "production", "scratch"])
def test_restore_shapes_refuse_production_and_off_allowlist_projects(root, project):
    calls = []
    with pytest.raises(guard.GuardError):
        runctx.run_compose(project, *neo4j_load(root / STAMP), backup_root=str(root), run=fake(calls))
    assert calls == []


def test_run_and_cp_need_the_scenarios_backup_root(root):
    for args in (neo4j_load(root / STAMP), cp(root / STAMP)):
        with pytest.raises(runctx.ConfigError, match="backup root"):
            runctx.run_compose("nstest-x", *args, run=fake([]))
    with pytest.raises(runctx.ConfigError):  # the root must be a real nsc-backup-* dir under the temp dir
        runctx.run_compose("nstest-x", *neo4j_load(root / STAMP), backup_root="/etc", run=fake([]))
    with pytest.raises(runctx.ConfigError):
        runctx.run_compose("nstest-x", "ps", backup_root=str(root), run=fake([]))  # only run/cp take one


def test_a_backup_root_outside_the_temp_dir_or_without_the_prefix_is_refused(tmp_path):
    other = tmp_path / "not-prefixed"
    other.mkdir()
    with pytest.raises(restore.RestoreRefused):
        restore.check_backup_root(other)
    with pytest.raises(restore.RestoreRefused):
        restore.check_backup_root("/var/tmp/../../etc")
    with pytest.raises(restore.RestoreRefused):
        restore.check_backup_root(tmp_path / "missing")
    with pytest.raises(restore.RestoreRefused):  # a symlink into a place that is not a backup root
        link = Path(tempfile.mkdtemp(prefix=restore.BACKUP_ROOT_PREFIX)) / "l"
        link.symlink_to("/etc")
        restore.check_backup_root(link)


@pytest.mark.parametrize("make", [
    lambda b: neo4j_load(b.parent / "elsewhere" / STAMP),                              # outside the root
    lambda b: neo4j_load(b / "qdrant"),                                                # wrong part directory
    lambda b: neo4j_load(str(b) + "/../" + STAMP),                                     # traversal
    lambda b: neo4j_load(b, "memory; rm -rf /"),                                       # database name
    lambda b: neo4j_load(b, "-x"),
    lambda b: neo4j_load(b)[:-1] + ("--overwrite-destination=false",),
    lambda b: neo4j_load(b) + ("--extra",),
    lambda b: neo4j_load(b)[:4] + ("--privileged",) + neo4j_load(b)[4:],
    lambda b: tuple(x.replace("neo4j-admin", "sh") for x in neo4j_load(b)),
    lambda b: qdrant_up(b)[:-1] + ("print(1)",),                                       # any other script
    lambda b: qdrant_up(b)[:6] + ("neo4j",) + qdrant_up(b)[7:],                        # wrong service for the qdrant volume
    lambda b: tuple(x.replace(":ro", "") for x in qdrant_up(b)),                       # volume must be read-only
    lambda b: qdrant_up(b)[:5] + (f"{b}/qdrant:/backups:ro", "-v", "/:/host") + qdrant_up(b)[6:],
    lambda b: ("run", "--rm", "--no-deps", "-T", "-v", "/var/run/docker.sock:/var/run/docker.sock", "neuralscape", "sh"),
    lambda b: ("run", "--rm", "--no-deps", "-T", "neuralscape", "python", "-c", "print(1)"),
    lambda b: ("run", "--no-deps", "-T", "-v", f"{b}/neo4j:/backups", "neo4j", "neo4j-admin"),
    lambda b: ("run", "--rm", "--no-deps", "-T", "--entrypoint", "sh", "neuralscape", "-c", "rm -rf /data"),
])
def test_refused_run_shapes(root, monkeypatch, make):
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    calls = []
    with pytest.raises(runctx.ConfigError):
        runctx.run_compose("nstest-x", *make(root / STAMP), backup_root=str(root), run=fake(calls))
    assert calls == []


def test_files_extract_only_with_the_backups_own_tar_on_stdin(root):
    b = root / STAMP
    good = str(b / "files" / restore.FILES_TAR)
    for stdin in (None, "/etc/passwd", str(b / "redis" / "dump.rdb"), str(root / "elsewhere.tar.gz"), str(b / "files" / "missing.tar.gz")):
        with pytest.raises(runctx.ConfigError):
            runctx.run_compose("nstest-x", *FILES_EXTRACT, backup_root=str(root), stdin_path=stdin, run=fake([]))
    with pytest.raises(runctx.ConfigError):  # stdin on any other form
        runctx.run_compose("nstest-x", *neo4j_load(b), backup_root=str(root), stdin_path=good, run=fake([]))
    with pytest.raises(runctx.ConfigError):  # the extract form with another command
        runctx.run_compose("nstest-x", *(FILES_EXTRACT[:-1] + ("-tf", "-")), backup_root=str(root), stdin_path=good, run=fake([]))


def test_cp_only_puts_the_backups_dump_rdb_into_the_redis_data_dir(root):
    b = root / STAMP
    for args in (
        ("cp", f"{b}/redis/dump.rdb", "redis:/etc/passwd"),
        ("cp", f"{b}/redis/dump.rdb", "neo4j:/data/dump.rdb"),
        ("cp", "/etc/passwd", "redis:/data/dump.rdb"),
        ("cp", f"{b}/redis/other.rdb", "redis:/data/dump.rdb"),
        ("cp", f"{b}/neo4j/dump.rdb", "redis:/data/dump.rdb"),
        ("cp", "redis:/data/dump.rdb", f"{b}/redis/dump.rdb"),   # the other direction
        ("cp", f"{b}/redis/dump.rdb", "redis:/data/dump.rdb", "extra"),
        ("cp", "-a", f"{b}/redis/dump.rdb", "redis:/data/dump.rdb"),
    ):
        with pytest.raises(runctx.ConfigError):
            runctx.run_compose("nstest-x", *args, backup_root=str(root), run=fake([]))


def test_destructive_forms_need_the_destructive_switch(root):
    calls = []
    with pytest.raises(runctx.ConfigError):
        runctx.run_compose("nstest-x", "exec", "-T", "redis", "redis-cli", "FLUSHALL", run=fake(calls))
    with pytest.raises(runctx.ConfigError):
        runctx.run_compose("nstest-x", *WIPE, backup_root=str(root), run=fake(calls))
    assert calls == []


def test_flush_forms_other_than_flushall_stay_refused(monkeypatch):
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    for args in (("exec", "-T", "redis", "redis-cli", "FLUSHDB"), ("exec", "-T", "redis", "redis-cli", "FLUSHALL", "ASYNC"),
                 ("exec", "-T", "neo4j", "redis-cli", "FLUSHALL"), ("exec", "redis", "redis-cli", "FLUSHALL"),
                 ("exec", "-T", "redis", "redis-cli", "CONFIG", "SET", "dir", "/")):
        with pytest.raises(runctx.ConfigError):
            runctx.run_compose("nstest-x", *args, run=fake([]))


# ── compose files and the backup script ──────────────────────────


def test_compose_files_become_f_flags_for_every_verb_and_the_cwd_is_the_repo_root(tmp_path, monkeypatch):
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    (tmp_path / "gate.yml").write_text("services: {}\n")
    monkeypatch.setenv("NS_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("NS_COMPOSE_FILES", "docker-compose.yml:gate.yml::")
    calls = []
    runctx.run_compose("nstest-x", "ps", run=fake(calls))
    cmd, kw = calls[0]
    assert cmd == ["docker", "compose", "-p", "nstest-x", "-f", str(tmp_path / "docker-compose.yml"), "-f", str(tmp_path / "gate.yml"), "ps"]
    assert kw["cwd"] == str(tmp_path)


@pytest.mark.parametrize("value", ["-f", "missing.yml", "a b.yml", "$(id).yml", "x;y.yml"])
def test_bad_compose_file_entries_are_refused(tmp_path, monkeypatch, value):
    monkeypatch.setenv("NS_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("NS_COMPOSE_FILES", value)
    with pytest.raises(runctx.ConfigError):
        runctx.run_compose("nstest-x", "ps", run=fake([]))


def test_backup_command_is_the_script_with_quiesce_and_a_fresh_destination(root, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "backup-volumes.sh").write_text("#!/bin/sh\n")
    (repo / "dc.yml").write_text("services: {}\n")
    monkeypatch.setenv("NS_REPO_ROOT", str(repo))
    monkeypatch.setenv("NS_COMPOSE_FILES", "dc.yml")
    cmd = restore.backup_command("nstest-x", root)
    assert cmd == [str(repo / "scripts" / "backup-volumes.sh"), "-p", "nstest-x", "--quiesce", "-d", str(root), "-f", str(repo / "dc.yml")]
    calls = []
    restore.run_backup("nstest-x", root, run=fake(calls))
    assert calls[0][0] == cmd and calls[0][1]["cwd"] == str(repo)


def test_backup_is_refused_for_production_a_foreign_destination_or_a_missing_script(root, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "backup-volumes.sh").write_text("#!/bin/sh\n")
    monkeypatch.setenv("NS_REPO_ROOT", str(repo))
    calls = []
    for project in ("neuralscape", "NeuralScape ", "prod"):
        with pytest.raises(guard.GuardError):
            restore.run_backup(project, root, run=fake(calls))
    for dest in (tmp_path / "plain", "/", "/var/backups", Path.home()):
        with pytest.raises(restore.RestoreRefused):
            restore.run_backup("nstest-x", dest, run=fake(calls))
    monkeypatch.setenv("NS_REPO_ROOT", str(tmp_path / "nowhere"))
    with pytest.raises(restore.RestoreRefused, match="not found"):
        restore.run_backup("nstest-x", root, run=fake(calls))
    assert calls == []


def test_find_backup_wants_exactly_one_complete_directory(root):
    assert restore.find_backup(root) == root / STAMP
    (root / "ns-backup-20260102T000000Z.partial").mkdir()
    assert restore.find_backup(root) == root / STAMP  # a partial is never a backup
    (root / "ns-backup-20260102T000000Z").mkdir()
    with pytest.raises(restore.RestoreRefused):
        restore.find_backup(root)


# ── stack helpers ────────────────────────────────────────────────


def test_rows_ready_names_the_first_service_that_is_not_ready():
    rows = [{"Service": "a", "State": "running", "Health": "healthy"}, {"Service": "b", "State": "running", "Health": ""}]
    assert stack.rows_ready(rows, ["a", "b"]) is None
    assert "c missing" in stack.rows_ready(rows, ["a", "c"])
    assert "exited" in stack.rows_ready([{"Service": "a", "State": "exited"}], ["a"])
    assert "starting" in stack.rows_ready([{"Service": "a", "State": "running", "Health": "starting"}], ["a"])


# ── opt-in gate, preflight shapes, step ordering ─────────────────


def test_the_destructive_gate_needs_the_switch_a_project_and_an_allowlisted_name():
    assert "NS_ALLOW_DESTRUCTIVE=1" in restore.gate_reason({}, "nstest-x")
    assert "NS_ALLOW_DESTRUCTIVE=1" in restore.gate_reason({"NS_ALLOW_DESTRUCTIVE": "0"}, "nstest-x")
    assert "--compose-project" in restore.gate_reason({"NS_ALLOW_DESTRUCTIVE": "1"}, None)
    assert restore.gate_reason({"NS_ALLOW_DESTRUCTIVE": "1"}, "nstest-x") is None
    for bad in ("neuralscape", " NeuralScape ", "production"):
        with pytest.raises(guard.GuardError):
            restore.gate_reason({"NS_ALLOW_DESTRUCTIVE": "1"}, bad)
    with pytest.raises(guard.GuardError):  # the allowlist override cannot admit another name
        restore.gate_reason({"NS_ALLOW_DESTRUCTIVE": "1", "NS_COMPOSE_ALLOWLIST": "bench-"}, "nstest-x")


def test_preflight_probe_shapes_run_and_nothing_else_slips_through(root):
    calls = []
    for args in (("config", "-q"),
                 ("run", "--rm", "--no-deps", "-T", "neo4j", "neo4j-admin", "--version"),
                 ("run", "--rm", "--no-deps", "-T", "--entrypoint", "tar", "neuralscape", "--version"),
                 ("run", "--rm", "--no-deps", "-T", "neuralscape", "python", "-c", restore.QDRANT_PING_SCRIPT)):
        runctx.run_compose("nstest-x", *args, run=fake(calls))
    assert len(calls) == 4
    for args in (("config",), ("config", "-q", "--no-interpolate"), ("config", "--volumes"),
                 ("run", "--rm", "--no-deps", "-T", "neo4j", "neo4j-admin", "database", "load", "memory"),
                 ("run", "--rm", "--no-deps", "-T", "neuralscape", "python", "-c", "print(1)"),
                 ("run", "--rm", "--no-deps", "-T", "--entrypoint", "tar", "neuralscape", "-xf", "/etc/x")):
        with pytest.raises(runctx.ConfigError):
            runctx.run_compose("nstest-x", *args, run=fake(calls))
    assert len(calls) == 4


def _fns(log, fail=(), restore_fails=0):
    left = {"restore": restore_fails}

    def mk(name):
        def fn():
            log.append(name)
            if name in fail:
                raise AssertionError(f"{name} broke")
            if name == "restore" and left["restore"] > 0:
                left["restore"] -= 1
                raise AssertionError("restore broke")
        return fn
    names = ["preflight", "seed", "backup", "after_backup_write", "wipe", "gone", "restore", *flow_mod.POST_STEPS]
    return {n: mk(n) for n in names}


def test_the_happy_path_runs_every_step_in_order():
    log = []
    f = flow_mod.run_flow(flow_mod.Flow(), _fns(log))
    assert log == ["preflight", "seed", "backup", "after_backup_write", "wipe", "gone", "restore", *flow_mod.POST_STEPS]
    assert f.order == log and not f.keep_backup and all(v == "" for v in f.steps.values())


@pytest.mark.parametrize("broken", ["preflight", "seed", "backup"])
def test_nothing_is_wiped_unless_preflight_seed_and_backup_succeeded(broken):
    log = []
    f = flow_mod.run_flow(flow_mod.Flow(), _fns(log, fail=(broken,)))
    assert "wipe" not in log and "restore" not in log and not f.keep_backup
    assert log[-1] == broken


def test_a_failed_wipe_is_still_followed_by_a_restore_and_the_proof_is_skipped():
    log = []
    f = flow_mod.run_flow(flow_mod.Flow(), _fns(log, fail=("wipe",)))
    assert log.index("restore") == log.index("wipe") + 1 and "gone" not in log


def test_post_checks_only_after_a_successful_restore_and_the_backup_is_kept_otherwise():
    log = []
    f = flow_mod.run_flow(flow_mod.Flow(), _fns(log, restore_fails=2))
    assert log.count("restore") == 2 and not any(n in log for n in flow_mod.POST_STEPS)
    assert f.keep_backup and not f.restore_ok


def test_one_retry_of_the_restore_recovers_and_the_first_failure_is_recorded():
    log = []
    f = flow_mod.run_flow(flow_mod.Flow(), _fns(log, restore_fails=1))
    assert log.count("restore") == 2 and log[-1] == "fresh_write"
    assert f.restore_ok and not f.keep_backup and "broke" in f.steps["restore_first_attempt"]


def test_the_backup_is_removed_only_when_no_wipe_happened_or_the_restore_succeeded():
    f = flow_mod.Flow()
    assert not f.keep_backup  # never wiped
    f.order.append("wipe")
    assert f.keep_backup  # wiped, not restored
    f.steps["restore"] = ""
    assert not f.keep_backup


# ── the fixture's own wiring (not just the units) ────────────────


def _scenario_module():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scenarios" / "test_backup_restore.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("br_scenario_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # dataclasses look the module up here
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


def _fixture_body(mod):
    f = mod.story
    return getattr(f, "__wrapped__", None) or f._get_wrapped_function()


def _drive(monkeypatch, restore_fails: bool):
    """Run the real `story` fixture body with fake steps and a fake ctx; return (story, root_exists_after_finalizer)."""
    from types import SimpleNamespace

    mod = _scenario_module()
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    monkeypatch.setattr(mod.stack, "ensure_running", lambda *a, **k: None)
    for name in ("_preflight", "_seed", "_backup", "_after_backup_marker", "_wipe", "_prove_gone", "_post_restore",
                 "_post_graph", "_post_task", "_post_artifact", "_fresh_write"):
        monkeypatch.setattr(mod, name, lambda *a, **k: None)
    monkeypatch.setattr(mod.stack, "restart_counts", lambda *a, **k: {})

    def fake_restore(*a, **k):
        if restore_fails:
            raise AssertionError("restore broke")

    monkeypatch.setattr(mod, "_restore", fake_restore)
    ctx = SimpleNamespace(compose_project="nstest-x", run_id="r1", project_id="acc-r1-proj")
    gen = _fixture_body(mod)(ctx, object())
    s = next(gen)
    root = s.root
    try:
        next(gen)
    except StopIteration:
        pass
    return s, root.exists()


def test_a_failed_restore_keeps_the_backup_directory_through_the_real_fixture(monkeypatch):
    s, exists = _drive(monkeypatch, restore_fails=True)
    try:
        assert exists, "the fixture finalizer deleted the only copy of the data"
        assert "KEPT at" in s.steps["restore"] and "remove that directory" in s.steps["restore"]
    finally:
        import shutil

        shutil.rmtree(s.root, ignore_errors=True)


def test_a_successful_restore_removes_the_backup_directory(monkeypatch):
    s, exists = _drive(monkeypatch, restore_fails=False)
    assert not exists


def test_the_fixture_gate_refuses_without_the_switch_and_for_foreign_projects(monkeypatch):
    from types import SimpleNamespace

    mod = _scenario_module()
    monkeypatch.delenv("NS_ALLOW_DESTRUCTIVE", raising=False)
    with pytest.raises(pytest.skip.Exception, match="NS_ALLOW_DESTRUCTIVE"):
        mod._gate(SimpleNamespace(compose_project="nstest-x"))
    monkeypatch.setenv("NS_ALLOW_DESTRUCTIVE", "1")
    for bad in ("neuralscape", "somethingelse"):
        with pytest.raises(guard.GuardError):
            mod._gate(SimpleNamespace(compose_project=bad))
    mod._gate(SimpleNamespace(compose_project="nstest-x"))  # allowed
    # and the real fixture body goes through the same gate: nothing is created when it refuses
    monkeypatch.delenv("NS_ALLOW_DESTRUCTIVE")
    with pytest.raises(pytest.skip.Exception):
        next(_fixture_body(mod)(SimpleNamespace(compose_project="nstest-x", run_id="r", project_id="p"), object()))
