"""scripts/backup-volumes.sh, exercised against a stand-in ``docker``.

No Docker is involved: a tiny ``docker`` executable on PATH records every
``docker compose`` call and plays the part of the redis / qdrant / neo4j /
API containers (writing the files a real run would produce). That is enough to
prove the script's own logic: the order of operations, that Neo4j is restarted
even when its dump fails, that a failed run never looks like a backup, that
retention keeps the newest N complete backups and never deletes a partial
one, and that every persistent compose volume is covered. The real stores are
exercised on the VPS gate.

Only the service tree exists in the packaged test image (no scripts/), so the
tests return early there; the host CI job runs them.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "backup-volumes.sh"

_STUB = r'''#!/usr/bin/env bash
# Stand-in for `docker`; only `docker compose ...` is supported.
set -u
LOG="$STUB_DIR/calls.log"
[ "${1:-}" = compose ] || { echo "stub docker: unsupported $*" >&2; exit 2; }
shift
while [ $# -gt 0 ]; do                      # skip -p NAME / -f FILE global flags
  case "$1" in -p|-f) shift 2 ;; *) break ;; esac
done
echo "$*" >> "$LOG"
cmd="$1"; shift
case "$cmd" in
  ps)                                        # ps --status running -q <svc>
    svc="${*: -1}"
    case " ${STUB_DOWN:-} " in *" $svc "*) ;; *) echo "cid-$svc" ;; esac ;;
  exec)                                      # exec -T redis redis-cli <ARGS>
    shift 3
    case "$1" in
      LASTSAVE) echo 100 ;;                  # frozen: a LASTSAVE-based wait would hang
      BGSAVE) echo 2 > "$STUB_DIR/bgsave_polls"; echo "Background saving started" ;;
      INFO)                                  # in progress for the first poll(s), then done
        n="$(cat "$STUB_DIR/bgsave_polls" 2>/dev/null || echo 0)"
        if [ "$n" -gt 0 ]; then echo $((n - 1)) > "$STUB_DIR/bgsave_polls"; flag=1; else flag=0; fi
        printf 'rdb_bgsave_in_progress:%s\r\nrdb_last_bgsave_status:%s\r\n' "$flag" "${STUB_BGSAVE_STATUS:-ok}" ;;
    esac ;;
  cp)                                        # cp redis:/data/dump.rdb <dest>
    printf 'RDB-BYTES' > "$2" ;;
  stop) ;;
  start) ;;
  up) ;;
  run)
    args=("$@")
    mount=""
    for ((i = 0; i < ${#args[@]}; i++)); do
      [ "${args[i]}" = "-v" ] && mount="${args[i+1]%%:*}"
    done
    joined=" ${args[*]} "
    case "$joined" in
      *" --entrypoint tar "*)                # files part: stream a real tar.gz to stdout
        tar -czf - -C "$STUB_DIR/data" ingest exemplars ;;
      *" neuralscape python - "*)            # qdrant part: swallow the script on stdin
        cat > "$STUB_DIR/qdrant-script.py"
        printf 'SNAP' > "$mount/memories.snap-1.snapshot"
        printf '{"memories": "memories.snap-1.snapshot"}' > "$mount/index.json" ;;
      *" neo4j neo4j-admin database dump "*)
        [ -z "${STUB_FAIL_DUMP:-}" ] || { echo "dump failed" >&2; exit 1; }
        db="${joined#*database dump }"; db="${db%% *}"
        printf 'DUMP' > "$mount/$db.dump" ;;
      *) echo "stub docker: unsupported run $*" >&2; exit 2 ;;
    esac ;;
  *) echo "stub docker: unsupported compose $cmd" >&2; exit 2 ;;
esac
'''


def _run(tmp_path: Path, *args: str, env: dict | None = None) -> tuple[subprocess.CompletedProcess, Path]:
    stub_dir = tmp_path / "stub"
    bin_dir = stub_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    docker = bin_dir / "docker"
    docker.write_text(_STUB, encoding="utf-8")
    docker.chmod(0o755)
    data = stub_dir / "data"
    (data / "ingest").mkdir(parents=True, exist_ok=True)
    (data / "exemplars").mkdir(parents=True, exist_ok=True)
    (data / "ingest" / "doc.txt").write_text("artifact", encoding="utf-8")
    dest = tmp_path / "backups"
    full_env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "STUB_DIR": str(stub_dir),
        **(env or {}),
    }
    full_env.pop("BACKUP_DIR", None)
    full_env.pop("COMPOSE_FILE", None)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dest", str(dest), *args],
        capture_output=True, text=True, env=full_env, timeout=120, check=False,
    )
    return result, stub_dir


def _calls(stub_dir: Path) -> list[str]:
    log = stub_dir / "calls.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def _index(calls: list[str], needle: str) -> int:
    return next(i for i, line in enumerate(calls) if needle in line)


def _complete(dest: Path) -> list[str]:
    return sorted(p.name for p in dest.glob("ns-backup-*") if p.is_dir() and not p.name.endswith(".partial"))


def _ok(tmp_path: Path, *args: str, **kw) -> tuple[subprocess.CompletedProcess, Path]:
    result, stub = _run(tmp_path, *args, **kw)
    assert result.returncode == 0, result.stderr
    return result, stub


# ── static ──────────────────────────────────────────────────────────


def test_script_is_valid_bash_and_documents_restore_for_every_volume() -> None:
    if not SCRIPT.is_file():
        return
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, check=False).returncode == 0
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -Eeuo pipefail" in text
    assert os.access(SCRIPT, os.X_OK), "scripts/backup-volumes.sh must be executable"
    for volume in ("neo4j_data", "qdrant_data", "redis_data", "ingest_data", "exemplar_data"):
        assert volume in text
    help_text = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True, check=False).stdout
    assert "RESTORE" in help_text and "neo4j-admin database load" in help_text
    assert "snapshots/upload" in help_text and "dump.rdb" in help_text


def test_every_compose_volume_that_holds_data_is_named_by_the_script() -> None:
    if not SCRIPT.is_file():
        return
    import yaml

    volumes = set(yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["volumes"])
    # test_venv is a cache, cbm_cache is rebuilt by re-indexing: neither is user data.
    data_volumes = volumes - {"test_venv", "cbm_cache"}
    text = SCRIPT.read_text(encoding="utf-8")
    missing = sorted(v for v in data_volumes if v not in text)
    assert not missing, f"compose volumes with no backup coverage: {missing}"


# ── behaviour ───────────────────────────────────────────────────────


def test_full_backup_layout_checksums_and_operation_order(tmp_path: Path) -> None:
    if not SCRIPT.is_file():
        return
    result, stub = _ok(tmp_path, "-f", "docker-compose.yml", "-p", "nsx")
    dest = tmp_path / "backups"
    final = Path(result.stdout.strip())
    assert final.parent == dest.resolve() and final.name.startswith("ns-backup-")
    assert _complete(dest) == [final.name]
    assert not list(dest.glob("*.partial"))

    assert (final / "redis" / "dump.rdb").read_bytes() == b"RDB-BYTES"
    assert json.loads((final / "qdrant" / "index.json").read_text()) == {"memories": "memories.snap-1.snapshot"}
    assert (final / "neo4j" / "memory.dump").read_bytes() == b"DUMP"
    with tarfile.open(final / "files" / "ingest-exemplars.tar.gz") as tar:
        assert "ingest/doc.txt" in tar.getnames()
    manifest = (final / "MANIFEST.txt").read_text()
    assert "parts=redis qdrant files neo4j" in manifest and "neo4j_database=memory" in manifest

    sums = (final / "SHA256SUMS").read_text().splitlines()
    listed = {}
    for line in sums:
        digest, _, name = line.partition("  ")
        listed[name.removeprefix("./")] = digest
        assert hashlib.sha256((final / name).read_bytes()).hexdigest() == digest, name
    assert {"neo4j/memory.dump", "redis/dump.rdb", "MANIFEST.txt", "qdrant/index.json"} <= set(listed)
    assert "SHA256SUMS" not in listed

    calls = _calls(stub)
    stop = _index(calls, "stop neo4j")
    dump = _index(calls, "neo4j-admin database dump memory --to-path=/backups")
    up = _index(calls, "up -d --no-deps --wait")
    assert stop < dump < up
    # the online parts finish before the only part with downtime starts
    assert _index(calls, "BGSAVE") < _index(calls, "cp redis:/data/dump.rdb") < stop
    assert _index(calls, "neuralscape python -") < stop
    assert _index(calls, "--entrypoint tar") < stop
    # the qdrant helper really talks to the snapshot API and cleans up after itself
    script = (stub / "qdrant-script.py").read_text()
    assert "/snapshots" in script and '"DELETE"' in script and "QDRANT_URL" in script


def test_neo4j_is_restarted_and_nothing_is_published_when_the_dump_fails(tmp_path: Path) -> None:
    if not SCRIPT.is_file():
        return
    result, stub = _run(tmp_path, env={"STUB_FAIL_DUMP": "1"})
    assert result.returncode != 0
    calls = _calls(stub)
    assert any("stop neo4j" in c for c in calls)
    assert any(c.startswith("up -d --no-deps --wait") and c.endswith("neo4j") for c in calls), (
        "neo4j must be brought back after a failed dump"
    )
    dest = tmp_path / "backups"
    assert _complete(dest) == [], "a failed run must not leave something that looks like a backup"
    assert len(list(dest.glob("ns-backup-*.partial"))) == 1
    assert "FAILED" in result.stderr


def test_a_stopped_service_aborts_before_neo4j_is_touched(tmp_path: Path) -> None:
    if not SCRIPT.is_file():
        return
    result, stub = _run(tmp_path, env={"STUB_DOWN": "qdrant"})
    assert result.returncode != 0 and "qdrant" in result.stderr
    assert not any("stop neo4j" in c for c in _calls(stub))
    assert _complete(tmp_path / "backups") == []


def test_skip_leaves_parts_out_and_never_stops_neo4j_when_skipped(tmp_path: Path) -> None:
    if not SCRIPT.is_file():
        return
    result, stub = _ok(tmp_path, "--skip", "neo4j,qdrant")
    final = Path(result.stdout.strip())
    assert sorted(p.name for p in final.iterdir()) == ["MANIFEST.txt", "SHA256SUMS", "files", "redis"]
    assert not any("neo4j" in c for c in _calls(stub))


def test_skipping_everything_and_bad_options_are_refused(tmp_path: Path) -> None:
    if not SCRIPT.is_file():
        return
    assert _run(tmp_path, "--skip", "neo4j,qdrant,redis,files")[0].returncode != 0
    assert _run(tmp_path, "--skip", "postgres")[0].returncode != 0
    assert _run(tmp_path, "--keep", "0")[0].returncode != 0
    assert _run(tmp_path, "--keep", "many")[0].returncode != 0
    assert _run(tmp_path, "--no-such-flag")[0].returncode != 0


def test_retention_keeps_the_newest_complete_backups_and_spares_partials(tmp_path: Path) -> None:
    if not SCRIPT.is_file():
        return
    dest = tmp_path / "backups"
    for stamp in ("20200101T000000Z", "20200102T000000Z", "20200103T000000Z", "20200104T000000Z"):
        (dest / f"ns-backup-{stamp}").mkdir(parents=True)
        (dest / f"ns-backup-{stamp}" / "MANIFEST.txt").write_text("old")
    partial = dest / "ns-backup-20200105T000000Z.partial"
    partial.mkdir()
    unrelated = dest / "notes"
    unrelated.mkdir()

    result, _ = _ok(tmp_path, "--keep", "3")
    kept = _complete(dest)
    new = Path(result.stdout.strip()).name
    assert kept == ["ns-backup-20200103T000000Z", "ns-backup-20200104T000000Z", new]
    assert partial.exists() and unrelated.exists()


def test_a_failed_run_never_prunes_good_backups(tmp_path: Path) -> None:
    if not SCRIPT.is_file():
        return
    dest = tmp_path / "backups"
    (dest / "ns-backup-20200101T000000Z").mkdir(parents=True)
    result, _ = _run(tmp_path, "--keep", "1", env={"STUB_FAIL_DUMP": "1"})
    assert result.returncode != 0
    assert _complete(dest) == ["ns-backup-20200101T000000Z"]


# ── redis wait, lock, quiesce ───────────────────────────────────────


def test_redis_wait_polls_info_persistence_not_lastsave(tmp_path: Path) -> None:
    if not SCRIPT.is_file():
        return
    _, stub = _ok(tmp_path)
    calls = _calls(stub)
    assert not any("LASTSAVE" in c for c in calls), "LASTSAVE has 1 s resolution: poll INFO persistence"
    assert sum("INFO persistence" in c for c in calls) >= 2, "must keep polling while the save is in progress"
    assert _index(calls, "BGSAVE") < _index(calls, "INFO persistence") < _index(calls, "cp redis:/data/dump.rdb")


def test_a_failed_bgsave_fails_the_backup(tmp_path: Path) -> None:
    if not SCRIPT.is_file():
        return
    result, stub = _run(tmp_path, env={"STUB_BGSAVE_STATUS": "err"})
    assert result.returncode != 0 and "BGSAVE" in result.stderr
    assert _complete(tmp_path / "backups") == []
    assert not any("stop neo4j" in c for c in _calls(stub))


def test_a_running_backup_blocks_a_second_one(tmp_path: Path) -> None:
    if not SCRIPT.is_file():
        return
    lock = tmp_path / "backups" / ".backup.lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text(str(os.getpid()))  # this test process is alive
    result, stub = _run(tmp_path)
    assert result.returncode != 0 and "another backup" in result.stderr
    assert _calls(stub) == [], "a blocked run must not touch docker"
    assert lock.exists(), "the other run's lock must be left alone"


def test_a_stale_lock_is_taken_over_and_the_lock_is_always_released(tmp_path: Path) -> None:
    if not SCRIPT.is_file():
        return
    dead = subprocess.Popen(["true"])
    dead.wait()
    lock = tmp_path / "backups" / ".backup.lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text(str(dead.pid))
    result, _ = _ok(tmp_path)
    assert "stale lock" in result.stderr
    assert not lock.exists(), "lock must be released after a successful run"
    failed, _ = _run(tmp_path, env={"STUB_FAIL_DUMP": "1"})
    assert failed.returncode != 0
    assert not lock.exists(), "lock must be released after a failed run"


def test_quiesce_stops_the_app_around_the_run_and_restarts_it_even_on_failure(tmp_path: Path) -> None:
    if not SCRIPT.is_file():
        return
    apps = "neuralscape neuralscape-worker neuralscape-graph-worker neuralscape-ingest-worker"
    _, stub = _ok(tmp_path, "--quiesce")
    calls = _calls(stub)
    assert f"stop {apps}" in calls and f"start {apps}" in calls
    assert _index(calls, f"stop {apps}") < _index(calls, "BGSAVE")
    assert _index(calls, "stop neo4j") < _index(calls, f"start {apps}")
    assert _index(calls, "up -d --no-deps --wait") < _index(calls, f"start {apps}")

    other = tmp_path / "second"
    other.mkdir()
    failed, stub2 = _run(other, "--quiesce", env={"STUB_FAIL_DUMP": "1"})
    assert failed.returncode != 0
    assert f"start {apps}" in _calls(stub2), "apps must come back after a failed run"


def test_without_quiesce_the_app_is_left_running(tmp_path: Path) -> None:
    if not SCRIPT.is_file():
        return
    _, stub = _ok(tmp_path)
    assert not any(c.startswith(("stop neuralscape", "start neuralscape")) for c in _calls(stub))


def test_usage_states_what_consistency_means() -> None:
    if not SCRIPT.is_file():
        return
    text = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True, check=False).stdout
    assert "--quiesce" in text and "NOT as of one instant" in text
