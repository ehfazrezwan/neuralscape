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
      LASTSAVE) cat "$STUB_DIR/lastsave" 2>/dev/null || echo 100 ;;
      BGSAVE) echo 101 > "$STUB_DIR/lastsave"; echo "Background saving started" ;;
      INFO) printf 'rdb_bgsave_in_progress:0\r\nrdb_last_bgsave_status:ok\r\n' ;;
    esac ;;
  cp)                                        # cp redis:/data/dump.rdb <dest>
    printf 'RDB-BYTES' > "$2" ;;
  stop) ;;
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
