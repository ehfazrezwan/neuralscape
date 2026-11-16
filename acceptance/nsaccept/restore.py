"""Plumbing for the destructive backup -> wipe -> restore scenario.

Everything docker-shaped here is a FIXED command shape, validated before it is
run, and every path it may touch lives under one fresh temp directory the
scenario created (``nsc-backup-*``). Nothing here can be aimed at production:
the compose project always goes through ``guard.check_compose_project``.

The restore shapes follow the RESTORE section of ``scripts/backup-volumes.sh``;
the only deliberate differences are noted next to each (the Qdrant upload runs
inside the API image against ``QDRANT_URL`` because the Qdrant port is not
published everywhere, and the files tar is read from a mounted directory by
stdin, as the script's own form does).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from . import guard

BACKUP_ROOT_PREFIX = "nsc-backup-"
BACKUP_DIR_RE = re.compile(r"ns-backup-\d{8}T\d{6}Z")
_HOST_PATH_RE = re.compile(r"/[\w./+-]+")
_DB_RE = re.compile(r"[A-Za-z][A-Za-z0-9.-]{0,62}")
_FILE_ARG_RE = re.compile(r"[\w./+@-]+")

FILES_TAR = "ingest-exemplars.tar.gz"

# Runs INSIDE the API image (``python -c``): uploads every snapshot named in
# index.json back to its collection. ``priority=snapshot`` makes the snapshot
# win over whatever the (possibly empty) collection holds, as the script says.
QDRANT_UPLOAD_SCRIPT = r'''
import json, os, sys, urllib.parse, urllib.request, uuid

base = os.environ["QDRANT_URL"].rstrip("/")
index = json.load(open("/backups/index.json"))
if not index:
    sys.exit("index.json lists no collections")
for name, fname in sorted(index.items()):
    data = open(os.path.join("/backups", fname), "rb").read()
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="snapshot"; filename="{fname}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    url = base + "/collections/" + urllib.parse.quote(name, safe="") + "/snapshots/upload?priority=snapshot&wait=true"
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req, timeout=900) as r:
        if r.status != 200:
            sys.exit(f"upload of {name} -> {r.status}")
    print(f"restored collection {name}", file=sys.stderr)
'''

# Wipes the ingest artifacts (the scenario's proof that the files part is restored
# from the backup, not left behind).
WIPE_INGEST_SCRIPT = "find /data/ingest -mindepth 1 -delete"


class RestoreRefused(RuntimeError):
    """A command shape or path the restore plumbing will not run."""


# ── service names / paths ────────────────────────────────────────


def api_service() -> str:
    return os.environ.get("NS_API_SERVICE", "neuralscape")


def neo4j_service() -> str:
    return os.environ.get("NS_NEO4J_SERVICE", "neo4j")


def redis_service() -> str:
    return os.environ.get("NS_REDIS_SERVICE", "redis")


def neo4j_database() -> str:
    """Same variable and default as scripts/backup-volumes.sh."""
    return os.environ.get("NEO4J_DATABASE") or "memory"


def repo_root() -> Path:
    return Path(os.environ.get("NS_REPO_ROOT") or Path(__file__).resolve().parents[2])


def compose_file_args(env: dict | None = None, root: Path | None = None) -> list[str]:
    """``-f <file>`` pairs from ``NS_COMPOSE_FILES`` (colon separated, the
    ``COMPOSE_FILE`` convention); relative paths resolve against the repo root.
    Empty when unset. Refuses option-looking or missing files."""
    env = os.environ if env is None else env
    root = root or repo_root()
    out: list[str] = []
    for part in (p.strip() for p in env.get("NS_COMPOSE_FILES", "").split(":")):
        if not part:
            continue
        if part.startswith("-") or not _FILE_ARG_RE.fullmatch(part):
            raise RestoreRefused(f"NS_COMPOSE_FILES entry {part!r} is not a plain file path")
        path = Path(part) if Path(part).is_absolute() else root / part
        if not path.is_file():
            raise RestoreRefused(f"NS_COMPOSE_FILES entry {part!r} is not a file ({path})")
        out += ["-f", str(path)]
    return out


def make_backup_root() -> Path:
    """A fresh ``nsc-backup-*`` directory under the system temp dir (mode 0700)."""
    return check_backup_root(tempfile.mkdtemp(prefix=BACKUP_ROOT_PREFIX))


def check_backup_root(path: str | os.PathLike) -> Path:
    """The only place a backup may be written or read: a real directory named
    ``nsc-backup-*`` somewhere under the system temp dir (symlinks resolved)."""
    resolved = Path(path).resolve()
    tmp = Path(tempfile.gettempdir()).resolve()
    if not resolved.is_dir():
        raise RestoreRefused(f"backup destination {str(path)!r} is not a directory")
    if tmp not in resolved.parents or not re.fullmatch(re.escape(BACKUP_ROOT_PREFIX) + r"[\w-]+", resolved.name):
        raise RestoreRefused(
            f"backup destination {str(path)!r} is not a fresh {BACKUP_ROOT_PREFIX}* directory under the temp dir"
        )
    if not _HOST_PATH_RE.fullmatch(str(resolved)):
        raise RestoreRefused(f"backup destination {str(resolved)!r} has characters this harness will not pass to docker")
    return resolved


def find_backup(root: str | os.PathLike) -> Path:
    """The single complete ``ns-backup-<stamp>`` directory under ``root`` (never ``.partial``)."""
    root = check_backup_root(root)
    done = sorted(p for p in root.iterdir() if p.is_dir() and BACKUP_DIR_RE.fullmatch(p.name))
    if len(done) != 1:
        raise RestoreRefused(f"expected exactly one complete backup under {root}, found {[p.name for p in done]}")
    return done[0]


def _part_dir(host: str, root: Path, part: str) -> Path:
    if not _HOST_PATH_RE.fullmatch(host or "") or os.path.normpath(host) != host:
        raise RestoreRefused(f"not a plain, normalised absolute path: {host!r}")
    p = Path(host).resolve()
    if p.name != part or not BACKUP_DIR_RE.fullmatch(p.parent.name) or p.parent.parent != root:
        raise RestoreRefused(f"{host!r} is not <backup root>/ns-backup-<stamp>/{part}")
    return p


def _volume(arg: str, root: Path, part: str, suffix: str) -> Path:
    m = re.fullmatch(r"(/[\w./+-]+):/backups" + re.escape(suffix), arg or "")
    if not m:
        raise RestoreRefused(f"volume {arg!r} is not <host dir>:/backups{suffix}")
    return _part_dir(m.group(1), root, part)


# ── validation of the new compose shapes ─────────────────────────


def destructive_allowed(env: dict | None = None) -> bool:
    return (os.environ if env is None else env).get("NS_ALLOW_DESTRUCTIVE") == "1"


def destructive_exec(args: tuple, env: dict | None = None) -> bool:
    """``exec -T <redis> redis-cli FLUSHALL``, and only with NS_ALLOW_DESTRUCTIVE=1."""
    return tuple(args) == ("exec", "-T", redis_service(), "redis-cli", "FLUSHALL") and destructive_allowed(env)


def validate_restore_args(
    args: tuple, backup_root: str | os.PathLike | None, stdin_path: str | os.PathLike | None = None,
    env: dict | None = None,
) -> None:
    """Raise ``RestoreRefused`` unless ``args`` (starting at the verb) is one of the
    fixed ``run`` / ``cp`` shapes below, with every host path inside ``backup_root``.

    run (all take ``--rm --no-deps -T``):
      neo4j load      -v <B>/neo4j:/backups <neo4j> neo4j-admin database load <db> --from-path=/backups --overwrite-destination=true
      qdrant upload   -v <B>/qdrant:/backups:ro <api> python -c <QDRANT_UPLOAD_SCRIPT>
      files extract   -i --entrypoint tar <api> -C /data -xzf -      (stdin = <B>/files/ingest-exemplars.tar.gz)
      ingest wipe     --entrypoint sh <api> -c <WIPE_INGEST_SCRIPT>  (needs NS_ALLOW_DESTRUCTIVE=1)
    cp:
      <B>/redis/dump.rdb  <redis>:/data/dump.rdb
    """
    a = tuple(args)
    verb = a[0] if a else ""
    if verb not in ("run", "cp"):
        raise RestoreRefused(f"{verb!r} is not a restore verb")
    if backup_root is None:
        raise RestoreRefused("restore commands need the scenario's backup root")
    root = check_backup_root(backup_root)
    if verb == "cp":
        if len(a) != 3 or a[2] != f"{redis_service()}:/data/dump.rdb":
            raise RestoreRefused(f"cp is only <B>/redis/dump.rdb {redis_service()}:/data/dump.rdb, got {a[1:]}")
        src = Path(a[1])
        _part_dir(str(src.parent), root, "redis")
        if src.name != "dump.rdb":
            raise RestoreRefused(f"cp source must be dump.rdb, got {src.name!r}")
        return
    if a[:4] != ("run", "--rm", "--no-deps", "-T"):
        raise RestoreRefused("run must start with --rm --no-deps -T")
    rest = a[4:]
    if stdin_path is not None and rest[:1] != ("-i",):
        raise RestoreRefused("stdin is only for the files extract form")
    if len(rest) == 9 and rest[0] == "-v" and rest[2] == neo4j_service():  # neo4j load
        _volume(rest[1], root, "neo4j", "")
        db = rest[6]
        if rest[3:6] != ("neo4j-admin", "database", "load") or not _DB_RE.fullmatch(db) or rest[7:] != (
            "--from-path=/backups", "--overwrite-destination=true",
        ):
            raise RestoreRefused(f"neo4j command shape refused: {rest[3:]}")
        return
    if len(rest) == 6 and rest[0] == "-v" and rest[2] == api_service():  # qdrant upload
        _volume(rest[1], root, "qdrant", ":ro")
        if rest[3:5] != ("python", "-c") or rest[5] != QDRANT_UPLOAD_SCRIPT:
            raise RestoreRefused("the qdrant upload form takes exactly the harness's own script")
        return
    if rest[:1] == ("-i",):  # files extract
        if rest != ("-i", "--entrypoint", "tar", api_service(), "-C", "/data", "-xzf", "-"):
            raise RestoreRefused(f"files extract shape refused: {rest}")
        if stdin_path is None:
            raise RestoreRefused("files extract needs its tar on stdin")
        src = Path(stdin_path)
        _part_dir(str(src.parent), root, "files")
        if src.name != FILES_TAR or not src.is_file():
            raise RestoreRefused(f"stdin must be <B>/files/{FILES_TAR}, got {str(stdin_path)!r}")
        return
    if rest == ("--entrypoint", "sh", api_service(), "-c", WIPE_INGEST_SCRIPT):  # ingest wipe
        if not destructive_allowed(env):
            raise RestoreRefused("wiping the ingest volume needs NS_ALLOW_DESTRUCTIVE=1")
        return
    raise RestoreRefused(f"run shape refused: {rest[:4]}")


# ── the backup script ────────────────────────────────────────────


def backup_command(project: str, dest: str | os.PathLike, files: list[str] | None = None, root: Path | None = None) -> list[str]:
    """``scripts/backup-volumes.sh -p <project> --quiesce -d <dest> [-f file]...``."""
    project = guard.check_compose_project(project)
    dest = check_backup_root(dest)
    root = root or repo_root()
    script = root / "scripts" / "backup-volumes.sh"
    if not script.is_file():
        raise RestoreRefused(f"{script} not found (set NS_REPO_ROOT to the checkout under test)")
    cmd = [str(script), "-p", project, "--quiesce", "-d", str(dest)]
    cmd += files if files is not None else compose_file_args(root=root)
    # compose_file_args already yields ["-f", path, ...]; -f is the script's option name too.
    return cmd


def run_backup(
    project: str, dest: str | os.PathLike, *, timeout: float = 1800.0,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> subprocess.CompletedProcess:
    """Run the real backup script (quiesced). Output is captured, never echoed here;
    the process environment is inherited and never printed."""
    cmd = backup_command(project, dest)
    return run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(repo_root()))
