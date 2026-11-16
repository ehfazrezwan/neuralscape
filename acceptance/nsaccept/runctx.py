"""Run context: unique ids and tokens for one acceptance run."""

from __future__ import annotations

import os
import random
import string
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable

from . import guard, restore, tokens


class ConfigError(RuntimeError):
    """Harness misconfiguration (exit code 2)."""


@dataclass
class Ctx:
    base_url: str
    profile: str
    run_id: str
    user_a: str
    user_b: str
    token_a: str = field(repr=False)
    token_b: str = field(repr=False)
    bad_token: str = field(repr=False)
    project_id: str
    token_source: str
    compose_project: str | None = None
    # Flags-on profiles: the profile's recorded flags (placeholders for secrets,
    # never values) and, when it names DICTATOR_USER_IDS, a token for the first
    # id so scenarios can act as the dictator (@needs("dictator")).
    profile_flags: dict = field(default_factory=dict)
    dictator_user: str | None = None
    token_dictator: str | None = field(default=None, repr=False)
    # The `legacy-key` profile: the gate-generated NEURALSCAPE_API_KEY (handed over through the
    # profile's `# harness-env:` header) and the one identity it is bound to. Never in the repr or results.
    legacy_key: str | None = field(default=None, repr=False)
    legacy_user: str | None = None

    def compose(
        self,
        *args: str,
        check: bool = True,
        timeout: float = 120.0,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        backup_root: str | None = None,
        stdin_path: str | None = None,
    ) -> subprocess.CompletedProcess:
        """``docker compose -p <project> <args...>`` against the campaign stack.

        Only usable when the run was given ``--compose-project`` (the same
        project the stack runs under). Refuses the production project name,
        and only lets container-lifecycle verbs through (pause/unpause/stop/
        start/restart/ps/logs/top/exec): nothing here can tear a project
        down or remove its volumes.
        """
        if not self.compose_project:
            raise ConfigError("ctx.compose needs --compose-project (or NS_COMPOSE_PROJECT)")
        return run_compose(
            self.compose_project, *args, check=check, timeout=timeout, run=run,
            backup_root=backup_root, stdin_path=stdin_path,
        )


_CONTAINER_ID = __import__("re").compile(r"^[0-9a-f]{12,64}$")


def container_restart_count(container_id: str, run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> int:
    """RestartCount of one container (``docker inspect``, read-only)."""
    if not _CONTAINER_ID.match(container_id or ""):
        raise ConfigError(f"not a container id: {container_id!r}")
    proc = run(
        ["docker", "inspect", "--format", "{{.RestartCount}}", container_id],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker inspect failed: {(proc.stderr or '').strip()[:200]}")
    return int((proc.stdout or "0").strip() or 0)


def parse_ps_json(text: str) -> list[dict]:
    """``docker compose ps --format json``: a JSON array or one object per line."""
    import json

    text = (text or "").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


COMPOSE_VERBS = frozenset({"pause", "unpause", "stop", "start", "restart", "ps", "logs", "top", "exec", "up", "run", "cp", "config"})
# `up` may only start existing containers: detached, no build, no recreate.
UP_REQUIRED_FLAGS = ("-d", "--no-build", "--no-recreate")
UP_ALLOWED_FLAGS = UP_REQUIRED_FLAGS + ("--no-deps",)


def read_only_exec(args: tuple) -> bool:
    """The two read-only inspections the flags-on scenarios need, and nothing else:
    ``exec -T <api> grep -rl -- <needle> /data/vault[/...]`` and
    ``exec -T <redis> redis-cli GET ns:connector:<id>``."""
    import re

    a = tuple(args)
    if len(a) == 8 and a[1:3] == ("-T", os.environ.get("NS_API_SERVICE", "neuralscape")) and a[3:6] == ("grep", "-rl", "--"):
        needle, root = a[6], a[7]
        return bool(needle) and not needle.startswith("-") and ".." not in root.split("/") and re.fullmatch(r"/data/vault(/[\w.-]+)*", root) is not None
    if len(a) == 6 and a[1:3] == ("-T", os.environ.get("NS_REDIS_SERVICE", "redis")) and a[3:5] == ("redis-cli", "GET"):
        return re.fullmatch(r"ns:connector:[\w.-]+", a[5]) is not None
    return False


def qdrant_service() -> str:
    return os.environ.get("NS_QDRANT_SERVICE", "qdrant")


def app_services() -> list[str]:
    return os.environ.get(
        "NS_APP_SERVICES", "neuralscape,neuralscape-worker,neuralscape-graph-worker,neuralscape-ingest-worker"
    ).split(",")


def run_compose(
    project: str,
    *args: str,
    check: bool = True,
    timeout: float = 120.0,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    backup_root: str | None = None,
    stdin_path: str | None = None,
) -> subprocess.CompletedProcess:
    """Run one allow-listed ``docker compose`` command. ``run``/``cp`` (the backup
    restore shapes in ``restore.py``) additionally need ``backup_root``, the
    scenario's own temp directory; ``stdin_path`` feeds the files-extract form."""
    project = guard.check_compose_project(project)
    if not args or args[0] not in COMPOSE_VERBS:
        raise ConfigError(f"compose verb {args[:1]!r} is not allowed (allowed: {sorted(COMPOSE_VERBS)})")
    if args[0] == "up":
        flags = {a for a in args[1:] if a.startswith("-")}
        names = [a for a in args[1:] if not a.startswith("-")]
        if not set(UP_REQUIRED_FLAGS) <= flags or not flags <= set(UP_ALLOWED_FLAGS):
            raise ConfigError(f"`up` takes exactly {UP_REQUIRED_FLAGS} (+ optional --no-deps), got {sorted(flags)}")
        if not names or not set(names) <= set(app_services()):
            raise ConfigError(f"`up` may only name the API/worker services {app_services()}, got {names}")
    if args[0] == "exec":
        qdrant_form = len(args) == 6 and args[1:5] == ("-T", qdrant_service(), "sh", "-c")
        if not qdrant_form and not read_only_exec(args) and not restore.destructive_exec(args):
            raise ConfigError(
                f"`exec` is only allowed as: exec -T {qdrant_service()} sh -c <script>, "
                "one of the read-only inspections (vault grep, connector vault record), "
                "or redis-cli FLUSHALL with NS_ALLOW_DESTRUCTIVE=1"
            )
    stdin_fh = None
    if args[0] in ("run", "cp", "config"):
        try:
            restore.validate_restore_args(args, backup_root, stdin_path)
        except restore.RestoreRefused as exc:
            raise ConfigError(str(exc)) from exc
    elif backup_root is not None or stdin_path is not None:
        raise ConfigError("backup_root / stdin_path only go with the restore shapes (run, cp)")
    try:
        files = restore.compose_file_args()
    except restore.RestoreRefused as exc:
        raise ConfigError(str(exc)) from exc
    cmd = ["docker", "compose", "-p", project, *files, *args]
    kw: dict = {"cwd": str(restore.repo_root())} if files else {}
    try:
        if stdin_path is not None:
            stdin_fh = open(stdin_path, "rb")
            kw["stdin"] = stdin_fh
        proc = run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    finally:
        if stdin_fh is not None:
            stdin_fh.close()
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"`docker compose -p {project} {' '.join(args)}` exited {proc.returncode}: "
            f"{(proc.stderr or '').strip()[:300]}"
        )
    return proc


def new_run_id() -> str:
    env = os.environ.get("NS_RUN_ID", "").strip().lower()
    if env:
        return env
    alphabet = string.digits + string.ascii_lowercase
    n, out = int(time.time()), ""
    while n:
        n, r = divmod(n, 36)
        out = alphabet[r] + out
    return out + "".join(random.choice(alphabet) for _ in range(3))


def build(
    base_url: str,
    profile: str,
    compose_project: str | None = None,
    profile_flags: dict | None = None,
    dictator_user: str | None = None,
    legacy_key: str | None = None,
) -> Ctx:
    run_id = new_run_id()
    project_id = f"acc-{run_id}-proj"
    compose_project = compose_project or os.environ.get("NS_COMPOSE_PROJECT") or None
    secret = os.environ.get("NS_TOKEN_SECRET", "").strip()
    env_a, env_b = os.environ.get("NS_TOKEN_A", "").strip(), os.environ.get("NS_TOKEN_B", "").strip()

    if compose_project:
        compose_project = guard.check_compose_project(compose_project)
        user_a, user_b = f"acc-{run_id}-a", f"acc-{run_id}-b"
        service = os.environ.get("NS_API_SERVICE", "neuralscape")
        try:
            token_a = tokens.mint_via_compose(compose_project, user_a, service)
            token_b = tokens.mint_via_compose(compose_project, user_b, service)
        except tokens.TokenError as exc:
            raise ConfigError(str(exc)) from exc
        source = f"compose:{compose_project}"
    elif secret:
        user_a, user_b = f"acc-{run_id}-a", f"acc-{run_id}-b"
        token_a, token_b = tokens.mint_local(user_a, secret), tokens.mint_local(user_b, secret)
        source = "NS_TOKEN_SECRET"
    elif env_a and env_b:
        user_a, user_b = os.environ.get("NS_USER_A", "").strip(), os.environ.get("NS_USER_B", "").strip()
        if not user_a or not user_b:
            raise ConfigError("NS_TOKEN_A/NS_TOKEN_B need NS_USER_A and NS_USER_B")
        if user_a == user_b:
            raise ConfigError("NS_USER_A and NS_USER_B must differ (isolation tests need two users)")
        token_a, token_b = env_a, env_b
        source = "NS_TOKEN_A/B"
    else:
        raise ConfigError(
            "no token source: pass --compose-project (or NS_COMPOSE_PROJECT), "
            "or set NS_TOKEN_SECRET, or set NS_TOKEN_A/NS_TOKEN_B with NS_USER_A/NS_USER_B"
        )
    token_dictator = None
    if dictator_user:
        # Only when a token can be minted for an arbitrary id (compose or local secret);
        # a run given two pre-made tokens has no dictator and @needs("dictator") skips.
        if compose_project:
            token_dictator = _mint_dictator(compose_project, dictator_user)
        elif secret:
            token_dictator = tokens.mint_local(dictator_user, secret)
    tokens.register_secret(legacy_key)
    return Ctx(
        base_url=base_url.rstrip("/"),
        profile=profile,
        run_id=run_id,
        user_a=user_a,
        user_b=user_b,
        token_a=token_a,
        token_b=token_b,
        bad_token=tokens.wrong_secret_token(user_a),
        project_id=project_id,
        token_source=source,
        compose_project=compose_project,
        profile_flags=dict(profile_flags or {}),
        dictator_user=dictator_user if token_dictator else None,
        token_dictator=token_dictator,
        legacy_key=legacy_key or None,
        legacy_user=((profile_flags or {}).get("NEURALSCAPE_API_KEY_USER_ID") or "").strip() or None,
    )


def _mint_dictator(compose_project: str, user: str) -> str:
    try:
        return tokens.mint_via_compose(compose_project, user, os.environ.get("NS_API_SERVICE", "neuralscape"))
    except tokens.TokenError as exc:
        raise ConfigError(str(exc)) from exc
