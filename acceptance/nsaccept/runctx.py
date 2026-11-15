"""Run context: unique ids and tokens for one acceptance run."""

from __future__ import annotations

import os
import random
import string
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable

from . import guard, tokens


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

    def compose(
        self,
        *args: str,
        check: bool = True,
        timeout: float = 120.0,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
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
        return run_compose(self.compose_project, *args, check=check, timeout=timeout, run=run)


COMPOSE_VERBS = frozenset({"pause", "unpause", "stop", "start", "restart", "ps", "logs", "top", "exec"})


def run_compose(
    project: str,
    *args: str,
    check: bool = True,
    timeout: float = 120.0,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> subprocess.CompletedProcess:
    if project.strip().lower() == guard.PROD_COMPOSE_PROJECT:
        raise guard.GuardError(
            f"refusing compose project '{guard.PROD_COMPOSE_PROJECT}' (the production project name)"
        )
    if not args or args[0] not in COMPOSE_VERBS:
        raise ConfigError(f"compose verb {args[:1]!r} is not allowed (allowed: {sorted(COMPOSE_VERBS)})")
    cmd = ["docker", "compose", "-p", project.strip(), *args]
    proc = run(cmd, capture_output=True, text=True, timeout=timeout)
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


def build(base_url: str, profile: str, compose_project: str | None = None) -> Ctx:
    run_id = new_run_id()
    project_id = f"acc-{run_id}-proj"
    compose_project = compose_project or os.environ.get("NS_COMPOSE_PROJECT") or None
    secret = os.environ.get("NS_TOKEN_SECRET", "").strip()
    env_a, env_b = os.environ.get("NS_TOKEN_A", "").strip(), os.environ.get("NS_TOKEN_B", "").strip()

    if compose_project:
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
    )
