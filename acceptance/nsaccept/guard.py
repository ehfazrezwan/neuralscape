"""Refuse to aim the acceptance run at production by accident.

The run writes real data (acc-* users, memories) and mints tokens, so it must
only ever target a throwaway campaign stack.
"""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

PROD_COMPOSE_PROJECT = "neuralscape"
PROD_PORT = 8199
#: Compose projects the harness may touch: disposable ones only. Override with NS_COMPOSE_ALLOWLIST (comma list of prefixes).
DEFAULT_COMPOSE_ALLOWLIST = ("nscampaign", "nstest-")


class GuardError(RuntimeError):
    """Refused: the target looks like production or a non-local stack (exit 2)."""


def _is_loopback(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_compose_project(project: str | None, env: dict | None = None) -> str:
    """THE one gate for every compose-project entry point; returns the normalised name.

    Docker lowercases project names, so the comparison is case-insensitive after strip. Refuses
    ``neuralscape`` and any name outside the allowlist of disposable prefixes.
    """
    env = os.environ if env is None else env
    name = (project or "").strip().lower()
    if not name:
        raise GuardError("a compose project name is required")
    if name == PROD_COMPOSE_PROJECT:
        raise GuardError(
            f"refusing compose project '{PROD_COMPOSE_PROJECT}' (the production project name); "
            "use a throwaway campaign project"
        )
    raw = env.get("NS_COMPOSE_ALLOWLIST", "")
    allow = tuple(p.strip().lower() for p in raw.split(",") if p.strip()) or DEFAULT_COMPOSE_ALLOWLIST
    if not name.startswith(allow):
        raise GuardError(f"refusing compose project {name!r}: not a disposable project (allowed prefixes: {', '.join(allow)})")
    return name


def check(base_url: str, compose_project: str | None, env: dict | None = None) -> None:
    env = os.environ if env is None else env
    if compose_project:
        check_compose_project(compose_project, env)
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if not host:
        raise GuardError(f"cannot parse a host from base URL {base_url!r}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port == PROD_PORT and env.get("NS_ALLOW_PROD_PORT") != "1":
        raise GuardError(
            f"refusing port {PROD_PORT} (the production API port); "
            "use a private port, or set NS_ALLOW_PROD_PORT=1 if this really is a throwaway stack"
        )
    if not _is_loopback(host) and env.get("NS_ALLOW_REMOTE") != "1":
        raise GuardError(
            f"refusing non-loopback host '{host}'; the gate targets 127.0.0.1. "
            "Set NS_ALLOW_REMOTE=1 only for a throwaway stack you own"
        )
