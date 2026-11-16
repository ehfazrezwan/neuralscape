"""Stack-level helpers for scenarios that stop, wipe and restart containers.

All docker access goes through ``ctx.compose`` (the allow-listed, project-scoped
wrapper). Pure functions (``check_*``) take plain data so they are self-tested
without a stack and reused as the real checks.
"""

from __future__ import annotations

import time
import httpx
import pytest

from .runctx import app_services, container_restart_count, parse_ps_json, qdrant_service
from . import restore

READY_WAIT = 180


def stateful_services() -> list[str]:
    return [restore.redis_service(), qdrant_service(), restore.neo4j_service()]


def all_services() -> list[str]:
    return [*stateful_services(), *app_services()]


def ps_rows(ctx, services: list[str]) -> list[dict]:
    out = ctx.compose("ps", "--all", "--format", "json", *services, check=False)
    return parse_ps_json(out.stdout or "")


def rows_ready(rows: list[dict], services: list[str]) -> str | None:
    """None when every service is running and not unhealthy, else why not."""
    by = {r.get("Service"): r for r in rows}
    for svc in services:
        row = by.get(svc)
        if row is None:
            return f"{svc} missing from `compose ps`"
        if row.get("State") != "running":
            return f"{svc} is {row.get('State')!r}"
        if row.get("Health") not in (None, "", "healthy"):
            return f"{svc} health is {row.get('Health')!r}"
    return None


def wait_ready(ctx, client, what: str, services: list[str] | None = None, timeout: float = READY_WAIT) -> None:
    """Every service running and healthy, /health answering 200 and a search
    succeeding; otherwise fail loudly with the ``ps`` output."""
    services = services or all_services()
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            why = rows_ready(ps_rows(ctx, services), services)
            if why is None:
                if httpx.get(ctx.base_url + "/health", timeout=10).status_code == 200:
                    client.search("readiness probe", limit=1)
                    return
                last = "/health is not 200"
            else:
                last = why
        except (httpx.HTTPError, AssertionError, ValueError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(3)
    ps = ctx.compose("ps", "--all", check=False).stdout
    pytest.fail(f"stack not ready {timeout:.0f}s after {what}; last: {last}\n{ps}")


def restart_counts(ctx, services: list[str] | None = None) -> dict[str, int]:
    rows = ps_rows(ctx, services or all_services())
    return {r["Service"]: container_restart_count(r["ID"]) for r in rows if r.get("ID") and r.get("Service")}


def check_no_new_restarts(before: dict[str, int], after: dict[str, int], services: list[str]) -> None:
    """No service restarted on its own: RestartCount did not move (deliberate
    ``compose stop`` / ``start`` do not move it)."""
    for svc in services:
        assert svc in before and svc in after, f"{svc}: no RestartCount recorded (before={sorted(before)}, after={sorted(after)})"
        assert after[svc] <= before[svc], f"{svc} restarted by itself: RestartCount {before[svc]} -> {after[svc]}"


def qdrant_http(ctx, method: str, path: str) -> str:
    """Call Qdrant's REST API from inside its own container (curl, else bash /dev/tcp)."""
    raw = (
        f"exec 3<>/dev/tcp/localhost/6333; printf '{method} {path} HTTP/1.1\\r\\nHost: localhost\\r\\n"
        f"Connection: close\\r\\n\\r\\n' >&3; cat <&3"
    )
    script = f"curl -sS -X {method} http://localhost:6333{path} 2>/dev/null || bash -c \"{raw}\""
    return ctx.compose("exec", "-T", qdrant_service(), "sh", "-c", script).stdout or ""


def wait_services(ctx, services: list[str], what: str, timeout: float = READY_WAIT) -> None:
    """Just these containers running and not unhealthy (no API needed); else fail with ``ps``."""
    deadline = time.monotonic() + timeout
    why = "not checked"
    while time.monotonic() < deadline:
        why = rows_ready(ps_rows(ctx, services), services)
        if why is None:
            return
        time.sleep(2)
    pytest.fail(f"{services} not ready {timeout:.0f}s after {what}: {why}\n{ctx.compose('ps', '--all', check=False).stdout}")


def ensure_running(ctx, client, what: str) -> None:
    """Finalizer: whatever happened, bring every service back and wait for ready."""
    ctx.compose("unpause", qdrant_service(), check=False)
    ctx.compose("start", *stateful_services(), check=False)
    ctx.compose("start", *app_services(), check=False)
    wait_ready(ctx, client, what)
