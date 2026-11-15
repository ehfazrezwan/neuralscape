"""Safety checks for lifecycle journeys that create and remove compose volumes.

The journey runner removes only the volumes of its own compose project and one sentinel volume.
These pure helpers take the container tool as a callable so the rules can be tested without
a Docker daemon.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

Run = Callable[..., Any]  # run("docker", ...) -> object with .stdout


class JourneySafetyError(RuntimeError):
    """A journey must not start or must not proceed."""


# Every compose project the harness starts, stops or removes volumes of must match an allowlist of
# disposable prefixes (`name == prefix` or `name.startswith(prefix)`). The production project name is
# refused outright (trimmed, any letter case, empty included). More prefixes can be added only by an
# explicit command-line option through `allow_prefix`, never from the environment.
DEFAULT_ALLOWED_PREFIXES: tuple[str, ...] = ("nscampaign", "nstest-", "ns-bench-")
PRODUCTION_PROJECT = "neuralscape"
_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")
_extra_prefixes: list[str] = []


def allowed_prefixes() -> tuple[str, ...]:
    return DEFAULT_ALLOWED_PREFIXES + tuple(_extra_prefixes)


def allow_prefix(prefix: object) -> str:
    """Add a disposable prefix (CLI option only). Anything that could reach production is refused."""

    if (not isinstance(prefix, str) or len(prefix) < 3 or not set(prefix) <= _NAME_CHARS
            or PRODUCTION_PROJECT.startswith(prefix) or prefix.startswith(PRODUCTION_PROJECT)):
        raise JourneySafetyError(f"refusing to allow project prefix {prefix!r}")
    if prefix not in _extra_prefixes:
        _extra_prefixes.append(prefix)
    return prefix


def reset_extra_prefixes() -> None:
    _extra_prefixes.clear()


def require_disposable_project(name: object) -> str:
    """Return the project name when it is a disposable benchmark project, else refuse."""

    if not isinstance(name, str) or not name.strip():
        raise JourneySafetyError("refusing to run docker: the compose project name is empty")
    if name.strip().casefold() == PRODUCTION_PROJECT:
        raise JourneySafetyError(f"refusing to run docker: {name!r} is the production compose project")
    prefixes = allowed_prefixes()
    if not any(name == p or name.startswith(p) for p in prefixes) or not set(name) <= _NAME_CHARS:
        raise JourneySafetyError(
            f"refusing to run docker: project {name!r} is not a disposable project "
            f"(lower-case name matching one of {', '.join(prefixes)})")
    return name


def require_disposable_volume(name: object, project: str) -> str:
    """The sentinel volume the journey creates and removes must be named after its own disposable project."""

    require_disposable_project(project)
    if not isinstance(name, str) or not name.startswith(project + "-") or name == project + "-":
        raise JourneySafetyError(f"refusing to remove volume {name!r}: not a volume of the disposable project")
    return name


def local_only_violations(compose_text: str) -> list[str]:
    """Lines of a compose file that would publish a port beyond loopback or reach the host's Docker/network.

    The harness stacks publish nothing. Anything that is published must bind a loopback address.
    """

    loopback = ("127.0.0.1:", "[::1]:", "localhost:")
    found: list[str] = []
    in_ports = False
    ports_indent = 0
    for raw in compose_text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        text = line.strip()
        if in_ports and indent <= ports_indent and not text.startswith("-"):
            in_ports = False
        if text.startswith("ports:"):
            in_ports, ports_indent = True, indent
            if text != "ports:" and text != "ports: []":
                found.append(text)
            continue
        if in_ports and text.startswith("-"):
            entry = text[1:].strip().strip("\"'")
            if not entry.startswith(loopback):
                found.append(f"ports entry {entry!r} does not bind a loopback address")
        elif in_ports and text.startswith(("published:", "target:", "host_ip:", "mode:", "protocol:")):
            if text.startswith("host_ip:") and text.split(":", 1)[1].strip().strip("\"'") not in ("127.0.0.1", "::1"):
                found.append(text)
            elif text.startswith("published:"):
                found.append("long-form published port: bind it with host_ip 127.0.0.1 in a short-form entry")
        if text.replace(" ", "").startswith(("network_mode:host", "network_mode:\"host\"", "privileged:true", "pid:host")):
            found.append(text)
        if "docker.sock" in text:
            found.append(text)
    return found


def require_local_only(compose_text: str, name: str = "compose file") -> None:
    violations = local_only_violations(compose_text)
    if violations:
        raise JourneySafetyError(f"refusing to start: {name} is not loopback-only: {'; '.join(violations)}")


def _lines(run: Run, *args: str) -> list[str]:
    return [line for line in run(*args).stdout.split() if line]


def leftovers(run: Run, project: str, sentinel: str) -> list[str]:
    """Names of anything an earlier journey with the same project name left behind."""

    require_disposable_volume(sentinel, project)
    found: list[str] = []
    found += [f"container:{n}" for n in _lines(
        run, "docker", "ps", "-a", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.Names}}")]
    found += [f"volume:{n}" for n in _lines(run, "docker", "volume", "ls", "--format", "{{.Name}}")
              if n.startswith(project + "_") or n == sentinel]
    found += [f"network:{n}" for n in _lines(run, "docker", "network", "ls", "--format", "{{.Name}}")
              if n.startswith(project + "_")]
    return found


def preflight(run: Run, project: str, sentinel: str) -> None:
    """Refuse to start while a project of the same name, its volumes or the sentinel exist."""

    existing = leftovers(run, project, sentinel)
    if existing:
        raise JourneySafetyError(
            "refusing to start: resources from an earlier run exist and would be removed by "
            f"`down -v`: {', '.join(sorted(existing))}")


def uninstall_report(before: Iterable[str], mid: Iterable[str], after: Iterable[str], project: str,
                     sentinel: str, mounted: Iterable[str]) -> dict[str, Any]:
    """Describe what `down -v` removed against the volumes that existed beforehand."""

    require_disposable_volume(sentinel, project)
    before, mid, after, mounted = set(before), set(mid), set(after), set(mounted)
    removed = mid - after
    own = {v for v in mid if v.startswith(project + "_")}
    # Volumes that are neither the project's nor mounted by its containers, measured over the
    # window of the removal itself (other work on a shared host can also add or drop volumes
    # earlier, which is reported separately and cannot be attributed to this run).
    unrelated = mid - own - mounted
    return {
        "project_volumes_before_down": sorted(own),
        "project_volumes_after_down": sorted(v for v in after if v.startswith(project + "_")),
        "unrelated_volumes_untouched": unrelated <= after,
        "unrelated_volumes_missing_after": sorted(unrelated - after),
        "sentinel_present": sentinel in after,
        "removed_volumes_not_project_named": sorted(v for v in removed if not v.startswith(project + "_")),
        "removed_volumes_not_project_named_and_not_mounted_by_project": sorted(
            v for v in removed if not v.startswith(project + "_") and v not in mounted),
        "vanished_since_start_not_ours": sorted(v for v in before - after if v not in mounted and not v.startswith(project + "_")),
    }


def uninstall_is_safe(report: dict[str, Any]) -> bool:
    return bool(report["unrelated_volumes_untouched"] and report["sentinel_present"]
                and not report["project_volumes_after_down"]
                and not report["removed_volumes_not_project_named_and_not_mounted_by_project"])
