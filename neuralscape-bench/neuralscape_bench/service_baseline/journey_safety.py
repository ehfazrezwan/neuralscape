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


def _lines(run: Run, *args: str) -> list[str]:
    return [line for line in run(*args).stdout.split() if line]


def leftovers(run: Run, project: str, sentinel: str) -> list[str]:
    """Names of anything an earlier journey with the same project name left behind."""

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

    before, mid, after, mounted = set(before), set(mid), set(after), set(mounted)
    removed = mid - after
    own = {v for v in mid if v.startswith(project + "_")}
    return {
        "project_volumes_before_down": sorted(own),
        "project_volumes_after_down": sorted(v for v in after if v.startswith(project + "_")),
        "unrelated_volumes_untouched": before <= after,
        "sentinel_present": sentinel in after,
        "removed_volumes_not_project_named": sorted(v for v in removed if not v.startswith(project + "_")),
        "removed_volumes_not_project_named_and_not_mounted_by_project": sorted(
            v for v in removed if not v.startswith(project + "_") and v not in mounted),
    }


def uninstall_is_safe(report: dict[str, Any]) -> bool:
    return bool(report["unrelated_volumes_untouched"] and report["sentinel_present"]
                and not report["project_volumes_after_down"]
                and not report["removed_volumes_not_project_named_and_not_mounted_by_project"])
