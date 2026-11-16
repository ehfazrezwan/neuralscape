"""Capability probes for @needs(...)."""

from __future__ import annotations

import os
import shutil


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def missing_reason(capability: str) -> str | None:
    """Return a skip reason when ``capability`` is absent, else None."""
    cap = capability.strip().lower()
    if cap == "node":
        return None if shutil.which("node") else "node not available"
    # "compose" is declared automatically when the run was given
    # --compose-project (the plugin exports NS_HAVE_COMPOSE=1); it also needs
    # a docker CLI on this host.
    if cap == "compose":
        if not _flag("NS_HAVE_COMPOSE"):
            return "compose not available (run with --compose-project)"
        return None if shutil.which("docker") else "compose not available (docker CLI not on PATH)"
    # Everything else is declared by the gate through NS_HAVE_<NAME>=1
    # (e.g. NS_HAVE_DOCLING=1 when the profile brought the docling service up).
    env = "NS_HAVE_" + cap.upper().replace("-", "_")
    if _flag(env):
        return None
    return f"{cap} not available (set {env}=1 when the stack provides it)"
