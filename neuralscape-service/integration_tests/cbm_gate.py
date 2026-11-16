"""Decide what a CBM flag-on test does when the cbm-bridge is (not) reachable.

Stdlib only so ``tests/test_cbm_gate_decision.py`` can import it without stores.

The convention (``conftest.unavailable``): under ``NS_INTEGRATION_REQUIRED`` (the
gate) a missing dependency FAILS, anywhere else the test SKIPS.  The CBM bridge
is not part of the standard store stack, so a gate profile that deliberately does
not run it must say so out loud:

* ``NS_INTEGRATION_WAIVE=cbm`` (comma list) turns the REQUIRED failure back into
  a skip for that profile, and only for the named dependency.
* the CBM profile sets ``CBM_BRIDGE_URL`` (+ ``CBM_ENABLED=true``) and waives
  nothing, so an unreachable bridge there fails with a message naming it.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Callable, Mapping

RUN, SKIP, FAIL = "run", "skip", "fail"
_TRUTHY = {"1", "true", "yes", "on"}


def integration_required(env: Mapping[str, str]) -> bool:
    """Same rule as ``conftest.integration_required``: the gate or GitHub Actions."""

    if env.get("GITHUB_ACTIONS", "").strip().lower() == "true":
        return True
    return env.get("NS_INTEGRATION_REQUIRED", "").strip().lower() in _TRUTHY


def waived(env: Mapping[str, str], dependency: str) -> bool:
    names = {n.strip().lower() for n in env.get("NS_INTEGRATION_WAIVE", "").split(",")}
    return dependency.lower() in names


def probe_bridge(url: str | None, timeout: float = 3.0) -> bool:
    if not url:
        return False
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as resp:  # noqa: S310
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def cbm_decision(
    env: Mapping[str, str] | None = None, probe: Callable[[str | None], bool] = probe_bridge
) -> tuple[str, str]:
    """Return ``(RUN|SKIP|FAIL, reason)`` for the CBM bridge test."""

    env = os.environ if env is None else env
    if probe(env.get("CBM_BRIDGE_URL") or None):
        return RUN, ""
    reason = (
        "cbm-bridge not reachable: set CBM_BRIDGE_URL to a running cbm-bridge "
        "(the standard store stack does not run it)"
    )
    if integration_required(env) and not waived(env, "cbm"):
        return FAIL, (
            f"NS_INTEGRATION_REQUIRED is set but {reason}; start the cbm-bridge profile or, "
            "for a gate profile that does not provide CBM, set NS_INTEGRATION_WAIVE=cbm"
        )
    return SKIP, reason
