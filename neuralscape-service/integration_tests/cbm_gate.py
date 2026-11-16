"""Decide what a CBM flag-on test does when the cbm-bridge is (not) reachable.

The convention (``conftest.unavailable``): under ``NS_INTEGRATION_REQUIRED`` (the
gate) a missing dependency FAILS, anywhere else the test SKIPS.  The CBM bridge
is not part of the standard store stack, so a gate profile that deliberately does
not run it must say so out loud:

* ``NS_INTEGRATION_WAIVE=cbm`` (comma list, exact names) turns the REQUIRED
  failure back into a skip for that profile, and only for the named dependency.
  It never applies when ``CBM_BRIDGE_URL`` is set: a gate that names a bridge
  meant to test CBM, so an unreachable one fails.
* the CBM profile sets ``CBM_BRIDGE_URL`` (+ ``CBM_ENABLED=true``) and waives
  nothing.

"Required" is ``conftest.integration_required()`` itself, read from
``os.environ`` (tests use ``monkeypatch.setenv``), so the truthy set cannot drift.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Callable
from urllib.parse import urlparse

RUN, SKIP, FAIL = "run", "skip", "fail"


def integration_required() -> bool:
    from integration_tests.conftest import integration_required as _required

    return _required()


def waived(dependency: str) -> bool:
    names = {n.strip().lower() for n in os.environ.get("NS_INTEGRATION_WAIVE", "").split(",")}
    return dependency.lower() in names


def probe_bridge(url: str | None, timeout: float = 3.0) -> bool:
    if not url or urlparse(url).scheme not in ("http", "https"):
        return False
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as resp:  # noqa: S310
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def cbm_decision(probe: Callable[[str | None], bool] = probe_bridge) -> tuple[str, str]:
    """Return ``(RUN|SKIP|FAIL, reason)`` for the CBM bridge test, from ``os.environ``."""

    url = os.environ.get("CBM_BRIDGE_URL") or None
    if probe(url):
        return RUN, ""
    reason = (
        "cbm-bridge not reachable: set CBM_BRIDGE_URL to a running cbm-bridge "
        "(the standard store stack does not run it)"
    )
    if url:
        if integration_required():
            return FAIL, f"CBM_BRIDGE_URL is set but the bridge is unreachable (waiver does not apply): {reason}"
        return SKIP, reason
    if integration_required() and not waived("cbm"):
        return FAIL, (
            f"NS_INTEGRATION_REQUIRED is set but {reason}; start the cbm-bridge profile or, "
            "for a gate profile that does not provide CBM, set NS_INTEGRATION_WAIVE=cbm"
        )
    return SKIP, reason
