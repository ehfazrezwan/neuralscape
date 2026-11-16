"""Evidence declarations: @proves, @profiles, @needs, @negative_control."""

from __future__ import annotations

from typing import Callable

import pytest

DEFAULT_PROFILES = ("default",)


def proves(*atlas_node_ids: str) -> Callable:
    """Declare which Feature Atlas node ids this scenario is evidence for."""
    if not atlas_node_ids:
        raise ValueError("@proves needs at least one atlas node id")
    return pytest.mark.ns_proves(*atlas_node_ids)


def profiles(*names: str) -> Callable:
    """Run this scenario only when the active --profile is one of ``names``."""
    if not names:
        raise ValueError("@profiles needs at least one profile name")
    return pytest.mark.ns_profiles(*names)


def needs(*capabilities: str) -> Callable:
    """Skip (with a reason) when a capability is absent: node, docling, ..."""
    return pytest.mark.ns_needs(*capabilities)


def expected_red(reference: str) -> Callable:
    """Mark a scenario as known to fail until ``reference`` (a PR) lands.

    NOT an xfail: the scenario still runs and a failure is still reported as ``fail``; the results
    JSON only carries ``expected_red`` (the reference) so a reader can tell known from new reds."""
    if not reference:
        raise ValueError("@expected_red needs a reference, e.g. '#515'")
    return pytest.mark.ns_expected_red(reference)


def negative_control(fn: Callable) -> Callable:
    """Mark a scenario as a negative control (reported with negative_control=true)."""
    return pytest.mark.ns_negative_control(fn)
