"""``@expected_red(reason)``: a scenario that is written against the behaviour a later slice delivers.

It is NOT a skip and NOT an xfail. The scenario still runs and a failure is still reported as ``fail``;
the label only lets a reader tell a known red from a new one. Two things carry the label, so it is
machine-readable with or without the results-schema change of the coverage-closure PR:

* the ``ns_expected_red`` pytest marker (the same marker name and one positional reference argument
  that PR's ``nsaccept.expected_red`` uses, so when it lands its plugin reports ``expected_red`` per
  scenario with no change here);
* an ``EXPECTED-RED[<reason>]`` prefix on the assertion message, which the results JSON ``detail``
  already carries today.

The reason is one of a closed set of neutral slice names (``REASONS``). A scenario that passes anyway is
simply a pass.

Note: the coverage-closure PR adds its own ``nsaccept.expected_red`` in ``decor.py`` with the same marker
name and one positional reference. Keep the two compatible; this module can then re-export that one.
"""

from __future__ import annotations

import functools
from typing import Callable

import pytest

REASONS = frozenset({
    "needs Phase 2 durable write cutover",
    "needs Phase 2 membership and onboarding",
    "needs Phase 2 onboarding policy",
    "needs Phase 2 tenant cells",
    "needs Phase 2 portable export",
})
TAG = "EXPECTED-RED"


def expected_red(reason: str) -> Callable:
    """Label a scenario that stays red until the slice named in ``reason`` is on dev."""
    if reason not in REASONS:
        raise ValueError(f"@expected_red reason must be one of {sorted(REASONS)}, got {reason!r}")

    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def run(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except AssertionError as exc:
                raise AssertionError(f"{TAG}[{reason}] {exc}") from None

        return pytest.mark.ns_expected_red(reason)(run)

    return decorate
