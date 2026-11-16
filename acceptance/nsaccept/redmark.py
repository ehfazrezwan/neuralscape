"""``@expected_red(reason)``: a scenario that is written against the behaviour a later slice delivers.

It is NOT a skip and NOT an xfail. The scenario still runs and a failure is still reported as ``fail``;
the label only lets a reader tell a known red from a new one. Two things carry the label, so it is
machine-readable with or without the results-schema change of the coverage-closure PR:

* the ``ns_expected_red`` pytest marker (the same marker name and one positional reference argument
  that PR's ``nsaccept.expected_red`` uses, so when it lands its plugin reports ``expected_red`` per
  scenario with no change here);
* an ``EXPECTED-RED[<reason>]`` prefix on the assertion message, which the results JSON ``detail``
  already carries today.

The reason names the slice and its lane in neutral words:
``needs Phase 2 <slice name> (lane p2 <map id>)``. A scenario that passes anyway is simply a pass.
"""

from __future__ import annotations

import functools
import re
from typing import Callable

import pytest

REASON_RE = re.compile(r"^needs Phase 2 [a-z][a-z \-]+ \(lane p2 p2-\d{2}(?:, p2-\d{2})*\)$")
TAG = "EXPECTED-RED"


def expected_red(reason: str) -> Callable:
    """Label a scenario that stays red until the slice named in ``reason`` is on dev."""
    if not REASON_RE.match(reason or ""):
        raise ValueError(f"@expected_red reason must look like 'needs Phase 2 <slice> (lane p2 p2-NN)', got {reason!r}")

    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def run(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except AssertionError as exc:
                raise AssertionError(f"{TAG}[{reason}] {exc}") from None

        return pytest.mark.ns_expected_red(reason)(run)

    return decorate
