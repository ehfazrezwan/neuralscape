"""Unexpected-skip gate for the bench suites (standard library only).

A green pytest run can hide skipped tests (a missing optional dependency, an
absent binary).  This script reads a pytest JUnit report and compares its
skips with ``ci/expected_skips.json``.  A run is rejected when it has
failures or errors, ran nothing, skipped a test that is not declared,
skipped a declared test for a different reason, or no longer skips a declared
test.

    python scripts/ci_skip_gate.py --junit junit.xml --manifest ci/expected_skips.json

Exit status: 0 accepted, 1 rejected, 2 unusable input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from xml.etree import ElementTree

ENTRY_KEYS = {"test", "reason_contains", "why"}


class GateInputError(Exception):
    """The JUnit report or the manifest cannot be interpreted."""


def load_manifest(path) -> dict[str, str]:
    """Return {test_id: reason_substring} from the manifest file."""

    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateInputError(f"manifest unreadable: {exc}") from exc
    if not isinstance(obj, dict) or obj.get("version") != 1:
        raise GateInputError("manifest version must be 1")
    skips = obj.get("skips")
    if not isinstance(skips, list):
        raise GateInputError("manifest needs a 'skips' list")
    expected: dict[str, str] = {}
    for entry in skips:
        if not isinstance(entry, dict):
            raise GateInputError("skip entries must be objects")
        unknown = set(entry) - ENTRY_KEYS
        if unknown:
            raise GateInputError(f"unknown keys in skip entry: {sorted(unknown)}")
        for key in ENTRY_KEYS:
            value = entry.get(key)
            if not (isinstance(value, str) and value.strip()):
                raise GateInputError(f"skip entry needs a non-empty {key!r}: {entry}")
        if entry["test"] in expected:
            raise GateInputError(f"duplicate skip entry {entry['test']!r}")
        expected[entry["test"]] = entry["reason_contains"]
    return expected


def load_junit(path_or_text) -> list[tuple[str, str, str]]:
    """Return [(test_id, outcome, detail)]; outcome is passed/skipped/failed/error."""

    try:
        if isinstance(path_or_text, str) and path_or_text.lstrip().startswith("<"):
            root = ElementTree.fromstring(path_or_text)
        else:
            root = ElementTree.parse(str(path_or_text)).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        raise GateInputError(f"junit report unreadable: {exc}") from exc
    cases = []
    for case in root.iter("testcase"):
        test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
        if case.find("failure") is not None:
            cases.append((test_id, "failed", ""))
        elif case.find("error") is not None:
            cases.append((test_id, "error", ""))
        elif (node := case.find("skipped")) is not None:
            text = " ".join((node.text or "").split())
            detail = f"{node.get('type', '')} {node.get('message', '')} {text}".strip()
            cases.append((test_id, "skipped", detail))
        else:
            cases.append((test_id, "passed", ""))
    return cases


def check(cases, expected: dict[str, str]) -> list[str]:
    """Return violation strings; empty means the run is accepted."""

    problems = []
    if not cases:
        problems.append("the report contains no test cases")
    seen: dict[str, str] = {}
    for test_id, outcome, detail in cases:
        if outcome in ("failed", "error"):
            problems.append(f"{outcome}: {test_id}")
        elif outcome == "skipped":
            seen[test_id] = detail
    for test_id, detail in sorted(seen.items()):
        if test_id not in expected:
            problems.append(f"unexpected skip: {test_id} [{detail}]")
        elif expected[test_id] not in detail:
            problems.append(
                f"skip reason changed: {test_id} [{detail}] lacks {expected[test_id]!r}"
            )
    for test_id in sorted(expected):
        if test_id not in seen:
            problems.append(f"declared skip did not skip: {test_id}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--junit", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    try:
        expected = load_manifest(args.manifest)
        cases = load_junit(args.junit)
    except GateInputError as exc:
        print(f"skip-gate: {exc}", file=sys.stderr)
        return 2
    problems = check(cases, expected)
    skipped = sum(1 for _, o, _ in cases if o == "skipped")
    print(
        f"skip-gate: {len(cases)} cases, {skipped} skipped, "
        f"{len(expected)} skips declared"
    )
    for problem in problems:
        print(f"skip-gate: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
