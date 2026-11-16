"""Static consistency: every @expected_red("#N") reference in the scenarios is listed in COVERAGE.md with a status.

Whether the cited PRs are still open is checked by hand when COVERAGE.md is refreshed (it needs the network)."""

import re
from pathlib import Path

ACCEPTANCE = Path(__file__).resolve().parents[1]
STATUSES = {"gap", "blocked", "dead-code", "not-started", "partial"}


def test_expected_red_references_appear_in_coverage_with_a_status():
    coverage = (ACCEPTANCE / "COVERAGE.md").read_text()
    rows = [line for line in coverage.splitlines() if line.startswith("| `")]
    assert rows and all(line.split("|")[2].strip() in STATUSES for line in rows), "COVERAGE.md rows need a known status"
    refs = set()
    for path in (ACCEPTANCE / "scenarios").glob("test_*.py"):
        refs |= set(re.findall(r'@expected_red\("(#\d+)"\)', path.read_text()))
    missing = sorted(r for r in refs if not any(r in row for row in rows))
    assert not missing, f"@expected_red references with no COVERAGE.md row naming them: {missing}"


def test_no_coverage_listed_node_is_claimed_by_a_non_control_scenario():
    """A node listed as uncovered must not carry an @proves on a scenario that is neither a negative control
    nor an expected-red scenario (a claim would make the listing wrong)."""
    import ast

    listed = set(re.findall(r"^\| `([^`]+)` \|", (ACCEPTANCE / "COVERAGE.md").read_text(), re.M))
    assert listed
    offenders = []
    for path in (ACCEPTANCE / "scenarios").glob("test_*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef):
                continue
            names = {getattr(d, "id", getattr(getattr(d, "func", None), "id", getattr(getattr(d, "func", None), "attr", getattr(d, "attr", "")))) for d in node.decorator_list}
            if names & {"negative_control", "expected_red"}:
                continue
            for d in node.decorator_list:
                if isinstance(d, ast.Call) and getattr(d.func, "id", getattr(d.func, "attr", "")) == "proves":
                    for arg in d.args:
                        if isinstance(arg, ast.Constant) and arg.value in listed:
                            offenders.append(f"{path.name}::{node.name} claims {arg.value}")
    assert not offenders, offenders
