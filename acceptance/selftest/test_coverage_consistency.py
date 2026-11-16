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
