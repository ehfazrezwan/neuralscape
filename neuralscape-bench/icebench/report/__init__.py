"""
ICEBench report generator (H4).

Turns raw results (icebench-v1 JSONL) + Track-Q ScoreReport into:
- Markdown report
- Self-contained HTML dashboard

Design principles (Fable audit targets):
- N/A ≠ 0 (never fabricate numbers for unsupported ops)
- DNF is a recorded result (not a blank)
- Medians reported with min/max
- Honest axes (start at 0, no truncation tricks)
"""

from .generator import generate_report

__all__ = ["generate_report"]
