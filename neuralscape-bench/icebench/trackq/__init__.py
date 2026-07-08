"""
Track-Q generator and scorer (H3).

H3 implements:
- Oracle tree-sitter pass for structural QA ground truth
- Structural QA query sets (symbol_lookup, neighbors_1hop, path_le4)
- Docstring-locate query sets (nl_locate)
- Normalization and scoring (hits@k, MRR, precision/recall)
- LSP spot-check (optional, requires pyright/gopls)
"""

from icebench.trackq.generate import generate_queries, generate_specs, QuerySpec
from icebench.trackq.score import score_results, normalize_answer, ScoreReport, OpScore
from icebench.trackq.oracle import TreeSitterOracle, Symbol, Edge

__all__ = [
    "generate_queries",  # runner-facing: returns list[dict] payloads
    "generate_specs",  # scorer/tests: returns list[QuerySpec] (payload + gold)
    "QuerySpec",
    "score_results",
    "normalize_answer",
    "ScoreReport",
    "OpScore",
    "TreeSitterOracle",
    "Symbol",
    "Edge",
]
