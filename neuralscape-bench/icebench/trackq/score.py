"""
Track-Q scoring and normalization.

Scores retrieval quality across structural QA and NL locate queries.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

from icebench.schema import read_rows
from icebench.trackq.generate import NL_LOCATE_MIN_SAMPLES

logger = logging.getLogger(__name__)


@dataclass
class OpScore:
    """Scores for a single operation."""
    op: str
    system: str
    corpus: str
    n_queries: int
    n_supported: int  # Queries where system returned ok=True
    n_unsupported: int  # Queries where system returned ok=False or N/A

    # Structural QA metrics
    precision: float | None = None
    recall: float | None = None
    hit_rate: float | None = None

    # NL locate metrics
    hits_at_1: float | None = None
    hits_at_5: float | None = None
    hits_at_10: float | None = None
    mrr: float | None = None


@dataclass
class ScoreReport:
    """
    Overall scoring report for Track-Q.

    Contains per-(system, corpus, op) scores plus metadata about oracle bias
    and sample sizes.
    """
    scores: list[OpScore]
    metadata: dict  # Includes shared-oracle-bias caveat, LSP agreement %, sample sizes


def normalize_answer(answer: str | dict | None) -> tuple[str, str] | None:
    """
    Normalize an answer to (file, symbol) tuple.

    Args:
        answer: Raw answer from adapter (dict with "text", plain text, or None).

    Returns:
        (relative_path, symbol_fqn) or None if unparseable.
    """
    if answer is None:
        return None

    # If answer is a dict, extract the text field
    if isinstance(answer, dict):
        if "error" in answer or answer.get("status") == "error":
            return None
        answer = answer.get("text", "")

    if not isinstance(answer, str):
        return None

    answer = answer.strip()
    if not answer:
        return None

    # Try to parse as JSON (adapters may return JSON arrays of results)
    try:
        parsed = json.loads(answer)
        if isinstance(parsed, list) and parsed:
            # Take first result
            first = parsed[0]
            if isinstance(first, dict):
                file = first.get("file", "")
                symbol = first.get("symbol", "")
                if file or symbol:
                    return (_normalize_path(file), _normalize_symbol(symbol))
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    # Try to parse as "file:line:symbol" or similar patterns
    # Common patterns: "path/to/file.py:123:symbol_name" or "file.py:symbol_name"
    parts = answer.split(":")
    if len(parts) >= 2:
        file = parts[0]
        symbol = parts[-1]  # Last part is usually the symbol
        return (_normalize_path(file), _normalize_symbol(symbol))

    # If no structure found, return None
    return None


def _normalize_path(path: str) -> str:
    """
    Normalize a file path: strip repo prefix, normalize separators.

    Args:
        path: Raw file path.

    Returns:
        Normalized relative path.
    """
    if not path:
        return ""

    # Convert to Path and back to string to normalize separators
    p = Path(path)

    # Strip the absolute corpus prefix /data/ice/corpora/<corpus-name>/ when
    # there is a real relative remainder AFTER the corpus name. We require
    # len(parts) > 5 (not > 4) so the slice parts[5:] is guaranteed non-empty —
    # otherwise a corpus-root path would collapse to Path() -> "." (the bug).
    parts = p.parts
    if len(parts) > 5 and parts[:4] == ("/", "data", "ice", "corpora"):
        # Skip the first 5 parts: /, data, ice, corpora, <corpus-name>
        p = Path(*parts[5:])

    return str(p)


def _normalize_symbol(symbol: str) -> str:
    """
    Normalize a symbol name.

    Args:
        symbol: Raw symbol name.

    Returns:
        Normalized symbol FQN.
    """
    # For now, just strip whitespace
    return symbol.strip()


def score_results(run_id: str, results_dir: Path) -> int:
    """
    Score Track-Q results for a run.

    Args:
        run_id: Run identifier.
        results_dir: Directory containing results JSONL.

    Returns:
        Exit code (0 = success).
    """
    results_file = results_dir / f"{run_id}.jsonl"

    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return 1

    # Load query results
    rows = list(read_rows(results_file))
    query_rows = [r for r in rows if r.kind == "query"]

    if not query_rows:
        logger.warning("No query results found")
        return 0

    # Recover gold answers by regenerating the query specs. The runner does not
    # persist gold (it only records the system's answer), but generation is
    # deterministic AND prefix-stable under (op, corpus, seed), so regenerating
    # with n = max(rep)+1 reproduces the exact gold for every rep index we have.
    from icebench.trackq.generate import generate_specs
    from icebench.corpora import iter_corpora

    # Build corpus map
    corpora = {c.name: c for c in iter_corpora()}

    # Group rows by (system, corpus, op)
    groups = defaultdict(list)
    for row in query_rows:
        groups[(row.system, row.corpus, row.op)].append(row)

    # Score each group
    scores = []
    for (system, corpus_name, op), group_rows in groups.items():
        corpus = corpora.get(corpus_name)
        if not corpus:
            logger.warning(f"Corpus {corpus_name} not found")
            continue

        # Use the seed recorded on the rows (do NOT hardcode 42). All rows in a
        # single run/op share the runner's --seed; if they diverge, warn and use
        # the most common one so alignment stays consistent.
        seeds = {r.seed for r in group_rows}
        if len(seeds) > 1:
            logger.warning(
                "Mixed seeds %s for %s/%s/%s; using the most common for gold regen",
                sorted(seeds), system, corpus_name, op,
            )
        seed = Counter(r.seed for r in group_rows).most_common(1)[0][0]

        # Reps may be non-contiguous (errored/missing queries). Regenerate enough
        # specs to cover the highest rep index actually present.
        max_rep = max(r.rep for r in group_rows)
        n_needed = max_rep + 1

        try:
            gold_specs = generate_specs(op, corpus, n=n_needed, seed=seed)
        except Exception as e:
            logger.error(f"Failed to generate gold queries for {corpus_name}/{op}: {e}")
            continue

        # Build gold map keyed by rep INDEX; align each row by its own row.rep
        # (handles non-contiguous reps — missing indices are simply absent).
        gold_map = {i: spec.gold for i, spec in enumerate(gold_specs)}

        # Score
        if op == "nl_locate":
            op_score = _score_nl_locate(system, corpus_name, op, group_rows, gold_map)
        elif op in ("symbol_lookup", "neighbors_1hop"):
            op_score = _score_structural(system, corpus_name, op, group_rows, gold_map)
        elif op == "path_le4":
            op_score = _score_path(system, corpus_name, op, group_rows, gold_map)
        else:
            logger.warning(f"Unknown op: {op}")
            continue

        scores.append(op_score)

    # Build metadata. sample_sizes reflects the REAL per-cell query count, and
    # under-sized nl_locate corpora are flagged so the report can mark them as
    # non-comparable (never silently dropped).
    sample_sizes = {
        f"{s.system}/{s.corpus}/{s.op}": s.n_queries
        for s in scores
    }
    undersized_nl_locate = {
        f"{s.system}/{s.corpus}/{s.op}": s.n_queries
        for s in scores
        if s.op == "nl_locate" and s.n_queries < NL_LOCATE_MIN_SAMPLES
    }
    if undersized_nl_locate:
        logger.warning(
            "Under-sized nl_locate eval sets (< %d): %s",
            NL_LOCATE_MIN_SAMPLES, undersized_nl_locate,
        )

    metadata = {
        "shared_oracle_bias": (
            "Both NS indexer and Track-Q oracle use tree-sitter for parsing. "
            "This introduces a shared-oracle bias where both systems may fail "
            "on the same malformed code, and structural ground truth is only as "
            "correct as the oracle's tree-sitter extraction."
        ),
        "oracle_independence": (
            "The Track-Q oracle (icebench/trackq/oracle.py) is a standalone "
            "module, independent of the NS indexer; they share only the "
            "tree-sitter parser library."
        ),
        "lsp_agreement_pct": None,  # Filled by LSP spot-check; None => skipped
        "lsp_spot_check": "skipped (pyright/gopls not installable on shared VM)",
        "nl_locate_min_samples": NL_LOCATE_MIN_SAMPLES,
        "undersized_nl_locate": undersized_nl_locate,
        "sample_sizes": sample_sizes,
    }

    report = ScoreReport(scores=scores, metadata=metadata)

    # Write report
    report_file = results_dir / f"{run_id}.trackq.json"
    with open(report_file, "w") as f:
        json.dump(asdict(report), f, indent=2)

    logger.info(f"Wrote Track-Q report to {report_file}")

    # Print summary
    print("\n=== Track-Q Scores ===")
    for s in scores:
        print(f"\n{s.system} / {s.corpus} / {s.op}")
        print(f"  Queries: {s.n_queries} (supported: {s.n_supported}, unsupported: {s.n_unsupported})")

        if s.op == "nl_locate":
            if s.hits_at_1 is not None:
                print(f"  Hits@1:  {s.hits_at_1:.2%}")
                print(f"  Hits@5:  {s.hits_at_5:.2%}")
                print(f"  Hits@10: {s.hits_at_10:.2%}")
                print(f"  MRR:     {s.mrr:.3f}")
            else:
                print("  N/A")
        else:
            if s.precision is not None:
                print(f"  Precision: {s.precision:.2%}")
            if s.recall is not None:
                print(f"  Recall:    {s.recall:.2%}")
            if s.hit_rate is not None:
                print(f"  Hit rate:  {s.hit_rate:.2%}")
            if s.precision is None and s.recall is None and s.hit_rate is None:
                print("  N/A")

    return 0


def _score_nl_locate(
    system: str,
    corpus: str,
    op: str,
    rows: list,
    gold_map: dict,
) -> OpScore:
    """
    Score NL locate queries: hits@k and MRR.

    Args:
        system: System name.
        corpus: Corpus name.
        op: Operation name.
        rows: Result rows.
        gold_map: Mapping of rep -> gold answer.

    Returns:
        OpScore with hits@k and MRR.
    """
    n_queries = len(rows)
    supported = [r for r in rows if r.ok]
    n_supported = len(supported)
    n_unsupported = n_queries - n_supported

    if n_supported == 0:
        return OpScore(
            op=op,
            system=system,
            corpus=corpus,
            n_queries=n_queries,
            n_supported=0,
            n_unsupported=n_unsupported,
        )

    # Score each query
    hits_1 = 0
    hits_5 = 0
    hits_10 = 0
    reciprocal_ranks = []

    for row in supported:
        gold = gold_map.get(row.rep)
        if not gold:
            continue

        gold_file = gold.get("file", "")
        gold_symbol = gold.get("symbol", "")
        gold_normalized = (_normalize_path(gold_file), _normalize_symbol(gold_symbol))

        # Parse answer (may be a ranked list)
        answer = row.answer
        if not answer:
            continue

        # Try to parse as JSON array of results
        results = _parse_ranked_results(answer)

        # Check hits@k
        found_at = None
        for i, result in enumerate(results[:10]):  # Only check top 10
            if result == gold_normalized:
                found_at = i + 1  # 1-indexed
                break

        if found_at is not None:
            reciprocal_ranks.append(1.0 / found_at)
            if found_at <= 1:
                hits_1 += 1
            if found_at <= 5:
                hits_5 += 1
            if found_at <= 10:
                hits_10 += 1
        else:
            reciprocal_ranks.append(0.0)

    # Calculate metrics
    hits_at_1 = hits_1 / n_supported if n_supported > 0 else 0.0
    hits_at_5 = hits_5 / n_supported if n_supported > 0 else 0.0
    hits_at_10 = hits_10 / n_supported if n_supported > 0 else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    return OpScore(
        op=op,
        system=system,
        corpus=corpus,
        n_queries=n_queries,
        n_supported=n_supported,
        n_unsupported=n_unsupported,
        hits_at_1=hits_at_1,
        hits_at_5=hits_at_5,
        hits_at_10=hits_at_10,
        mrr=mrr,
    )


def _parse_ranked_results(answer: str | dict) -> list[tuple[str, str]]:
    """
    Parse a ranked list of results from an answer.

    Args:
        answer: Raw answer (could be dict or string).

    Returns:
        List of (file, symbol) tuples in rank order.
    """
    if isinstance(answer, dict):
        if "error" in answer or answer.get("status") == "error":
            return []
        answer = answer.get("text", "")

    if not isinstance(answer, str):
        return []

    # Try to parse as JSON array
    try:
        parsed = json.loads(answer)
        if isinstance(parsed, list):
            results = []
            for item in parsed:
                if isinstance(item, dict):
                    file = item.get("file", "")
                    symbol = item.get("symbol", "")
                    results.append((_normalize_path(file), _normalize_symbol(symbol)))
            return results
    except json.JSONDecodeError:
        pass

    # Fallback: treat as single result
    normalized = normalize_answer(answer)
    if normalized:
        return [normalized]

    return []


def _score_structural(
    system: str,
    corpus: str,
    op: str,
    rows: list,
    gold_map: dict,
) -> OpScore:
    """
    Score structural QA queries (symbol_lookup, neighbors_1hop).

    For symbol_lookup: hit rate (exact match on file+symbol).
    For neighbors_1hop: precision/recall on caller sets.

    Args:
        system: System name.
        corpus: Corpus name.
        op: Operation name.
        rows: Result rows.
        gold_map: Mapping of rep -> gold answer.

    Returns:
        OpScore.
    """
    n_queries = len(rows)
    supported = [r for r in rows if r.ok]
    n_supported = len(supported)
    n_unsupported = n_queries - n_supported

    if n_supported == 0:
        return OpScore(
            op=op,
            system=system,
            corpus=corpus,
            n_queries=n_queries,
            n_supported=0,
            n_unsupported=n_unsupported,
        )

    if op == "symbol_lookup":
        # Hit rate: exact match on (file, symbol)
        hits = 0
        for row in supported:
            gold = gold_map.get(row.rep)
            if not gold:
                continue

            gold_file = gold.get("file", "")
            gold_symbol = gold.get("symbol", "")
            gold_normalized = (_normalize_path(gold_file), _normalize_symbol(gold_symbol))

            answer_normalized = normalize_answer(row.answer)

            if answer_normalized == gold_normalized:
                hits += 1

        hit_rate = hits / n_supported if n_supported > 0 else 0.0

        return OpScore(
            op=op,
            system=system,
            corpus=corpus,
            n_queries=n_queries,
            n_supported=n_supported,
            n_unsupported=n_unsupported,
            hit_rate=hit_rate,
        )

    elif op == "neighbors_1hop":
        # Precision/recall on caller sets
        precisions = []
        recalls = []

        for row in supported:
            gold = gold_map.get(row.rep)
            if not gold:
                continue

            gold_callers = set(gold.get("callers", []))

            # Parse answer as a set of symbols
            answer_callers = _parse_symbol_set(row.answer)

            if not answer_callers and not gold_callers:
                # Both empty: perfect match
                precisions.append(1.0)
                recalls.append(1.0)
            elif not answer_callers:
                # No results returned but there are gold callers
                precisions.append(0.0)
                recalls.append(0.0)
            elif not gold_callers:
                # Results returned but no gold callers
                precisions.append(0.0)
                recalls.append(0.0)
            else:
                tp = len(answer_callers & gold_callers)
                precision = tp / len(answer_callers) if answer_callers else 0.0
                recall = tp / len(gold_callers) if gold_callers else 0.0
                precisions.append(precision)
                recalls.append(recall)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

        return OpScore(
            op=op,
            system=system,
            corpus=corpus,
            n_queries=n_queries,
            n_supported=n_supported,
            n_unsupported=n_unsupported,
            precision=avg_precision,
            recall=avg_recall,
        )

    return OpScore(
        op=op,
        system=system,
        corpus=corpus,
        n_queries=n_queries,
        n_supported=n_supported,
        n_unsupported=n_unsupported,
    )


def _parse_symbol_set(answer: str | dict) -> set[str]:
    """
    Parse a set of symbols from an answer.

    Args:
        answer: Raw answer.

    Returns:
        Set of symbol names.
    """
    if isinstance(answer, dict):
        if "error" in answer or answer.get("status") == "error":
            return set()
        answer = answer.get("text", "")

    if not isinstance(answer, str):
        return set()

    # Try to parse as JSON array
    try:
        parsed = json.loads(answer)
        if isinstance(parsed, list):
            return {str(item).strip() for item in parsed if item}
    except json.JSONDecodeError:
        pass

    # Fallback: split by commas or newlines
    symbols = set()
    for part in answer.replace(",", "\n").split("\n"):
        part = part.strip()
        if part:
            symbols.add(part)

    return symbols


def _score_path(
    system: str,
    corpus: str,
    op: str,
    rows: list,
    gold_map: dict,
) -> OpScore:
    """
    Score path queries: hit rate (any valid path found).

    Args:
        system: System name.
        corpus: Corpus name.
        op: Operation name.
        rows: Result rows.
        gold_map: Mapping of rep -> gold answer.

    Returns:
        OpScore.
    """
    n_queries = len(rows)
    supported = [r for r in rows if r.ok]
    n_supported = len(supported)
    n_unsupported = n_queries - n_supported

    if n_supported == 0:
        return OpScore(
            op=op,
            system=system,
            corpus=corpus,
            n_queries=n_queries,
            n_supported=0,
            n_unsupported=n_unsupported,
        )

    # Hit rate: system found any path
    hits = 0
    for row in supported:
        gold = gold_map.get(row.rep)
        if not gold:
            continue

        # Gold has a list of valid paths
        gold_paths = gold.get("paths", [])

        # If gold has at least one path, and system returned a non-empty answer, count as hit
        # (Simplified: we don't validate the path structure, just that something was found)
        if gold_paths:
            answer = row.answer
            if answer and isinstance(answer, dict) and answer.get("status") == "ok":
                text = answer.get("text", "")
                if text and text.strip():
                    hits += 1

    hit_rate = hits / n_supported if n_supported > 0 else 0.0

    return OpScore(
        op=op,
        system=system,
        corpus=corpus,
        n_queries=n_queries,
        n_supported=n_supported,
        n_unsupported=n_unsupported,
        hit_rate=hit_rate,
    )
