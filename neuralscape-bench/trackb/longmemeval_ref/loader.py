"""Dataset loading for LongMemEval.

Reuses the existing neuralscape_bench.accuracy.suites.longmemeval loader
(read-only) to avoid duplication. The shape is well-tested and stable.
"""

from __future__ import annotations

from pathlib import Path

from neuralscape_bench.accuracy.suites.longmemeval import FILES, parse, read_json

DATASET_DIR = Path(__file__).parent.parent.parent / "datasets" / "longmemeval"


def load_longmemeval_s(*, sample: int | None = None, seed: int = 42):
    """Load LongMemEval_S from the on-disk dataset.

    Returns a SuiteData object with conversations and qa_items.
    If sample is provided, stratified sample across question types.
    """
    path = DATASET_DIR / FILES["longmemeval_s"]
    if not path.exists():
        raise FileNotFoundError(
            f"LongMemEval_S dataset not found at {path}. "
            "Expected datasets/longmemeval/longmemeval_s_cleaned.json in the repo."
        )

    data = parse(read_json(path), variant="longmemeval_s")

    if sample is not None:
        from neuralscape_bench.accuracy.sampling import stratified_sample

        data.qa_items = stratified_sample(data.qa_items, sample, seed=seed)
        keep = {qa.conv_id for qa in data.qa_items}
        # Each question owns its haystack — drop conversations for unsampled questions
        data.conversations = [c for c in data.conversations if c.conv_id in keep]

    return data
