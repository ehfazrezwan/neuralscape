"""
Deterministic task set for the token-savings navigation benchmark.

We reuse the ICEBench Track-Q generator (`icebench.trackq.generate.generate_specs`)
so the tasks are oracle-backed (tree-sitter ground truth) and deterministic +
prefix-stable under a fixed seed. A token-savings *task* is a Track-Q spec plus a
natural-language prompt phrased the way a developer would ask an LLM to navigate:

  * locate         (nl_locate)      "Where is the code that <docstring>?"
  * symbol_lookup  (symbol_lookup)  "Where is `<Symbol>` defined?"
  * neighbors      (neighbors_1hop) "What calls `<Symbol>`?"

The task set is balanced across the three goal-relevant op classes and sliced
deterministically (seed 42), so every arm sees the identical task list.
"""

import logging
from dataclasses import dataclass, field

from icebench.adapters.base import Corpus
from icebench.trackq.generate import generate_specs

logger = logging.getLogger(__name__)

# The three navigation op classes the North Star cares about, mapped to their
# underlying Track-Q op names. `locate` and `symbol_lookup` are "find a location";
# `neighbors` is the "what-connects" follow-up.
OP_CLASSES = {
    "locate": "nl_locate",
    "symbol_lookup": "symbol_lookup",
    "neighbors": "neighbors_1hop",
}

# Default tasks per op class -> 3 * 10 = 30 total (brief: 20-40).
DEFAULT_PER_OP = 10


@dataclass
class TokenSaveTask:
    """A single navigation task: a prompt + oracle-backed gold answer."""

    task_id: str
    op_class: str  # "locate" | "symbol_lookup" | "neighbors"
    track_q_op: str  # underlying generate_specs op
    prompt: str  # natural-language question posed to the agent
    payload: dict  # the Track-Q payload (carries the corpus + query keys)
    gold: dict  # oracle ground truth (op-specific shape)
    corpus_name: str
    metadata: dict = field(default_factory=dict)


def _clip(text: str, limit: int = 600) -> str:
    """Clip a docstring for the prompt so a giant docstring can't dominate token
    counts (and to keep the NL question realistic)."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …"


def _prompt_for(op_class: str, payload: dict, gold: dict) -> str:
    """Phrase the navigation question the way a developer would ask an LLM."""
    if op_class == "locate":
        desc = _clip(payload.get("query", ""))
        return (
            "Where in this codebase is the function or method whose documented "
            f"purpose is:\n\n    \"{desc}\"\n\n"
            "Report the single best matching definition as `path/to/file.py:LINE`."
        )
    if op_class == "symbol_lookup":
        sym = payload.get("symbol", "")
        return (
            f"Where is the symbol `{sym}` defined in this codebase? "
            "Report its definition location as `path/to/file.py:LINE`."
        )
    if op_class == "neighbors":
        sym = payload.get("symbol", "")
        return (
            f"Which functions or methods call `{sym}` in this codebase? "
            "List every direct caller by name (one per line)."
        )
    raise ValueError(f"unknown op_class: {op_class}")


def build_task_set(
    corpus: Corpus,
    per_op: int = DEFAULT_PER_OP,
    seed: int = 42,
    op_classes: list[str] | None = None,
) -> list[TokenSaveTask]:
    """
    Build the deterministic, balanced task set for a corpus.

    Args:
        corpus: The corpus to navigate (checkout on disk + language).
        per_op: Number of tasks per op class.
        seed: Fixed seed for reproducibility (default 42).
        op_classes: Subset of op classes to include (default: all three).

    Returns:
        A flat list of TokenSaveTask, grouped by op class in a stable order.
    """
    op_classes = op_classes or list(OP_CLASSES.keys())
    tasks: list[TokenSaveTask] = []

    for op_class in op_classes:
        track_q_op = OP_CLASSES[op_class]
        # generate_specs is prefix-stable: taking the first `per_op` is a stable
        # slice of the identical sequence every arm/run sees.
        specs = generate_specs(track_q_op, corpus, n=per_op, seed=seed)
        specs = specs[:per_op]
        if len(specs) < per_op:
            logger.warning(
                "op_class %s: only %d specs available (< requested %d)",
                op_class, len(specs), per_op,
            )
        for i, spec in enumerate(specs):
            tasks.append(
                TokenSaveTask(
                    task_id=f"{op_class}-{i:03d}",
                    op_class=op_class,
                    track_q_op=track_q_op,
                    prompt=_prompt_for(op_class, spec.payload, spec.gold),
                    payload=spec.payload,
                    gold=spec.gold,
                    corpus_name=corpus.name,
                    metadata={"seed": seed},
                )
            )

    logger.info(
        "built %d token-savings tasks (%d op classes x ~%d each) on %s",
        len(tasks), len(op_classes), per_op, corpus.name,
    )
    return tasks
