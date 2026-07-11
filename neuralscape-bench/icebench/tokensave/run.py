"""
Token-savings benchmark orchestrator.

Runs ONE (arm, condition) matrix over the task set and writes a JSONL of per-task
rows plus a summary JSON. Invoke it once per (arm, condition) — e.g. baseline
once (arm-independent), then the memory condition once per arm — and aggregate
with :mod:`report`. This mirrors the nl_locate A/B driver: each measured
invocation is independent and resumable.

Usage:
  # Baseline (no code memory) — arm-independent, run once:
  python -m icebench.tokensave.run --condition baseline --arm none \
      --out /data/ice-v2/results/raw/tokensave-baseline.jsonl

  # With memory, native arm (server configured for that arm):
  python -m icebench.tokensave.run --condition memory --arm native \
      --out /data/ice-v2/results/raw/tokensave-memory-native.jsonl
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from icebench.adapters.base import Corpus
from icebench.tokensave.agent import (
    BASELINE_SYSTEM,
    MEMORY_SYSTEM,
    DEFAULT_MAX_STEPS,
    run_agent,
)
from icebench.tokensave.gold import first_hop_hit, score_answer
from icebench.tokensave.tasks import DEFAULT_PER_OP, build_task_set
from icebench.tokensave.tools import (
    FILE_TOOL_DECLS,
    MEMORY_TOOL_DECL,
    SUBMIT_TOOL_DECL,
    FileTools,
    MemoryTool,
    ToolResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tokensave.run")

DEFAULT_CORPUS_PATH = "/data/ice/corpora/small-py@8a4ce842564ae94ab050062db8525196ad476c19"
DEFAULT_API_URL = "http://localhost:8699"


def _make_dispatch(file_tools: FileTools, memory_tool: MemoryTool | None):
    def dispatch(name: str, args: dict) -> ToolResult:
        if name == "list_dir":
            return file_tools.list_dir(args.get("path", "."))
        if name == "grep":
            return file_tools.grep(args.get("pattern", ""), args.get("path", "."))
        if name == "read_file":
            return file_tools.read_file(
                args.get("path", ""), args.get("start_line", 1), args.get("end_line")
            )
        if name == "code_memory" and memory_tool is not None:
            return memory_tool.code_memory(
                op=args.get("op", ""),
                symbol=args.get("symbol", ""),
                query=args.get("query", ""),
            )
        return ToolResult(f"unknown or unavailable tool: {name}", ok=False)

    return dispatch


def _run_one(task, rep, llm, corpus, condition, api_url, knowledge_system, max_steps):
    file_tools = FileTools(corpus.path)
    memory_tool = None
    if condition == "memory":
        memory_tool = MemoryTool(
            api_url=api_url,
            corpus_name=corpus.name,
            knowledge_system=knowledge_system,
        )
    dispatch = _make_dispatch(file_tools, memory_tool)

    t0 = time.perf_counter()
    try:
        result = run_agent(llm, dispatch, task.prompt, max_steps=max_steps)
    finally:
        if memory_tool is not None:
            memory_tool.close()  # avoid HTTP client FD leaks over long runs
    latency_s = time.perf_counter() - t0

    ans_score = score_answer(task.op_class, result.answer, task.gold)

    fh1 = fh5 = False
    first = result.first_memory_step()
    if condition == "memory" and result.first_tool() == "code_memory" and first is not None:
        fh1 = first_hop_hit(task.op_class, first.raw, task.gold, k=1)
        fh5 = first_hop_hit(task.op_class, first.raw, task.gold, k=5)

    return {
        "arm": None,  # filled by caller
        "condition": condition,
        "knowledge_system": knowledge_system if condition == "memory" else None,
        "task_id": task.task_id,
        "op_class": task.op_class,
        "corpus": corpus.name,
        "rep": rep,
        "correct": ans_score.correct,
        "answer": result.answer,
        "gold": task.gold if task.op_class != "locate" else {k: task.gold[k] for k in ("file", "symbol")},
        "score_detail": ans_score.detail,
        "tokens": result.usage.as_dict(),
        "n_tool_calls": result.n_tool_calls,
        "n_file_reads": result.n_file_reads,
        "n_memory_calls": result.n_memory_calls,
        "steps": result.steps,
        "stopped_reason": result.stopped_reason,
        "first_tool": result.first_tool(),
        "first_hop_hit_1": fh1,
        "first_hop_hit_5": fh5,
        "latency_s": latency_s,
        "trace": [
            {"step": s.step, "tool": s.tool, "args": s.args, "ok": s.ok,
             "obs_len": len(s.observation)}
            for s in result.trace
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, choices=["baseline", "memory"])
    ap.add_argument("--arm", default="none", help="Arm label for tagging output.")
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--per-op", type=int, default=DEFAULT_PER_OP)
    ap.add_argument("--op-classes", default="locate,symbol_lookup,neighbors")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--api-url", default=DEFAULT_API_URL)
    ap.add_argument("--knowledge-system", default="code-native")
    ap.add_argument("--corpus-path", default=DEFAULT_CORPUS_PATH)
    ap.add_argument("--corpus-name", default="small-py")
    ap.add_argument("--corpus-language", default="python")
    ap.add_argument("--model", default=None, help="Override ICE_TOKENSAVE_MODEL.")
    ap.add_argument("--workers", type=int, default=3, help="Concurrency (Gemini conc <= 4).")
    ap.add_argument("--limit", type=int, default=0, help="Limit tasks (0=all) for smoke runs.")
    args = ap.parse_args()

    corpus = Corpus(
        name=args.corpus_name,
        path=args.corpus_path,
        repo_sha=args.corpus_path.split("@")[-1] if "@" in args.corpus_path else "",
        language=args.corpus_language,
        loc=0,
        file_count=0,
    )

    op_classes = [o.strip() for o in args.op_classes.split(",") if o.strip()]
    tasks = build_task_set(corpus, per_op=args.per_op, seed=args.seed, op_classes=op_classes)
    if args.limit:
        tasks = tasks[: args.limit]
    logger.info("condition=%s arm=%s tasks=%d reps=%d", args.condition, args.arm, len(tasks), args.reps)

    # Tool decls + system prompt for this condition. A fresh GeminiClient is
    # built PER TASK inside the worker (below) rather than shared across the
    # ThreadPoolExecutor — this sidesteps any question of google-genai client
    # thread-safety and avoids shared mutable state. Client construction is cheap.
    from icebench.tokensave.llm import GeminiClient

    if args.condition == "memory":
        tool_decls = [MEMORY_TOOL_DECL, *FILE_TOOL_DECLS, SUBMIT_TOOL_DECL]
        system = MEMORY_SYSTEM
    else:
        tool_decls = [*FILE_TOOL_DECLS, SUBMIT_TOOL_DECL]
        system = BASELINE_SYSTEM

    model_kwargs = {}
    if args.model:
        model_kwargs["model"] = args.model

    # Report the resolved model once (constructed and discarded — cheap).
    _probe = GeminiClient(tool_decls=tool_decls, system_instruction=system, **model_kwargs)
    resolved_model = _probe.model
    _probe.close()

    units = [(t, rep) for t in tasks for rep in range(args.reps)]
    rows = []
    started = time.time()

    def _work(unit):
        task, rep = unit
        try:
            llm = GeminiClient(tool_decls=tool_decls, system_instruction=system, **model_kwargs)
            try:
                row = _run_one(
                    task, rep, llm, corpus, args.condition,
                    args.api_url, args.knowledge_system, args.max_steps,
                )
            finally:
                llm.close()
            row["arm"] = args.arm
            return row
        except Exception as e:  # noqa: BLE001 - one task failing shouldn't kill the run
            logger.exception("task %s rep %d failed: %s", task.task_id, rep, e)
            return {
                "arm": args.arm, "condition": args.condition, "task_id": task.task_id,
                "op_class": task.op_class, "rep": rep, "error": str(e),
                "correct": False, "tokens": {"total_tokens": 0},
            }

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(_work, u): u for u in units}
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            done += 1
            if done % 5 == 0 or done == len(units):
                logger.info("  [%d/%d] done", done, len(units))

    # Stable order for reproducible output.
    rows.sort(key=lambda r: (r.get("op_class", ""), r.get("task_id", ""), r.get("rep", 0)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # Inline quick summary.
    n = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    total_tokens = sum(r.get("tokens", {}).get("total_tokens", 0) for r in rows)
    fh1 = sum(1 for r in rows if r.get("first_hop_hit_1"))
    summary = {
        "arm": args.arm,
        "condition": args.condition,
        "knowledge_system": args.knowledge_system if args.condition == "memory" else None,
        "model": resolved_model,
        "seed": args.seed,
        "corpus": args.corpus_name,
        "n_rows": n,
        "n_correct": correct,
        "correctness": correct / n if n else 0.0,
        "total_tokens": total_tokens,
        "mean_tokens_per_task": total_tokens / n if n else 0.0,
        "first_hop_hit_1_rate": fh1 / n if n else 0.0,
        "wall_s": time.time() - started,
    }
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %d rows -> %s", n, out)
    logger.info("summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
