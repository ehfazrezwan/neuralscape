"""
STRETCH arm — precise-neighbors upper bound (proxy for tree-sitter-stack-graphs).

The core arms show native code memory returns ~0 useful callers (the (b)
resolution gap is unfunded), so the memory arms LOSE on the "what-connects"
(neighbors) op class. The brief's stretch asks whether *precise* neighbors — e.g.
a wired tree-sitter-stack-graphs Python resolver — would add token savings.

Rather than integrate the stack-graphs Rust/Python toolchain (a large lift, out
of scope for a v1 proxy), this measures the **upper bound**: a `code_memory`
neighbors tool that returns the ORACLE caller set (the same tree-sitter ground
truth stack-graphs would aim to reproduce). This answers "does precise neighbors
save tokens, and how much at best?" honestly, and bounds the value of investing
in a real stack-graphs resolver. It is NOT a shipped path and NOT a claim that
stack-graphs achieves this exactly — it is the ceiling.

No NS stack is involved (the tool returns oracle data directly), so this needs no
quiesce. Token counts are load-independent regardless.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from icebench.adapters.base import Corpus
from icebench.tokensave.agent import MEMORY_SYSTEM, run_agent
from icebench.tokensave.gold import first_hop_hit, score_answer
from icebench.tokensave.tasks import build_task_set
from icebench.tokensave.tools import (
    FILE_TOOL_DECLS,
    MEMORY_TOOL_DECL,
    SUBMIT_TOOL_DECL,
    FileTools,
    ToolResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tokensave.stretch")

DEFAULT_CORPUS_PATH = "/data/ice/corpora/small-py@8a4ce842564ae94ab050062db8525196ad476c19"


class OracleNeighborsMemory:
    """A code_memory tool whose neighbors op returns the oracle caller set.

    locate/symbol_lookup are intentionally unsupported here (this arm exists only
    to bound the neighbors op); if the agent asks for them it is told so and must
    use file tools — but the task set for this arm is neighbors-only.
    """

    def __init__(self, gold_by_symbol: dict):
        self._gold = gold_by_symbol  # symbol -> [callers]

    def code_memory(self, op: str, symbol: str = "", query: str = "") -> ToolResult:
        if op != "neighbors":
            return ToolResult(
                f"code_memory[{op}]: unsupported in the precise-neighbors arm.",
                raw={"op": op}, ok=False,
            )
        callers = self._gold.get(symbol)
        if callers is None:
            return ToolResult(
                "code_memory[neighbors]: no callers returned.",
                raw={"op": op, "callers": []},
            )
        body = "code_memory[neighbors] callers:\n" + "\n".join(sorted(callers)[:200])
        return ToolResult(body, raw={"op": op, "callers": list(callers)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-op", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--corpus-path", default=DEFAULT_CORPUS_PATH)
    ap.add_argument("--corpus-name", default="small-py")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    corpus = Corpus(
        name=args.corpus_name, path=args.corpus_path,
        repo_sha=args.corpus_path.split("@")[-1], language="python", loc=0, file_count=0,
    )
    tasks = build_task_set(corpus, per_op=args.per_op, seed=args.seed, op_classes=["neighbors"])
    gold_by_symbol = {t.gold["symbol"]: t.gold["callers"] for t in tasks}

    from icebench.tokensave.llm import GeminiClient

    tool_decls = [MEMORY_TOOL_DECL, *FILE_TOOL_DECLS, SUBMIT_TOOL_DECL]
    model_kwargs = {"model": args.model} if args.model else {}
    llm = GeminiClient(tool_decls=tool_decls, system_instruction=MEMORY_SYSTEM, **model_kwargs)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _work(task):
        ft = FileTools(corpus.path)
        mem = OracleNeighborsMemory(gold_by_symbol)

        def dispatch(name, a):
            if name == "list_dir":
                return ft.list_dir(a.get("path", "."))
            if name == "grep":
                return ft.grep(a.get("pattern", ""), a.get("path", "."))
            if name == "read_file":
                return ft.read_file(a.get("path", ""), a.get("start_line", 1), a.get("end_line"))
            if name == "code_memory":
                return mem.code_memory(a.get("op", ""), a.get("symbol", ""), a.get("query", ""))
            return ToolResult(f"unknown tool {name}", ok=False)

        t0 = time.perf_counter()
        res = run_agent(llm, dispatch, task.prompt, max_steps=args.max_steps)
        latency = time.perf_counter() - t0
        sc = score_answer("neighbors", res.answer, task.gold)
        fh1 = fh5 = False
        first = res.first_memory_step()
        if res.first_tool() == "code_memory" and first is not None:
            fh1 = first_hop_hit("neighbors", first.raw, task.gold, k=1)
            fh5 = first_hop_hit("neighbors", first.raw, task.gold, k=5)
        return {
            "arm": "native+stackgraphs(upperbound)", "condition": "memory",
            "op_class": "neighbors", "task_id": task.task_id, "rep": 0,
            "correct": sc.correct, "answer": res.answer,
            "gold": task.gold, "score_detail": sc.detail,
            "tokens": res.usage.as_dict(), "n_tool_calls": res.n_tool_calls,
            "n_file_reads": res.n_file_reads, "n_memory_calls": res.n_memory_calls,
            "steps": res.steps, "stopped_reason": res.stopped_reason,
            "first_tool": res.first_tool(), "first_hop_hit_1": fh1, "first_hop_hit_5": fh5,
            "latency_s": latency,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(_work, t) for t in tasks]):
            rows.append(fut.result())
    rows.sort(key=lambda r: r["task_id"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    import statistics
    tok = [r["tokens"]["total_tokens"] for r in rows]
    print(json.dumps({
        "arm": "native+stackgraphs(upperbound)", "n": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "mean_tokens": statistics.fmean(tok), "median_tokens": statistics.median(tok),
        "first_hop_hit_1": sum(r["first_hop_hit_1"] for r in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
