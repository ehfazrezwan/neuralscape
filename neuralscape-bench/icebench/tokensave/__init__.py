"""
Token-Savings Navigation Benchmark (ICE v2 — the REAL North Star).

The code memory exists primarily to *reduce the tokens an LLM burns reading files
to find things* — "let the LLM know where something is without reading a bunch of
files first." Benchmark recall (Track-Q) is a proxy; the metric this package
measures directly is:

    tokens_saved = baseline_tokens - with_memory_tokens

per navigation task, plus the **first-hop hit rate** (did the code memory return
the right location on the first query, so the agent never had to read a file?).

Two conditions, one Gemini ReAct agent, same corpus, same oracle-backed gold:

  * Baseline (no code memory): the agent has ONLY file tools (list/grep/read) over
    the corpus checkout. It reads files until it can answer — this is the
    "reads a bunch of files" cost.
  * With code memory: the agent has the SAME file tools PLUS a `code_memory` tool
    routed to NS `/v1/code-graph/{locate,query,neighbors}`. It is told to prefer
    the memory; a first-hop hit means the memory answered without any file read.

Arms vary the *server-side* memory backend (the harness is arm-agnostic — the arm
label just tags the output, exactly like the nl_locate A/B):

  1. native            — the shipped locate + symbol lookup (embeddings off).
  2. native+embedder   — the nl_locate A/B winner (local CodeRankEmbed + BM25).
  3. native+stackgraphs (STRETCH) — precise callers/callees for neighbors.

This is an honest v1 proxy: small corpus (pallets/click @ 8a4ce84), synthetic
oracle tasks, a single small LLM. See ICE_V2_TOKEN_SAVINGS.md for caveats.
"""
