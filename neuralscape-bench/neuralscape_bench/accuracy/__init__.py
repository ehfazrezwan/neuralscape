"""Memory-accuracy benchmark battery (roadmap E5).

Runs the published competitor suites against an isolated Neuralscape stack:

- **LoCoMo** (snap-research) — mem0's headline benchmark, also Honcho.
- **LongMemEval** S + M (xiaowu0162) — Zep/Graphiti paper, MemPalace, Honcho.
- **DMR** (MemGPT MSC-Self-Instruct) — used in the Zep paper.
- **BEAM** (vectorize-io agent-memory-benchmark) — Honcho scores on it.
- **ConvoMem** (Salesforce) — reproduced by MemPalace.
- **MemBench** (ACL 2025, import-myself/Membench) — reproduced by MemPalace.

Pipeline phases (each idempotent/resumable):

    fetch → ingest → answer → judge → report      (+ estimate, offline)

Datasets are downloaded at run time into ``neuralscape-bench/datasets/``
(gitignored — never committed). Raw per-question outputs (which contain
conversation text) stay gitignored under ``results/raw/``; only aggregate
results JSON + summary markdown are committed.
"""
