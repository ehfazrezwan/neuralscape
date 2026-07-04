# Neuralscape memory-accuracy battery

Generated: 2026-07-04T09:00:32.857035+00:00  
NS commit: `124d45f`

Config: k=10, reasoning_level=high, judge=gemini-2.5-flash, sample=full, seed=42.

| Suite | NS accuracy (LLM-judge) | NS retrieval R@k | Questions judged | Published competitor figures* |
|---|---|---|---|---|
| LoCoMo | *not yet run* | *not yet run* | 0 | mem0 66.9% (LLM-judge overall); mem0-graph 68.4% (LLM-judge overall); Honcho 89.9% (LLM-judge overall) |
| LongMemEval_S | *not yet run* | *not yet run* | 0 | Honcho 90.4% (LLM-judge overall); Zep ~71-79% (LLM-judge overall); MemPalace 96.6% (retrieval R@5 (not answer accuracy)) |
| LongMemEval_M | *not yet run* | *not yet run* | 0 | Zep see paper (LLM-judge overall) |
| DMR (MSC-Self-Instruct) | 57.0% | — | 500 | MemGPT 93.4% (LLM-judge); Zep 94.8% (LLM-judge) |
| BEAM | 27.8% | 0.0% (R@10, n=400) | 400 | Honcho see honcho.dev/evals (per-tier scores) |
| ConvoMem | *not yet run* | *not yet run* | 0 | MemPalace see repo results (retrieval recall) |
| MemBench | *not yet run* | *not yet run* | 0 | MemPalace see repo results (retrieval recall) |

\* Competitor numbers are **their self-reported figures** on their own (possibly different) configurations — answer model, retrieval depth, and judge model all vary between publications. They are context, not a controlled comparison.

## Per-type breakdown

### DMR (MSC-Self-Instruct)

| Question type | n | judged | accuracy |
|---|---|---|---|
| dmr | 500 | 500 | 57.0% |

### BEAM

| Question type | n | judged | accuracy |
|---|---|---|---|
| abstention | 40 | 40 | 90.0% |
| contradiction_resolution | 40 | 40 | 0.0% |
| event_ordering | 40 | 40 | 0.0% |
| information_extraction | 40 | 40 | 20.0% |
| instruction_following | 40 | 40 | 45.0% |
| knowledge_update | 40 | 40 | 37.5% |
| multi_session_reasoning | 40 | 40 | 7.5% |
| preference_following | 40 | 40 | 62.5% |
| summarization | 40 | 40 | 2.5% |
| temporal_reasoning | 40 | 40 | 12.5% |
