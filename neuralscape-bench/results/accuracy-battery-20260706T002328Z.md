# Neuralscape memory-accuracy battery

Generated: 2026-07-06T00:23:28.680288+00:00  
NS commit: `dcc0dd7`

Config: k=10, reasoning_level=high, judge=gemini-2.5-flash, sample=full, seed=42.

| Suite | NS accuracy (LLM-judge) | NS retrieval R@k | Questions judged | Published competitor figures* |
|---|---|---|---|---|
| LoCoMo | 50.0% | 70.5% (R@10, n=1982) | 1986 | mem0 66.9% (LLM-judge overall); mem0-graph 68.4% (LLM-judge overall); Honcho 89.9% (LLM-judge overall) |
| LongMemEval_S | *not yet run* | *not yet run* | 0 | Honcho 90.4% (LLM-judge overall); Zep ~71-79% (LLM-judge overall); MemPalace 96.6% (retrieval R@5 (not answer accuracy)) |
| LongMemEval_M | *not yet run* | *not yet run* | 0 | Zep see paper (LLM-judge overall) |
| DMR (MSC-Self-Instruct) | 62.8% | — | 500 | MemGPT 93.4% (LLM-judge); Zep 94.8% (LLM-judge) |
| BEAM | 27.8% | — | 400 | Honcho see honcho.dev/evals (per-tier scores) |
| ConvoMem | 14.4% | 19.1% (R@10, n=1047) | 1047 | MemPalace see repo results (retrieval recall) |
| MemBench | *not yet run* | *not yet run* | 0 | MemPalace see repo results (retrieval recall) |

\* Competitor numbers are **their self-reported figures** on their own (possibly different) configurations — answer model, retrieval depth, and judge model all vary between publications. They are context, not a controlled comparison.

## Per-type breakdown

### LoCoMo

| Question type | n | judged | accuracy |
|---|---|---|---|
| 1-multi-hop | 282 | 282 | 34.8% |
| 2-temporal | 321 | 321 | 30.8% |
| 3-open-domain | 96 | 96 | 25.0% |
| 4-single-hop | 841 | 841 | 42.2% |
| 5-adversarial | 446 | 446 | 93.5% |

### DMR (MSC-Self-Instruct)

| Question type | n | judged | accuracy |
|---|---|---|---|
| dmr | 500 | 500 | 62.8% |

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

### ConvoMem

| Question type | n | judged | accuracy |
|---|---|---|---|
| abstention_evidence | 200 | 200 | 9.0% |
| assistant_facts_evidence | 200 | 200 | 5.0% |
| changing_evidence | 199 | 199 | 1.0% |
| implicit_connection_evidence | 74 | 74 | 58.1% |
| preference_evidence | 182 | 182 | 41.2% |
| user_evidence | 192 | 192 | 1.6% |
