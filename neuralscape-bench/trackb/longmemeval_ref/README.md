# Track B: LongMemEval Reference Harness

**Track B** runs Neuralscape through the published LongMemEval reference evaluation methodology, producing NS accuracy numbers directly comparable to the LongMemEval figures cited by competitors (Zep, MemPalace, Honcho) — kept clearly separate from Track A (NSBench's own methodology).

## What This Is

This harness faithfully reproduces the **published LongMemEval evaluation protocol**:

1. **Ingest**: Store each question's haystack sessions into NS, preserving temporal context (dated sessions).
2. **Retrieve**: For each question, retrieve top-k memories from NS.
3. **Answer**: Generate an answer using NS `/v1/ask` endpoint (which mirrors the LME reference's retrieval + answer-generation pipeline).
4. **Grade**: Judge correctness using LongMemEval's QA-correctness metric with the standard judge (`gemini-2.5-flash`, temp 0).

### Faithfulness Statement

**What matches the published LME reference harness:**
- Dataset: LongMemEval_S (cleaned release, 500 questions, 6 types)
- Ingest protocol: dated sessions, one NS user per question/haystack
- Answer protocol: retrieve top-k → generate answer (via NS `/v1/ask`)
- Grading: LLM judge with abstention handling, per-question_type breakdown
- Metrics: QA correctness (overall + per-type)

**What's adapted for NS:**
- Storage backend: NS REST API instead of the reference's in-memory store
- Answer generation: NS `/v1/ask` (which internally does retrieval + LLM answer generation) instead of direct LLM calls, making this comparable to how Zep/MemPalace/Honcho run LME
- Retrieval recall@k is computed as a **diagnostic** (NOT the headline number) — LME's R@k definition, clearly tagged in reports

**Why comparable:**
The published LME benchmark tests whether a memory system can correctly answer questions from stored conversational context. Competitors (Zep, MemPalace, Honcho) cite their LME scores as evidence of memory accuracy. This harness runs NS through the same evaluation, producing directly comparable QA-correctness numbers.

### Diagnostic Caveat: Retrieval Recall@k

This harness reports **retrieval recall@k** as a diagnostic metric (LME's R@k definition: whether any of the top-k retrieved memories came from the gold evidence sessions). **This is NOT the headline number.** The headline is **QA correctness** — whether the final answer is correct per the LLM judge. R@k is useful for debugging retrieval issues but is not how LME scores memory systems in the published benchmark.

## Requirements

- Python 3.12+
- Running Neuralscape bench stack (orchestrator-managed, or `docker compose -f neuralscape-bench/docker-compose.bench.yml up`)
- `GOOGLE_API_KEY` for the LLM judge

## Usage

### Against the orchestrator bench stack

```bash
cd neuralscape-bench/trackb/longmemeval_ref
python run.py \
  --target http://localhost:8398 \
  --token $BENCH_TOKEN \
  --judge-key $GOOGLE_API_KEY \
  --sample 50 \
  --seed 42 \
  --k 10
```

### Phases

By default, all phases run in sequence:
1. `ingest` — store haystacks into NS
2. `answer` — retrieve + generate answers
3. `judge` — grade answers
4. `report` — aggregate results

To run specific phases:
```bash
python run.py --target http://localhost:8398 --phases ingest answer
```

### Output

- `results/raw/answers-trackb-lme.jsonl` — per-question answers (gitignored, contains conversation text)
- `results/raw/judged-trackb-lme.jsonl` — per-question verdicts
- `results/trackb-lme-<timestamp>.json` — aggregate results (overall + per-type accuracy, tagged Track B)
- `results/trackb-lme-<timestamp>.md` — markdown summary

All runs are **resumable**: already-processed items are skipped.

## Methodology Tags

All result files are tagged:
```json
{
  "harness": "longmemeval-ref (Track B)",
  "backbone": "neuralscape",
  "judge": "gemini-2.5-flash",
  "embedder": "<NS's configured embedder>",
  "dataset": "LongMemEval_S (xiaowu0162/longmemeval-cleaned)"
}
```

These tags make it clear that Track B results use the LME reference methodology and are directly comparable to competitor LME scores.

## Directory Structure

```
trackb/longmemeval_ref/
├── README.md              # This file
├── run.py                 # CLI entrypoint
├── harness.py             # Core harness logic
├── loader.py              # Dataset loading
├── ingest.py              # Ingest protocol (dated sessions)
├── answer.py              # Answer protocol (retrieve + generate)
├── grade.py               # Grading protocol (LLM judge)
├── report.py              # Result aggregation
├── conftest.py            # Pytest imports (if needed)
└── tests/
    ├── test_loader.py     # Dataset parsing tests
    ├── test_ingest.py     # Ingest mapping tests (mocked)
    ├── test_answer.py     # Answer protocol tests (mocked)
    ├── test_grade.py      # Judge parsing tests
    └── test_report.py     # Aggregation tests
```
