# mem0 LoCoMo Evaluation (Track B)

**Track B** evaluation: Neuralscape evaluated using mem0's published LoCoMo methodology. Results are directly comparable to mem0's published numbers on their own terms.

## What is LoCoMo?

LoCoMo (Long Conversation Memory, ACL 2024) is mem0's headline benchmark for long multi-session conversational memory. The dataset contains:

- 10 very long two-person conversations (~19-35 sessions each)
- ~200 QA pairs per conversation across 5 categories:
  1. **multi-hop** - requires connecting multiple pieces of information
  2. **temporal** - requires understanding time relationships
  3. **open-domain** - general knowledge questions
  4. **single-hop** - direct fact retrieval
  5. **adversarial** - unanswerable (correct behavior = abstain)

## Methodology Faithfulness

This harness reproduces mem0's LoCoMo evaluation as faithfully as possible:

### What Matches mem0's Published Eval

1. **Dataset**: Same locomo10.json from snap-research/locomo
2. **Ingestion**: Per-speaker turns stored in memory layer, session-by-session
3. **Answer Protocol**: 
   - Retrieve top-k memories for each question
   - Generate answer using retrieved context
   - NS `/v1/ask` endpoint mirrors mem0's answer generation prompt
4. **Judge**: LLM judge (gemini-3.1-flash-lite @ temp=0) evaluating correctness
5. **Metrics**: Overall + category-wise accuracy, retrieval R@k
6. **Abstention Handling**: Category 5 (adversarial) requires "not mentioned" / abstention

### Adaptations

1. **Backend**: Neuralscape REST API instead of mem0's Python SDK
2. **Answer Generation**: NS `/v1/ask` (C3 reasoning-tiered QA) instead of mem0's direct LLM call
   - We use `reasoning_level=high` by default (mirrors mem0's multi-search behavior)
   - The prompt structure is equivalent to mem0's answer prompt
3. **Judge Model**: gemini-3.1-flash-lite (locked standard) instead of mem0's original judge model
   - Same prompt structure as mem0's correctness judge
   - Same evaluation criteria

### Why Comparable

The core evaluation loop is identical:
1. Ingest conversations -> memory layer
2. For each question: retrieve top-k -> generate answer
3. Judge: correct/incorrect via LLM
4. Report: category-wise accuracy

The backend swap (mem0 -> NS) is the point of the benchmark. The methodology (dataset, protocol, judge) remains constant.

## Running the Harness

### Prerequisites

- Neuralscape API running (orchestrator bench stack or local)
- `GOOGLE_API_KEY` environment variable (for judge)
- Dataset: `locomo10.json` (auto-located at `../../datasets/locomo/locomo10.json`)

### Full Run (All Phases)

```bash
cd /data/nsbench/neuralscape/neuralscape-bench

# Against orchestrator bench stack
python -m trackb.mem0_locomo.run \
  --target http://localhost:8398 \
  --token $BENCH_TOKEN

# Local dev (no auth)
python -m trackb.mem0_locomo.run \
  --target http://localhost:8199
```

### Phase-by-Phase

```bash
# 1. Ingest only
python -m trackb.mem0_locomo.run \
  --target http://localhost:8398 \
  --phases ingest

# 2. Answer only (requires prior ingest)
python -m trackb.mem0_locomo.run \
  --target http://localhost:8398 \
  --phases answer

# 3. Judge only (requires answers.jsonl)
python -m trackb.mem0_locomo.run \
  --target http://localhost:8398 \
  --phases judge

# 4. Report only (requires judged.jsonl)
python -m trackb.mem0_locomo.run \
  --target http://localhost:8398 \
  --phases report
```

### Sampling (Mini-Run)

```bash
# Run on 50 QA items (stratified by category)
python -m trackb.mem0_locomo.run \
  --target http://localhost:8398 \
  --sample 50 \
  --seed 42
```

### Options

```
--target URL           NS API base URL (required)
--token TOKEN          Bearer token (optional for local dev)
--dataset PATH         Path to locomo10.json (default: auto-locate)
--sample N             Use stratified sample of N QA items
--seed SEED            Random seed for sampling (default: 42)
--k K                  Retrieval top-k (default: 10)
--reasoning-level LVL  NS reasoning level: minimal/low/medium/high (default: high)
--concurrency-ingest N Ingest concurrency (default: 2)
--concurrency-answer N Answer concurrency (default: 5)
--concurrency-judge N  Judge concurrency (default: 5)
--phases PHASE...      Phases to run: ingest answer judge report (default: all)
--output-dir DIR       Output directory (default: ./trackb_results)
```

## Output

Results are written to `--output-dir` (default: `./trackb_results/`):

- `ingest_manifest.json` - Ingestion state (idempotent/resumable)
- `answers.jsonl` - Raw per-QA records (question, answer, retrieval metrics)
- `judged.jsonl` - Judged records (answers + judgment field)
- `report.json` - Full report (metrics, config, metadata)
- `report.md` - Human-readable summary

### Report Format

```json
{
  "harness": "mem0-locomo (Track B)",
  "backbone": "neuralscape",
  "judge": "gemini-3.1-flash-lite",
  "embedder": "text-embedding-004",
  "config": {
    "k": 10,
    "reasoning_level": "high"
  },
  "timestamp": "2026-07-06T...",
  "metrics": {
    "overall_accuracy": 0.78,
    "category_accuracy": {
      "1-multi-hop": 0.72,
      "2-temporal": 0.80,
      "3-open-domain": 0.65,
      "4-single-hop": 0.88,
      "5-adversarial": 0.75
    },
    "retrieval_r_at_k": 0.82,
    "abstention_accuracy": 0.75,
    "total": 200,
    "correct": 156,
    "incorrect": 40,
    "errors": 4
  }
}
```

## Resumability & Idempotence

All phases are resumable:

1. **Ingest**: `ingest_manifest.json` tracks completed sessions (never re-store)
2. **Answer**: `answers.jsonl` is appended; already-answered QA IDs are skipped
3. **Judge**: Judges only records in `answers.jsonl` (outputs to `judged.jsonl`)
4. **Report**: Generates from `judged.jsonl` (can re-run anytime)

If interrupted, re-run the same command to resume.

## Testing

Unit tests (mocked NS client) are in `tests/`. Run:

```bash
cd /data/nsbench/neuralscape/neuralscape-bench
uv run pytest trackb/mem0_locomo -v
```

Tests cover:
- Dataset loading & parsing
- Ingest message mapping
- Answer context assembly
- Judge response parsing
- Category aggregation math
- Report generation

All tests use `httpx.MockTransport` - no live NS stack required.

## Track B vs Track A

- **Track A** (NSBench): Our own methodology, our harness, multiple datasets
- **Track B** (this): Competitors' published methodology, our backend

Keep results clearly separated. Track B numbers are comparable to mem0's published LoCoMo figures. Track A numbers are our own broader accuracy battery.

## Citation

LoCoMo dataset from:
- Repository: https://github.com/snap-research/locomo
- Paper: "LoCoMo: Long Conversation Memory" (ACL 2024)
- See LICENSE.txt in upstream repo before redistribution
