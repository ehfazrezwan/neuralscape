# mem0 Track A Control Harness

A harness that runs the vendored mem0 library as the memory layer under the NSBench accuracy battery, with the backbone, judge and embedder set to the same models the Track A runs use.

**Status:** no result from this harness is committed to the repository. The runs it was built for are paused (see the benchmark README). Treat everything below as a description of what the code does, not of any outcome.

## What This Is

A self-contained harness that runs mem0 (the in-repo `mem0/` subtree) as the memory layer under the NSBench accuracy battery. It is meant to vary the memory layer (NS vs mem0) while holding the settings below constant; the caveats at the end list where the two sides still differ:

- **Dataset**: Same (LoCoMo, BEAM, ConvoMem)
- **Backbone**: `gemini-3.1-flash-lite`
- **Embedder**: `gemini-embedding-001`
- **Judge**: `gemini-3.1-flash-lite` (temp 0)
- **Vector store**: Qdrant local/on-disk (isolated)

It is the mem0 control for Track A.

## Configuration

### Locked Models (Controlled Comparison)

All model ids live in `config.py` as the single source of truth; `report.py`
reads those same constants so the reported config always matches what the
mem0 Memory / judge were actually configured with.

```python
# config.py (single source of truth)
BACKBONE_MODEL  = "gemini-3.1-flash-lite"   # mem0 LLM (extraction + generation)
EMBEDDER_MODEL  = "gemini-embedding-001"    # mem0 embedder
JUDGE_MODEL     = "gemini-3.1-flash-lite"   # NSBench GeminiJudge (temp 0)
ANSWER_MODEL    = BACKBONE_MODEL            # answer generation == backbone
```

The vendored mem0 `GeminiLLM` passes `config.model` straight through to the
`google-genai` client (see `mem0/mem0/llms/gemini.py`), so the **same**
`gemini-3.1-flash-lite` model id can be used verbatim, with no model-string
translation. That lets both sides name the same backbone.

### mem0 Memory Configuration

Each conversation is ingested into a separate `mem0.Memory` instance with a unique `user_id` (format: `{suite}-{conv_id}`). The config dict (built by `config.py`):

```python
{
  "llm": {
    "provider": "gemini",              # mem0 provider registry key
    "config": {
      "model": "gemini-3.1-flash-lite",
      "temperature": 0.0,
      "api_key": "<GOOGLE_API_KEY>"
    }
  },
  "embedder": {
    "provider": "gemini",             # GoogleGenAIEmbedding
    "config": {
      "model": "gemini-embedding-001",
      "embedding_dims": 768,
      "api_key": "<GOOGLE_API_KEY>"
    }
  },
  "vector_store": {
    "provider": "qdrant",
    "config": {
      "collection_name": "mem0_tracka",
      "embedding_model_dims": 768,
      "path": ".mem0_tracka_qdrant/",
      "on_disk": true
    }
  },
  "version": "v1.1"
}
```

## Answer Generation (Faithfulness Caveat)

mem0-the-library has **no `/ask` endpoint** — it only provides `.add()` and `.search()`. We generate answers with:

1. **Retrieval**: `memory.search(question, top_k=k, filters={"user_id": user_id})` → top-k memories (mem0 returns `{"results": [...]}`; the entity id MUST go inside `filters` — top-level `user_id`/`limit` kwargs are rejected by the library)
2. **Answer prompt**: Mirrors NSBench's `/ask` logic (retrieved memories + question → "provide a concise factual answer")
3. **Generation**: Direct Gemini call with the same `gemini-3.1-flash-lite` backbone (temp 0)

**Faithfulness**: The answer prompt is designed to mirror NSBench's reasoning approach, but it's **not byte-identical** — retrieval→context assembly parity is approximate. Keep this in mind for any mem0-vs-NS comparison.

## Usage

### Prerequisites

1. **NSBench deps** (suite loaders + judge):
   ```bash
   cd neuralscape-bench && uv sync
   ```

2. **mem0 installed (editable)**: mem0 must be **installed as a package**, not merely on `PYTHONPATH`. `mem0/__init__.py` runs `importlib.metadata.version("mem0ai")` at import time, so a bare `PYTHONPATH=.../mem0` fails with `PackageNotFoundError`. Install it editable from the in-repo subtree, plus its Gemini extra (`google-genai`):
   ```bash
   # from the repo root (the mem0/ subtree is the mem0ai package)
   uv pip install -e mem0/
   uv pip install google-genai   # required by mem0's GeminiLLM / embedder
   ```
   Verify: `python -c "from mem0.memory.main import Memory; print('ok')"`. If the import still fails, that is an orchestrator env-wiring detail — the harness imports mem0 lazily and unit tests run with a mock, so `pytest` stays green without it.

3. **GOOGLE_API_KEY** in the environment (Gemini inference + embeddings).

### Fetch Datasets (Offline, Free)

```bash
cd neuralscape-bench
uv run python -m neuralscape_bench.accuracy.run --suite locomo --phase fetch
uv run python -m neuralscape_bench.accuracy.run --suite beam --phase fetch
uv run python -m neuralscape_bench.accuracy.run --suite convomem --phase fetch
```

### Run Full Pipeline (Paid Phases)

```bash
cd neuralscape-bench

# LoCoMo (full)
uv run python -m trackb.mem0_tracka.run \
  --suite locomo \
  --phase ingest --phase answer --phase judge --phase report \
  --k 10

# BEAM 100k (sample for dev)
uv run python -m trackb.mem0_tracka.run \
  --suite beam \
  --phase ingest --phase answer --phase judge --phase report \
  --sample 50 --seed 42 \
  --k 10

# ConvoMem
uv run python -m trackb.mem0_tracka.run \
  --suite convomem \
  --phase ingest --phase answer --phase judge --phase report \
  --k 10
```

### Phases

- **ingest**: Load suite → ingest conversations into mem0 Memory instances
- **answer**: Answer QA items using mem0 retrieval + LLM generation
- **judge**: Judge answers with GeminiJudge (same as NSBench)
- **report**: Generate JSON + markdown reports

### Outputs

- **Raw**: `trackb/mem0_tracka/raw/answers-{suite}.jsonl`, `judged-{suite}.jsonl`
- **Results**: `results/mem0-tracka-{suite}-<timestamp>.json` (aggregate metrics)
- **Markdown**: `results/mem0-tracka-{suite}-<timestamp>.md` (human report)

## Metrics

### Headline: LLM-Judged QA Accuracy

End-to-end correctness: retrieval → answer generation → judge verdict (same judge as NS). Reported overall + per-category breakdown.

### Diagnostic: R@k

Retrieval recall at k (session-level attribution, lexical match over distilled memories). Secondary metric; accuracy is the headline.

## Orchestrator Integration

A run needs the following (no run is currently scheduled):

1. **mem0 installed editable**: `uv pip install -e mem0/` (a bare `PYTHONPATH` does NOT work — `mem0/__init__.py` needs the `mem0ai` package metadata) plus `uv pip install google-genai`.
2. **Isolated vector store**: Pass `--vector-store-path` to avoid colliding with any production Qdrant.
3. **GOOGLE_API_KEY**: Set in the environment (inference + embeddings).
4. **Dataset pre-fetch**: Run fetch phases offline first (free, no API calls).
5. **Run isolation**: Pass `--run-label <id>` to keep distinct runs' raw files separate; the answer phase also truncates its own answers file each run so re-runs never blend stale records into the report.

## Testing

All tests use mocked mem0 (no live imports required):

```bash
cd neuralscape-bench
uv run pytest trackb/mem0_tracka/tests/ -v
```

In CI (`bench-ci`), these tests run from the installed wheel via `--pyargs trackb.mem0_tracka.tests`, with no network, as part of `scripts/ci_run_suite.sh`. The skip gate applies to them like every other bench suite. The package is shipped in the wheel (`trackb` is in `pyproject.toml`).

Coverage:
- Config dict generation (`test_config.py`)
- Conversation→messages mapping (`test_ingest.py`)
- Answer prompt rendering + retrieval→generation (`test_answer.py`)
- Report aggregation + markdown rendering (`test_report.py`)
- CLI/run wiring (`test_run.py`)
- Full pipeline integration with mocks (`test_integration.py`)

## Files

All files are self-contained under `trackb/mem0_tracka/`:

```
trackb/mem0_tracka/
├── __init__.py
├── __main__.py          # python -m entrypoint
├── config.py            # Locked models + mem0 config dict builder
├── ingest.py            # Conversation→mem0.add mapping
├── answer.py            # mem0.search + LLM answer generation
├── report.py            # Result aggregation + markdown
├── run.py               # CLI entrypoint
├── raw/                 # JSONL outputs (gitignored)
├── tests/
│   ├── __init__.py
│   ├── conftest.py      # Fixtures (mock mem0, sample data)
│   ├── test_config.py
│   ├── test_ingest.py
│   ├── test_answer.py
│   ├── test_report.py
│   ├── test_run.py
│   └── test_integration.py
└── README.md            # This file
```

No edits to shared files (`neuralscape-bench/pyproject.toml`, `neuralscape_bench/`, `mem0/`, service code). Imports from NSBench are read-only reuse (suite loaders, judge, metrics).

## Caveats & Faithfulness

1. **Answer prompt parity**: Designed to mirror NSBench `/ask` logic, but not byte-identical. Retrieval→context parity is approximate.
2. **mem0 version**: Vendored subtree at v2.0.2 (the NS fork restoring the OSS graph layer).
3. **Configured models**: this harness configures `gemini-3.1-flash-lite` as backbone, `gemini-embedding-001` as embedder and `gemini-3.1-flash-lite` as judge (temp 0). That the Neuralscape side uses the same models depends on how that side is configured; check it per run.
4. **Comparison scope**: This is mem0-the-library under OUR harness, not mem0's own eval harness (which uses different answer models/judges). It says nothing about how mem0 performs under its own published method.

## License

Apache 2.0 (same as the parent Neuralscape repo).
