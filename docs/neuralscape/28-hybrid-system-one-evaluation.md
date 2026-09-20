# Hybrid System One evaluation

Evaluated 2026-09-21. Branch `codex/hybrid-system-one`, based on local `dev`
`c4b83ff8af2ad18cfc2bd81b4d4eec2ff857de39`. This is an opt-in implementation
and small live pilot, **not a production rollout or a claim of production accuracy**.
All new flags default to false. A Typesafe key alone changes no behavior.

## Recommendation

Proceed with a sandbox trial of Jev for graph entity/edge decisions. Retain the
original model for uncertainty, conflicts, extraction and synthesis. Consider
Jev policy reranking separately as a quality feature, not a saving over the
current deterministic ranking. Bucket-aware category mapping improved after
redesign and merits sandbox testing with partial fallback. Do not promote
Needle: its initial pilot added work without saving calls.

The strongest finding is selective replacement, not replacing every LLM.
Component savings were substantial; the graph write path saved only about 12%
of measured generation cost, and tail latency worsened. Embeddings, extraction,
database operations and fallback still dominate parts of the system.

## What changed, including Mem0 and Graphiti

| Existing capability | Implementation in this branch | Remaining boundary |
|---|---|---|
| Graphiti entity resolution | Batched Jev choices over existing candidate IDs plus NONE | Low confidence, provider failure, oversized inputs use original resolver |
| Graphiti edge deduplication | Jev compares candidate facts: duplicate/distinct/conflict/uncertain | Conflict/uncertainty and invalidation-only duplicates use original temporal resolver |
| Mem0 graph integration | NS `MemoryGraph` attaches the shared decision adapter to Graphiti's client | No pretend generative provider; extraction client remains intact |
| Imported OKF type labels | Exact/alias mapping first; scoped core buckets, shared definitions, bounded Jev batches, per-item fallback | Improved pilot; still opt-in pending larger quality evaluation |
| Combined recall ranking | Optional Jev Noul scores on already-authorized candidates, stable reorder only | Adds API work to current fusion; does not increase candidate recall |
| Single literal fact extraction | Research pilot only; no runtime lane ships in this PR | Pilot accepted 0/12 and the native path lacked process isolation |
| Mem0 general fact extraction / conversation extraction | Retained | Produces novel text, roles, dates, sensitivity/provenance fields |
| Graphiti entity/relationship extraction, attributes, summaries | Retained | Open-ended entities, relations and text are not fixed choices |
| Graphiti temporal conflict resolution | Retained behind Jev abstention | Invalidation is a consequential mutation; confidence alone is not sufficient |
| Conversation compiler, lint/repair, flush, answer generation | Retained | Compilation, repair and answers require generation |
| Dreaming consolidation, reflection, cards, librarian, strategy synthesis | Retained | Rewrites/merges, temporal reframing, novel cluster labels and prose; destructive actions need existing gates |
| Vector and graph embeddings | Retained | Jev is not an embedding replacement |
| Existing RRF, hashing, filtering, exact-match dedup, adapter rules | Retained | Deterministic work should not acquire model cost |

Important fork detail: this repository's Mem0 graph adapter is NS-maintained;
it is not just the current upstream Mem0 API. Both Graphiti maintenance call
sites and the actual Mem0-to-Graphiti configuration path were modified and tested.
The current Mem0 path is already ADD-only rather than an assumed older two-call
extraction/update pipeline. There is no second obsolete update call to remove.

Implementation map:

- `graphiti/graphiti_core/llm_client/jev_client.py`: one reusable HTTP and validation layer.
- `graphiti/graphiti_core/utils/maintenance/{node,edge}_operations.py`: structured decision seams.
- `mem0/mem0/{graphs/configs,memory/graphiti_memory}.py`: config and graph wiring.
- `neuralscape-service/hybrid_inference.py`: service classification/ranking and experimental literal extraction.
- `neuralscape-service/{config,memory/search,memory/write,ingest/okf_bundle}.py`: independently controlled capabilities.
- `neuralscape-service/scripts/benchmark_hybrid*.py`: repeatable private evaluation harnesses.

## Live measurements

These are USD **standard token list-price estimates**, not invoices, total
infrastructure costs or projected production bills. Gemini 3.1 Flash Lite was
priced at $0.25/M input and $1.50/M output, including thinking output; Jev 1.13.0
at $0.042/M input and zero output charge. Fallback calls are included. Model usage
was read from actual API responses. No cache discount is assumed. See
[Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) and
[Typesafe models](https://docs.typesafe.ai/models).

### Component pilot

Six entity cases, six edge cases, four extraction cases, three repeats each.
Same frozen fixtures and original production prompt builders for the baseline;
baseline ran before substitutions. macOS 26.6.2 arm64, Python 3.13.14. Calls were
sequential; network latency is included. Repeated fixtures are not independent
examples. Extraction scoring checks category plus a required fragment, not full
semantic correctness.

| Arm | Checks | Median | p95 | Measured generation cost |
|---|---:|---:|---:|---:|
| Original Gemini | 48/48 | 872 ms | 986 ms | $0.011656 |
| Jev + original fallback | 48/48 | 366 ms | 1,159 ms | $0.003803 |

About 58% lower median and 67% lower cost **on this workload**, with a worse
p95. Entity-resolution cost was $0.005390 → $0.000387; edge-resolution cost
$0.003803 → $0.000971. Extraction was not substituted. Hybrid made 36 Jev and
15 Gemini calls: 12 extraction calls and three conflict fallbacks. More provider
calls can still cost less, but serial fallback can worsen tail latency.

A separately authored challenge set added four multi-candidate/homonym entity
cases and eight negation, numeric, version and qualifier edge cases, repeated
three times. Baseline: 33/36, median 810 ms, p95 904 ms, $0.008730. Hybrid:
36/36, median 352 ms, p95 1,372 ms, $0.003297, 12 fallbacks. The baseline misses
were one qualifier/subset case repeated three times. Hybrid also delegated that
case to Gemini, so its different result is **not evidence of superior Jev
reasoning**. The strict hand label treats losing a qualifier as non-equivalence;
this policy needs agreement against a larger human-reviewed corpus.

### Actual Neo4j + Qdrant paths

The graph benchmark calls `MemoryService.store_raw(add_to_graph=True)` through
the real Mem0 adapter into isolated Neo4j and Qdrant. It writes a fact, a
paraphrase, then a dated employer change, and inspects persisted Cypher results.
It does not infer graph success from a successful vector write.

All **21 checks passed** for baseline, hybrid, and missing-Jev-key fallback:

- Idempotent duplicate write returns the same memory ID without new model calls.
- Vector-only and combined recall, plus empty-namespace isolation.
- Persisted entities and relationships, one reused Ada entity, invalidated old employer, current new employer.
- Exact expected graph group, graph-only recall and isolation.
- Populated competing tenant and project using the same entity name; each can retrieve its own employer without the other tenant's employer.
- Own graph excludes foreign entities; vector/combined search excludes foreign tenant and project facts.
- Relationship endpoints remain in the same group; foreign writes cannot invalidate the original current relationship.

| Expanded warm-store run | Median write | Slowest of 3 writes | Generation cost for 3 writes |
|---|---:|---:|---:|
| Baseline | 9.019 s | 11.790 s | $0.003801 |
| Jev hybrid | 8.562 s | 14.804 s | $0.003362 |
| Missing-key fallback | 7.797 s | 11.397 s | $0.003724 |

Cost fell 11.5%, median fell 5.1%, but the slowest write increased 25.6%.
Only three writes per arm: do not interpret this as a stable latency gain or
meaningful p95 estimate. The outage arm being faster than baseline illustrates
noise. Earlier cold baseline and warm reruns are retained, not silently dropped.
The first cold baseline includes index setup and is **not** an apples-to-apples
speedup denominator. Synthetic competing-namespace writes are additional
validation; their events are recorded separately, outside this three-write cost.

This path includes embedding latency but **does not meter embedding cost**.
It excludes HTTP/MCP handling, Redis queue wait, worker scheduling, ingestion
parsing and generated answers. It is not an entire-product cost baseline.

### Policy reranking

Six fixed candidate sets, three repeats; authority/version, historical state,
refunds, environment, and named-person relevance. Arms alternate order between
repeats. No LLM judge generated the labels.

| Reranker | Top-1 checks | Median | p95 | API token cost for 18 sets |
|---|---:|---:|---:|---:|
| Jev | 18/18 | 316 ms | 879 ms | $0.000434 |
| Gemini JSON permutation | 18/18 | 840 ms | 894 ms | $0.001008 |
| Local MiniLM L6, CPU, 2 threads | 12/18 | 14.6 ms | 19.6 ms | $0 |

MiniLM missed both official-version-policy cases on all repeats. It is a query/
document relevance model, not an instruction-following policy model. Jev cost
57% less than the **hypothetical Gemini reranker**, not less than today's RRF.
The input-order comparator is arbitrary fixture order, **not measured current
search quality**. No search-recall gain is established here.

Local MiniLM initialization was 156 ms, excluding download; ONNX file was
90,992,115 bytes. At an illustrative $0.05/CPU-hour, occupied inference time for
18 sets is ~$0.0000036. This excludes idle capacity, initialization, memory,
operations and hardware purchase; local is not literally free. Model:
[Xenova/ms-marco-MiniLM-L-6-v2](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2).

### Category buckets: initial failure, then targeted redesign

Imported type mapping used six short labels against the runtime taxonomy,
including registered adapter categories, repeated three times. Both arms matched
12/18 hand labels. Labels such as “active blocker” are ambiguous across core and
adapter categories, and the existing baseline prompt supplies category names
without their descriptions. Jev abstained for every batch: hybrid median
1,427 ms vs baseline 831 ms; cost $0.001087 vs $0.000400. This was the initial
implementation, not the final bucket-aware design.

The user's explicit bucket requirement led to a tighter implementation:

- Default to `CORE_MEMORY_CATEGORIES` (all 13), not the process-wide registry of
  every installed adapter. Explicit adapter taxonomies can be passed separately.
- Reuse canonical bucket descriptions in shared request state. Each independent
  Choice enumerates the complete applicable bucket set plus UNCERTAIN, points
  to that shared state, and supplies its own label explicitly.
- Keep known/embedded categories and deterministic aliases free of model calls.
- Deduplicate labels within the request and batch at most 64 independent choices
  per request. No tenant-content cache and no serial five-family decision tree.
- Retain confident answers and send only unresolved labels to the original
  mapper. Reject malformed responses. The fallback cannot overwrite accepted
  labels or escape the selected category set.

This follows [Typesafe's workflow guidance](https://docs.typesafe.ai/concepts/how-to-build-with-system-one)
and [Choice guidance](https://docs.typesafe.ai/primitives/choice): structured
state, bounded choices, independent parallel questions, deterministic control
flow and uncertainty routing. The 0.9 threshold was not lowered to improve
benchmark acceptance.

On the original six development labels, final shared-state routing accepted
5/6 directly per repeat; the remaining label used Gemini. Combined accuracy
was 18/18, cost $0.000353 vs $0.000397 baseline, but median was 1,328 ms vs
861 ms because of serial fallback. Moving definitions into shared state reduced
measured input tokens for that hybrid test from 8,988 to 5,601 versus the earlier
scoped design that repeated them per question. These are development examples,
not independent validation after policy refinement.

Thirteen new labels (one per core bucket), three repeats, were then tested with
an additional **Gemini control using the same core descriptions and policy**:

| Held-out label arm | Correct labels | Median batch | Slowest of 3 batches | Cost for 39 labels |
|---|---:|---:|---:|---:|
| Existing broad-taxonomy Gemini mapper | 32/39 | 961 ms | 1,203 ms | $0.000782 |
| Bucket-aware Jev | 39/39 | 383 ms | 1,113 ms | $0.000382 |
| Bucket-aware Gemini control | 39/39 | 911 ms | 970 ms | $0.000920 |

Jev accepted all held-out labels without fallback. Relative to the equally
informed Gemini control: about 58% lower median and 58% lower token cost, but a
slower first/cold batch. Both models benefited from correct taxonomy scoping;
the quality improvement over the old mapper must **not** be attributed entirely
to Jev. The new labels were authored after the six development labels and not
used for further policy tuning; they remain a tiny synthetic test, not an
independent external benchmark. Alias matching is deliberately bypassed in this
direct classifier comparison; real imports should avoid these calls whenever
an alias already resolves the type.

### Negative result: Needle literal extraction

Needle 3 (`cactus-needle==3.0.1`) accepted **0/12** literal extraction attempts.
All used the original model. A raw high-confidence example extracted only
“concise answers” and classified it as convention from “I prefer concise
answers without emojis.” The exact-source guard rejected it; lowering that
guard to manufacture savings would lose meaning. Candidate classification is
not sufficiently reliable merely because a function call is valid.

The prototype rejected missing/nonfinite/low confidence and required the entire
source as a verbatim quote, but native inference lacked process-level
crash/timeout isolation. The service lane and dependency were removed from this
PR after review; this negative result remains research context only.

The actual downloaded Needle archive was 35,335,380 bytes, SHA-256
`c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38`.
Python's download client encountered a certificate-chain error on this host;
the official artifacts were fetched with TLS verification intact and checked
against published hashes. No TLS verification was disabled. See the
[Needle repository](https://github.com/cactus-compute/needle) and
[Needle 3 model](https://huggingface.co/Cactus-Compute/needle3).

### Next extraction design: selection rather than generation

Jev's inability to generate prose does **not** imply that the entire extraction
workflow must remain an LLM call. A plausible redesign is deterministic source
segmentation → batched durable-fact selection plus bucket classification →
verbatim storage with source/speaker/time provenance. Selecting candidates is
compatible with Jev's typed API. Normalizing pronouns, splitting compound facts,
joining evidence across messages and resolving implicit dates/corrections still
require additional machinery or fallback. Verbatim passage selection must not
be mislabeled as equivalent atomic fact extraction.

This redesign has now been implemented behind `JEV_EXTRACTION_ENABLED=false`;
see [the extraction follow-up](29-jev-extraction-evaluation.md) for the baseline,
held-out comparison and actual extraction-to-graph/vector checks. It selects
source sentences and preserves original generation for unsupported inputs. The
failed Needle experiment did not establish that Jev candidate selection would
fail. Adding Jev after a Gemini call that already extracts and categorizes facts
would not remove that call; this implementation instead runs selection first.

## Research and competitor implications

The research and YouTube workflows informed this implementation; all four
supplied videos were reviewed through downloaded captions, not inferred from
their titles. The latest reranking video was reviewed before implementation.
Captions cannot establish details visible only in diagrams or unspoken demos.

- [System One introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
  [Jev explained](https://www.youtube.com/watch?v=vj7hysh0mOI), and
  [Open Jev models](https://www.youtube.com/watch?v=53wDOI_7x8I): bounded choices
  and scores are a good fit; they do not imply free-form generation or a
  drop-in replacement for every model client. This pilot used hosted Jev,
  not an evaluated self-hosted Jev deployment.
- [Needle 3 video](https://www.youtube.com/watch?v=qbN559fQn7k): small local
  function/slot models are plausible for literal structured inputs. Our
  extraction result did not validate a broader replacement claim.
- [Reranking video](https://www.youtube.com/watch?v=UhGH8cNG0qs): policy-aware
  candidate scoring motivated the optional recall reranker. Its reported
  BM25/demo accuracy gain is not a NeuralScape result. The linked Colab notebook
  was inaccessible; its code was not inspected.

The browser-use repo was inspected at
`1231850a0bf1a0c0341fe408ef1668dbbfdfac46`. Its
[model implementation](https://github.com/browser-use/jev-ultrafast/blob/1231850a0bf1a0c0341fe408ef1668dbbfdfac46/jev_ultrafast/model.py)
batches operation/target choices, uses deterministic candidate IDs, and retains
a generative helper for typed text. That pattern maps to graph resolution and
rank selection. Its [performance report](https://github.com/browser-use/jev-ultrafast/blob/1231850a0bf1a0c0341fe408ef1668dbbfdfac46/docs/performance.md)
compares old/new browser machinery using the same models: three runs per arm,
9.450 → 7.092 s. The ~$0.00006272 demo figure covers the text helper, not all Jev
costs. It is not evidence that all LLMs can be replaced or a production cost SLA.

| Competitor / adjacent project | Primary-source pattern | Implication |
|---|---|---|
| [Mem0](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm) | ADD-only extraction, removal of redundant update machinery | Eliminate unnecessary work before replacing remaining models; our fork already incorporates this direction |
| [Zep / Graphiti](https://help.getzep.com/graphiti/working-with-data/searching) | Hybrid vector/BM25/RRF; local cross-encoder option | Don't confuse constructing a cross-encoder client with invoking an LLM on every search |
| [Graphiti model guidance](https://help.getzep.com/graphiti/configuration/llm-configuration) | Structured extraction reliability matters | Keep generation/extraction quality gates separate from cheap decisions |
| [Cognee GLiNER](https://docs.cognee.ai/python-api/cognify#llm-free-extraction-with-gliner) | Bounded LLM-free entity/relation extraction, deterministic summaries; limitations on temporal/custom instructions | Closest precedent for a selective small-model graph path; candidate for a separate benchmark, not implemented here |
| [Supermemory](https://supermemory.ai/product/) | Background memory processing, graph/profile claims | Offloading work is not removing model cost; no verified evidence of Jev usage or sufficient public unit economics |
| [Letta](https://docs.letta.com/configuration/memory) | Persistent memory and background agent work | Schedule synthesis off the critical path; don't pretend synthesis became classification |

No primary evidence found that these competitors use Jev internally. GLiNER2 is
a promising future candidate for fixed ontologies, not a validated replacement
for Graphiti's temporal reasoning in this branch. Likewise, local BGE/MiniLM
can improve economical relevance ranking, but policy compliance must be tested
separately. Hosted vendor marketing and this tiny pilot are different evidence
classes.

## Safety and rollout boundaries

- Jev uses the documented [System One API](https://docs.typesafe.ai/api), pinned
  `jev-1.13.0`, no retries on the fast path, connection reuse, bounded payloads
  and candidate counts, and response/model-version validation.
- Choices must have the exact allowed answer set, finite normalized probabilities,
  valid selected IDs, and both selected probability/confidence ≥0.9. These are
  routing thresholds, **not calibrated correctness guarantees**; see
  [Typesafe confidence](https://docs.typesafe.ai/confidence).
- The original model owns detected conflicts and uncertainty. Jev can still be
  confidently wrong; a mistaken DISTINCT or entity merge remains a residual
  risk. The current checks do not establish a production false-merge rate.
- Graph async decisions have a deadline; synchronous HTTP calls have per-phase
  timeouts. A timed-out thread may finish later, and possible paid usage must
  not be called zero. This is not a universal request wall-clock SLA.
- Reranking runs after authorization and kind filters, never on internal
  `vector_only` write probes. It permutes rows without deleting them, altering
  stored facts, or relabeling retrieval scores as Jev confidence.
- Safe telemetry includes operation, duration, model, usage and fallback reason,
  not content, credentials or raw provider errors. Reports are ignored under
  `.nsbench-reports/`; pilot inputs are synthetic, not private user memories.
- Enabling hosted Jev sends authorized candidate text to an additional provider.
  Review retention, residency and account terms before using real memories.
  No training/distillation from Jev outputs was performed; labels are handwritten.
- The environment-variable skill informed the empty-key template and
  explicit private-file loading. This Python service keeps its existing
  gitignored `.env` convention; no Vercel linkage or configuration was changed.

Promote only after a larger human-reviewed, representative replay demonstrates
acceptable false merge/invalidation rates, recall quality and total cost,
including embeddings, retries, fallback, queues and concurrency. Freeze the
corpus and compare paired/interleaved runs. Test long contexts, multilingual
inputs, prompt injection, outages and rate limiting. Run full API/ARQ load and
failure recovery separately. No production traffic replay, LoCoMo/LongMemEval
study, concurrent-load measurement or self-hosted Jev economics was completed.

## Reproduce and test safely

From the worktree root, start the isolated backing stores (different loopback
ports; no production volumes):

```bash
docker compose -p ns-hybrid-eval -f neuralscape-bench/docker-compose.hybrid.yml up -d
cd neuralscape-service
uv sync --frozen --extra code-graph
```

Supply an **absolute path** to your private key file in place of
`/absolute/path/private.env`. It needs `GOOGLE_API_KEY` and `TYPESAFE_API_KEY`.
Scripts load it explicitly without printing values. The graph benchmark
overrides database URLs to `127.0.0.1:17687`, `:16333`, and `:16379`, disables
the gateway, and creates unique test namespaces. It leaves test data intact.
Wait until the stores are ready, then run baseline before hybrid:

```bash
.venv/bin/python scripts/benchmark_hybrid.py --arm baseline --env-file /absolute/path/private.env --output ../.nsbench-reports/hybrid/baseline-new.json
.venv/bin/python scripts/benchmark_hybrid.py --arm hybrid --env-file /absolute/path/private.env --output ../.nsbench-reports/hybrid/hybrid-new.json
# Add --challenge to both commands for the additional graph decision cases.
.venv/bin/python scripts/benchmark_hybrid_graph.py --arm baseline --env-file /absolute/path/private.env --output ../.nsbench-reports/hybrid/graph-baseline-new.json
.venv/bin/python scripts/benchmark_hybrid_graph.py --arm hybrid --env-file /absolute/path/private.env --output ../.nsbench-reports/hybrid/graph-hybrid-new.json
.venv/bin/python scripts/benchmark_hybrid_graph.py --arm hybrid --simulate-jev-outage --env-file /absolute/path/private.env --output ../.nsbench-reports/hybrid/graph-fallback-new.json
.venv/bin/python scripts/benchmark_hybrid_rerank.py --env-file /absolute/path/private.env --output ../.nsbench-reports/hybrid/rerank-new.json --local
.venv/bin/python scripts/benchmark_hybrid_categories.py --env-file /absolute/path/private.env --output ../.nsbench-reports/hybrid/categories-new.json
# Additional labels and an equally informed Gemini control:
.venv/bin/python scripts/benchmark_hybrid_categories.py --held-out --scoped-control --env-file /absolute/path/private.env --output ../.nsbench-reports/hybrid/categories-control-new.json
.venv/bin/pytest tests/ --ignore=tests/test_async_pipeline.py -q
```

The rerank script downloads MiniLM on first `--local` run; use
`--local-model-path /absolute/path/to/model` for pre-provisioned artifacts.
Downloads are not included in warm latency. The rejected Needle prototype is
not included in the service runtime or benchmark command surface.

For a sandbox service trial, set `JEV_GRAPH_ENABLED=true` first; leave
`JEV_CATEGORIES_ENABLED` and `JEV_RERANK_ENABLED` false. Restart graph workers
because clients are constructed at initialization.
Test reranking separately with `JEV_RERANK_ENABLED=true`; restart API/search
workers. Reverting flags and restarting restores the original model paths;
it does **not undo graph mutations** already made, so use isolated data first.

Container gate, from the repository root:

```bash
docker build --target test -f neuralscape-service/Dockerfile -t ns-hybrid-gate .
docker run --rm ns-hybrid-gate
docker build --target runtime -f neuralscape-service/Dockerfile -t ns-hybrid-runtime .
```

Verification before the extraction follow-up: **2,672 passed, 2 skipped** on both the
host and the built test container. Existing deprecation/mock-coroutine warnings
remain; this is not a warning-free suite. Fifty new hybrid contract tests cover
provider errors, malformed choices/scores, timeout fallback, category scoping,
partial acceptance, batch limits, graph decision wiring and safe reranking. The
initial Needle contract tests were removed with the rejected runtime lane.
`git diff --check` passes.

The container suite excludes the existing live-ARQ test module. The separate
graph benchmark above provides actual DB coverage, not API/queue coverage.
Stop the sandbox without deleting its data with
`docker compose -p ns-hybrid-eval -f neuralscape-bench/docker-compose.hybrid.yml stop`.

Private report files from this evaluation: `baseline.json`, `jev-pilot.json`,
`jev-needle-pilot.json`, `challenge-{baseline,hybrid}.json`,
`graph-{baseline,baseline-warm,hybrid,fallback}.json`,
`graph-expanded-{baseline,hybrid,fallback}.json`, `rerank.json`, `categories.json`,
`categories-{scoped,shared,heldout,control-heldout}.json`.
They are intentionally not checked into this public repository.
