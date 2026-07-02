# Knowledge Adapters + the Trading-Strategy Adapter

Neuralscape ingests different *kinds* of knowledge through pluggable **knowledge
adapters**. An adapter swaps the parts of the ingest pipeline that are
domain-specific — the **taxonomy**, the **chunking strategy**, the **fact
extractor**, and the **knowledge-graph ontology** — while keeping the **fixed
metadata envelope** (`source_ref`, `memory_kind`, content-hash dedup, scope,
visibility, provenance) byte-for-byte identical across adapters.

The first non-default adapter is `trading_strategy`, which ingests trading books
into a queryable *and* executable strategy graph.

## The envelope vs. the taxonomy

| Fixed (never per-adapter) | Swappable (per-adapter) |
|---|---|
| `id`, `memory`, timestamps | `category` set + descriptions |
| `owner_user_id`, `visibility`, `scope`, `project_id` | chunking strategy |
| `memory_kind` (`fact`/`passage`), `source_ref` | fact-extraction prompt/parser |
| content-hash dedup | Graphiti `entity_types`/`edge_types`/`edge_type_map`/`custom_extraction_instructions` |
| `agent_id`/`run_id` | synthesis policy (`synthesizer`, `synthesis_group_key`) |

A guardrail test (`tests/test_adapters.py`) asserts a `default`-adapter ingest is
equivalent to the pre-adapter path.

## Selecting an adapter

`adapter` is threaded end-to-end and defaults to `"default"` (current behavior):

- REST: `POST /v1/ingest`, `/v1/ingest/text`, `/v1/ingest/files` accept an
  `adapter` field/form-field.
- MCP: `ingest_document` and `ingest_text` take an `adapter` argument.
- Unknown adapter names are **rejected loudly (400/422) at the REST/MCP request
  boundary** — a typo'd adapter silently ingesting without the taxonomy/ontology
  the caller asked for would be worse than an error. Only **worker-side**
  resolution degrades to `default`, and only for jobs already queued when an
  adapter was removed.

```bash
curl -F 'files=@naked-forex.pdf' \
     -F 'adapter=trading_strategy' \
     -F 'tags=strategy:naked-forex-reversal' \
     -F 'visibility=shared' \
     "$NS/v1/ingest/files"
```

The `strategy:<name>` tag is how the strategy synthesizer groups a book's rules
into one playbook — set it per book/strategy at ingest time.

## Architecture (the four seams)

```
adapters/
  base.py                 KnowledgeAdapter (frozen dataclass) + ADAPTER_REGISTRY + get_adapter
  trading/
    profile.py            registers the trading_strategy adapter + its taxonomy
    ontology.py           Graphiti entity/edge types (Strategy, Setup, EntryCondition, …)
    chunking.py           SectionAwareStrategy (book/section-aware)
    extractor.py          TradingStrategyExtractor (rule_ast + executable_expression + citation)
    exemplars.py          visual setup exemplars (see below)
ingest/
  chunking_strategies.py  ChunkingStrategy protocol + registry (ParagraphAwareStrategy default)
  extractors.py           FactExtractor protocol + registry (DefaultExtractor default)
  pipeline.py             resolves get_adapter(doc.adapter); uses its chunker/extractor/ontology
```

1. **Taxonomy** — adapters call `schemas.register_categories(...)` at import to add
   categories to the shared `MEMORY_CATEGORIES` (additive; the core 13 are never
   changed) so `store_raw`, the request validators, and the parser all accept them.
2. **Chunking** — `IngestDoc.adapter` → `get_chunking_strategy(adapter.chunking_strategy)`.
3. **Extraction** — `extract_facts_only(text, extractor=get_extractor(adapter.extractor))`.
4. **Graph ontology** — resolved *per-ingest* and passed at `add_episode` call time
   (not baked into `get_mem0_config`): `adapter.graph_ontology_kwargs()` →
   `enrich_graph(graph_ontology=)` → `MemoryGraph.add` → `add_episode`.

### Deferred fact enrichment (book-scale ingests)

Ingested facts are stored **vector-fast** and their (slow, ~minutes-per-fact)
Graphiti enrichment is **deferred onto the graph queue**: `ingest_document`
returns one `graph_jobs` entry per newly-created fact, and the ingest worker
enqueues each as a `process_graph_enrichment` job carrying the adapter *name*
(the ontology itself isn't queue-serializable — the graph worker re-resolves it
via `get_adapter`). Without this, a book's hundreds of facts × minutes of
entity extraction each would blow the ingest `job_timeout` and re-run the whole
Docling parse on every ARQ retry. Dedup hits produce no job. Unknown adapter
names **fail loudly (422) at the API/MCP boundary**; only already-queued jobs
degrade to the default adapter.

## The trading-strategy adapter

### Taxonomy
12 categories: `strategy`, `setup`, `entry_rule`, `exit_rule`, `stop_rule`,
`take_profit_rule`, `market_condition`, `sr_concept`, `risk_rule`,
`psychology_rule`, `checklist`, `glossary`.

### Ontology (`adapters/trading/ontology.py`)
Entities model an *executable* strategy: `Strategy`, `Setup`, `EntryCondition`,
`StopLoss`, `TakeProfit`, `ExitCondition`, `RiskRule`, `SupportResistanceZone`,
`MarketRegime`, `RuleNode` (AST), `Timeframe`, `Instrument`, `Signal` (LEAN
`Insight` shape), `Backtest`, `Ensemble`, `VisualExemplar`, …

**The 3-part gate (fidelity rule):** a catalyst is only a trade when it prints
**on a zone** (and, for continuation setups, in the right regime). This is
encoded as hard `REQUIRES` edges — the `edge_type_map` constrains
`(Setup, SupportResistanceZone)` to `[REQUIRES]` so a compiled strategy can't
fire a setup out of context.

> **#1111 hedge.** Graphiti has a known gap populating *custom edge attributes*.
> So all load-bearing, compiler-facing data (`rule_ast`, `executable_expression`,
> offsets, anchors) lives on **entity** nodes; edge types are thin markers.

### Extractor
`TradingStrategyExtractor` extracts **executable rules, not prose**. Each rule
memory keeps, inline: the statement, a machine-checkable `rule_ast`
(boolean/expression tree), an `executable_expression`, and the verbatim
`source_quote` + `page_ref`. This makes each rule recallable by vector search,
compilable by Bellwether, and rich enough for Graphiti to extract the ontology.

### Section-aware chunking
Books build strictly (philosophy → risk → zones → setups → exits → psychology).
`SectionAwareStrategy` breaks on heading/`CHAPTER` boundaries and keeps a setup's
identification + entry + stop + target in one chunk (falling back to
paragraph-chunking within oversized sections, with absolute spans preserved).

## Cumulative synthesis — the strategy playbook synthesizer

`extensions/strategy_synthesizer/` clones the `wiki_synthesizer` pattern, grouped
by **strategy** instead of `(group_id × category)`:

- Per `(owner, strategy)`, scroll the strategy's rule memories, incrementally
  merge them via Gemini into one canonical **playbook** page
  (`Playbooks/<owner>/<strategy>.md`) with sections *Thesis, Setups, Entry, Stop,
  Targets, Exits, Risk, Market Conditions, Gotchas, Version Updates*.
- **Non-destructive:** source memories are immutable; playbook pages are
  versioned append-only (`version_number++`); contradictions resolve via
  Graphiti bi-temporal invalidation. If the source memory-id set is unchanged,
  the LLM merge is **skipped** (idempotent).
- Patches contributing Neo4j nodes with a `strategy_playbook_path` back-reference.
- Cron on the graph worker at `:55` (staggered after wiki-synth `:45`), gated by
  `STRATEGY_SYNTHESIZER_ENABLED` (dark by default).

> **Visibility note.** Unlike the wiki synthesizer (shared-only by design), the
> strategy synthesizer includes **private** trading memories: playbooks are
> keyed per-owner (`Playbooks/<owner>/…`), but they land in the one shared vault
> volume, which the operator can read. Fine for a single-user deployment (and
> the feature is dark by default); revisit before enabling on a multi-user
> instance — either filter to `visibility=shared` or split the vault per owner.

## Visual setup exemplars (chart images)

Trading books show annotated pictures of setups. With `EXEMPLAR_STORE_ENABLED`
and `DOCLING_EXTRACT_IMAGES`, the trading adapter (in the ingest worker):

1. extracts embedded figures from the file via docling-serve
   (`image_export_mode=embedded`);
2. stores each image's bytes in an object store (v1: local dir, `file://` URI in
   `source_ref.stored_path`, keyed by content hash — `EXEMPLAR_STORE_DIR`);
3. **vision-describes** it with a multimodal model (Opus 4.8 via the LLM gateway,
   `EXEMPLAR_VISION_MODEL`) into `{setup_name, direction, visual_description,
   key_levels, caption}`;
4. writes a normal `setup`/`visual_exemplar` memory whose body **is** the
   description (so it embeds into Qdrant — the v1 text-proxy recall index) plus a
   `VisualExemplar` graph node (`EXEMPLIFIES → Setup`).

Notes:
- Text + images come from **one** Docling conversion (`extract_text_and_images`) —
  a book PDF takes minutes per parse, so the two paths never convert separately.
- Re-ingesting a book does **not** duplicate exemplars: the vision description is
  nondeterministic, so idempotency keys on the **image content hash**
  (`source_ref.external_id`) via a pre-store lookup, not on the memory body.
- Book-scale PDFs are best uploaded as page-range **slices** (a single 290-page
  job would blow the ingest `job_timeout`) under one `strategy:` tag. Pass
  `page_offset` per slice (pages 61–80 → `page_offset=60`) so exemplar
  `page_ref`s stay relative to the original book, not the slice; the offset is
  part of the deterministic job-id, so re-uploading a slice with a corrected
  offset is a new job, not a coalesced dup.
- Image bytes are retrievable over HTTP: `GET /v1/ingest/exemplars/{image_id}`
  (owner-scoped — resolution goes through the caller's exemplar memory). The API
  container mounts the exemplar volume for this.

Consumption (a chart-vision agent retrieving exemplars and applying a bounded,
cited confidence boost) lives in **Bellwether**, not NS. v2 upgrade: true CLIP
image embeddings in a separate Qdrant collection.

## Config flags (all dark by default)

| Env | Purpose |
|---|---|
| `STRATEGY_SYNTHESIZER_ENABLED` | turn on the playbook synthesizer cron |
| `STRATEGY_SYNTHESIZER_CRON_HOURS` | cadence (default 6h) |
| `EXEMPLAR_STORE_ENABLED` | turn on visual-exemplar ingestion |
| `EXEMPLAR_STORE_DIR` | object-store dir (v1 local) |
| `EXEMPLAR_VISION_MODEL` | multimodal model id (empty ⇒ gateway default) |
| `DOCLING_EXTRACT_IMAGES` | request embedded images from docling-serve |

Visual exemplars require the LLM gateway (`LLM_GATEWAY_ENABLED`) for the
multimodal describe step.

## The bridge to Bellwether (out of scope for NS)

The graph → runnable-`Strategy` **compiler** lives in Bellwether (it depends on
Bellwether's `Strategy`/`OrderIntent`), reading the strategy graph via the
Neuralscape MCP tools. NS owns the semantic + procedural layers (ontology,
provenance, synthesis); Bellwether owns execution + validation (CPCV/DSR/PBO,
written back as `Backtest` nodes). The strategy graph is the interchange format.
