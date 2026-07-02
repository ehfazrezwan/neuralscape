# Plan: Pluggable Knowledge Adapters + a Trading‑Strategy Learning Adapter

**Audience:** a Claude Code session working in the `neuralscape` repo.
**Author:** handed off from the Bellwether project (the trading platform that will *consume* this).
**Status:** design + implementation plan. Nothing here is built yet.

> This document is self‑contained. It cites the exact files/seams in `neuralscape-service` you'll
> touch, the taxonomy to build, and the phased implementation with acceptance criteria. Verify
> every file:line against the current tree before editing (line numbers drift).

---

## 0. The two asks (from the product owner, verbatim intent)

1. **Make Neuralscape accept different *kinds* of knowledge‑graph ingestion** — pluggable "data
   adapters" that **keep the same fixed metadata envelope** but swap the *taxonomy* (category set,
   controlled vocabularies), the *chunking* strategy, the *fact‑extraction* prompt, and the
   *knowledge‑graph entity/edge types*.
2. **Build the first such adapter: a "trading‑strategy learning" adapter** that ingests trading
   books (starting with *Naked Forex*) page‑by‑page and slowly builds a **canonical, cumulative
   understanding of each strategy** — where *each new synthesis is the sum of all lessons learned
   so far, without destructively rewriting old memories*. Eventually the owner wants to **pick and
   combine strategies and run them in permutational combinations** (this connects to Bellwether's
   `Strategy` contract + evaluation harness).

Design north star: **the graph must be both *queryable* (so the agent can recall a strategy) and
*executable* (so the strategy can be compiled into a runnable Bellwether `Strategy`)**, and the
synthesis must be **additive + provenance‑linked + non‑destructive**.

---

## 1. What already exists (grounded in the code)

### 1.1 The memory model — a fixed envelope + a swappable taxonomy (`schemas.py`)
- **Fixed metadata envelope** (do NOT change per adapter): `id`, `memory` (content), `created_at/updated_at`,
  `tags`, `owner_user_id`, `visibility`, `agent_id/run_id`, `scope` (global/project), `project_id`,
  and the data‑layer fields `memory_kind` (`fact`|`passage`) + `source_ref` (`SourceDescriptor`:
  `connector_id`, `chunk_index`, `span`, `content_hash`, `retrieval` handle).
- **Taxonomy‑specific fields** (this is the swappable part): `category` (the 13‑item
  `MemoryCategory` enum + `MEMORY_CATEGORIES` dict ~L115), and the memory‑model‑v2 vocabularies
  `domain` (`DOMAIN_VOCAB`), `observation_type` (`OBSERVATION_TYPE_VOCAB`), `concepts`
  (`CONCEPT_VOCAB`), `source_type` (~L172‑209).

**Key realization:** the envelope already cleanly separates "storage identity/provenance" (fixed)
from "how this fact is classified" (category + vocabs). An adapter overrides only the latter.

### 1.2 The ingestion pipeline (`ingest/pipeline.py`, `ingest/chunking.py`, `prompts.py`, `memory_service.py`)
`ingest_document(service, IngestDoc)` runs two paths:
- **Passages path:** `chunk_text()` (paragraph‑aware, `max_chars=1500`, `overlap=150`) →
  `store_raw(memory_kind="passage", add_to_graph=False)` (fast, vector‑only).
- **Facts path:** `memory_service.extract_facts_only()` → **one hardcoded**
  `CODING_ASSISTANT_EXTRACTION_PROMPT` (`prompts.py`) → emits `[category] fact` lines →
  `store_raw(memory_kind="fact", add_to_graph=True)` → graph enrichment.

There is currently **no way to choose a taxonomy, chunking strategy, or extraction prompt** — all
three are hardcoded. These are your three primary seams.

### 1.3 The knowledge graph (Graphiti on Neo4j)
- `store_raw → enrich_graph → mem0/graphiti_memory.py MemoryGraph.add → graphiti.add_episode`.
- **Graphiti already supports custom ontology** via `add_episode(entity_types=..., edge_types=...,
  edge_type_map=..., excluded_entity_types=..., custom_extraction_instructions=...)` — but
  **Neuralscape does not pass any of these today** (`get_mem0_config()` builds the `graph_store`
  block without entity/edge types; `MemoryGraph.add()` calls `add_episode` with none). This is the
  fourth seam.
- Graphiti gives you **bi‑temporal validity** (`valid_at`/`invalid_at` + `created_at`/`expired_at`),
  **contradiction‑as‑invalidation** (never deletion — history preserved), and **entity dedup**
  (re‑ingesting the same setup across books consolidates rather than fragments). These are exactly
  the non‑destructive properties ask #2 needs — lean on them, don't reinvent them.

### 1.4 The extension system (`extensions/`) — the synthesis precedent
- Extensions implement `NeuralscapeExtension` (`extensions/base.py`: `manifest`, `startup/shutdown`,
  `on_event`, `get_routes`), discovered by `ExtensionRegistry` at worker startup, driven by cron
  (slow `GraphWorkerSettings` queue) or events (`extensions/events.py`).
- **`wiki_synthesizer` is the blueprint for cumulative synthesis.** Cron‑driven, per
  `(group_id, category)`: scroll Qdrant for source memories → read the existing wiki page's
  frontmatter `source_memory_ids` → **skip the LLM if the id‑set is unchanged (idempotent)** → else
  run `INCREMENTAL_MERGE_PROMPT` (existing body + new memories → merged body) → write a **new
  versioned page** (`synthesis_count++`) → patch Neo4j nodes with a `wiki_path` back‑reference.
  **Source memories are never mutated; only synthesis pages are appended.** This *is* "each new
  synthesis = sum of all lessons." The trading synthesizer clones this pattern, grouped by
  **`strategy_name`** instead of category.

---

## 2. Architecture — Part A: the pluggable "Knowledge Adapter" (ask #1)

Introduce one new abstraction that bundles everything an adapter overrides, plus a registry.

### 2.1 The `KnowledgeAdapter` profile
New module `adapters/` (sibling to `ingest/`). An adapter is a declarative profile + a few
pluggable callables:

```python
# adapters/base.py
class KnowledgeAdapter(BaseModel):
    name: str                      # "default" | "trading_strategy"
    version: str

    # (1) Taxonomy — overrides schemas.py constants for THIS adapter only
    categories: dict[str, str]                 # category -> description
    global_categories: set[str]
    project_categories: set[str]
    concept_vocab: set[str] | None = None      # override CONCEPT_VOCAB
    observation_type_vocab: set[str] | None = None

    # (2) Chunking — a strategy id resolved from a registry
    chunking_strategy: str = "paragraph_aware"
    default_max_chars: int = 1500
    default_overlap: int = 150

    # (3) Extraction — a FactExtractor id resolved from a registry
    extractor: str = "default"

    # (4) Graph ontology — Graphiti custom types (Pydantic classes)
    entity_types: dict[str, type[BaseModel]] | None = None
    edge_types: dict[str, type[BaseModel]] | None = None
    edge_type_map: dict[tuple[str, str], list[str]] | None = None
    custom_extraction_instructions: str | None = None

    # (5) Synthesis policy — which synthesizer + grouping key
    synthesizer: str | None = None             # e.g. "strategy_synthesizer"
    synthesis_group_key: str | None = None      # e.g. "strategy_name"

ADAPTER_REGISTRY: dict[str, KnowledgeAdapter] = {"default": DEFAULT_ADAPTER}
def get_adapter(name: str = "default") -> KnowledgeAdapter: ...
```

The `"default"` adapter simply re‑exposes today's behavior (the 13 categories, paragraph chunker,
`CODING_ASSISTANT_EXTRACTION_PROMPT`, no custom graph types) so **nothing regresses**.

### 2.2 The three pluggable strategies (make the hardcoded pieces registries)
- **Chunking** — `ingest/chunking_strategies.py`: a `ChunkingStrategy` protocol (`chunk(text,
  max_chars, overlap) -> list[Chunk]`) + `CHUNKING_STRATEGIES` registry. Move today's `chunk_text`
  into `ParagraphAwareStrategy`. Add `SectionAwareStrategy` for books (see §3.3).
- **Fact extraction** — `ingest/extractors.py`: a `FactExtractor` protocol
  (`build_messages(text) -> list[msg]`, `parse(response) -> list[(category, content)]`) +
  `FACT_EXTRACTORS` registry. Wrap today's prompt as `DefaultExtractor`; add
  `TradingStrategyExtractor` (see §3.4).
- **Graph types** — thread `entity_types/edge_types/edge_type_map` from the adapter down through
  `enrich_graph → MemoryGraph.add → add_episode`. This is plumbing‑only but touches
  `memory_service.py`, `mem0/graphiti_memory.py`, and `config.get_mem0_config()`.

### 2.3 Selecting an adapter (thread one new field end‑to‑end)
Add `adapter: str = "default"` to: the `ingest_document` **MCP tool** (`mcp_server.py` ~L271) and
`ingest_text`; the `IngestDocumentRequest` schema; the `IngestDoc` dataclass; and
`ingest/pipeline.py:ingest_document`, which resolves `get_adapter(doc.adapter)` and uses its
chunker/extractor/graph‑types. **Backward‑compatible:** default preserves current behavior.

> **Guardrail:** the *fixed envelope* (`source_ref`, `memory_kind`, `content_hash` dedup, scope,
> visibility, provenance) must be identical across adapters. Only category/vocab/chunking/
> extraction/graph‑types vary. Add a test asserting a `default`‑adapter ingest is byte‑for‑byte
> equivalent to today's path.

---

## 3. Architecture — Part B: the Trading‑Strategy adapter (ask #2)

### 3.1 The trading taxonomy (categories)
Grounded in the *Naked Forex* concept analysis. These become the adapter's `categories`:

| category | what it captures |
|---|---|
| `strategy` | a named strategy container (thesis, bias, member setups) |
| `setup` | a tradable pattern/catalyst (kangaroo tail, big shadow, last kiss, …) |
| `entry_rule` | the trigger (order type, reference price, offset) |
| `exit_rule` | dynamic trade management (zone/split/ladder/three‑bar/trailing) |
| `stop_rule` | stop‑loss placement |
| `take_profit_rule` | fixed target placement (RR multiple / next zone) |
| `market_condition` | regime/context gate (trend, range, exhaustion, session) |
| `sr_concept` | support/resistance **zones** (the core primitive) + round numbers/trendlines |
| `risk_rule` | position sizing, risk‑per‑trade, survival limits |
| `psychology_rule` | discipline/temperament guardrails (gunner vs runner) |
| `checklist` | the ordered routine that gates a trade |
| `glossary` | definitions + aliases (kangaroo tail ≈ pin bar) for cross‑book normalization |

(`timeframe` and `instrument` are better modeled as graph entities than memory categories — see §3.2.)

### 3.2 The graph ontology (Graphiti custom types)
Define Pydantic `BaseModel`s in `adapters/trading/ontology.py`. **The docstring is load‑bearing —
Graphiti uses it to classify.** Avoid reserved fields (`uuid`, `name`, `group_id`, `labels`,
`created_at`, `summary`, `attributes`, `name_embedding`); make attributes `Optional`; use precise
types; write rich `Field(description=...)`.

**Entity types:** `Strategy`, `Setup`, `Instrument`, `Timeframe`, `Indicator` (rare in Naked Forex
but needed for other books), `MarketRegime`, `SupportResistanceZone`, `Signal` (copy QuantConnect
LEAN's `Insight`: `symbol`, `direction`, `period`, `type`, `magnitude`, `confidence`, `weight`,
`source_model`), `EntryCondition`, `ExitCondition`, `StopLoss`, `TakeProfit`, `RiskRule`,
`RuleNode` (an AST node), `Backtest` (validation results).

**Edge types:** `HAS_SETUP`, `TRADES`, `ON_TIMEFRAME`, `USES_INDICATOR`, `HAS_ENTRY`, `HAS_EXIT`,
`HAS_STOP`, `HAS_TARGET`, `CONSTRAINED_BY` (→RiskRule), `ACTIVE_IN_REGIME`, `REQUIRES` (the gate —
Setup→SupportResistanceZone / Setup→MarketRegime), `HAS_CHILD` (bracket triad), `DERIVED_FROM`
(provenance → source episode/chunk), `COMPOSED_OF` (Ensemble→Strategy), `VALIDATED_BY`
(Strategy→Backtest).

**The fidelity rule (do not skip):** *Naked Forex*'s central law is the **3‑part gate** — a
catalyst is only a trade **when it prints on a zone** (and, for continuation setups, in the right
regime). Encode this as **hard `REQUIRES` edges** so a compiled strategy can never fire a setup out
of context. This is the single biggest book→execution fidelity risk.

### 3.3 Chunking for books (`SectionAwareStrategy`)
Books have chapter/section structure and the concepts build strictly (philosophy → risk → zones →
setups → exits → psychology). Chunk on heading/section boundaries with generous overlap so a
setup's entry/stop/exit stay in one chunk. Stamp each chunk's `source_ref` with `page`/`chapter`
(extend `SourceDescriptor` usage; the field already supports `span` + arbitrary provenance). Rule
text must retain a verbatim `source_quote` + `page_ref` for citation and auditing.

### 3.4 The trading fact extractor (`TradingStrategyExtractor`)
A domain prompt that extracts **executable rules, not prose**. For each rule it must emit:
- the `[category]` prefix (from §3.1),
- a machine‑checkable **`rule_ast`** — a boolean/expression tree over OHLC/zone/equity
  (`{op: AND|OR|GT|LT|crossesAbove|closeInThird|engulfs|…, left, right}` or a leaf
  `{fact, operator, value}`). This is the industry‑standard rule‑engine representation and is both
  queryable and compilable.
- an **`executable_expression`** (e.g. `buy_stop = pattern.high + offset_pips`,
  `stop = pattern.low - offset_pips`, `size = equity*risk_pct / (stop_distance*pip_value)`),
- the `source_quote` + `page_ref`.

Pass `custom_extraction_instructions` to `add_episode` so Graphiti's own entity extraction is
trading‑aware too.

### 3.5 Worked example — the Kangaroo Tail encoded (target output)
```
Strategy "Naked Forex — Reversal"  —HAS_SETUP→  Setup "Kangaroo Tail"
Setup "Kangaroo Tail":
  identification rule_ast:  AND[
     closeInThird(candle, 3rd),  openInThird(candle, 3rd),
     GT(range(candle), max(range, last=10)),  insideRange(candle, prev),
     roomToLeft(candle) ]
  —REQUIRES→ SupportResistanceZone (membership_test(price) true)     # the gate
  —HAS_ENTRY→ EntryCondition {order:'stop', ref: candle.high, offset:'+5pips+spread',
                              time_validity:'next 1 candle else cancel'}
  —HAS_STOP→  StopLoss {anchor: candle.low, offset:'-few pips', type:'emergency'}
  —HAS_TARGET→ TakeProfit {anchor:'next opposing zone'}  # plus 75%-to-stop drawdown-cut exit
  —ON_TIMEFRAME→ Timeframe {H1+, best on H4/D1}
```
This is enough for the graph→code compiler (§4) to emit a Bellwether bracketed `OrderIntent`.

---

## 4. Architecture — Part C: cumulative synthesis (`strategy_synthesizer` extension)

Clone `wiki_synthesizer`, grouped by **`strategy_name`**. New dir
`extensions/strategy_synthesizer/` (`__init__.py`, `manifest.json` with `hooks: []`, `config.py`,
`synthesizer.py`, `prompts.py`, `strategy_renderer.py`, `graph_patcher.py`). Wire a cron into
`GraphWorkerSettings` (stagger vs the wiki cron), gated by `STRATEGY_SYNTHESIZER_ENABLED`.

**Per strategy, per run:**
1. Scroll Qdrant for that strategy's memories (filter on `metadata.strategy_name` or the
   `strategy` category + a tag).
2. Read the existing **playbook page** frontmatter `source_memory_ids`; **skip if unchanged**
   (idempotent — the key efficiency trick).
3. Else run a `PLAYBOOK_MERGE_PROMPT` (existing playbook body + new lessons → a merged canonical
   playbook with sections: *Thesis, Setups, Entry, Stop, Targets, Exits, Risk, Market Conditions,
   Gotchas, Version Updates*). Force **citation alignment** — every synthesized claim links to its
   source memory (`DERIVED_FROM`), to prevent drift.
4. Write a **new versioned playbook page** (`version_number++`) and patch the contributing Neo4j
   nodes with `strategy_playbook_path`.

**Non‑destructive by construction:** source memories are immutable; playbook pages are versioned
(append‑only); and *contradictions between lessons* are handled by **Graphiti's bi‑temporal edge
invalidation** (superseded facts get `invalid_at`, never deleted). So "each new memory form = the
sum of all previous lessons" is satisfied at two levels: the **playbook** (human‑readable canonical
sum) and the **graph** (queryable, time‑travellable canonical sum).

> Optional later upgrade (RAPTOR/GraphRAG‑style): add a hierarchical layer — per‑setup summaries →
> per‑strategy playbook → cross‑strategy "market playbook" — all as append‑only synthesis nodes
> with `DERIVED_FROM` provenance. Not required for v1.

---

## 5. Architecture — Part D: the bridge to Bellwether (graph → runnable strategy)

This is where "pick and combine strategies and run them" lands. **Keep parse and execute
separate** (LLMs parse rules reliably but execute end‑to‑end poorly):

1. **Compile (deterministic):** a `graph → Bellwether Strategy` compiler walks a `Strategy` node's
   `RuleNode` AST and emits a concrete `decide(signals, state) -> OrderIntent | None` implementing
   Bellwether's `Strategy` protocol (already exists in `bellwether/eval/strategy.py`). Entry/stop/
   target map directly onto Bellwether's **bracketed `OrderIntent`** (which already carries
   `stop_loss`/`take_profit` and enforces the reward:risk gate). A **slot‑filler/validator refuses
   to emit** until required slots (zone source, offset, risk %) are filled.
2. **Validate (already built in Bellwether):** run the compiled strategy through Bellwether's
   evaluation harness — walk‑forward/**CPCV**, **Deflated Sharpe**, **PBO**, trial register — and
   write results back as a `Backtest` node (`VALIDATED_BY`), advancing `Strategy.status`
   (`draft → backtested → validated → live`). Store `trial_count` as first‑class metadata (DSR
   needs it).
3. **Compose/permute:** an `Ensemble` node `COMPOSED_OF` several `Strategy` nodes; each emits
   normalized `Signal`s (the LEAN `Insight` shape, stamped with `source_model`); a weighting/gating
   layer combines them (this is exactly Bellwether's `Signal.weight` + `WeightingPolicy` seam, and
   the planned Stage‑3 learned gate). Regime‑gate ensembles via `ACTIVE_IN_REGIME`.

**Division of labor:** Neuralscape owns the *semantic* layer (the strategy ontology + provenance +
synthesis) and *procedural* layer (the compiled skill, versioned by `Strategy.uuid`); Bellwether
owns *execution + validation*. The compiler is the contract between them — recommend it lives in
**Bellwether** (it depends on Bellwether's `Strategy`/`OrderIntent`), reading the strategy graph via
the existing Neuralscape MCP tools.

---

## 6. Implementation phases (ordered, each independently shippable + testable)

**Phase 0 — Adapter seam (no new behavior).** Introduce `adapters/` + `KnowledgeAdapter` +
`ADAPTER_REGISTRY` with only `"default"`. Convert the hardcoded chunker + extractor into registries
with the current logic as the default entries. Thread `adapter="default"` through MCP →
`IngestDoc` → pipeline. **Acceptance:** existing ingest tests pass unchanged; a new test asserts
`default` ingest == pre‑refactor output (envelope, categories, chunk spans identical).

**Phase 1 — Graph custom‑types plumbing.** Thread `entity_types/edge_types/edge_type_map/
custom_extraction_instructions` from adapter → `get_mem0_config`/`enrich_graph` →
`MemoryGraph.add` → `add_episode`. **Acceptance:** with `default` (no types) behavior is unchanged;
a unit test injects a toy entity type and asserts it reaches `add_episode` (mock Graphiti).

**Phase 2 — Trading adapter (taxonomy + chunker + extractor + ontology).** Build
`adapters/trading/` : the taxonomy profile, `SectionAwareStrategy`, `TradingStrategyExtractor`
(rule_ast + executable_expression + source_quote/page_ref), and the Graphiti ontology (§3.2).
Register it. **Acceptance:** ingest a small hand‑crafted "kangaroo tail" passage via
`adapter="trading_strategy"` → assert a `Setup` entity + `REQUIRES`(zone) edge + an
`EntryCondition` with a parseable `rule_ast` appear in the graph, and memories carry the trading
categories + `page_ref`.

**Phase 3 — Strategy synthesizer extension.** Implement `strategy_synthesizer` (cron + incremental
merge, grouped by `strategy_name`, versioned playbook pages, `DERIVED_FROM` citations, graph
patch). **Acceptance:** ingest two batches of lessons for one strategy across two runs → run #2
produces a **new playbook version** that contains run #1's lessons *plus* the new ones, source
memories unchanged; re‑running with no new memories is a **no‑op (skipped_unchanged)**.

**Phase 4 — Ingest Naked Forex end‑to‑end.** Fetch the PDF (owner will provide), extract text
page‑by‑page (respect the dependency order: philosophy → risk → back‑testing → **zones** → the six
setups → exits → psychology), ingest with the trading adapter, let the synthesizer build the
canonical playbooks. **Acceptance:** query the graph for "kangaroo tail" → get the full
setup with entry/stop/target/gate + citations; a `Naked Forex — Reversal` playbook exists and reads
as a coherent strategy guide.

**Phase 5 (Bellwether repo) — the compiler + validation writeback.** Build the graph→`Strategy`
compiler + slot‑filler in Bellwether; validate via the existing harness; write `Backtest` nodes
back. **Acceptance:** one Naked Forex setup compiles to a runnable `decide()`, runs through
`simulate_nautilus`, and its DSR/PBO land back on the `Strategy` node.

**Phase 6 — Composition/permutation.** `Ensemble` nodes + normalized `Signal` combination +
regime gating, validated as a family (PBO across the permutations). This is the owner's end goal.

---

## 7. Decisions (locked 2026‑07‑02) + residual risks

**Locked decisions — build to these:**
- **v1 scope = Phases 0–4** (adapter framework + trading adapter + strategy synthesizer + Naked
  Forex ingested & synthesized into canonical playbooks). Phases 5–6 (compiler, composition) come
  after a real book is fully in.
- **Compiler ownership = Bellwether.** The graph→runnable‑`Strategy` compiler lives in Bellwether
  (it depends on Bellwether's `Strategy`/`OrderIntent`); it reads the strategy graph via the
  Neuralscape MCP tools. NS stays domain‑agnostic; **the strategy graph is the interchange format**.
- **Custom graph types = per‑adapter, resolved per‑ingest** (Phase 1). A project may mix content, so
  types are chosen by the adapter on each `add_episode`, not fixed per‑project.
- **Synthesis grouping key = the canonical strategy name**, normalized via the `glossary` category +
  Graphiti entity dedup (so "kangaroo tail" ≈ "pin bar" collapse to one). Prefer the book's own name
  as canonical.

**Residual risks (engineering, not owner decisions):**
- **Rule fidelity vs. paywalled specifics.** Some pip offsets/stops in the research came from
  practitioner reproductions, not verbatim text. Ingesting the real PDF fixes this — keep
  `source_quote`/`page_ref` on every rule so discrepancies are auditable, and prefer the book's
  words over any secondary value.
- **Chunking granularity for executable rules.** Section‑aware chunking must keep a setup's
  entry+stop+exit together; validate on the Naked Forex setup chapters and tune.
- **Graphiti edge‑attribute gap.** Custom **edge attribute** population has a known gap (issue
  #1111) — verify on the pinned version during Phase 1 and fall back to entity attributes if needed.

---

## 8. Concrete seam checklist (files to touch in `neuralscape-service`)

| Phase | File | Change |
|---|---|---|
| 0 | `adapters/base.py` (new) | `KnowledgeAdapter`, `ADAPTER_REGISTRY`, `get_adapter`, `DEFAULT_ADAPTER` |
| 0 | `ingest/chunking_strategies.py` (new) | `ChunkingStrategy` protocol + registry; move `chunk_text` into `ParagraphAwareStrategy` |
| 0 | `ingest/extractors.py` (new) | `FactExtractor` protocol + registry; wrap current prompt as `DefaultExtractor` |
| 0 | `ingest/pipeline.py` | `IngestDoc.adapter`; resolve adapter; use its chunker/extractor |
| 0 | `schemas.py` | `IngestDocumentRequest.adapter`; keep envelope constants but allow adapter override of category set/vocabs |
| 0 | `mcp_server.py` | add `adapter` param to `ingest_document` + `ingest_text` |
| 1 | `config.py` | `get_mem0_config()` accepts/propagates entity/edge types |
| 1 | `mem0/mem0/memory/graphiti_memory.py` | `MemoryGraph.add(... entity_types, edge_types, edge_type_map)` → `add_episode` |
| 1 | `memory_service.py` | `enrich_graph(... adapter)` passes graph types down |
| 2 | `adapters/trading/{profile,ontology,extractor,chunking}.py` (new) | the trading adapter |
| 3 | `extensions/strategy_synthesizer/*` (new) | cron synthesizer (clone `wiki_synthesizer`) |
| 3 | `worker.py` | register `synthesize_strategy_playbooks_cron` on `GraphWorkerSettings` |

---

## Appendix A — Naked Forex reference (for the trading adapter's prompts/tests)

**The 3‑part gate (the law):** identify S/R **zones** → wait for price to reach a zone → trade only
when a **catalyst** prints on the zone. A catalyst off a zone is not a trade.

**Setups (each on a zone unless noted):**
- **Last Kiss** (Ch5, continuation): break of a zone/trendline → retest ("kiss") → enter in break
  direction; stop beyond the retested level; target next zone / runner exit.
- **Big Shadow** (Ch6, reversal): 2‑candle engulfing (outside bar) larger than prior ~5–10 candles,
  close near extreme; enter on break of its extreme (+5pips/H1+); stop opposite end; RR 1:2–1:3.
- **Wammies & Moolahs** (Ch7): double‑bottom with a *higher* low (Wammie, bullish) / double‑top
  with a *lower* high (Moolah, bearish); ≥6 (pref 20+) candles between touches; stop beyond the
  first touch; target next zone.
- **Kangaroo Tail** (Ch8, reversal): pin bar — open&close in same third, long opposite tail, range
  > last ~10, inside prior candle's range, room to the left, on a zone; stop beyond tail tip;
  75%‑to‑stop drawdown‑cut exit.
- **Big Belt** (Ch9): weekend/Monday‑gap fade at a zone; enter in fill direction; stop opposite
  extreme.
- **Trendy Kangaroo** (Ch10, continuation): a kangaroo tail inside a **trend's** 3–10‑candle pause
  (NOT at a reversal zone); enter with trend; runner (ladder/three‑bar) exits.

**Exits:** Zone (gunner), Split (scale‑out), Ladder (trail via successive zones, BE at first),
Three‑Bar (trail behind lowest‑low/highest‑high of last 3). **Risk:** ~1–2%/trade, RR 1:2–1:3,
size = risk$ ÷ stop‑pips ÷ pip‑value, back‑test before trading. **Timeframes:** H1+ (best H4/D1).
**Instruments:** spot FX (Big Belt is FX‑Monday‑specific); price‑action portable to stocks/indices/
crypto.

## Appendix B — Key references
- Graphiti custom entity/edge types + `add_episode` signature; bi‑temporal invalidation & dedup
  (getzep/graphiti docs & repo; Zep paper arXiv 2501.13956).
- Rule‑as‑AST / boolean tree (json‑rules‑engine; JBS GP paper); LEAN `Insight` signal schema;
  NautilusTrader config/behavior split + bracket orders; FIBO/FITS/GEAKG executable ontologies.
- Synthesis: RAPTOR (2401.18059), GraphRAG (2404.16130), Generative Agents reflection, mem0
  (2504.19413, use its ADD/UPDATE decision but Graphiti invalidation not deletion), anti‑drift
  (2308.15022, 2505.15291).
- Learn‑to‑execute: parse‑vs‑execute split (2412.04856), WALL‑E 2.0 (2504.15785), AlphaAgent AST
  complexity regularizer (2502.16789), Voyager skill library (2305.16291), CoALA (2309.02427);
  validation via Deflated Sharpe (Bailey/López de Prado) + CPCV (KBS 2024).

*(Full source list in the Bellwether research dossier that produced this plan.)*
