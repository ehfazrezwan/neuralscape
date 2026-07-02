# Knowledge Adapters — Implementation Notes (PR #92)

Companion to `22-knowledge-adapters.md`. That doc explains *how the system
works*; this one records *what was built, in what order, which decisions were
made at implementation time, what the reviews and live E2E caught, and why the
code looks the way it does*. Written at the end of the branch
(`feat/knowledge-adapters`, 7 commits, targeting `dev`).

Source specs: `docs/TRADING_KNOWLEDGE_ADAPTER_PLAN.md` (Phases 0–4 = v1) and
`docs/VISUAL_EXEMPLARS_SPEC.md`.

---

## 1. Build order and why it was safe

Each phase was independently green before the next started:

| Step | Content | Gate |
|---|---|---|
| Phase 0 | Adapter seam: registry + pluggable chunker/extractor, `adapter=` threaded end-to-end | Guardrail test: `default`-adapter ingest == pre-adapter output (chunk spans, envelope, extractor identity) |
| Phase 1 | Graph custom-types plumbing to `add_episode` | Toy-ontology test proves threading; default path forwards **zero** extra kwargs |
| Phase 2 | Trading adapter (taxonomy, ontology, chunker, extractor) | 25 unit tests incl. the REQUIRES gate + thin-edge assertions |
| Phase 3 | `strategy_synthesizer` extension + cron | create → idempotent-skip → versioned-accumulate tests |
| Exemplars | Object store + vision describe + harvest + endpoint | Degrade-path + dedup + endpoint tests |
| Review round | 12 findings fixed (see §5) | Full suite + live E2E (see §6) |

The ordering mattered for commit hygiene too: the commits are sequenced so each
is import-clean (e.g. the synthesizer extension lands *before* the worker
commit, because `GraphWorkerSettings`' class body resolves the synthesizer's
cron hours at import time).

## 2. Deviations from the plan (deliberate, with reasons)

**`KnowledgeAdapter` is a frozen dataclass, not a Pydantic model.** The plan
sketched `class KnowledgeAdapter(BaseModel)`. Rejected at implementation: the
profile holds `type[BaseModel]` values (Graphiti entity classes) and
tuple-keyed dicts (`edge_type_map: dict[tuple[str, str], list[str]]`) that
don't round-trip Pydantic validation — and never need to, because adapters are
declarative singletons built at import, never parsed from user input. Only the
adapter **name** (a string) crosses process/queue boundaries; each side
re-resolves the profile from `ADAPTER_REGISTRY`.

**Graph types are passed at `add_episode` call time, not via
`get_mem0_config()`.** The plan's seam checklist said to thread types through
`config.get_mem0_config()`. That would make the ontology a *store-level*
constant; the locked decision (§7 of the plan) is *per-adapter, resolved
per-ingest*, because one deployment mixes content. So the ontology rides the
call chain instead: `adapter.graph_ontology_kwargs()` →
`enrich_graph(graph_ontology=)` → `MemoryGraph.add(**kwargs)` →
`add_episode(**kwargs)`. `get_mem0_config()` is untouched.

**Taxonomy extension is additive mutation of the shared registries.**
`schemas.register_categories()` mutates `MEMORY_CATEGORIES` /
`GLOBAL_CATEGORIES` / `PROJECT_CATEGORIES` / `FLEXIBLE_CATEGORIES` in place at
adapter import. Alternative considered: per-adapter category namespaces with
validation context threaded everywhere — rejected as invasive surgery on
`store_raw`, every request validator, and the parser, for no isolation benefit
(categories are already namespaced *behaviorally*: trading categories are kept
out of `CATEGORY_VAULT_PATHS`, so the wiki synthesizer never touches them).
Consequences accepted: trading categories are valid for any write once
registered, and appear in `/v1/categories`.

**Trading categories are flexible-scope.** Not forced global or project —
scope follows the caller's `project_id`, same rule as `domain_knowledge`.
Trading knowledge is reference material; the deployment decides its scoping.

**Playbooks are keyed `(owner, strategy)`, not `(group_id, category)`.** The
wiki synthesizer's grouping key can't express "one canonical page per
strategy." The strategy tag (`strategy:<slug>`, set at upload) is the grouping
key; `owner_user_id` partitions multi-user instances. Facts without the tag
are skipped (they still exist as memories — they just can't be attributed to a
playbook).

## 3. The load-bearing design points

**The #1111 hedge shaped the ontology.** Graphiti had a known gap populating
custom *edge* attributes (issue #1111). So every compiler-facing datum —
`rule_ast`, `executable_expression`, anchors, offsets, `source_quote`,
`page_ref` — lives on **entity** attributes; the 16 edge classes are
*empty* marker models whose only payload is their docstring, with semantics
enforced by `EDGE_TYPE_MAP`. Live E2E later showed entity attributes populate
fully on our pinned subtree (see §6), so the hedge cost nothing and the risk
never materialized.

**The 3-part gate is structural, not prose.** `EDGE_TYPE_MAP[("Setup",
"SupportResistanceZone")] == ["REQUIRES"]` means Graphiti *cannot* classify a
Setup→Zone relationship as anything but the gate. The extraction prompt states
the rule; the type map enforces it.

**Deferred fact enrichment (the biggest post-review change).** Originally
ingested facts wrote to the graph inline on the ingest worker (inherited from
the PR #89 design, fine for typical docs). Book math breaks it: hundreds of
facts × ~minutes of Graphiti extraction each ≫ the 900s ingest `job_timeout`
→ ARQ kills + retries → Docling re-parses + Gemini re-extracts, forever. Now
`ingest_document` stores facts vector-fast and returns `graph_jobs` (one per
*newly created* fact — dedup hits produce none); the worker enqueues each onto
the graph queue with the adapter **name**; `process_graph_enrichment`
re-resolves the ontology. Two side-fixes fell out: the deferred path had been
dropping `source_ref` (so deferred facts lost their `(:Source)`/`DERIVED_FROM`
linkage — now threaded), and enqueue failure falls back to inline enrichment
(a lost enqueue would otherwise mean a permanently vector-only fact, since a
dedup'd retry never re-emits the job).

**Boundary-loud, worker-lenient adapter resolution.** A typo'd
`adapter="trading-strategy"` must not silently ingest a book *without* the
trading ontology — that's worse than an error, because the caller believes the
knowledge landed structured. So request schemas/endpoints 400/422 on unknown
names (`schemas.validate_adapter_name`, imported lazily to break the
`adapters ↔ schemas` cycle), while `get_adapter()` degrades to `default` only
for jobs already queued when an adapter was removed.

**Vision-describe degrades, never loses.** Exemplar images are stored
content-addressed *before* the describe call; a missing gateway, a rejected
image, or garbage JSON yields `described: False` with a minimal memory — the
description can be backfilled, the image cannot be un-lost.

**Exemplar idempotency keys on image bytes, not memory content.** The vision
description is nondeterministic, so `store_raw`'s content-hash dedup can't
make exemplar re-ingest idempotent. `find_existing_exemplar` pre-checks
`source_ref.external_id` (sha256 of the image bytes) + owner.

## 4. Bug classes worth remembering (found on this branch)

1. **Deterministic-key drift** (3rd occurrence in NS): ingest job-ids lacked
   `adapter`, and the file job-id lacked `visibility`. Any parameter that
   changes a write's *semantics* must be in every deterministic job-id and
   dedup key, or same-content requests coalesce onto a stale job's cached
   result. When adding a new semantic dimension, grep `_generate_job_id` and
   `_find_by_content_hash` call sites.
2. **Import-time side effects need eager entrypoint imports**: taxonomy
   registration ran only when something imported `ingest.pipeline` (lazy in
   every handler), so `MEMORY_CATEGORIES` had 13 or 25 entries depending on
   request history. Fixed with `import adapters` at the top of `main.py`,
   `mcp_server.py`, `worker.py`.
3. **Frozen snapshots of extensible registries**: `prompts.py` held a
   dict-comprehension copy of `MEMORY_CATEGORIES` from import time — unused,
   but stale-by-construction once the taxonomy became extensible. Deleted.
4. **Lazy `_memory` deref**: extensions doing direct Qdrant scrolls must call
   `service._get_memory()`, never `service._memory` (None until first use).
   Hit twice (synthesizer via E2E, exemplar lookup via Copilot).
5. **Single-page Qdrant scrolls lie**: the synthesizer's first scroll took one
   page with a misnamed "per-playbook" limit applied globally — silently
   dropping rules *and* making the idempotency check see a
   stable-but-incomplete source set. Paginate to exhaustion; cap per-group
   after grouping.

## 5. Review round (12 findings, all addressed)

Self-review + fixes, each with regression tests: (1) eager adapter imports;
(2) adapter+visibility in job-ids; (3) paginated synth scroll; (4) deferred
enrichment; (5) single Docling conversion for text+images; (6) 422 on unknown
adapter; (7) image-hash exemplar dedup; (8) owner-scoped
`GET /v1/ingest/exemplars/{image_id}` (resolution goes *through the caller's
memory*, so foreign ids 404 without probing the store); (9) synthesizer
visibility caveat documented; (10) guarded built-in adapter import (a broken
adapter degrades to "unavailable" instead of killing default ingest — pipeline
imports `adapters` at module top); (11) search category-filter cap raised;
(12) stale scope snapshot removed.

Copilot round: sha256 digest for the image-harvest dedup (was a 64-byte prefix
— distinct same-format charts could collide), the docs' stale
"degrades-to-default" bullet, and the `_find_exemplar_point` lazy-init deref.

## 6. Live E2E (isolated stack) — what it proved and what it caught

Method: second compose project (`-p nstest`) with an override file — `!reset`
to empty published ports, `!override` to replace them (the tags are not
interchangeable), **named volumes replacing the vault bind-mount** (otherwise
the test synthesizer writes into the live vault — the `-p` isolation only
covers named volumes), heavy/outward services excluded, `down -v` teardown.

Proved, against real Qdrant/Neo4j/Redis/Gemini:

- 422 on adapter typo at the boundary; 202 + `graph_jobs_enqueued: 9` for a
  2-chapter trading doc; all 9 deferred jobs enriched on the graph worker.
- Neo4j: `Setup`/`RuleNode`/`StopLoss`/`SupportResistanceZone`/
  `EntryCondition`/… labels; edge names `REQUIRES`/`HAS_ENTRY`/`HAS_STOP`/
  `HAS_CHILD`/`CONSTRAINED_BY`; the `(Kangaroo Tail)-[REQUIRES]->(zone)` gate;
  and **fully populated entity attributes** (`order_type='buy stop'`,
  `offset='5 pips + spread'`,
  `executable_expression='buy_stop = candle.high + offset_pips(5) + spread'`)
  — closing the #1111 question on our pinned subtree.
- Playbook v1 synthesized from 9 rules with every section + inline citations;
  second run `skipped_unchanged`; passages dedup exactly on re-ingest.
- Exemplar path: content-addressed store, describe-degrade (the gateway
  rejected deliberately-invalid image bytes; nothing was lost), image-hash
  dedup, owner-scoped HTTP download round-trip.

Caught (bugs no unit test could see):

1. **Dockerfile never COPY'd `adapters/`** — every stage lists source dirs
   explicitly; new top-level packages must be added to builder+runtime+test.
2. **Docling inlines figures into `md_content`** as
   `![Image](data:image/png;base64,…)` with `image_export_mode=embedded` — no
   `pictures[]` array on our build. Two fixes: the harvester regex-scans
   data-URIs *inside* strings, and the Markdown is stripped of base64 before
   chunking (a picture-heavy book would otherwise flood passages with
   megabytes of image bytes).
3. **`service._memory` lazy-init crash** on manual synthesizer invocation.

## 7. Known limitations (deliberate, documented)

- **Fact rewording on forced re-ingest**: identical content re-extracted by
  Gemini produces reworded facts that bypass content-hash dedup. Layered
  mitigation: job-id coalescing catches the accidental case; the 6-hour
  semantic dedup cron (0.95 cosine) collapses the rest. A forced re-ingest
  still near-duplicates rules until the cron runs.
- **Private memories reach vault playbooks** (per-owner folders, one volume).
  Single-user OK; filter to shared or split the vault before multi-user.
- **`page_ref` is None for md-inline images** (no `prov` container to read).
- **v1 exemplar recall is a text proxy**; CLIP image embeddings deferred.
- **Naked Forex ingest itself** awaits the PDF; Phases 5–6 (compiler,
  ensembles) live in Bellwether. The strategy graph is the interchange format.

## 8. Runbook: ingesting a trading book

```bash
# Flags (all dark by default): STRATEGY_SYNTHESIZER_ENABLED,
# EXEMPLAR_STORE_ENABLED, DOCLING_EXTRACT_IMAGES (+ LLM gateway for vision).
curl -F 'files=@naked-forex.pdf' \
     -F 'adapter=trading_strategy' \
     -F 'tags=strategy:naked-forex-reversal' \
     -F 'visibility=shared' \
     "$NS/v1/ingest/files"
# Poll /v1/memories/status/{task_id}; facts enrich on the graph queue;
# the synthesizer cron (graph worker, :55) builds/refreshes the playbook.
```
