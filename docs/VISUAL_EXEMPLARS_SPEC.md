# Spec: Visual Setup Exemplars (chart images from trading books)

**Companion to** `TRADING_KNOWLEDGE_ADAPTER_PLAN.md`. Scope kept intentionally lean — enough to build.

## The idea
Trading books show **pictures of real setups** (an annotated kangaroo tail, a big shadow on a
zone). Those images carry information the prose doesn't. We want to:
1. **Understand** each setup image at ingest time and **store that understanding** (+ the image).
2. Let Bellwether's **chart‑vision agent reference stored exemplars at inference time** and treat a
   strong visual match as a **high‑confidence indicator** for the corresponding setup.

## Is this a Neuralscape task? Split of responsibilities
Yes — mostly, because **images are a data type NS doesn't natively store today** (NS holds text
memories + a Neo4j graph + Qdrant vectors). So NS grows a small "visual exemplar" capability; the
consumption side lives in Bellwether. The **same multimodal model** (Claude Opus 4.8 via the LLM
gateway) describes book exemplars *and* reads live charts — so both live in one shared "visual
vocabulary" and embedding space, which is what makes retrieval work.

| Concern | Owner |
|---|---|
| Store image bytes | **NS** (object store; see below) |
| Vision‑describe the exemplar → structured understanding | **NS ingest** (Opus 4.8) — reuses Bellwether's describer prompt style |
| `VisualExemplar` graph node + link to `Setup` + provenance | **NS** (trading adapter ontology) |
| Recall index (find exemplars similar to a live chart) | **NS** (Qdrant) |
| Retrieve exemplars + boost confidence during a live read | **Bellwether** (`ChartVisionAnalyzer`) |

## Data model (NS — extend the trading adapter)
Add one entity type + one edge to the trading ontology (`adapters/trading/ontology.py`):

- **`VisualExemplar`** — fields: `setup_name`, `direction` (bullish/bearish), `caption`,
  `visual_description` (the model's structured read: tail/body position, zone location, "room to the
  left", relative size — the *checkable visual features*), `key_levels` (annotated prices if
  legible), `chart_context` (timeframe/instrument if shown), `image_uri`, `page_ref`,
  `source_quote` (nearby caption text). Avoid Graphiti reserved field names.
- Edge **`EXEMPLIFIES`** (`VisualExemplar → Setup`) and reuse **`DERIVED_FROM`**
  (`VisualExemplar → source page/chunk`).

Store the **understanding as a normal NS memory** too (category `setup`, tagged
`visual_exemplar`) so it's recalled by the existing text pipeline — the memory body *is* the
`visual_description`, which means it embeds into Qdrant like everything else (this is the v1 recall
index — no new vector infra needed).

## Image storage (the "we need to store data like that" part)
NS memories are text; image **bytes** need a blob store. Recommendation:
- Put bytes in an **object store** (S3/MinIO in prod, a `vault/exemplars/` dir in dev). The memory
  model already has `source_ref.stored_path` — use it to hold the URI. Key by `content_hash`.
- No new NS schema needed beyond populating `stored_path`; add a tiny storage helper +
  config (`EXEMPLAR_STORE_URI`).

## Ingestion flow (NS trading adapter)
Per book image (extracted alongside the page text):
1. Save bytes → object store → `image_uri`.
2. Call **Opus 4.8 (multimodal)** with a structured "describe this trading‑setup image" prompt →
   `{setup_name, direction, visual_description, key_levels, caption}`. Use the **same visual
   feature vocabulary** as Bellwether's live chart read so descriptions are comparable.
3. Store a `VisualExemplar` node (`EXEMPLIFIES → Setup`, `DERIVED_FROM → page`) **and** a
   `setup`/`visual_exemplar` memory whose body is the description, with `source_ref.stored_path =
   image_uri` + `page_ref`.

> Multimodal note: `ingest_document` is text‑only today. Simplest path: do the vision‑describe step
> in the adapter's page extractor (out‑of‑band), then ingest the resulting **text description** the
> normal way with the image URI in `source_ref`. A dedicated `ingest_image` MCP tool is a later
> nicety, not required for v1.

## Retrieval + consumption (Bellwether `ChartVisionAnalyzer`)
When reading a live chart for a candidate setup:
1. **Retrieve** top‑k exemplars for that setup from NS (recall on `setup`+`visual_exemplar`,
   filtered by `setup_name`/`direction`).
2. **Few‑shot the read**: inject the exemplar image(s) (or their `visual_description`) into the
   multimodal prompt as labeled references ("here is a canonical kangaroo tail from the book").
3. **Boost on match**: if the model judges the live chart a strong match to an exemplar, raise the
   emitted `Signal.confidence`/`weight` and **cite the exemplar** (provenance → book page). Keep the
   boost bounded and auditable — a matched exemplar is *evidence*, still gated by the risk layer.

## Recall fidelity — v1 vs v2
- **v1 (recommended):** embed the **text descriptions** (reuse existing Qdrant text vectors).
  Because one model writes both the book‑exemplar and the live‑chart descriptions, text similarity
  is a good proxy for visual similarity. Zero new infra.
- **v2 (upgrade):** true **image embeddings** (a CLIP‑style multimodal encoder) in a **separate
  Qdrant collection**; compare a live‑chart image vector to exemplar vectors for genuine visual
  similarity. Add when v1's text‑proxy proves too coarse.

## Phased build
1. **NS:** object‑store helper + `stored_path` wiring; `VisualExemplar` type + `EXEMPLIFIES` edge.
2. **NS:** the vision‑describe step in the trading adapter's page extractor (Opus 4.8, shared
   prompt) → store exemplar node + memory + image.
3. **NS:** ingest Naked Forex's setup images → exemplars linked to their `Setup` nodes.
4. **Bellwether:** `ChartVisionAnalyzer` retrieves exemplars, few‑shots the read, applies a bounded
   confidence boost with citation.
5. **(Later)** v2 image‑embedding recall collection.

**Acceptance:** ingest a labeled kangaroo‑tail image → a `VisualExemplar` (`EXEMPLIFIES → Setup`)
with a stored `image_uri` + description exists; Bellwether recalls it for a live kangaroo‑tail read,
and a matched exemplar raises the signal's confidence with a citation back to the book page.

## Decisions (locked 2026‑07‑02)
- **Object store = local vault dir for now.** Write bytes to `vault/exemplars/<content_hash>.png`
  and set `source_ref.stored_path = file:///…/<hash>.png`. Zero new infra; swap to MinIO/S3 later
  **behind the same `stored_path` interface** (no call‑site changes). Add `EXEMPLAR_STORE_URI`
  config now so the swap is a one‑line change.
- **Boost policy = additive + bounded.** A strong exemplar match **raises** `Signal.confidence`/
  `weight` within a cap and **always cites** the book page; a match is **never required** for a
  setup to fire. It's corroborating evidence, still fully risk‑gated. (Pick the cap during Phase 4 —
  start small, e.g. a bounded bump, and tune against the eval harness.)
- **Recall = text‑proxy for v1.** Embed the exemplars' `visual_description` text (reuse the existing
  Qdrant text vectors); rely on the shared Opus‑4.8 describer so book‑exemplar and live‑chart
  descriptions land in the same space. **Defer true image embeddings (CLIP + a separate Qdrant
  collection) to v2**, added only if the text proxy proves too coarse.
