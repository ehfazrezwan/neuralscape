# F2 — Native Code-Intel Engine (DRAFT v1, 2026-07-05)

Builds on the 2026-07-04 decision: absorb codebase-memory-mcp's *capabilities*
and graphify's *semantics* onto NS-native storage instead of wrapping either
binary. This spec turns that decision + the F1 seam map into an implementable
plan.

## The division of labor (the product requirement)

When Neuralscape is used for coding:

- **Structure** — "who calls X", "what imports Y", "map this architecture",
  "where is the symbol" — is answered by the **code-intel engine** (derived
  index, rebuilt on reindex, never trusted as memory).
- **Semantics** — "why is it this way", "what did we decide", "what's the
  gotcha here", "how does this subsystem work" — is answered by **NS native
  storage** (recall over decision/architecture/rationale memories).
- The two meet at **anchors**, so a structure answer carries its attached
  semantics and a semantic memory points at real code entities.

The plugin already enforces the split at the prompt layer (session-start
`CODE_GRAPH` probe, the "defer to the code graph" disclosure footer,
compile-observations rejecting structural facts). F2 makes the server side
real.

## What exists (F1) and what it fixes for us

The stable contract F2 must keep (everything downstream depends on these four
invariants — worker, plugin probe, e2e script, three test suites):

1. Tool surface: `query_code_graph` / `get_code_neighbors` / `code_path`
   (MCP `mcp_server.py:1122-1189` + REST twins `main.py:1241-1319`), text-out.
2. Availability gate degrading to 501 / error-JSON (`adapters/code_graph/__init__.py:34-54`).
3. Owner-scoped `graph_id → find_artifact` resolution (`query.py:59-87`).
4. `source_ref` retrieval handles resolving through NS's own surface
   (`ingest/code_graph.py:74-101`).

What F1 lacks: **no engine protocol** (`query.py` reaches into graphify's
private serve helpers), no `locate`, no blast radius, no incremental index, no
memory↔code bridge, and the graph is a static `graph.json` artifact.

## Architecture

### 1. `CodeIntelEngine` protocol (net-new — `adapters/code_graph/engine.py`)

```python
class CodeIntelEngine(Protocol):
    def query(self, question, *, mode, depth, token_budget) -> str: ...
    def neighbors(self, label, *, relation_filter) -> str: ...
    def path(self, source, target, *, max_hops) -> str: ...
    # F2 additions
    def locate(self, query, *, k) -> list[LocateHit]:          # hybrid graph+dense+BM25
    def detect_changes(self, since) -> ChangeReport: ...        # real blast-radius BFS
    def semantic_layer(self) -> list[SemanticFact]: ...         # ports semantic.py
    def index(self, source, *, incremental=True) -> IndexReport: ...
    def export_snapshot(self) -> bytes: ...                     # index-in-CI artifact
```

Two implementations, selected per graph ref:

- **`GraphifyJsonEngine`** — wraps today's `query.py`/`semantic.py` verbatim.
  Stays the default for `.json` artifact graph_ids during transition; `index`,
  `detect_changes`, `locate` raise NotSupported.
- **`NativeEngine`** — the F2 engine. Selected when the graph ref is an
  NS-indexed repo (`repo:<name>` refs) rather than a JSON artifact.

The three existing tools route through the protocol → zero contract change.
`locate` and `code_impact` become new MCP tools + REST twins, registered under
the same availability gate (native engine needs no extra, so on F2-enabled
deployments the gate is simply "engine configured").

### 2. Native storage — own label-space in the shared Neo4j

- Same driver, same bridge-dispatch discipline (`_run_on_bridge`,
  `memory_service.py:729-760`); the extension-owned-label-space template is
  `extensions/dreaming/graph_patcher.py` (raw MERGE with deadlock retries).
- Labels prefixed and disjoint from Graphiti's: `(:CodeRepo)`, `(:CodeFile)`,
  `(:CodeSymbol {fqn, kind, file, span, degree, community_id})`,
  edges `CALLS | IMPORTS | DEFINES | INHERITS | REFERENCES` with an
  `extraction` property (`extracted|inferred|ambiguous`) mapped to the A1
  epistemic levels exactly as `semantic.py:_confidence_mapping` does today.
- Partition key `code_space = "code--{owner}--{repo}"` on every node — this IS
  the **code workspace** (WORKSPACES_SPEC.md): never enumerated by dreaming,
  never searched by recall, torn down and rebuilt on reindex.
- Real openCypher with server-side timeouts replaces CBM's param-swallowing
  lexer; an optional `code_cypher` read-only tool (dictator/ops-gated) exposes
  it.

### 3. Anchors — the memory↔code bridge (survives reindex)

- `(:CodeAnchor {repo, fqn})` — keyed on repo + fully-qualified symbol name,
  **never node UUIDs**. Reindex deletes/rebuilds `CodeSymbol` nodes and
  re-attaches them to existing anchors (`(:CodeSymbol)-[:ANCHORED]->(:CodeAnchor)`).
- Memories about code carry `source_ref{connector_type:"code_graph",
  external_id:"<repo>::<fqn>"}`; the existing dreaming `graph_patcher`
  Source-node mechanism links them — a decision memory holds a real edge to the
  function it concerns.
- **Enrichment both ways**:
  - code→answer: `query`/`neighbors`/`locate` responses append attached
    memories (decisions/gotchas/bugfix history) found via the anchor — this is
    "NS native storage provides the semantics" made concrete: one tool call
    returns structure *plus* the why.
  - code→memory: `detect_changes` on reindex emits liveness events (symbol
    deleted / signature rewritten) → anchored memories get flagged for the
    dreaming sweep's `temporal_reframe` — invalidation grounded in code ground
    truth. (The `"ambiguous"` tag plumbing already exists but is unconsumed —
    this is its consumer.)

### 4. Indexer

- **tree-sitter via `tree-sitter-language-pack`**, curated set first: Python,
  TS/JS, Go, Rust, Java. Per-language `.scm` query files for symbol/edge
  extraction; heuristic FQN resolution v1 (LSP-grade later).
- Runs on the **ingest queue** (bulk work never starves fast paths — same
  isolation argument as document ingest). Incremental by file content-hash;
  full rebuild is just re-index with the same anchors.
- **Louvain communities computed at index time and persisted** on
  `community_id` (closes CBM's never-persisted-communities gap); god-node
  degree and surprise scores persisted likewise, so `semantic_layer()` (the
  ported `semantic.py` distillation: community/hotspot/boundary/rationale
  facts) runs off stored properties, not a per-call NetworkX pass.
- Semantic-layer facts keep going where F1 puts them: **project memory
  workspace** (they are curated, stable knowledge *about* the code and should
  surface in recall) — with the liveness flags above keeping them honest.

### 5. `locate` — hybrid code retrieval

- Symbol cards (name + signature + docstring + first lines) embedded through
  the existing embedding pipeline into a **separate Qdrant collection
  `code_index`** (code workspace ⇒ never mixed into memory recall), plus BM25
  and graph-degree signals; returns `file:line` hits with anchor ids.
- Serves both the MCP tool and the plugin's grep-steering PreToolUse hook
  (steer-never-block, <100 ms budget, silent skip — per the WT6 Read Gate
  lesson).

### 6. Absorbed-capability checklist

| from | capability | F2 form |
|---|---|---|
| CBM | snapshot artifact / index-in-CI | `export_snapshot()` → content-addressed ingest artifact; CI indexes, deployment imports |
| CBM | detect_changes | real blast-radius BFS over CALLS/IMPORTS (CBM's depth param was a no-op) |
| CBM | openCypher | real Neo4j + timeouts (CBM's lexer silently swallowed $params) |
| CBM | ADR management | one-way ADR→decision-memory sync (CBM's was a whole-doc blob upsert) |
| CBM | graph.db.zst reader | optional one-shot migration importer |
| graphify | semantic distillation | `semantic.py` ported onto persisted properties |
| graphify | confidence tags | `extraction` property → epistemic levels (same mapping) |
| graphify | graph.json path | `GraphifyJsonEngine` kept as the artifact-file engine |

## Sequencing

- **E1** — engine protocol + `GraphifyJsonEngine` wrapper. Pure refactor, zero
  behavior change, lands the seam. (Small; can ship independently.)
- **E2** — native indexer + Neo4j label-space + parity on the 3 existing tools
  for one language (Python), `repo:` graph refs.
- **E3** — `locate` + `code_index` embeddings + language set expansion.
- **E4** — anchors + memory enrichment both ways (the product payoff).
- **E5** — `detect_changes` + dreaming liveness consumer.
- **E6** — snapshot export / CI flow + CBM migration reader.

Prereq per the 2026-07-04 decision: sequenced after the memory_service
refactor. E1 has no such dependency and can go first.

## Non-goals (v1)

- 158 languages; LSP-grade resolution; watching filesystems (index is
  pull/CI-driven); exposing raw Cypher to normal users; mirroring the raw code
  graph into Graphiti (F1's hard rule stands).
