# Phase D: Deterministic Knowledge System Router

**Status:** Phase D (the resolver) is built and current. The Phase E fusion composition and the Phase G persistence described below as "later" have since landed on dev; this page was updated to say so. Current behavior lives in `knowledge/router.py`, `knowledge/index_store.py` and `knowledge/fusion.py`.  
**See also:** `ICE_V2_KNOWLEDGE_SYSTEMS_PLAN.md` §4, `ICE_V2_DECISIONS.md` #3

## What This Is

The router (`knowledge/router.py`) is a three-layer deterministic resolver that decides which `KnowledgeSystem`(s) to query for a given request. It operates on registry entries via `eligible_systems` and **never branches on `transport`** (per DECISIONS.md cross-cutting rule).

Resolution order (first hit wins):

1. **Explicit override**: optional `knowledge_system` param (additive to all recall/code tools); existing `graph_id` ref-shape dispatch (`repo:`/`code--`/`.json`) subsumed as an explicit code signal.
2. **Project config default**: per-project settings document `{code_systems: [...], fuse_code_into_recall: bool (DEFAULT TRUE per decision #3), default_engine: str}`. Editable via REST/MCP; set at index time.
3. **Deterministic signals** (only for generic `recall_memories`/`ask_memory` when layer 2 permits code fusion):
   - Project has ≥1 healthy indexed code_space (necessary condition — cheap registry lookup).
   - Query-shape gate for the EXTRA code leg: FQN-ish token (`\w+\.\w+\(`, `::`), path-like token (`\w+/\w+\.\w{1,4}`), backticked snake_case/CamelCase identifier, or structural keywords (who calls / where is / defined / imports / blast radius).
   - Plain-prose recall on a code project does NOT fan out to code (recall precision + latency floor).
   - Ambiguity → base-only (additive: code is an enrichment leg, never a replacement).

## Router Overhead

Budget: <1 ms (measured in `tests/test_router.py::test_router_overhead`). No LLM, no network, only table lookups and cheap regex checks.

## Phase D vs Phase E

**Phase D** is the resolver: it picks the route and records the decision. It is still what decides whether a recall gets a code leg, via `RouteDecision.wants_code_fusion` and `RouteDecision.code_system_names`. When the decision is base-only (plain prose, no indexed code, `fuse_code_into_recall=False`), generic recall output is unchanged from the plain memory path.

**Phase E (fusion) has landed.** When the router decides a code leg is warranted, `recall_memories` in `mcp_server.py` runs the code leg next to the base legs and `knowledge/fusion.py` composes sections (`[structure]`, `[semantics]`, `[memory]`) instead of interleaving scores. So generic recall on a code project is no longer byte-identical to the plain memory path when the coding-signal gate passes; it is byte-identical only when the gate does not pass. A project can also set `default_engine="auto"` (or a caller can pass `knowledge_system="auto"`) to let `knowledge/autoroute.py` pick the engine per operation.

## Project Config

Per-project knowledge routing settings. The config is **persisted, not in-memory only**: `set_project_config` in `knowledge/router.py` writes through `knowledge/index_store.py` (`save_project_config`), which stores it in Redis under a per-project key and keeps a process-local copy. `get_project_config` consults the in-process mirror (`_PROJECT_CONFIGS` in `router.py`) first and falls back to `load_project_config` on a miss, so a config set by the ingest worker at index time becomes visible to the API process on its first read. Redis is best-effort: with Redis down (unit tests, outages) the store degrades to the in-memory copy. A miss returns `None`, which keeps recall on the base path. Known limit as of this writing: once a project's config is cached in a long-lived process, a rewrite made by a different process is not guaranteed to be seen until that process restarts; a bounded read-through re-read for the mirror is in review (PR #548), and `knowledge/router.py` is the source of truth for whichever behavior has merged.

```python
@dataclass
class ProjectKnowledgeConfig:
    project_id: str
    code_systems: list[str] = []          # ["code-cbm"] or ["code-graphify"] etc.
    fuse_code_into_recall: bool = True    # DEFAULT TRUE per decision #3
    default_engine: str | None = None     # "code-cbm", "code-graphify" or "auto"
    code_space: str | None = None         # code_space indexed for this project (set at index time)
    op_preference: dict[str, list[str]] | None = None  # per-op engine preference override
```

### REST Endpoints

- `GET /v1/projects/{project_id}/knowledge-config` — get current config (returns default if not set)
- `PUT /v1/projects/{project_id}/knowledge-config` — set/update config (admin-only: 403 otherwise. Admin = the listed `DICTATOR_USER_IDS`; when that list is empty, the default user)

### MCP Tools

- `get_project_knowledge_config(project_id)` — get current config
- `set_project_knowledge_config(project_id, code_systems?, fuse_code_into_recall?, default_engine?)` — set/update config (dictator-only, same gate as the PUT; returns a JSON `error` otherwise)

## Decision #3 Rationale (fusion ON by default)

Per `ICE_V2_DECISIONS.md` #3: `fuse_code_into_recall` defaults **ON**, BUT the code leg fires ONLY when the deterministic coding-signal gate passes (code project + code-shaped signal, or a code MCP tool). Plain-prose recall on a code project stays base-only.

This preserves the +20% latency floor (Phase-D gate still binding) while enabling code enrichment for coding-shaped queries by default. No per-project opt-in required to get fusion; opt-OUT is available by setting `fuse_code_into_recall=False`.

## Routing Table (reviewable logic)

See `tests/test_router.py` for exhaustive unit tests. Each test asserts: systems resolved, rationale, layer hit.

### Layer 1 Examples

- `knowledge_system="code-cbm"` → `[code-cbm]` (layer 1)
- `graph_id="repo:myrepo"` → `[code-native]` (layer 1)
- `graph_id="code--user--myrepo"` → `[code-native]` (layer 1)
- Unknown/unhealthy/capability-mismatch → fallback to `[ns-memory]`

### Layer 2 Examples

- Code tool + project has `code_systems=["code-cbm"]` → `[code-cbm]` (layer 2)
- Generic recall + project has code + `fuse_code_into_recall=True` + coding signal → `[ns-memory]` plus the project's code system(s) as the fusion leg (`wants_code_fusion=True`) (layer 2)
- Generic recall + `fuse_code_into_recall=False` → `[ns-memory]` (layer 2)

### Layer 3 Examples

- Generic recall + project has code + coding signal detected → `[ns-memory]` plus the code leg (`wants_code_fusion=True`) (layer 3)
- Generic recall + plain prose (no coding signal) → `[ns-memory]` (layer 3)
- Generic recall + project has no code → `[ns-memory]` (layer 3, default)

## Coding Signal Detection

Query-shape gate (layer 3 only):

- **FQN-ish tokens:** `foo.bar()`, `foo::bar`
- **Path-like tokens:** `src/utils/helper.py`, `lib/foo/bar.ts`
- **Backticked identifiers:** `` `some_function` ``, `` `ClassName` ``
- **Structural keywords:** "who calls", "what calls", "where is", "defined", "definition", "imports", "blast radius", "impact", "callers", "callees", "dependencies", "dependents"

Plain prose (no coding signal): "What were the design decisions?", "Tell me about the architecture", "Why did we choose this approach?"

## Files

- `neuralscape-service/knowledge/router.py` — the router
- `neuralscape-service/tests/test_router.py` — exhaustive unit tests
- `neuralscape-service/main.py` — REST endpoints for project config
- `neuralscape-service/knowledge/index_store.py` — Redis-backed persistence for project config and code-index metadata
- `neuralscape-service/knowledge/fusion.py` — section composer used when the router asks for a code leg
- `neuralscape-service/mcp_server.py` — MCP tools wired to the router; `recall_memories` acts on the decision (code leg + fusion)
