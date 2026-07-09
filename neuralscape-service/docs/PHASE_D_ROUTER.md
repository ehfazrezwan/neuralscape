# Phase D: Deterministic Knowledge System Router

**Status:** Phase D complete (resolve only; fusion composition is Phase E)  
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

**Phase D (this phase):** Resolves the route and logs the decision. Generic recall output is **byte-identical to today** unless `knowledge_system` is explicitly given. This preserves the latency floor and keeps D/E separable.

**Phase E (fusion):** Uses the route decision to compose section-based fusion responses (structure + semantics + memory). Generic recall will then include code legs when the router decided they're warranted.

## Project Config

Per-project knowledge routing settings (stored in-memory in Phase D; Phase E will persist to Redis/Neo4j):

```python
@dataclass
class ProjectKnowledgeConfig:
    project_id: str
    code_systems: list[str] = []          # ["code-cbm"] or ["code-graphify"] etc.
    fuse_code_into_recall: bool = True    # DEFAULT TRUE per decision #3
    default_engine: str | None = None     # "code-cbm" or "code-graphify"
```

### REST Endpoints

- `GET /v1/projects/{project_id}/knowledge-config` — get current config (returns default if not set)
- `PUT /v1/projects/{project_id}/knowledge-config` — set/update config

### MCP Tools

- `get_project_knowledge_config(project_id)` — get current config
- `set_project_knowledge_config(project_id, code_systems?, fuse_code_into_recall?, default_engine?)` — set/update config

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
- Generic recall + project has code + `fuse_code_into_recall=True` + coding signal → `[ns-memory]` (Phase D; Phase E will add code leg) (layer 2)
- Generic recall + `fuse_code_into_recall=False` → `[ns-memory]` (layer 2)

### Layer 3 Examples

- Generic recall + project has code + coding signal detected → `[ns-memory]` (Phase D; Phase E will add code leg) (layer 3)
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
- `neuralscape-service/mcp_server.py` — MCP tools wired to router (Phase D: logs decision, doesn't change output)
