# Neuralscape Pluggable-Engine Architecture: Ports, Default Adapters, and Per-Use-Case Routing

*Design doc. Builds on doc 13 (coupling), doc 14 (version-drift), doc 15 (feature parity). Goal: turn Neuralscape from "is a mem0 instance that holds a graphiti instance" into an orchestration layer that drives **any** vector store and **any** temporal graph store — with mem0 and graphiti as the **default, first-class** adapters, not hard dependencies — and with the ability to **route different use-cases to different engines.***

---

## 1. Design goals (in priority order)

1. **Opinionated defaults, pluggable internals.** Out of the box, NS *is* mem0+graphiti+Qdrant+Neo4j and behaves exactly as today. Pluggability is an internal seam, not a config burden users must understand.
2. **Reuse the frontier engineering — swap the DATABASE, not the ENGINE.** mem0 stays the vector orchestrator; graphiti stays the temporal-graph engine. Their own factories (`VectorStoreFactory` 23 backends, `GraphDriver` 4 backends) do the backend construction. NS does not reimplement vector math or graph traversal.
3. **A true bring-your-own contract.** A third party can implement an abstract base class, register it, and run NS on an engine NS has never heard of — *provided* they satisfy the port contract and declare their capabilities.
4. **Per-use-case engine selection.** A single deployment can route, e.g., coding-agent memory to one graph engine and general-workflow memory to another, chosen per request. *(New requirement — §6.)*
5. **Graceful capability degradation.** Backends differ (doc 15 §4): not all support BM25, cross-encoder, typed ontology, or `STARTS WITH`. Ports advertise capabilities; the orchestrator adapts instead of crashing.

**Non-goal:** making *arbitrary* third-party engines trivially integrable. Realistically, a new backend needs an adapter author who understands both that backend and the port contract. The ABC makes the contract explicit and testable; it does not make integration zero-effort.

---

## 2. The core seam: two Ports + a Registry + a Router

Today `MemoryService` holds one `mem0.Memory` and pulls `graphiti` + the private `_bridge` off `_memory.graph` (`memory_service.py:295,339-341`). The re-attach shim (`:311-337`) only exists to compensate for mem0 2.x dropping auto graph-init. **The Port split deletes that shim entirely.**

```
                         ┌──────────────────────────────────────────┐
                         │              MemoryService                │   ← engine-agnostic
                         │  (extraction, taxonomy, group_id algebra, │     business logic
                         │   merge, dedup policy, scopes)            │
                         └───────────────┬──────────────────────────┘
                                         │ asks the Router for ports
                              ┌──────────▼───────────┐
                              │     EngineRouter      │  picks (VectorPort, GraphPort)
                              │  by request Profile   │  per request  (§6)
                              └──────────┬───────────┘
                                         │ looks up in
                              ┌──────────▼───────────┐
                              │    EngineRegistry     │  name → adapter factory
                              └──────┬─────────┬──────┘
                       ┌─────────────┘         └─────────────┐
            ┌──────────▼──────────┐            ┌─────────────▼─────────────┐
            │   VectorPort (ABC)  │            │      GraphPort (ABC)       │
            └──────────┬──────────┘            └─────────────┬─────────────┘
       ┌───────────────┼───────────────┐         ┌───────────┼─────────────────┐
 Mem0VectorAdapter  PgvectorAdapter  <BYO>   GraphitiAdapter  FalkorAdapter   <BYO>
 (default, wraps    (delegates to            (default, wraps  (delegates to
  mem0 +            mem0's factory)           graphiti +       graphiti's
  VectorStoreFactory)                         GraphDriver)     driver roster)
```

Two seams, derived from the *real* call sites in doc 15 / the seam-map (not invented):

### 2.1 `VectorPort` (ABC) — minimum method set

Every method below exists because NS calls it today (citations are the originating sites):

```python
class VectorPort(ABC):
    # identity / capabilities
    @property
    @abstractmethod
    def capabilities(self) -> "VectorCapabilities": ...
    embedding_dims: int            # port owns this, not config (config.py:335,369,394)
    collection: str

    # embedding (mem0 embedding_model today)
    def embed(self, text, action) -> list[float]: ...           # :659,788,1392,2856
    def embed_batch(self, texts, action) -> list[list[float]]: ...# :1092

    # write
    def insert(self, ids, vectors, payloads) -> None: ...        # :700,1095
    def update(self, id, content=None, payload=None) -> None: ...# :1989  (closes mem0 gap)
    def delete(self, id) -> None: ...                            # :2094,2179,2746
    def delete_all(self, scope: Scope) -> None: ...             # :2069

    # read
    def get(self, id) -> Memory | None: ...                      # :1861,1986,2021
    def query(self, vector|query, *, filter: PortFilter,        # :811,1420 (raw query_points)
              limit, with_payload=True) -> list[Hit]: ...        #  + :1197,2862 (mem0 search)
    def scroll(self, *, filter: PortFilter, limit, offset,       # :857,968,1225,2225,2725,2780
               with_payload, with_vectors) -> tuple[list, Cursor]: ...

    # provenance (optional — see capabilities.history)
    def add_history(self, id, prev, content, op, at) -> None: ...# :707,1104
```

`PortFilter` is an **engine-neutral filter AST** (equality, `in`, is-null, OR-of-conditions — the exact operators NS uses at `:790-807,1175-1183,1396-1418,2208-2219`). Each adapter compiles it to its backend's dialect (Qdrant `Filter`, pgvector SQL `WHERE`, …). This is what frees NS from importing `qdrant_client.models` everywhere.

### 2.2 `GraphPort` (ABC) — minimum method set

```python
class GraphPort(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> "GraphCapabilities": ...

    def add(self, data, *, group_id, user_id,                    # :544,719,2006
            entity_types=None, edge_types=None) -> None: ...     #  (wires doc15 gap #2)
    def search(self, query, *, group_ids, limit,                 # :1565,1695,2626; main.py:572
               recipe: GraphRecipe = DEFAULT) -> GraphResults: ...
    def get_nodes(self, group_ids, limit) -> list[Node]: ...     # :2264; main.py:402
    def get_edges(self, group_ids, limit) -> list[Edge]: ...     # :2296,2692; main.py:440
    def get_episodes(self, group_ids, reference_time, last_n): ...# :2330; main.py:482
    def get_communities(self, group_ids, limit): ...             # :2366; main.py:524
    def delete_episode(self, uuid) -> None: ...                  # :2390 (raw Cypher today)
    def expire_edges_matching(self, query, group_ids, substr): ...# :2625-2636
    def expire_all_edges(self, group_ids) -> None: ...           # :2690-2698
    def distinct_project_ids(self, user_prefix, shared_prefix): ...# :1930-1949 (raw Cypher)
    def attach_memory_id(self, group_id, memory_id, vis, owner, window): ...# graph_patcher
    def fetch_backrefs(self, uuids) -> dict: ...                 # :1621-1639 (raw Cypher)
    def close(self) -> None: ...                                 # :448; main.py:170
```

The three raw-Cypher operations (`distinct_project_ids`, `fetch_backrefs`, `attach_memory_id`) become **named port methods** — each adapter implements them in its own dialect (Neo4j keeps today's Cypher; FalkorDB/Kuzu/Neptune supply their own). This is the single change that retires the Neo4j hard-binding from doc 15 §5.

**The `group_id` tenancy algebra** (`_build_group_id`/`_get_group_ids`, `memory_service.py:206-259`) stays in `MemoryService` (it is NS policy, backend-agnostic) but is passed *into* the port — adapters never parse group-id strings, they only filter by them.

### 2.3 Capability negotiation (why doc 15 §4 can't just be "turned on")

```python
@dataclass(frozen=True)
class GraphCapabilities:
    typed_ontology: bool        # graphiti: yes
    cross_encoder_rerank: bool  # graphiti: yes (off today)
    communities: bool
    sagas: bool
    temporal_invalidation: bool
    recipes: frozenset[GraphRecipe]

@dataclass(frozen=True)
class VectorCapabilities:
    bm25_hybrid: bool           # mem0: yes (unused today)
    entity_boost: bool
    rerank: bool
    filter_ops: frozenset[FilterOp]   # eq/in/isnull/or/... per backend
    history: bool               # mem0: yes; many backends: no
```

`MemoryService` checks capabilities before using an optional feature (e.g. "rerank if `cap.rerank` else return raw order"). This is how the same orchestration code runs on a feature-rich default and a minimal BYO backend without branching everywhere.

---

## 3. Default adapters (the opinionated path)

- **`Mem0VectorAdapter`** wraps `mem0.Memory` and delegates *construction* to mem0's `VectorStoreFactory` — so all 23 mem0 vector stores are reachable by config, and Qdrant stays the default. It also exposes the richer `query`/`scroll` primitives NS needs (today's raw-client leaks) as **first-class adapter methods**, so `MemoryService` never touches `QdrantClient` again. Capabilities advertise mem0's BM25/entity-boost/rerank as *available* (wiring them on is then a `MemoryService` policy choice — doc 15 §4 #3).
- **`GraphitiAdapter`** wraps a `Graphiti` instance constructed directly (dropping the mem0 graph-store shim and the `graphiti_memory.py` impersonation — see doc 14 §8). It delegates backend construction to graphiti's `GraphDriver` roster (Neo4j default; FalkorDB viable as of 0.29.2 per doc 14 §3). The three raw-Cypher methods live here for the Neo4j case.

**Migration is mechanical and behavior-preserving:** every current call site already maps 1:1 to a port method (the citations in §2 are those call sites). Phase 1 is "introduce the ports, point them at mem0/graphiti, change call sites to go through the port" — no behavior change, no new backend yet.

---

## 4. The bring-your-own contract

A third party ships a package that:
1. Subclasses `VectorPort` and/or `GraphPort`.
2. Implements every abstract method + a truthful `capabilities` property.
3. Registers via entry point or config: `EngineRegistry.register("mygraph", MyGraphAdapter)`.
4. Passes the **port conformance test suite** NS ships (one parametrized pytest module that runs the same scenarios against any adapter — the contract is the test).

What we promise BYO authors: a stable port ABC + filter AST + capability flags + a conformance suite. What we *don't* promise: that an arbitrary engine maps cleanly — temporal-graph semantics in particular (bi-temporal validity, contradiction-invalidation) are graphiti's hard-won value, and a BYO graph engine that lacks them advertises `temporal_invalidation=False` and loses that feature. That trade-off is explicit, not hidden.

---

## 5. Configuration model

Today: flat literals (`config.py:402` `"provider":"qdrant"`, `:406` graph, `:335/369/394` dims). Proposed: **named engine definitions** + a default selection.

```yaml
engines:
  vector:
    default:   { adapter: mem0, store: qdrant, embedder: gemini, dims: 768 }
    pg:        { adapter: mem0, store: pgvector, embedder: gemini, dims: 768 }
  graph:
    neo4j:     { adapter: graphiti, driver: neo4j }      # default
    falkor:    { adapter: graphiti, driver: falkordb }
    codegraph: { adapter: mygraph, dsn: ... }            # BYO

defaults: { vector: default, graph: neo4j }
```

Backward-compatible: absent `engines:`, NS synthesizes the `default`/`neo4j` definitions from today's env vars. Existing deployments need **zero** config change.

---

## 6. Per-use-case engine routing (the new requirement)

> *"Use a different graph engine for coding-agent use cases vs. normal non-coding-heavy workflows."*

A **Profile** is a named routing target attached to a request; the **EngineRouter** maps profile → concrete engine names, then hands `MemoryService` the right `(VectorPort, GraphPort)` pair for that request.

```yaml
profiles:
  coding:   { graph: codegraph, vector: default }   # code-aware graph engine
  general:  { graph: neo4j,     vector: default }    # falls back to defaults
routing:
  default_profile: general
  # how a request's profile is resolved, first match wins:
  rules:
    - when: { project_id_matches: ".*-(svc|app|api|service)$" }  # heuristic
      use: coding
    - when: { metadata: { domain: coding } }                     # explicit tag
      use: coding
    - when: { client_declared_profile: true }                    # caller sets it
      use: "$declared"
```

**How a request acquires a profile (priority):**
1. **Explicit** — REST body / MCP tool arg `profile: "coding"` (most reliable; the coding agent declares itself).
2. **Derived from `domain`** — NS already tags memories with a `domain` (`coding|research|…`, see the observation pipeline). `domain == coding → coding` profile is a natural, already-present signal.
3. **Heuristic** — project-id patterns / repo presence.
4. **Default** — `general`.

**Routing key.** The profile is part of the **storage scope**, not just read-time, because a memory written under the `coding` graph engine must be read back from the same one. Cleanest model: **profile selects a backend; backend owns its own data store.** So `group_id` stays the tenant key *within* an engine, and the (profile→engine) choice picks *which* engine. A memory's profile is persisted in its payload so reads route consistently. Cross-profile recall (query both engines and merge) is an opt-in `MemoryService` policy, reusing the existing dual-backend merge logic.

**Why this falls out cleanly:** §2's Router already selects ports per request. Profiles are just *named* selections with resolution rules. No new seam — it's the Router's routing table. The same machinery lets a future deployment route, e.g., a cheap local vector store for ephemeral working-memory and a managed one for long-term semantic memory.

**Constraint to flag:** per-use-case engines multiply operational surface (now you run Neo4j *and* the code-graph engine) and complicate cross-use-case recall. Recommend shipping it as **opt-in** (single-engine remains the default) and starting with **two profiles** (`coding`, `general`) over the *same* port ABC, so the only thing that varies is the adapter instance — not the contract.

---

## 7. Phasing (maps onto doc 13's P0–P4)

| Phase | Scope | Risk | Depends on sync? |
|---|---|---|---|
| **P0** | Define `VectorPort`/`GraphPort` ABCs + `PortFilter` AST + capability dataclasses + conformance test suite. No call-site changes. | none (additive) | no |
| **P1** | Implement `Mem0VectorAdapter` + `GraphitiAdapter` over **current** Qdrant/Neo4j; migrate `MemoryService` call sites to the ports; **delete the re-attach shim** and raw-`QdrantClient`/raw-Cypher leaks. Behavior-preserving. | medium (touches many call sites; conformance suite is the safety net) | no |
| **P2** | `EngineRegistry` + `EngineRouter` + named-engine config (`§5`) with the synthesized-default back-compat. Still single engine. | low | no |
| **P3** | **Per-use-case profiles** (`§6`): profile resolution, profile-in-payload, opt-in cross-profile recall. Add a 2nd graph adapter (FalkorDB or BYO code-graph) to prove the routing. | medium | FalkorDB needs graphiti ≥0.29.2 (doc 14) |
| **P4** | Turn on previously-inert capabilities behind flags: typed ontology (doc 15 #2, **no sync, do early**), BM25/entity-boost, cross-encoder rerank, multi-layer recipes. | per-feature | typed ontology: no; others vary |

**Sequencing note:** the **typed-ontology wire-in (doc 15 gap #2)** is independent of all of this and the highest value-to-risk item — it can ship *now* on the current `graphiti_memory.py:320` path, before P0, as a quick win that also de-risks P4.

---

## 8. Open questions

1. **~~Drop the `graphiti_memory.py` impersonation in P1, or keep it?~~ DECIDED (2026-06-22): drop it.** `GraphitiAdapter` will construct `Graphiti` directly and `MemoryService` will reach it through `GraphPort` — retiring both the "pretend to be a mem0 graph provider" indirection and the `memory_service.py:300-341` re-attach shim (doc 14 §8). P1 owns this removal; `graphiti_memory.py` is deleted once the adapter reaches parity.
2. **Does `MemoryService` keep its own extraction, or delegate to mem0 via the port?** Doc 15 §6 lists extraction as a "stop-forking" candidate. Out of scope for the port mechanics, but the port should not *prevent* delegating later.
3. **Profile persistence vs. cross-profile recall cost.** Storing profile-in-payload is simple; merged cross-profile recall multiplies query fan-out. Decide default recall scope (same-profile only?) in P3.
4. **Conformance-suite depth.** How much temporal/bi-temporal behavior must a BYO `GraphPort` prove vs. advertise-and-skip? Defines how "real" the BYO contract is.
5. **Embedding-dim heterogeneity across profiles.** Two vector engines with different dims/embedders complicate cross-profile merge — likely forbid mixing dims within a recall set.
