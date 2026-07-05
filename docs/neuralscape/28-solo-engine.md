# 28 — Solo Engine

**Status:** Design approved, implementation in progress on the `solo-engine` integration branch.
**Audience:** contributors building or reviewing solo-mode work.
**Baseline:** `dev @ 4c65df6` (post memory-package split #136, post subtree prune #137).

---

## 1. Vision: one engine, two layers

Neuralscape today is a team-shaped deployment: FastAPI + three ARQ workers + Neo4j + Qdrant + Redis (+ Docling), eight containers for one user's memories. That is the right shape for a shared, multi-tenant memory server and the wrong shape for an individual who just wants an agent that remembers.

The Solo Engine is **the same Neuralscape** — same extraction, same memory model, same hybrid retrieval, same MCP/REST surface — packaged as **one process on the user's machine** with embedded storage and no external services beyond the LLM API.

The strategic point is bigger than a lightweight SKU. The solo engine is designed to become the **local layer** of a two-layer system:

- **Layer 1 — Solo:** a single local daemon. All memory lives on the user's disk. No server, no account.
- **Layer 2 — Team (later):** the same local daemon, now paired with an upstream team server. Local remains the fast read path; the server is authoritative; a sync protocol keeps the local slice converged. A solo user "joins a team" by having the team engine **adopt** their local store (§8) — their data migrates, their privacy is preserved.

This is the Git model: fully functional offline-first local engine; the network layer is an *addition*, not a requirement. Every design decision below is made so that Layer 2 is a feature flag away, not a rewrite.

### Non-negotiable constraint

**Solo mode must not regress retrieval quality or speed relative to the full stack.** The product offering — extraction quality, hybrid vector+graph recall, contradiction resolution, provenance — is identical in both modes. Retrieval parity is enforced by a benchmark gate (§7): the solo stack must match the server stack on the retrieval suite before anything merges to `dev`.

## 2. Goals and non-goals

**Goals**

1. One-command install and start for a solo user: no Docker, no Neo4j, no Redis, no Qdrant server.
2. Full feature parity on the interactive path: remember / recall / ask / ingest / checkpoint / timeline / MCP tools, including graph memory.
3. All state under one directory (`~/.neuralscape/`), trivially backed up, trivially portable.
4. The existing Claude Code plugin works against solo mode **unchanged** (it already defaults to `http://localhost:8199`).
5. A designed (not bolted-on) solo → team migration path that preserves privacy.
6. Keep the team deployment completely unaffected: `NS_MODE=team` (the default) must be byte-identical in behavior to today.

**Non-goals (this phase)**

- The Layer-2 sync protocol itself (edge daemon syncing against a team server). We design *for* it; we do not build it.
- A GUI/installer app. The installer is a CLI script; the UI comes later.
- Multi-user solo. Solo is exactly one local user (constant local user id).
- Local LLM inference. Extraction still uses the configured LLM API; that is the one external dependency.

## 3. Where the codebase already is (audit summary)

An audit at `dev @ 95edf81` (still true at `4c65df6`) found the runtime is already ~70% solo-capable:

| Concern | Status on dev |
|---|---|
| Writes without Redis | Every MCP/REST write path falls back to inline synchronous storage on enqueue connection failure (`"fallback": "sync"`). Only deferred graph-enrichment jobs are skipped. |
| Non-queue Redis features | Savings meter, SSE event stream, session summarizer, extraction settings, dreaming state: all wrapped in "never raises, degrade to empty". |
| MCP without FastAPI | `mcp_server.py` stdio mode instantiates the service directly; full remember/recall works with zero other processes. |
| Vector store | Embedded on-disk Qdrant is already the config default (`qdrant_url` unset → `~/.neuralscape/qdrant`). |
| Document parsing | `DOCLING_ENABLED=false` → in-process MarkItDown fallback (loses figure extraction and scan-grade PDF fidelity — acceptable solo trade-off). |
| Ingest artifacts | Plain local directory (`~/.neuralscape/ingest`); the "volume" is deployment convention, not code. |
| Crons | All optional maintenance (dedup, expiry, dreaming sweep, playbook synth, connector sync, auto-compile). None gate the interactive path. |
| Graph degradation | If Graphiti fails to init, store/recall degrade gracefully to vector-only. |

**The three real blockers:**

1. **Config gates** — `config.py` `validate_required()` refuses to start without `NEO4J_URI`, `NEO4J_PASSWORD`, `REDIS_URL` (`config.py:568-573`). Pure config-level; the runtime doesn't need them.
2. **Hardcoded graph driver** — the NS mem0 fork constructs `Neo4jDriver` unconditionally (`mem0/mem0/memory/graphiti_memory.py:196`). No provider switch exists.
3. **Untested Kuzu driver** — the graphiti subtree ships a *complete* embedded `KuzuDriver` (`graphiti_core/driver/kuzu_driver.py` + full `driver/kuzu/operations/` package, including the NS Saga schema), but zero tests exercise it.

## 4. Architecture

### 4.1 Process model: one daemon, not stdio-per-session

**Solo mode is a single long-running localhost daemon** serving REST + MCP-over-HTTP on `:8199`, exactly like the containerized API does today.

Why not MCP stdio per client session (which already works)? Because **embedded Qdrant and embedded Kuzu are both single-process stores**. Two concurrent Claude Code sessions each spawning a stdio server would deadlock on the storage locks. One daemon per machine owns the embedded files; every client (plugin, CLI, future UI) talks HTTP to it. This also happens to be the plugin's existing default (`http://localhost:8199`), so the plugin needs zero changes.

Stdio mode remains supported for the degenerate single-session case and for debugging, but the installer sets up the daemon.

### 4.2 Component swap table

| Component | Team stack (unchanged) | Solo engine |
|---|---|---|
| API + MCP | uvicorn in API container | same process, same code, launched by service unit |
| Vector store | Qdrant server | embedded Qdrant (`qdrant_path`) — already exists |
| Graph store | Neo4j container via `Neo4jDriver` | **embedded Kuzu file via the subtree `KuzuDriver`** |
| Task queue | Redis + 3 ARQ worker processes | **in-process `TaskBackend` (asyncio)** |
| Crons | ARQ cron jobs on workers | **in-process scheduler** (subset) |
| Doc parsing | Docling container (MarkItDown fallback) | MarkItDown in-process |
| Artifacts | shared volume | `~/.neuralscape/ingest` — already works |
| SSE events / savings / summarizer | Redis-backed | in-process equivalents or disabled (§5.5) |
| LLM + embeddings | LLM API | LLM API (unchanged) |

Kuzu's single-writer lockfile model — the reason it was rejected as a *shared* store — is exactly correct here: one process owns one file. We pin frozen Kuzu `0.11.3` (MIT, extensions bundled, works indefinitely). If the actively-maintained API-compatible fork (LadybugDB) matures — Graphiti driver merged upstream, sustained release cadence, >1 maintainer — swapping is a dependency bump behind the same seam.

### 4.3 Configuration profile

A single new setting drives everything:

```
NS_MODE = team | solo        # default: team (today's behavior, bit-for-bit)
```

`NS_MODE=solo` flips defaults; every one of them individually overridable:

```
graph_provider   = kuzu          (team: neo4j)
kuzu_path        = ~/.neuralscape/graph.kuzu
task_backend     = inline        (team: redis)
scheduler        = inproc        (team: off — ARQ crons own it)
docling_enabled  = false
qdrant_url       = unset         (embedded, already the default)
auth             = local single-user (constant local user id; no token auth)
```

`validate_required()` becomes mode-aware: solo requires only the LLM API key; team keeps today's gates exactly. **Guardrail:** `NS_MODE=solo` with a `qdrant_url` or `redis_url` explicitly set is a config *error*, not a silent hybrid — hybrid topologies are Layer 2's job, not accidental config.

## 5. Component design

### 5.1 Graph backend seam

`graphiti_memory.py` gains a `graph_provider` parameter threaded from `config.get_mem0_config()`:

- `neo4j` → today's `Neo4jDriver(uri, user, password, database)` — untouched default.
- `kuzu` → `KuzuDriver(db=<kuzu_path>)` from the subtree.

This is the same parameterization the FalkorDB evaluation identified; building it mode-agnostically keeps future backends (or a licensed alternative) one registry entry away. Notes:

- **group_id semantics unchanged.** Solo still writes `user--<local_user_id>` (and project groups). This is deliberate — it makes solo stores *adoptable* (§8) without rewriting graph namespaces.
- Kuzu reifies `RELATES_TO` edges through an intermediate `RelatesToNode_` (it cannot fulltext-index edge properties). This is internal to the graphiti driver, but any NS-authored raw Cypher that pattern-matches edges must be checked against it (§7.2).
- Kuzu stores datetimes natively but the driver round-trips through `parse_db_date`; NS raw Cypher using `datetime()` functions needs Kuzu-dialect equivalents.

### 5.2 Task backend

Extract the enqueue seam into a small interface with two implementations:

- **`RedisTaskBackend`** — today's `task_manager.py` + ARQ, untouched.
- **`InlineTaskBackend`** — an in-process `asyncio` queue with a small worker pool inside the daemon:
  - Same public contract: writes return 202-style envelopes with a `task_id`; `queue_status`/task polling read from an in-memory task table (bounded, LRU-evicted; optionally journaled to `~/.neuralscape/tasks.jsonl` so a restart doesn't strand pollers).
  - Two lanes mirroring the queue split: a *fast* lane (vector writes) and a *slow* lane (graph enrichment) with a concurrency cap of 1–2, so a bulk ingest still can't starve interactive writes — the same isolation philosophy as the three ARQ queues, collapsed into one process.
  - The existing sync-fallback paths remain as a safety net but stop being the *mechanism*: solo writes are first-class async, not exception-handling.

This kills the current solo-degradation caveat (`graph_jobs_skipped`): deferred graph enrichment runs in-process on the slow lane, so **solo memories get full graph treatment**, same as team.

### 5.3 In-process scheduler

A minimal asyncio periodic runner (no new dependency; ~a page of code) registered at daemon startup when `scheduler=inproc`, running the solo-relevant subset on the slow lane:

- `expire_old_memories` (daily) — TTL hygiene.
- `dedup` (daily, single-user scope) — quality maintenance.
- `dream_sweep` — only if dreaming is enabled (off by default, unchanged).
- Auto-compile check (hourly window) — the plugin depends on this for observation compilation.

Wiki/playbook synthesis and connector sync stay available but off by default in solo.

### 5.4 Ingest

No design work needed: `DOCLING_ENABLED=false` + MarkItDown + local artifact dir all exist. Ingest jobs route through the `InlineTaskBackend` slow lane. Documented solo trade-off: no figure extraction, lower fidelity on scanned/complex PDFs. (A later opt-in can run Docling as an on-demand sidecar for heavy users.)

### 5.5 Redis-adjacent features in solo

| Feature | Solo behavior |
|---|---|
| Task status | in-memory table (+ optional journal), same API shape |
| SSE event stream | in-process pub/sub (an `asyncio` fan-out replacing the Redis channel) — kept alive because it is also Layer 2's invalidation feed |
| Savings meter | in-process counters, flushed to a local JSON ledger |
| Session summarizer / context assembler | in-process dict with the same TTL semantics, or disabled at first ship if effort demands — degrade path already exists |
| Extraction settings | read from a local JSON file instead of Redis keys |
| Identity store (federated login) | disabled — solo has exactly one local identity |

None of these are load-bearing for recall correctness; each already has a "Redis down" degrade path today, which is the floor we improve on.

## 6. Distribution and installer

```
uvx neuralscape init          # or: curl -fsSL https://<repo>/install.sh | sh
```

The wizard asks one question — **Solo or Team?**

**Solo path:**
1. Installs the service package with the `solo` extra (`kuzu`, `markitdown`; no arq/redis client needed at runtime but harmless to keep).
2. Prompts for the LLM API key → writes `~/.neuralscape/env`.
3. Creates `~/.neuralscape/{qdrant,graph.kuzu,ingest,logs}`.
4. Writes and starts a service unit (launchd on macOS, systemd --user on Linux) running the daemon on `:8199`.
5. Prints the plugin install one-liner and a `neuralscape doctor` self-check (health, write/recall round-trip, storage paths).

**Team path:** drops the compose file / points at the Helm chart — exactly today's flow.

Upgrade story: `uvx neuralscape upgrade` = package bump + daemon restart. Embedded-store schema migrations ship with the package and run at daemon startup (Kuzu schema is versioned by the graphiti driver's `setup_schema`).

### 6.1 Who installs what — the onboarding matrix

**You install an engine only if you host one; you configure a connection if you use one.** `neuralscape init` is for engine *hosts*. Memory *consumers* — team members — never run it.

| Onboarding path | Local install | Configuration |
|---|---|---|
| **Solo self-hosted** | `uvx neuralscape init` → local daemon | none — the plugin already defaults to `http://localhost:8199`; onboarding guides the user through Claude Code / Cowork setup |
| **Team member** | **none** | plugin/connector config: team base URL + auth (token, or the team's OAuth provider) — exactly today's flow |
| **Team admin** (hosts the team engine) | compose / Helm | today's team deployment, unchanged |
| **Solo user joining a team** | nothing new | adoption flow (§8) migrates the local store into the team engine; the user then flips their plugin URL to the team server — or (Layer 2) keeps the daemon as an edge cache |
| **Team member + edge cache** (Layer 2, later) | same binary, `edge` mode | the daemon gets the team URL + auth and syncs the user's slice; the plugin points back at `localhost:8199` |

Two properties are hard requirements, not conveniences:

1. **The team-member path stays zero-install.** A team member's entire onboarding is "paste URL, authenticate." The solo daemon must never become a prerequisite for team use.
2. **The edge cache is an opt-in upgrade of the team-member path, not a fourth product.** It is the same binary as the solo engine running in a different mode: a team member who wants offline recall or local-disk read latency runs `neuralscape init` and picks *edge* instead of *solo*. Everyone else keeps the thin client. (This also means a solo user who joins a team loses nothing — their daemon flips to edge mode instead of being uninstalled.)

### 6.2 Multiple backends on one machine

"Can a user belong to multiple teams?" decomposes into routing vs. federation:

- **Routing (in scope, cheap):** the plugin already resolves a per-project `PROJECT_ID`; adding a per-project *URL + auth* override maps cleanly to "work project → team A engine, side project → team B engine, everything else → my solo daemon." Exactly one backend answers any given recall, so correctness is trivial.
- **Federation (out of scope):** a single recall fanned out across several engines with merged ranking, cross-engine dedup, and unified provenance is a genuinely hard distributed-retrieval problem. Not planned; the edge daemon's slice-sync (Layer 2) is as far as we go.

## 7. Retrieval parity plan (the merge gate)

Parity is defined and enforced, not assumed.

### 7.1 Test matrix

- Parameterize the graph-touching unit suites across `neo4j` and `kuzu` providers (the subtree KuzuDriver currently has **zero** test coverage — this is the highest-risk item in the whole project).
- CI on the `solo-engine` branch runs the full unit suite in both modes; `NS_MODE=team` runs must be untouched-green at every fold-in.

### 7.2 NS-authored raw Cypher inventory

Eleven sites (per the Neo4j coupling audit — now relocated into the `memory/` package and `extensions/`): graph patchers (wiki/strategy), `list_projects` group scans, `delete_episode`, shared-group listing, plus two offline scripts. Each gets: Kuzu-dialect port (or provider branch), a test against the reified-edge schema, and validation of `datetime()`/`coalesce`/`STARTS WITH` behavior. The wiki-synthesizer's `TransientError` retry (a Neo4j exception) gets a provider-neutral wrapper.

### 7.3 Benchmark gate

Before `solo-engine` → `dev`:

- **Accuracy:** the retrieval benchmark suite (DMR at minimum, ideally the fuller battery) runs against a fully-landed solo store; scores must be within noise (≤1pp) of the same-commit team stack. Same wave discipline as the benchmark protocol: never benchmark a known-buggy pipeline.
- **Latency:** p50/p95 for recall and ask on solo hardware must be ≤ the team-stack numbers measured on the same machine (expected: solo is *faster* — no network hops, no container overlay; embedded Kuzu's analytical reads are strong at this graph scale). Regressions are release blockers.
- **Write throughput:** a bulk-ingest smoke (e.g., a book) must complete with full graph enrichment and interactive writes staying responsive (slow-lane isolation working).

## 8. Solo → team adoption

The flow that makes solo a front door rather than a silo. Designed now, built when Layer 2 lands; solo-side prerequisites ship in this phase.

### 8.1 Export bundle

`neuralscape export` produces a single archive:

- **Memories:** full vector payloads (content, category, metadata, provenance, `occurred_at`, content hashes) — the vector store is the system of record for memory rows.
- **Graph:** the episode log (episodic nodes + raw content) rather than a Neo4j/Kuzu dump. Graph state is *re-derivable*: the team server re-runs enrichment from episodes on its own backend. This sidesteps cross-backend graph serialization entirely and reuses the existing re-ingestion capability.
- **Artifacts:** the ingest directory, preserving `source_ref` integrity.
- **Manifest:** engine version, local user id, project ids, counts, content-hash index (for import dedup and verification).

The bundle format is versioned and documented — it doubles as solo backup/restore (`neuralscape export` / `neuralscape import`), which is worth shipping regardless of teams.

### 8.2 Server-side adopt (Layer 2 feature, contract fixed now)

1. Team admin (or the user, self-serve) initiates adoption with the bundle + the user's authenticated team identity.
2. Identity re-key: the solo local user id maps onto the team identity; every `group_id` is rewritten `user--<local>` → `user--<team-id>` on import (the mechanism the existing identity re-key script already implements for Neo4j).
3. **Privacy invariant: everything imports as `visibility=private`.** Adoption moves data *into* the team engine, never into the shared pool. Sharing remains an explicit per-memory/per-pool act by the user afterward. The team server's group-scoped isolation (native `group_id` scoping at query time) is what makes this a namespace decision, not a trust decision.
4. Import dedup via content hashes (the write path's idempotency machinery); episodes re-enrich on the server's graph queue.
5. The user's local daemon either retires or (Layer 2) flips into edge mode against the team server.

### 8.3 What this phase must get right for adoption to work later

- Stable local user id (constant, recorded in the manifest).
- `group_id` discipline identical to team (§5.1).
- Content hashes and `source_ref` present on every solo write (already the case).
- Episode raw content retained (`store_raw_episode_content` on, as today).

## 9. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| KuzuDriver latent bugs (zero existing tests) | **High** | Test matrix first (§7.1) — before any feature work sits on it; bench gate; fallback below |
| Kuzu upstream frozen (archived, MIT) | Medium | Pinned 0.11.3 works indefinitely for embedded use; LadybugDB is the API-compatible succession path; the provider seam keeps Neo4j available |
| Retrieval parity gap on Kuzu hybrid search (RediSearch-style FTS differences, reified edges) | Medium-High | Bench gate is a hard merge blocker; fallback: solo ships with a single-container Neo4j option (`graph_provider=neo4j` + one `docker run`) — worse UX, zero parity risk |
| Inline task backend starves interactive path under bulk ingest | Medium | Two-lane design + throughput smoke test (§7.3) |
| Solo config drift breaking team mode | Medium | `NS_MODE=team` default is bit-identical; CI runs both modes on every fold-in |
| Embedded-store corruption (power loss) on user machines | Low-Medium | Kuzu WAL + Qdrant on-disk both crash-safe by design; `neuralscape doctor` verifies; export/import is the backup story |
| A second process opening embedded stores (user runs stdio while daemon lives) | Low | Detect the lock, fail with a clear "daemon already running — connect via HTTP" message |

## 10. Delivery plan

Branch discipline: **`solo-engine`** is the integration branch (off `dev @ 4c65df6`). Each work unit is a worktree branched off `solo-engine`, folded back into `solo-engine` on completion — one active worktree at a time. Nothing merges to `dev` until the §7.3 gate passes. Periodic `dev → solo-engine` merges keep drift bounded (the subtrees just got pruned; expect churn).

| # | Work unit (worktree) | Contents | Exit criteria |
|---|---|---|---|
| 0 ✅ | `solo/design-doc` | this document | folded into `solo-engine` |
| 1 ✅ | `solo/config-profile` | `NS_MODE`, mode-aware `validate_required`, defaults matrix, guardrails | team mode bit-identical; unit tests for both profiles |
| 2 ✅ | `solo/graph-provider-seam` | `graph_provider` threading (config → mem0 fork → driver construction), kuzu extra in service deps | neo4j default untouched; kuzu constructs and round-trips a smoke write/read |
| 3 ✅ | `solo/kuzu-test-matrix` | parameterize graph unit suites across providers; port/branch the 11 raw Cypher sites; provider-neutral retry wrapper | both providers green on the graph suites |
| 4 ✅ | `solo/inline-task-backend` | `TaskBackend` interface, inline impl (two lanes), task-status table, wire MCP/REST enqueue seam | full write path incl. graph enrichment works Redis-less; team path untouched |
| 5 ✅ | `solo/inproc-scheduler` + Redis-adjacent swaps | scheduler, in-process SSE/savings/settings equivalents | solo daemon runs maintenance without workers |
| 6 ✅ | `solo/installer` | `neuralscape init/doctor/export/import`, service units, docs | clean-machine install → plugin round-trip in <5 min |
| 7 | `solo/parity-bench` | bench harness against solo stack; DMR + latency runs | **NEXT / deferred** — §7.3 gate green → open the `solo-engine → dev` PR |

Estimated effort: units 1–2 are days; unit 3 is the long pole (~2–3 weeks, where Kuzu surprises will surface); units 4–6 ~1–2 weeks combined; unit 7 is run-and-measure.

## 11. Open questions

1. **Package name/entry point** for the installer (`neuralscape` CLI as a new thin package wrapping the service?) — decide in unit 6.
2. **Local embeddings for solo** — self-hosted embeddings (per the scaling roadmap) would make solo recall fully offline-capable and quota-free; out of scope here but the config seam should not assume the embedder is remote.
3. **Session summarizer in solo v1** — in-process port vs. ship-disabled; decide by effort during unit 5.
4. **LadybugDB adoption criteria** — revisit once its Graphiti driver merges upstream and it shows a second year of releases.
5. **Per-project backend routing in the plugin** (§6.2) — spec the override precedence (env → project marker → global config) when unit 6 lands.
