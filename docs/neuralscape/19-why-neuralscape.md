# Why Neuralscape

### Your team's shared memory for the agentic era.

Knowledge shouldn't live in one person's head — or get trapped in one person's chat history. On a team, the same lessons get re-learned, the same questions get re-asked, and every AI assistant starts from zero, for every person, every day. Neuralscape is the **shared memory layer for teams and their agents**: what *anyone* — a teammate or an agent acting for them — learns is captured once, stays true over time, and is instantly available to everyone who should have it.

One person figures something out. Now the whole team — and every agent working on the team's behalf — already knows it.

---

## For developers

### The problem, precisely

Most "memory" features are single-player. They remember *your* chats for *you*. But real work is a team sport: shared codebases, shared conventions, shared decisions, and now a growing fleet of agents (Claude Code, CI bots, internal copilots) all acting on the same projects. When memory is siloed per-user or per-session:

- The same context gets re-taught by every person and re-discovered by every agent.
- A decision made in one person's session is invisible to everyone else's tools.
- Knowledge walks out the door when someone changes teams.

And the storage problem underneath is just as hard. You need **semantic recall** (a vector store: "find things like this") *and* **structured, time-aware relationships** (a knowledge graph: "how do these relate, and what changed?"). Building that — extraction, deduplication, temporal invalidation, *and* the multi-tenant scoping that makes it safe to share — is a multi-month project. So teams ship the siloed compromise.

### What Neuralscape is

Neuralscape is a **production-grade, multi-tenant agentic memory layer**. It unifies a vector store and a temporal knowledge graph behind one clean API, and it's built from the ground up around **scopes and sharing** so a whole team and its agents can safely draw from — and contribute to — the same memory.

It stands on two best-in-class open-source engines — **mem0** for vector orchestration and **Graphiti** for the temporal graph — and adds the orchestration, structure, and multi-tenancy that turn them into a shared memory *product* rather than two libraries you wire together yourself.

### Built for teams, first

- **Shared project pools.** Memory is scoped: **global** (a person's own context) and **project** (shared across everyone working on it). A lesson learned in a project pool is immediately usable by every teammate and every agent on that project — while private memories stay private. This isn't a feature bolted on later; the scope-and-visibility model is the core of the design.
- **Agents are first-class citizens.** Every memory can record *who* learned it — a person, Claude Code, a Slack bot, a CI pipeline — as provenance, while everyone can still *use* it. Agents don't get walled into their own silos; they enrich the team's shared knowledge and read from it.
- **Knowledge that outlives the session — and the seat.** Because memory lives in a shared layer, not a chat log, it survives session resets, tool switches, and team changes. Onboarding a new person (or a new agent) means pointing them at memory that's already rich.

### What makes it different (beyond sharing)

- **Two memory systems, one surface.** Vector recall for "what's relevant," a temporal graph for "how things relate and how they changed." Store twice, query both, merge — you never choose between them.
- **Memory that knows about time.** Facts aren't immutable rows. When something changes — a switched database, a revised convention, a teammate's new role — the graph records the new truth *and* when the old one stopped being true. Recall reflects what's true *now*, with history intact instead of overwritten.
- **Structure, not a blob.** Every memory is classified into a 13-category taxonomy (preferences, tech stack, conventions, decisions, architecture, and more) and scoped, so context loads *organized* and you can ask for exactly the slice you need.
- **Agent-native integrations.** A REST API for your services, a native **MCP server** so Claude and other agents read and write memory as a first-class tool, and a Claude Code / Cowork plugin that auto-loads team context at the start of a session. Writes are async (returns immediately); reads are synchronous and fast.
- **You own it.** Self-hostable, opinionated sensible defaults, no lock-in to a hosted black box. Your team's memory, your infrastructure.

### Built on two best-in-class engines — on purpose

Neuralscape doesn't reinvent vector search or graph extraction. It deliberately stands on the two leading open-source memory engines and tracks their upstreams:

- **[mem0](https://github.com/mem0ai/mem0)** — the vector-orchestration engine. Battle-tested fact storage and semantic retrieval, with a roster of 20+ vector-store backends.
- **[Graphiti](https://github.com/getzep/graphiti)** — the temporal knowledge-graph engine. Real bi-temporal modeling: it tracks not just *what* is true but *when* it became true and when it stopped, with automatic contradiction handling.

Choosing both, and reusing their frontier engineering, means Neuralscape inherits their improvements over time while we focus on the layer that's actually missing: turning two powerful-but-separate libraries into one coherent, structured, **shared**, time-aware memory that a whole team of humans and agents can trust.

### Why Neuralscape instead of mem0 or Graphiti directly?

If those engines are so good, why not just use them? Because they're **engines**, not an assembled, multi-tenant memory *system*. Using them directly, you're the one who has to build everything in the right-hand column.

| | mem0 (alone) | Graphiti (alone) | **Neuralscape** |
|---|---|---|---|
| **Semantic (vector) recall** | ✅ | ❌ | ✅ (via mem0) |
| **Temporal knowledge graph** | ❌ (OSS graph store removed) | ✅ | ✅ (via Graphiti) |
| **Both, queried together & merged** | ❌ | ❌ | ✅ one call, one answer |
| **One API for both** | — | — | ✅ REST + MCP |
| **Multi-user scopes & shared team pools** | ❌ (single-user namespacing) | ⚠️ raw `group_id`, you design it | ✅ built-in global/project + private/shared |
| **Structured categories out of the box** | ❌ (hosted-only) | ❌ | ✅ 13-category taxonomy |
| **Agent provenance (who learned it)** | ⚠️ DIY | ⚠️ DIY | ✅ first-class |
| **Async write pipeline + queue** | ⚠️ DIY | ⚠️ DIY | ✅ Redis/ARQ, 202-and-poll |
| **MCP server + Claude Code/Cowork plugin** | ❌ | ❌ | ✅ ships with it |
| **Self-host the whole stack** | ✅ | ✅ | ✅ one `docker compose up` |

**The one-liner:** mem0 and Graphiti are world-class parts. Neuralscape is the assembled machine — the orchestration, the structure, and the multi-tenant sharing you'd otherwise spend months building yourself.

### Already using mem0 or Graphiti? Bring your data.

Adopting Neuralscape doesn't mean starting over. Because it's built *on* these engines, your existing memories have a home here:

- **From mem0** — export your memories and bulk-import them through Neuralscape's batch API; content-hash dedup makes re-runs safe.
- **From Graphiti** — either point Neuralscape at your existing Neo4j and remap scopes in place (fast, lossless, preserves your graph history), or replay your episodes through Neuralscape so they also gain vector recall and structured categories.

See **[Migrating from mem0 or Graphiti](./20-migrating-from-mem0-or-graphiti.md)** for the step-by-step recipes.

### The five-minute version

```
Write:  client → API / MCP → (async) → extract → store in vector + graph → done
Read:   client → API / MCP → query vector + graph (your scope + shared pools) → merge → answer
```

Backed by Qdrant (vectors), Neo4j (graph), Redis (queue). One `docker compose up` and your team has a memory service every person and agent can talk to.

---

## For everyone else

### The short version

AI assistants are brilliant but forgetful — and they forget *separately* for every person. Neuralscape gives your whole team one shared memory, so knowledge stops living in one head, and everyone's assistant picks up where the team left off.

### Think of it like this

Imagine everyone on your team hires a brilliant new assistant every morning — and each one has total amnesia. You all re-explain the same projects, the same preferences, the same decisions, every day, in parallel. Worse, when one person teaches *their* assistant something, nobody else's assistant ever finds out.

Neuralscape is the **shared team notebook** those assistants finally get to keep. When anyone writes something down — a preference, a tool the team uses, a decision and the reason behind it — it's there for everyone, and for every assistant working on the team's behalf. And when something **changes**, the notebook understands the new fact replaces the old one, instead of believing both.

### What you get

- **Knowledge that's shared, not siloed.** What one teammate learns, the whole team's assistants can use.
- **Less repeating yourselves.** Your tools remember the team's context and preferences — once, for everyone.
- **Fewer mistakes from stale information.** Memory updates when reality does.
- **It survives turnover.** Knowledge stays with the team, not just the person who happened to learn it.
- **Your data stays yours.** It can run on your own infrastructure — not locked inside someone else's service.

---

## Where we're going *(aspirational)*

Today Neuralscape is opinionated: it runs beautifully on mem0 + Graphiti, backed by Qdrant and Neo4j. That's the default, and it works.

But memory shouldn't be married to one database. Our north star is to make Neuralscape the **unifying memory layer** — the orchestration brain that can drive *any* vector store and *any* temporal graph database underneath, while keeping mem0 and Graphiti as first-class, batteries-included defaults. On that path:

- **Bring your own engine.** A clean adapter contract so you can run Neuralscape on the vector store and graph database your stack already standardizes on — without rewriting your memory logic.
- **The right engine for the job.** Different workloads have different needs. A coding-agent's memory and a general-knowledge assistant's memory don't have to live on the same engine — Neuralscape will route each use case to the backend that serves it best.
- **More of the engines' power, surfaced.** Richer hybrid retrieval, a typed knowledge graph that understands your domain, and smarter ranking — squeezing the maximum out of the engines underneath while you keep writing to one simple API.

The promise stays the same the whole way: **your team writes to one shared memory layer, and it does the hard part** — so your people *and* your agents finally remember, together.

---

*Neuralscape — the shared memory layer for the agentic era. Multi-tenant, time-aware, team-ready.*
