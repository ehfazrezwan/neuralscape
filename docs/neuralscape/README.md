# Neuralscape — Reference Docs

Comprehensive reference for the Neuralscape codebase. Read in order or jump to whatever you need:

| # | Page | What it covers |
|---|---|---|
| 00 | [Overview](./00-overview.md) | Hub: architecture-in-prose, audience-routed quick links, key-concepts cheatsheet |
| 01 | [Getting Started](./01-getting-started.md) | Step-by-step setup: clone → `.env` → docker compose → first memory write/search → Claude Code/Cowork plugin install |
| 02 | [Service Architecture](./02-service-architecture.md) | FastAPI route surface (legacy + v1), lifespan, MCP HTTP mount, health checks, middleware |
| 03 | [Memory Model](./03-memory-model.md) | 13 categories, 5 type groups, 2 scopes, `group_id` format, `CATEGORY_VAULT_PATHS`, Pydantic models |
| 04 | [Memory Service Core](./04-memory-service-core.md) | `MemoryService` class, extraction pipeline, dual-write to Qdrant + Neo4j, hybrid search, dedup, graph re-ingestion |
| 05 | [LLM Extraction](./05-llm-extraction.md) | Gemini extraction prompt, JSON parser + regex fallback, embeddings, mem0 factory wiring, retry/backoff |
| 06 | [Storage Backends](./06-storage-backends.md) | Qdrant payload + `metadata.` filter prefix, Neo4j + Graphiti, group_id format, subtree deps, sync-upstream |
| 07 | [Async Pipeline](./07-async-pipeline.md) | ARQ workers, Redis queue, deterministic job IDs, status polling, dedup cron, failure modes |
| 08 | [MCP Server](./08-mcp-server.md) | 7 MCP tools, stdio + Streamable HTTP transports, identity model, Redis fallback |
| 09 | [Plugin System](./09-plugin-system.md) | Claude Code + Cowork plugin, hook manifests, adapters, conversation_compiler extension, vault dual-write |
| 10 | [Testing](./10-testing.md) | 11 test files, fixtures, mocking strategy, integration tests, run commands |
| 11 | [Deployment](./11-deployment.md) | Docker compose stack, two-stage Dockerfile, env vars, structlog, helper scripts, gotchas |
| 12 | [UI PRD](./12-ui-prd.md) | Product requirements for the human-facing web UI: features, audience, constraints, states, accessibility |
| 21 | [Document & File Ingestion](./21-document-ingestion.md) | Ingest files/folders/zips + manual context; Docling/MarkItDown parsing; artifact storage; dedicated ingest worker |

## Quick links by audience

- **"I want to set up Neuralscape locally."** → [01-getting-started](./01-getting-started.md), then [11-deployment](./11-deployment.md) for the full env-var reference.
- **"I want to install the Claude Code / Cowork plugin."** → [01-getting-started](./01-getting-started.md) Step 8, with deep detail in [09-plugin-system](./09-plugin-system.md).
- **"I want to understand how memories flow through the system."** → [04-memory-service-core](./04-memory-service-core.md) for in-process logic, then [07-async-pipeline](./07-async-pipeline.md) for the queue path.
- **"I'm adding a new agent client (Cursor, Copilot, custom)."** → [09-plugin-system](./09-plugin-system.md) for the adapter pattern, then [03-memory-model](./03-memory-model.md) for the canonical category contract.
- **"I'm interfacing via MCP from a client agent."** → [08-mcp-server](./08-mcp-server.md) for the 7 tools, [03-memory-model](./03-memory-model.md) for the category enums.
- **"I'm debugging a stuck async write."** → [07-async-pipeline](./07-async-pipeline.md), then [06-storage-backends](./06-storage-backends.md), then [11-deployment](./11-deployment.md).
- **"I'm building the web UI."** → [12-ui-prd](./12-ui-prd.md) for the feature scope, then [03-memory-model](./03-memory-model.md) and [02-service-architecture](./02-service-architecture.md) for the data shapes the UI consumes.

## Source of truth

These files are mirrored from the maintainer's Obsidian vault. The repo copies are the canonical reference for any contributor — they render on github.com without needing Obsidian. If you're updating one of these pages, edit the repo copy directly.
