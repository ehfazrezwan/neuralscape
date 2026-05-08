---
title: Neuralscape — Overview
date: 2026-05-06
tags: [reference, neuralscape, overview, hub]
source: handwritten
---

# Neuralscape — Overview

## Overview

Neuralscape is a production-grade agentic memory layer that fuses **mem0** (vector store) and **Graphiti** (temporal knowledge graph) behind a single service. It exposes both a REST API (FastAPI) and an MCP server so AI agents — Claude Code, OpenClaw, Cursor, custom clients — can store and retrieve structured memories about users and projects through one shared backend. Writes are asynchronous: a `POST` returns `202 Accepted` plus a `task_id`, the actual LLM extraction and dual-write happen on an ARQ worker, and the client polls for completion. Reads are synchronous: a `POST /v1/search` runs hybrid Qdrant + Graphiti retrieval inline and returns `200`. Memories are organized along a 13-category × 5-type-group taxonomy with two scopes (`global`, `project`) and dual-written into Qdrant, Neo4j, and an Obsidian vault.

## Architecture in prose

**Write path.** A client (FastAPI client, MCP tool, or the conversation_compiler plugin) calls a write endpoint. The API constructs a deterministic SHA-256 job ID, enqueues the job onto the Redis-backed ARQ queue, and returns `202 {task_id, poll_url}` immediately. The ARQ worker (a separate process) dequeues the job, runs Gemini extraction against the conversation, batch-embeds the resulting facts, upserts them into Qdrant, and ingests the cleaned conversation as a Graphiti episode in Neo4j. For each stored fact it emits a `memory_stored` event so the conversation_compiler extension can write the fact into the Obsidian vault and rebuild the category index. Result lands in Redis at `arq:result:<job_id>` and the client polls the status endpoint until `completed`. See [async-pipeline](./07-async-pipeline.md) and [memory-service-core](./04-memory-service-core.md).

**Read path.** Searches and context fetches run synchronously through `MemoryService` on the request thread. Vector recall hits Qdrant (with optional dual-scope merge for project queries that should also see global memories), graph recall hits Graphiti via an async bridge using `EDGE_HYBRID_SEARCH_RRF`, and the two result sets are deduplicated, interleaved, and truncated to the requested limit before returning `200`. Health, listing, and CRUD endpoints follow the same synchronous pattern. See [storage-backends](./06-storage-backends.md) and [service-architecture](./02-service-architecture.md).

## Documentation map

- [getting-started](./01-getting-started.md) — Step-by-step setup: clone, `.env`, docker compose, first memory write/search, optional plugin + MCP install, troubleshooting
- [service-architecture](./02-service-architecture.md) — FastAPI route surface (legacy + v1), lifespan, MCP HTTP mount, health checks, middleware
- [memory-service-core](./04-memory-service-core.md) — `MemoryService` class, extraction pipeline, dual-write to Qdrant + Neo4j, hybrid search, dedup, graph re-ingestion
- [plugin-system](./09-plugin-system.md) — Claude Code plugin, hook manifests, adapter pattern (claude-code/openclaw/generic), conversation_compiler extension, vault dual-write
- [mcp-server](./08-mcp-server.md) — 7 MCP tools, stdio + Streamable HTTP transports, identity model, Redis fallback
- [async-pipeline](./07-async-pipeline.md) — ARQ workers, Redis queue, deterministic job IDs, status polling, dedup cron, failure modes
- [memory-model](./03-memory-model.md) — 13 categories, 5 type groups, 2 scopes, group_id format, `CATEGORY_VAULT_PATHS`, Pydantic models
- [llm-extraction](./05-llm-extraction.md) — Gemini extraction prompt, JSON parser + regex fallback, embeddings, mem0 factory wiring, retry/backoff
- [storage-backends](./06-storage-backends.md) — Qdrant payload + `metadata.` filter prefix, Neo4j + Graphiti, group_id format, subtree deps, sync-upstream
- [testing](./10-testing.md) — 11 test files, fixtures, mocking strategy, integration tests, run commands
- [deployment](./11-deployment.md) — Docker compose stack, two-stage Dockerfile, env vars, structlog, helper scripts, gotchas

## Quick links by audience

- **"I want to understand the data flow."** Start with [memory-service-core](./04-memory-service-core.md) for the in-process write/read logic, then [async-pipeline](./07-async-pipeline.md) for how writes get queued and reconciled.
- **"I'm adding a new agent client (Cursor, Copilot, custom)."** Read [plugin-system](./09-plugin-system.md) first (adapter pattern + checklist), then [memory-model](./03-memory-model.md) for the canonical category and scope contract.
- **"I'm debugging a stuck async write."** Trace through [async-pipeline](./07-async-pipeline.md) (queue, polling, retries), then [storage-backends](./06-storage-backends.md) (where the actual writes land), then [deployment](./11-deployment.md) (compose stack, env vars, the Redis-down sync fallback).
- **"I'm a memory-data consumer."** [memory-model](./03-memory-model.md) is canonical for categories, scopes, and `group_id`; [memory-service-core](./04-memory-service-core.md) documents the actual filter shapes and search semantics.
- **"I'm setting up the stack locally."** [getting-started](./01-getting-started.md) is the guided walk-through; [deployment](./11-deployment.md) is the deeper compose + env reference; [testing](./10-testing.md) verifies with the unit suite.
- **"I'm interfacing via MCP from a client agent."** [mcp-server](./08-mcp-server.md) for the 7 tools and transports, [memory-model](./03-memory-model.md) for the category enums callers must supply.

## Key concepts cheatsheet

- **13 categories across 5 type groups.** Semantic (`preference`, `personal_fact`, `technical_skill`, `domain_knowledge`), Project (`tech_stack`, `convention`, `architecture`, `dependency`), Episodic (`decision`, `interaction`), Procedural (`workflow`, `procedure`), Working (`task_context`). Defined in `schemas.py:18-40`. Full table in [memory-model](./03-memory-model.md).
- **2 scopes.** `global` (user-wide) and `project` (project-scoped, requires `project_id`). Semantic categories default to `global`; Project categories default to `project`; flexible categories follow the caller.
- **`group_id` format.** `"global"` or `"project--{project_id}"` — **double-hyphen**, locked in by commit `2edd9b5`. Project searches query both `["global", "project--{id}"]` and merge results.
- **Qdrant filter prefix.** Filter keys for nested fields must be `metadata.{key}` (e.g. `metadata.scope`, `metadata.category`, `metadata.project_id`) because they live under the payload's `metadata` object. Locked in by commit `6c94f20`. Forgetting the prefix silently returns zero results.
- **Async write semantics.** `POST /v1/memories` and `POST /v1/memories/raw` return `202 {task_id, poll_url}`; clients poll `GET /v1/memories/status/{task_id}` until `status == "completed"` or `"failed"`. Redis-down fallback: the API runs the write synchronously and returns `200` with the materialized memories, so clients must branch on status code.
- **Reads are sync.** `POST /v1/search`, `POST /v1/graph/search`, `GET /v1/context/*`, and all CRUD endpoints return `200` inline. No polling.

## Repo pointers

- Project root: `C:\Users\aydin_gb3tlqh\Documents\projects\neuralscape`
- Project guide: `C:\Users\aydin_gb3tlqh\Documents\projects\neuralscape\CLAUDE.md`
- Service code: `C:\Users\aydin_gb3tlqh\Documents\projects\neuralscape\neuralscape-service\`
- Plugin code: `C:\Users\aydin_gb3tlqh\Documents\projects\neuralscape\neuralscape-plugin\`
- Compose stack: `C:\Users\aydin_gb3tlqh\Documents\projects\neuralscape\docker-compose.yml`
- Subtree deps: `C:\Users\aydin_gb3tlqh\Documents\projects\neuralscape\mem0\` and `...\graphiti\`

## Related

- [getting-started](./01-getting-started.md) — guided setup walkthrough
- Projects/neuralscape/README (also tracked as `Projects/neuralscape/README` in the maintainer's Obsidian vault) — project hub (commands, env, branch policy)
- [memory-model](./03-memory-model.md) — canonical schema reference
- [memory-service-core](./04-memory-service-core.md) — write/read engine
- [async-pipeline](./07-async-pipeline.md) — Redis/ARQ pipeline
- [deployment](./11-deployment.md) — operational reference
