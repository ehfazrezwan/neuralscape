---
title: Getting Started
date: 2026-05-06
tags: [reference, neuralscape, getting-started, setup, quickstart]
source: handwritten
---

# Getting Started

This page walks you from a clean machine to a running neuralscape stack with a memory written, retrieved, and (optionally) wired into Claude Code via the plugin or MCP. Allow ~10 minutes if Docker images are warm; ~25 minutes on first pull.

By the end you'll have:
- The full stack running (Neo4j, Redis, Qdrant, neuralscape API, ARQ worker)
- One memory stored asynchronously and retrieved via hybrid search
- (Optional) The Claude Code plugin installed so context auto-injects on `SessionStart`
- (Optional) The MCP server reachable from any MCP-aware client

For deeper detail on any step, follow the wikilinks. Most ops gotchas live in [11-deployment](./11-deployment.md).

## Prerequisites

| Tool | Version | Used for |
|------|---------|----------|
| Docker + Compose v2 | recent | Backing services + app containers |
| Git | any | Clone the repo |
| `uv` | 0.4+ | Required only if running outside Docker |
| Python | 3.12+ | Required only if running outside Docker |
| Node.js | 18+ | Required only if building the plugin |
| Gemini API key | — | LLM extraction + embeddings (`GOOGLE_API_KEY`) |

You don't need to install Python/uv/Node if you're staying entirely inside Docker — the runtime image already has them.

## Step 1 — Clone the repo

```bash
git clone https://github.com/ehfazrezwan/neuralscape.git
cd neuralscape
```

Branch policy: feature work targets `dev`; release PRs go `dev → main`. See Projects/neuralscape (also tracked as `Projects/neuralscape/README` in the maintainer's Obsidian vault).

## Step 2 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set the two required secrets:

```bash
GOOGLE_API_KEY=ya29...        # https://aistudio.google.com/apikey
NEO4J_PASSWORD=change-me      # any password; Neo4j will use it on first start
```

Optional knobs (defaults are reasonable):
- `NEURALSCAPE_API_KEY` — if set, all endpoints except `/health` require `Authorization: Bearer <key>`
- `OBSIDIAN_VAULT_PATH` — host path mounted into the container at `/data/vault` (defaults to `./vault`)
- `MCP_TRANSPORT` — `http` (default in compose) or `stdio`

The full env-var reference lives in [11-deployment](./11-deployment.md).

## Step 3 — Create the vault directory

The conversation-compiler writes to a host-mounted vault. The compose file expects it to exist before startup:

```bash
mkdir -p vault
```

(If you already have an Obsidian vault, point `OBSIDIAN_VAULT_PATH` at it instead — auto-generated category folders will be created alongside your existing notes.)

## Step 4 — Start the stack

The full route brings up all five services with health-gated startup:

```bash
docker compose up --build -d
```

Wait ~30s for healthchecks. Check status:

```bash
docker compose ps
```

All five services (`neo4j`, `redis`, `qdrant`, `neuralscape`, `neuralscape-worker`) should show `(healthy)`.

### Alternative — backing services only

If you'd rather run the API and worker locally for fast reload:

```bash
docker compose up neo4j redis qdrant -d
cd neuralscape-service
uv sync
uv run python main.py                # terminal 1: API
uv run arq worker.WorkerSettings     # terminal 2: ARQ worker
```

## Step 5 — Verify health

```bash
curl http://localhost:8199/health
```

Expected:

```json
{
  "status": "ok",
  "service": "neuralscape",
  "checks": {"redis": "ok", "vector_store": "ok", "graph_store": "ok"}
}
```

If `vector_store` is `unreachable`, the call returns 503 — see [06-storage-backends](./06-storage-backends.md) for Qdrant troubleshooting. If `redis` is `degraded`, async writes will silently fall back to sync (returns 200 instead of 202) — see [07-async-pipeline](./07-async-pipeline.md).

## Step 6 — Write your first memory

Async write returns `202 Accepted` with a `task_id` you poll:

```bash
TASK=$(curl -s -X POST http://localhost:8199/v1/memories/raw \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Prefers TypeScript over JavaScript for new projects",
    "user_id": "you",
    "category": "preference",
    "scope": "global"
  }' | jq -r '.task_id')

# Poll until the worker finishes (typically 1–3 seconds)
curl -s http://localhost:8199/v1/memories/status/$TASK | jq
```

`status` transitions `queued → processing → completed`. The completed payload contains the stored memory's UUID, category, scope, and `source: "vector"`. Schema details: [03-memory-model](./03-memory-model.md).

## Step 7 — Search and retrieve

```bash
curl -s -X POST http://localhost:8199/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "language preferences", "user_id": "you", "limit": 5}' | jq
```

The response merges Qdrant vector hits with Graphiti graph hits, dedup'd by content. See [04-memory-service-core](./04-memory-service-core.md) for the hybrid-search internals.

## Step 8 (multi-user) — Issue per-user tokens

When you're ready to add a second user (or you want auth turned on at all for a single-user install), generate per-user HMAC tokens.

1. **Pick a signing secret.** Add to `.env`:

   ```bash
   NEURALSCAPE_USER_TOKEN_SECRET=<32+ char random string>
   ```

   Restart the stack so the env var lands in the API container:

   ```bash
   docker compose up -d
   ```

2. **Issue tokens for each user.**

   ```bash
   cd neuralscape-service
   NEURALSCAPE_USER_TOKEN_SECRET=<same secret> \
     uv run python scripts/issue_user_token.py --user alice --days 30
   ```

   Output is a `base64url(payload).hmac` two-segment token. Give it to Alice; she sets it as `API_KEY` in her plugin config. Repeat for every user.

3. **Verify with a request.**

   ```bash
   curl -H "Authorization: Bearer <alice-token>" \
        http://localhost:8199/v1/context/global
   ```

   The server reads `user_id` from the token's signed claims (Alice), not from the body. Cross-user spoofing returns 400.

Memory visibility now matters: see the [memory-model doc](./03-memory-model.md#multi-user-model-v22) for the per-category default visibility table (`preference` → private; `tech_stack` → shared; etc.). If you have legacy memories you'd like to make team-visible after the fact, run `scripts/bulk_promote_visibility.py` (dry-run by default) and `scripts/migrate_graph_groups.py` to rewrite the Graphiti group_ids.

For a strictly single-user install you can skip this step entirely — `NEURALSCAPE_API_KEY` (legacy shared key, or no auth at all) keeps working.

## Step 9 (optional) — Install the Claude Code / Cowork plugin

The plugin ships through a marketplace catalog at the repo root, so installation is two commands. From inside Claude Code:

```text
/plugin marketplace add ehfazrezwan/neuralscape
/plugin install neuralscape@neuralscape-plugins
```

Claude Code prompts for three values defined in the manifest's `userConfig`:

| Prompt | Example | Notes |
|---|---|---|
| Neuralscape service URL | `http://localhost:8199` | Whatever you used in Step 5 |
| API key (optional) | (leave empty for local dev) | Bearer token; stored in the OS keychain when set |
| Your user ID | `you` | Stable identifier so memories are scoped to you |

That's it — the post-install build runs automatically (`postinstall` script in `package.json`), the marketplace registers the plugin's hooks, and the next session you open will fire `SessionStart` and inject memory context via `additionalContext`.

**Cowork:** the marketplace install loads the skills, but **hooks don't fire in Cowork** and the token prompt is unreliable there — so the auto inject/capture loop above won't run. Use the **MCP OAuth connector + standing context** instead. Full runbook: [`../../COWORK.md`](../../COWORK.md).

**MCP tools come for free.** The plugin's `.mcp.json` points at the deployed `/mcp/` endpoint, so installing the plugin auto-wires the seven Neuralscape MCP tools (`recall_memories`, `remember`, etc.) into Claude Code's tool list.

**Slash commands:** four discoverable commands ship with the plugin — `/neuralscape:status`, `/neuralscape:search`, `/neuralscape:sync`, `/neuralscape:config`. Claude can also invoke them automatically when the user's question fits.

Plugin internals (hooks, adapters for Claude Code/OpenClaw/generic, the `ConversationTurn` contract, the v2 manifest, slash commands): [09-plugin-system](./09-plugin-system.md).

## Step 10 (optional) — Connect via MCP

The MCP server runs in two modes:

**Streamable HTTP** (default in docker-compose, `MCP_TRANSPORT=http`): mounted at `http://localhost:8199/mcp/`. Point any MCP HTTP client at that endpoint.

**stdio** (for Claude Desktop or stdio-only clients): add an entry to your MCP config:

```json
{
  "mcpServers": {
    "neuralscape": {
      "command": "uv",
      "args": ["run", "python", "mcp_server.py"],
      "cwd": "/abs/path/to/neuralscape/neuralscape-service",
      "env": {"MCP_TRANSPORT": "stdio", "GOOGLE_API_KEY": "..."}
    }
  }
}
```

The 7 tools exposed (`recall_memories`, `remember`, `remember_conversation`, `get_project_context`, `search_knowledge_graph`, `list_memories`, `delete_memories`) are documented in [08-mcp-server](./08-mcp-server.md).

## Troubleshooting

- **Health 503 on `/health`** → Qdrant unreachable. With local on-disk Qdrant (no `QDRANT_URL`), check `~/.neuralscape/qdrant` is writable; otherwise verify the `qdrant` container is up.
- **202 returns 200 instead** → Redis-down sync fallback fired. Check `docker compose logs redis`.
- **Project graph search returns nothing for old data** → `group_id` format migration needed (was `project:{id}`, now `project--{id}`). Run `cypher-shell -u neo4j -p $NEO4J_PASSWORD < neuralscape-service/scripts/migrate-group-ids.cypher`.
- **429 from Gemini** → free-tier quota; the service auto-retries with exponential backoff and falls back to `gemini-2.5-flash`. Tune `LLM_RETRY_MAX_DELAY` if needed. See [05-llm-extraction](./05-llm-extraction.md).
- **Empty extractions** → conversation may have been all junk-filtered (tool logs, file edits). The extraction prompt skips those by design.

## Where to next

- Run unit tests: `cd neuralscape-service && uv run pytest tests/ --ignore=tests/test_async_pipeline.py -v` ([10-testing](./10-testing.md))
- Understand the data model: [03-memory-model](./03-memory-model.md)
- Trace a write end-to-end: [04-memory-service-core](./04-memory-service-core.md) → [07-async-pipeline](./07-async-pipeline.md)
- Add a new client adapter (Cursor, Copilot, …): [09-plugin-system](./09-plugin-system.md)

## Related

- [00-overview](./00-overview.md)
- [02-service-architecture](./02-service-architecture.md)
- [11-deployment](./11-deployment.md)
- [08-mcp-server](./08-mcp-server.md)
- Projects/neuralscape (also tracked as `Projects/neuralscape/README` in the maintainer's Obsidian vault)
