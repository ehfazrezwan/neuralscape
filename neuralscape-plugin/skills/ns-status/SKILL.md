---
name: ns-status
description: Check that Neuralscape is reachable and show the current plugin configuration. Use when the user asks "is neuralscape working", "is my memory service up", "what URL is neuralscape using", or any health/connectivity question. Works in both Claude Code and Claude Cowork.
---

# Neuralscape — Status

Confirm Neuralscape is reachable and show what the plugin is currently using. The reachability check uses the **MCP connector** (works in both platforms); the local `/health` probe is an additional Claude Code readout when a service URL is configured.

## What to do

1. **MCP reachability (primary, both platforms)** — call `list_memories(limit: 1)` via the MCP tool. If it returns without error, the connector is reachable; report `MCP connector: reachable`. If it errors, report the error verbatim and that the connector is unreachable.
2. **Background queues (MCP, both platforms)** — call `queue_status()`. Report the per-queue pending depths (`main`/`graph`/`ingest`) and the `caught_up` flag — this answers "are my writes done?" without any REST polling.
3. **Read config (Claude Code)** — the same precedence the hooks use. `URL` from `CLAUDE_PLUGIN_OPTION_URL` (userConfig), else `NEURALSCAPE_URL` (env), else the **baked connector URL** (`mcpServers.neuralscape.url` in `$CLAUDE_PLUGIN_ROOT/.mcp.json`, trailing `/mcp/` removed; ignore a missing or `${...}` value), else `http://localhost:8199` (default); `user_id` from `CLAUDE_PLUGIN_OPTION_USER_ID`, else `NEURALSCAPE_USER_ID`, else the OS username (`os-fallback`); API-key **state only** from `CLAUDE_PLUGIN_OPTION_API_KEY` (fallback `NEURALSCAPE_API_KEY`) — never echo the value. Note where each value came from (`userConfig` / `env` / `baked .mcp.json` / `default`).
4. **Local `/health` probe (Claude Code only, when a URL resolved by step 3 and curl are present)** — `GET <URL>/health` with a 5-second timeout, `Authorization: Bearer <key>` if set. Report the per-backend checks. This is the **one sanctioned REST call in this skill** — an ops probe of the backing stores that has no MCP equivalent. (`/health/live` is the dependency-free liveness endpoint; `/health` is the deep readiness check that probes Redis/Qdrant/Neo4j.) Never use REST for any *memory* operation.
5. **Render a compact status block.** Two shapes depending on what's available:

   **Claude Code (env + URL present):**
   ```
   Neuralscape — status
     MCP connector: reachable
     queues: main 0 · graph 1 · ingest 0 (caught up: no)
     URL:     https://neuralscape.example.com  (baked .mcp.json)
     user_id: aydin  (userConfig)
     API key: set  (userConfig)
     /health: 200 OK
       redis:        ok
       vector_store: ok
       graph_store:  ok
   ```

   **Cowork / connector mode (hooks not running: no local env, no resolvable URL):**
   ```
   Neuralscape — status
     MCP connector: reachable
     queues: main 0 · graph 0 · ingest 0 (caught up: yes)
     identity: from OAuth token (no local user_id/URL set — expected in Cowork)
     /health: not available here (no local service URL)
   ```

   If the local `/health` call fails (timeout, 5xx, connection refused) in Claude Code, report the error verbatim and suggest checking `docs/neuralscape/01-getting-started.md` Step 4 in the Neuralscape repo. A 401/403 means the API key is wrong or expired (fix API_KEY via `/plugin config`); a 404 means the URL is wrong. **Do not** treat a missing local URL as an error in Cowork — connector mode is normal there.

## Notes

- This skill only reads. It must never write memories or modify config.
- If the user asks to *change* the URL or user_id, point them at the `ns-config` skill — in Claude Code those live in the keychain (sensitive) or settings.json (non-sensitive) via `/plugin config neuralscape@neuralscape-plugins`; in Cowork identity comes from the OAuth connector, not local config.
