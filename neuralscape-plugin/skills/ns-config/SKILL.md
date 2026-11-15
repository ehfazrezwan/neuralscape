---
name: ns-config
description: Show the current Neuralscape plugin configuration (URL, user_id, API-key state) and how identity is resolved on this platform. Use when the user asks "what's my neuralscape config", "where is neuralscape pointing", "am I logged in to neuralscape", or wants to verify their settings. Works in both Claude Code and Claude Cowork.
---

# Neuralscape — Config

Display the active configuration without leaking secrets, and explain how your identity is resolved on the current platform.

## What to do

1. **Read each value** with the same precedence the hooks use (the `CLAUDE_PLUGIN_OPTION_*` env vars are set in Claude Code; generally **unset** in Claude Cowork):
   - `URL` ← `CLAUDE_PLUGIN_OPTION_URL` (userConfig) || `NEURALSCAPE_URL` (env) || the **baked connector URL** || `"http://localhost:8199"` (default). The baked URL is `mcpServers.neuralscape.url` in the plugin's own `.mcp.json` (`$CLAUDE_PLUGIN_ROOT/.mcp.json`), with the trailing `/mcp/` removed; ignore it if it is missing, unparsable, or still a `${...}` template.
   - `USER_ID` ← `CLAUDE_PLUGIN_OPTION_USER_ID` (userConfig) || `NEURALSCAPE_USER_ID` (env) || OS username (`USER` / `USERNAME`, `os-fallback`)
   - `API_KEY` ← `CLAUDE_PLUGIN_OPTION_API_KEY` (userConfig) || `NEURALSCAPE_API_KEY` (env) (read-only check)

2. **Pick the branch that matches reality:**

   **A — Claude Code / local config present** (`CLAUDE_PLUGIN_OPTION_*` or `NEURALSCAPE_*` set). Render a table:

   | Setting | Value | Source |
   |---|---|---|
   | URL | `https://neuralscape.example.com` | userConfig / env / baked .mcp.json / default |
   | user_id | `aydin` | userConfig |
   | API key | set (32 chars) | userConfig (sensitive) |

   Identify each source as `userConfig` (modern `CLAUDE_PLUGIN_OPTION_*`), `env` (legacy `NEURALSCAPE_*`), `baked .mcp.json` (URL only: the connector URL shipped with the plugin), or `default` / `os-fallback` (last resort). Never print the API key value — only its length and "set" / "unset". If `USER_ID` resolves only to the OS-username fallback, warn the user to set it explicitly so memories don't collide between machines (SessionStart already prints a one-line notice of this when the URL is not local).

   The hooks can be running even when no `CLAUDE_PLUGIN_OPTION_*` / `NEURALSCAPE_*` var is set: they then use the baked URL, so "no env vars" alone does **not** mean connector mode when the baked URL resolves and the hooks are installed.

   **B — Cowork / connector mode** (the hooks do not run and none of the `CLAUDE_PLUGIN_OPTION_*` env vars are set). **Do not imply misconfiguration** — this is normal in Cowork. Report instead:

   > **Connector mode (Claude Cowork).** Your identity is resolved from the OAuth token on the Neuralscape custom connector — there is no local URL, user_id, or API key, and none are needed. Memory operations are scoped to whoever authenticated the connector. To change the connection, manage it under Settings → Connectors.

3. **Prompt how to change** (Claude Code branch only):

   > To change any setting, run `/plugin config neuralscape@neuralscape-plugins`. Sensitive values (API key) are stored in the system keychain.

## Notes

- Read-only. Must never write memories or hit any non-`/health` endpoint.
