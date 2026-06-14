---
name: ns-config
description: Show the current Neuralscape plugin configuration (URL, user_id, API-key state) and how identity is resolved on this platform. Use when the user asks "what's my neuralscape config", "where is neuralscape pointing", "am I logged in to neuralscape", or wants to verify their settings. Works in both Claude Code and Claude Cowork.
---

# Neuralscape — Config

Display the active configuration without leaking secrets, and explain how your identity is resolved on the current platform.

## What to do

1. **Read each value** from the manifest-supplied env vars (these are set in Claude Code; generally **unset** in Claude Cowork):
   - `URL` ← `process.env.CLAUDE_PLUGIN_OPTION_URL` || `process.env.NEURALSCAPE_URL` || `"http://localhost:8199"` (default)
   - `USER_ID` ← `process.env.CLAUDE_PLUGIN_OPTION_USER_ID` || `process.env.NEURALSCAPE_USER_ID` || OS username (`USER` / `USERNAME`)
   - `API_KEY` ← `process.env.CLAUDE_PLUGIN_OPTION_API_KEY` || `process.env.NEURALSCAPE_API_KEY` (read-only check)

2. **Pick the branch that matches reality:**

   **A — Claude Code / local config present** (`CLAUDE_PLUGIN_OPTION_*` or `NEURALSCAPE_*` set). Render a table:

   | Setting | Value | Source |
   |---|---|---|
   | URL | `https://neuralscape.example.com` | userConfig |
   | user_id | `aydin` | userConfig |
   | API key | set (32 chars) | userConfig (sensitive) |

   Identify each source as `userConfig` (modern `CLAUDE_PLUGIN_OPTION_*`), `env` (legacy `NEURALSCAPE_*`), or `default` / `os-fallback` (last resort). Never print the API key value — only its length and "set" / "unset". If `USER_ID` resolves only to the OS-username fallback, warn the user to set it explicitly so memories don't collide between machines.

   **B — Cowork / connector mode** (none of the `CLAUDE_PLUGIN_OPTION_*` env vars are set). **Do not imply misconfiguration** — this is normal in Cowork. Report instead:

   > **Connector mode (Claude Cowork).** Your identity is resolved from the OAuth token on the Neuralscape custom connector — there is no local URL, user_id, or API key, and none are needed. Memory operations are scoped to whoever authenticated the connector. To change the connection, manage it under Settings → Connectors.

3. **Prompt how to change** (Claude Code branch only):

   > To change any setting, run `/plugin config neuralscape@neuralscape-plugins`. Sensitive values (API key) are stored in the system keychain.

## Notes

- Read-only. Must never write memories or hit any non-`/health` endpoint.
