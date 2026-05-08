---
name: config
description: Show the current Neuralscape plugin configuration (URL, user_id, API-key state). Use when the user asks "what's my neuralscape config", "where is neuralscape pointing", "am I logged in to neuralscape", or wants to verify their settings.
---

# Neuralscape — Config

Display the active plugin configuration without leaking secrets.

## What to do

1. **Read each value** from the manifest-supplied env vars:
   - `URL` ← `process.env.CLAUDE_PLUGIN_OPTION_URL` || `process.env.NEURALSCAPE_URL` || `"http://localhost:8199"` (default)
   - `USER_ID` ← `process.env.CLAUDE_PLUGIN_OPTION_USER_ID` || `process.env.NEURALSCAPE_USER_ID` || OS username (`USER` / `USERNAME`)
   - `API_KEY` ← `process.env.CLAUDE_PLUGIN_OPTION_API_KEY` || `process.env.NEURALSCAPE_API_KEY` (read-only check)

2. **Render** as a markdown table:

   | Setting | Value | Source |
   |---|---|---|
   | URL | `https://neuralscape.example.com` | userConfig |
   | user_id | `aydin` | userConfig |
   | API key | set (32 chars) | userConfig (sensitive) |

   For each row, identify the source as `userConfig` (modern, came from `CLAUDE_PLUGIN_OPTION_*`), `env` (legacy `NEURALSCAPE_*`), or `default` / `os-fallback` (last-resort). Never print the API key value itself — only its length and "set" / "unset".

3. **Prompt the user how to change** any value at the bottom:

   > To change any setting, run `/plugin config neuralscape@neuralscape-plugins`. Sensitive values (API key) are stored in the system keychain.

## Notes

- Read-only. Must never write memories or hit any non-`/health` endpoint.
- If `USER_ID` resolves only to the OS-username fallback, surface that as a warning — the user should set it explicitly so memories don't accidentally collide between machines.
