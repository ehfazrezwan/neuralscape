---
name: status
description: Check Neuralscape service health and show the current plugin configuration. Use when the user asks "is neuralscape working", "is my memory service up", "what URL is neuralscape using", or any health/connectivity question.
---

# Neuralscape — Status

Run this skill when the user wants to confirm the Neuralscape service is reachable and see which URL / user_id / API key the plugin is currently using.

## What to do

1. **Read the configured URL** from `process.env.CLAUDE_PLUGIN_OPTION_URL`. If empty, fall back to `process.env.NEURALSCAPE_URL`. If both are empty, report the default `http://localhost:8199` and tell the user to set the URL via `/plugin config neuralscape@neuralscape-plugins`.
2. **Read the user ID** from `process.env.CLAUDE_PLUGIN_OPTION_USER_ID` (or `NEURALSCAPE_USER_ID`). If unset, surface a clear error.
3. **Detect API-key state** — only report whether `CLAUDE_PLUGIN_OPTION_API_KEY` (or `NEURALSCAPE_API_KEY`) is set. Never echo the value itself.
4. **Hit `GET <URL>/health`** with a 5-second timeout. Use the `Authorization: Bearer <key>` header if an API key is set.
5. **Render a compact status block** like:

   ```
   Neuralscape — status
     URL:     https://neuralscape.example.com
     user_id: aydin
     API key: set
     /health: 200 OK
       redis:        ok
       vector_store: ok
       graph_store:  ok
   ```

   If the health call fails (timeout, 5xx, connection refused), report the error verbatim and suggest the user check `docs/neuralscape/01-getting-started.md` Step 4 in the Neuralscape repo.

## Notes

- This skill only reads. It must never write memories or modify config.
- If the user asks to *change* the URL or user_id, point them at the `config` skill — they'll need to re-run `/plugin config neuralscape@neuralscape-plugins` since values are stored in the keychain (sensitive) or settings.json (non-sensitive).
