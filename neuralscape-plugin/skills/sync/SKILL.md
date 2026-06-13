---
name: sync
description: Manually flush the current conversation to Neuralscape's compiler. Use after important sessions when the user wants context captured immediately, or when they say "sync my conversation", "save this to memory", or "compile this session now".
---

# Neuralscape — Sync (manual flush)

Trigger an immediate flush + compile of the current session, mid-conversation rather than waiting for the Stop hook.

## Which path to take

This skill has a deterministic HTTP conversation-compiler path (Claude Code) and an MCP path (everywhere else). Pick the path **before** doing anything:

0. **Is a service URL (`CLAUDE_PLUGIN_OPTION_URL` / `NEURALSCAPE_URL`) set AND is `curl` available?**
   - **Yes** → this is the Claude Code fast path. Use the HTTP conversation-compiler flow in steps 1–5 below.
   - **No** (no local URL/env — almost certainly **Claude Cowork**) → **do not** attempt curl and **do not** error. Invoke `/neuralscape:save-session` instead — it does the same job (extract + store the conversation) via the MCP `remember_conversation` tool, which works without any local config. Tell the user you're using the MCP path because no local service URL is configured, then stop.

## What to do (HTTP path — Claude Code)

1. **Read config** — `CLAUDE_PLUGIN_OPTION_URL`, `CLAUDE_PLUGIN_OPTION_USER_ID` (legacy fallbacks supported). If `USER_ID` is missing, fall back to the OS username rather than aborting.
2. **Locate the transcript** — Claude Code exposes the path on the `transcript_path` field of hook input. From inside a skill (no hook stdin) you have to walk the conversation history yourself: take the last N user/assistant message pairs from the live conversation (typically the most recent 10) and treat them as the batch to flush.
3. **For each turn** (user message + assistant response pair), POST to `<URL>/v1/extensions/conversation-compiler/flush` with body:

   ```json
   {
     "user_message": "<user's text>",
     "assistant_response": "<assistant's text>",
     "session_id": "<current session id, or 'manual-sync'>",
     "channel": "claude-code-manual-sync",
     "timestamp": "<now ISO-8601>",
     "project_id": "<basename of cwd>",
     "user_id": "<USER_ID>"
   }
   ```

   Skip turns where the assistant response is shorter than 20 chars or matches `NO_REPLY` / `[heartbeat]` / `[system]` patterns — that's the same noise filter the Stop hook applies.

4. **After all turns flush successfully**, POST to `<URL>/v1/extensions/conversation-compiler/compile`:

   ```json
   {
     "date": "<today YYYY-MM-DD>",
     "user_id": "<USER_ID>"
   }
   ```

5. **Report counts** — "Flushed X turns, triggered compile for YYYY-MM-DD." If anything failed, list the failures briefly.

## Notes

- The Stop hook does this automatically at session end (Claude Code) — only run `sync` manually for "save this conversation right now" cases.
- `/neuralscape:save-session` is the MCP-native equivalent and works on every platform. `sync` delegates to it automatically when no local service URL is configured (e.g. in Cowork).
- Flush requests return `202 Accepted` with a task_id; you do not need to poll. The compile request returns immediately too.
- The full pipeline is documented at `docs/neuralscape/09-plugin-system.md#conversation-compiler-extension` in the Neuralscape repo.
