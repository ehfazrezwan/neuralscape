---
name: sync
description: Manually flush the current conversation to Neuralscape's compiler. Use after important sessions when the user wants context captured immediately, or when they say "sync my conversation", "save this to memory", or "compile this session now".
---

# Neuralscape — Sync (manual flush)

Trigger an immediate flush + compile of the current session, mid-conversation rather than waiting for the Stop hook.

## Which path to take

This skill has a deterministic HTTP conversation-compiler path (Claude Code) and an MCP path (everywhere else). The compiler `flush`/`compile` endpoints are the **only sanctioned REST calls in this skill** — they have no MCP equivalent (they feed the hook-parity compiler pipeline, not the memory store). Every other memory operation, here and everywhere, goes through the MCP tools; never substitute curl for a memory read or write.

**Never hand-write curl for these calls.** The plugin ships a helper, `node "${CLAUDE_PLUGIN_ROOT}/scripts/sync.js"`, that is the same code the Stop hook runs: it resolves the service URL the hooks' way (explicit `URL` option → the URL baked into the plugin's `.mcp.json` → `http://localhost:8199`), sends the configured `API_KEY` as a Bearer token, applies the deterministic noise filter, and **fail-closed redacts `<private>…</private>` text** before anything leaves the machine. Hand-written requests would have none of that (a token-protected server answers 401, and private text would be sent).

Pick the path **before** doing anything:

0. **Is `${CLAUDE_PLUGIN_ROOT}` set (Claude Code) and is `node` available?**
   - **Yes** → Claude Code fast path: steps 1–4 below.
   - **No** (no plugin root — almost certainly **Claude Cowork**) → **do not** attempt the helper and **do not** error. Invoke `/neuralscape:save-session` instead — it does the same job via MCP (`checkpoint` or `remember_conversation`), which works without any local config. Tell the user you're using the MCP path because there is no local plugin runtime, then stop.

## What to do (helper path — Claude Code)

1. **Check config (optional but cheap)** — run `node "${CLAUDE_PLUGIN_ROOT}/scripts/sync.js" print-config`. It prints `{url, url_source, api_key_set, user_id, user_id_source, project_id}` and never the key. If `user_id` is empty, tell the user to set the plugin `USER_ID` option and stop; otherwise the OS username is used automatically.
2. **Collect the turns** — Claude Code exposes the transcript path only to hooks, so from inside a skill take the last N user/assistant message pairs from the live conversation (typically the most recent 10) and treat them as the batch to flush. Pass the text verbatim; the helper redacts private spans itself.
3. **Flush and compile** — the helper POSTs each turn to `/v1/extensions/conversation-compiler/flush` and then `/v1/extensions/conversation-compiler/compile`. Pipe one JSON object to it:

   ```bash
   node "${CLAUDE_PLUGIN_ROOT}/scripts/sync.js" flush <<'EOF'
   {"turns": [{"user_message": "<user's text>", "assistant_response": "<assistant's text>", "session_id": "<current session id, or omit for manual-sync>", "timestamp": "<ISO-8601, optional>"}],
    "compile": true, "date": "<today YYYY-MM-DD>"}
   EOF
   ```

   The helper skips turns whose assistant response is shorter than 20 chars or matches `NO_REPLY` / `[heartbeat]` / `[system]` patterns (the same deterministic filter as the Stop hook, `MIN_RESPONSE_LENGTH = 20` in `flush.ts`; the MCP `/neuralscape:save-session` path uses a semantic filter instead, so the two differ on purpose). It sends `channel: "claude-code-manual-sync"`, the resolved `project_id` and `user_id`, and — only when every turn was delivered — triggers the compile for `date`.
4. **Report counts** from the helper's JSON line (`{"attempted", "flushed", "failed", "compiled"}`): "Flushed X turns, triggered compile for YYYY-MM-DD." Exit code `1` means the service refused or was unreachable (say which turns were not delivered; they can be re-run), `2` means bad input/config.

## Notes

- The Stop hook does this automatically at session end (Claude Code) — only run `sync` manually for "save this conversation right now" cases.
- `/neuralscape:save-session` is the MCP-native equivalent and works on every platform. `sync` delegates to it automatically when no local service URL is configured (e.g. in Cowork).
- Flush requests return `202 Accepted` with a task_id; you do not need to poll. The compile request returns immediately too. The helper has no network side effect beyond the configured service.
- The full pipeline is documented at `docs/neuralscape/09-plugin-system.md#conversation-compiler-extension` in the Neuralscape repo.
