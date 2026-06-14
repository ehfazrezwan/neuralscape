---
name: capture
description: Manually compile the current session's PostToolUse observation buffer into Neuralscape memories. Use when the user types `/neuralscape:capture` or asks to "flush memories now" / "save what we've done so far".
---

# Capture (Manual Compile)

## When to use

User-triggered alternative to the automatic UserPromptSubmit threshold. Useful when:

- You've done a lot of work and want memories saved before context grows further.
- You want to capture insights before stepping away from the session.
- You're debugging memory capture and want to see what gets stored from the current buffer.

## What to do

1. **Locate the buffer directory** — `${CLAUDE_PLUGIN_DATA}/observations/` (falls back to `~/.neuralscape/observations/` if `CLAUDE_PLUGIN_DATA` isn't set).

2. **If that directory doesn't exist, or contains no `.jsonl` files** → there is no observation buffer in this environment. This is the expected state in **Claude Cowork** (no `PostToolUse` hook ever runs there) and in an unconfigured CLI. **Do not error.** Tell the user:

   > No tool-observation buffer exists here — passive per-tool capture only runs in Claude Code with hooks active. To save this session's knowledge, use `/neuralscape:save-session` (extracts facts from the conversation) or `/neuralscape:remember` (save one fact). Want me to do that now?

   Then stop.

3. **If a buffer exists** — read the current session's buffer at `${...}/observations/${session_id}.jsonl`. The session_id is whatever is in the current Claude Code session. If you can't determine it from the environment, use the explicit path the user gave you; **otherwise fail closed and ask for the explicit buffer path.** Do NOT guess by picking the most recently modified `.jsonl` — prior sessions leave their buffers in the directory, so "most recent" can point at a different session and compile/truncate the wrong one.

4. Invoke the `compile-observations` skill on that buffer path. Its rubric and v2 schema are the source of truth — don't duplicate them here.

5. Confirm to the user how many memories were stored vs. how many observations were skipped as noise.

6. If the buffer file is present but empty, tell the user "nothing to capture" and truncate it (and remove any `.stale` marker) so the buffer is reset — mirroring how `compile-observations` treats an empty/noisy buffer, so the manual and automatic flows stay consistent.

## Privacy reminder

Re-state when relevant: the skill skips `<private>` tags, API-key-shaped strings, and env-var values. If the user explicitly *wants* something private captured, they'll need to use the API directly with their own redaction.
