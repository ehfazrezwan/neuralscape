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

1. Read the current session's buffer at:
   `${CLAUDE_PLUGIN_DATA}/observations/${session_id}.jsonl`
   (Falls back to `~/.neuralscape/observations/${session_id}.jsonl` if `CLAUDE_PLUGIN_DATA` isn't set.)
   The session_id is whatever is in the current Claude Code session — if you can't read it from the environment, use the path the user gave you, or list `${CLAUDE_PLUGIN_DATA}/observations/` and pick the most recently modified `.jsonl`.

2. Invoke the `compile-observations` skill on that buffer path. Its rubric and v2 schema are the source of truth — don't duplicate them here.

3. Confirm to the user how many memories were stored vs. how many observations were skipped as noise.

4. If the buffer is empty, tell the user "nothing to capture" and don't write anything.

## Privacy reminder

Re-state when relevant: the skill skips `<private>` tags, API-key-shaped strings, and env-var values. If the user explicitly *wants* something private captured, they'll need to use the API directly with their own redaction.
