---
name: compile-observations
description: Read a JSONL buffer of captured tool observations and extract significant memories using the Neuralscape memory model v2. Use this when a UserPromptSubmit, SessionStart, or `/neuralscape:capture` instruction asks you to compile observations from a buffer file path.
---

# Compile Observations

## When to use

Invoked by:

- The **UserPromptSubmit hook** when the per-session buffer reaches the compile threshold (default: 25 observations or 30 minutes since the oldest entry).
- The **SessionStart hook** when there are unprocessed buffers from prior sessions.
- The **Stop hook's stale-marker handoff** — picked up at the next SessionStart.
- The user manually via **`/neuralscape:capture`**.

The trigger always supplies the path(s) to one or more JSONL files to process.

## What to do

1. **Read each provided buffer file.** Each line is a JSON row:
   ```json
   {"ts": "...", "session_id": "...", "cwd": "...", "project_id": "...", "user_id": "...", "tool": "Edit|Write|Bash|...", "input": {...}, "output": "..."}
   ```
2. **Group consecutive rows that work on the same target** (same file, same topic, same command intent). One *work unit* per group.
3. **For each work unit, decide: is this *significant*?** Apply the quality rubric below. If not, skip.
4. **For each significant work unit, write ONE memory** and submit it via `mcp__plugin_neuralscape_neuralscape__remember` with the v2 fields filled in.
5. **After all calls succeed, truncate every buffer file that was processed** by using the `Write` tool with the buffer's absolute path and content `""`. (This preserves the file for next session writes; the `Write` tool accepts only `file_path` and `content`.)

## Quality rubric — keep these

A memory is significant if it satisfies AT LEAST ONE:

- **Decision** — a non-obvious choice was made (which library, approach, vendor, or scope cut, with the why).
- **Discovery** — something non-obvious was learned about the code, system, domain, person, market, or process.
- **Gotcha** — a surprising behavior worth remembering for next time.
- **Pattern** — a recurring approach was applied or established.
- **Bugfix** — a defect was identified and corrected, with the root cause.
- **Convention** — a project norm was established or applied.
- **Architecture** — a structural choice was made (system, org, info architecture).
- **Outcome** — a meeting, conversation, or task produced a result worth remembering.

## Quality rubric — skip these

- A routine file edit / read / search.
- A tool invocation logged for its own sake ("ran ls", "edited utils.ts").
- Anything tied to *this* session that won't matter in 30 days.
- Anything containing `<private>`, an API-key shape (`sk-...`, `gsk_...`, `ghp_...`, etc.), passwords, or env-var values.
- Tool errors that don't reveal anything — but DO keep errors that surface a real bug or constraint.

## Tone & format

Wiki entry, not log line. Lead with the substance, not the tooling.

- ❌ "Edited utils.ts to fix the offset bug"
- ✓ "The `.neuralscape-offset` file should be written *after* `flushTurns` resolves, not before — otherwise a failed flush leaves the cursor advanced and we drop turns. The fix uses a `pendingOffsets` Map staged in extract, committed by `commitClaudeCodeFlush()`."

- ❌ "Bash output: npm install completed"
- ✓ "Pin pydantic to >=2.5 — the `field_validator` decorator behavior we rely on changed in 2.5; older versions silently accept invalid enum values."

- ❌ "Met with Sarah today"
- ✓ "Sarah confirmed Q3 OKR for the analytics team is shifted to Q4 because the data-platform team's migration won't land in time. Plan the dashboard work after Aug 15."

## v2 fields to fill (per `mcp__plugin_neuralscape_neuralscape__remember` call)

- `content` (required) — the wiki sentence
- `category` (required) — one of:
  - **Semantic**: `preference`, `personal_fact`, `technical_skill`, `domain_knowledge`
  - **Project**: `tech_stack`, `convention`, `architecture`, `dependency`
  - **Episodic**: `decision`, `interaction`
  - **Procedural**: `workflow`, `procedure`
  - **Working**: `task_context`
- `scope` — `global` or `project` (defaults by category, but set explicitly when ambiguous)
- `project_id` — derive from each row's `project_id` (basename of `cwd`); use the dominant one in the work unit
- `domain` — `coding | research | meeting | writing | ops | personal | general`. Infer from the buffer contents.
- `observation_type` — `bugfix | feature | refactor | decision | discovery | gotcha | pattern | trade_off | research_note | meeting_outcome | task_plan | fact`
- `concepts` — 1–3 from `how-it-works | why-it-exists | what-changed | problem-solution | gotcha | pattern | trade-off | open-question | next-step | blocker`
- `tags` — free-form, ≤5 items (keep them short and specific)
- `source_type` — always `tool_extraction`
- `confidence` — your honest 0.0–1.0 self-assessment of memory quality
- `related_memory_ids` — leave **unset** in this skill flow. The rule above is one memory per work unit, so there is no "second memory from the same work unit" to link. (The field exists for higher-level callers that knowingly emit multiple linked memories.)
- `expires_at` — only for `task_context` entries that should auto-purge (e.g. set to 30 days from now)

## Throughput target

A typical session of 50–100 captured rows should yield **3–10 memories**, not 50. Err on the side of fewer, denser memories. If you're unsure whether something is significant, it isn't.

## After compile

After every successful set of `remember` calls for a buffer file:

1. Use the `Write` tool with the buffer's full path and content `""` to truncate it (do NOT delete — the next session may write to it).
2. The `.stale` marker file (if present) is removed automatically by the next `truncateBuffer` write — but if the skill is called outside that path, also remove `<buffer_path>.stale` if it exists.

If a buffer is empty or the rubric finds zero significant memories, that's a normal outcome — truncate the file and move on.
