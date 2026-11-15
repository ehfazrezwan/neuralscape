# Changelog

All notable changes to the `neuralscape` Claude plugin are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project adheres to [Semantic Versioning](https://semver.org/).

## [2.9.8] - 2026-11-16

### Fixed
- Privacy: the session note's `completed` took the final assistant entry of the last turn and redacted it on its own, so a `<private>` span opened in an earlier assistant entry and closed in the final one lost its opening tag and its tail was POSTed to `/v1/checkpoint`. Redaction is now decided over the whole turn's assistant text (the spans are found in the joined text and masked in place) before the final entry is cut out, and every turn is redacted over its whole text before any note field (request, learned, next steps) picks a sentence out of it.
- Privacy: an unclosed `<private>` was redacted only to the end of its line or sentence, so a multi-line private span leaked its later lines. Text that leaves the machine or is written to disk is now redacted fail-closed (`redactPrivate(text, { failClosed: true })`: an unmatched opener redacts through the end of the text): observation rows (every input string, redacted before the 600-character elision, and the output, including JSON-stringified object responses), conversation turns sent to the flush route, the session note, the compact snapshot, and the framework adapter's remembered inbound message. The bounded default mode stays for text that does not leave the machine (the read gate's re-redaction of stored text before it is shown locally).
- The Stop hook POSTed `/compile` after every assistant turn, and the service recompiles the whole date each time. The compile is now windowed per session: a Stop sends it only when it delivered at least one turn and the last successful compile for the session is at least `COMPILE_INTERVAL_MIN` (default 15) minutes old; the SessionEnd hook compiles whatever is still uncompiled. A failed compile leaves its date marked, so the next Stop retries even inside the window, and a date that is still marked when the day changes is compiled too. The state is a small file next to the observation buffers (`<session>.compile.lastturn`, aged out after two weeks). Other clients keep their single trigger at session end. If a session is killed before SessionEnd runs, up to one interval of its last turns waits for the next compile of that date.
- The wait for a lagging transcript re-read and re-parsed the whole transcript on every poll (about 30 times in 3 seconds). It now parses the file once and then only the bytes appended since (a half-written last line and a split multi-byte character wait for the rest; a file that shrank, or a transcript stored as one JSON document, is read in full again).
- A last turn whose transcript entry had not landed when its Stop ran stayed pending until a later Stop that might never come. The SessionEnd hook now delivers pending turns (under the same transcript lock) before it builds the session note, then compiles.
- The Stop-hook documentation no longer says a pending turn is "delivered once": delivery is normally once, and the lock's residual race is documented in 2.9.5.

### Notes
- A user prompt that has no assistant text and is followed by another prompt (the user interrupted and asked again) is still skipped, on purpose: the conversation compiler needs an answer to compile from, the flush route drops turns without one as noise, and the next prompt carries the intent. A test pins it.

## [2.9.7] - 2026-11-16

### Fixed
- The UserPromptSubmit compile instruction told Claude to call `remember` once per memory, contradicting the `compile-observations` skill (one `checkpoint` call for 3 or more memories, `remember` for 1-2). It now states the same rule and points at the skill for the rubric and the field list instead of repeating them.
- The instruction was re-injected on every prompt until the buffer was truncated, so an ignored or still-running compile re-prompted every turn. It is now injected once per threshold crossing: a marker next to the buffer (`<buffer>.jsonl.compile-armed`) records the row count and time of the last injection, and it re-arms when the buffer grows by another `COMPILE_THRESHOLD`, when the last injection is `COMPILE_AGE_MIN` old, or after the buffer was truncated and refilled. Thresholds and defaults are unchanged.
- `CONTEXT_MODE=full` skipped the resume-after-compaction block, the "Previously..." block and the identity cards. It now renders the same continuity blocks as the default index mode (standards, identity cards, "Resuming after compaction" on compact/resume/fork, "Previously..."), then the full text of the remaining memories within the existing 8000-character budget; the resume snapshot and the session note are not repeated in the list. The default stays `index`.
- The adapter for the third-party agent framework dropped every real `message:sent` event: its documented payload carries the assistant text but no user message, and the shared noise filter skips turns with an empty user message. The adapter now remembers the user text from the `message:received` event (per session, consumed by the next `message:sent`) and pairs the two; `hooks/openclaw-hooks.json` subscribes to `message:received` as well. Turns were always global scope; the project now comes from a `workspaceDir` / `cwd` in the payload when there is one, or the `PROJECT_ID` override, and stays global otherwise. The manifest no longer uses the unconfirmed `${PLUGIN_ROOT}` variable (the README installs to `~/.openclaw/hooks/neuralscape/`, and the manifest uses that absolute path); what could not be verified against the framework's public docs is listed in the README.

- Review fixes in the same release: the excluded-projects check of the framework adapter's hooks now uses the project the adapter files the turn under (it used the runner's cwd); the compile instruction re-arms after a truncate-and-refill that passes the old row count between two prompts (the marker records the first row's timestamp); `.compile-armed` markers are garbage-collected with the other per-session leftovers; the adapter's remembered inbound messages are written atomically, one file per message, with `<private>` spans redacted before they reach disk.

### Added
- README section "Server endpoints the plugin calls", and a test that fails when it differs from the `neuralscapeGet` / `neuralscapePost` call sites in `src/` or the REST paths the skills name.
- Tests for the adapter (detection, turn extraction, session-end threshold, project resolution) and subprocess tests running the built `conversation-turn.js`, `session-end.js`, `user-prompt-submit.js` and `session-start.js` (with `CONTEXT_MODE=full`) against a fixture server.

## [2.9.6] - 2026-11-16

### Fixed
- A rejected token, a wrong URL and a dead service all produced the same "memory service unreachable — check `docker compose ps`" notice. The HTTP client now throws typed errors (`NeuralscapeHttpError` with the status, `NeuralscapeNetworkError` for connection failures and timeouts) and one classifier sorts a failure into auth (401/403), not found (404), other client error (4xx), server (5xx) or network. SessionStart words the notice per kind: a rejected key says so and points at `/plugin config neuralscape@neuralscape-plugins` (never printing the key); a 404 says the URL is probably wrong; another 4xx says the request was rejected; 5xx and network failures keep the unreachable wording. Only server and network failures count toward the fail-loud streak (the service answered the others, so "unreachable for N events" would be false); auth, 404 and other 4xx leave the streak untouched, and a success still resets it. PreToolUse, SessionEnd (session summary) and PreCompact use the same classification when deciding whether to count a failure. All hooks still exit 0 and never block.
- An install with no `USER_ID` identifies as the OS login name (a deliberate fallback, unchanged), but the user was never told. When that fallback is in effect and the service URL is not localhost/127.0.0.1/::1, SessionStart now adds one line saying which id is used and how to set `USER_ID`. It does not name the service host.
- The `ns-config` and `ns-status` skills resolved the URL from env vars only and treated "no env vars" as connector mode. They now follow the precedence the hooks use (userConfig, then env, then the baked `.mcp.json` URL, then localhost), show where each value came from, and keep the Cowork connector-mode branch for when the hooks do not run.
- Forks were not warned that the connector URL committed in `.mcp.json` is the upstream channel's. The README now says so at the top, and `npm run bake -- --check` reports the baked URL and the API base the hooks derive from it (`--expect <url>` makes it exit 1 on a mismatch, for a fork's own CI). The committed URL itself is unchanged.

## [2.9.5] - 2026-11-16

### Fixed
- The transcript lock's recovery mutex could be lost. Two hooks that found the same dead recovery mutex both cleared it; if a live recoverer had taken a new one in between, the slower cleaner deleted that live mutex too, and two recoverers could then act at once, one removing the abandoned lock and a hook acquiring a fresh one that the other recoverer then removed. The code of the clearing step also did not match its comment (it never checked that what it moved aside was the record it had judged). Recovery mutexes are now named after the record being recovered plus a generation number, nobody deletes a mutex but the recoverer that holds it (a mutex whose recorder is dead is passed over by taking the next generation), and a recoverer checks once more that it still holds its mutex before it renames the lock. The dead generations are deleted only after the abandoned record is gone, and it can never come back.
- A lock path that exists but cannot be read (no read permission, or a directory) was treated as absent, so the hook retried without sleeping or checking its deadline and ran past the wait bound. Only a missing file counts as absent now; an unreadable lock is never reclaimed and the hook returns at once, skipping its flush with a log line (the turns wait for the next flush). Every retry path now checks the deadline.
- The liveness probe treated any `process.kill(pid, 0)` error other than `EPERM` as proof that the owner was gone, so an unexpected error could make a live owner's lock reclaimable. Only `ESRCH` proves the owner is gone now; `EPERM` and every other error mean the owner is believed alive.
- A hook waiting on a held lock spawned `ps` on every 50 ms poll to check whether the owner's pid had been reused. The verdict is now kept per pid and record for the length of the wait, so `ps` runs once.
- The same transcript reached through a symlink, a relative path or `..` segments used a different lock file, so two such hooks did not exclude each other. The lock file now sits next to the transcript's canonical path.
- Leftovers of crashed hooks (`.tmp` and `.stale.*` files) are swept during recovery once older than the stale bound (a crashed recoverer's mutex files are passed over, never swept), a failed write no longer leaves its temporary file, and an exhausted recovery chain logs a clear line. A transcript that does not exist yet gets the same lock path it will have once it exists (canonical parent directory plus name). Hooks of versions before 2.9.5 locked the path as given, so for a transcript reached through a symlink the old and new versions briefly use two lock files while an old hook is still running.
- Correction to the 2.9.3 entry: the per-transcript lock normally serializes overlapping hooks but does not promise exactly-once delivery; that sentence has been reworded. A recovery mutex written by a pre-2.9.3 hook (no start time) whose pid was reused by a live process was already passed over by the `ps` rule; there is a test for it now.

## [2.9.4] - 2026-11-16

### Fixed
- A running session's observation buffer is no longer offered to another session for compilation. SessionStart offered every non-empty buffer of another session, so a second terminal told the new session to compile and then empty the first terminal's live buffer. The "session ended" marker came from the Stop hook, which fires after every assistant turn, so every live buffer already carried one and the marker could not tell live from finished. The marker (now `<buffer>.jsonl.ended`) is written once by the SessionEnd hook, before any network call. A buffer is offered only when it is non-empty, belongs to another session, and either has an end marker that is not older than the buffer's last write (a session that resumed and wrote again is not ended), or has not been written to for 12 hours (a crashed session never fires SessionEnd; set `PENDING_IDLE_HOURS`, or the legacy `NEURALSCAPE_PENDING_IDLE_HOURS` variable, to change the window). The Stop hook no longer writes any marker.
- Markers written by 2.9.3 and earlier (`<buffer>.jsonl.stale`, from the Stop hook) say nothing about the session ending, so they are ignored and deleted; a buffer that only has such a marker is offered once it has been idle for 12 hours. The compile skill empties a buffer with the Write tool and cannot delete files, so end markers of emptied or missing buffers were left behind forever; SessionStart now removes them.
- The pending-buffer hand-off no longer depends on the service being reachable. When the context fetch failed (service down or a bad token) SessionStart returned before looking at the buffers, so they piled up during an outage; the compile prompt is now appended to the "service unreachable" notice as well.
- A compaction no longer re-injects another terminal's snapshot. The "Resuming after compaction" block showed the newest three snapshots of any session in the project. Snapshots now carry a `session:<id>` tag (snapshots from earlier versions are matched by the session id in their text). After a compaction (Claude Code keeps the session id) only that session's snapshots are shown. On `resume` and `fork`, where Claude Code starts a new session id, the new transcript begins with the earlier session's entries, so the earlier id(s) are read from the head of `transcript_path` (at most 8 MiB, only the `sessionId` key) and only those sessions' snapshots are shown; with no earlier id found, none. `fork` is now handled like `resume`.
- The snapshot is also written to a file next to the observation buffers. `POST /v1/memories/raw` only enqueues the write (it answers 202), so the SessionStart that follows a compaction can run before the snapshot is searchable, and not at all when the service is down. After a compaction SessionStart reads that file as well, and shows the snapshot once when the service returns the same text. The copy is written before any network call (a hanging service cannot prevent it, even with the 60 s hook timeout), and, like the end marker, with mode 0600. Snapshot files older than two weeks are removed.
- PreCompact delivers the turn for clients that have no transcript. Since 2.9.3 it delivered no turns for any client; that is right for Claude Code (the Stop hook delivers its turns and owns the cursor) but lost the turn of a generic client whose turn arrives in the hook payload. Those clients' turn is flushed again from PreCompact. With no cursor, a turn the generic conversation-turn hook already delivered could be sent twice, so the flush now remembers a hash of the last delivered non-Claude-Code turn per session (`<session>.lastturn` next to the buffers, removed after two weeks) and skips an identical repeat; Claude Code turns, which have a cursor, are not filtered. Claude Code is unchanged.
- The compile-observations and capture skills describe the real hand-off: when buffers are offered, how to empty one (the Write tool with an empty string), what happens to the end marker, and that `checkpoint` (three or more memories) or `remember` (one or two) is used consistently.
- Verified, no change needed: PostToolUse stores the tool result (`tool_response`, fixed in 2.9.3) in the buffer rows' `output`, so the compile skill sees real output.

## [2.9.3] - 2026-10-28

### Fixed
- PostToolUse now captures tool output. Claude Code sends it as `tool_response` (string or object), but the hook read `tool_output`, so every observation row had `output: ""`. Object responses are stored as JSON; `tool_output` is still accepted as a fallback. `<private>` spans are now redacted before truncation, so a span split by the size cap cannot leak its kept half.
- Conversation capture now reads the transcript format Claude Code actually writes. Entries carry their text under `message.content` (a string for prompts, a list of blocks for assistant output), but the adapter only looked at a top-level `content`, so every turn looked empty and was dropped as noise while the flush cursor still advanced. The Stop and PreCompact flushes and the transcript half of the SessionEnd note therefore delivered nothing. The adapter now reads `message.content` (falling back to a top-level `content`), treats a turn as one real prompt plus the assistant text up to the next real prompt, and ignores tool results, tool calls, thinking, injected meta entries and sub-agent (sidechain) entries.
- The flush cursor no longer moves past a prompt that has not been answered yet; it stays unread and is delivered by the next flush.
- Turns that earlier versions failed to capture because of this bug are not recovered: a cursor written by an earlier version is used exactly as it is and never moved backwards, so nothing that was already delivered can be sent again.
- Turns are delivered by the Stop hook only. PreCompact no longer delivers turns: compaction adds a summary entry to the transcript (flagged, and never captured) next to the entries it summarizes, so the Stop hook delivers everything, and a second delivering hook could only race it over the cursor or split a turn that is still running. PreCompact still stores its compact snapshot, built from the last real prompts of the transcript whether or not they were delivered, including a prompt that is still being answered.
- The transcript is written asynchronously and can lag the conversation, so the Stop hook delivers the last turn only once the transcript holds the final answer that the Stop event reports (`last_assistant_message`; whitespace is ignored, and it waits up to three seconds for the transcript to catch up). Otherwise the turn stays pending, with its prompt unread, and a later flush delivers it whole; a turn is never delivered without its final answer, and a short turn is dropped as noise only when its final answer is present.
- The hooks that deliver turns (Stop, and the generic conversation-turn hook) now hold a per-transcript lock (a lock file next to the offset file, created atomically with the owner's pid and time) around reading the cursor, delivering and writing the cursor, so two hooks that overlap are normally delivered one after the other (the lock does not promise that each turn is delivered exactly once: a residual race, described below, can deliver a turn twice). A lock is recovered only when its owner is gone and it is older than 90 seconds: the pid is not alive, or (on Linux, from the boot id and the process start time recorded in the lock) the record is from another boot or its pid now belongs to a different process. A lock whose owner might be alive is never reclaimed, however old, so a suspended or slow owner cannot be overtaken: where there is no `/proc` (macOS), a live pid whose process `ps` shows started after the lock was written is not its owner (the pid was reused, or the lock predates a reboot) and the lock is recovered, so a stale lock cannot wedge capture behind an unrelated process; a live pid that may still be the owner is waited for, then the hook skips its flush with a log line saying so. Recoverers are serialized, the recovery mutex follows the same rule (it is never cleared by age alone), and a recovery never puts back a lock it moved by mistake. The owner checks that it still holds the lock before every post and before writing the cursor, and stops cleanly if it does not; the cursor is only ever written forward. Node has no portable operating-system file lock without a native module, so a very small window remains (a recoverer's rename racing a successor's lock, which needs an owner or recoverer that is declared dead while it is alive, for example one running in another pid namespace): at worst one turn is delivered twice, and the cursor never moves backwards.
- A turn's text is bounded to what the flush API accepts (50,000 characters in each of the user message and the assistant response; longer text is cut and ends with a marker saying how many characters were left out), so one very long turn can no longer be refused and block every turn after it. Every delivery failure is treated as transient: the flush stops at that turn and the cursor stays before it, so the turn is retried by the next flush. No turn is skipped because the service refused it.
- Compaction summaries (entries flagged `isCompactSummary` or shown only in the transcript view) and local slash-command wrappers (`<command-name>`, `<local-command-stdout>`, …) are no longer captured as things the user said, so an automatic compaction no longer sends the whole session summary to extraction attributed to the user.
- The SessionEnd note's `completed` field is taken from the last assistant message of the final turn, not from the start of the turn's whole assistant text (which begins with progress commentary when the turn used tools).

## [2.9.2] - 2026-09-07

### Fixed
- Bundled MCP connector now points at the production instance so prod is the single source of truth (the URL is a hardcoded literal because Cowork cannot read env vars).

## [2.9.1] - 2026-07-05

### Changed

- **Skills refreshed to the 22-tool MCP census** (dev @ `64baf3d`). Every
  skill now carries an explicit **MCP-first rule**: the MCP tools are the
  primary surface, and an agent must never switch to `curl`/REST while an
  MCP tool is available. The REST API deliberately mirrors the MCP
  surface, so REST remains documented where it belongs: the `search`
  skill keeps its `/v1/search` fallback (gated on the MCP tool being
  truly absent), and the three calls with no MCP equivalent are labeled
  sanctioned exceptions in place — binary file upload (`/v1/ingest/files`,
  `ingest` skill), the conversation-compiler `flush`/`compile` pair
  (`sync` skill), and the `/health` ops probe (`ns-status` skill).
- **Route-by-job guidance baked into the skills** (measured tiers:
  instant ≈ 0.1–0.2 s metadata ops · ~1.6 s embed-backed ops · 3–7 s
  LLM-backed ops). `recall_memories` stays the non-negotiable default
  for relevance retrieval — the instant tools serve *different jobs*
  (known-ID fetch, chronology, inventory), never cheaper substitutes —
  and skills teach an escalation ladder (broaden → re-run → `ask_memory`)
  instead of pre-emptive downgrading. `ask_memory` is reserved for
  question-shaped requests that want a synthesized, cited answer, with
  `reasoning_level` as the agent-chosen effort knob.
- **New tools woven into existing skills** (no new skills): index-first
  retrieval (`recall_memories(index_only=true)` → `get_memories`) in
  `search`/`recall`; `timeline` and `get_card` in `recall`; `occurred_at`
  guidance (absent-means-unknown, never fabricated) and
  `edit_memory`-vs-re-`remember` correction routing in `remember`;
  `checkpoint` as the preferred batch path in `save-session` and
  `compile-observations` (≥3 memories → one call); `queue_status` for
  "did my writes land?" in `ns-status`/`ingest`/`remember`;
  `ingest_document` with connector `source` provenance in `ingest`.

## [2.9.0] - 2026-07-04

### Added

- **File Read Gate (roadmap D3).** A new `PreToolUse` hook on the `Read` tool
  (`scripts/pre-tool-use.js`): when the target file is larger than
  `READ_GATE_MIN_BYTES` (default 1500) and Neuralscape memories reference it
  (recency-bounded `GET /v1/memories` fetch + basename verification — NS
  memories carry no structured `files_read`/`files_modified` metadata, so a
  literal content mention is the strongest signal, and the hybrid search's
  Graphiti pass is too slow for a PreToolUse budget), the read is denied and
  replaced with a ranked per-file memory
  timeline (`#id | when | title | ~tokens`) plus an escalation menu
  (`get_memories` → `timeline` → `recall_memories(index_only=true)`).
  Ranking prefers memories that *modified* the file (modify-typed
  observations, then modification verbs) and more file-specific observations;
  capped at 10 rows. Hard safety rules: at most once per file per session
  (state-file dedup) — the second Read of the same path ALWAYS passes (the
  override is simply reading again); small files, binary/media extensions,
  excluded projects, and an unreachable service bypass entirely (allow,
  exit 0). Config: `READ_GATE_ENABLED` (default true), `READ_GATE_MIN_BYTES`.
- **Excluded-project globs (roadmap D4).** `EXCLUDED_PROJECTS`
  (comma-separated, `*`/`?` wildcards, case-insensitive;
  `NS_EXCLUDED_PROJECTS` env alias) is honored by every capturing hook:
  PostToolUse observation capture, the Stop conversation flush, the
  SessionEnd session note, SessionStart context injection, and the read gate.
- **Skip-tools list for observation capture (roadmap D4).** The PostToolUse
  built-in skip set now also drops harness plumbing observed as noise in real
  buffers (`Skill`, `ToolSearch`, `AskUserQuestion`, `TaskUpdate`, `TaskStop`,
  `TaskList`, `Workflow`, `Monitor`, `WaitForMcpServers`); `SKIP_TOOLS` adds
  more names without replacing the defaults.
- **Fail-loud unreachable threshold (roadmap D4).** Hooks count CONSECUTIVE
  transport failures in a small state file (any success resets it). Once the
  streak reaches `FAIL_LOUD_THRESHOLD` (default 3), the SessionStart notice
  upgrades to `unreachable for N consecutive events — check docker compose
  ps` instead of the generic degrade line. Exit codes are unchanged (always
  0 — never block).

### Changed

- The hook exit-code taxonomy gains the PreToolUse rule: malformed stdin
  exits 0 there (exit 2 would BLOCK the tool call, so the never-block
  principle wins, same as Stop/UserPromptSubmit).

## [2.8.0] - 2026-07-04

### Added

- **Progressive-disclosure SessionStart injection (roadmap D1).** The default
  `CONTEXT_MODE=index` now injects the *map*, not the memories: a day-grouped
  index table (`#id | time | type | title | ~tokens`) built from the recalled
  context, topped by the dreaming sweep's identity card(s) when available, a
  savings header (`index: N memories, ~X tokens vs ~Y full`), and an
  escalation footer teaching index → `recall_memories(index_only=true)` →
  `get_memories(ids=[...])` → `timeline`. Budget-bounded via
  `INDEX_BUDGET_TOKENS` (default 1500). `CONTEXT_MODE=full` restores the
  legacy full-content injection.
- **Structured session summaries + "Previously…" continuity (roadmap D2).** A
  new `SessionEnd` hook (`scripts/session-summary.js`) distills the session
  into `{request, investigated, learned, completed, next_steps}` from the
  transcript + observation buffer (deterministic, no LLM) and stores it via
  ONE `POST /v1/checkpoint` call as a `session_note`. The next SessionStart
  parses the latest note back into a compact "Previously…" block, next steps
  first. Dense memories still come from compile-observations — the note is a
  continuity artifact, never a duplicate store.
- **F2 code-graph deferral policy.** When the service reports a resolvable
  Graphify code graph (`CODE_GRAPH=auto` probes at session start; `on`/`off`
  override), the escalation footer directs code-*structure* questions to the
  NS `query_code_graph` / `get_code_neighbors` / `code_path` tools, and the
  compile-observations rubric demotes purely-structural observations
  (structure rots; decisions/gotchas/rationale don't).
- **`<private>…</private>` redaction (roadmap D4).** Redacted in everything
  the plugin stores or transmits: PostToolUse observation rows (before they
  hit disk), conversation-compiler flush bodies, and session-note fields.
  Fail-closed: an unclosed `<private>` redacts to end-of-text.
- **Hook exit-code taxonomy (roadmap D4)**, documented in the README:
  transport failure → exit 0 + a one-line notice (never block the session),
  malformed hook stdin → exit 2 (client bug, fail loud) on hooks where exit 2
  cannot block (SessionStart / SessionEnd / PostToolUse); Stop and
  UserPromptSubmit stay exit-0-lenient because exit 2 there has blocking
  semantics.

### Changed

- SessionStart degrades explicitly when the service is down: a one-line
  `[neuralscape] memory service unreachable` notice is injected instead of a
  silent skip.

## [2.7.1] - 2026-07-03

### Fixed

- **Marketplace installs were broken: hook bundles were never shipped.**
  `neuralscape-plugin/.gitignore` excluded `scripts/*.js` — the esbuild output
  the hooks actually invoke (`node "${CLAUDE_PLUGIN_ROOT}/scripts/*.js"`). A
  fresh install from the git marketplace therefore had **no `scripts/`
  directory at all**, and every prompt surfaced `UserPromptSubmitHookError`
  (SessionStart/PostToolUse/Stop failed too, just less visibly). The built
  bundles are now committed; rebuild with `npm run build` whenever `src/`
  changes.
- `package.json` version brought in line with the plugin manifest (was stuck
  at 2.4.0).

## [2.7.0] - 2026-07-03

### Added

- **`edit_memory` + `retag_memories` MCP tools** (tool count 12 → 14). Memories
  are now editable **in place** — ID, author, and creation time all preserved —
  instead of the lossy delete+recreate workaround. `edit_memory` patches a
  single memory (content, category, project_id, tags, visibility, v2 fields;
  explicit `null` clears a field); `retag_memories` bulk-applies metadata ops
  (add/remove tags, set category/project) over a filter set with a `dry_run`
  preview — the housekeeping tool for stamping a project code across a whole
  ingest.
- **Permission split for shared memories.** Organizational metadata
  (tags/category/project) on shared memories is editable by any authenticated
  teammate; content and visibility changes remain owner-or-dictator; private
  memories stay owner-only; the `standard` tier stays dictator-only.

### Fixed

- Content edits through the service previously **wiped all Neuralscape
  metadata** on the memory (category, scope, tags, visibility, owner) — the
  nested metadata is now merged through, and graph work (content re-ingest,
  project/visibility partition migration) runs on the graph queue instead of
  the request thread.
- Marketplace/plugin version desync (2.5.0 vs 2.6.0) — both now 2.7.0.

## [2.5.0] - 2026-07-01

### Added

- **`/neuralscape:ingest` skill.** Loads files, folders, a `.zip`, or a block of
  pasted context into memory. In Claude Code (service URL + `curl`) it uploads to
  the new `POST /v1/ingest/files` endpoint so binaries (PDF, MS Office, HTML) are
  parsed server-side; in Cowork (no filesystem) it ingests pasted text via the
  new MCP `ingest_text` tool. Content is chunked into passages + distilled facts.
- **`ingest_text` MCP tool** (tool count 9 → 10). Manually provide a block of
  context; it's persisted as a Markdown artifact and the memories reference it.

### Changed

- Ingested files and manual context are now **stored as artifacts** on a server
  volume (organized into `{user}/{project}/{category}/` subfolders) and every
  produced memory's `source_ref` points back to the artifact
  (`GET /v1/ingest/artifacts/{file_id}`) — nothing is ingested sourcelessly.
- Rich-format parsing (PDF + MS Office docx/xlsx/pptx + HTML) is handled by a
  **Docling** container, with an in-process MarkItDown fallback.
- Bulk ingestion + connector sync moved to a **dedicated ingest worker/queue**,
  isolated from latency-sensitive memory reads/writes.

## [2.4.0] - 2026-06-15

### Changed

- **Baked-URL distribution model (Cowork-compatible connector).** The MCP
  connector URL in `.mcp.json` is now a **literal value baked per distribution
  channel** instead of `${user_config.URL}`. Cowork bundles the connector
  read-only and does not interpolate `${user_config.*}`, so the template
  rendered as a locked, invalid field; a literal URL is read by both Cowork and
  Claude Code. The bundled connector now appears pre-configured in Cowork —
  users just **Connect** (OAuth), with no manual URL entry. Verified end-to-end:
  Cowork loads the skills, connects, and calls the MCP tools.
- **OAuth-only connector auth.** The static `Authorization: Bearer
  ${user_config.API_KEY}` header was removed from `.mcp.json`. Auth is handled
  by the service's OAuth flow (dynamic client registration + PKCE), so no key is
  configured for the connector on either platform.
- **Single source of truth for the service URL.** The Claude Code hooks now
  derive their API base from the same baked `.mcp.json` (`readBakedUrl`), with
  precedence: explicit `URL` override → baked URL → `localhost`. One place to
  change the URL.

### Added

- **`npm run bake` (`tools/bake.mjs`).** Stamps a distribution channel into the
  committed files: rewrites the literal connector URL in `.mcp.json` and,
  optionally, the marketplace `name`/`owner`. A self-hoster or vendor forks the
  repo and runs one command (`npm run bake -- --url https://their-host`) instead
  of hand-editing JSON. Validates https (loopback http allowed), normalizes a
  base or full `/mcp/` URL, rejects templates; `--dry-run` previews.
- **`npm run package` (`tools/package.mjs`).** Builds a clean, installable zip
  under `dist/` (single top-level dir, zero symlinks) for local `--plugin-dir`
  installs. Note: the **marketplace flow is the supported Cowork path**; Cowork's
  local zip upload is unreliable.
- **Distribution & self-hosting docs.** README + `COWORK.md` document the three
  channels (official / self-host fork / vendor's existing marketplace) and the
  low-effort OAuth self-host setup (two env vars + public HTTPS).

## [2.3.0] - 2026-06-13

### Added

- **Cross-platform memory skills.** Four new MCP-driven skills that work
  in both Claude Code and Claude Cowork (no hooks, no local config
  required): `recall` (load relevant context before acting), `remember`
  (save one fact), `save-session` (extract facts from the conversation —
  the Cowork stand-in for the Stop hook), and `project` (list/pick/create
  the project to scope memory to, important in Cowork which has no working
  directory).
- **`list_projects` MCP tool + `GET /v1/projects` endpoint.** Returns the
  caller's distinct project_ids — private projects plus all team-shared
  projects. Projects are implicit (no separate entity); the list is derived
  from Neo4j `group_id`s via an index-backed `DISTINCT` query (not by scanning
  memories), so it stays cheap even for very large stores. Brings the
  documented MCP tool count to 8.
- **Deterministic `project_id` resolution.** Hooks and skills now resolve the
  project id with a stable precedence — `PROJECT_ID` override → a
  `.neuralscape-project` marker file (walked up from the cwd) → git-repo-root
  basename → working-directory basename — so every subdirectory of a repo
  reports ONE id. Previously `project_id` was just the cwd basename, which
  fragmented memories across a monorepo's sibling folders. A
  `.neuralscape-project` marker is committed at this repo's root pinning it to
  `neuralscape`.
- **Near-duplicate project guard.** The `project` and `remember` skills now
  fuzzy-match a user-typed project name against existing projects
  (case- and separator-insensitive) and confirm before creating a variant
  like `Neuralscape` next to `neuralscape` — the main guard against naming
  drift in Cowork, where there's no working directory to derive the id from.
- **Persist project selection to disk.** The `project` skill can now offer to
  write the chosen id to a `.neuralscape-project` marker at the repo root
  (via `git rev-parse --show-toplevel`), turning a one-session pick into a
  durable, repo-wide default the SessionStart hook reads automatically. Always
  writes at the repo root (never a subdirectory) and skips redundant/global
  cases.

### Changed

- **Skills are now MCP-first and degrade gracefully.** `search` prefers the
  `recall_memories` MCP tool (HTTP `/v1/search` kept only as a Claude Code
  fast path); `ns-status` probes reachability via MCP and reports a
  connector-mode block when no local URL is set; `ns-config` shows a
  Cowork "connector mode" branch instead of implying misconfiguration.
- **Hook-dependent skills no longer error in Cowork.** `capture` and
  `compile-observations` detect a missing observation buffer and redirect to
  `save-session`/`remember` instead of failing; `sync` delegates to
  `save-session` (MCP) when no local service URL is configured.

## [2.2.1] - 2026-06-13

### Fixed

- **Skill name collision with built-in slash commands.** The `status` and
  `config` skills declared bare frontmatter names identical to Claude
  Code's built-in `/status` and `/config`, so typing `/status` resolved
  to the plugin skill instead of the built-in. Renamed them to `ns-status`
  and `ns-config` (invoked as `/neuralscape:ns-status` and
  `/neuralscape:ns-config`); the built-ins are no longer shadowed.

## [2.2.0] - 2026-05-11

### Added

- **Multi-user support (single-team model).** Each authenticated user has
  a personal memory pool plus access to a shared team knowledge pool.
  Private memories (preferences, personal facts, task context) stay
  visible only to their writer; shared memories (tech_stack, convention,
  architecture, decision, etc.) are visible to anyone authenticated to
  the Neuralscape instance.
- **HMAC-signed per-user tokens.** The API_KEY field now accepts either
  a per-user token (issued via `scripts/issue_user_token.py --user
  <name>` on the backend) or the legacy single-shared key. Per-user
  tokens carry `user_id` in their signed claims, so the server can
  enforce identity without trusting the request body.
- **Default visibility per category.** The compile-observations skill
  now defaults each new memory to private or shared based on its
  category (personal categories private, team categories shared),
  with an explicit override mechanism documented in `SKILL.md`.
- **`visibility` field** on the `remember` MCP tool input schema.
- **`include_shared` + `visibility` filters** on `recall_memories` to
  scope retrieval to one pool or the other.

### Changed

- API_KEY userConfig description rewritten to describe both per-user
  token and legacy shared-key formats.
- USER_ID userConfig description notes it must match the token's
  user_id when using per-user tokens.

### Compatibility

- All existing v2.1.x memories continue to work unchanged. They have no
  `metadata.visibility` field, so the server treats them as **private**
  to their original writer — no cross-user leakage. To bulk-promote a
  category to the shared pool retroactively, run
  `scripts/bulk_promote_visibility.py --owner <user_id> --category
  <category> --to shared --apply`.
- Graphiti graph entries written before v2.2.0 sit under the legacy
  `group_id="global"` / `"project--..."` namespaces; the new search
  doesn't see them until you run
  `scripts/migrate_graph_groups.py --owner <user_id> --apply` once
  per environment.

## [2.1.0] - 2026-05-09

### Added
- **Incremental memory capture (PostToolUse hook).** A new `post-tool-use.js`
  hook fires after every tool invocation, filters read-only tools (Read, Glob,
  Grep, NotebookRead, etc.) and trivial Bash commands at the gate, truncates
  large `tool_output` (head 800 + tail 200), and appends one JSONL row to a
  per-session buffer at `${CLAUDE_PLUGIN_DATA}/observations/{session_id}.jsonl`.
  Pure recorder — zero LLM calls in the hook itself, sub-50ms typical.
- **Client-side extraction (compile-observations skill).** A new
  `skills/compile-observations/SKILL.md` instructs Claude to read the buffer,
  group consecutive ops into work units, apply a quality rubric (decision /
  discovery / gotcha / pattern / bugfix / convention / architecture / outcome),
  and submit each significant work unit as a single memory via the
  `mcp__plugin_neuralscape_neuralscape__remember` MCP tool. Runs on the user's
  Claude Code subscription tokens — backend Gemini is bypassed entirely for
  capture.
- **UserPromptSubmit trigger.** A new `user-prompt-submit.js` hook checks the
  per-session buffer on every user message; when it reaches the configured
  threshold (default 25 observations) or age (default 30 minutes), it injects
  `additionalContext` asking Claude to invoke the compile-observations skill
  before responding. Long-running sessions get fresh memories without waiting
  for a session boundary.
- **SessionStart fallback.** The session-start hook now scans for unprocessed
  buffers from prior sessions and prepends a compile prompt to its
  `additionalContext` so short sessions that didn't hit the threshold still
  flush on the next opening.
- **Stop-time stale marker.** session-end now writes a `.stale` sentinel next
  to any non-empty buffer so the next SessionStart picks it up even if the
  user closes Claude Code without sending another prompt.
- **`/neuralscape:capture` slash command** for on-demand compilation.
- **userConfig prompts**: `COMPILE_THRESHOLD` and `COMPILE_AGE_MIN` for
  tuning the in-session compile cadence (defaults 25 / 30).

### Changed
- **Memory model v2** (backend, additive). The plugin now passes 7 new
  optional fields through to Neuralscape on every `remember` call: `domain`,
  `observation_type`, `concepts`, `source_type`, `related_memory_ids`,
  `confidence`, `expires_at`. Existing v1 memories continue to render
  unchanged with these fields as null. See
  `docs/neuralscape/03-memory-model.md` for the full vocabulary tables.
- **Domain-neutral category descriptions.** All 13 category descriptions
  rewritten to apply equally to coding, research, meetings, writing, and
  ops work — the team uses Claude Code for more than just code.
- `readConfig` is now exported from `utils.ts` so the new hooks can read
  user-configurable thresholds.

### Notes
- No migration required. v1 memories continue to work as-is; v2 fields
  populate only on new writes that supply them.
- The 13 memory categories are unchanged — only their descriptions and
  surrounding metadata gained domain-neutrality.
- Backend additions to support the new flow: `POST /v1/memories/raw/batch`,
  content-hash dedup on insert (idempotent re-flushes), and an
  `expire_old_memories` cron.

## [2.0.2] - 2026-05-08

### Fixed
- `.mcp.json` now uses `${user_config.URL}` and `${user_config.API_KEY}`
  for substitution instead of `${CLAUDE_PLUGIN_OPTION_URL}` and
  `${CLAUDE_PLUGIN_OPTION_API_KEY}`. The `CLAUDE_PLUGIN_OPTION_*` form
  is an environment-variable reference resolved when Claude Code spawns
  a subprocess (stdio MCP servers, hook scripts). HTTP MCP servers are
  invoked directly by Claude Code with no subprocess, so that form
  never resolved — the validator reported "Missing environment
  variables" even though the userConfig values were present in
  `~/.claude/settings.json` and `~/.claude/.credentials.json`. The
  `${user_config.<KEY>}` form is the documented substitution for
  declarative manifest fields and resolves correctly from pluginConfigs.

## [2.0.1] - 2026-05-08

### Fixed
- Drop the explicit `"hooks": "./hooks/hooks.json"` reference from
  `plugin.json`. Claude Code 2.1+ auto-loads the standard
  `hooks/hooks.json` from the plugin root; declaring it explicitly causes
  a "Duplicate hooks file detected" load error in `/doctor`. Hooks still
  fire identically — this is a manifest-only correction.

## [2.0.0] - 2026-05-07

### Breaking
- Configuration now flows through Claude Code/Cowork's `userConfig` prompts at
  install time. The plugin reads `CLAUDE_PLUGIN_OPTION_URL`,
  `CLAUDE_PLUGIN_OPTION_API_KEY`, and `CLAUDE_PLUGIN_OPTION_USER_ID`. Legacy
  `NEURALSCAPE_URL` / `NEURALSCAPE_API_KEY` / `NEURALSCAPE_USER_ID` env vars
  remain as a fallback for one release; they will be dropped in 3.0.
- The `DEFAULT_USER_ID` no longer falls back to a hardcoded `"ehfaz"` value.
  If neither the manifest prompt nor an env var supplies a user ID, the plugin
  uses the OS user (`$USER` / `%USERNAME%`) and logs an error if that's also
  unset, instead of silently writing memories under the wrong identity.

### Added
- `marketplace.json` with full Anthropic schema so users can install via
  `/plugin marketplace add ehfazrezwan/neuralscape` followed by
  `/plugin install neuralscape@neuralscape-plugins`.
- Slash command skills under `skills/`: `status`, `search`, `sync`, `config`.
- Bundled remote MCP via `.mcp.json` pointing at the configured service's
  Streamable HTTP `/mcp/` endpoint — installs the 7 Neuralscape MCP tools
  alongside the plugin without separate setup.
- `LICENSE` (MIT).
- This `CHANGELOG.md`.

### Fixed
- `extractClaudeCodeTurns` no longer advances the transcript offset before
  the flush completes. Offsets are staged in memory and persisted by a new
  `commitClaudeCodeFlush()` after `flushTurns` returns, so a crash mid-flush
  leaves the cursor at its prior position and the next session re-flushes.
- `getProjectId` now uses `path.parse(cwd).name` instead of a `/`-only split,
  so Windows backslash paths resolve correctly.
- All three hooks now write a clear stderr line when `user_id` is missing
  rather than silently no-op'ing or writing under a wrong identity.

### Notes
- `hooks/openclaw-hooks.json` continues to use `${PLUGIN_ROOT}` rather than
  `${CLAUDE_PLUGIN_ROOT}` because OpenClaw's hook runner expansion convention
  has not been confirmed. If OpenClaw mirrors Claude Code, switch this in 2.1.

## [1.0.0]

Initial release. SessionStart context injection, Stop-time conversation
flush + compile, Claude Code + OpenClaw + generic adapters.
