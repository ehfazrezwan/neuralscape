# Conversation Compiler Extension

Automatic memory capture from conversations. Implements Karpathy's "LLM Wiki" pattern + coleam00's "Claude Memory Compiler" concept as a NeuralScape extension.

## What It Does

1. **Flush** — Runs each conversation turn through the main extraction path (the same one `remember_conversation` uses: one prompt, the core category set, speaker and `occurred_at` handling, content-hash dedup) in a single LLM call, stores the facts in NeuralScape, then mirrors the shared ones into the Obsidian vault (category folders and the daily log). A turn yields one memory shape whichever route it arrives by.

2. **Compile** — Synthesizes daily logs into structured articles: session summaries, project pages, decision records, and research articles. Updates the vault index and triggers dedup for each user whose entries were compiled (the semantic phase is skipped when `DREAMING_ENABLED` is on, like the dedup cron). Daily-log entries carry the memory's core category: `decision` feeds the decision article and `domain_knowledge` feeds the research article.

3. **Lint** — Runs 7 health checks on the vault: broken links, orphan pages, stale content, missing cross-references, contradictions (LLM-powered), data gaps, and index drift.

4. **Query** — Index-guided retrieval: reads the vault index to find relevant pages, retrieves content, and synthesizes answers via Gemini. Optionally files answers back as new pages.

## The vault is one shared tree (single-operator by design)

The Obsidian vault is a single directory tree shared by every user the server
serves: the category folders, the daily logs, the compiled articles, `index.md`
and `log.md` are not partitioned by user. Only memories with `shared` visibility
are ever written to it (private memories never reach the vault), but everything
that is written lands in the same place and is readable by whoever can read the
vault, and `POST /query` and `POST /lint` operate over the whole tree.

That is the intended design: the vault belongs to one operator (the person or
team who runs the server and reads the vault), and the extension is not a
per-user store. Do not point it at a vault that several unrelated people should
not see together. Each daily-log entry records its owner (`_(user: ...)_`) so
the post-compile dedup can run for every user whose entries were compiled.

## Configuration

All via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSIDIAN_VAULT_PATH` | `./vault` (or `/data/vault` in Docker) | Root path to the Obsidian vault. The maintainer's local default is `~/Documents/Obsidian/<vault-name>`; pick whatever works for you. |
| `COMPILER_LLM_MODEL` | (NeuralScape default) | Gemini model for extraction/compilation |
| `COMPILE_AFTER_HOUR` | `18` | Hour (24h) after which auto-compilation runs |
| `AUTO_COMPILE` | `true` | Whether to auto-compile daily logs |

## API Endpoints

All routes are mounted at `/v1/extensions/conversation-compiler/`.

### `POST /flush`
Submit a conversation turn for fact extraction. Returns 202 (async via ARQ) or sync result if ARQ unavailable.

```json
{
  "user_message": "...",
  "assistant_response": "...",
  "session_id": "abc123",
  "channel": "slack",
  "timestamp": "2026-04-07T10:30:00",
  "project_id": "neuralscape",
  "user_id": "ehfaz"
}
```

### `POST /compile`
Trigger compilation for a date or all pending.

```json
{
  "date": "2026-04-07",
  "user_id": "ehfaz"
}
```

### `POST /query`
Query the knowledge base.

```json
{
  "question": "How does the dedup system work?",
  "file_back": false,
  "user_id": "ehfaz"
}
```

### `POST /lint`
Run vault health checks.

```json
{
  "structural_only": false
}
```

### `GET /status`
Get extension status and stats.

## Vault Structure

```
vault/
  Daily/          # YYYY-MM-DD.md — raw extraction logs
  Sessions/       # YYYY-MM-DD.md — compiled session summaries
  Projects/       # <project>/README.md — project knowledge pages
  Decisions/      # <slug>.md — decision records with rationale
  Research/       # <topic>.md — investigation/research articles
  index.md        # Auto-maintained vault index
  log.md          # Chronological event log
```

## Event Hooks

- `conversation_turn` — Triggers flush extraction
- `session_end` — Flushes remaining context + checks if auto-compile needed
- `compile_requested` — Triggers compilation

## ARQ Tasks

- `process_conversation_flush` — Async flush extraction
- `process_conversation_compile` — Async compilation
- `auto_compile_check` — Periodic cron (runs after `COMPILE_AFTER_HOUR`)
