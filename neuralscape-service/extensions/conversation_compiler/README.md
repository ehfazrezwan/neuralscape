# Conversation Compiler Extension

Automatic memory capture from conversations. Implements Karpathy's "LLM Wiki" pattern + coleam00's "Claude Memory Compiler" concept as a NeuralScape extension.

## What It Does

1. **Flush** — Extracts facts, decisions, preferences, and patterns from conversation turns using Gemini. Stores them in NeuralScape and appends to daily log files in the Obsidian vault.

2. **Compile** — Synthesizes daily logs into structured articles: session summaries, project pages, decision records, and research articles. Updates the vault index and triggers dedup.

3. **Lint** — Runs 7 health checks on the vault: broken links, orphan pages, stale content, missing cross-references, contradictions (LLM-powered), data gaps, and index drift.

4. **Query** — Index-guided retrieval: reads the vault index to find relevant pages, retrieves content, and synthesizes answers via Gemini. Optionally files answers back as new pages.

## Configuration

All via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSIDIAN_VAULT_PATH` | `~/Documents/Obsidian/KITT/K.I.T.T.` | Root path to the Obsidian vault |
| `COMPILER_LLM_MODEL` | (NeuralScape default) | Gemini model for extraction/compilation |
| `COMPILE_AFTER_HOUR` | `18` | Hour (24h) after which auto-compilation runs |
| `AUTO_COMPILE` | `true` | Whether to auto-compile daily logs |
| `NEURALSCAPE_URL` | `http://localhost:8199` | NeuralScape API URL |

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
