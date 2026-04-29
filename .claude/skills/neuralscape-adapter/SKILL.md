---
name: neuralscape-adapter
description: Add new agent framework adapters to NeuralScape's unified conversation hook system. Use when integrating a new AI client (Cursor, Copilot, custom agent), adding a client adapter, or extending the plugin to support another agent framework. Covers adapter creation, client detection, hook manifest, and build config.
---

# NeuralScape Client Adapter

Guide for adding support for a new agent framework to NeuralScape's conversation capture system.

## Architecture

NeuralScape uses an adapter pattern to support multiple AI clients. All clients share the same processing pipeline — only the input extraction differs.

```
stdin JSON → detect client → adapter extracts turns → core/flush → NeuralScape API
```

### Key files

| File | Purpose |
|------|---------|
| `neuralscape-plugin/src/core/types.ts` | `ConversationTurn`, `TurnExtractor`, `SessionEndExtractor` interfaces |
| `neuralscape-plugin/src/core/flush.ts` | Shared filter + flush logic (ALL clients use this) |
| `neuralscape-plugin/src/core/compile.ts` | Shared compile trigger (ALL clients use this) |
| `neuralscape-plugin/src/adapters/detect.ts` | Client detection + extractor lookup |
| `neuralscape-plugin/src/adapters/openclaw.ts` | Reference: OpenClaw adapter (1 turn per event) |
| `neuralscape-plugin/src/adapters/claude-code.ts` | Reference: Claude Code adapter (N turns from transcript) |
| `neuralscape-plugin/src/adapters/generic.ts` | Reference: simplest adapter (direct JSON) |
| `neuralscape-plugin/src/hooks/conversation-turn.ts` | Entry point: detect → adapter → flush |
| `neuralscape-plugin/src/hooks/session-end.ts` | Entry point: detect → adapter → flush + compile |
| `neuralscape-plugin/src/utils.ts` | HTTP client, stdin parsing, identity helpers |

### Shared interfaces (from `core/types.ts`)

```typescript
interface ConversationTurn {
  userMessage: string;
  assistantResponse: string;
  sessionId: string;
  channel: string;        // e.g. "cursor", "copilot", "my-agent"
  timestamp: string;      // ISO 8601
  projectId?: string;
  userId: string;
}

interface SessionEndInput {
  date: string;           // YYYY-MM-DD
  userId: string;
  shouldCompile: boolean;
}

type TurnExtractor = (raw: Record<string, unknown>) => ConversationTurn[] | Promise<ConversationTurn[]>;
type SessionEndExtractor = (raw: Record<string, unknown>) => SessionEndInput | Promise<SessionEndInput>;
```

## Instructions

### Step 1: Understand the client's data shape

Before writing code, determine what JSON the new client sends via stdin when a conversation turn completes or a session ends. Document:

1. What fields identify this client (for detection)?
2. Where is the user message? The assistant response?
3. Is there a session ID? A project/workspace path?
4. Does the client send one turn at a time, or a full transcript?
5. What event fires at session end?

### Step 2: Create the adapter file

Create `neuralscape-plugin/src/adapters/{client-name}.ts`:

```typescript
/**
 * {ClientName} adapter — extracts conversation turns from {ClientName}'s event format.
 *
 * Input: { ... describe the expected stdin shape ... }
 * Output: ConversationTurn[] 
 */

import type { ConversationTurn, SessionEndInput } from "../core/types.js";
import { getUserId, getProjectId } from "../utils.js";

/**
 * Extract conversation turns from a {ClientName} event.
 */
export function extract{ClientName}Turns(
  raw: Record<string, unknown>
): ConversationTurn[] {
  // Map client-specific fields to ConversationTurn
  return [
    {
      userMessage: /* extract from raw */,
      assistantResponse: /* extract from raw */,
      sessionId: /* extract or default to "unknown" */,
      channel: "{client-name}",
      timestamp: /* extract or default to new Date().toISOString() */,
      projectId: /* extract workspace/project if available */,
      userId: getUserId(),
    },
  ];
}

/**
 * Extract session-end metadata from a {ClientName} event.
 */
export function extract{ClientName}SessionEnd(
  raw: Record<string, unknown>
): SessionEndInput {
  return {
    date: /* extract or default to today */,
    userId: getUserId(),
    shouldCompile: true,
  };
}
```

Key rules:
- Always return `ConversationTurn[]` (even for a single turn, wrap in array)
- Use `getUserId()` from utils for the user ID
- Use `getProjectId(cwd)` if the client provides a working directory
- Set `channel` to a unique string identifying this client
- Default missing fields gracefully — never throw on missing optional data

### Step 3: Register the adapter in detect.ts

Edit `neuralscape-plugin/src/adapters/detect.ts`:

1. Add the new client type to `ClientType`:
```typescript
export type ClientType = "openclaw" | "claude-code" | "{client-name}" | "generic";
```

2. Add the import:
```typescript
import { extract{ClientName}Turns, extract{ClientName}SessionEnd } from "./{client-name}.js";
```

3. Add detection logic in `detectClient()` — add BEFORE the `"generic"` fallback:
```typescript
export function detectClient(raw: Record<string, unknown>): ClientType {
  if (raw.transcript_path || raw.hook_event_name) return "claude-code";
  if (raw.type && raw.action) return "openclaw";
  if (raw.{unique_field}) return "{client-name}";  // ADD THIS
  return "generic";
}
```

4. Add cases to both switch statements in `getTurnExtractor()` and `getSessionEndExtractor()`:
```typescript
case "{client-name}":
  return extract{ClientName}Turns;
```

### Step 4: Handle session-end behavior (if needed)

If the new client needs to flush turns at session end (like Claude Code does with transcripts), edit `neuralscape-plugin/src/hooks/session-end.ts`:

```typescript
// Add the client to the flush-before-compile block
if (client === "claude-code" || client === "{client-name}") {
  const turnExtractor = getTurnExtractor(client);
  const turns = await turnExtractor(raw);
  await flushTurns(turns);
}
```

Only do this if the client sends conversation data at session end rather than per-turn.

### Step 5: Create a hook manifest (if the client uses one)

If the new client has its own hook configuration format, create `neuralscape-plugin/hooks/{client-name}-hooks.json`. Use the OpenClaw manifest as a template:

```json
{
  "description": "NeuralScape memory hooks for {ClientName}",
  "hooks": {
    "{conversation-event}": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "node \"${PLUGIN_ROOT}/scripts/conversation-turn.js\"",
            "timeout": 15,
            "async": true
          }
        ]
      }
    ],
    "{session-end-event}": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "node \"${PLUGIN_ROOT}/scripts/session-end.js\"",
            "timeout": 15,
            "async": true
          }
        ]
      }
    ]
  }
}
```

### Step 6: Build and test

```bash
cd neuralscape-plugin
npm run build
```

Test with a sample payload:
```bash
echo '{ ... client-specific JSON ... }' | node scripts/conversation-turn.js
```

Expected output: `{"continue":true,"suppressOutput":true}` followed by either a successful flush or a connection error (if NeuralScape isn't running).

Test session end:
```bash
echo '{ ... client-specific JSON ... }' | node scripts/session-end.js
```

### Step 7: Verify the full pipeline

With NeuralScape running (`cd neuralscape-service && uv run python main.py`):

1. Pipe a conversation turn — check that `/flush` is called and facts appear in the vault
2. Pipe a session end — check that `/compile` is called
3. Verify the daily log has the entry (`Daily/{date}.md`)
4. Verify the category folder has the entry (e.g., `Semantic/Preferences/entries.md`)

## Reference adapters

### Simplest: generic.ts (start here)

Single turn from direct JSON. No detection logic needed (it's the fallback). Good starting point — copy and modify.

**File**: `neuralscape-plugin/src/adapters/generic.ts`

### Per-turn: openclaw.ts

Extracts one turn from an event payload. Demonstrates field mapping from a nested `context` object.

**File**: `neuralscape-plugin/src/adapters/openclaw.ts`

### Batch/transcript: claude-code.ts

Reads a transcript file, pairs user/assistant messages, tracks offset to avoid re-flushing. Demonstrates async extraction with file I/O.

**File**: `neuralscape-plugin/src/adapters/claude-code.ts`

## Checklist

Before submitting:

- [ ] Adapter file created at `src/adapters/{client-name}.ts`
- [ ] Exports `extract{ClientName}Turns` (returns `ConversationTurn[]`)
- [ ] Exports `extract{ClientName}SessionEnd` (returns `SessionEndInput`)
- [ ] `ClientType` union updated in `detect.ts`
- [ ] Detection condition added in `detectClient()` — before `"generic"` fallback
- [ ] Both `getTurnExtractor()` and `getSessionEndExtractor()` have new cases
- [ ] `session-end.ts` updated if client needs flush-before-compile
- [ ] Hook manifest created (if client uses one)
- [ ] `npm run build` succeeds
- [ ] Tested with sample stdin payload
- [ ] Channel name is unique and descriptive
