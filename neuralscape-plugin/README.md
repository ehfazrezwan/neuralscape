# NeuralScape Plugin

Persistent memory hooks for AI agent platforms. Automatically captures context and observations during sessions and sends them to [NeuralScape](https://github.com/ehfazrezwan/neuralscape) for long-term memory storage.

## Supported Platforms

| Platform   | Hooks                                          |
|------------|-------------------------------------------------|
| Claude Code | SessionStart, PostToolUse, Stop                |
| OpenClaw   | message:sent (conversation turn), session:end  |

## Setup

### Prerequisites

- Node.js 18+
- A running NeuralScape instance (default: `http://localhost:8199`)

### Install & Build

```bash
cd neuralscape-plugin
npm install
npm run build
```

This generates bundled scripts in `scripts/`.

### Environment Variables

| Variable              | Default                  | Description                        |
|-----------------------|--------------------------|------------------------------------|
| `NEURALSCAPE_URL`     | `http://localhost:8199`  | NeuralScape API base URL           |
| `NEURALSCAPE_API_KEY` | *(empty)*                | Bearer token for authenticated APIs |
| `NEURALSCAPE_USER_ID` | `ehfaz`                  | Default user ID for memory storage |

## Claude Code Integration

Install the plugin via Claude Code's plugin system. The hooks manifest at `hooks/hooks.json` registers three hooks:

- **SessionStart** — Fetches stored memories and injects them as context
- **PostToolUse** — Captures tool observations (writes, edits, commands) as raw memories
- **Stop** — Marks session boundaries

## OpenClaw Integration

The OpenClaw hooks capture conversation turns and trigger end-of-session memory compilation via NeuralScape's `conversation-compiler` extension.

### Hook Manifest

The `hooks/openclaw-hooks.json` manifest defines two hooks:

- **message:sent** — Fires after every assistant response. Sends the conversation turn (user message + assistant response) to the conversation-compiler flush endpoint.
- **session:end** — Fires when a session ends. Triggers compilation of the day's captured turns into durable memories.

### Installing as OpenClaw Managed Hooks

OpenClaw discovers hooks from `~/.openclaw/hooks/`. To install the conversation-turn hook as a managed hook, create a directory with a `HOOK.md` and `handler.ts`:

```
~/.openclaw/hooks/neuralscape-conversation-turn/
├── HOOK.md
└── handler.ts
```

**HOOK.md:**
```markdown
---
name: neuralscape-conversation-turn
description: "Sends conversation turns to NeuralScape conversation-compiler"
metadata:
  openclaw:
    events: ["message:sent"]
    requires:
      env: ["NEURALSCAPE_URL"]
    always: true
---
```

**handler.ts** — Wrap the built script or inline the logic:
```typescript
import { execFile } from "node:child_process";
import { resolve } from "node:path";

const SCRIPT = resolve(
  process.env.HOME || "~",
  "path/to/neuralscape-plugin/scripts/conversation-turn.js"
);

export default async (event: any) => {
  if (event.type !== "message" || event.action !== "sent") return;

  const child = execFile("node", [SCRIPT], { timeout: 15000 });
  child.stdin?.write(JSON.stringify(event));
  child.stdin?.end();
};
```

Repeat for `session-summary` with event `session:end`.

### Stdin Payload

When invoked, the scripts read JSON from stdin. They accept two formats:

**Direct invocation (for testing):**
```json
{
  "user_message": "What is the weather?",
  "assistant_response": "I don't have access to weather data...",
  "session_id": "session-abc123",
  "channel": "telegram",
  "timestamp": "2026-04-07T05:00:00.000Z",
  "project_id": null,
  "user_id": "ehfaz"
}
```

**OpenClaw InternalHookEvent:**
```json
{
  "type": "message",
  "action": "sent",
  "sessionKey": "session-abc123",
  "timestamp": "2026-04-07T05:00:00.000Z",
  "context": {
    "content": "I don't have access to weather data...",
    "channelId": "telegram",
    "userMessage": "What is the weather?"
  }
}
```

### Filtering

The conversation-turn hook skips trivial exchanges to avoid noise:

- **Heartbeats** — Messages like `.`, `ping`, `heartbeat`, `/heartbeat`
- **NO_REPLY responses** — `NO_REPLY`, `[NO_REPLY]`, empty responses
- **System messages** — Prefixed with `[system]`, `[auto_reply]`, `[internal]`, etc.
- **Short responses** — Responses under 20 characters

### Verifying It Works

1. Start NeuralScape: `cd neuralscape-service && python main.py`
2. Send a test turn:
   ```bash
   echo '{"user_message":"hello","assistant_response":"Hi! How can I help you today? Let me know what you need.","session_id":"test","channel":"cli"}' | node scripts/conversation-turn.js
   ```
3. Check NeuralScape logs for the flush request
4. Trigger compilation:
   ```bash
   echo '{"date":"2026-04-07","user_id":"ehfaz"}' | node scripts/session-summary.js
   ```

### API Endpoints Used

| Endpoint                                          | Method | Purpose                          |
|--------------------------------------------------|--------|----------------------------------|
| `/v1/extensions/conversation-compiler/flush`     | POST   | Send a single conversation turn  |
| `/v1/extensions/conversation-compiler/compile`   | POST   | Trigger end-of-session compilation |
| `/v1/context/{project_id}`                       | GET    | Fetch stored context (Claude Code) |
| `/v1/memories/raw`                               | POST   | Store raw observations (Claude Code) |
