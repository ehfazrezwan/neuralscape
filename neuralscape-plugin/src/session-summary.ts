/**
 * Session summary hook — triggers end-of-session compilation in NeuralScape's
 * conversation-compiler extension.
 *
 * Fires on session:end (or OpenClaw equivalent) to compile the day's captured
 * conversation turns into durable memories.
 *
 * Accepts two input formats via stdin:
 *   1. Direct / testing:  { session_id, user_id, date }
 *   2. OpenClaw event:    { type, action, sessionKey, context: { ... } }
 *
 * Runs async so it never blocks the session teardown.
 */

import {
  getUserId,
  logError,
  neuralscapePost,
  outputContinue,
  parseStdin,
} from "./utils.js";

// ── Input types ──────────────────────────────────────────────────

interface SessionEndInput {
  // Direct invocation format
  session_id?: string;
  user_id?: string;
  date?: string;

  // OpenClaw InternalHookEvent format
  type?: string;
  action?: string;
  sessionKey?: string;
  timestamp?: string;
  context?: {
    sessionId?: string;
    messageCount?: number;
  };
}

// ── Main ─────────────────────────────────────────────────────────

async function main(): Promise<void> {
  // Output continue immediately so we never block the caller
  outputContinue();

  try {
    const raw = (await parseStdin()) as SessionEndInput;

    // If OpenClaw provides a messageCount, skip if no meaningful turns occurred
    const messageCount = raw.context?.messageCount;
    if (messageCount !== undefined && messageCount < 2) return;

    const userId = raw.user_id || getUserId();
    const date =
      raw.date ||
      (raw.timestamp
        ? new Date(raw.timestamp).toISOString().split("T")[0]
        : new Date().toISOString().split("T")[0]);

    await neuralscapePost("/v1/extensions/conversation-compiler/compile", {
      date,
      user_id: userId,
    }).catch((error) => {
      logError("Failed to trigger session compilation", error);
    });
  } catch (error) {
    logError("session-summary hook failed", error);
  }
}

main();
