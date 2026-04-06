/**
 * Conversation turn hook — captures completed conversation turns from OpenClaw
 * and sends them to NeuralScape's conversation-compiler for automatic memory
 * extraction.
 *
 * Accepts two input formats via stdin:
 *   1. Direct / testing:  { user_message, assistant_response, session_id, ... }
 *   2. OpenClaw event:    { type, action, sessionKey, context: { content, channelId, ... } }
 *
 * Runs async (fire-and-forget) so it never blocks OpenClaw's response delivery.
 */

import {
  getUserId,
  isHeartbeat,
  isNoReply,
  isSystemMessage,
  logError,
  neuralscapePost,
  outputContinue,
  parseStdin,
} from "./utils.js";

// Responses shorter than this are likely not worth capturing
const MIN_RESPONSE_LENGTH = 20;

// ── Input types ──────────────────────────────────────────────────

interface DirectInput {
  user_message?: string;
  assistant_response?: string;
  session_id?: string;
  channel?: string;
  timestamp?: string;
  project_id?: string | null;
  user_id?: string;
}

interface OpenClawEventInput {
  type?: string;
  action?: string;
  sessionKey?: string;
  timestamp?: string;
  context?: {
    to?: string;
    content?: string;
    success?: boolean;
    channelId?: string;
    conversationId?: string;
    // Extended fields — when OpenClaw pairs the inbound message
    userMessage?: string;
    user_message?: string;
  };
}

type ConversationTurnInput = DirectInput & OpenClawEventInput;

interface NormalizedTurn {
  userMessage: string;
  assistantResponse: string;
  sessionId: string;
  channel: string;
  timestamp: string;
  projectId: string | null;
  userId: string;
}

// ── Normalization ────────────────────────────────────────────────

function normalizeInput(raw: ConversationTurnInput): NormalizedTurn {
  // Direct invocation format (preferred)
  if (raw.user_message !== undefined || raw.assistant_response !== undefined) {
    return {
      userMessage: raw.user_message || "",
      assistantResponse: raw.assistant_response || "",
      sessionId: raw.session_id || "unknown",
      channel: raw.channel || "unknown",
      timestamp: raw.timestamp || new Date().toISOString(),
      projectId: raw.project_id ?? null,
      userId: raw.user_id || getUserId(),
    };
  }

  // OpenClaw InternalHookEvent format
  const ctx = raw.context || {};
  return {
    userMessage: ctx.userMessage || ctx.user_message || "",
    assistantResponse: ctx.content || "",
    sessionId: raw.sessionKey || "unknown",
    channel: ctx.channelId || "unknown",
    timestamp: raw.timestamp || new Date().toISOString(),
    projectId: null,
    userId: getUserId(),
  };
}

// ── Filtering ────────────────────────────────────────────────────

function shouldSkipTurn(userMessage: string, assistantResponse: string): boolean {
  if (isHeartbeat(userMessage)) return true;
  if (isNoReply(assistantResponse)) return true;
  if (isSystemMessage(userMessage)) return true;
  if (assistantResponse.length < MIN_RESPONSE_LENGTH) return true;
  return false;
}

// ── Main ─────────────────────────────────────────────────────────

async function main(): Promise<void> {
  // Output continue immediately so we never block the caller
  outputContinue();

  try {
    const raw = (await parseStdin()) as ConversationTurnInput;
    const turn = normalizeInput(raw);

    if (shouldSkipTurn(turn.userMessage, turn.assistantResponse)) return;

    await neuralscapePost("/v1/extensions/conversation-compiler/flush", {
      user_message: turn.userMessage,
      assistant_response: turn.assistantResponse,
      session_id: turn.sessionId,
      channel: turn.channel,
      timestamp: turn.timestamp,
      project_id: turn.projectId,
      user_id: turn.userId,
    }).catch((error) => {
      logError("Failed to flush conversation turn", error);
    });
  } catch (error) {
    logError("conversation-turn hook failed", error);
  }
}

main();
