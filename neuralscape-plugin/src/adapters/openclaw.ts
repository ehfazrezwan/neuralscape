/**
 * OpenClaw adapter — extracts conversation turns from OpenClaw's event format.
 *
 * Input: { type, action, sessionKey, context: { content, channelId, userMessage } }
 * Output: ConversationTurn[] (single element)
 */

import type { ConversationTurn, SessionEndInput } from "../core/types.js";
import { getUserId } from "../utils.js";

interface OpenClawContext {
  to?: string;
  content?: string;
  success?: boolean;
  channelId?: string;
  conversationId?: string;
  userMessage?: string;
  user_message?: string;
  sessionId?: string;
  messageCount?: number;
}

/**
 * Extract a single conversation turn from an OpenClaw message:sent event.
 */
export function extractOpenClawTurns(
  raw: Record<string, unknown>
): ConversationTurn[] {
  const ctx = (raw.context || {}) as OpenClawContext;

  return [
    {
      userMessage: ctx.userMessage || ctx.user_message || "",
      assistantResponse: ctx.content || "",
      sessionId: (raw.sessionKey as string) || "unknown",
      channel: ctx.channelId || "unknown",
      timestamp: (raw.timestamp as string) || new Date().toISOString(),
      projectId: undefined,
      userId: getUserId(),
    },
  ];
}

/**
 * Extract session-end metadata from an OpenClaw session:end event.
 */
export function extractOpenClawSessionEnd(
  raw: Record<string, unknown>
): SessionEndInput {
  const ctx = (raw.context || {}) as OpenClawContext;
  const messageCount = ctx.messageCount;

  const timestampDate = raw.timestamp ? new Date(raw.timestamp as string) : null;
  const date =
    timestampDate && !Number.isNaN(timestampDate.getTime())
      ? timestampDate.toISOString().split("T")[0]
      : new Date().toISOString().split("T")[0];

  return {
    date,
    userId: getUserId(),
    shouldCompile: messageCount === undefined || messageCount >= 2,
  };
}
