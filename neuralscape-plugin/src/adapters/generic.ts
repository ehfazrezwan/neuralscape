/**
 * Generic adapter — extracts a conversation turn from direct JSON input.
 *
 * Input: { user_message, assistant_response, session_id, channel, ... }
 * Output: ConversationTurn[] (single element)
 *
 * Use this for manual testing or future agent frameworks that send
 * the normalized format directly.
 */

import type { ConversationTurn, SessionEndInput } from "../core/types.js";
import { getUserId } from "../utils.js";

/**
 * Extract a single conversation turn from direct JSON input.
 */
export function extractGenericTurns(
  raw: Record<string, unknown>
): ConversationTurn[] {
  return [
    {
      userMessage: (raw.user_message as string) || "",
      assistantResponse: (raw.assistant_response as string) || "",
      sessionId: (raw.session_id as string) || "unknown",
      channel: (raw.channel as string) || "api",
      timestamp: (raw.timestamp as string) || new Date().toISOString(),
      projectId: (raw.project_id as string) || undefined,
      userId: (raw.user_id as string) || getUserId(),
    },
  ];
}

/**
 * Extract session-end metadata from direct JSON input.
 */
export function extractGenericSessionEnd(
  raw: Record<string, unknown>
): SessionEndInput {
  const date =
    (raw.date as string) || new Date().toISOString().split("T")[0];

  return {
    date,
    userId: (raw.user_id as string) || getUserId(),
    shouldCompile: true,
  };
}
