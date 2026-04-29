/**
 * Shared flush logic — filters and sends conversation turns to NeuralScape.
 *
 * Used by ALL client adapters. This is the single processing path
 * regardless of whether the turn came from OpenClaw, Claude Code, or
 * a future agent framework.
 */

import {
  isHeartbeat,
  isNoReply,
  isSystemMessage,
  logError,
  neuralscapePost,
} from "../utils.js";
import type { ConversationTurn } from "./types.js";

/** Responses shorter than this are likely not worth capturing. */
const MIN_RESPONSE_LENGTH = 20;

/**
 * Check if a conversation turn should be skipped (noise filtering).
 */
export function shouldSkipTurn(turn: ConversationTurn): boolean {
  const trimmed = turn.assistantResponse.trim();

  if (isHeartbeat(turn.userMessage)) return true;
  if (isNoReply(turn.assistantResponse)) return true;
  if (isSystemMessage(turn.userMessage)) return true;
  if (isSystemMessage(turn.assistantResponse)) return true;
  if (trimmed.length < MIN_RESPONSE_LENGTH) return true;
  return false;
}

/**
 * Flush conversation turns to the conversation-compiler extension.
 *
 * Filters noise, then POSTs each turn sequentially. Errors are logged
 * but do not throw — this is fire-and-forget.
 */
export async function flushTurns(turns: ConversationTurn[]): Promise<number> {
  let flushed = 0;

  for (const turn of turns) {
    if (shouldSkipTurn(turn)) continue;

    try {
      await neuralscapePost("/v1/extensions/conversation-compiler/flush", {
        user_message: turn.userMessage,
        assistant_response: turn.assistantResponse,
        session_id: turn.sessionId,
        channel: turn.channel,
        timestamp: turn.timestamp,
        project_id: turn.projectId ?? null,
        user_id: turn.userId,
      });
      flushed++;
    } catch (error) {
      logError("Failed to flush conversation turn", error);
    }
  }

  return flushed;
}
