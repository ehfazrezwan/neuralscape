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
  redactPrivate,
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
 * Delivery accounting for a flush (audit 27 #34b): callers that maintain a
 * transcript cursor must advance it only past the successfully-delivered
 * prefix, so the result reports where delivery stopped.
 */
export interface FlushResult {
  /** Turns handed in. */
  attempted: number;
  /** POSTs that succeeded (noise-skipped turns are not counted here). */
  flushed: number;
  /**
   * Index (into the input array) of the first turn whose POST failed, or
   * null when every non-skipped turn was delivered. Turns after this index
   * were NOT attempted — they stay re-flushable next session.
   */
  firstFailedIndex: number | null;
}

/**
 * Flush conversation turns to the conversation-compiler extension.
 *
 * Filters noise, then POSTs each turn sequentially, stopping at the first
 * failure (if the service is down, later POSTs would fail too, and stopping
 * keeps the undelivered turns a contiguous re-flushable suffix). Never
 * throws — failures are logged and reported via the result.
 */
export async function flushTurns(turns: ConversationTurn[]): Promise<FlushResult> {
  let flushed = 0;

  for (let i = 0; i < turns.length; i++) {
    const turn = turns[i];
    // Noise counts as delivered: there is nothing to re-flush later.
    if (shouldSkipTurn(turn)) continue;

    try {
      // <private>…</private> spans never leave the machine (D4).
      await neuralscapePost("/v1/extensions/conversation-compiler/flush", {
        user_message: redactPrivate(turn.userMessage),
        assistant_response: redactPrivate(turn.assistantResponse),
        session_id: turn.sessionId,
        channel: turn.channel,
        timestamp: turn.timestamp,
        project_id: turn.projectId ?? null,
        user_id: turn.userId,
      });
      flushed++;
    } catch (error) {
      logError(`Failed to flush conversation turn ${i + 1}/${turns.length}; leaving the rest for the next flush`, error);
      return { attempted: turns.length, flushed, firstFailedIndex: i };
    }
  }

  return { attempted: turns.length, flushed, firstFailedIndex: null };
}
