/**
 * Shared flush logic — filters and sends conversation turns to NeuralScape.
 *
 * Used by ALL client adapters. This is the single processing path
 * regardless of whether the turn came from OpenClaw, Claude Code, or
 * a future agent framework.
 */

import {
  HttpStatusError,
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
 * The flush route accepts at most this many characters in each of ``user_message`` and ``assistant_response``
 * (the service's request schema).  JavaScript counts UTF-16 code units, which is never fewer than the service's
 * code points, so a text within this length is always accepted.
 */
export const MAX_FLUSH_TEXT = 50000;

/**
 * Bound ``value`` to ``limit`` characters.  Text beyond the limit is cut and replaced by a marker that says how
 * much was left out, so the stored turn shows plainly that it is partial.  Never cuts a surrogate pair in two.
 */
export function boundText(value: string, limit: number = MAX_FLUSH_TEXT): string {
  if (value.length <= limit) return value;
  const omitted = (n: number) => `\n[truncated by the plugin: ${n} more characters]`;
  let keep = limit - omitted(value.length).length;
  const unit = value.charCodeAt(keep - 1);
  if (unit >= 0xd800 && unit <= 0xdbff) keep -= 1; // the cut would split a surrogate pair
  return value.slice(0, keep) + omitted(value.length - keep);
}

/** HTTP statuses that mean the service refused THIS payload for good (retrying cannot help): bad request, too large, invalid. */
const PERMANENT_REJECTIONS = new Set([400, 413, 422]);

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
        user_message: boundText(redactPrivate(turn.userMessage)),
        assistant_response: boundText(redactPrivate(turn.assistantResponse)),
        session_id: turn.sessionId,
        channel: turn.channel,
        timestamp: turn.timestamp,
        project_id: turn.projectId ?? null,
        user_id: turn.userId,
      });
      flushed++;
    } catch (error) {
      if (error instanceof HttpStatusError && PERMANENT_REJECTIONS.has(error.status)) {
        // The service will never accept this turn: skip it (like noise) so it cannot block every later turn.
        logError(`Skipping conversation turn ${i + 1}/${turns.length}: the service rejected it (HTTP ${error.status}) and will not accept it on retry`);
        continue;
      }
      logError(`Failed to flush conversation turn ${i + 1}/${turns.length}; leaving the rest for the next flush`, error);
      return { attempted: turns.length, flushed, firstFailedIndex: i };
    }
  }

  return { attempted: turns.length, flushed, firstFailedIndex: null };
}
