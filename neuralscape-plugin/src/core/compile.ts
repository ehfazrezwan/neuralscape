/**
 * Shared compile trigger — tells NeuralScape to compile the day's facts.
 *
 * Used by ALL client adapters at session end.
 */

import { logError, neuralscapePost } from "../utils.js";
import type { SessionEndInput } from "./types.js";

/**
 * Trigger compilation for the given date and user.
 */
export async function triggerCompile(input: SessionEndInput): Promise<void> {
  if (!input.shouldCompile) return;

  try {
    await neuralscapePost("/v1/extensions/conversation-compiler/compile", {
      date: input.date,
      user_id: input.userId,
    });
  } catch (error) {
    logError("Failed to trigger compilation", error);
  }
}
