/**
 * Shared compile trigger — tells NeuralScape to compile the day's facts.
 *
 * Used by ALL client adapters at session end. For Claude Code the Stop hook
 * fires after every assistant turn, so the trigger there is windowed (see
 * compileForSession) instead of one POST per turn.
 */

import {
  type CompileState,
  logError,
  neuralscapePost,
  readCompileState,
  readConfig,
  writeCompileState,
} from "../utils.js";
import type { SessionEndInput } from "./types.js";

/** Stop compiles a session's delivered turns at most this often (SessionEnd always compiles what is left). */
export const DEFAULT_COMPILE_INTERVAL_MIN = 15;

/** Minutes between Stop compiles (`COMPILE_INTERVAL_MIN`; 0 compiles after every Stop that delivered a turn). */
export function getCompileIntervalMin(): number {
  const parsed = parseFloat(readConfig("COMPILE_INTERVAL_MIN", ""));
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : DEFAULT_COMPILE_INTERVAL_MIN;
}

/**
 * Trigger compilation for the given date and user. Resolves true when the
 * service accepted the request (or there was nothing to ask for), false when
 * the POST failed.
 */
export async function triggerCompile(input: SessionEndInput): Promise<boolean> {
  if (!input.shouldCompile) return true;

  try {
    await neuralscapePost("/v1/extensions/conversation-compiler/compile", {
      date: input.date,
      user_id: input.userId,
    });
    return true;
  } catch (error) {
    logError("Failed to trigger compilation", error);
    return false;
  }
}

/**
 * Windowed compile for a Claude Code session.
 *
 * The service compiles a whole date from the turns it holds, idempotently, so
 * compiling after every turn recompiles the same day again and again. Instead
 * the session keeps a small state file: turns delivered mark their date
 * dirty; a compile is sent when something is dirty and either `final` (the
 * session is ending: nothing may be left behind) or the last successful
 * compile is at least `intervalMin` old. A failed POST leaves the date
 * dirty for the next try; a date older than today that is still dirty (the
 * session crossed midnight) is compiled too.
 *
 * Returns true when a compile request was sent.
 */
export async function compileForSession(
  sessionId: string,
  input: SessionEndInput,
  options: { delivered: number; final: boolean; intervalMin?: number; now?: number },
): Promise<boolean> {
  const now = options.now ?? Date.now();
  const intervalMs = (options.intervalMin ?? getCompileIntervalMin()) * 60_000;
  const state: CompileState = await readCompileState(sessionId);

  const before = JSON.stringify(state);
  if (options.delivered > 0 && !state.dirtyDates.includes(input.date)) {
    state.dirtyDates = [...state.dirtyDates, input.date].slice(-3);
  }

  const due = state.dirtyDates.length > 0 && (options.final || now - state.compiledAt >= intervalMs);
  if (!due) {
    if (JSON.stringify(state) !== before) await writeCompileState(sessionId, state);
    return false;
  }

  const failed: string[] = [];
  for (const date of state.dirtyDates) {
    if (!(await triggerCompile({ ...input, date }))) failed.push(date);
  }
  state.dirtyDates = failed;
  if (failed.length === 0) state.compiledAt = now;
  await writeCompileState(sessionId, state);
  return true;
}
