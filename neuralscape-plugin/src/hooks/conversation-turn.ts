/**
 * Conversation turn hook — unified entry point for all clients.
 *
 * Auto-detects the client (OpenClaw, Claude Code, or generic),
 * extracts conversation turns via the appropriate adapter,
 * and flushes them through the shared core pipeline.
 *
 * Runs async (fire-and-forget) so it never blocks the caller.
 */

import { commitClaudeCodeFlush } from "../adapters/claude-code.js";
import { detectClient, getTurnExtractor } from "../adapters/detect.js";
import { flushTurns } from "../core/flush.js";
import {
  getProjectId,
  hasUserId,
  isProjectExcluded,
  logError,
  outputContinue,
  parseStdin,
} from "../utils.js";

async function main(): Promise<void> {
  outputContinue();

  if (!hasUserId()) {
    logError(
      "missing user_id — run `/plugin config neuralscape@neuralscape-plugins` to set USER_ID (or set NEURALSCAPE_USER_ID env var as legacy fallback); skipping turn capture",
    );
    return;
  }

  try {
    const raw = (await parseStdin()) as Record<string, unknown>;

    // Excluded projects (D4): no turn capture from an excluded project.
    if (isProjectExcluded(getProjectId(raw.cwd as string | undefined))) return;

    const client = detectClient(raw);
    const extractor = getTurnExtractor(client);
    const turns = await extractor(raw);
    const result = await flushTurns(turns);
    // Persist the transcript cursor only past delivered turns (audit 27
    // #34b) — without this commit the offset file is never written and
    // every flush re-reads the transcript from zero.
    if (client === "claude-code") {
      await commitClaudeCodeFlush(raw, result);
    }
  } catch (error) {
    logError("conversation-turn hook failed", error);
  }
}

main();
