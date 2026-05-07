/**
 * Conversation turn hook — unified entry point for all clients.
 *
 * Auto-detects the client (OpenClaw, Claude Code, or generic),
 * extracts conversation turns via the appropriate adapter,
 * and flushes them through the shared core pipeline.
 *
 * Runs async (fire-and-forget) so it never blocks the caller.
 */

import { detectClient, getTurnExtractor } from "../adapters/detect.js";
import { flushTurns } from "../core/flush.js";
import { hasUserId, logError, outputContinue, parseStdin } from "../utils.js";

async function main(): Promise<void> {
  outputContinue();

  if (!hasUserId()) {
    logError(
      "missing user_id — set NEURALSCAPE_USER_ID via /plugin config or env var; skipping turn capture",
    );
    return;
  }

  try {
    const raw = (await parseStdin()) as Record<string, unknown>;
    const client = detectClient(raw);
    const extractor = getTurnExtractor(client);
    const turns = await extractor(raw);
    await flushTurns(turns);
  } catch (error) {
    logError("conversation-turn hook failed", error);
  }
}

main();
