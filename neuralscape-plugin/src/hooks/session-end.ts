/**
 * Session-end hook — unified entry point for all clients.
 *
 * For Claude Code: reads the transcript, batch-flushes all new turns,
 * then triggers compilation.
 * For OpenClaw / generic: triggers compilation directly.
 *
 * Runs async (fire-and-forget) so it never blocks session teardown.
 */

import { commitClaudeCodeFlush } from "../adapters/claude-code.js";
import {
  detectClient,
  getSessionEndExtractor,
  getTurnExtractor,
} from "../adapters/detect.js";
import { triggerCompile } from "../core/compile.js";
import { flushTurns } from "../core/flush.js";
import { hasUserId, logError, outputContinue, parseStdin } from "../utils.js";

async function main(): Promise<void> {
  outputContinue();

  if (!hasUserId()) {
    logError(
      "missing user_id — run `/plugin config neuralscape@neuralscape-plugins` to set USER_ID (or set NEURALSCAPE_USER_ID env var as legacy fallback); skipping flush",
    );
    return;
  }

  try {
    const raw = (await parseStdin()) as Record<string, unknown>;
    const client = detectClient(raw);

    // Claude Code: flush all turns from transcript before compiling.
    // Commit the transcript offset only after flushTurns returns so a
    // crash mid-flush leaves the cursor at its prior position.
    if (client === "claude-code") {
      const turnExtractor = getTurnExtractor(client);
      const turns = await turnExtractor(raw);
      await flushTurns(turns);
      await commitClaudeCodeFlush(raw);
    }

    // All clients: trigger compilation
    const sessionExtractor = getSessionEndExtractor(client);
    const input = await sessionExtractor(raw);
    await triggerCompile(input);
  } catch (error) {
    logError("session-end hook failed", error);
  }
}

main();
