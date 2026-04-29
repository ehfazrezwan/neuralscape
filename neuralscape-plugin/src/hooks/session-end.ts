/**
 * Session-end hook — unified entry point for all clients.
 *
 * For Claude Code: reads the transcript, batch-flushes all new turns,
 * then triggers compilation.
 * For OpenClaw / generic: triggers compilation directly.
 *
 * Runs async (fire-and-forget) so it never blocks session teardown.
 */

import {
  detectClient,
  getSessionEndExtractor,
  getTurnExtractor,
} from "../adapters/detect.js";
import { triggerCompile } from "../core/compile.js";
import { flushTurns } from "../core/flush.js";
import { logError, outputContinue, parseStdin } from "../utils.js";

async function main(): Promise<void> {
  outputContinue();

  try {
    const raw = (await parseStdin()) as Record<string, unknown>;
    const client = detectClient(raw);

    // Claude Code: flush all turns from transcript before compiling
    if (client === "claude-code") {
      const turnExtractor = getTurnExtractor(client);
      const turns = await turnExtractor(raw);
      await flushTurns(turns);
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
