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
import {
  getBufferPath,
  getBufferStats,
  hasUserId,
  logError,
  markBufferStale,
  outputContinue,
  parseStdin,
} from "../utils.js";

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

    // All clients: trigger compilation of conversation-compiler
    const sessionExtractor = getSessionEndExtractor(client);
    const input = await sessionExtractor(raw);
    await triggerCompile(input);

    // Mark this session's PostToolUse observation buffer as stale so the
    // next SessionStart picks it up if compile-observations didn't already
    // flush it via the UserPromptSubmit threshold during this session.
    try {
      const sessionId = (raw.session_id as string | undefined) || "unknown";
      const stats = await getBufferStats(getBufferPath(sessionId));
      if (stats.lineCount > 0) {
        await markBufferStale(sessionId);
      }
    } catch (error) {
      logError("stale-marker write failed (non-critical)", error);
    }
  } catch (error) {
    logError("session-end hook failed", error);
  }
}

main();
