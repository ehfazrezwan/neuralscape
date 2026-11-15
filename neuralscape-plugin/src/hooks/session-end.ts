/**
 * Session-end hook — unified entry point for all clients.
 *
 * For Claude Code: reads the transcript, batch-flushes all new turns,
 * then triggers compilation.
 * For OpenClaw / generic: triggers compilation directly.
 *
 * Runs async (fire-and-forget) so it never blocks session teardown.
 *
 * Registered on Stop, which fires after EVERY assistant turn, not once at the
 * end of the session. It therefore must not mark the observation buffer as
 * finished: that marker is written by the SessionEnd hook (session-summary).
 */

import { commitClaudeCodeFlush } from "../adapters/claude-code.js";
import {
  detectClient,
  getSessionEndExtractor,
  resolveEventProjectId,
  getTurnExtractor,
} from "../adapters/detect.js";
import { triggerCompile } from "../core/compile.js";
import { flushTurns } from "../core/flush.js";
import { type LockHandle, withTranscriptLock } from "../core/transcript-lock.js";
import {
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
      "missing user_id — run `/plugin config neuralscape@neuralscape-plugins` to set USER_ID (or set NEURALSCAPE_USER_ID env var as legacy fallback); skipping flush",
    );
    return;
  }

  try {
    const raw = (await parseStdin()) as Record<string, unknown>;

    // Excluded projects (D4): no conversation capture leaves an excluded project.
    const client = detectClient(raw);
    if (isProjectExcluded(resolveEventProjectId(raw, client))) return;


    // Claude Code: flush all turns from transcript before compiling.
    // The offset commit is bounded by the flush RESULT (audit 27 #34b):
    // full success commits the whole window, a partial failure commits only
    // past the delivered prefix, and an all-failed flush leaves the cursor
    // untouched so the next session re-flushes everything.
    if (client === "claude-code") {
      const turnExtractor = getTurnExtractor(client);
      const deliver = async (handle?: LockHandle): Promise<void> => {
        const turns = await turnExtractor(raw);
        const result = await flushTurns(turns, { proceed: handle?.held });
        await commitClaudeCodeFlush(raw, result, handle?.held);
      };
      // Read cursor, deliver and write cursor as one step per transcript: an overlapping hook waits for it.
      const transcriptPath = raw.transcript_path as string | undefined;
      if (transcriptPath) await withTranscriptLock(transcriptPath, deliver);
      else await deliver();
    }

    // All clients: trigger compilation of conversation-compiler
    const sessionExtractor = getSessionEndExtractor(client);
    const input = await sessionExtractor(raw);
    await triggerCompile(input);
  } catch (error) {
    logError("session-end hook failed", error);
  }
}

main();
