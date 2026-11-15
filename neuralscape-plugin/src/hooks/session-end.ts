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

import {
  detectClient,
  getSessionEndExtractor,
  resolveEventProjectId,
} from "../adapters/detect.js";
import { compileForSession, triggerCompile } from "../core/compile.js";
import { deliverClaudeCodeTurns } from "../core/deliver.js";
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


    const sessionExtractor = getSessionEndExtractor(client);
    const input = await sessionExtractor(raw);

    // Claude Code: flush all turns from transcript before compiling.
    // The offset commit is bounded by the flush RESULT (audit 27 #34b):
    // full success commits the whole window, a partial failure commits only
    // past the delivered prefix, and an all-failed flush leaves the cursor
    // untouched so the next session re-flushes everything.
    // Stop fires after EVERY assistant turn, so the compile request is
    // windowed per session (see compileForSession); the SessionEnd hook
    // sends whatever is still uncompiled when the session ends.
    if (client === "claude-code") {
      const delivered = await deliverClaudeCodeTurns(raw);
      await compileForSession((raw.session_id as string) || "unknown", input, { delivered, final: false });
      return;
    }

    // Other clients (a session:end event, a generic POST): one trigger at the end of the session.
    await triggerCompile(input);
  } catch (error) {
    logError("session-end hook failed", error);
  }
}

main();
