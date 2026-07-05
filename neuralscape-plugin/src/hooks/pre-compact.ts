/**
 * PreCompact hook — compact-resilience snapshot.
 *
 * Claude Code fires this just before compacting the session (stdin:
 * `{session_id, transcript_path, cwd, hook_event_name: "PreCompact",
 * trigger: "manual"|"auto", custom_instructions}`). Compaction
 * lossy-summarizes in-context state, so before it happens this hook:
 *
 *   1. flushes any not-yet-delivered conversation turns through the SAME
 *      adapter path as the conversation-turn hook — including the
 *      offset-commit step, so the cursor advances only past delivered
 *      turns (audit 27 #34b) and nothing said pre-compact is lost to the
 *      turn-capture pipeline;
 *   2. stores ONE small `task_context` "compact snapshot" memory (tagged
 *      `compact_snapshot`) via the EXISTING POST /v1/memories/raw
 *      endpoint. The compact-aware SessionStart re-injects it under
 *      "Resuming after compaction".
 *
 * Fire-and-forget: outputContinue() first, everything caught and logged —
 * this hook must NEVER block or fail the compact.
 */

import { commitClaudeCodeFlush } from "../adapters/claude-code.js";
import { detectClient, getTurnExtractor } from "../adapters/detect.js";
import {
  buildCompactSnapshotContent,
  buildCompactSnapshotPayload,
} from "../core/compact.js";
import { flushTurns } from "../core/flush.js";
import {
  getProjectId,
  getUserId,
  hasUserId,
  isProjectExcluded,
  logError,
  neuralscapePost,
  outputContinue,
  parseStdin,
  recordTransportFailure,
  resetTransportFailures,
} from "../utils.js";

async function main(): Promise<void> {
  outputContinue();

  if (!hasUserId()) {
    logError(
      "missing user_id — run `/plugin config neuralscape@neuralscape-plugins` to set USER_ID (or set NEURALSCAPE_USER_ID env var as legacy fallback); skipping compact snapshot",
    );
    return;
  }

  try {
    const raw = (await parseStdin()) as Record<string, unknown>;

    const projectId = getProjectId(raw.cwd as string | undefined);

    // Excluded projects (D4): nothing leaves an excluded project.
    if (isProjectExcluded(projectId)) return;

    // 1. Flush undelivered turns — mirrors conversation-turn.ts exactly,
    //    including the offset commit bounded by delivered turns (#34b).
    const client = detectClient(raw);
    const extractor = getTurnExtractor(client);
    const turns = await extractor(raw);
    const result = await flushTurns(turns);
    if (client === "claude-code") {
      await commitClaudeCodeFlush(raw, result);
    }

    // 2. One compact-snapshot memory so the post-compact SessionStart can
    //    re-anchor the session. Small, redacted, bounded.
    const content = buildCompactSnapshotContent({
      sessionId: (raw.session_id as string) || "unknown",
      projectId,
      trigger: (raw.trigger as string) || "auto",
      when: new Date(),
      capturedTurns: result.flushed,
      recentUserMessages: turns.map((t) => t.userMessage),
    });

    try {
      await neuralscapePost(
        "/v1/memories/raw",
        buildCompactSnapshotPayload(content, getUserId(), projectId),
      );
      await resetTransportFailures();
    } catch (error) {
      // Transport failure → log + count toward the fail-loud threshold
      // surfaced at next SessionStart. The compact itself is unaffected.
      await recordTransportFailure();
      logError("compact snapshot store failed (service unreachable?)", error);
    }
  } catch (error) {
    logError("pre-compact hook failed", error);
  }
}

// Skip auto-running main() when imported by the test harness.
if (process.env.NEURALSCAPE_TEST !== "1") {
  main();
}
