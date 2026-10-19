/**
 * PreCompact hook — compact-resilience snapshot.
 *
 * Claude Code fires this just before compacting the session (stdin:
 * `{session_id, transcript_path, cwd, hook_event_name: "PreCompact",
 * trigger: "manual"|"auto", custom_instructions}`). Compaction
 * lossy-summarizes in-context state, so before it happens this hook:
 *
 *   1. does NOT deliver conversation turns.  The Stop hook delivers them, and
 *      compaction adds a flagged summary entry to the transcript next to the entries it summarizes,
 *      so nothing is lost by leaving them for it; delivering here as well
 *      would race the Stop hook over the cursor and could split a turn that
 *      is still running;
 *   2. stores ONE small `task_context` "compact snapshot" memory (tagged
 *      `compact_snapshot`) via the EXISTING POST /v1/memories/raw
 *      endpoint. The compact-aware SessionStart re-injects it under
 *      "Resuming after compaction".
 *
 * Fire-and-forget: outputContinue() first, everything caught and logged —
 * this hook must NEVER block or fail the compact.
 */

import { extractRecentUserPrompts } from "../adapters/claude-code.js";
import { detectClient, getTurnExtractor } from "../adapters/detect.js";
import {
  buildCompactSnapshotContent,
  buildCompactSnapshotPayload,
} from "../core/compact.js";
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

    // 1. The snapshot shows the prompts that were under way, read from the transcript independently of what
    //    has been delivered (a prompt that is still being answered is included).  Other clients have no
    //    transcript: their extractor's turns (no cursor involved) supply the prompts as before.
    const client = detectClient(raw);
    const recentUserMessages =
      client === "claude-code"
        ? await extractRecentUserPrompts(raw)
        : (await getTurnExtractor(client)(raw)).map((t) => t.userMessage);

    // 2. One compact-snapshot memory so the post-compact SessionStart can
    //    re-anchor the session. Small, redacted, bounded.
    const content = buildCompactSnapshotContent({
      sessionId: (raw.session_id as string) || "unknown",
      projectId,
      trigger: (raw.trigger as string) || "auto",
      when: new Date(),
      recentUserMessages,
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
