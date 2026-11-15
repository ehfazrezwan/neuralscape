/**
 * PreCompact hook — compact-resilience snapshot.
 *
 * Claude Code fires this just before compacting the session (stdin:
 * `{session_id, transcript_path, cwd, hook_event_name: "PreCompact",
 * trigger: "manual"|"auto", custom_instructions}`). Compaction
 * lossy-summarizes in-context state, so before it happens this hook:
 *
 *   1. Claude Code: does NOT deliver conversation turns.  The Stop hook
 *      delivers them, and compaction adds a flagged summary entry to the
 *      transcript next to the entries it summarizes, so nothing is lost by
 *      leaving them for it; delivering here as well would race the Stop hook
 *      over the cursor and could split a turn that is still running.
 *      Other clients (generic payload, no transcript and no cursor): the turn
 *      arrives in the payload and no other hook is guaranteed to deliver it
 *      before the compact, so it is flushed here;
 *   2. stores ONE small `task_context` "compact snapshot" memory (tagged
 *      `compact_snapshot`, plus `session:<id>`) via the EXISTING
 *      POST /v1/memories/raw endpoint, and keeps a copy next to the
 *      observation buffers. That endpoint only enqueues the write (202), so
 *      the SessionStart that follows the compaction may run before the
 *      memory is searchable; the compact-aware SessionStart reads the local
 *      copy first and re-injects it under "Resuming after compaction".
 *
 * Fire-and-forget: outputContinue() first, everything caught and logged —
 * this hook must NEVER block or fail the compact.
 */

import { extractRecentUserPrompts } from "../adapters/claude-code.js";
import { detectClient, getTurnExtractor } from "../adapters/detect.js";
import { flushTurns } from "../core/flush.js";
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
  writeLocalCompactSnapshot,
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

    // 1. Claude Code: the snapshot shows the prompts that were under way, read from the transcript independently
    //    of what has been delivered (a prompt that is still being answered is included); the Stop hook owns
    //    delivery and the cursor.  Other clients have no transcript and no cursor: their extractor's turns
    //    supply the prompts, and are delivered here (once, no cursor involved) before the context is lost.
    const client = detectClient(raw);
    let recentUserMessages: string[];
    let turnsToDeliver: Awaited<ReturnType<ReturnType<typeof getTurnExtractor>>> = [];
    if (client === "claude-code") {
      recentUserMessages = await extractRecentUserPrompts(raw);
    } else {
      turnsToDeliver = await getTurnExtractor(client)(raw);
      recentUserMessages = turnsToDeliver.map((t) => t.userMessage);
    }

    // 2. One compact-snapshot memory so the post-compact SessionStart can
    //    re-anchor the session. Small, redacted, bounded.
    const sessionId = (raw.session_id as string) || "unknown";
    const when = new Date();
    const content = buildCompactSnapshotContent({
      sessionId,
      projectId,
      trigger: (raw.trigger as string) || "auto",
      when,
      recentUserMessages,
    });

    // The local copy comes FIRST: it needs no network, so a hanging service
    // (this hook has a 60 s timeout) cannot prevent it.
    if (sessionId !== "unknown") {
      await writeLocalCompactSnapshot({
        sessionId,
        projectId,
        content,
        createdAt: when.toISOString(),
      });
    }

    // Clients with no transcript: deliver the turn here (see the header).
    if (turnsToDeliver.length > 0) await flushTurns(turnsToDeliver);

    try {
      await neuralscapePost(
        "/v1/memories/raw",
        buildCompactSnapshotPayload(content, getUserId(), projectId, when, sessionId),
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
