/**
 * Compact-snapshot builders (compact-resilience loop).
 *
 * Compaction lossy-summarizes a session's in-context state. Just before it
 * happens, the PreCompact hook stores ONE small `task_context` memory — a
 * "compact snapshot" — via the EXISTING POST /v1/memories/raw endpoint
 * (pre-categorized fact, no LLM extraction, no new server surface). The
 * compact-aware SessionStart re-injects it under "Resuming after
 * compaction" when the post-compact session starts.
 *
 * Pure module: no I/O, fully unit-tested. The hook entry point owns stdin,
 * flushing, and the POST.
 */

import { redactPrivate } from "../utils.js";

/** Tag every snapshot carries so SessionStart can pick them out. */
export const COMPACT_SNAPSHOT_TAG = "compact_snapshot";

/** Content prefix — also the fallback detector for tag-less legacy rows. */
export const COMPACT_SNAPSHOT_MARKER = "Compact snapshot:";

/** Hard cap on the whole snapshot body — this is a marker, not a transcript. */
export const MAX_SNAPSHOT_CHARS = 1500;

/** How many trailing user messages the snapshot preserves. */
const MAX_TAIL_MESSAGES = 3;

/** Per-message cap inside the tail block. */
const MAX_MESSAGE_CHARS = 300;

/** Snapshots are continuity artifacts — expire them after two weeks. */
const SNAPSHOT_TTL_MS = 14 * 24 * 60 * 60 * 1000;

export interface CompactSnapshotInputs {
  sessionId: string;
  projectId?: string;
  /** "manual" (/compact) or "auto" (context-window pressure). */
  trigger: string;
  when: Date;
  /** Turns the pre-compact flush actually delivered to NS. */
  capturedTurns: number;
  /** The last few user messages before the compact (tail is kept). */
  recentUserMessages: string[];
}

/** Squash a message to a single redacted line, bounded to `maxChars`. */
function squashMessage(text: string, maxChars = MAX_MESSAGE_CHARS): string {
  const clean = redactPrivate(text).replace(/\s+/g, " ").trim();
  if (clean.length <= maxChars) return clean;
  return clean.slice(0, maxChars - 1).trimEnd() + "…";
}

/**
 * Build the snapshot body: one marker line naming session/project/trigger/
 * time/turn-count, then the tail of the last user messages as bullets.
 * `<private>…</private>` spans never leave the machine (D4). Bounded to
 * MAX_SNAPSHOT_CHARS no matter what the transcript held.
 */
export function buildCompactSnapshotContent(inputs: CompactSnapshotInputs): string {
  const project = inputs.projectId || "(no project)";
  const lines: string[] = [
    `${COMPACT_SNAPSHOT_MARKER} session ${inputs.sessionId} in project ${project} ` +
      `compacted (${inputs.trigger}) at ${inputs.when.toISOString()} ` +
      `after ${inputs.capturedTurns} captured turn(s).`,
  ];

  const tail = inputs.recentUserMessages
    .map((m) => squashMessage(m ?? ""))
    .filter((m) => m.length > 0)
    .slice(-MAX_TAIL_MESSAGES);

  if (tail.length > 0) {
    lines.push("Last user messages before compaction:");
    for (const msg of tail) lines.push(`- ${msg}`);
  }

  const content = lines.join("\n");
  if (content.length <= MAX_SNAPSHOT_CHARS) return content;
  return content.slice(0, MAX_SNAPSHOT_CHARS - 1).trimEnd() + "…";
}

/**
 * Body for the existing POST /v1/memories/raw endpoint (RawMemoryRequest):
 * a pre-categorized `task_context` fact tagged `compact_snapshot`, scoped
 * to the project when one resolves, with a finite expiry so snapshots stay
 * a rolling window instead of accumulating forever.
 */
export function buildCompactSnapshotPayload(
  content: string,
  userId: string,
  projectId: string | undefined,
  when: Date = new Date(),
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    content,
    user_id: userId,
    category: "task_context",
    scope: projectId ? "project" : "global",
    tags: [COMPACT_SNAPSHOT_TAG],
    source_type: "tool_extraction",
    expires_at: new Date(when.getTime() + SNAPSHOT_TTL_MS).toISOString(),
  };
  if (projectId) payload.project_id = projectId;
  return payload;
}
