/**
 * SessionEnd hook — structured Stop summary (roadmap D2).
 *
 * Fires ONCE when the session actually ends (clear/logout/exit) — unlike
 * Stop, which fires after every assistant turn. Builds a
 * `{request, investigated, learned, completed, next_steps}` note from the
 * full transcript + this session's observation buffer and stores it via
 * ONE `POST /v1/checkpoint` call (session_note only, zero memories — the
 * buffer's memories still belong to the compile-observations flow).
 *
 * The server renders it as a single `task_context` memory tagged
 * `session_note`; the next SessionStart parses it back into the
 * "Previously…" block.
 *
 * Failure taxonomy (D4): transport failure → notice on stderr, exit 0
 * (never block session teardown); malformed stdin → exit 2 (client bug —
 * SessionEnd exit 2 only surfaces stderr, it cannot block anything).
 */

import { readFile } from "node:fs/promises";

import { extractAllTurnPairs } from "../adapters/claude-code.js";
import {
  type ObservationRow,
  buildCheckpointPayload,
  buildSessionNote,
} from "../core/session-note.js";
import {
  MalformedHookInputError,
  getBufferPath,
  getProjectId,
  getUserId,
  hasUserId,
  logError,
  neuralscapePost,
  outputContinue,
  parseStdinStrict,
} from "../utils.js";

/** Parse the session's observation buffer (already-truncated → []). */
export async function readObservationRows(sessionId: string): Promise<ObservationRow[]> {
  try {
    const content = await readFile(getBufferPath(sessionId), "utf-8");
    return content
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => {
        try {
          return JSON.parse(l) as ObservationRow;
        } catch {
          return null;
        }
      })
      .filter((r): r is ObservationRow => r !== null);
  } catch {
    return [];
  }
}

/* v8 ignore start — main() is the integration entry; pure logic lives in core/session-note.ts */
async function main(): Promise<void> {
  outputContinue();

  let raw: Record<string, unknown>;
  try {
    raw = (await parseStdinStrict()) as Record<string, unknown>;
  } catch (error) {
    if (error instanceof MalformedHookInputError) {
      logError(error.message);
      process.exit(2); // client bug — fail loud (SessionEnd cannot block)
    }
    logError("session-summary stdin read failed", error);
    return;
  }

  try {
    if (!hasUserId()) return;

    const sessionId = (raw.session_id as string | undefined) || "unknown";
    const turns = await extractAllTurnPairs(raw);
    const observations = await readObservationRows(sessionId);

    const note = buildSessionNote(turns, observations);
    if (!note) return; // trivial session — nothing worth carrying forward

    const userId = getUserId();
    const projectId = getProjectId(raw.cwd as string | undefined);

    try {
      await neuralscapePost(
        "/v1/checkpoint",
        buildCheckpointPayload(note, userId, projectId),
      );
    } catch (error) {
      // Transport failure → notice + exit 0 (never block session teardown).
      logError("session summary checkpoint failed (service unreachable?)", error);
      return;
    }
  } catch (error) {
    logError("session-summary hook failed", error);
  }
}

// Skip auto-running main() when imported by the test harness.
if (process.env.NEURALSCAPE_TEST !== "1") {
  main();
}
/* v8 ignore stop */
