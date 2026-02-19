/**
 * Stop hook — fires when Claude finishes responding. Stores a session
 * completion marker so Neuralscape can track session boundaries.
 *
 * Runs async so it never blocks the session ending.
 */

import {
  getUserId,
  getProjectId,
  logError,
  neuralscapePost,
  outputContinue,
  parseStdin,
} from "./utils.js";

async function main(): Promise<void> {
  // Always output continue immediately
  outputContinue();

  try {
    const input = await parseStdin();
    const userId = getUserId();
    const projectId = getProjectId(input.cwd);
    const sessionId = input.session_id;

    // Store a session-end marker
    const body: Record<string, unknown> = {
      content: `Session ${sessionId || "unknown"} completed in project ${projectId || "unknown"}`,
      user_id: userId,
      category: "interaction",
      scope: projectId ? "project" : "global",
      tags: ["auto_captured", "session_end"],
    };

    if (projectId) {
      body.project_id = projectId;
    }

    await neuralscapePost("/v1/memories/raw", body).catch((error) => {
      logError("Failed to store session marker", error);
    });
  } catch (error) {
    logError("stop hook failed", error);
  }
}

main();
