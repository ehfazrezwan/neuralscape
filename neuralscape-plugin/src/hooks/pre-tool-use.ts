/**
 * PreToolUse hook on the `Read` tool — the File Read Gate (roadmap D3).
 *
 * When Claude is about to Read a file that (a) is larger than
 * READ_GATE_MIN_BYTES and (b) Neuralscape already holds memories about,
 * the gate DENIES the read and substitutes a ranked per-file memory
 * timeline (`#id | when | title | ~tokens` rows) plus an escalation menu —
 * the memories usually answer the question for a fraction of the tokens.
 *
 * Hard safety rules (never get in the user's way):
 *   - fires at most ONCE per file per session (state file dedup) — the
 *     second Read of the same path is ALWAYS allowed (that IS the override);
 *   - bypasses small files, binary/media extensions, excluded projects;
 *   - NS unreachable → allow + exit 0 (never-block taxonomy), counting the
 *     failure toward the fail-loud threshold;
 *   - malformed stdin → allow + exit 0 (exit 2 on PreToolUse BLOCKS the
 *     tool call, so the never-block principle wins — same as Stop/UPS);
 *   - READ_GATE_ENABLED=false turns the whole thing off.
 */

import { stat } from "node:fs/promises";

import {
  type HookInput,
  type NeuralscapeMemory,
  getProjectId,
  getUserId,
  hasUserId,
  isProjectExcluded,
  loadGatedFiles,
  logError,
  neuralscapeGet,
  outputContinue,
  outputHookResult,
  parseStdin,
  readConfig,
  recordGatedFile,
  recordTransportFailure,
  resetTransportFailures,
} from "../utils.js";

import {
  DEFAULT_READ_GATE_MIN_BYTES,
  READ_GATE_FETCH_LIMIT,
  buildDenyOutput,
  isBypassedExtension,
  rankFileMemories,
  renderReadGateReason,
} from "../core/read-gate.js";

// ── Config knobs ─────────────────────────────────────────────────

export function getReadGateEnabled(): boolean {
  const raw = readConfig("READ_GATE_ENABLED", "true").toLowerCase();
  return raw !== "false" && raw !== "0" && raw !== "off" && raw !== "no";
}

export function getReadGateMinBytes(): number {
  const parsed = parseInt(readConfig("READ_GATE_MIN_BYTES", ""), 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : DEFAULT_READ_GATE_MIN_BYTES;
}

/** The Read target, if this event is a gateable Read invocation. */
export function targetFilePath(input: HookInput): string | null {
  if (input.tool_name !== "Read") return null;
  const raw = (input.tool_input as Record<string, unknown> | undefined)?.file_path;
  if (typeof raw !== "string" || !raw.trim()) return null;
  return raw.trim();
}

/* v8 ignore start — main() is the integration entry, exercised by the
   subprocess suite (tests/read-gate-subprocess.test.ts) against the built
   bundle with a local fixture server; pure logic lives in core/read-gate.ts */

/**
 * Fetch the memories to match against this file: the newest
 * READ_GATE_FETCH_LIMIT memories from `GET /v1/memories` (newest-first,
 * project-scoped when a project resolves, ~100ms) rather than
 * `POST /v1/search` — the hybrid search also runs a Graphiti pass whose
 * latency routinely exceeds the PreToolUse hook budget, and the
 * verification filter needs the literal path mention anyway; recall's
 * semantic ranking adds nothing here. The project scope also keeps bulk
 * global ingest (e.g. book passages) from flooding the recency window.
 */
async function fetchCandidateMemories(
  userId: string,
  projectId: string | undefined,
): Promise<NeuralscapeMemory[]> {
  const params: Record<string, string> = {
    user_id: userId,
    limit: String(READ_GATE_FETCH_LIMIT),
  };
  if (projectId) params.project_id = projectId;
  const response = await neuralscapeGet("/v1/memories", params);
  return Array.isArray(response) ? (response as NeuralscapeMemory[]) : [];
}

async function main(): Promise<void> {
  let input: HookInput;
  try {
    // Lenient parse on purpose: exit 2 on PreToolUse has BLOCKING semantics
    // (it would deny the Read), so malformed stdin logs + allows instead.
    input = await parseStdin();
  } catch (error) {
    logError("pre-tool-use stdin read failed", error);
    outputContinue();
    return;
  }

  try {
    if (!getReadGateEnabled() || !hasUserId()) {
      outputContinue();
      return;
    }

    const filePath = targetFilePath(input);
    if (!filePath || isBypassedExtension(filePath)) {
      outputContinue();
      return;
    }

    // Small files are cheaper to just read than to gate.
    let sizeBytes: number;
    try {
      const info = await stat(filePath);
      if (!info.isFile()) {
        outputContinue();
        return;
      }
      sizeBytes = info.size;
    } catch {
      // Missing/unreadable file — let Read produce its own error.
      outputContinue();
      return;
    }
    if (sizeBytes <= getReadGateMinBytes()) {
      outputContinue();
      return;
    }

    const projectId = getProjectId(input.cwd);
    if (isProjectExcluded(projectId)) {
      outputContinue();
      return;
    }

    // Once per file per session — a repeat Read is the documented override,
    // and repeated large-file reads shouldn't re-pay the search round-trip.
    const sessionId = input.session_id || "unknown";
    const gated = await loadGatedFiles(sessionId);
    if (gated.has(filePath)) {
      outputContinue();
      return;
    }

    let results: NeuralscapeMemory[];
    try {
      results = await fetchCandidateMemories(getUserId(), projectId);
    } catch (error) {
      // Transport failure → allow, exit 0 (never-block) + count toward the
      // fail-loud threshold surfaced at next SessionStart.
      await recordTransportFailure();
      logError("read-gate memory fetch failed (service unreachable?) — allowing read", error);
      outputContinue();
      return;
    }
    await resetTransportFailures();

    const ranked = rankFileMemories(results, filePath);
    // Record the check either way: deny → the retry passes; no hits → the
    // next Read of this file skips the fetch entirely.
    await recordGatedFile(sessionId, filePath);

    if (ranked.length === 0) {
      outputContinue();
      return;
    }

    outputHookResult(buildDenyOutput(renderReadGateReason(filePath, sizeBytes, ranked)));
  } catch (error) {
    logError("pre-tool-use hook failed", error);
    outputContinue();
  }
}

// Skip auto-running main() when imported by the test harness.
if (process.env.NEURALSCAPE_TEST !== "1") {
  main();
}
/* v8 ignore stop */
