/**
 * PreToolUse hook on the `Read` tool — the File Read Gate (roadmap D3,
 * reworked per audit 27 #31/#32).
 *
 * When Claude is about to Read a file that (a) is larger than
 * READ_GATE_MIN_BYTES and (b) Neuralscape already holds memories about,
 * the gate STEERS: it injects `additionalContext` with a ranked per-file
 * memory index (`#id | when | title | ~tokens` rows) plus an escalation
 * menu — while the Read ALWAYS proceeds. It never denies and never
 * substitutes memory titles for real file contents.
 *
 * Cost control (audit 27 #31 — this spawns synchronously on every Read):
 *   - the NS fetch happens AT MOST ONCE per session: index-level rows
 *     (`GET /v1/memories?fields=index`, capped at READ_GATE_FETCH_LIMIT)
 *     are cached in a session-scoped file; later Reads match in-process;
 *   - a hard time budget (READ_GATE_TIME_BUDGET_MS, default 2s) bounds the
 *     one fetch — on timeout/error the hook exits 0 with a plain allow and
 *     caches the failure so the gate stays quiet for the session;
 *   - context is injected at most ONCE per file per session (state file).
 *
 * Hard safety rules (never get in the user's way):
 *   - bypasses small files, binary/media extensions, excluded projects;
 *   - NS unreachable/slow → allow + exit 0 (never-block taxonomy), counting
 *     the failure toward the fail-loud threshold;
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
  loadReadGateIndexCache,
  logError,
  neuralscapeGet,
  outputContinue,
  outputHookResult,
  parseStdin,
  readConfig,
  recordGatedFile,
  recordTransportFailure,
  resetTransportFailures,
  saveReadGateIndexCache,
} from "../utils.js";

import {
  DEFAULT_READ_GATE_MIN_BYTES,
  DEFAULT_READ_GATE_TIME_BUDGET_MS,
  READ_GATE_FETCH_LIMIT,
  buildSteerOutput,
  isBypassedExtension,
  rankFileMemories,
  renderReadGateContext,
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

export function getReadGateTimeBudgetMs(): number {
  const parsed = parseInt(readConfig("READ_GATE_TIME_BUDGET_MS", ""), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_READ_GATE_TIME_BUDGET_MS;
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
 * Fetch the candidate index rows: the newest READ_GATE_FETCH_LIMIT
 * index-level rows from `GET /v1/memories?fields=index` (newest-first,
 * project-scoped when a project resolves) rather than `POST /v1/search` —
 * the hybrid search runs a Graphiti pass whose latency routinely exceeds
 * the PreToolUse hook budget, and the verification filter needs literal
 * path mentions anyway. `fields=index` drops content payloads server-side
 * (audit 27 #31); older servers ignore the param and return full rows,
 * which match even better — both shapes work.
 */
async function fetchCandidateMemories(
  userId: string,
  projectId: string | undefined,
): Promise<NeuralscapeMemory[]> {
  const params: Record<string, string> = {
    user_id: userId,
    limit: String(READ_GATE_FETCH_LIMIT),
    fields: "index",
  };
  if (projectId) params.project_id = projectId;
  const response = await neuralscapeGet("/v1/memories", params);
  return Array.isArray(response) ? (response as NeuralscapeMemory[]) : [];
}

const FETCH_TIMEOUT: unique symbol = Symbol("read-gate-fetch-timeout");

/** The one NS fetch under the hard time budget. */
async function fetchWithBudget(
  userId: string,
  projectId: string | undefined,
): Promise<NeuralscapeMemory[] | typeof FETCH_TIMEOUT> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      fetchCandidateMemories(userId, projectId),
      new Promise<typeof FETCH_TIMEOUT>((resolve) => {
        timer = setTimeout(() => resolve(FETCH_TIMEOUT), getReadGateTimeBudgetMs());
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
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
    try {
      const info = await stat(filePath);
      if (!info.isFile() || info.size <= getReadGateMinBytes()) {
        outputContinue();
        return;
      }
    } catch {
      // Missing/unreadable file — let Read produce its own error.
      outputContinue();
      return;
    }

    const projectId = getProjectId(input.cwd);
    if (isProjectExcluded(projectId)) {
      outputContinue();
      return;
    }

    // Steer at most once per file per session — repeated Reads of the same
    // path shouldn't re-inject the same context.
    const sessionId = input.session_id || "unknown";
    const gated = await loadGatedFiles(sessionId);
    if (gated.has(filePath)) {
      outputContinue();
      return;
    }

    // Session-scoped index cache (audit 27 #31): the NS fetch happens at
    // most once per session; a cached failure keeps the gate quiet.
    let rows: NeuralscapeMemory[];
    const cached = await loadReadGateIndexCache(sessionId);
    if (cached) {
      if (!cached.ok) {
        outputContinue();
        return;
      }
      rows = cached.rows;
    } else {
      let fetched: NeuralscapeMemory[] | typeof FETCH_TIMEOUT;
      try {
        fetched = await fetchWithBudget(getUserId(), projectId);
      } catch (error) {
        // Transport failure → allow, exit 0 (never-block) + count toward the
        // fail-loud threshold surfaced at next SessionStart.
        await recordTransportFailure();
        await saveReadGateIndexCache(sessionId, false, []);
        logError("read-gate memory fetch failed (service unreachable?) — allowing read", error);
        outputContinue();
        return;
      }
      if (fetched === FETCH_TIMEOUT) {
        await recordTransportFailure();
        await saveReadGateIndexCache(sessionId, false, []);
        logError(
          `read-gate memory fetch exceeded the ${getReadGateTimeBudgetMs()}ms budget — allowing read`,
        );
        outputContinue();
        return;
      }
      await resetTransportFailures();
      rows = fetched;
      await saveReadGateIndexCache(sessionId, true, rows);
    }

    const ranked = rankFileMemories(rows, filePath);
    // Record the check either way so this file never re-triggers this session.
    await recordGatedFile(sessionId, filePath);

    if (ranked.length === 0) {
      outputContinue();
      return;
    }

    // Steer, never block: additionalContext rides alongside the Read.
    outputHookResult(buildSteerOutput(renderReadGateContext(filePath, ranked)));
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
