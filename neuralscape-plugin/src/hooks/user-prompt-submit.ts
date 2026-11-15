/**
 * UserPromptSubmit hook — primary trigger for the compile-observations skill.
 *
 * On every user message, check the per-session observation buffer. If it has
 * accumulated beyond the threshold (or its oldest entry is older than the
 * configured age), inject `additionalContext` instructing Claude to compile
 * the buffer using the skill before responding to the user's prompt. The
 * instruction is injected once per threshold crossing (see shouldInjectNow),
 * not on every prompt.
 *
 * The cost lands on the same Claude turn that was about to handle the prompt
 * anyway — no extra API round-trip, fully covered by the user's subscription.
 */

import { readFile, unlink, writeFile } from "node:fs/promises";

import {
  type BufferStats,
  getBufferPath,
  getBufferStats,
  hasUserId,
  logError,
  outputContinue,
  outputHookResult,
  parseStdin,
  readConfig,
} from "../utils.js";

// Defaults tuned for ~50-100 tool sessions; configurable via plugin userConfig.
export const DEFAULT_COMPILE_THRESHOLD = 25;
export const DEFAULT_COMPILE_AGE_MIN = 30; // minutes
export const HARD_CAP = 500; // forced compile or warning above this

export function shouldCompile(stats: BufferStats, threshold: number, ageMinutes: number): { compile: boolean; reason: string } {
  if (stats.lineCount === 0) return { compile: false, reason: "empty" };
  if (stats.lineCount >= HARD_CAP) {
    return { compile: true, reason: `hard-cap reached (${stats.lineCount} ≥ ${HARD_CAP})` };
  }
  if (stats.lineCount >= threshold) {
    return { compile: true, reason: `threshold reached (${stats.lineCount} ≥ ${threshold})` };
  }
  if (stats.oldestTs) {
    const ageMs = Date.now() - new Date(stats.oldestTs).getTime();
    const ageMin = ageMs / 60_000;
    if (ageMin >= ageMinutes) {
      return { compile: true, reason: `buffer aged (${ageMin.toFixed(0)}min ≥ ${ageMinutes}min)` };
    }
  }
  return { compile: false, reason: "below threshold and not aged" };
}

export function buildCompileInstruction(stats: BufferStats, reason: string): string {
  return `# Neuralscape — Compile Observations

The PostToolUse hook has captured **${stats.lineCount} tool observations** in this session's buffer (${reason}). Buffer path:

\`${stats.path}\`

**Before responding to the user's prompt above**, run the \`compile-observations\` skill on this buffer. The skill holds the quality rubric and the field list; follow it. In short:

1. Read the JSONL rows, group them into work units, and keep only the significant ones (decision, discovery, gotcha, pattern, bugfix, convention, architecture, outcome).
2. Submit the memories through the MCP tools, never curl: ONE \`mcp__plugin_neuralscape_neuralscape__checkpoint\` call carrying all of them when there are 3 or more, individual \`mcp__plugin_neuralscape_neuralscape__remember\` calls when there are 1-2.
3. After the calls succeed, truncate the buffer by writing an empty string to the same path with the \`Write\` tool.
4. Then continue with the user's request. This note is shown once per threshold crossing, not on every prompt.

Throughput target: ${stats.lineCount} captured rows should yield ~3-10 dense memories, not ${stats.lineCount}. Skip routine file edits, reads, searches. Never store API keys, tokens, or env vars.`;
}

// ── Once-per-crossing cool-down ──────────────────────────────────

/** What the hook remembers about the last time it injected the instruction for a buffer. */
export interface CompileMarker {
  /** Buffer line count when the instruction was injected. */
  lineCount: number;
  /** Injection time (ms since epoch). */
  injectedAt: number;
}

/** Path of the per-buffer cool-down marker (next to the buffer; the buffer's own name keeps it per-session). */
export function getCompileMarkerPath(bufferPath: string): string {
  return bufferPath + ".compile-armed";
}

/**
 * Should the instruction be injected NOW, given that the buffer qualifies for
 * compilation? The instruction used to ride along on every prompt until the
 * buffer was truncated, so an ignored (or still running) compile re-prompted
 * on every turn. Now it is injected once per crossing and re-armed when
 *   - there is no marker (first crossing),
 *   - the buffer is smaller than at the last injection (it was truncated and refilled),
 *   - the buffer grew by another `threshold` rows since the last injection, or
 *   - the last injection is itself `ageMinutes` old.
 */
export function shouldInjectNow(
  stats: BufferStats,
  marker: CompileMarker | null,
  threshold: number,
  ageMinutes: number,
  now: number = Date.now(),
): boolean {
  if (!marker) return true;
  if (stats.lineCount < marker.lineCount) return true;
  if (stats.lineCount >= marker.lineCount + threshold) return true;
  if (now - marker.injectedAt >= ageMinutes * 60_000) return true;
  return false;
}

export async function readCompileMarker(bufferPath: string): Promise<CompileMarker | null> {
  try {
    const parsed = JSON.parse(await readFile(getCompileMarkerPath(bufferPath), "utf-8"));
    if (Number.isFinite(parsed?.lineCount) && Number.isFinite(parsed?.injectedAt)) {
      return { lineCount: parsed.lineCount, injectedAt: parsed.injectedAt };
    }
  } catch {
    // missing or unreadable: behave as if never injected
  }
  return null;
}

export async function writeCompileMarker(bufferPath: string, marker: CompileMarker): Promise<void> {
  try {
    await writeFile(getCompileMarkerPath(bufferPath), JSON.stringify(marker), { encoding: "utf-8", mode: 0o600 });
  } catch (error) {
    logError("writing compile marker failed (non-critical)", error);
  }
}

export async function clearCompileMarker(bufferPath: string): Promise<void> {
  try {
    await unlink(getCompileMarkerPath(bufferPath));
  } catch {
    // none — fine
  }
}

export function getThreshold(): number {
  const raw = readConfig("COMPILE_THRESHOLD", "");
  const parsed = parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_COMPILE_THRESHOLD;
}

export function getAgeMinutes(): number {
  const raw = readConfig("COMPILE_AGE_MIN", "");
  const parsed = parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_COMPILE_AGE_MIN;
}

/* v8 ignore start — main() is the integration entry; pure logic above is unit-tested */
async function main(): Promise<void> {
  try {
    if (!hasUserId()) {
      outputContinue();
      return;
    }
    const input = await parseStdin();
    const sessionId = input.session_id || "unknown";
    const stats = await getBufferStats(getBufferPath(sessionId));
    const threshold = getThreshold();
    const ageMin = getAgeMinutes();

    const decision = shouldCompile(stats, threshold, ageMin);
    if (!decision.compile) {
      // Below the threshold again (the buffer was truncated): re-arm.
      await clearCompileMarker(stats.path);
      outputContinue();
      return;
    }

    const marker = await readCompileMarker(stats.path);
    if (!shouldInjectNow(stats, marker, threshold, ageMin)) {
      outputContinue();
      return;
    }
    await writeCompileMarker(stats.path, { lineCount: stats.lineCount, injectedAt: Date.now() });

    outputHookResult({
      continue: true,
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext: buildCompileInstruction(stats, decision.reason),
      },
    });
  } catch (error) {
    logError("user-prompt-submit hook failed", error);
    outputContinue();
  }
}

// Skip auto-running main() when imported by the test harness.
if (process.env.NEURALSCAPE_TEST !== "1") {
  main();
}
/* v8 ignore stop */
