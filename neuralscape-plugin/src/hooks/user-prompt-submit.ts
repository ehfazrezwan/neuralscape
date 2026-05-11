/**
 * UserPromptSubmit hook — primary trigger for the compile-observations skill.
 *
 * On every user message, check the per-session observation buffer. If it has
 * accumulated beyond the threshold (or its oldest entry is older than the
 * configured age), inject `additionalContext` instructing Claude to compile
 * the buffer using the skill before responding to the user's prompt.
 *
 * The cost lands on the same Claude turn that was about to handle the prompt
 * anyway — no extra API round-trip, fully covered by the user's subscription.
 */

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

**Before responding to the user's prompt above**, please run the \`compile-observations\` skill on this buffer:

1. Read each JSONL row and group consecutive ops on the same target into work units.
2. For each *significant* work unit (decision / discovery / gotcha / pattern / bugfix / convention / architecture / outcome — see the skill's quality rubric), submit one wiki-quality memory via the \`mcp__plugin_neuralscape_neuralscape__remember\` MCP tool with:
   - \`category\`: one of the 13 NS categories
   - \`domain\`: coding | research | meeting | writing | ops | personal | general
   - \`observation_type\`: bugfix | feature | refactor | decision | discovery | gotcha | pattern | trade_off | research_note | meeting_outcome | task_plan | fact
   - \`concepts\`: 1-3 from how-it-works | why-it-exists | what-changed | problem-solution | gotcha | pattern | trade-off | open-question | next-step | blocker
   - \`source_type\`: "tool_extraction"
   - \`confidence\`: your 0.0-1.0 self-assessment
3. After the calls succeed, truncate the buffer (write empty string with \`flag: "w"\`).
4. Then continue with the user's request.

Throughput target: ${stats.lineCount} captured rows should yield ~3-10 dense memories, not ${stats.lineCount}. Skip routine file edits, reads, searches. Skip anything tied to *this* session that won't matter in 30 days. Never store API keys, tokens, or env vars.`;
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
      outputContinue();
      return;
    }

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
