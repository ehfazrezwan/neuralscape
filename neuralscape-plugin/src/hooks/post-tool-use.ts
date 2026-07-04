/**
 * PostToolUse hook — captures tool observations to a per-session JSONL buffer
 * for later compilation by the `compile-observations` skill.
 *
 * Pure recorder: NO LLM, NO HTTP, fast. Filters read-only tools at the gate
 * and truncates large outputs before persisting. Always exits 0 — must never
 * block tool execution.
 */

import {
  type HookInput,
  MalformedHookInputError,
  appendObservation,
  getObservationDir,
  getProjectId,
  getUserId,
  hasUserId,
  isProjectExcluded,
  logError,
  outputContinue,
  parseStdinStrict,
  pickRelevantInput,
  readConfig,
  redactPrivate,
  truncateOutput,
} from "../utils.js";

// Tools whose outputs are not worth capturing — read-only, already-managed,
// or pure harness plumbing. The plumbing set (Skill, ToolSearch,
// AskUserQuestion, Task*/Workflow/Monitor, WaitForMcpServers) was chosen by
// inspecting real observation buffers: those rows are UI/orchestration
// noise that compile-observations always skips anyway (roadmap D4).
export const DENY_TOOLS = new Set<string>([
  "Read",
  "Glob",
  "Grep",
  "NotebookRead",
  "TodoWrite", // managed by Claude Code's own todo system
  "ListMcpResourcesTool",
  "ReadMcpResourceTool",
  "ExitPlanMode",
  "EnterPlanMode",
  "EnterWorktree",
  "ExitWorktree",
  "Skill", // the tools the skill runs get captured; the invocation is noise
  "ToolSearch",
  "AskUserQuestion",
  "TaskUpdate",
  "TaskStop",
  "TaskList",
  "Workflow",
  "Monitor",
  "WaitForMcpServers",
]);

/** DENY_TOOLS plus any user-configured additions (SKIP_TOOLS, comma-separated). */
export function getSkipTools(): Set<string> {
  const extra = readConfig("SKIP_TOOLS", "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  if (extra.length === 0) return DENY_TOOLS;
  return new Set([...DENY_TOOLS, ...extra]);
}

// Bash commands that are routine inspection/navigation — skip.
export const BORING_BASH_RE =
  /^\s*(ls|pwd|cd|cat|head|tail|less|more|stat|file|which|where|tree|echo|date|whoami|uname|hostname|env|set|unset|history)\b/;

// Tool names whose use is internal Neuralscape plumbing — capturing them
// would create feedback loops (e.g. compile-observations calling `remember`
// would generate observations about its own writes).
export const NEURALSCAPE_TOOL_PREFIX = "mcp__plugin_neuralscape_";

export function asString(v: unknown): string {
  if (typeof v === "string") return v;
  if (v == null) return "";
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

/**
 * Normalize a path for prefix comparison: forward slashes, lower-cased on
 * Windows, trimmed. We need this because the observation_dir resolved
 * locally may differ in slash direction from what the tool input contains.
 */
function normalizePath(p: string): string {
  if (!p) return "";
  let out = p.replace(/\\/g, "/").trim();
  // Strip trailing slash for cleaner prefix checks
  if (out.endsWith("/")) out = out.slice(0, -1);
  if (process.platform === "win32") out = out.toLowerCase();
  return out;
}

/** Pure decision: should this PostToolUse event be captured? */
export function shouldCapture(input: HookInput): boolean {
  const toolName = input.tool_name;
  if (!toolName) return false;
  if (getSkipTools().has(toolName)) return false;
  // Excluded projects (D4): never capture anything from a project matching
  // an EXCLUDED_PROJECTS glob.
  if (isProjectExcluded(getProjectId(input.cwd))) return false;
  // Skip Neuralscape's own MCP tools to avoid recording our own writes.
  if (toolName.startsWith(NEURALSCAPE_TOOL_PREFIX)) return false;
  if (toolName === "Bash") {
    const cmd = asString((input.tool_input as Record<string, unknown> | undefined)?.command);
    if (BORING_BASH_RE.test(cmd)) return false;
  }
  // For file-writing tools, skip when the target is inside our observation
  // buffer directory — otherwise the truncation that compile-observations
  // performs becomes a new observation, creating an infinite loop.
  if (toolName === "Write" || toolName === "Edit" || toolName === "NotebookEdit") {
    const filePath = asString(
      (input.tool_input as Record<string, unknown> | undefined)?.file_path
        ?? (input.tool_input as Record<string, unknown> | undefined)?.notebook_path,
    );
    if (filePath) {
      const obsDir = normalizePath(getObservationDir());
      const target = normalizePath(filePath);
      if (target.startsWith(obsDir + "/") || target === obsDir) return false;
    }
  }
  return true;
}

/** Build the observation row that gets appended to the per-session buffer.
 *  `<private>…</private>` spans are redacted BEFORE anything hits disk (D4). */
export function buildObservationRow(input: HookInput): Record<string, unknown> {
  const picked = pickRelevantInput(input.tool_name!, input.tool_input);
  for (const [k, v] of Object.entries(picked)) {
    if (typeof v === "string") picked[k] = redactPrivate(v);
  }
  return {
    ts: new Date().toISOString(),
    session_id: input.session_id || "unknown",
    cwd: input.cwd,
    project_id: getProjectId(input.cwd),
    user_id: getUserId(),
    tool: input.tool_name,
    input: picked,
    output: redactPrivate(truncateOutput(asString(input.tool_output))),
  };
}

/* v8 ignore start — main() is the integration entry; pure logic above is unit-tested */
async function main(): Promise<void> {
  // Always emit continue first so even a thrown error below cannot block.
  outputContinue();

  let input: HookInput;
  try {
    input = await parseStdinStrict();
  } catch (error) {
    if (error instanceof MalformedHookInputError) {
      logError(error.message);
      process.exit(2); // client bug — fail loud (PostToolUse already ran; cannot block)
    }
    logError("post-tool-use stdin read failed", error);
    return;
  }

  try {
    if (!hasUserId()) {
      // Quietly skip when not configured — no logging noise per tool.
      return;
    }

    if (!shouldCapture(input)) return;

    await appendObservation(buildObservationRow(input));
  } catch (error) {
    // Never let a hook failure surface to the user.
    logError("post-tool-use hook failed", error);
  }
}

// Skip auto-running main() when imported by the test harness.
if (process.env.NEURALSCAPE_TEST !== "1") {
  main();
}
/* v8 ignore stop */
