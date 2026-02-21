/**
 * PostToolUse hook — captures tool observations and fires them to Neuralscape
 * as raw memories. Runs async (fire-and-forget) so it never blocks Claude.
 *
 * Filtering: captures all tool observations except known diagnostic/read-only
 * commands (git status, docker logs, curl localhost, ls, grep, etc.).
 */

import {
  type HookInput,
  getUserId,
  getProjectId,
  logError,
  neuralscapePost,
  outputContinue,
  parseStdin,
} from "./utils.js";

// Tool → category mapping for auto-captured observations
const TOOL_CATEGORY_MAP: Record<string, string> = {
  Write: "task_context",
  Edit: "task_context",
  Bash: "task_context",
  WebFetch: "task_context",
  WebSearch: "task_context",
  Task: "task_context",
  NotebookEdit: "task_context",
};

// Tools that produce too much noise — already filtered by matcher in hooks.json,
// but double-check here
const SKIP_TOOLS = new Set([
  "Glob",
  "Grep",
  "Read",
  "AskUserQuestion",
  "ListMcpResourcesTool",
  "ReadMcpResourceTool",
  "TaskList",
  "TaskGet",
  "TaskCreate",
  "TaskUpdate",
]);

// ── Bash command filtering ──────────────────────────────────────────
// Only capture commands that change state. Skip read-only/diagnostic commands.

// Commands to skip (diagnostic/read-only). Everything else is captured.
const BASH_SKIP_PATTERNS = [
  /^(cat|head|tail|less|more)\s+/,
  /^(ls|ll|find|tree|wc|du|df)\s+/,
  /^(grep|rg|ag|ack)\s+/,
  /^(echo|printf)\s+/,
  /^(git\s+(status|log|diff|show|branch|remote|stash\s+list))/,
  /^(docker\s+(logs|ps|exec|inspect|images))/,
  /^(docker\s+compose\s+(ps|logs|config))/,
  /^(redis-cli|mongo|psql|mysql)\s+/,
  /^curl\s+.*(-s|--silent).*localhost/,  // Diagnostic curls to local services
  /^(which|where|type|command)\s+/,
  /^(env|printenv|set)\s*$/,
  /^(node|python3?)\s+(-c|--check|-e)\s+/,  // One-liner scripts
  /^source\s+/,
  /^sleep\s+/,
];

function shouldCaptureBashCommand(command: string): boolean {
  const trimmed = command.trim();

  // Skip empty or very short commands
  if (trimmed.length < 3) return false;

  // Check skip patterns (deny list) — everything else is captured
  for (const pattern of BASH_SKIP_PATTERNS) {
    if (pattern.test(trimmed)) return false;
  }

  return true;
}

// ── File path filtering ─────────────────────────────────────────────
// Skip writes/edits to files that aren't interesting

const SKIP_FILE_PATTERNS = [
  /\/node_modules\//,
  /\/__pycache__\//,
  /\.pyc$/,
  /\/\.git\//,
  /package-lock\.json$/,
  /uv\.lock$/,
  /\.log$/,
];

function shouldCaptureFilePath(filePath: string): boolean {
  for (const pattern of SKIP_FILE_PATTERNS) {
    if (pattern.test(filePath)) return false;
  }
  return true;
}

// ── Tool summarization ──────────────────────────────────────────────

function summarizeTool(input: HookInput): string | null {
  const toolName = input.tool_name || "unknown";
  const toolInput = input.tool_input || {};

  if (SKIP_TOOLS.has(toolName)) return null;

  switch (toolName) {
    case "Write": {
      const filePath = toolInput.file_path as string | undefined;
      if (!filePath || !shouldCaptureFilePath(filePath)) return null;
      return `Wrote file ${filePath}`;
    }
    case "Edit": {
      const filePath = toolInput.file_path as string | undefined;
      if (!filePath || !shouldCaptureFilePath(filePath)) return null;
      return `Edited file ${filePath}`;
    }
    case "Bash": {
      const command = toolInput.command as string | undefined;
      if (!command || !shouldCaptureBashCommand(command)) return null;
      const shortCmd = command.length > 120 ? command.slice(0, 120) + "..." : command;
      return `Ran command: ${shortCmd}`;
    }
    case "WebFetch": {
      const url = toolInput.url as string | undefined;
      return url ? `Fetched web content from ${url}` : null;
    }
    case "WebSearch": {
      const query = toolInput.query as string | undefined;
      return query ? `Searched web for: ${query}` : null;
    }
    case "Task": {
      const desc = toolInput.description as string | undefined;
      const agentType = toolInput.subagent_type as string | undefined;
      return desc
        ? `Launched ${agentType || "agent"} task: ${desc}`
        : null;
    }
    case "NotebookEdit": {
      const notebookPath = toolInput.notebook_path as string | undefined;
      return notebookPath ? `Edited notebook ${notebookPath}` : null;
    }
    default:
      return null;
  }
}

function deriveScope(
  category: string,
  projectId: string | undefined
): string {
  const projectCategories = new Set([
    "tech_stack",
    "convention",
    "architecture",
    "dependency",
  ]);
  if (projectCategories.has(category) && projectId) return "project";
  if (projectId) return "project";
  return "global";
}

async function main(): Promise<void> {
  // Always output continue immediately so we never block Claude
  outputContinue();

  try {
    const input = await parseStdin();
    const toolName = input.tool_name;

    if (!toolName || SKIP_TOOLS.has(toolName)) return;

    const summary = summarizeTool(input);
    if (!summary) return;

    const userId = getUserId();
    const projectId = getProjectId(input.cwd);
    const category = TOOL_CATEGORY_MAP[toolName] || "task_context";
    const scope = deriveScope(category, projectId);

    // Fire-and-forget POST to Neuralscape
    const body: Record<string, unknown> = {
      content: summary,
      user_id: userId,
      category,
      scope,
      tags: ["auto_captured", `tool:${toolName}`],
    };

    if (projectId && scope === "project") {
      body.project_id = projectId;
    }

    await neuralscapePost("/v1/memories/raw", body).catch((error) => {
      // Silently fail — don't disrupt Claude's workflow
      logError("Failed to store observation", error);
    });
  } catch (error) {
    logError("post-tool-use hook failed", error);
  }
}

main();
