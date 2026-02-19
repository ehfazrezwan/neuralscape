/**
 * PostToolUse hook — captures tool observations and fires them to Neuralscape
 * as raw memories. Runs async (fire-and-forget) so it never blocks Claude.
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

function summarizeTool(input: HookInput): string | null {
  const toolName = input.tool_name || "unknown";
  const toolInput = input.tool_input || {};

  if (SKIP_TOOLS.has(toolName)) return null;

  switch (toolName) {
    case "Write": {
      const filePath = toolInput.file_path as string | undefined;
      return filePath ? `Wrote file ${filePath}` : null;
    }
    case "Edit": {
      const filePath = toolInput.file_path as string | undefined;
      return filePath ? `Edited file ${filePath}` : null;
    }
    case "Bash": {
      const command = toolInput.command as string | undefined;
      if (!command) return null;
      // Truncate long commands
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
      return `Used tool ${toolName}`;
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
