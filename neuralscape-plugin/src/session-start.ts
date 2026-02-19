/**
 * SessionStart hook — fetches stored context from Neuralscape and injects it
 * into the Claude Code session as additionalContext.
 */

import {
  type ContextResponse,
  type NeuralscapeMemory,
  getUserId,
  getProjectId,
  logError,
  neuralscapeGet,
  outputContinue,
  outputWithContext,
  parseStdin,
} from "./utils.js";

// Category display order (most important first)
const CATEGORY_ORDER = [
  "preference",
  "convention",
  "architecture",
  "tech_stack",
  "dependency",
  "workflow",
  "procedure",
  "decision",
  "personal_fact",
  "technical_skill",
  "domain_knowledge",
  "interaction",
  "task_context",
];

const CATEGORY_LABELS: Record<string, string> = {
  preference: "Preferences",
  personal_fact: "Personal",
  technical_skill: "Skills",
  domain_knowledge: "Domain Knowledge",
  tech_stack: "Tech Stack",
  convention: "Conventions",
  architecture: "Architecture",
  dependency: "Dependencies",
  decision: "Decisions",
  interaction: "Interactions",
  workflow: "Workflows",
  procedure: "Procedures",
  task_context: "Recent Context",
};

// Target max tokens for injection (~4 chars per token)
const MAX_CHARS = 8000;

function formatMemories(categories: Record<string, NeuralscapeMemory[]>): string {
  const sections: string[] = [];
  let totalChars = 0;

  for (const cat of CATEGORY_ORDER) {
    const memories = categories[cat];
    if (!memories || memories.length === 0) continue;

    const label = CATEGORY_LABELS[cat] || cat;
    const lines: string[] = [`## ${label}`];

    for (const mem of memories) {
      const line = `- ${mem.memory}`;
      if (totalChars + line.length > MAX_CHARS) break;
      lines.push(line);
      totalChars += line.length;
    }

    if (lines.length > 1) {
      sections.push(lines.join("\n"));
    }

    if (totalChars >= MAX_CHARS) break;
  }

  if (sections.length === 0) return "";

  return `# Neuralscape Memory Context\n\n${sections.join("\n\n")}`;
}

async function main(): Promise<void> {
  try {
    const input = await parseStdin();
    const userId = getUserId();
    const projectId = getProjectId(input.cwd);

    // Fetch context from Neuralscape
    let context: ContextResponse;
    try {
      if (projectId) {
        context = (await neuralscapeGet(`/v1/context/${projectId}`, {
          user_id: userId,
        })) as ContextResponse;
      } else {
        context = (await neuralscapeGet("/v1/context/global", {
          user_id: userId,
        })) as ContextResponse;
      }
    } catch {
      // Neuralscape unavailable — continue without context
      outputContinue();
      return;
    }

    // Format and inject
    const formatted = formatMemories(context.categories || {});
    if (formatted) {
      outputWithContext(formatted);
    } else {
      outputContinue();
    }
  } catch (error) {
    logError("session-start hook failed", error);
    outputContinue();
  }
}

main();
