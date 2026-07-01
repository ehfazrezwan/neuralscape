/**
 * SessionStart hook — fetches stored context from Neuralscape and injects it
 * into the Claude Code session as additionalContext.
 */

import {
  type ContextResponse,
  type NeuralscapeMemory,
  getUserId,
  getProjectId,
  hasUserId,
  listPendingBuffers,
  logError,
  neuralscapeGet,
  outputContinue,
  outputWithContext,
  parseStdin,
} from "../utils.js";

import { CATEGORY_LABELS, CATEGORY_ORDER } from "../types.js";

// Target max tokens for injection (~4 chars per token)
const MAX_CHARS = 8000;
// Reserved budget for authoritative standards, on top of MAX_CHARS. Standards
// are binding and must never be truncated away by a large recalled context.
const STANDARDS_MAX_CHARS = 3000;

function formatStandards(standards: NeuralscapeMemory[] | undefined): string {
  if (!standards || standards.length === 0) return "";
  const lines: string[] = [
    "# ⚖️ Neuralscape AUTHORITATIVE Standards (binding)",
    "",
    "These are organization standards set by a Neuralscape dictator. They are " +
      "BINDING directives, not preferences. On any conflict they OVERRIDE " +
      "personal preferences and project conventions. Follow them unless the " +
      "user explicitly overrides them in this session.",
    "",
  ];
  let total = 0;
  for (const mem of standards) {
    const line = `- ${mem.memory}`;
    if (total + line.length > STANDARDS_MAX_CHARS) break;
    lines.push(line);
    total += line.length;
  }
  return lines.join("\n");
}

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
    if (!hasUserId()) {
      logError(
        "missing user_id — run `/plugin config neuralscape@neuralscape-plugins` to set USER_ID (or set NEURALSCAPE_USER_ID env var as legacy fallback); skipping context injection",
      );
      outputContinue();
      return;
    }

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

    // Detect any unprocessed observation buffers left behind by prior sessions
    // and ask Claude to compile them before responding to the user.
    let pendingNote = "";
    try {
      const pending = await listPendingBuffers(input.session_id);
      if (pending.length > 0) {
        const totalRows = pending.reduce((s, b) => s + b.lineCount, 0);
        const paths = pending.map((b) => `\`${b.path}\``).join(", ");
        pendingNote =
          `\n\n# Neuralscape — Pending Observations\n\n` +
          `${pending.length} buffer file(s) from prior sessions hold ` +
          `${totalRows} unprocessed tool observations: ${paths}.\n\n` +
          `**Before responding to the user's first prompt**, run the ` +
          `\`compile-observations\` skill on each of these paths to extract ` +
          `significant memories using the v2 memory model. Skip noise ` +
          `(routine reads/edits/searches). After memories submit successfully, ` +
          `truncate each buffer file. Then continue with the user's request.`;
      }
    } catch (error) {
      logError("listPendingBuffers failed (non-critical)", error);
    }

    // Format and inject. Standards are prepended (outside the MAX_CHARS budget)
    // as a binding block so they are never truncated by recalled context.
    const standardsBlock = formatStandards(context.standards);
    const formatted = formatMemories(context.categories || {});
    const contextBody = standardsBlock
      ? formatted
        ? `${standardsBlock}\n\n---\n\n${formatted}`
        : standardsBlock
      : formatted;
    const combined = (contextBody || "") + pendingNote;
    if (combined.trim()) {
      outputWithContext(combined);
    } else {
      outputContinue();
    }
  } catch (error) {
    logError("session-start hook failed", error);
    outputContinue();
  }
}

main();
