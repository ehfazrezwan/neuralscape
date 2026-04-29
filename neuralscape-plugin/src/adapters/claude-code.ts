/**
 * Claude Code adapter — extracts conversation turns from the session transcript.
 *
 * Primary: reads the transcript file, parses user/assistant turn pairs,
 *          respects offset to avoid re-flushing.
 * Fallback: if no transcript, uses prompt + last_assistant_message for the last turn.
 */

import { readFile } from "node:fs/promises";
import { getProjectId, getUserId, logError } from "../utils.js";
import type { ConversationTurn, SessionEndInput } from "../core/types.js";

const OFFSET_SUFFIX = ".neuralscape-offset";

interface TranscriptMessage {
  type: string; // "user" | "assistant" | "tool_use" | "tool_result" | ...
  content?: string | Array<{ type: string; text?: string }>;
  role?: string;
}

/**
 * Read the flush offset (last processed message index) for a transcript.
 */
async function readFlushOffset(transcriptPath: string): Promise<number> {
  try {
    const content = await readFile(transcriptPath + OFFSET_SUFFIX, "utf-8");
    const offset = parseInt(content.trim(), 10);
    return Number.isNaN(offset) ? 0 : offset;
  } catch {
    return 0;
  }
}

/**
 * Write the flush offset after successful processing.
 */
async function writeFlushOffset(
  transcriptPath: string,
  offset: number
): Promise<void> {
  const { writeFile } = await import("node:fs/promises");
  await writeFile(transcriptPath + OFFSET_SUFFIX, String(offset), "utf-8");
}

/**
 * Extract text content from a transcript message.
 */
function extractText(msg: TranscriptMessage): string {
  if (typeof msg.content === "string") return msg.content;
  if (Array.isArray(msg.content)) {
    return msg.content
      .filter((block) => block.type === "text" && block.text)
      .map((block) => block.text!)
      .join("\n");
  }
  return "";
}

/**
 * Parse a transcript file into user/assistant turn pairs.
 */
function parseTranscript(
  content: string,
  startOffset: number
): { turns: Array<{ user: string; assistant: string }>; newOffset: number } {
  let messages: TranscriptMessage[];
  try {
    // Try JSON array first
    messages = JSON.parse(content);
  } catch {
    // Try JSONL (one object per line)
    messages = content
      .split("\n")
      .filter((line) => line.trim())
      .map((line) => {
        try {
          return JSON.parse(line);
        } catch {
          return null;
        }
      })
      .filter(Boolean) as TranscriptMessage[];
  }

  // Filter to user and assistant messages only
  const relevant = messages.filter(
    (m) =>
      m.type === "user" ||
      m.type === "assistant" ||
      m.role === "user" ||
      m.role === "assistant"
  );

  // Skip already-processed messages
  const newMessages = relevant.slice(startOffset);

  // Pair consecutive user/assistant messages
  const turns: Array<{ user: string; assistant: string }> = [];
  for (let i = 0; i < newMessages.length - 1; i++) {
    const msg = newMessages[i];
    const next = newMessages[i + 1];
    const msgRole = msg.type || msg.role;
    const nextRole = next.type || next.role;

    if (msgRole === "user" && nextRole === "assistant") {
      turns.push({
        user: extractText(msg),
        assistant: extractText(next),
      });
      i++; // skip the assistant message on next iteration
    }
  }

  return { turns, newOffset: startOffset + newMessages.length };
}

/**
 * Extract conversation turns from a Claude Code Stop event.
 *
 * Reads the transcript file, parses turn pairs, respects offset.
 * Falls back to prompt + last_assistant_message if transcript unavailable.
 */
export async function extractClaudeCodeTurns(
  raw: Record<string, unknown>
): Promise<ConversationTurn[]> {
  const transcriptPath = raw.transcript_path as string | undefined;
  const sessionId = (raw.session_id as string) || "unknown";
  const cwd = raw.cwd as string | undefined;
  const userId = getUserId();
  const projectId = getProjectId(cwd);
  const timestamp = new Date().toISOString();

  // Primary: read transcript file
  if (transcriptPath) {
    try {
      const content = await readFile(transcriptPath, "utf-8");
      const offset = await readFlushOffset(transcriptPath);
      const { turns, newOffset } = parseTranscript(content, offset);

      if (turns.length > 0) {
        await writeFlushOffset(transcriptPath, newOffset);

        return turns.map((t) => ({
          userMessage: t.user,
          assistantResponse: t.assistant,
          sessionId,
          channel: "claude-code",
          timestamp,
          projectId,
          userId,
        }));
      }
    } catch (error) {
      logError("Failed to read transcript, falling back to prompt fields", error);
    }
  }

  // Fallback: use prompt + last_assistant_message
  const prompt = raw.prompt as string | undefined;
  const lastAssistant = raw.last_assistant_message as string | undefined;

  if (prompt && lastAssistant) {
    return [
      {
        userMessage: prompt,
        assistantResponse: lastAssistant,
        sessionId,
        channel: "claude-code",
        timestamp,
        projectId,
        userId,
      },
    ];
  }

  return [];
}

/**
 * Extract session-end metadata from a Claude Code Stop event.
 */
export function extractClaudeCodeSessionEnd(
  raw: Record<string, unknown>
): SessionEndInput {
  return {
    date: new Date().toISOString().split("T")[0],
    userId: getUserId(),
    shouldCompile: true,
  };
}
