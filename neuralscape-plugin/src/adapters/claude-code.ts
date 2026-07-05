/**
 * Claude Code adapter — extracts conversation turns from the session transcript.
 *
 * Primary: reads the transcript file, parses user/assistant turn pairs,
 *          respects offset to avoid re-flushing.
 * Fallback: if no transcript, uses prompt + last_assistant_message for the last turn.
 */

import { readFile } from "node:fs/promises";
import { getProjectId, getUserId, logError } from "../utils.js";
import type { FlushResult } from "../core/flush.js";
import type { ConversationTurn, SessionEndInput } from "../core/types.js";

const OFFSET_SUFFIX = ".neuralscape-offset";

/**
 * Offsets staged by extractClaudeCodeTurns, committed only after the caller
 * reports delivery (audit 27 #34b): `turnEndOffsets[i]` is the transcript
 * cursor value just past turn i's messages, so a partial flush can advance
 * exactly to the end of the delivered prefix; `finalOffset` covers the whole
 * parsed window (including trailing unpaired messages) and is committed only
 * on full success.
 */
interface StagedFlush {
  turnEndOffsets: number[];
  finalOffset: number;
}

const pendingOffsets = new Map<string, StagedFlush>();

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
): {
  turns: Array<{ user: string; assistant: string; endOffset: number }>;
  newOffset: number;
} {
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

  // Pair consecutive user/assistant messages. Each turn records the cursor
  // value just past its own messages so a partially-delivered flush can
  // commit exactly the delivered prefix (audit 27 #34b).
  const turns: Array<{ user: string; assistant: string; endOffset: number }> = [];
  for (let i = 0; i < newMessages.length - 1; i++) {
    const msg = newMessages[i];
    const next = newMessages[i + 1];
    const msgRole = msg.type || msg.role;
    const nextRole = next.type || next.role;

    if (msgRole === "user" && nextRole === "assistant") {
      turns.push({
        user: extractText(msg),
        assistant: extractText(next),
        endOffset: startOffset + i + 2,
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
        // Stage per-turn offsets; commitClaudeCodeFlush() persists a cursor
        // only as far as the caller reports turns actually delivered.
        pendingOffsets.set(transcriptPath, {
          turnEndOffsets: turns.map((t) => t.endOffset),
          finalOffset: newOffset,
        });

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
 * Read ALL user/assistant turn pairs of a session transcript from the
 * beginning, ignoring (and never touching) the flush offset. Used by the
 * SessionEnd summary hook (roadmap D2), which needs the whole session to
 * build its structured note — unlike the flush path, it is not
 * incremental. Returns [] when the transcript is missing/unreadable.
 */
export async function extractAllTurnPairs(
  raw: Record<string, unknown>
): Promise<Array<{ user: string; assistant: string }>> {
  const transcriptPath = raw.transcript_path as string | undefined;
  if (!transcriptPath) return [];
  try {
    const content = await readFile(transcriptPath, "utf-8");
    return parseTranscript(content, 0).turns;
  } catch (error) {
    logError("Failed to read transcript for session summary", error);
    return [];
  }
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

/**
 * Persist the offset staged by extractClaudeCodeTurns(), bounded by what the
 * flush actually DELIVERED (audit 27 #34b):
 *
 * - full success → commit the full parsed-window offset;
 * - partial failure → commit only past the successfully-POSTed prefix, so
 *   the next session re-flushes exactly the undelivered turns;
 * - nothing delivered (first turn failed) → leave the cursor untouched.
 *
 * If the hook crashes between extract and commit, the on-disk offset stays
 * at its prior value and the next session re-flushes (the backend dedups).
 */
export async function commitClaudeCodeFlush(
  raw: Record<string, unknown>,
  result: FlushResult
): Promise<void> {
  const transcriptPath = raw.transcript_path as string | undefined;
  if (!transcriptPath) return;

  const staged = pendingOffsets.get(transcriptPath);
  if (staged === undefined) return;
  pendingOffsets.delete(transcriptPath);

  let newOffset: number;
  if (result.firstFailedIndex === null) {
    newOffset = staged.finalOffset;
  } else if (result.firstFailedIndex <= 0) {
    return; // nothing confirmed delivered — cursor stays where it was
  } else {
    const committed = staged.turnEndOffsets[result.firstFailedIndex - 1];
    if (committed === undefined) return;
    newOffset = committed;
  }

  try {
    await writeFlushOffset(transcriptPath, newOffset);
  } catch (error) {
    logError("Failed to commit transcript flush offset", error);
  }
}
