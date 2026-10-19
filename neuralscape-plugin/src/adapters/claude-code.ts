/**
 * Claude Code adapter — extracts conversation turns from the session transcript.
 *
 * Primary: reads the transcript file, parses user/assistant turn pairs,
 *          respects offset to avoid re-flushing.
 * Fallback: if no transcript, uses prompt + last_assistant_message for the last turn.
 */

import { readFile, rename, unlink, writeFile } from "node:fs/promises";
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

type ContentBlocks = string | Array<{ type?: string; text?: string }>;

/**
 * One transcript entry. Claude Code writes `{type, message: {role, content}, isSidechain?, isMeta?, ...}` with
 * `content` a string (user prompts) or a block list (assistant text, tool calls, user tool results); older and
 * simplified producers put `content` at the top level, which is still read.
 */
interface TranscriptMessage {
  type?: string; // "user" | "assistant" | "system" | "summary" | ...
  content?: ContentBlocks;
  message?: { role?: string; content?: ContentBlocks; stop_reason?: string | null };
  role?: string;
  isSidechain?: boolean;
  isMeta?: boolean;
  isCompactSummary?: boolean;
  isVisibleInTranscriptOnly?: boolean;
}

/**
 * Read the flush offset (last processed message index) for a transcript.  An offset is used exactly as it
 * was written: it is never moved backwards, so nothing that was already delivered can be delivered again.
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
 * Write the flush offset in one atomic step: a temporary file in the same directory, renamed over the offset
 * file, so an interrupted write can never leave a half-written cursor.  A failure keeps the previous offset
 * (and removes the temporary file).
 */
async function writeFlushOffset(
  transcriptPath: string,
  offset: number
): Promise<void> {
  const target = transcriptPath + OFFSET_SUFFIX;
  const temporary = `${target}.${process.pid}.tmp`;
  try {
    await writeFile(temporary, String(offset), "utf-8");
    await rename(temporary, target);
  } catch (error) {
    await unlink(temporary).catch(() => undefined);
    throw error;
  }
}

function roleOf(msg: TranscriptMessage): string | undefined {
  return msg.type || msg.role;
}

/** Text of a `content` value: a string as is, otherwise the `text` blocks (tool calls, results, thinking are not text). */
function textOf(content: ContentBlocks | undefined): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((block) => block && block.type === "text" && block.text)
      .map((block) => block.text!)
      .join("\n");
  }
  return "";
}

/**
 * User entries Claude Code writes for local slash commands and their output (`<command-name>`, `<command-message>`,
 * `<command-args>`, `<local-command-stdout>`, `<local-command-stderr>`, `<local-command-caveat>`): not something the
 * user said to the assistant.
 */
const LOCAL_COMMAND_WRAPPER = /^\s*<(?:command-name|command-message|command-args|local-command-[a-z]+)>/;

/**
 * Extract the conversation text of a transcript entry. Real entries carry it under `message.content`; a top-level
 * `content` (simplified producers) is the fallback. Entries that are not the user's conversation yield no text:
 * meta entries (injected context), sidechain entries (sub-agent conversations), compaction summaries
 * (`isCompactSummary`, shown only in the transcript view: `isVisibleInTranscriptOnly`; the whole "continued from a
 * previous conversation" summary, which must never be attributed to the user) and local-command wrappers.
 */
function extractText(msg: TranscriptMessage): string {
  if (msg.isMeta === true || msg.isSidechain === true) return "";
  if (msg.isCompactSummary === true || msg.isVisibleInTranscriptOnly === true) return "";
  const text = textOf(msg.message?.content ?? msg.content);
  if ((roleOf(msg) ?? msg.message?.role) === "user" && LOCAL_COMMAND_WRAPPER.test(text)) return "";
  return text;
}

function parseMessages(content: string): TranscriptMessage[] {
  try {
    // Try JSON array first
    const parsed = JSON.parse(content);
    if (Array.isArray(parsed)) return parsed;
  } catch {
    // not one JSON document: JSONL below
  }
  // JSONL (one object per line; a single-line file parses as one object above and lands here too)
  return content
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

/** The user/assistant entries, in order. An offset counts positions in this list. */
function relevantMessages(content: string): TranscriptMessage[] {
  return parseMessages(content).filter((m) => {
    const role = roleOf(m) ?? m.message?.role;
    return role === "user" || role === "assistant";
  });
}

interface ParsedTurn {
  user: string;
  /** Every assistant text entry of the turn, joined: what is delivered. */
  assistant: string;
  /** The text of the turn's LAST assistant entry (the wrap-up), kept apart for the session note. */
  final: string;
  endOffset: number;
}

/**
 * Parse a transcript file into user/assistant turns.
 *
 * A turn is one real user prompt (an entry with text; a user entry that only carries tool results has none and
 * does not end a turn) and every assistant text entry up to the next real prompt. Entries without text
 * (thinking, tool calls, tool results) are skipped for pairing but still count as positions, so offsets stay
 * comparable with those written before. The returned `newOffset` never passes a user prompt that has no
 * assistant text yet: it stays unread and is picked up by the next flush.
 */
function parseTranscript(
  content: string,
  startOffset: number
): {
  turns: ParsedTurn[];
  newOffset: number;
} {
  const window = relevantMessages(content).slice(startOffset);
  const turns: ParsedTurn[] = [];
  let user: { text: string; index: number } | null = null;
  let assistantParts: string[] = [];

  window.forEach((entry, position) => {
    const absolute = startOffset + position;
    const role = roleOf(entry) ?? entry.message?.role;
    const text = extractText(entry);
    if (!text) return;
    if (role === "user") {
      if (user && assistantParts.length > 0) {
        turns.push({ user: user.text, assistant: assistantParts.join("\n"), final: assistantParts[assistantParts.length - 1], endOffset: absolute });
      }
      user = { text, index: absolute };
      assistantParts = [];
    } else if (user) {
      assistantParts.push(text);
    }
  });

  const windowEnd = startOffset + window.length;
  let newOffset = windowEnd;
  if (user) {
    const pending: { text: string; index: number } = user;
    if (assistantParts.length > 0) {
      turns.push({ user: pending.text, assistant: assistantParts.join("\n"), final: assistantParts[assistantParts.length - 1], endOffset: windowEnd });
    } else {
      newOffset = pending.index; // an unanswered turn is not consumed
    }
  }
  return { turns, newOffset };
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
): Promise<Array<{ user: string; assistant: string; final: string }>> {
  const transcriptPath = raw.transcript_path as string | undefined;
  if (!transcriptPath) return [];
  try {
    const content = await readFile(transcriptPath, "utf-8");
    return parseTranscript(content, 0).turns.map(({ user, assistant, final }) => ({ user, assistant, final }));
  } catch (error) {
    logError("Failed to read transcript for session summary", error);
    return [];
  }
}

/**
 * The text of the last ``limit`` real user prompts of the transcript, oldest first, whether or not they were
 * delivered and including a prompt that has no answer yet.  The PreCompact snapshot uses it: it must show the
 * task that is under way, and it does not depend on what the flush has delivered.  Returns [] when the
 * transcript is missing or unreadable.
 */
export async function extractRecentUserPrompts(raw: Record<string, unknown>, limit = 10): Promise<string[]> {
  const transcriptPath = raw.transcript_path as string | undefined;
  if (!transcriptPath) return [];
  try {
    const content = await readFile(transcriptPath, "utf-8");
    const prompts = relevantMessages(content)
      .filter((entry) => (roleOf(entry) ?? entry.message?.role) === "user")
      .map(extractText)
      .filter((text) => text.length > 0);
    return prompts.slice(-limit);
  } catch (error) {
    logError("Failed to read transcript for the compact snapshot", error);
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
 * at its prior value and the next session re-flushes. A re-sent turn is a new extraction call: the flush route
 * does not de-duplicate turns (memory-level de-duplication happens when the turns are compiled).
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
