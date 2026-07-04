/**
 * Structured Stop-summary heuristics (roadmap D2).
 *
 * Builds a `{request, investigated, learned, completed, next_steps}`
 * session note from what the plugin already has at session end — the
 * transcript's user/assistant turns and the PostToolUse observation
 * buffer. Deterministic, no LLM: the note is a continuity artifact
 * ("Previously…" at next SessionStart), not a wiki memory — dense
 * memories still come from the compile-observations flow, which this
 * deliberately does NOT duplicate.
 */

import { isHeartbeat, isSystemMessage, redactPrivate } from "../utils.js";
import type { SessionNoteFields } from "./disclosure.js";

export type { SessionNoteFields };

export interface ObservationRow {
  ts?: string;
  session_id?: string;
  cwd?: string;
  project_id?: string;
  user_id?: string;
  tool?: string;
  input?: Record<string, unknown>;
  output?: string;
}

const FIELD_MAX_CHARS = 400;

function squash(text: string, maxChars = FIELD_MAX_CHARS): string {
  const clean = redactPrivate(text).replace(/\s+/g, " ").trim();
  if (clean.length <= maxChars) return clean;
  return clean.slice(0, maxChars - 1).trimEnd() + "…";
}

/** Claude Code wraps slash-command invocations in XML-ish envelopes. */
function isCommandEnvelope(message: string): boolean {
  const t = message.trimStart();
  return t.startsWith("<command-name>") || t.startsWith("<local-command") || t.startsWith("<command-message>");
}

/** First substantive user message — what the session was actually about. */
export function extractRequest(turns: Array<{ user: string; assistant: string }>): string | undefined {
  for (const turn of turns) {
    const msg = turn.user ?? "";
    if (!msg.trim()) continue;
    if (isHeartbeat(msg) || isSystemMessage(msg) || isCommandEnvelope(msg)) continue;
    return squash(msg, 300);
  }
  return undefined;
}

/** Compact inventory of what the tool observations touched. */
export function extractInvestigated(rows: ObservationRow[]): string | undefined {
  if (!rows || rows.length === 0) return undefined;
  const files = new Set<string>();
  let commands = 0;
  let web = 0;
  let agents = 0;
  for (const row of rows) {
    const input = row.input ?? {};
    const filePath = (input.file_path ?? input.notebook_path) as string | undefined;
    if (typeof filePath === "string" && filePath) {
      // Keep the tail of the path — enough to recognize, cheap to render.
      files.add(filePath.split(/[\\/]/).slice(-2).join("/"));
    }
    if (row.tool === "Bash") commands++;
    if (row.tool === "WebFetch" || row.tool === "WebSearch") web++;
    if (row.tool === "Task" || row.tool === "Agent") agents++;
  }
  const parts: string[] = [];
  if (files.size > 0) {
    const names = [...files].slice(0, 5).join(", ");
    parts.push(`worked in ${files.size} file(s) (${names}${files.size > 5 ? ", …" : ""})`);
  }
  if (commands > 0) parts.push(`ran ${commands} command(s)`);
  if (web > 0) parts.push(`${web} web lookup(s)`);
  if (agents > 0) parts.push(`${agents} subagent task(s)`);
  if (parts.length === 0) return undefined;
  return squash(parts.join("; "));
}

const LEARNED_RE =
  /(?:turns out|discovered|found (?:that|out)|root cause|the (?:issue|problem|bug) (?:was|is)|gotcha:|learned that)/i;

/** Sentences from assistant turns that read like discoveries. */
export function extractLearned(turns: Array<{ user: string; assistant: string }>): string | undefined {
  const hits: string[] = [];
  for (const turn of turns) {
    const text = turn.assistant ?? "";
    if (!text) continue;
    for (const sentence of text.replace(/\s+/g, " ").split(/(?<=[.!?])\s+/)) {
      if (LEARNED_RE.test(sentence)) {
        hits.push(sentence.trim());
        if (hits.length >= 2) break;
      }
    }
    if (hits.length >= 2) break;
  }
  if (hits.length === 0) return undefined;
  return squash(hits.join(" "));
}

/** The final assistant message is usually the wrap-up — use its head. */
export function extractCompleted(turns: Array<{ user: string; assistant: string }>): string | undefined {
  for (let i = turns.length - 1; i >= 0; i--) {
    const text = (turns[i].assistant ?? "").trim();
    if (!text || isSystemMessage(text)) continue;
    return squash(text, 350);
  }
  return undefined;
}

const NEXT_HEADING_RE = /^(?:#+\s*)?(?:\*\*)?\s*(next steps?|remaining|follow[- ]?ups?|todo)\b/i;
const BULLET_RE = /^\s*(?:[-*+•]|\d+[.)])\s+(.+)/;

/**
 * Look for a "next steps"-style heading (or bold label) in the last two
 * assistant messages and collect the bullets under it.
 */
export function extractNextSteps(turns: Array<{ user: string; assistant: string }>): string | undefined {
  const tail = turns.slice(-2).reverse();
  for (const turn of tail) {
    const lines = (turn.assistant ?? "").split("\n");
    for (let i = 0; i < lines.length; i++) {
      if (!NEXT_HEADING_RE.test(lines[i].trim())) continue;
      const bullets: string[] = [];
      for (let j = i + 1; j < lines.length && bullets.length < 5; j++) {
        const m = lines[j].match(BULLET_RE);
        if (m) bullets.push(m[1].trim());
        else if (lines[j].trim() && bullets.length > 0) break;
      }
      if (bullets.length > 0) return squash(bullets.join("; "));
      // Heading with inline content ("Next steps: do X, then Y").
      const inline = lines[i].replace(NEXT_HEADING_RE, "").replace(/^[:\s*]+/, "").trim();
      if (inline) return squash(inline);
    }
  }
  return undefined;
}

/**
 * Build the full structured note. Returns null when the session produced
 * nothing worth carrying forward (no substantive turns and no observations).
 */
export function buildSessionNote(
  turns: Array<{ user: string; assistant: string }>,
  observations: ObservationRow[],
): SessionNoteFields | null {
  const note: SessionNoteFields = {};
  const request = extractRequest(turns);
  const investigated = extractInvestigated(observations);
  const learned = extractLearned(turns);
  const completed = extractCompleted(turns);
  const nextSteps = extractNextSteps(turns);
  if (request) note.request = request;
  if (investigated) note.investigated = investigated;
  if (learned) note.learned = learned;
  if (completed) note.completed = completed;
  if (nextSteps) note.next_steps = nextSteps;
  return Object.keys(note).length > 0 ? note : null;
}

/**
 * The ONE checkpoint call's payload (C4): a session note, zero memories —
 * observation-derived memories stay with compile-observations (no
 * double-store).
 */
export function buildCheckpointPayload(
  note: SessionNoteFields,
  userId: string,
  projectId?: string,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    memories: [],
    session_note: note,
    user_id: userId,
  };
  if (projectId) payload.project_id = projectId;
  return payload;
}
