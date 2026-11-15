/**
 * OpenClaw adapter — extracts conversation turns from OpenClaw's event format.
 *
 * Input: { type, action, sessionKey, context: { content, channelId, userMessage } }
 * Output: ConversationTurn[] (single element)
 *
 * The framework's documented `message:sent` context carries the assistant's
 * outgoing text (`content`, `to`, `success`, `channelId`, …) and NO user
 * message; the user's text arrives in the earlier `message:received` event
 * (`content`, `from`, `channelId`, …). So `message:received` is remembered per
 * session (a small file per message next to the observation buffers, consumed by the next
 * `message:sent`) and paired into the turn. A payload that carries the user
 * message itself (`userMessage` / `user_message`) is used as is.
 * Source: https://docs.openclaw.ai/automation/hooks/event-types.md
 */

import { randomBytes } from "node:crypto";
import { mkdir, readdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { join as pathJoin } from "node:path";

import type { ConversationTurn, SessionEndInput } from "../core/types.js";
import { getObservationDir, getProjectId, getUserId, logError, readConfig, redactPrivate } from "../utils.js";

interface OpenClawContext {
  to?: string;
  content?: string;
  success?: boolean;
  channelId?: string;
  conversationId?: string;
  userMessage?: string;
  user_message?: string;
  sessionId?: string;
  messageCount?: number;
  /** Present on some framework events (e.g. agent/command events); not on message:sent. */
  workspaceDir?: string;
  cwd?: string;
}

/**
 * Resolve the project for a turn. A working-directory signal in the payload
 * (`context.workspaceDir`, `context.cwd`, or a top-level `cwd` /
 * `workspaceDir`) goes through the same resolution as every other hook
 * (override setting, `.neuralscape-project` marker, git root, basename).
 * Without one the payload says nothing about a project: only the explicit
 * `PROJECT_ID` override applies, otherwise the turn stays global-scope. (The
 * hook runner's own cwd is not a project signal, so it is never consulted.)
 */
export function resolveOpenClawProjectId(raw: Record<string, unknown>): string | undefined {
  const ctx = (raw.context || {}) as OpenClawContext;
  const dir = [ctx.workspaceDir, ctx.cwd, raw.workspaceDir, raw.cwd].find(
    (v): v is string => typeof v === "string" && v.trim() !== "",
  );
  if (dir) return getProjectId(dir);
  return readConfig("PROJECT_ID", "").trim() || undefined;
}

const INBOUND_SUFFIX = ".lastturn";

/** Filename stem shared by every remembered inbound message of one session. */
function inboundStem(raw: Record<string, unknown>, ctx: OpenClawContext): string {
  const key = (raw.sessionKey as string) || ctx.conversationId || ctx.channelId || "unknown";
  const safe = key.replace(/[^A-Za-z0-9._-]/g, "_") || "unknown";
  return `openclaw-${safe}.inbound.`;
}

/**
 * Remember the user's text (private spans redacted BEFORE it touches disk).
 * One file per inbound message, written under a temp name and renamed into
 * place, so a concurrent reader never sees a half-written file and two
 * inbound messages before a reply do not overwrite each other. Files end in
 * `.lastturn`, so the SessionStart orphan sweep ages them out.
 */
async function rememberInbound(raw: Record<string, unknown>, ctx: OpenClawContext): Promise<void> {
  const text = redactPrivate(ctx.content || "");
  if (!text.trim()) return;
  try {
    const dir = getObservationDir();
    await mkdir(dir, { recursive: true });
    const unique = `${String(Date.now()).padStart(15, "0")}-${process.pid}-${randomBytes(3).toString("hex")}`;
    const final = pathJoin(dir, `${inboundStem(raw, ctx)}${unique}${INBOUND_SUFFIX}`);
    const tmp = `${final}.tmp`;
    await writeFile(tmp, text, { encoding: "utf-8", mode: 0o600 });
    await rename(tmp, final);
  } catch (error) {
    logError("openclaw: could not remember the inbound message", error);
  }
}

/** Consume the session's remembered inbound messages (oldest first, joined). Temp files are never read. */
async function takeInbound(raw: Record<string, unknown>, ctx: OpenClawContext): Promise<string> {
  const stem = inboundStem(raw, ctx);
  try {
    const dir = getObservationDir();
    const names = (await readdir(dir)).filter((n) => n.startsWith(stem) && n.endsWith(INBOUND_SUFFIX)).sort();
    const parts: string[] = [];
    for (const name of names) {
      const path = pathJoin(dir, name);
      try {
        const text = await readFile(path, "utf-8");
        // Claim it by unlinking; if another hook already did, it is theirs.
        await unlink(path);
        if (text.trim()) parts.push(text);
      } catch {
        // gone (claimed by a concurrent hook) — skip
      }
    }
    return parts.join("\n");
  } catch {
    return "";
  }
}

/**
 * Extract the conversation turn from an OpenClaw `message:sent` event, paired
 * with the session's last `message:received`. A `message:received` event
 * itself yields no turn: it is only remembered.
 */
export async function extractOpenClawTurns(
  raw: Record<string, unknown>
): Promise<ConversationTurn[]> {
  const ctx = (raw.context || {}) as OpenClawContext;

  if (raw.type === "message" && raw.action === "received") {
    await rememberInbound(raw, ctx);
    return [];
  }

  return [
    {
      userMessage: ctx.userMessage || ctx.user_message || (await takeInbound(raw, ctx)),
      assistantResponse: ctx.content || "",
      sessionId: (raw.sessionKey as string) || "unknown",
      channel: ctx.channelId || "unknown",
      timestamp: (raw.timestamp as string) || new Date().toISOString(),
      projectId: resolveOpenClawProjectId(raw),
      userId: getUserId(),
    },
  ];
}

/**
 * Extract session-end metadata from an OpenClaw session:end event.
 */
export function extractOpenClawSessionEnd(
  raw: Record<string, unknown>
): SessionEndInput {
  const ctx = (raw.context || {}) as OpenClawContext;
  const messageCount = ctx.messageCount;

  const timestampDate = raw.timestamp ? new Date(raw.timestamp as string) : null;
  const date =
    timestampDate && !Number.isNaN(timestampDate.getTime())
      ? timestampDate.toISOString().split("T")[0]
      : new Date().toISOString().split("T")[0];

  return {
    date,
    userId: getUserId(),
    shouldCompile: messageCount === undefined || messageCount >= 2,
  };
}
