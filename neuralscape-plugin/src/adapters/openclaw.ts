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
 * session (a small file next to the observation buffers, consumed by the next
 * `message:sent`) and paired into the turn. A payload that carries the user
 * message itself (`userMessage` / `user_message`) is used as is.
 * Source: https://docs.openclaw.ai/automation/hooks/event-types.md
 */

import { mkdir, readFile, unlink, writeFile } from "node:fs/promises";
import { join as pathJoin } from "node:path";

import type { ConversationTurn, SessionEndInput } from "../core/types.js";
import { getObservationDir, getProjectId, getUserId, logError, readConfig } from "../utils.js";

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

/** File (next to the observation buffers) holding the last inbound user message of a session. */
function inboundPath(raw: Record<string, unknown>, ctx: OpenClawContext): string {
  const key = (raw.sessionKey as string) || ctx.conversationId || ctx.channelId || "unknown";
  const safe = key.replace(/[^A-Za-z0-9._-]/g, "_") || "unknown";
  return pathJoin(getObservationDir(), `openclaw-${safe}.inbound.lastturn`);
}

async function rememberInbound(raw: Record<string, unknown>, ctx: OpenClawContext): Promise<void> {
  try {
    await mkdir(getObservationDir(), { recursive: true });
    await writeFile(inboundPath(raw, ctx), ctx.content || "", { encoding: "utf-8", mode: 0o600 });
  } catch (error) {
    logError("openclaw: could not remember the inbound message", error);
  }
}

async function takeInbound(raw: Record<string, unknown>, ctx: OpenClawContext): Promise<string> {
  const path = inboundPath(raw, ctx);
  try {
    const text = await readFile(path, "utf-8");
    await unlink(path);
    return text;
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
