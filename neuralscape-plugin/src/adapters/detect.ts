/**
 * Client detection — auto-detects which agent framework sent the stdin data.
 */

import type { SessionEndExtractor, TurnExtractor } from "../core/types.js";
import { extractClaudeCodeTurns, extractClaudeCodeSessionEnd } from "./claude-code.js";
import { extractGenericTurns, extractGenericSessionEnd } from "./generic.js";
import { extractOpenClawTurns, extractOpenClawSessionEnd, resolveOpenClawProjectId } from "./openclaw.js";
import { getProjectId } from "../utils.js";

export type ClientType = "openclaw" | "claude-code" | "generic";

/**
 * Detect the client type from the shape of the stdin JSON.
 */
export function detectClient(raw: Record<string, unknown>): ClientType {
  // Claude Code hooks include transcript_path or hook_event_name
  if (raw.transcript_path || raw.hook_event_name) return "claude-code";
  // OpenClaw events have type + action fields
  if (raw.type && raw.action) return "openclaw";
  return "generic";
}

/**
 * Get the turn extractor for a given client type.
 */
export function getTurnExtractor(client: ClientType): TurnExtractor {
  switch (client) {
    case "openclaw":
      return extractOpenClawTurns;
    case "claude-code":
      return extractClaudeCodeTurns;
    case "generic":
      return extractGenericTurns;
  }
}

/**
 * Get the session-end extractor for a given client type.
 */
export function getSessionEndExtractor(client: ClientType): SessionEndExtractor {
  switch (client) {
    case "openclaw":
      return extractOpenClawSessionEnd;
    case "claude-code":
      return extractClaudeCodeSessionEnd;
    case "generic":
      return extractGenericSessionEnd;
  }
}

/**
 * The project a hook event belongs to, for the excluded-projects check. It
 * must be the same id the adapter files the event under: the framework
 * adapter resolves it from the payload (workspace directory or the
 * PROJECT_ID override, else none), every other client from `cwd`.
 */
export function resolveEventProjectId(
  raw: Record<string, unknown>,
  client: ClientType = detectClient(raw),
): string | undefined {
  if (client === "openclaw") return resolveOpenClawProjectId(raw);
  return getProjectId(raw.cwd as string | undefined);
}
