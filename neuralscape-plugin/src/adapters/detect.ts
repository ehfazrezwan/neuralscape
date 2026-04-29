/**
 * Client detection — auto-detects which agent framework sent the stdin data.
 */

import type { SessionEndExtractor, TurnExtractor } from "../core/types.js";
import { extractClaudeCodeTurns, extractClaudeCodeSessionEnd } from "./claude-code.js";
import { extractGenericTurns, extractGenericSessionEnd } from "./generic.js";
import { extractOpenClawTurns, extractOpenClawSessionEnd } from "./openclaw.js";

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
