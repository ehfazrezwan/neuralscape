/**
 * Shared interfaces for the adapter pattern.
 *
 * All client adapters normalize their input into these types.
 * Core processing (flush, compile) operates on these types only.
 */

export interface ConversationTurn {
  userMessage: string;
  assistantResponse: string;
  sessionId: string;
  channel: string;
  timestamp: string;
  projectId?: string;
  userId: string;
}

export interface SessionEndInput {
  date: string;
  userId: string;
  shouldCompile: boolean;
}

/**
 * Extracts conversation turns from client-specific stdin data.
 * Returns one turn (OpenClaw, generic) or many turns (Claude Code transcript).
 */
export type TurnExtractor = (
  raw: Record<string, unknown>
) => ConversationTurn[] | Promise<ConversationTurn[]>;

/**
 * Extracts session-end metadata from client-specific stdin data.
 */
export type SessionEndExtractor = (
  raw: Record<string, unknown>
) => SessionEndInput | Promise<SessionEndInput>;
