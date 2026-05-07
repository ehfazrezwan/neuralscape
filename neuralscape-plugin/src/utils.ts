/**
 * Shared utilities for Neuralscape Claude Code plugin hooks.
 */

import { parse as parsePath } from "node:path";

// ── Configuration ────────────────────────────────────────────────
//
// Plugin reads config from `userConfig` prompts the user fills in at install
// time; Claude Code/Cowork expose those values as CLAUDE_PLUGIN_OPTION_<KEY>
// env vars. Legacy NEURALSCAPE_<KEY> env vars stay supported for one release.

function readConfig(key: string, fallback = ""): string {
  const modern = process.env[`CLAUDE_PLUGIN_OPTION_${key}`]?.trim();
  if (modern) return modern;
  const legacy = process.env[`NEURALSCAPE_${key}`]?.trim();
  if (legacy) return legacy;
  return fallback;
}

const NEURALSCAPE_URL = readConfig("URL", "http://localhost:8199").replace(/\/$/, "");

const NEURALSCAPE_API_KEY = readConfig("API_KEY", "");

const NEURALSCAPE_USER_ID =
  readConfig("USER_ID", "") ||
  process.env.USER?.trim() ||
  process.env.USERNAME?.trim() ||
  "";

const REQUEST_TIMEOUT_MS = 8000;

function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (NEURALSCAPE_API_KEY) {
    headers["Authorization"] = `Bearer ${NEURALSCAPE_API_KEY}`;
  }
  return headers;
}

// ── Types ────────────────────────────────────────────────────────

export interface HookInput {
  session_id?: string;
  transcript_path?: string;
  cwd?: string;
  permission_mode?: string;
  hook_event_name?: string;
  tool_name?: string;
  tool_input?: Record<string, unknown>;
  tool_output?: string;
  prompt?: string;
  stop_hook_active?: boolean;
  last_assistant_message?: string;
}

export interface HookOutput {
  continue?: boolean;
  suppressOutput?: boolean;
  stopReason?: string;
  systemMessage?: string;
  hookSpecificOutput?: {
    hookEventName?: string;
    additionalContext?: string;
    [key: string]: unknown;
  };
}

export interface NeuralscapeMemory {
  id: string;
  memory: string;
  category?: string;
  scope?: string;
  project_id?: string;
  tags?: string[];
  created_at?: string;
  source?: string;
}

export interface ContextResponse {
  status: string;
  user_id: string;
  project_id?: string;
  categories: Record<string, NeuralscapeMemory[]>;
}

// ── Stdin Parsing ────────────────────────────────────────────────

export async function parseStdin(): Promise<HookInput> {
  return new Promise((resolve) => {
    let data = "";
    process.stdin.setEncoding("utf-8");
    process.stdin.on("data", (chunk: string) => {
      data += chunk;
    });
    process.stdin.on("end", () => {
      try {
        resolve(JSON.parse(data));
      } catch {
        resolve({});
      }
    });
    // If stdin is already closed or empty, resolve after a short delay
    if (process.stdin.readableEnded) {
      try {
        resolve(JSON.parse(data));
      } catch {
        resolve({});
      }
    }
  });
}

// ── Identity ─────────────────────────────────────────────────────

export function getUserId(): string {
  return NEURALSCAPE_USER_ID;
}

export function hasUserId(): boolean {
  return NEURALSCAPE_USER_ID.length > 0;
}

export function getServiceUrl(): string {
  return NEURALSCAPE_URL;
}

export function hasApiKey(): boolean {
  return NEURALSCAPE_API_KEY.length > 0;
}

/**
 * Project ID is the basename of the working directory.
 * Uses path.parse so it handles both POSIX (/home/foo/bar) and
 * Windows (C:\Users\foo\bar) path separators.
 */
export function getProjectId(cwd?: string): string | undefined {
  const dir = cwd || process.cwd();
  const name = parsePath(dir).name;
  return name || undefined;
}

// ── HTTP Client ──────────────────────────────────────────────────

export async function neuralscapeGet(
  endpoint: string,
  params?: Record<string, string>
): Promise<unknown> {
  const url = new URL(`${NEURALSCAPE_URL}${endpoint}`);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      url.searchParams.set(key, value);
    }
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(url.toString(), {
      method: "GET",
      headers: authHeaders(),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

export async function neuralscapePost(
  endpoint: string,
  body: Record<string, unknown>
): Promise<unknown> {
  const url = `${NEURALSCAPE_URL}${endpoint}`;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

// ── Output Helpers ───────────────────────────────────────────────

export function outputHookResult(result: HookOutput): void {
  process.stdout.write(JSON.stringify(result));
}

export function outputContinue(): void {
  outputHookResult({ continue: true, suppressOutput: true });
}

export function outputWithContext(context: string, hookEventName: string = "SessionStart"): void {
  outputHookResult({
    continue: true,
    hookSpecificOutput: {
      hookEventName,
      additionalContext: context,
    },
  });
}

// ── Message Filtering ───────────────────────────────────────────

const HEARTBEAT_PATTERNS = [
  /^\s*\.{1,3}\s*$/,
  /^heartbeat$/i,
  /^ping$/i,
  /^keepalive$/i,
  /^\/heartbeat$/i,
  /^\s*$/,
];

export function isHeartbeat(message: string): boolean {
  const trimmed = message.trim();
  if (trimmed.length === 0) return true;
  return HEARTBEAT_PATTERNS.some((p) => p.test(trimmed));
}

export function isNoReply(response: string): boolean {
  const trimmed = response.trim();
  return (
    trimmed === "NO_REPLY" ||
    trimmed === "[NO_REPLY]" ||
    trimmed === "(no reply)" ||
    trimmed.length === 0
  );
}

const SYSTEM_MESSAGE_PATTERNS = [
  /^\[system\]/i,
  /^\[auto[-_]?reply\]/i,
  /^\[heartbeat\]/i,
  /^\[status\]/i,
  /^\[internal\]/i,
];

export function isSystemMessage(message: string): boolean {
  return SYSTEM_MESSAGE_PATTERNS.some((p) => p.test(message.trim()));
}

// ── Logging ──────────────────────────────────────────────────────

export function logError(message: string, error?: unknown): void {
  const errMsg = error instanceof Error ? error.message : String(error || "");
  process.stderr.write(`[neuralscape] ${message}${errMsg ? `: ${errMsg}` : ""}\n`);
}
