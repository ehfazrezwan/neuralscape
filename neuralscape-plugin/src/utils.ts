/**
 * Shared utilities for Neuralscape Claude Code plugin hooks.
 */

import { appendFile, mkdir, readdir, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join as pathJoin, parse as parsePath } from "node:path";

// ── Configuration ────────────────────────────────────────────────
//
// Plugin reads config from `userConfig` prompts the user fills in at install
// time; Claude Code/Cowork expose those values as CLAUDE_PLUGIN_OPTION_<KEY>
// env vars. Legacy NEURALSCAPE_<KEY> env vars stay supported for one release.

export function readConfig(key: string, fallback = ""): string {
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
  /* v8 ignore next — fallback when neither $USER nor $USERNAME set */
  "";

const REQUEST_TIMEOUT_MS = 8000;

/* v8 ignore start — pre-existing helper, only used by ignored HTTP fns */
function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (NEURALSCAPE_API_KEY) {
    headers["Authorization"] = `Bearer ${NEURALSCAPE_API_KEY}`;
  }
  return headers;
}
/* v8 ignore stop */

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

/* v8 ignore start — pre-existing utility, exercised by integration only */
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
/* v8 ignore stop */

// ── Observation Buffer (PostToolUse → compile-observations) ─────

/**
 * Resolves the directory where per-session observation buffers live.
 *
 * Prefers the official `CLAUDE_PLUGIN_DATA` env var (Claude Code 2.x+);
 * falls back to `~/.neuralscape/observations` for older clients.
 */
export function getObservationDir(): string {
  const claudeData = process.env.CLAUDE_PLUGIN_DATA?.trim();
  if (claudeData) {
    return pathJoin(claudeData, "observations");
  }
  return pathJoin(homedir(), ".neuralscape", "observations");
}

/** Path to the JSONL buffer for a given session id. */
export function getBufferPath(sessionId: string): string {
  // Sanitize session id: replace anything that isn't a safe filename char
  const safe = sessionId.replace(/[^A-Za-z0-9._-]/g, "_") || "unknown";
  return pathJoin(getObservationDir(), `${safe}.jsonl`);
}

/** Path to the per-session "stale" marker dropped by the Stop hook. */
export function getStaleMarkerPath(sessionId: string): string {
  return getBufferPath(sessionId) + ".stale";
}

/**
 * Append a single observation row (already-stringified-able JSON) to the
 * current session's buffer. Creates the directory on first call. Errors are
 * swallowed and logged — the hook must never block tool execution.
 */
export async function appendObservation(row: unknown): Promise<void> {
  try {
    const dir = getObservationDir();
    await mkdir(dir, { recursive: true });
    const sessionId = (row as { session_id?: string }).session_id || "unknown";
    const path = getBufferPath(sessionId);
    await appendFile(path, JSON.stringify(row) + "\n", "utf-8");
  } catch (error) {
    logError("appendObservation failed", error);
  }
}

/**
 * Stats about a single buffer file: line count and the timestamp of the
 * oldest row. Used by the UserPromptSubmit threshold check.
 */
export interface BufferStats {
  path: string;
  sessionId: string;
  lineCount: number;
  oldestTs: string | null;
  isStale: boolean; // true if a `.stale` marker exists alongside the buffer
}

/**
 * Compute stats for a buffer file. Returns lineCount=0 if the file is missing.
 */
export async function getBufferStats(path: string): Promise<BufferStats> {
  const sessionId = parsePath(path).name;
  let lineCount = 0;
  let oldestTs: string | null = null;
  try {
    const { readFile } = await import("node:fs/promises");
    const content = await readFile(path, "utf-8");
    const lines = content.split("\n").filter((l) => l.trim());
    lineCount = lines.length;
    if (lines.length > 0) {
      try {
        const first = JSON.parse(lines[0]);
        oldestTs = typeof first.ts === "string" ? first.ts : null;
      } catch {
        // ignore parse errors — buffer rotation will eventually GC
      }
    }
  } catch {
    // file missing or unreadable: lineCount stays 0
  }
  let isStale = false;
  try {
    await stat(path + ".stale");
    isStale = true;
  } catch {
    isStale = false;
  }
  return { path, sessionId, lineCount, oldestTs, isStale };
}

/**
 * List every non-empty buffer file in the observation directory whose
 * session id is NOT the current one (i.e. left behind by prior sessions).
 */
export async function listPendingBuffers(currentSessionId?: string): Promise<BufferStats[]> {
  const dir = getObservationDir();
  let entries: string[] = [];
  try {
    entries = await readdir(dir);
  } catch {
    return [];
  }
  const results: BufferStats[] = [];
  for (const name of entries) {
    if (!name.endsWith(".jsonl")) continue;
    const path = pathJoin(dir, name);
    const stats = await getBufferStats(path);
    if (stats.lineCount === 0) continue;
    if (currentSessionId && stats.sessionId === currentSessionId) continue;
    results.push(stats);
  }
  return results;
}

/**
 * Truncate a buffer file to empty (preserves the file/inode for reuse).
 * Also removes any `.stale` marker. Used by the compile-observations skill
 * after successfully POSTing all extracted memories.
 */
export async function truncateBuffer(path: string): Promise<void> {
  try {
    await writeFile(path, "", "utf-8");
    try {
      const { unlink } = await import("node:fs/promises");
      await unlink(path + ".stale");
    } catch {
      // no marker — fine
    }
  } catch (error) {
    logError(`truncateBuffer failed for ${path}`, error);
  }
}

/**
 * Write a small sentinel file next to the buffer indicating that the Stop
 * hook flushed without compilation. The next SessionStart picks these up.
 */
export async function markBufferStale(sessionId: string): Promise<void> {
  try {
    const dir = getObservationDir();
    await mkdir(dir, { recursive: true });
    await writeFile(getStaleMarkerPath(sessionId), new Date().toISOString(), "utf-8");
  } catch (error) {
    logError("markBufferStale failed", error);
  }
}

// ── Tool input/output shaping for PostToolUse capture ─────────────

/**
 * Truncate a long string by keeping the head and tail. Returns the original
 * if shorter than head+tail+marker.
 */
export function truncateOutput(text: string, head = 800, tail = 200): string {
  if (typeof text !== "string") {
    text = String(text ?? "");
  }
  const marker = " … [truncated] … ";
  const minLen = head + tail + marker.length;
  if (text.length <= minLen) return text;
  return text.slice(0, head) + marker + text.slice(text.length - tail);
}

/**
 * Pick only the high-signal subset of a tool's input. We deliberately drop
 * full file contents and large patches — they bloat the buffer and the
 * skill doesn't need them to write a memory.
 */
export function pickRelevantInput(toolName: string, input: Record<string, unknown> | undefined): Record<string, unknown> {
  if (!input || typeof input !== "object") return {};
  const KEEP_BY_TOOL: Record<string, string[]> = {
    Edit: ["file_path", "old_string", "new_string"],
    Write: ["file_path"],
    NotebookEdit: ["notebook_path", "cell_id", "edit_mode"],
    Bash: ["command", "description", "timeout"],
    BashOutput: ["bash_id"],
    Task: ["description", "subagent_type", "prompt"],
    WebFetch: ["url", "prompt"],
    WebSearch: ["query"],
  };
  const keep = KEEP_BY_TOOL[toolName];
  const out: Record<string, unknown> = {};
  if (keep) {
    for (const k of keep) {
      if (k in input) out[k] = input[k];
    }
  } else {
    // Unknown tool — keep top-level scalar/short values only
    for (const [k, v] of Object.entries(input)) {
      if (typeof v === "string") {
        out[k] = v.length > 400 ? v.slice(0, 400) + "…" : v;
      } else if (typeof v === "number" || typeof v === "boolean") {
        out[k] = v;
      }
    }
  }
  // Bonus truncation for large payloads in Edit (old_string/new_string can
  // be huge for whole-file rewrites)
  for (const k of ["old_string", "new_string", "prompt", "command"]) {
    const v = out[k];
    if (typeof v === "string" && v.length > 600) {
      out[k] = v.slice(0, 600) + "…";
    }
  }
  return out;
}
