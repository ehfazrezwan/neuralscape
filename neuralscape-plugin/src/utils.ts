/**
 * Shared utilities for Neuralscape Claude Code plugin hooks.
 */

// ── Configuration ────────────────────────────────────────────────

const NEURALSCAPE_URL =
  process.env.NEURALSCAPE_URL?.replace(/\/$/, "") || "http://localhost:8199";

const NEURALSCAPE_API_KEY = process.env.NEURALSCAPE_API_KEY || "";

const DEFAULT_USER_ID = process.env.NEURALSCAPE_USER_ID || "ehfaz";

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
  return DEFAULT_USER_ID;
}

export function getProjectId(cwd?: string): string | undefined {
  const dir = cwd || process.cwd();
  // Use the last component of the path as project_id
  const parts = dir.split("/").filter(Boolean);
  return parts.length > 0 ? parts[parts.length - 1] : undefined;
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

// ── Logging ──────────────────────────────────────────────────────

export function logError(message: string, error?: unknown): void {
  const errMsg = error instanceof Error ? error.message : String(error || "");
  process.stderr.write(`[neuralscape] ${message}${errMsg ? `: ${errMsg}` : ""}\n`);
}
