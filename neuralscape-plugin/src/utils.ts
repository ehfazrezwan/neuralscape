/**
 * Shared utilities for Neuralscape Claude Code plugin hooks.
 */

import { appendFile, mkdir, open, readdir, readFile, stat, unlink, writeFile } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join as pathJoin, parse as parsePath } from "node:path";

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

// Single source of truth for the service URL: the base URL is baked into the
// plugin's `.mcp.json` (the connector endpoint) per distribution channel. The
// hooks derive the same base from that file so there is exactly one place to
// change it. Strips the trailing `/mcp/` connector suffix to get the API base.
// Ignores an un-baked `${...}` template (transitional) and any read/parse error.
// Parses with the URL constructor (not a raw regex) so query/hash/port variants
// normalize correctly before the `/mcp/` segment is removed.
export function readBakedUrl(): string {
  const root = process.env.CLAUDE_PLUGIN_ROOT?.trim();
  if (!root) return "";
  try {
    const cfg = JSON.parse(readFileSync(pathJoin(root, ".mcp.json"), "utf8"));
    const raw = cfg?.mcpServers?.neuralscape?.url;
    if (typeof raw !== "string" || raw.includes("${")) return "";
    const parsed = new URL(raw);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return "";
    // `parsed.origin` already excludes query/hash; `parsed.pathname` is the path
    // alone. Build the base directly — never assign back to `parsed.pathname`,
    // which the URL object re-normalizes "" to "/".
    const base = parsed.pathname.replace(/\/mcp\/?$/, "").replace(/\/$/, "");
    return `${parsed.origin}${base}`;
  } catch {
    return "";
  }
}

// Precedence: explicit config override → baked `.mcp.json` URL → local dev.
const NEURALSCAPE_URL = (
  readConfig("URL", "") ||
  readBakedUrl() ||
  "http://localhost:8199"
).replace(/\/$/, "");

const NEURALSCAPE_API_KEY = readConfig("API_KEY", "");

// Deliberate zero-config fallback: an unconfigured install identifies as the
// OS username so hook captures stay continuous on single-user/local servers.
// Removing it would orphan memories existing installs stored under that id.
// Token-authenticated servers ignore client-claimed ids (identity comes from
// the Bearer token), so this can't spoof identity on multi-user deployments.
const CONFIGURED_USER_ID = readConfig("USER_ID", "");
const NEURALSCAPE_USER_ID =
  CONFIGURED_USER_ID ||
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
  /** PostToolUse: the tool's result as Claude Code sends it — a string for
   *  some tools, an object (e.g. Bash `{stdout, stderr, ...}`) for most. */
  tool_response?: unknown;
  /** @deprecated Not sent by Claude Code; kept as a fallback for older or
   *  non-Claude-Code callers. Use `tool_response`. */
  tool_output?: unknown;
  prompt?: string;
  stop_hook_active?: boolean;
  last_assistant_message?: string;
  /** SessionStart: "startup" | "resume" | "clear" | "compact". */
  source?: string;
  /** PreCompact: "manual" (/compact) or "auto". */
  trigger?: string;
  /** PreCompact: instructions passed to /compact, if any. */
  custom_instructions?: string;
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
  /** Full content. Absent/empty on index-level rows (`fields=index`). */
  memory?: string;
  category?: string;
  scope?: string;
  project_id?: string;
  tags?: string[];
  created_at?: string;
  source?: string;
  // Memory-model v2 / retrieval economics (C1) — null on legacy memories.
  observation_type?: string | null;
  title?: string | null;
  token_estimate?: number | null;
}

export interface ContextResponse {
  status: string;
  user_id: string;
  project_id?: string;
  categories: Record<string, NeuralscapeMemory[]>;
  // Authoritative org standards (visibility=standard). Rendered as a binding
  // block that overrides personal preferences on conflict. Empty unless the
  // server has STANDARDS_ENABLED.
  standards?: NeuralscapeMemory[];
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

/**
 * Thrown by parseStdinStrict when the hook payload isn't valid JSON.
 * Per the hook exit-code taxonomy (see README), this is a CLIENT BUG:
 * hooks that can safely fail loud exit 2 so the malformed input is
 * surfaced instead of silently swallowed.
 */
export class MalformedHookInputError extends Error {
  constructor(detail: string) {
    super(`malformed hook input (client bug): ${detail}`);
    this.name = "MalformedHookInputError";
  }
}

/**
 * Strict variant of parseStdin: rejects with MalformedHookInputError on
 * empty or non-JSON stdin instead of resolving `{}`. New hooks use this
 * to implement the exit-code taxonomy; parseStdin stays lenient for the
 * legacy paths that must never change behavior mid-release.
 */
export async function parseStdinStrict(): Promise<HookInput> {
  const data: string = await new Promise((resolve) => {
    let buf = "";
    process.stdin.setEncoding("utf-8");
    process.stdin.on("data", (chunk: string) => {
      buf += chunk;
    });
    process.stdin.on("end", () => resolve(buf));
    if (process.stdin.readableEnded) resolve(buf);
  });
  if (!data.trim()) {
    throw new MalformedHookInputError("empty stdin");
  }
  try {
    return JSON.parse(data) as HookInput;
  } catch {
    throw new MalformedHookInputError("stdin is not valid JSON");
  }
}
/* v8 ignore stop */

// ── Excluded projects (roadmap D4) ───────────────────────────────
//
// Comma-separated globs of project ids the plugin must never capture from
// or inject into. Config: EXCLUDED_PROJECTS (CLAUDE_PLUGIN_OPTION_ /
// NEURALSCAPE_ prefixes), plus the roadmap-spelled `NS_EXCLUDED_PROJECTS`
// raw env var as an alias. Matching is case-insensitive; `*` and `?` are
// the only wildcards.

/** Compile one project glob to an anchored, case-insensitive RegExp. */
export function globToRegExp(glob: string): RegExp {
  const escaped = glob
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replace(/\*/g, ".*")
    .replace(/\?/g, ".");
  return new RegExp(`^${escaped}$`, "i");
}

export function getExcludedProjectGlobs(): string[] {
  const raw = readConfig("EXCLUDED_PROJECTS", "") || process.env.NS_EXCLUDED_PROJECTS?.trim() || "";
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

// isProjectExcluded runs in hook hot paths (once per PostToolUse event) —
// cache the compiled regexes keyed by the raw glob list so config parsing
// and RegExp construction happen once per process, not once per tool call.
let excludedGlobCacheKey: string | null = null;
let excludedGlobCache: RegExp[] = [];

function getExcludedProjectRegexps(): RegExp[] {
  const globs = getExcludedProjectGlobs();
  const key = globs.join(",");
  if (key !== excludedGlobCacheKey) {
    excludedGlobCacheKey = key;
    excludedGlobCache = globs.map(globToRegExp);
  }
  return excludedGlobCache;
}

/** True when the resolved project id matches any excluded-projects glob. */
export function isProjectExcluded(projectId: string | undefined): boolean {
  if (!projectId) return false;
  return getExcludedProjectRegexps().some((re) => re.test(projectId));
}

// ── Privacy: <private> tag redaction (roadmap D4) ────────────────

/**
 * One `<private>` tag, tolerant of whitespace, attributes and case (`< private >`, `<PRIVATE reason="x">`,
 * `</ private>`), and of the HTML-escaped spelling (`&lt;private&gt;`). ASCII only: full-width or zero-width-joined
 * look-alikes are not recognised. `private` must be a whole word, so `<privateer>` and `<private-note>` are not
 * tags. Group 1 is set for a closer.
 */
const PRIVATE_TAG = /(?:<|&lt;)\s{0,20}(\/)?\s{0,20}private(?![\w-])[^<>]{0,200}?(?:>|&gt;)/gi;

/**
 * Fail-closed matcher: the same, but a tag whose terminator is missing (`<private` at the end of the text,
 * `<private foo` with no `>`) still counts, so a half-written opener cannot hide a span from the redactor.
 * Both patterns are bounded (whitespace and attribute lengths are capped, no nested quantifiers), so the work
 * is linear in the text even for a megabyte of repeated openers.
 */
const PRIVATE_TAG_LOOSE = /(?:<|&lt;)\s{0,20}(\/)?\s{0,20}private(?![\w-])(?:[^<>]{0,200}?(?:>|&gt;))?/gi;

export interface PrivateSpan {
  start: number;
  end: number;
  /** An opener that is never closed (it runs to the end of the text). */
  unclosed: boolean;
}

/**
 * Find the private spans of a text. Spans NEST: an opener inside a span deepens it and the span ends at the
 * closer that balances the outermost opener, so text between an inner and the outer closer stays inside.
 * An opener that is never balanced ends at the end of the text (`unclosed`).
 * With `failClosed`, a closer with no opener before it (the opener was in text that was cut away or sits in
 * another entry) makes everything from the previous span's end through that closer a span.
 */
export function findPrivateSpans(text: string, failClosed = false): PrivateSpan[] {
  const spans: PrivateSpan[] = [];
  let depth = 0;
  let start = 0;
  let floor = 0; // end of the last span: a stray closer redacts back to here
  for (const tag of text.matchAll(failClosed ? PRIVATE_TAG_LOOSE : PRIVATE_TAG)) {
    const at = tag.index as number;
    const end = at + tag[0].length;
    if (tag[1] === undefined) {
      if (depth === 0) start = at;
      depth++;
    } else if (depth > 0) {
      depth--;
      if (depth === 0) {
        spans.push({ start, end, unclosed: false });
        floor = end;
      }
    } else if (failClosed) {
      spans.push({ start: floor, end, unclosed: false });
      floor = end;
    }
  }
  if (depth > 0) spans.push({ start, end: text.length, unclosed: true });
  return spans;
}

/**
 * Redact `<private>…</private>` spans from text the plugin is about to
 * store or transmit. Case-insensitive. Idempotent. Spans nest (see
 * findPrivateSpans) and the tag is matched tolerantly (whitespace,
 * attributes, `&lt;private&gt;`).
 *
 * An UNMATCHED opener (e.g. prose *about* the tag: "use <private> to mark
 * secrets") is bounded by default (audit 27 #34a): it redacts at most through
 * the next sentence boundary or end of line — never to end-of-string — and
 * leaves a `[redacted:unclosed]` warning marker so the truncation is visible
 * instead of silently eating the rest of the transcript.
 *
 * `failClosed` is for text that LEAVES the machine or is written to disk:
 * an unmatched opener redacts through the END OF THE TEXT, and a closer with
 * no opener redacts back to the start of the text. A private span can run
 * over several lines or sentences (a tool result, a multi-paragraph answer),
 * so bounding it to the line would publish the rest of the span;
 * over-redacting prose that merely mentions the tag is the safe error.
 */
export function redactPrivate(text: string, options: { failClosed?: boolean } = {}): string {
  if (typeof text !== "string") return text;
  const failClosed = options.failClosed === true;
  const spans = findPrivateSpans(text, failClosed);
  if (spans.length === 0) return text;
  let out = "";
  let pos = 0;
  for (const span of spans) {
    out += text.slice(pos, span.start);
    pos = span.end;
    if (!span.unclosed) {
      out += "[redacted]";
    } else if (failClosed) {
      out += "[redacted:unclosed]";
    } else {
      // Bounded: through the earliest sentence end or the end of the line, then carry on with the rest.
      const rest = text.slice(span.start);
      const bound = /^[^\n]*?(?:[.!?](?=\s|$)|(?=\n)|$)/.exec(rest);
      const cut = bound ? bound[0].length : rest.length;
      out += "[redacted:unclosed]";
      pos = span.start + cut;
      out += redactPrivate(text.slice(pos), options);
      return out;
    }
  }
  return out + text.slice(pos);
}

/**
 * Per-session compile bookkeeping for the Stop hook, next to the buffers.
 * The name ends in `.lastturn` so the SessionStart orphan sweep ages it out
 * with the other per-session leftovers.
 */
export const COMPILE_STATE_SUFFIX = ".compile.lastturn";

export interface CompileState {
  /** Dates (YYYY-MM-DD) with turns delivered since their last successful compile. */
  dirtyDates: string[];
  /** When a compile last succeeded (ms since epoch); 0 when never. */
  compiledAt: number;
}

export function getCompileStatePath(sessionId: string): string {
  const safe = sessionId.replace(/[^A-Za-z0-9._-]/g, "_") || "unknown";
  return pathJoin(getObservationDir(), `${safe}${COMPILE_STATE_SUFFIX}`);
}

export async function readCompileState(sessionId: string): Promise<CompileState> {
  try {
    const parsed = JSON.parse(await readFile(getCompileStatePath(sessionId), "utf-8"));
    const dirtyDates = Array.isArray(parsed?.dirtyDates)
      ? parsed.dirtyDates.filter((d: unknown): d is string => typeof d === "string").slice(-3)
      : [];
    const compiledAt = Number.isFinite(parsed?.compiledAt) ? parsed.compiledAt : 0;
    return { dirtyDates, compiledAt };
  } catch {
    return { dirtyDates: [], compiledAt: 0 };
  }
}

export async function writeCompileState(sessionId: string, state: CompileState): Promise<void> {
  try {
    await mkdir(getObservationDir(), { recursive: true });
    await writeFile(getCompileStatePath(sessionId), JSON.stringify(state), { encoding: "utf-8", mode: 0o600 });
  } catch (error) {
    logError("writing compile state failed (non-critical)", error);
  }
}

/* v8 ignore start — pre-existing utility, exercised by integration only */
// ── Identity ─────────────────────────────────────────────────────

export function getUserId(): string {
  return NEURALSCAPE_USER_ID;
}

export function hasUserId(): boolean {
  return NEURALSCAPE_USER_ID.length > 0;
}

/** Where the effective user id came from: the user's own config, the OS
 *  login-name fallback (see the comment above NEURALSCAPE_USER_ID), or nowhere. */
export function getUserIdSource(): "config" | "os" | "none" {
  if (CONFIGURED_USER_ID) return "config";
  return NEURALSCAPE_USER_ID ? "os" : "none";
}

export function getServiceUrl(): string {
  return NEURALSCAPE_URL;
}

export function hasApiKey(): boolean {
  return NEURALSCAPE_API_KEY.length > 0;
}

/** File name the plugin walks up the tree to find, to pin one project id
 *  across every subdirectory of a repo (e.g. a monorepo whose service and
 *  plugin live in sibling folders). Its trimmed first line, if any, IS the
 *  project id; an empty marker falls back to the marker directory's basename. */
const PROJECT_MARKER = ".neuralscape-project";

/**
 * Walk up from `start` (inclusive) to the filesystem root looking for an
 * entry named `name`. Returns the absolute path of the containing directory,
 * or undefined. Synchronous on purpose — `getProjectId` is sync and called
 * from hook hot paths.
 */
function findUp(start: string, name: string): string | undefined {
  let dir = start;
  // Bounded by the root check below (dirname(root) === root), so this can't loop forever.
  for (;;) {
    if (existsSync(pathJoin(dir, name))) return dir;
    const parent = dirname(dir);
    if (parent === dir) return undefined;
    dir = parent;
  }
}

/**
 * Resolve the project id with a deterministic precedence so the SAME repo
 * always reports the SAME id regardless of which subdirectory a hook runs in:
 *
 *   1. Explicit config override — `PROJECT_ID` (CLAUDE_PLUGIN_OPTION_PROJECT_ID
 *      / NEURALSCAPE_PROJECT_ID). Pins one id everywhere.
 *   2. A `.neuralscape-project` marker file found by walking up from `cwd`.
 *      Its trimmed first line is the id; an empty marker → its dir basename.
 *   3. The git repo root basename (walk up for `.git`).
 *   4. Legacy fallback: the basename of `cwd`.
 *
 * Uses path.parse so it handles both POSIX and Windows separators.
 */
export function getProjectId(cwd?: string): string | undefined {
  const override = readConfig("PROJECT_ID", "").trim();
  if (override) return override;

  const dir = cwd || process.cwd();

  const markerDir = findUp(dir, PROJECT_MARKER);
  if (markerDir) {
    try {
      const id = readFileSync(pathJoin(markerDir, PROJECT_MARKER), "utf8").trim().split(/\r?\n/)[0]?.trim();
      if (id) return id;
    } catch {
      /* unreadable marker — fall through to its directory basename */
    }
    return parsePath(markerDir).name || undefined;
  }

  const gitDir = findUp(dir, ".git");
  if (gitDir) return parsePath(gitDir).name || undefined;

  return parsePath(dir).name || undefined;
}

// ── HTTP Client ──────────────────────────────────────────────────

/** The service answered, but not with 2xx. `message` keeps the legacy
 *  `HTTP <status>: <text>` shape so log lines and old assertions still match. */
export class NeuralscapeHttpError extends Error {
  readonly status: number;
  constructor(status: number, statusText = "") {
    super(`HTTP ${status}: ${statusText}`);
    this.name = "NeuralscapeHttpError";
    this.status = status;
  }
}

/** No usable answer: DNS/connection failure, reset, or the request timeout. */
export class NeuralscapeNetworkError extends Error {
  constructor(detail: string) {
    super(`network error: ${detail}`);
    this.name = "NeuralscapeNetworkError";
  }
}

/** What a failed call means for the user. `auth`, `not_found` and `client`
 *  mean the service IS reachable but the plugin's setup is wrong; `server`
 *  and `network` mean the service is down or failing. */
export type FailureKind = "auth" | "not_found" | "client" | "server" | "network";

export function classifyFailure(error: unknown): FailureKind {
  if (error instanceof NeuralscapeHttpError) {
    if (error.status === 401 || error.status === 403) return "auth";
    if (error.status === 404) return "not_found";
    if (error.status >= 500) return "server";
    return "client";
  }
  // NeuralscapeNetworkError, a bare fetch/abort error, or anything unexpected:
  // treated as a transport failure, the pre-existing behavior.
  return "network";
}

/** Failures that count toward the "unreachable" streak: the service is down
 *  or failing. Config problems (auth/404/other 4xx) do not: the service answered. */
export function isUnreachableKind(kind: FailureKind): boolean {
  return kind === "network" || kind === "server";
}

/** Count a failed call toward the unreachable streak when (and only when) it
 *  was a transport/server failure. Returns the kind, and the new count when
 *  the streak advanced. Never throws. */
export async function recordFailure(
  error: unknown,
): Promise<{ kind: FailureKind; count: number | null }> {
  const kind = classifyFailure(error);
  const count = isUnreachableKind(kind) ? await recordTransportFailure() : null;
  return { kind, count };
}

function wrapTransportError(error: unknown): unknown {
  if (error instanceof NeuralscapeHttpError || error instanceof NeuralscapeNetworkError) return error;
  // fetch() rejects with TypeError on connection failure; abort() on timeout.
  if (error instanceof Error && (error.name === "AbortError" || error.name === "TimeoutError")) {
    return new NeuralscapeNetworkError(`timed out after ${REQUEST_TIMEOUT_MS}ms`);
  }
  if (error instanceof TypeError) {
    return new NeuralscapeNetworkError("could not connect");
  }
  // e.g. a 2xx body that is not JSON: leave as is (classified as network, as before).
  return error;
}

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
      throw new NeuralscapeHttpError(response.status, response.statusText);
    }
    return await response.json();
  } catch (error) {
    throw wrapTransportError(error);
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
      throw new NeuralscapeHttpError(response.status, response.statusText);
    }
    return await response.json();
  } catch (error) {
    throw wrapTransportError(error);
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

/**
 * Path to the per-session "session ended" marker, written once by the
 * SessionEnd hook. (Versions up to 2.9.3 wrote `<buffer>.stale` from the Stop
 * hook, which fires after every turn and so says nothing about the session
 * having ended; those markers are ignored and removed.)
 */
export function getEndedMarkerPath(sessionId: string): string {
  return getBufferPath(sessionId) + ".ended";
}

/** Marker suffix written by plugin versions up to 2.9.3 (meaningless; cleaned up). */
const LEGACY_STALE_SUFFIX = ".stale";
const ENDED_SUFFIX = ".ended";

/** Cool-down marker of the UserPromptSubmit compile instruction, next to the buffer. */
export const COMPILE_ARMED_SUFFIX = ".compile-armed";

/** Local copy of the last compact snapshot, next to the buffers. */
const COMPACT_SNAPSHOT_FILE_SUFFIX = ".compact-snapshot.json";

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
  /** Last write to the buffer (ms since epoch); 0 when the file is missing. */
  mtimeMs: number;
  /**
   * True when the SessionEnd hook marked this session as ended and the buffer
   * has not been written since (a session that resumed and kept working is not ended).
   */
  isEnded: boolean;
}

/**
 * Compute stats for a buffer file. Returns lineCount=0 if the file is missing.
 */
export async function getBufferStats(path: string): Promise<BufferStats> {
  const sessionId = parsePath(path).name;
  let lineCount = 0;
  let oldestTs: string | null = null;
  try {
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
  let mtimeMs = 0;
  try {
    mtimeMs = (await stat(path)).mtimeMs;
  } catch {
    mtimeMs = 0;
  }
  let isEnded = false;
  try {
    const marker = await stat(path + ENDED_SUFFIX);
    isEnded = lineCount > 0 && marker.mtimeMs >= mtimeMs;
  } catch {
    isEnded = false;
  }
  return { path, sessionId, lineCount, oldestTs, mtimeMs, isEnded };
}

/** A buffer nobody has written to for this long belongs to a session that is gone (it crashed before SessionEnd). */
export const DEFAULT_PENDING_IDLE_HOURS = 12;

/** Hours of silence after which an un-ended buffer counts as abandoned (`PENDING_IDLE_HOURS`, default 12). */
export function getPendingIdleHours(): number {
  const parsed = parseFloat(readConfig("PENDING_IDLE_HOURS", ""));
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_PENDING_IDLE_HOURS;
}

/** Local compact snapshots are continuity artifacts like the stored ones: dropped after two weeks. */
const COMPACT_SNAPSHOT_FILE_MAX_AGE_MS = 14 * 24 * 60 * 60 * 1000;

/**
 * Remove marker files that can no longer matter. The compile-observations
 * skill empties a buffer with the Write tool, which cannot delete the
 * markers next to it, so they would otherwise accumulate:
 *   - `.ended` markers whose buffer is empty or missing;
 *   - `.stale` markers written by plugin versions up to 2.9.3 (they came
 *     from the per-turn Stop hook and carry no information);
 *   - local compact snapshots older than two weeks.
 * Never throws.
 */
export async function cleanupOrphanMarkers(entries: string[], now: number = Date.now()): Promise<void> {
  const dir = getObservationDir();
  const buffers = new Set(entries.filter((n) => n.endsWith(".jsonl")));
  for (const name of entries) {
    try {
      if (name.endsWith(".jsonl" + LEGACY_STALE_SUFFIX)) {
        await unlink(pathJoin(dir, name));
      } else if (name.endsWith(".jsonl" + ENDED_SUFFIX)) {
        const bufferName = name.slice(0, -ENDED_SUFFIX.length);
        let empty = !buffers.has(bufferName);
        if (!empty) empty = (await stat(pathJoin(dir, bufferName))).size === 0;
        if (empty) await unlink(pathJoin(dir, name));
      } else if (name.endsWith(".jsonl" + COMPILE_ARMED_SUFFIX)) {
        // Meaningful only while its buffer holds rows; also bounded by age
        // like the other per-session leftovers.
        const bufferName = name.slice(0, -COMPILE_ARMED_SUFFIX.length);
        let remove = !buffers.has(bufferName);
        if (!remove) remove = (await stat(pathJoin(dir, bufferName))).size === 0;
        if (!remove) remove = now - (await stat(pathJoin(dir, name))).mtimeMs > COMPACT_SNAPSHOT_FILE_MAX_AGE_MS;
        if (remove) await unlink(pathJoin(dir, name));
      } else if (name.endsWith(COMPACT_SNAPSHOT_FILE_SUFFIX) || name.endsWith(".lastturn")) {
        const age = now - (await stat(pathJoin(dir, name))).mtimeMs;
        if (age > COMPACT_SNAPSHOT_FILE_MAX_AGE_MS) await unlink(pathJoin(dir, name));
      }
    } catch {
      // raced with another hook or already gone — fine
    }
  }
}

/**
 * List the non-empty buffers left behind by sessions that are OVER.
 *
 * A buffer belongs to a finished session when it is not the current one and
 *   (a) the SessionEnd hook marked it ended and it has not been written since, or
 *   (b) nobody has written to it for `PENDING_IDLE_HOURS` (default 12): the
 *       session crashed or was killed and SessionEnd never fired.
 * A buffer that is being written by another terminal's live session is neither,
 * so it is never offered for compilation and truncation. Markers from empty
 * or missing buffers are removed as a side effect.
 */
export async function listPendingBuffers(
  currentSessionId?: string,
  now: number = Date.now(),
): Promise<BufferStats[]> {
  const dir = getObservationDir();
  let entries: string[] = [];
  try {
    entries = await readdir(dir);
  } catch {
    return [];
  }
  await cleanupOrphanMarkers(entries, now);
  const idleMs = getPendingIdleHours() * 60 * 60 * 1000;
  const results: BufferStats[] = [];
  for (const name of entries) {
    if (!name.endsWith(".jsonl")) continue;
    const path = pathJoin(dir, name);
    const stats = await getBufferStats(path);
    if (stats.lineCount === 0) continue;
    if (currentSessionId && stats.sessionId === currentSessionId) continue;
    const abandoned = now - stats.mtimeMs >= idleMs;
    if (!stats.isEnded && !abandoned) continue;
    results.push(stats);
  }
  return results;
}

/**
 * Truncate a buffer file to empty (preserves the file/inode for reuse).
 * Also removes the end marker. (The compile-observations skill itself empties
 * buffers with the Write tool; the next `listPendingBuffers` call removes the
 * marker it leaves behind.)
 */
export async function truncateBuffer(path: string): Promise<void> {
  try {
    await writeFile(path, "", "utf-8");
    for (const suffix of [ENDED_SUFFIX, LEGACY_STALE_SUFFIX]) {
      try {
        await unlink(path + suffix);
      } catch {
        // no marker — fine
      }
    }
  } catch (error) {
    logError(`truncateBuffer failed for ${path}`, error);
  }
}

/**
 * Write the "session ended" marker next to the buffer. Called once by the
 * SessionEnd hook (never by Stop, which fires after every turn). The next
 * SessionStart, in any terminal, treats the buffer as ready to compile
 * as long as it is not written to again after the marker.
 */
export async function markBufferEnded(sessionId: string): Promise<void> {
  try {
    const dir = getObservationDir();
    await mkdir(dir, { recursive: true });
    await writeFile(getEndedMarkerPath(sessionId), new Date().toISOString(), { encoding: "utf-8", mode: 0o600 });
  } catch (error) {
    logError("markBufferEnded failed", error);
  }
}

// ── Local compact snapshot (survives a slow or unreachable service) ──

export interface LocalCompactSnapshot {
  sessionId: string;
  projectId?: string;
  content: string;
  createdAt: string;
}

/** Path to the local copy of a session's latest compact snapshot. */
export function getCompactSnapshotPath(sessionId: string): string {
  const safe = sessionId.replace(/[^A-Za-z0-9._-]/g, "_") || "unknown";
  return pathJoin(getObservationDir(), `${safe}${COMPACT_SNAPSHOT_FILE_SUFFIX}`);
}

/**
 * Keep a copy of the compact snapshot on disk. POST /v1/memories/raw only
 * enqueues the write (202), so the SessionStart that follows the compaction
 * can run before the memory is searchable; this copy is read first.
 */
export async function writeLocalCompactSnapshot(snapshot: LocalCompactSnapshot): Promise<void> {
  try {
    await mkdir(getObservationDir(), { recursive: true });
    await writeFile(getCompactSnapshotPath(snapshot.sessionId), JSON.stringify(snapshot), { encoding: "utf-8", mode: 0o600 });
  } catch (error) {
    logError("writeLocalCompactSnapshot failed (non-critical)", error);
  }
}

/**
 * Session ids that came before this one, read from the head of a resumed or forked transcript.
 * Claude Code starts a resumed/forked session under a NEW id, and the new transcript begins with
 * the earlier session's entries, which carry their own `sessionId`. Returns the distinct ids that
 * differ from `currentSessionId`, in order of appearance. Reads at most the first 8 MiB; only the
 * `sessionId` key of each line is looked at. Empty when nothing is found or the file is unreadable.
 */
export async function readPredecessorSessionIds(
  transcriptPath: string | undefined,
  currentSessionId: string | undefined,
): Promise<string[]> {
  if (!transcriptPath) return [];
  const MAX_BYTES = 8 * 1024 * 1024;
  let handle;
  try {
    handle = await open(transcriptPath, "r");
    const buf = Buffer.alloc(MAX_BYTES);
    const { bytesRead } = await handle.read(buf, 0, MAX_BYTES, 0);
    const text = buf.subarray(0, bytesRead).toString("utf-8");
    const ids: string[] = [];
    for (const line of text.split("\n")) {
      if (!line.includes('"sessionId"')) continue;
      try {
        const id = JSON.parse(line)?.sessionId;
        if (typeof id === "string" && id && id !== currentSessionId && !ids.includes(id)) ids.push(id);
      } catch {
        // a partial last line or a non-JSON line
      }
    }
    return ids;
  } catch {
    return [];
  } finally {
    await handle?.close();
  }
}

/** Read a session's local compact snapshot; null when absent or unreadable. */
export async function readLocalCompactSnapshot(sessionId: string): Promise<LocalCompactSnapshot | null> {
  try {
    const parsed = JSON.parse(await readFile(getCompactSnapshotPath(sessionId), "utf-8"));
    if (typeof parsed?.content !== "string" || typeof parsed?.createdAt !== "string") return null;
    return {
      sessionId: String(parsed.sessionId ?? sessionId),
      projectId: typeof parsed.projectId === "string" ? parsed.projectId : undefined,
      content: parsed.content,
      createdAt: parsed.createdAt,
    };
  } catch {
    return null;
  }
}

// ── Fail-loud transport-failure counter (roadmap D4) ─────────────
//
// A tiny state file counting CONSECUTIVE Neuralscape-unreachable events
// across hooks. Any hook that hits a transport failure increments it; any
// successful call resets it. When the count reaches the threshold, the
// SessionStart notice upgrades from the generic one-liner to a fail-loud
// "unreachable for N events — check `docker compose ps`" message. All
// still exit 0 — the counter changes the message, never the contract.

export const DEFAULT_FAIL_LOUD_THRESHOLD = 3;

export function getFailLoudThreshold(): number {
  const parsed = parseInt(readConfig("FAIL_LOUD_THRESHOLD", ""), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_FAIL_LOUD_THRESHOLD;
}

/** Counter lives next to the observation buffers (same data dir). */
export function getFailureCounterPath(): string {
  return pathJoin(getObservationDir(), "unreachable.count");
}

export async function getTransportFailureCount(): Promise<number> {
  try {
    const raw = await readFile(getFailureCounterPath(), "utf-8");
    const parsed = parseInt(raw.trim(), 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
  } catch {
    return 0;
  }
}

/** Increment the consecutive-failure counter; returns the new count. */
export async function recordTransportFailure(): Promise<number> {
  try {
    const next = (await getTransportFailureCount()) + 1;
    await mkdir(getObservationDir(), { recursive: true });
    await writeFile(getFailureCounterPath(), String(next), "utf-8");
    return next;
  } catch (error) {
    logError("recordTransportFailure failed", error);
    return 1;
  }
}

/** Any successful NS call resets the streak. */
export async function resetTransportFailures(): Promise<void> {
  try {
    await unlink(getFailureCounterPath());
  } catch {
    // no counter — already clean
  }
}

// ── Read-gate session state (roadmap D3) ─────────────────────────
//
// Session-scoped dedup: the gate fires at most once per file per session.
// The state file records every path already checked (denied or not), so a
// second Read of the same path — the user/agent override — always passes,
// and repeated large-file reads don't pay the search round-trip again.

function safeSessionName(sessionId: string): string {
  return sessionId.replace(/[^A-Za-z0-9._-]/g, "_") || "unknown";
}

export function getReadGateStatePath(sessionId: string): string {
  return pathJoin(getObservationDir(), `${safeSessionName(sessionId)}.readgate.json`);
}

export async function loadGatedFiles(sessionId: string): Promise<Set<string>> {
  try {
    const raw = await readFile(getReadGateStatePath(sessionId), "utf-8");
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return new Set(parsed.filter((p): p is string => typeof p === "string"));
    }
    return new Set();
  } catch {
    return new Set();
  }
}

export async function recordGatedFile(sessionId: string, filePath: string): Promise<void> {
  try {
    const gated = await loadGatedFiles(sessionId);
    gated.add(filePath);
    await mkdir(getObservationDir(), { recursive: true });
    await writeFile(getReadGateStatePath(sessionId), JSON.stringify([...gated]), "utf-8");
  } catch (error) {
    logError("recordGatedFile failed", error);
  }
}

// ── Read-gate session index cache (audit 27 #31) ─────────────────
//
// The candidate-memory fetch happens AT MOST ONCE per session AND project:
// the first gateable Read fetches the index-level rows and caches them
// here; every later Read (any file) in the same project matches against
// the cache with zero network I/O. The key includes the resolved project
// id (Copilot, PR #126): a session whose cwd moves to a different repo
// must NOT be steered by the previous project's rows — the switch triggers
// exactly one fresh project-scoped fetch. A failed/timed-out fetch caches
// a failure sentinel — the gate then stays quiet for that project for the
// rest of the session instead of re-paying the round trip on every
// large-file Read while the service is down.

export interface ReadGateIndexCache {
  /** False when the one fetch failed/timed out (gate stays quiet). */
  ok: boolean;
  fetchedAt: string;
  rows: NeuralscapeMemory[];
}

export function getReadGateCachePath(sessionId: string, projectId?: string): string {
  const project = safeSessionName(projectId || "global");
  return pathJoin(
    getObservationDir(),
    `${safeSessionName(sessionId)}.${project}.readgate-index.json`,
  );
}

export async function loadReadGateIndexCache(
  sessionId: string,
  projectId?: string,
): Promise<ReadGateIndexCache | null> {
  try {
    const raw = await readFile(getReadGateCachePath(sessionId, projectId), "utf-8");
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && typeof parsed.ok === "boolean") {
      return {
        ok: parsed.ok,
        fetchedAt: typeof parsed.fetchedAt === "string" ? parsed.fetchedAt : "",
        rows: Array.isArray(parsed.rows) ? (parsed.rows as NeuralscapeMemory[]) : [],
      };
    }
    return null;
  } catch {
    return null;
  }
}

export async function saveReadGateIndexCache(
  sessionId: string,
  projectId: string | undefined,
  ok: boolean,
  rows: NeuralscapeMemory[],
): Promise<void> {
  try {
    await mkdir(getObservationDir(), { recursive: true });
    const payload: ReadGateIndexCache = { ok, fetchedAt: new Date().toISOString(), rows };
    await writeFile(getReadGateCachePath(sessionId, projectId), JSON.stringify(payload), "utf-8");
  } catch (error) {
    logError("saveReadGateIndexCache failed", error);
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
