/**
 * Skill helper CLI — the plugin's own code path for the skills that must make
 * REST calls (sync, ingest file upload, search fallback), so they resolve the
 * service URL, send the Bearer token and redact `<private>` text exactly the
 * way the hooks do instead of hand-writing curl.
 *
 *   node sync.js print-config
 *       Prints {url, url_source, api_key_set, user_id, user_id_source, project_id}.
 *       Never prints the API key.
 *   node sync.js flush            (JSON on stdin)
 *       {"turns":[{"user_message","assistant_response","session_id"?,"timestamp"?}],
 *        "compile"?: true, "date"?: "YYYY-MM-DD", "project_id"?: "..."}
 *       Same noise filter, fail-closed `<private>` redaction and Bearer auth as
 *       the Stop hook, then (optionally) the compile trigger.
 *   node sync.js request <GET|POST> </v1/...>   (JSON body on stdin for POST)
 *       An authenticated call to the configured service; prints the JSON answer.
 *   node sync.js upload [--field k=v]... <file>...
 *       Multipart POST /v1/ingest/files with the configured URL and Bearer.
 *
 * Exit codes: 0 ok, 1 the service refused/was unreachable or a flush failed, 2 bad usage/input.
 */

import { readFile } from "node:fs/promises";
import { basename } from "node:path";

import { triggerCompile } from "../core/compile.js";
import { flushTurns } from "../core/flush.js";
import type { ConversationTurn } from "../core/types.js";
import {
  classifyFailure,
  getProjectId,
  getServiceUrl,
  getUserId,
  getUserIdSource,
  hasApiKey,
  neuralscapeGet,
  neuralscapePost,
  readBakedUrl,
  readConfig,
} from "../utils.js";

class UsageError extends Error {}

function print(value: unknown): void {
  process.stdout.write(`${JSON.stringify(value)}\n`);
}

async function readStdin(): Promise<string> {
  let data = "";
  process.stdin.setEncoding("utf-8");
  for await (const chunk of process.stdin) data += chunk as string;
  return data;
}

async function readJsonStdin(): Promise<Record<string, unknown>> {
  const text = await readStdin();
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new UsageError("stdin is not valid JSON");
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new UsageError("stdin must be a JSON object");
  }
  return parsed as Record<string, unknown>;
}

export function printConfig(): void {
  const urlSource = readConfig("URL", "") ? "config" : readBakedUrl() ? "baked" : "default";
  print({
    url: getServiceUrl(),
    url_source: urlSource,
    api_key_set: hasApiKey(),
    user_id: getUserId(),
    user_id_source: getUserIdSource(),
    project_id: getProjectId() ?? null,
  });
}

async function flush(): Promise<number> {
  const input = await readJsonStdin();
  const rawTurns = input.turns;
  if (!Array.isArray(rawTurns)) throw new UsageError('"turns" must be an array');
  const userId = getUserId();
  if (!userId) throw new UsageError("no user id configured (set the plugin USER_ID option)");
  const projectId = typeof input.project_id === "string" && input.project_id ? input.project_id : getProjectId();

  const turns: ConversationTurn[] = rawTurns.map((t) => {
    const turn = (t ?? {}) as Record<string, unknown>;
    return {
      userMessage: typeof turn.user_message === "string" ? turn.user_message : "",
      assistantResponse: typeof turn.assistant_response === "string" ? turn.assistant_response : "",
      sessionId: typeof turn.session_id === "string" && turn.session_id ? turn.session_id : "manual-sync",
      channel: "claude-code-manual-sync",
      timestamp: typeof turn.timestamp === "string" && turn.timestamp ? turn.timestamp : new Date().toISOString(),
      projectId,
      userId,
    };
  });

  const result = await flushTurns(turns);
  const failed = result.firstFailedIndex !== null;
  let compiled: boolean | null = null;
  if (input.compile === true && !failed) {
    const date = typeof input.date === "string" && input.date ? input.date : new Date().toISOString().slice(0, 10);
    compiled = await triggerCompile({ date, userId, shouldCompile: true });
  }
  print({ attempted: result.attempted, flushed: result.flushed, failed, first_failed_index: result.firstFailedIndex, compiled });
  return failed || compiled === false ? 1 : 0;
}

async function request(method: string, path: string): Promise<number> {
  if (method !== "GET" && method !== "POST") throw new UsageError("method must be GET or POST");
  if (!path.startsWith("/v1/")) throw new UsageError("path must start with /v1/");
  const answer = method === "GET" ? await neuralscapeGet(path) : await neuralscapePost(path, await readJsonStdin());
  print(answer);
  return 0;
}

const UPLOAD_TIMEOUT_MS = 120_000;

async function upload(args: string[]): Promise<number> {
  const form = new FormData();
  const files: string[] = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--field") {
      const pair = args[++i] ?? "";
      const eq = pair.indexOf("=");
      if (eq < 1) throw new UsageError("--field expects key=value");
      form.append(pair.slice(0, eq), pair.slice(eq + 1));
    } else files.push(args[i]);
  }
  if (files.length === 0) throw new UsageError("upload needs at least one file");
  for (const file of files) form.append("files", new Blob([new Uint8Array(await readFile(file))]), basename(file));

  const key = readConfig("API_KEY", "");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPLOAD_TIMEOUT_MS);
  try {
    const response = await fetch(`${getServiceUrl()}/v1/ingest/files`, {
      method: "POST",
      headers: key ? { Authorization: `Bearer ${key}` } : {},
      body: form,
      signal: controller.signal,
    });
    const text = await response.text();
    if (!response.ok) {
      process.stderr.write(`[neuralscape] upload refused: HTTP ${response.status} ${response.statusText}\n`);
      return 1;
    }
    process.stdout.write(`${text}\n`);
    return 0;
  } finally {
    clearTimeout(timer);
  }
}

async function main(): Promise<number> {
  const [mode, ...rest] = process.argv.slice(2);
  switch (mode) {
    case "print-config":
      printConfig();
      return 0;
    case "flush":
      return flush();
    case "request":
      return request((rest[0] ?? "").toUpperCase(), rest[1] ?? "");
    case "upload":
      return upload(rest);
    default:
      throw new UsageError("usage: sync.js print-config | flush | request <GET|POST> </v1/...> | upload [--field k=v]... <file>...");
  }
}

main().then(
  (code) => process.exit(code),
  (error) => {
    if (error instanceof UsageError) {
      process.stderr.write(`[neuralscape] ${error.message}\n`);
      process.exit(2);
    }
    const kind = classifyFailure(error);
    process.stderr.write(`[neuralscape] ${kind}: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exit(1);
  },
);
