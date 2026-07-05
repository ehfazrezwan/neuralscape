/**
 * Compact-resilience loop, piece 3 — registration + the built PreCompact
 * bundle end-to-end.
 *
 * Registration: hooks/hooks.json must wire PreCompact at scripts/
 * pre-compact.js and esbuild must emit that bundle. Behavior (subprocess,
 * against a local HTTP fixture standing in for Neuralscape): the hook
 * flushes undelivered transcript turns to the conversation-compiler,
 * commits the offset past delivered turns, POSTs exactly ONE tagged
 * compact-snapshot to /v1/memories/raw — and NEVER exits non-zero or
 * blocks the compact, even with the service dead or stdin malformed.
 *
 * `npm test` builds first (pretest), so scripts/pre-compact.js exists.
 */

import { spawn } from "node:child_process";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeEach, describe, expect, it } from "vitest";

const PLUGIN_ROOT = join(__dirname, "..");
const SCRIPT = join(PLUGIN_ROOT, "scripts", "pre-compact.js");

// A port that refuses connections immediately.
const DEAD_URL = "http://127.0.0.1:9";

// ── Registration ─────────────────────────────────────────────────

describe("PreCompact registration", () => {
  it("hooks.json wires PreCompact to the pre-compact bundle", async () => {
    const manifest = JSON.parse(
      await readFile(join(PLUGIN_ROOT, "hooks", "hooks.json"), "utf-8"),
    );
    const entries = manifest.hooks?.PreCompact;
    expect(Array.isArray(entries)).toBe(true);
    const command = entries[0]?.hooks?.[0];
    expect(command?.type).toBe("command");
    expect(command?.command).toContain("scripts/pre-compact.js");
    expect(command?.command).toContain("${CLAUDE_PLUGIN_ROOT}");
    // Fire-and-forget like the other capture hooks — never blocks the compact.
    expect(command?.async).toBe(true);
    expect(typeof command?.timeout).toBe("number");
  });

  it("esbuild.config.js lists the pre-compact entrypoint", async () => {
    const config = await readFile(join(PLUGIN_ROOT, "esbuild.config.js"), "utf-8");
    expect(config).toContain("src/hooks/pre-compact.ts");
  });
});

// ── Fixture server ───────────────────────────────────────────────

let server: Server;
let serverUrl: string;
let requests: Array<{ path: string; body: Record<string, unknown> }> = [];

await new Promise<void>((resolve) => {
  server = createServer((req, res) => {
    let body = "";
    req.on("data", (d) => (body += d));
    req.on("end", () => {
      requests.push({ path: req.url ?? "", body: body ? JSON.parse(body) : {} });
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok" }));
    });
  });
  server.listen(0, "127.0.0.1", () => {
    const address = server.address();
    serverUrl = `http://127.0.0.1:${typeof address === "object" && address ? address.port : 0}`;
    resolve();
  });
});

afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

beforeEach(() => {
  requests = [];
});

// ── Harness ──────────────────────────────────────────────────────

interface RunResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

async function runHook(stdin: string, env: Record<string, string> = {}): Promise<RunResult> {
  const dataDir = await mkdtemp(join(tmpdir(), "ns-precompact-data-"));
  return new Promise((resolve, reject) => {
    const child = spawn("node", [SCRIPT], {
      env: {
        PATH: process.env.PATH ?? "",
        NEURALSCAPE_USER_ID: "pre-compact-test",
        NEURALSCAPE_URL: serverUrl,
        CLAUDE_PLUGIN_DATA: dataDir,
        ...env,
      },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    child.on("error", reject);
    child.on("close", (code) => resolve({ code, stdout, stderr }));
    child.stdin.write(stdin);
    child.stdin.end();
  });
}

const MESSAGES = [
  { type: "user", content: "first user prompt with plenty of substance here" },
  { type: "assistant", content: "first assistant answer with plenty of substance here" },
  { type: "user", content: "second user prompt about the compact loop work" },
  { type: "assistant", content: "second assistant answer with plenty of substance here" },
];

async function makeTranscript(): Promise<{ path: string; dir: string }> {
  const dir = await mkdtemp(join(tmpdir(), "ns-precompact-"));
  const path = join(dir, "transcript.jsonl");
  await writeFile(path, MESSAGES.map((m) => JSON.stringify(m)).join("\n") + "\n", "utf-8");
  return { path, dir };
}

function stdinFor(transcriptPath: string, cwd: string, trigger = "auto"): string {
  return JSON.stringify({
    session_id: "sess-precompact",
    transcript_path: transcriptPath,
    cwd,
    hook_event_name: "PreCompact",
    trigger,
    custom_instructions: "",
  });
}

/** Exit 0 + continue:true — the compact must never be blocked. */
function expectNeverBlocks(r: RunResult): void {
  expect(r.code).toBe(0);
  const out = JSON.parse(r.stdout);
  expect(out.continue).toBe(true);
}

// ── Behavior ─────────────────────────────────────────────────────

describe("pre-compact bundle end-to-end", () => {
  it("flushes turns, commits the offset, and stores one tagged snapshot", async () => {
    const { path, dir } = await makeTranscript();
    const r = await runHook(stdinFor(path, dir, "manual"));
    expectNeverBlocks(r);

    const flushes = requests.filter((q) => q.path.includes("/conversation-compiler/flush"));
    const raws = requests.filter((q) => q.path.includes("/v1/memories/raw"));
    expect(flushes.length).toBe(2);
    expect(raws.length).toBe(1);

    const snapshot = raws[0].body;
    expect(snapshot.category).toBe("task_context");
    expect(snapshot.tags).toEqual(["compact_snapshot"]);
    expect(snapshot.user_id).toBe("pre-compact-test");
    expect(String(snapshot.content)).toContain("Compact snapshot: session sess-precompact");
    expect(String(snapshot.content)).toContain("(manual)");
    expect(String(snapshot.content)).toContain("second user prompt about the compact loop work");

    // Offset committed past both delivered turns (audit 27 #34b path).
    const offset = await readFile(path + ".neuralscape-offset", "utf-8");
    expect(parseInt(offset, 10)).toBe(4);
  });

  it("still snapshots when the transcript has no new turns", async () => {
    const { path, dir } = await makeTranscript();
    await writeFile(path + ".neuralscape-offset", "4", "utf-8"); // all delivered
    const r = await runHook(stdinFor(path, dir));
    expectNeverBlocks(r);

    const flushes = requests.filter((q) => q.path.includes("/conversation-compiler/flush"));
    const raws = requests.filter((q) => q.path.includes("/v1/memories/raw"));
    expect(flushes.length).toBe(0);
    expect(raws.length).toBe(1);
    expect(String(raws[0].body.content)).toContain("after 0 captured turn(s)");
  });

  it("exits 0 with the service dead — the compact is never failed", async () => {
    const { path, dir } = await makeTranscript();
    const r = await runHook(stdinFor(path, dir), { NEURALSCAPE_URL: DEAD_URL });
    expectNeverBlocks(r);
  });

  it("exits 0 on malformed stdin (fire-and-forget contract)", async () => {
    const r = await runHook("this is not json");
    expectNeverBlocks(r);
  });

  it("captures nothing from an excluded project", async () => {
    const { path, dir } = await makeTranscript();
    const r = await runHook(stdinFor(path, dir), {
      NEURALSCAPE_EXCLUDED_PROJECTS: "ns-precompact-*",
    });
    expectNeverBlocks(r);
    expect(requests.length).toBe(0);
  });
});
