/**
 * Subprocess tests for the built pre-tool-use bundle — the gate decision
 * matrix end-to-end after audit 27 #31/#32: STEERING output shape
 * (additionalContext, never a deny), exit codes per the never-block
 * taxonomy, once-per-file steering, the once-per-SESSION index fetch
 * (cache file), the index-level fetch parameters, and the hard time
 * budget. A local HTTP fixture server stands in for Neuralscape so nothing
 * here depends on the live stack.
 *
 * `npm test` builds first (pretest), so scripts/pre-tool-use.js exists.
 */

import { spawn } from "node:child_process";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";

const PLUGIN_ROOT = join(__dirname, "..");
const SCRIPT = join(PLUGIN_ROOT, "scripts", "pre-tool-use.js");

// A port that refuses connections immediately.
const DEAD_URL = "http://127.0.0.1:9";

interface RunResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

function runGate(stdin: string, env: Record<string, string> = {}): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    const child = spawn("node", [SCRIPT], {
      env: {
        PATH: process.env.PATH ?? "",
        NEURALSCAPE_USER_ID: "read-gate-test",
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

/** Exit 0 and NO permission decision — the Read always proceeds. */
function expectNeverBlocks(r: RunResult): void {
  expect(r.code).toBe(0);
  if (r.stdout.trim()) {
    const out = JSON.parse(r.stdout);
    expect(out.hookSpecificOutput?.permissionDecision).toBeUndefined();
    expect(out.hookSpecificOutput?.permissionDecisionReason).toBeUndefined();
  }
}

/** A pass-through with no steering context at all. */
function expectQuietAllow(r: RunResult): void {
  expectNeverBlocks(r);
  if (r.stdout.trim()) {
    expect(JSON.parse(r.stdout).hookSpecificOutput?.additionalContext).toBeUndefined();
  }
}

function steerContext(r: RunResult): string {
  const out = JSON.parse(r.stdout);
  const ctx = out.hookSpecificOutput?.additionalContext;
  expect(typeof ctx).toBe("string");
  return ctx as string;
}

// ── Fixture server: fake GET /v1/memories ────────────────────────
//
// The gate fetches the recency-bounded index-level memory list ONCE per
// session (fields=index; not the hybrid /v1/search, whose Graphiti pass
// blows the PreToolUse latency budget).

let server: Server;
let serverUrl: string;
let fetchCalls = 0;
let fetchUrls: string[] = [];
let mode: "ok" | "http500" | "slow" = "ok";

const FIXTURE_MEMORIES = [
  {
    id: "mem-modified",
    memory: "",
    observation_type: "bugfix",
    title: "Fixed flaky debounce in src/gate-fixture-module.ts",
    token_estimate: 90,
    created_at: "2026-07-02T10:00:00Z",
  },
  {
    id: "mem-read",
    memory: "",
    title: "src/gate-fixture-module.ts reads the shared config singleton",
    observation_type: "discovery",
    token_estimate: 60,
    created_at: "2026-07-03T10:00:00Z",
  },
  {
    id: "mem-other",
    memory: "",
    title: "src/other-fixture-module.ts owns the retry queue",
    observation_type: "discovery",
    token_estimate: 45,
    created_at: "2026-07-03T11:00:00Z",
  },
  {
    id: "mem-unrelated",
    memory: "",
    title: "The deploy pipeline flips blue/green on Sundays",
    created_at: "2026-07-01T10:00:00Z",
  },
];

beforeAll(async () => {
  server = createServer((req, res) => {
    if (req.method === "GET" && req.url?.startsWith("/v1/memories")) {
      fetchCalls++;
      fetchUrls.push(req.url);
      if (mode === "http500") {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ detail: "boom" }));
        return;
      }
      const reply = () => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(FIXTURE_MEMORIES));
      };
      if (mode === "slow") {
        setTimeout(reply, 1500);
        return;
      }
      reply();
      return;
    }
    res.writeHead(404).end();
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (typeof address === "object" && address) serverUrl = `http://127.0.0.1:${address.port}`;
});

afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

beforeEach(() => {
  mode = "ok";
  fetchUrls = [];
});

// ── Fixture files ────────────────────────────────────────────────

async function makeEnv(): Promise<{
  dataDir: string;
  largeFile: string;
  otherLargeFile: string;
  unknownLargeFile: string;
  smallFile: string;
  largePng: string;
}> {
  const dataDir = await mkdtemp(join(tmpdir(), "ns-gate-e2e-"));
  const srcDir = join(dataDir, "src");
  await mkdir(srcDir, { recursive: true });
  const largeFile = join(srcDir, "gate-fixture-module.ts");
  await writeFile(largeFile, `// fixture\n${"x".repeat(2500)}\n`, "utf-8");
  const otherLargeFile = join(srcDir, "other-fixture-module.ts");
  await writeFile(otherLargeFile, `// fixture\n${"y".repeat(2500)}\n`, "utf-8");
  const unknownLargeFile = join(srcDir, "nobody-remembers-me.ts");
  await writeFile(unknownLargeFile, `// fixture\n${"z".repeat(2500)}\n`, "utf-8");
  const smallFile = join(srcDir, "tiny.ts");
  await writeFile(smallFile, "export const x = 1;\n", "utf-8");
  const largePng = join(srcDir, "big-image.png");
  await writeFile(largePng, Buffer.alloc(5000, 7));
  return { dataDir, largeFile, otherLargeFile, unknownLargeFile, smallFile, largePng };
}

function readEvent(filePath: string, sessionId: string, cwd: string): string {
  return JSON.stringify({
    session_id: sessionId,
    cwd,
    hook_event_name: "PreToolUse",
    tool_name: "Read",
    tool_input: { file_path: filePath },
  });
}

// ── The decision matrix ──────────────────────────────────────────

describe("read gate subprocess matrix (steer, never block)", () => {
  it("large file + memories → additionalContext with ranked rows, Read proceeds", async () => {
    const { dataDir, largeFile } = await makeEnv();
    const r = await runGate(readEvent(largeFile, "s-steer", dataDir), {
      NEURALSCAPE_URL: serverUrl,
      CLAUDE_PLUGIN_DATA: dataDir,
    });
    expectNeverBlocks(r);
    const out = JSON.parse(r.stdout);
    expect(out.hookSpecificOutput.hookEventName).toBe("PreToolUse");
    const ctx = steerContext(r);
    expect(ctx).toContain("[Neuralscape]");
    expect(ctx).toContain("`#id | when | title | ~tokens`");
    // ranked: the memory that MODIFIED the file precedes the read-only one;
    // unrelated + other-file rows are filtered by the tail-match pass
    expect(ctx.indexOf("#mem-modified")).toBeGreaterThanOrEqual(0);
    expect(ctx.indexOf("#mem-modified")).toBeLessThan(ctx.indexOf("#mem-read"));
    expect(ctx).not.toContain("mem-unrelated");
    expect(ctx).not.toContain("mem-other");
    // escalation menu, but never deny/override framing
    expect(ctx).toContain("get_memories");
    expect(ctx).toContain("timeline");
    expect(ctx).not.toContain("Skipped reading");
  }, 15000);

  it("fetches index-level fields with a sane row cap (audit 27 #31)", async () => {
    const { dataDir, largeFile } = await makeEnv();
    await runGate(readEvent(largeFile, "s-params", dataDir), {
      NEURALSCAPE_URL: serverUrl,
      CLAUDE_PLUGIN_DATA: dataDir,
    });
    const url = new URL(fetchUrls[fetchUrls.length - 1], serverUrl);
    expect(url.searchParams.get("fields")).toBe("index");
    const limit = parseInt(url.searchParams.get("limit") ?? "0", 10);
    expect(limit).toBeGreaterThan(0);
    expect(limit).toBeLessThanOrEqual(200);
  }, 15000);

  it("steers once per file per session: the second Read is quiet", async () => {
    const { dataDir, largeFile } = await makeEnv();
    const env = { NEURALSCAPE_URL: serverUrl, CLAUDE_PLUGIN_DATA: dataDir };
    const first = await runGate(readEvent(largeFile, "s-dedup", dataDir), env);
    expect(steerContext(first)).toContain("#mem-modified");
    const callsAfterFirst = fetchCalls;
    const second = await runGate(readEvent(largeFile, "s-dedup", dataDir), env);
    expectQuietAllow(second);
    expect(fetchCalls).toBe(callsAfterFirst);
  }, 20000);

  it("fetches at most ONCE per session: a different file reuses the cache", async () => {
    const { dataDir, largeFile, otherLargeFile, unknownLargeFile } = await makeEnv();
    const env = { NEURALSCAPE_URL: serverUrl, CLAUDE_PLUGIN_DATA: dataDir };
    await runGate(readEvent(largeFile, "s-cache", dataDir), env);
    const callsAfterFirst = fetchCalls;
    // Second file: steered from the CACHED index — zero additional fetches.
    const other = await runGate(readEvent(otherLargeFile, "s-cache", dataDir), env);
    expect(steerContext(other)).toContain("#mem-other");
    expect(fetchCalls).toBe(callsAfterFirst);
    // Third file with no matching memories: quiet, still no fetch.
    const unknown = await runGate(readEvent(unknownLargeFile, "s-cache", dataDir), env);
    expectQuietAllow(unknown);
    expect(fetchCalls).toBe(callsAfterFirst);
  }, 30000);

  it("a project switch within one session triggers a fresh project-scoped fetch (Copilot, PR #126)", async () => {
    const { dataDir } = await makeEnv(); // observation/cache dir only
    const projA = await mkdtemp(join(tmpdir(), "ns-proj-a-"));
    const projB = await mkdtemp(join(tmpdir(), "ns-proj-b-"));
    await mkdir(join(projA, "src"), { recursive: true });
    await mkdir(join(projB, "src"), { recursive: true });
    const fileA = join(projA, "src", "gate-fixture-module.ts");
    const fileA2 = join(projA, "src", "other-fixture-module.ts");
    const fileB = join(projB, "src", "gate-fixture-module.ts");
    for (const f of [fileA, fileA2, fileB]) {
      await writeFile(f, `// fixture\n${"x".repeat(2500)}\n`, "utf-8");
    }
    const projectA = projA.split("/").filter(Boolean).pop()!;
    const projectB = projB.split("/").filter(Boolean).pop()!;
    const env = { NEURALSCAPE_URL: serverUrl, CLAUDE_PLUGIN_DATA: dataDir };

    await runGate(readEvent(fileA, "s-projswitch", projA), env);
    const callsAfterA = fetchCalls;
    expect(new URL(fetchUrls[fetchUrls.length - 1], serverUrl).searchParams.get("project_id")).toBe(projectA);

    // Same session, cwd now resolves a DIFFERENT project: the cached rows
    // belong to project A and must NOT be served — a fresh project-B-scoped
    // fetch happens instead.
    await runGate(readEvent(fileB, "s-projswitch", projB), env);
    expect(fetchCalls).toBe(callsAfterA + 1);
    expect(new URL(fetchUrls[fetchUrls.length - 1], serverUrl).searchParams.get("project_id")).toBe(projectB);

    // Back in project A: its cache is still valid — no third fetch.
    await runGate(readEvent(fileA2, "s-projswitch", projA), env);
    expect(fetchCalls).toBe(callsAfterA + 1);
  }, 30000);

  it("still steers the same file in a DIFFERENT session (fresh fetch)", async () => {
    const { dataDir, largeFile } = await makeEnv();
    const env = { NEURALSCAPE_URL: serverUrl, CLAUDE_PLUGIN_DATA: dataDir };
    await runGate(readEvent(largeFile, "s-one", dataDir), env);
    const callsAfterFirst = fetchCalls;
    const other = await runGate(readEvent(largeFile, "s-two", dataDir), env);
    expect(steerContext(other)).toContain("#mem-modified");
    expect(fetchCalls).toBe(callsAfterFirst + 1);
  }, 20000);

  it("small file → quiet allow without calling the API", async () => {
    const { dataDir, smallFile } = await makeEnv();
    const before = fetchCalls;
    const r = await runGate(readEvent(smallFile, "s-small", dataDir), {
      NEURALSCAPE_URL: serverUrl,
      CLAUDE_PLUGIN_DATA: dataDir,
    });
    expectQuietAllow(r);
    expect(fetchCalls).toBe(before);
  }, 15000);

  it("binary/media extension → quiet allow without calling the API", async () => {
    const { dataDir, largePng } = await makeEnv();
    const before = fetchCalls;
    const r = await runGate(readEvent(largePng, "s-png", dataDir), {
      NEURALSCAPE_URL: serverUrl,
      CLAUDE_PLUGIN_DATA: dataDir,
    });
    expectQuietAllow(r);
    expect(fetchCalls).toBe(before);
  }, 15000);

  it("NS unreachable → allow, exit 0, failure counted (never-block)", async () => {
    const { dataDir, largeFile } = await makeEnv();
    const r = await runGate(readEvent(largeFile, "s-down", dataDir), {
      NEURALSCAPE_URL: DEAD_URL,
      CLAUDE_PLUGIN_DATA: dataDir,
    });
    expectQuietAllow(r);
    expect(r.stderr).toContain("allowing read");
  }, 15000);

  it("HTTP failure is cached: the gate stays quiet for the session without refetching", async () => {
    mode = "http500";
    const { dataDir, largeFile, otherLargeFile } = await makeEnv();
    const env = { NEURALSCAPE_URL: serverUrl, CLAUDE_PLUGIN_DATA: dataDir };
    const first = await runGate(readEvent(largeFile, "s-500", dataDir), env);
    expectQuietAllow(first);
    expect(first.stderr).toContain("allowing read");
    const callsAfterFirst = fetchCalls;
    mode = "ok"; // even with the service healthy again, this session stays quiet
    const second = await runGate(readEvent(otherLargeFile, "s-500", dataDir), env);
    expectQuietAllow(second);
    expect(fetchCalls).toBe(callsAfterFirst);
  }, 20000);

  it("a slow NS blows the time budget → allow, exit 0, no steering (audit 27 #31)", async () => {
    mode = "slow";
    const { dataDir, largeFile } = await makeEnv();
    const started = Date.now();
    const r = await runGate(readEvent(largeFile, "s-slow", dataDir), {
      NEURALSCAPE_URL: serverUrl,
      CLAUDE_PLUGIN_DATA: dataDir,
      NEURALSCAPE_READ_GATE_TIME_BUDGET_MS: "200",
    });
    expectQuietAllow(r);
    expect(r.stderr).toContain("budget");
    expect(Date.now() - started).toBeLessThan(10000);
  }, 15000);

  it("excluded project → quiet allow without calling the API", async () => {
    const { dataDir, largeFile } = await makeEnv();
    const projectId = dataDir.split("/").filter(Boolean).pop()!;
    const before = fetchCalls;
    const r = await runGate(readEvent(largeFile, "s-excluded", dataDir), {
      NEURALSCAPE_URL: serverUrl,
      CLAUDE_PLUGIN_DATA: dataDir,
      NEURALSCAPE_EXCLUDED_PROJECTS: projectId,
    });
    expectQuietAllow(r);
    expect(fetchCalls).toBe(before);
  }, 15000);

  it("READ_GATE_ENABLED=false → quiet allow without calling the API", async () => {
    const { dataDir, largeFile } = await makeEnv();
    const before = fetchCalls;
    const r = await runGate(readEvent(largeFile, "s-disabled", dataDir), {
      NEURALSCAPE_URL: serverUrl,
      CLAUDE_PLUGIN_DATA: dataDir,
      NEURALSCAPE_READ_GATE_ENABLED: "false",
    });
    expectQuietAllow(r);
    expect(fetchCalls).toBe(before);
  }, 15000);

  it("non-Read tool events and missing files are ignored", async () => {
    const { dataDir, largeFile } = await makeEnv();
    const env = { NEURALSCAPE_URL: serverUrl, CLAUDE_PLUGIN_DATA: dataDir };
    const before = fetchCalls;
    const edit = await runGate(
      JSON.stringify({ session_id: "s-x", cwd: dataDir, tool_name: "Edit", tool_input: { file_path: largeFile } }),
      env,
    );
    expectQuietAllow(edit);
    const missing = await runGate(readEvent(join(dataDir, "no-such-file.ts"), "s-x", dataDir), env);
    expectQuietAllow(missing);
    expect(fetchCalls).toBe(before);
  }, 20000);

  it("malformed stdin → allow + exit 0 (exit 2 would BLOCK on PreToolUse)", async () => {
    const { dataDir } = await makeEnv();
    const r = await runGate("this is not json", {
      NEURALSCAPE_URL: serverUrl,
      CLAUDE_PLUGIN_DATA: dataDir,
    });
    expectQuietAllow(r);
  }, 15000);
});
