/**
 * Subprocess tests for the built pre-tool-use bundle (roadmap D3) — the gate
 * decision matrix end-to-end: deny JSON shape per the hooks API, exit codes
 * per the never-block taxonomy, and once-per-file-per-session dedup across
 * real invocations. A local HTTP fixture server stands in for Neuralscape so
 * nothing here depends on the live stack.
 *
 * `npm test` builds first (pretest), so scripts/pre-tool-use.js exists.
 */

import { spawn } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

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

function expectAllow(r: RunResult): void {
  expect(r.code).toBe(0);
  if (r.stdout.trim()) {
    const out = JSON.parse(r.stdout);
    expect(out.hookSpecificOutput?.permissionDecision).toBeUndefined();
  }
}

// ── Fixture server: fake GET /v1/memories ────────────────────────
//
// The gate fetches the recency-bounded memory LIST (not the hybrid
// /v1/search, whose Graphiti pass blows the PreToolUse latency budget).

let server: Server;
let serverUrl: string;
let searchCalls = 0;

const FIXTURE_BASENAME = "gate-fixture-module.ts";

const FIXTURE_MEMORIES = [
  {
    id: "mem-modified",
    memory: `Fixed the flaky debounce in src/${FIXTURE_BASENAME} by pinning the timer.`,
    observation_type: "bugfix",
    title: `Fixed flaky debounce in ${FIXTURE_BASENAME}`,
    token_estimate: 90,
    created_at: "2026-07-02T10:00:00Z",
  },
  {
    id: "mem-read",
    memory: `${FIXTURE_BASENAME} plus helper-a.ts and helper-b.ts all read the shared config singleton.`,
    observation_type: "discovery",
    created_at: "2026-07-03T10:00:00Z",
  },
  {
    id: "mem-unrelated",
    memory: "The deploy pipeline flips blue/green on Sundays.",
    created_at: "2026-07-01T10:00:00Z",
  },
];

beforeAll(async () => {
  server = createServer((req, res) => {
    if (req.method === "GET" && req.url?.startsWith("/v1/memories")) {
      searchCalls++;
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(FIXTURE_MEMORIES));
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

// ── Fixture files ────────────────────────────────────────────────

async function makeEnv(): Promise<{ dataDir: string; largeFile: string; smallFile: string; largePng: string }> {
  const dataDir = await mkdtemp(join(tmpdir(), "ns-gate-e2e-"));
  const largeFile = join(dataDir, "gate-fixture-module.ts");
  await writeFile(largeFile, `// fixture\n${"x".repeat(2500)}\n`, "utf-8");
  const smallFile = join(dataDir, "tiny.ts");
  await writeFile(smallFile, "export const x = 1;\n", "utf-8");
  const largePng = join(dataDir, "big-image.png");
  await writeFile(largePng, Buffer.alloc(5000, 7));
  return { dataDir, largeFile, smallFile, largePng };
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

describe("read gate subprocess matrix", () => {
  it("large file + memories → deny with ranked timeline + override text", async () => {
    const { dataDir, largeFile } = await makeEnv();
    const r = await runGate(readEvent(largeFile, "s-deny", dataDir), {
      NEURALSCAPE_URL: serverUrl,
      CLAUDE_PLUGIN_DATA: dataDir,
    });
    expect(r.code).toBe(0);
    const out = JSON.parse(r.stdout);
    expect(out.hookSpecificOutput.hookEventName).toBe("PreToolUse");
    expect(out.hookSpecificOutput.permissionDecision).toBe("deny");
    const reason: string = out.hookSpecificOutput.permissionDecisionReason;
    expect(reason).toContain("[Neuralscape Read Gate]");
    expect(reason).toContain("`#id | when | title | ~tokens`");
    // ranked: the memory that MODIFIED the file precedes the read-only one;
    // the unrelated fixture hit is filtered by the verification pass
    expect(reason.indexOf("#mem-modified")).toBeGreaterThanOrEqual(0);
    expect(reason.indexOf("#mem-modified")).toBeLessThan(reason.indexOf("#mem-read"));
    expect(reason).not.toContain("mem-unrelated");
    // escalation menu + exact override instruction
    expect(reason).toContain("get_memories");
    expect(reason).toContain("timeline");
    expect(reason).toContain("Read the same path again");
  }, 15000);

  it("fires once per file per session: the second Read is allowed", async () => {
    const { dataDir, largeFile } = await makeEnv();
    const env = { NEURALSCAPE_URL: serverUrl, CLAUDE_PLUGIN_DATA: dataDir };
    const first = await runGate(readEvent(largeFile, "s-dedup", dataDir), env);
    expect(JSON.parse(first.stdout).hookSpecificOutput.permissionDecision).toBe("deny");
    const callsAfterFirst = searchCalls;
    const second = await runGate(readEvent(largeFile, "s-dedup", dataDir), env);
    expectAllow(second);
    // and the retry didn't pay a second search round-trip
    expect(searchCalls).toBe(callsAfterFirst);
  }, 20000);

  it("still gates the same file in a DIFFERENT session", async () => {
    const { dataDir, largeFile } = await makeEnv();
    const env = { NEURALSCAPE_URL: serverUrl, CLAUDE_PLUGIN_DATA: dataDir };
    await runGate(readEvent(largeFile, "s-one", dataDir), env);
    const other = await runGate(readEvent(largeFile, "s-two", dataDir), env);
    expect(JSON.parse(other.stdout).hookSpecificOutput.permissionDecision).toBe("deny");
  }, 20000);

  it("small file → allow without calling the API", async () => {
    const { dataDir, smallFile } = await makeEnv();
    const before = searchCalls;
    const r = await runGate(readEvent(smallFile, "s-small", dataDir), {
      NEURALSCAPE_URL: serverUrl,
      CLAUDE_PLUGIN_DATA: dataDir,
    });
    expectAllow(r);
    expect(searchCalls).toBe(before);
  }, 15000);

  it("binary/media extension → allow without calling the API", async () => {
    const { dataDir, largePng } = await makeEnv();
    const before = searchCalls;
    const r = await runGate(readEvent(largePng, "s-png", dataDir), {
      NEURALSCAPE_URL: serverUrl,
      CLAUDE_PLUGIN_DATA: dataDir,
    });
    expectAllow(r);
    expect(searchCalls).toBe(before);
  }, 15000);

  it("NS unreachable → allow, exit 0, failure counted (never-block)", async () => {
    const { dataDir, largeFile } = await makeEnv();
    const r = await runGate(readEvent(largeFile, "s-down", dataDir), {
      NEURALSCAPE_URL: DEAD_URL,
      CLAUDE_PLUGIN_DATA: dataDir,
    });
    expectAllow(r);
    expect(r.stderr).toContain("allowing read");
  }, 15000);

  it("excluded project → allow without calling the API", async () => {
    const { dataDir, largeFile } = await makeEnv();
    const projectId = dataDir.split("/").filter(Boolean).pop()!;
    const before = searchCalls;
    const r = await runGate(readEvent(largeFile, "s-excluded", dataDir), {
      NEURALSCAPE_URL: serverUrl,
      CLAUDE_PLUGIN_DATA: dataDir,
      NEURALSCAPE_EXCLUDED_PROJECTS: projectId,
    });
    expectAllow(r);
    expect(searchCalls).toBe(before);
  }, 15000);

  it("READ_GATE_ENABLED=false → allow without calling the API", async () => {
    const { dataDir, largeFile } = await makeEnv();
    const before = searchCalls;
    const r = await runGate(readEvent(largeFile, "s-disabled", dataDir), {
      NEURALSCAPE_URL: serverUrl,
      CLAUDE_PLUGIN_DATA: dataDir,
      NEURALSCAPE_READ_GATE_ENABLED: "false",
    });
    expectAllow(r);
    expect(searchCalls).toBe(before);
  }, 15000);

  it("non-Read tool events and missing files are ignored", async () => {
    const { dataDir, largeFile } = await makeEnv();
    const env = { NEURALSCAPE_URL: serverUrl, CLAUDE_PLUGIN_DATA: dataDir };
    const before = searchCalls;
    const edit = await runGate(
      JSON.stringify({ session_id: "s-x", cwd: dataDir, tool_name: "Edit", tool_input: { file_path: largeFile } }),
      env,
    );
    expectAllow(edit);
    const missing = await runGate(readEvent(join(dataDir, "no-such-file.ts"), "s-x", dataDir), env);
    expectAllow(missing);
    expect(searchCalls).toBe(before);
  }, 20000);

  it("malformed stdin → allow + exit 0 (exit 2 would BLOCK on PreToolUse)", async () => {
    const { dataDir } = await makeEnv();
    const r = await runGate("this is not json", {
      NEURALSCAPE_URL: serverUrl,
      CLAUDE_PLUGIN_DATA: dataDir,
    });
    expect(r.code).toBe(0);
    if (r.stdout.trim()) {
      expect(JSON.parse(r.stdout).hookSpecificOutput?.permissionDecision).toBeUndefined();
    }
  }, 15000);
});
