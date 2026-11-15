/**
 * Failure kinds and the identity notice, end to end: the BUILT hook bundles
 * run as subprocesses against a fixture HTTP server that answers a chosen
 * status (or is down).
 *
 * A bad token (401/403), a wrong URL (404), another 4xx, a failing service
 * (5xx) and a dead one each get their own SessionStart notice; only the last
 * two count toward the "unreachable" streak; the counter resets on success;
 * pre-tool-use shares the classification; every hook exits 0.
 *
 * `npm test` builds first (pretest), so scripts/*.js exist.
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeEach, describe, expect, it } from "vitest";

const PLUGIN_ROOT = join(__dirname, "..");
const PROJECT = "failure-kinds-fixture";
const TOKEN = "fixture-token-do-not-print-0123456789";
const DEAD_URL = "http://127.0.0.1:9";

// ── Fixture server ───────────────────────────────────────────────

let server: Server;
let serverUrl: string;
let status = 200;
let seenAuth: Array<string | undefined> = [];

await new Promise<void>((resolve) => {
  server = createServer((req, res) => {
    seenAuth.push(req.headers.authorization);
    req.resume();
    req.on("end", () => {
      res.writeHead(status, { "Content-Type": "application/json" });
      if (status !== 200) {
        res.end(JSON.stringify({ detail: "fixture failure" }));
      } else if ((req.url ?? "").startsWith("/v1/context/")) {
        res.end(JSON.stringify({ status: "ok", user_id: "u", categories: {} }));
      } else if ((req.url ?? "").startsWith("/v1/memories")) {
        res.end("[]");
      } else {
        res.end("{}");
      }
    });
  });
  server.listen(0, "127.0.0.1", () => {
    const a = server.address();
    serverUrl = `http://127.0.0.1:${typeof a === "object" && a ? a.port : 0}`;
    resolve();
  });
});

afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

beforeEach(() => {
  status = 200;
  seenAuth = [];
});

// ── Harness ──────────────────────────────────────────────────────

interface RunResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

function runHook(script: string, stdin: string, env: Record<string, string>): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    const child = spawn("node", [join(PLUGIN_ROOT, "scripts", script)], {
      env: { PATH: process.env.PATH ?? "", NEURALSCAPE_PROJECT_ID: PROJECT, ...env },
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

const startInput = JSON.stringify({ session_id: "s-fk", cwd: PLUGIN_ROOT, hook_event_name: "SessionStart", source: "startup" });

function ctx(r: RunResult): string {
  return JSON.parse(r.stdout).hookSpecificOutput?.additionalContext ?? "";
}

async function newDataDir(): Promise<string> {
  const dataDir = await mkdtemp(join(tmpdir(), "ns-failkind-sp-"));
  await mkdir(join(dataDir, "observations"), { recursive: true });
  return dataDir;
}

const counterPath = (dataDir: string) => join(dataDir, "observations", "unreachable.count");
const readCounter = async (dataDir: string) =>
  existsSync(counterPath(dataDir)) ? (await readFile(counterPath(dataDir), "utf-8")).trim() : null;

const baseEnv = (dataDir: string, url = serverUrl) => ({
  NEURALSCAPE_URL: url,
  NEURALSCAPE_USER_ID: "fk-user",
  NEURALSCAPE_API_KEY: TOKEN,
  CLAUDE_PLUGIN_DATA: dataDir,
});

// ── SessionStart, per failure kind ───────────────────────────────

describe("session-start words the notice by failure kind", () => {
  it.each([401, 403])("HTTP %i → key-rejected notice, no unreachable wording, no counter, token never printed", async (code) => {
    status = code;
    const dataDir = await newDataDir();
    const r = await runHook("session-start.js", startInput, baseEnv(dataDir));
    expect(r.code).toBe(0);
    const out = ctx(r);
    expect(out).toContain("rejected the API key");
    expect(out).toContain("/plugin config neuralscape@neuralscape-plugins");
    expect(out).not.toContain("unreachable");
    expect(out).not.toContain("docker compose");
    expect(r.stdout + r.stderr).not.toContain(TOKEN);
    expect(seenAuth.some((h) => h === `Bearer ${TOKEN}`)).toBe(true); // it WAS sent; just never echoed
    expect(await readCounter(dataDir)).toBeNull();
  }, 15000);

  it("HTTP 404 → check-the-URL notice, no counter", async () => {
    status = 404;
    const dataDir = await newDataDir();
    const r = await runHook("session-start.js", startInput, baseEnv(dataDir));
    expect(r.code).toBe(0);
    const out = ctx(r);
    expect(out).toContain("404");
    expect(out).toContain("URL");
    expect(out).not.toContain("unreachable");
    expect(await readCounter(dataDir)).toBeNull();
  }, 15000);

  it("HTTP 422 → generic client-rejection notice, no counter", async () => {
    status = 422;
    const dataDir = await newDataDir();
    const r = await runHook("session-start.js", startInput, baseEnv(dataDir));
    expect(r.code).toBe(0);
    expect(ctx(r)).toContain("rejected the context request");
    expect(await readCounter(dataDir)).toBeNull();
  }, 15000);

  it("HTTP 500 → unreachable notice and the streak advances", async () => {
    status = 500;
    const dataDir = await newDataDir();
    const r = await runHook("session-start.js", startInput, baseEnv(dataDir));
    expect(r.code).toBe(0);
    expect(ctx(r)).toContain("memory service unreachable");
    expect(await readCounter(dataDir)).toBe("1");
  }, 15000);

  it("a closed port → unreachable notice; three in a row go fail-loud", async () => {
    const dataDir = await newDataDir();
    const env = baseEnv(dataDir, DEAD_URL);
    expect(ctx(await runHook("session-start.js", startInput, env))).toContain("memory service unreachable");
    await runHook("session-start.js", startInput, env);
    const third = await runHook("session-start.js", startInput, env);
    expect(third.code).toBe(0);
    expect(ctx(third)).toContain("unreachable for 3 consecutive events");
    expect(await readCounter(dataDir)).toBe("3");
  }, 30000);

  it("auth failures neither advance nor clear a running streak; success clears it", async () => {
    const dataDir = await newDataDir();
    await writeFile(counterPath(dataDir), "2", "utf-8");
    status = 401;
    await runHook("session-start.js", startInput, baseEnv(dataDir));
    expect(await readCounter(dataDir)).toBe("2");
    status = 200;
    const ok = await runHook("session-start.js", startInput, baseEnv(dataDir));
    expect(ok.code).toBe(0);
    expect(ctx(ok)).not.toContain("unreachable");
    expect(await readCounter(dataDir)).toBeNull();
  }, 20000);
});

// ── A second hook shares the classification ──────────────────────

describe("pre-tool-use shares the classification", () => {
  async function gate(dataDir: string, env: Record<string, string>, sessionId: string): Promise<RunResult> {
    const big = join(dataDir, "big-fixture.ts");
    await writeFile(big, `// fixture\n${"x".repeat(2500)}\n`, "utf-8");
    return runHook(
      "pre-tool-use.js",
      JSON.stringify({
        session_id: sessionId,
        cwd: dataDir,
        hook_event_name: "PreToolUse",
        tool_name: "Read",
        tool_input: { file_path: big },
      }),
      env,
    );
  }

  it("401 on the read-gate fetch: allow, exit 0, labelled auth, streak untouched", async () => {
    status = 401;
    const dataDir = await newDataDir();
    const r = await gate(dataDir, baseEnv(dataDir), "g-401");
    expect(r.code).toBe(0);
    expect(r.stderr).toContain("(auth)");
    expect(r.stderr).not.toContain(TOKEN);
    expect(await readCounter(dataDir)).toBeNull();
  }, 15000);

  it("500 on the read-gate fetch: allow, exit 0, labelled server, streak advances", async () => {
    status = 500;
    const dataDir = await newDataDir();
    const r = await gate(dataDir, baseEnv(dataDir), "g-500");
    expect(r.code).toBe(0);
    expect(r.stderr).toContain("(server)");
    expect(await readCounter(dataDir)).toBe("1");
  }, 15000);
});

// ── Identity notice ──────────────────────────────────────────────

describe("identity notice when USER_ID is unset", () => {
  // A hostname that is not localhost, mapped to the fixture server by a
  // preload that patches dns.lookup for that one name.
  const REMOTE_HOST = "ns-remote.fixture.test";
  let preload: string;
  async function remoteUrl(): Promise<string> {
    if (!preload) {
      preload = join(await mkdtemp(join(tmpdir(), "ns-dns-preload-")), "preload.cjs");
      await writeFile(
        preload,
        `const dns=require("node:dns");const orig=dns.lookup;
dns.lookup=function(h,o,cb){if(typeof o==="function"){cb=o;o={};}
if(h==="${REMOTE_HOST}"){return o&&o.all?cb(null,[{address:"127.0.0.1",family:4}]):cb(null,"127.0.0.1",4);}
return orig.call(dns,h,o,cb);};`,
        "utf-8",
      );
    }
    return `http://${REMOTE_HOST}:${new URL(serverUrl).port}`;
  }

  const noUserIdEnv = (dataDir: string, url: string) => ({
    NEURALSCAPE_URL: url,
    USER: "oslogin-fixture",
    CLAUDE_PLUGIN_DATA: dataDir,
  });

  it("OS-login fallback against a non-local URL → one line naming the id, not the host", async () => {
    const dataDir = await newDataDir();
    const url = await remoteUrl();
    const r = await runHook("session-start.js", startInput, {
      ...noUserIdEnv(dataDir, url),
      NODE_OPTIONS: `--require ${preload}`,
    });
    expect(r.code).toBe(0);
    const out = ctx(r);
    expect(out).toContain('"oslogin-fixture"');
    expect(out).toContain("USER_ID is not set");
    expect(out).not.toContain(REMOTE_HOST);
    expect(out.match(/USER_ID is not set/g)).toHaveLength(1);
  }, 15000);

  it("configured USER_ID → no notice", async () => {
    const dataDir = await newDataDir();
    const url = await remoteUrl();
    const r = await runHook("session-start.js", startInput, {
      ...noUserIdEnv(dataDir, url),
      NEURALSCAPE_USER_ID: "chosen-id",
      NODE_OPTIONS: `--require ${preload}`,
    });
    expect(r.code).toBe(0);
    expect(ctx(r)).not.toContain("USER_ID is not set");
  }, 15000);

  it("OS-login fallback against a local service → no notice", async () => {
    const dataDir = await newDataDir();
    const r = await runHook("session-start.js", startInput, noUserIdEnv(dataDir, serverUrl));
    expect(r.code).toBe(0);
    expect(ctx(r)).not.toContain("USER_ID is not set");
  }, 15000);
});
