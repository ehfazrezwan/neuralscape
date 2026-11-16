/**
 * The skills' helper CLI (scripts/sync.js) end to end, BUILT bundle against a
 * recording server: the config it resolves is the hooks' config (explicit URL
 * > baked .mcp.json > localhost), every request carries the Bearer token, and
 * `<private>` text never reaches a POST body.
 */

import { spawn } from "node:child_process";
import { createServer, type IncomingHttpHeaders, type Server } from "node:http";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { PLUGIN_ROOT } from "./helpers/fixture.js";

interface Seen {
  method: string;
  path: string;
  headers: IncomingHttpHeaders;
  raw: string;
}

let server: Server;
let url = "";
const seen: Seen[] = [];
const failPaths = new Set<string>();

beforeAll(async () => {
  server = createServer((req, res) => {
    let raw = "";
    req.on("data", (d) => (raw += d));
    req.on("end", () => {
      const path = req.url ?? "";
      seen.push({ method: req.method ?? "", path, headers: req.headers, raw });
      if (failPaths.has(path)) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end("{}");
        return;
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok", path }));
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", () => r()));
  const addr = server.address();
  url = `http://127.0.0.1:${typeof addr === "object" && addr ? addr.port : 0}`;
});
afterAll(() => new Promise<void>((r) => server.close(() => r())));
beforeEach(() => {
  seen.length = 0;
  failPaths.clear();
});

const FLUSH = "/v1/extensions/conversation-compiler/flush";
const COMPILE = "/v1/extensions/conversation-compiler/compile";
const KEY = "tok-SECRET-1234";

function run(args: string[], stdin: string, env: Record<string, string>) {
  return new Promise<{ code: number | null; stdout: string; stderr: string }>((resolve, reject) => {
    const data = mkdtempSync(join(tmpdir(), "ns-sync-data-"));
    const child = spawn("node", [join(PLUGIN_ROOT, "scripts", "sync.js"), ...args], {
      env: { PATH: process.env.PATH ?? "", CLAUDE_PLUGIN_DATA: data, ...env },
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

function pluginRootWith(mcpUrl?: string): string {
  const root = mkdtempSync(join(tmpdir(), "ns-root-"));
  if (mcpUrl) writeFileSync(join(root, ".mcp.json"), JSON.stringify({ mcpServers: { neuralscape: { type: "http", url: mcpUrl } } }));
  return root;
}

const LONG = (s: string) => `${s} — long enough to be captured as a turn.`;

describe("print-config resolves the URL the way the hooks do", () => {
  it("default install (no URL option): the baked .mcp.json URL, /mcp/ suffix stripped", async () => {
    const r = await run(["print-config"], "", { CLAUDE_PLUGIN_ROOT: pluginRootWith("https://memory.example.test/mcp/"), USER: "alice" });
    expect(r.code).toBe(0);
    const cfg = JSON.parse(r.stdout);
    expect(cfg).toMatchObject({ url: "https://memory.example.test", url_source: "baked", api_key_set: false, user_id: "alice", user_id_source: "os" });
  });

  it("nothing configured and nothing baked: localhost", async () => {
    const r = await run(["print-config"], "", { CLAUDE_PLUGIN_ROOT: pluginRootWith(), USER: "alice" });
    expect(JSON.parse(r.stdout)).toMatchObject({ url: "http://localhost:8199", url_source: "default" });
  });

  it("an explicit URL option wins over the baked one; the key is reported as set, never printed", async () => {
    const r = await run(["print-config"], "", {
      CLAUDE_PLUGIN_ROOT: pluginRootWith("https://memory.example.test/mcp/"),
      CLAUDE_PLUGIN_OPTION_URL: "http://127.0.0.1:9/",
      CLAUDE_PLUGIN_OPTION_API_KEY: KEY,
      CLAUDE_PLUGIN_OPTION_USER_ID: "bob",
      CLAUDE_PLUGIN_OPTION_PROJECT_ID: "proj-x",
    });
    const cfg = JSON.parse(r.stdout);
    expect(cfg).toMatchObject({ url: "http://127.0.0.1:9", url_source: "config", api_key_set: true, user_id: "bob", user_id_source: "config", project_id: "proj-x" });
    expect(r.stdout + r.stderr).not.toContain(KEY);
  });
});

describe("flush sends the Bearer token and never a private span", () => {
  const env = (extra: Record<string, string> = {}) => ({
    CLAUDE_PLUGIN_OPTION_URL: url,
    CLAUDE_PLUGIN_OPTION_API_KEY: KEY,
    CLAUDE_PLUGIN_OPTION_USER_ID: "sync-user",
    CLAUDE_PLUGIN_OPTION_PROJECT_ID: "sync-proj",
    ...extra,
  });

  it("authenticates flush + compile and keeps <private> text (closed and unclosed) out of the body", async () => {
    const payload = {
      turns: [
        { user_message: "tell me the plan <private>CLOSED-SECRET</private>", assistant_response: LONG("Here is the plan") , session_id: "s1" },
        { user_message: "next <private>UNCLOSED-SECRET carries on", assistant_response: LONG("Done, with <private>ANSWER-SECRET</private> removed"), session_id: "s1" },
        { user_message: "short", assistant_response: "ok" },
      ],
      compile: true,
      date: "2026-11-16",
    };
    const r = await run(["flush"], JSON.stringify(payload), env());
    expect(r.code).toBe(0);
    expect(JSON.parse(r.stdout)).toMatchObject({ attempted: 3, flushed: 2, failed: false, compiled: true });

    const flushes = seen.filter((s) => s.path === FLUSH);
    expect(flushes).toHaveLength(2);
    const compiles = seen.filter((s) => s.path === COMPILE);
    expect(compiles).toHaveLength(1);
    for (const call of [...flushes, ...compiles]) expect(call.headers.authorization).toBe(`Bearer ${KEY}`);
    const everything = seen.map((s) => s.raw).join("\n");
    for (const secret of ["CLOSED-SECRET", "UNCLOSED-SECRET", "ANSWER-SECRET"]) expect(everything).not.toContain(secret);
    const body = JSON.parse(flushes[0].raw);
    expect(body).toMatchObject({ channel: "claude-code-manual-sync", project_id: "sync-proj", user_id: "sync-user", session_id: "s1" });
    expect(body.user_message).toContain("[redacted]");
    expect(JSON.parse(compiles[0].raw)).toEqual({ date: "2026-11-16", user_id: "sync-user" });
    expect(r.stdout + r.stderr).not.toContain(KEY);
  });

  it("uses the baked URL when no URL option is set (default install)", async () => {
    // The baked URL here is the fixture itself, so the call is observable.
    const root = pluginRootWith(`${url}/mcp/`);
    const r = await run(["flush"], JSON.stringify({ turns: [{ user_message: "hello there", assistant_response: LONG("Greetings") }] }), {
      CLAUDE_PLUGIN_ROOT: root,
      CLAUDE_PLUGIN_OPTION_USER_ID: "sync-user",
    });
    expect(r.code).toBe(0);
    expect(seen.filter((s) => s.path === FLUSH)).toHaveLength(1);
    expect(seen[0].headers.authorization).toBeUndefined();
  });

  it("a refused flush exits 1 and does not trigger the compile", async () => {
    failPaths.add(FLUSH);
    const r = await run(["flush"], JSON.stringify({ turns: [{ user_message: "hello there", assistant_response: LONG("Greetings") }], compile: true }), env());
    expect(r.code).toBe(1);
    expect(JSON.parse(r.stdout)).toMatchObject({ failed: true, first_failed_index: 0, compiled: null });
    expect(seen.some((s) => s.path === COMPILE)).toBe(false);
  });

  it("rejects malformed input with exit 2 and sends nothing", async () => {
    const bad = await run(["flush"], "not json", env());
    expect(bad.code).toBe(2);
    const noTurns = await run(["flush"], "{}", env());
    expect(noTurns.code).toBe(2);
    expect(seen).toHaveLength(0);
  });
});

describe("request and upload carry the Bearer token to the resolved URL", () => {
  const withUrl = () => ({ CLAUDE_PLUGIN_OPTION_URL: url, CLAUDE_PLUGIN_OPTION_API_KEY: KEY, CLAUDE_PLUGIN_OPTION_USER_ID: "u" });

  it("request POST /v1/search", async () => {
    const r = await run(["request", "POST", "/v1/search"], JSON.stringify({ query: "q", user_id: "u", limit: 10 }), withUrl());
    expect(r.code).toBe(0);
    expect(JSON.parse(r.stdout)).toMatchObject({ status: "ok" });
    expect(seen).toHaveLength(1);
    expect(seen[0].headers.authorization).toBe(`Bearer ${KEY}`);
    expect(JSON.parse(seen[0].raw)).toEqual({ query: "q", user_id: "u", limit: 10 });
  });

  it("request refuses a path outside /v1/ and a method other than GET/POST", async () => {
    expect((await run(["request", "POST", "/admin"], "{}", withUrl())).code).toBe(2);
    expect((await run(["request", "DELETE", "/v1/x"], "{}", withUrl())).code).toBe(2);
    expect(seen).toHaveLength(0);
  });

  it("request exits 1 when the service refuses", async () => {
    failPaths.add("/v1/search");
    expect((await run(["request", "POST", "/v1/search"], "{}", withUrl())).code).toBe(1);
  });

  it("upload sends a multipart POST /v1/ingest/files with the token and fields", async () => {
    const dir = mkdtempSync(join(tmpdir(), "ns-up-"));
    mkdirSync(dir, { recursive: true });
    const file = join(dir, "notes.md");
    writeFileSync(file, "# hello upload");
    const r = await run(["upload", "--field", "user_id=u", "--field", "category=domain_knowledge", file], "", withUrl());
    expect(r.code).toBe(0);
    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ method: "POST", path: "/v1/ingest/files" });
    expect(seen[0].headers.authorization).toBe(`Bearer ${KEY}`);
    expect(String(seen[0].headers["content-type"])).toContain("multipart/form-data");
    expect(seen[0].raw).toContain("hello upload");
    expect(seen[0].raw).toContain('name="category"');
    expect(seen[0].raw).toContain('filename="notes.md"');
  });

  it("upload usage errors exit 2; a refused upload exits 1", async () => {
    expect((await run(["upload"], "", withUrl())).code).toBe(2);
    expect((await run(["upload", "--field", "nokey"], "", withUrl())).code).toBe(2);
    failPaths.add("/v1/ingest/files");
    const dir = mkdtempSync(join(tmpdir(), "ns-up-"));
    const file = join(dir, "a.md");
    writeFileSync(file, "x");
    expect((await run(["upload", file], "", withUrl())).code).toBe(1);
  });

  it("an unknown mode exits 2", async () => {
    expect((await run(["bogus"], "", withUrl())).code).toBe(2);
  });
});
