/**
 * Session lifecycle, end to end, against the BUILT hook bundles (subprocesses)
 * and a fixture HTTP server standing in for Neuralscape.
 *
 * 1. Stale-buffer handoff: a buffer is offered to the next session for
 *    compilation only once its session is over (SessionEnd marker, or a long
 *    idle window for a crash); the Stop hook (after every turn) says nothing;
 *    the hand-off still happens while the service is down.
 * 2. Resume after compaction: PreCompact on a real-shaped Claude Code
 *    transcript -> the POST /v1/memories/raw body -> served back by
 *    GET /v1/context/<project> -> SessionStart(source: "compact") renders the
 *    last user prompts under "Resuming after compaction", only for the
 *    session that compacted, and also when the stored write is not yet
 *    searchable (it is only enqueued) or the service is down.
 * 3. PreCompact for a client with no transcript still delivers its turn.
 *
 * `npm test` builds first (pretest), so scripts/*.js exist.
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, utimes, writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeEach, describe, expect, it } from "vitest";

const PLUGIN_ROOT = join(__dirname, "..");
const DEAD_URL = "http://127.0.0.1:9";
const PROJECT = "lifecycle-fixture-project";

// ── Fixture server ───────────────────────────────────────────────

interface Recorded {
  method: string;
  path: string;
  body: Record<string, unknown>;
}

let server: Server;
let serverUrl: string;
let requests: Recorded[] = [];
/** Memories the fixture returns from GET /v1/context/<project> (the server's stored state). */
let stored: Array<Record<string, unknown>> = [];

await new Promise<void>((resolve) => {
  server = createServer((req, res) => {
    let body = "";
    req.on("data", (d) => (body += d));
    req.on("end", () => {
      const path = req.url ?? "";
      requests.push({ method: req.method ?? "", path, body: body ? JSON.parse(body) : {} });
      if (req.method === "GET" && path.startsWith("/v1/context/")) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            status: "ok",
            user_id: "lifecycle-test",
            categories: stored.length > 0 ? { task_context: stored } : {},
          }),
        );
        return;
      }
      if (req.method === "GET") {
        // dreaming card, code-graph probe, ...: not available on this fixture
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end("{}");
        return;
      }
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
  stored = [];
});

// ── Harness ──────────────────────────────────────────────────────

interface RunResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

function runHook(
  script: string,
  stdin: string,
  dataDir: string,
  env: Record<string, string> = {},
): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    const child = spawn("node", [join(PLUGIN_ROOT, "scripts", script)], {
      env: {
        PATH: process.env.PATH ?? "",
        NEURALSCAPE_USER_ID: "lifecycle-test",
        NEURALSCAPE_URL: serverUrl,
        NEURALSCAPE_PROJECT_ID: PROJECT,
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

function additionalContext(r: RunResult): string {
  return JSON.parse(r.stdout).hookSpecificOutput?.additionalContext ?? "";
}

async function newDataDir(): Promise<{ dataDir: string; obsDir: string }> {
  const dataDir = await mkdtemp(join(tmpdir(), "ns-lifecycle-data-"));
  const obsDir = join(dataDir, "observations");
  await mkdir(obsDir, { recursive: true });
  return { dataDir, obsDir };
}

const ROW = JSON.stringify({ ts: "2026-11-16T00:00:00Z", tool: "Bash", input: "make", output: "ok" }) + "\n";

async function seedBuffer(obsDir: string, sessionId: string, ageMs: number): Promise<string> {
  const path = join(obsDir, `${sessionId}.jsonl`);
  await writeFile(path, ROW, "utf-8");
  const t = new Date(Date.now() - ageMs);
  await utimes(path, t, t);
  return path;
}

const startInput = (sessionId: string, source = "startup") =>
  JSON.stringify({
    session_id: sessionId,
    cwd: PLUGIN_ROOT,
    hook_event_name: "SessionStart",
    source,
  });

const HOUR = 60 * 60 * 1000;

// ── 1. Stale-buffer handoff ──────────────────────────────────────

describe("stale-buffer handoff", () => {
  it("does not offer another terminal's live buffer, even after its Stop hook ran", async () => {
    const { dataDir, obsDir } = await newDataDir();
    const live = await seedBuffer(obsDir, "terminal-a", 20 * 1000);

    // Terminal A finishes a turn: Stop fires (session-end.js). This must NOT mark the buffer as finished.
    const stop = await runHook(
      "session-end.js",
      JSON.stringify({ session_id: "terminal-a", cwd: PLUGIN_ROOT, hook_event_name: "Stop" }),
      dataDir,
    );
    expect(stop.code).toBe(0);
    expect(existsSync(live + ".ended")).toBe(false);
    expect(existsSync(live + ".stale")).toBe(false);

    // Terminal B starts.
    const r = await runHook("session-start.js", startInput("terminal-b"), dataDir);
    expect(r.code).toBe(0);
    expect(additionalContext(r)).not.toContain("Pending Observations");
    expect(additionalContext(r)).not.toContain(live);
  });

  it("offers the buffer once the SessionEnd hook has fired", async () => {
    const { dataDir, obsDir } = await newDataDir();
    const buffer = await seedBuffer(obsDir, "terminal-a", 60 * 1000);

    // SessionEnd for terminal A, with the service down: the marker is local and must still be written.
    const end = await runHook(
      "session-summary.js",
      JSON.stringify({ session_id: "terminal-a", cwd: PLUGIN_ROOT, hook_event_name: "SessionEnd", reason: "exit" }),
      dataDir,
      { NEURALSCAPE_URL: DEAD_URL },
    );
    expect(end.code).toBe(0);
    expect(existsSync(buffer + ".ended")).toBe(true);

    const r = await runHook("session-start.js", startInput("terminal-b"), dataDir);
    const ctx = additionalContext(r);
    expect(ctx).toContain("# Neuralscape — Pending Observations");
    expect(ctx).toContain(buffer);
  });

  it("writes no marker for an empty buffer at SessionEnd", async () => {
    const { dataDir, obsDir } = await newDataDir();
    await writeFile(join(obsDir, "terminal-a.jsonl"), "", "utf-8");
    const end = await runHook(
      "session-summary.js",
      JSON.stringify({ session_id: "terminal-a", cwd: PLUGIN_ROOT, hook_event_name: "SessionEnd" }),
      dataDir,
    );
    expect(end.code).toBe(0);
    expect(existsSync(join(obsDir, "terminal-a.jsonl.ended"))).toBe(false);
  });

  it("offers a crashed session's buffer (never ended) after the idle window", async () => {
    const { dataDir, obsDir } = await newDataDir();
    const buffer = await seedBuffer(obsDir, "crashed", 13 * HOUR);
    const r = await runHook("session-start.js", startInput("terminal-b"), dataDir);
    expect(additionalContext(r)).toContain(buffer);
  });

  it("still emits the pending note when the context fetch fails (service down / bad token)", async () => {
    const { dataDir, obsDir } = await newDataDir();
    const buffer = await seedBuffer(obsDir, "ended-earlier", 2 * HOUR);
    await writeFile(buffer + ".ended", "x", "utf-8"); // written now, after the last buffer write

    const r = await runHook("session-start.js", startInput("terminal-b"), dataDir, { NEURALSCAPE_URL: DEAD_URL });
    expect(r.code).toBe(0);
    const ctx = additionalContext(r);
    expect(ctx).toContain("memory service unreachable");
    expect(ctx).toContain("# Neuralscape — Pending Observations");
    expect(ctx).toContain(buffer);
  });

  it("removes the end marker the compile skill leaves behind after it empties the buffer", async () => {
    const { dataDir, obsDir } = await newDataDir();
    const buffer = await seedBuffer(obsDir, "done", 2 * HOUR);
    await writeFile(buffer + ".ended", "x", "utf-8");
    await writeFile(buffer, "", "utf-8"); // the skill truncates with the Write tool
    const r = await runHook("session-start.js", startInput("terminal-b"), dataDir);
    expect(additionalContext(r)).not.toContain("Pending Observations");
    expect(existsSync(buffer + ".ended")).toBe(false);
  });
});

// ── 2. Resume after compaction ───────────────────────────────────

// The real Claude Code shape: text under message.content (a string for the user, text blocks for the assistant).
function transcriptLines(prompts: string[]): string {
  const rows: unknown[] = [];
  for (const p of prompts) {
    rows.push({ type: "user", message: { role: "user", content: p } });
    rows.push({
      type: "assistant",
      message: { role: "assistant", content: [{ type: "text", text: `Answer to: ${p} with plenty of substance` }] },
    });
  }
  return rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
}

const PROMPTS = [
  "set up the fixture server for the plugin tests",
  "now wire the pre-compact bundle into the loop",
  "check that the resume block shows my last prompts",
];

async function runPreCompact(dataDir: string, sessionId: string, prompts: string[], env: Record<string, string> = {}) {
  const dir = await mkdtemp(join(tmpdir(), "ns-lifecycle-transcript-"));
  const transcript = join(dir, `${sessionId}.jsonl`);
  await writeFile(transcript, transcriptLines(prompts), "utf-8");
  const r = await runHook(
    "pre-compact.js",
    JSON.stringify({
      session_id: sessionId,
      transcript_path: transcript,
      cwd: PLUGIN_ROOT,
      hook_event_name: "PreCompact",
      trigger: "auto",
      custom_instructions: "",
    }),
    dataDir,
    env,
  );
  expect(r.code).toBe(0);
  return r;
}

/** What the service would return for the memory the PreCompact hook POSTed. */
function storeFromRaw(): Record<string, unknown> {
  const raw = requests.find((q) => q.method === "POST" && q.path.includes("/v1/memories/raw"));
  expect(raw).toBeDefined();
  return {
    id: `mem-${stored.length + 1}`,
    memory: raw!.body.content,
    category: raw!.body.category,
    tags: raw!.body.tags,
    created_at: new Date().toISOString(),
  };
}

describe("resume after compaction (PreCompact -> service -> SessionStart)", () => {
  it("renders the last user prompts of the session that compacted", async () => {
    const { dataDir } = await newDataDir();
    await runPreCompact(dataDir, "sess-compacted", PROMPTS);

    // The service stores what the hook sent; the next context fetch serves it back.
    stored = [storeFromRaw()];
    requests = [];

    const r = await runHook("session-start.js", startInput("sess-compacted", "compact"), dataDir);
    expect(r.code).toBe(0);
    const ctx = additionalContext(r);
    expect(ctx).toContain("## Resuming after compaction");
    for (const p of PROMPTS) expect(ctx).toContain(p);
    expect(ctx).toContain("Compact snapshot: session sess-compacted");
  });

  it("does not re-inject another terminal's snapshot after a compaction", async () => {
    const { dataDir } = await newDataDir();
    await runPreCompact(dataDir, "terminal-a", ["terminal A prompt about the billing export"]);
    const snapshotA = storeFromRaw();
    requests = [];
    await runPreCompact(dataDir, "terminal-b", ["terminal B prompt about the retry loop"]);
    const snapshotB = storeFromRaw();
    stored = [snapshotA, snapshotB]; // B is newer, A is older: both are in the project's memories
    requests = [];

    // Terminal A compacts again: it must see its own snapshot only.
    const r = await runHook("session-start.js", startInput("terminal-a", "compact"), dataDir);
    const ctx = additionalContext(r);
    expect(ctx).toContain("## Resuming after compaction");
    expect(ctx).toContain("terminal A prompt about the billing export");
    expect(ctx).not.toContain("terminal B prompt about the retry loop");
  });

  it("still renders the snapshot when the stored write is not searchable yet (enqueued, 202)", async () => {
    const { dataDir } = await newDataDir();
    await runPreCompact(dataDir, "sess-compacted", PROMPTS);
    stored = []; // the worker has not processed the queued raw write yet

    const r = await runHook("session-start.js", startInput("sess-compacted", "compact"), dataDir);
    const ctx = additionalContext(r);
    expect(ctx).toContain("## Resuming after compaction");
    expect(ctx).toContain(PROMPTS[2]);
  });

  it("does not render the snapshot twice when both the stored and the local copy exist", async () => {
    const { dataDir } = await newDataDir();
    await runPreCompact(dataDir, "sess-compacted", PROMPTS);
    stored = [storeFromRaw()];
    const r = await runHook("session-start.js", startInput("sess-compacted", "compact"), dataDir);
    const ctx = additionalContext(r);
    expect(ctx.split(PROMPTS[2]).length - 1).toBe(1);
  });

  it("renders the local snapshot even when the service is unreachable", async () => {
    const { dataDir } = await newDataDir();
    await runPreCompact(dataDir, "sess-compacted", PROMPTS);
    const r = await runHook("session-start.js", startInput("sess-compacted", "compact"), dataDir, {
      NEURALSCAPE_URL: DEAD_URL,
    });
    const ctx = additionalContext(r);
    expect(ctx).toContain("memory service unreachable");
    expect(ctx).toContain("## Resuming after compaction");
    expect(ctx).toContain(PROMPTS[1]);
  });

  it("sends the session id as a tag on the stored snapshot", async () => {
    const { dataDir } = await newDataDir();
    await runPreCompact(dataDir, "sess-compacted", PROMPTS);
    const raw = requests.find((q) => q.path.includes("/v1/memories/raw"));
    expect(raw?.body.tags).toEqual(["compact_snapshot", "session:sess-compacted"]);
  });

  async function resumedTranscript(previousId: string, newId: string): Promise<string> {
    const dir = await mkdtemp(join(tmpdir(), "ns-lifecycle-resumed-"));
    const path = join(dir, `${newId}.jsonl`);
    const rows = [
      { type: "user", sessionId: previousId, message: { role: "user", content: "earlier prompt" } },
      { type: "user", sessionId: newId, message: { role: "user", content: "resumed prompt" } },
    ];
    await writeFile(path, rows.map((r) => JSON.stringify(r)).join("\n") + "\n", "utf-8");
    return path;
  }

  for (const source of ["resume", "fork"]) {
    it(`on ${source} (a NEW session id) renders the predecessor session's snapshot read from the transcript`, async () => {
      const { dataDir } = await newDataDir();
      await runPreCompact(dataDir, "sess-old", PROMPTS);
      const mine = storeFromRaw();
      requests = [];
      await runPreCompact(dataDir, "sess-unrelated", ["unrelated terminal prompt about billing"]);
      stored = [mine, storeFromRaw()];
      requests = [];

      const transcript = await resumedTranscript("sess-old", "sess-new-id");
      const r = await runHook(
        "session-start.js",
        JSON.stringify({ session_id: "sess-new-id", transcript_path: transcript, cwd: PLUGIN_ROOT, hook_event_name: "SessionStart", source }),
        dataDir,
      );
      const ctx = additionalContext(r);
      expect(ctx).toContain("## Resuming after compaction");
      expect(ctx).toContain(PROMPTS[2]);
      expect(ctx).not.toContain("unrelated terminal prompt about billing");
    });

    it(`on ${source} with no predecessor id in the transcript renders no snapshot`, async () => {
      const { dataDir } = await newDataDir();
      await runPreCompact(dataDir, "sess-unrelated", ["unrelated terminal prompt about billing"]);
      stored = [storeFromRaw()];
      requests = [];
      const r = await runHook(
        "session-start.js",
        JSON.stringify({ session_id: "sess-new-id", cwd: PLUGIN_ROOT, hook_event_name: "SessionStart", source }),
        dataDir,
      );
      expect(additionalContext(r)).not.toContain("Resuming after compaction");
      expect(additionalContext(r)).not.toContain("unrelated terminal prompt about billing");
    });
  }

  it("writes the local snapshot before the network, so a hanging service cannot prevent it", async () => {
    const { dataDir, obsDir } = await newDataDir();
    const hang = createServer(() => {
      /* never answers */
    });
    await new Promise<void>((resolve) => hang.listen(0, "127.0.0.1", () => resolve()));
    const addr = hang.address();
    const hangUrl = `http://127.0.0.1:${typeof addr === "object" && addr ? addr.port : 0}`;
    const dir = await mkdtemp(join(tmpdir(), "ns-lifecycle-hang-"));
    const transcript = join(dir, "t.jsonl");
    await writeFile(transcript, transcriptLines(PROMPTS), "utf-8");
    const child = spawn("node", [join(PLUGIN_ROOT, "scripts", "pre-compact.js")], {
      env: {
        PATH: process.env.PATH ?? "",
        NEURALSCAPE_USER_ID: "lifecycle-test",
        NEURALSCAPE_URL: hangUrl,
        NEURALSCAPE_PROJECT_ID: PROJECT,
        CLAUDE_PLUGIN_DATA: dataDir,
      },
      stdio: ["pipe", "pipe", "pipe"],
    });
    child.stdin.write(
      JSON.stringify({ session_id: "sess-hang", transcript_path: transcript, cwd: PLUGIN_ROOT, hook_event_name: "PreCompact", trigger: "auto" }),
    );
    child.stdin.end();
    const file = join(obsDir, "sess-hang.compact-snapshot.json");
    const deadline = Date.now() + 4000;
    while (!existsSync(file) && Date.now() < deadline) await new Promise((r) => setTimeout(r, 50));
    const stillRunning = child.exitCode === null;
    child.kill("SIGKILL");
    await new Promise<void>((resolve) => hang.close(() => resolve()));
    expect(existsSync(file)).toBe(true);
    expect(stillRunning).toBe(true); // the POST is still hanging: the file did not wait for it
  });
});

// ── 3. PreCompact for a client with no transcript ────────────────

describe("pre-compact for a generic client", () => {
  it("delivers the turn that arrives in the payload, then stores the snapshot", async () => {
    const { dataDir } = await newDataDir();
    const r = await runHook(
      "pre-compact.js",
      JSON.stringify({
        session_id: "generic-sess",
        user_message: "generic client prompt before the compact",
        assistant_response: "generic client answer with plenty of substance here",
        channel: "api",
      }),
      dataDir,
    );
    expect(r.code).toBe(0);

    const flushes = requests.filter((q) => q.path.includes("/conversation-compiler/flush"));
    expect(flushes.length).toBe(1);
    expect(flushes[0].body.user_message).toBe("generic client prompt before the compact");
    expect(flushes[0].body.session_id).toBe("generic-sess");

    const raws = requests.filter((q) => q.path.includes("/v1/memories/raw"));
    expect(raws.length).toBe(1);
    expect(String(raws[0].body.content)).toContain("generic client prompt before the compact");
  });

  it("does not deliver a turn twice when the conversation-turn hook already delivered it (no cursor)", async () => {
    const { dataDir } = await newDataDir();
    const payload = JSON.stringify({
      session_id: "generic-dup",
      user_message: "generic client prompt delivered twice?",
      assistant_response: "generic client answer with plenty of substance here",
      channel: "api",
    });
    expect((await runHook("conversation-turn.js", payload, dataDir)).code).toBe(0);
    expect((await runHook("pre-compact.js", payload, dataDir)).code).toBe(0);
    expect(requests.filter((q) => q.path.includes("/conversation-compiler/flush")).length).toBe(1);

    // A different turn in the same session is still delivered.
    const next = JSON.stringify({ ...JSON.parse(payload), user_message: "a different prompt entirely" });
    await runHook("pre-compact.js", next, dataDir);
    expect(requests.filter((q) => q.path.includes("/conversation-compiler/flush")).length).toBe(2);
  });

  it("Claude Code still delivers no turns from PreCompact", async () => {
    const { dataDir } = await newDataDir();
    await runPreCompact(dataDir, "sess-cc", PROMPTS);
    expect(requests.filter((q) => q.path.includes("/conversation-compiler/flush")).length).toBe(0);
    expect(requests.filter((q) => q.path.includes("/v1/memories/raw")).length).toBe(1);
  });
});
