/**
 * Real Claude Code transcript shape: entries are `{type, message: {role, content}, ...}` with `content` a string
 * (user prompts) or a block list (assistant text, tool calls, user tool results). Capture used to read only a
 * top-level `content`, so every real turn looked empty, was dropped as noise, and the flush offset still advanced:
 * the Stop flush, the PreCompact flush and the SessionEnd note delivered nothing.
 *
 * The fixtures below are synthetic: they reproduce the key structure only (never real transcript text).
 */

import { spawn } from "node:child_process";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeEach, describe, expect, it } from "vitest";

const PLUGIN_ROOT = join(__dirname, "..");

// ── Fixture server (up BEFORE utils.js resolves its base URL) ────

let server: Server;
let requests: Array<{ path: string; body: Record<string, any> }> = [];
let failFlushFrom = -1;
let flushCount = 0;

await new Promise<void>((resolve) => {
  server = createServer((req, res) => {
    let body = "";
    req.on("data", (d) => (body += d));
    req.on("end", () => {
      const parsed = body ? JSON.parse(body) : {};
      requests.push({ path: req.url ?? "", body: parsed });
      if ((req.url ?? "").includes("/conversation-compiler/flush")) {
        const n = flushCount++;
        if (failFlushFrom >= 0 && n >= failFlushFrom) {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "boom" }));
          return;
        }
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok" }));
    });
  });
  server.listen(0, "127.0.0.1", resolve);
});
const address = server.address();
const SERVER_URL = `http://127.0.0.1:${typeof address === "object" && address ? address.port : 0}`;
process.env.NEURALSCAPE_URL = SERVER_URL;
process.env.NEURALSCAPE_USER_ID = "shape-test";
process.env.NEURALSCAPE_TEST = "1";

const { commitClaudeCodeFlush, extractAllTurnPairs, extractClaudeCodeTurns, legacyRewindOffset } = await import(
  "../src/adapters/claude-code.js"
);
const { flushTurns } = await import("../src/core/flush.js");

afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

beforeEach(() => {
  requests = [];
  failFlushFrom = -1;
  flushCount = 0;
});

// ── Real-shape fixtures (synthetic content) ──────────────────────

const SESSION = "11111111-2222-3333-4444-555555555555";
let seq = 0;
const common = () => ({
  uuid: `uuid-${++seq}`,
  parentUuid: seq > 1 ? `uuid-${seq - 1}` : null,
  sessionId: SESSION,
  cwd: "/tmp/synthetic-project",
  timestamp: "2026-10-18T10:00:00.000Z",
  isSidechain: false,
});
const user = (text: string) => ({ type: "user", ...common(), message: { role: "user", content: text } });
const assistant = (...blocks: unknown[]) => ({ type: "assistant", ...common(), message: { role: "assistant", model: "synthetic-model", content: blocks } });
const text = (t: string) => ({ type: "text", text: t });
const thinking = { type: "thinking", thinking: "synthetic reasoning that is not conversation text" };
const toolUse = { type: "tool_use", id: "toolu_synthetic_1", name: "Read", input: { file_path: "/tmp/synthetic.txt" } };
const toolResultEntry = () => ({
  type: "user", ...common(),
  message: { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_synthetic_1", content: "TOOL-RESULT-SENTINEL synthetic file body" }] },
});
const meta = (t: string) => ({ type: "user", ...common(), isMeta: true, message: { role: "user", content: t } });
const sidechain = (t: string) => ({ type: "assistant", ...common(), isSidechain: true, message: { role: "assistant", content: [text(t)] } });
const bookkeeping = [
  { type: "summary", summary: "synthetic summary entry", leafUuid: "uuid-0" },
  { type: "file-history-snapshot", messageId: "m1", snapshot: {} },
  { type: "system", subtype: "synthetic", content: "SYSTEM-ENTRY-SENTINEL top-level content that is not conversation", ...common() },
];

const PROMPT_ONE = "Please set up the retry logic for the sync worker with exponential backoff";
const PROMPT_TWO = "Now document the retry settings in the README for operators";
const FIRST_TEXT = "I will look at the sync worker first.";
const SECOND_TEXT = "The retry logic is now in place with exponential backoff and jitter; the tests pass.";
const THIRD_TEXT = "Documented the retry settings in a new README section for operators.";

/** Relevant (user/assistant) entries: 0 user, 1 assistant, 2 tool_result user, 3 assistant, 4 meta user, 5 user, 6 assistant, 7 sidechain. */
function realSession(): unknown[] {
  return [
    bookkeeping[0],
    bookkeeping[1],
    user(PROMPT_ONE),
    assistant(thinking, text(FIRST_TEXT), toolUse),
    toolResultEntry(),
    assistant(text(SECOND_TEXT)),
    bookkeeping[2],
    meta("META-SENTINEL injected context that is not a user prompt"),
    user(PROMPT_TWO),
    assistant(text(THIRD_TEXT)),
    sidechain("SIDECHAIN-SENTINEL sub-agent text that is not the main conversation"),
  ];
}

async function transcript(entries: unknown[], jsonl = true): Promise<{ path: string; dir: string; raw: Record<string, unknown> }> {
  const dir = await mkdtemp(join(tmpdir(), "ns-shape-"));
  const path = join(dir, "transcript.jsonl");
  await writeFile(path, jsonl ? entries.map((e) => JSON.stringify(e)).join("\n") + "\n" : JSON.stringify(entries), "utf-8");
  return { path, dir, raw: { transcript_path: path, session_id: SESSION, cwd: dir } };
}

const offsetOf = async (path: string) => parseInt(await readFile(path + ".neuralscape-offset", "utf-8"), 10);
const markerOf = async (path: string) => readFile(path + ".neuralscape-offset.format", "utf-8").catch(() => null);

// ── Reading real-shape entries ───────────────────────────────────

describe("real transcript shape: turns", () => {
  it("reads text under message.content: a string for the user, text blocks for the assistant", async () => {
    const { raw } = await transcript(realSession());
    const turns = await extractClaudeCodeTurns(raw);
    expect(turns.length).toBe(2);
    expect(turns[0].userMessage).toBe(PROMPT_ONE);
    expect(turns[0].assistantResponse).toBe(`${FIRST_TEXT}\n${SECOND_TEXT}`); // one turn spans a tool loop
    expect(turns[1].userMessage).toBe(PROMPT_TWO);
    expect(turns[1].assistantResponse).toBe(THIRD_TEXT);
  });

  it("never pairs tool results, thinking, tool calls, meta entries, sidechains or bookkeeping entries", async () => {
    const { raw } = await transcript(realSession());
    const joined = JSON.stringify(await extractClaudeCodeTurns(raw));
    for (const sentinel of ["TOOL-RESULT-SENTINEL", "META-SENTINEL", "SIDECHAIN-SENTINEL", "SYSTEM-ENTRY-SENTINEL", "synthetic reasoning", "Read"]) {
      expect(joined).not.toContain(sentinel);
    }
  });

  it("a user entry that only carries a tool result does not end a turn", async () => {
    const { raw } = await transcript([user(PROMPT_ONE), assistant(text(FIRST_TEXT), toolUse), toolResultEntry(), assistant(text(SECOND_TEXT))]);
    const turns = await extractClaudeCodeTurns(raw);
    expect(turns.length).toBe(1);
    expect(turns[0].assistantResponse).toContain(SECOND_TEXT);
  });

  it("reads a transcript written as one JSON array and as a single JSONL line", async () => {
    const arrayForm = await transcript(realSession(), false);
    expect((await extractClaudeCodeTurns(arrayForm.raw)).length).toBe(2);
    const single = await transcript([user(PROMPT_ONE), assistant(text(SECOND_TEXT))].slice(0, 1));
    expect(await extractClaudeCodeTurns(single.raw)).toEqual([]); // one unanswered prompt: nothing to deliver
    const oneLine = await mkdtemp(join(tmpdir(), "ns-shape-"));
    const p = join(oneLine, "t.jsonl");
    await writeFile(p, JSON.stringify(user(PROMPT_ONE)) + "\n", "utf-8");
    expect(await extractAllTurnPairs({ transcript_path: p })).toEqual([]);
  });

  it("keeps reading the flattened shape (top-level content) unchanged", async () => {
    const { raw } = await transcript([
      { type: "user", content: "flattened user prompt with plenty of text" },
      { type: "assistant", content: [{ type: "text", text: "flattened assistant answer with plenty of text" }] },
    ]);
    const turns = await extractClaudeCodeTurns(raw);
    expect(turns.map((t) => [t.userMessage, t.assistantResponse])).toEqual([
      ["flattened user prompt with plenty of text", "flattened assistant answer with plenty of text"],
    ]);
  });

  it("extractAllTurnPairs (the SessionEnd note's input) reads the real shape from the start", async () => {
    const { raw } = await transcript(realSession());
    const pairs = await extractAllTurnPairs(raw);
    expect(pairs.map((p) => p.user)).toEqual([PROMPT_ONE, PROMPT_TWO]);
    expect(pairs[0].assistant).toContain(SECOND_TEXT);
  });
});

// ── The offset only moves past delivered turns ───────────────────

describe("real transcript shape: offsets", () => {
  it("a full delivery commits the window and the format marker", async () => {
    const { path, raw } = await transcript(realSession());
    const turns = await extractClaudeCodeTurns(raw);
    const result = await flushTurns(turns);
    expect(result.flushed).toBe(2);
    await commitClaudeCodeFlush(raw, result);
    expect(await offsetOf(path)).toBe(8); // eight user/assistant entries
    expect(await markerOf(path)).toBe("2");
    expect(await extractClaudeCodeTurns(raw)).toEqual([]);
  });

  it("a failed delivery leaves the offset where it was (and writes no marker)", async () => {
    failFlushFrom = 0;
    const { path, raw } = await transcript(realSession());
    const result = await flushTurns(await extractClaudeCodeTurns(raw));
    await commitClaudeCodeFlush(raw, result);
    await expect(readFile(path + ".neuralscape-offset", "utf-8")).rejects.toThrow();
    expect(await markerOf(path)).toBeNull();
  });

  it("a partial failure advances only past the delivered turn", async () => {
    failFlushFrom = 1;
    const { path, raw } = await transcript(realSession());
    const result = await flushTurns(await extractClaudeCodeTurns(raw));
    await commitClaudeCodeFlush(raw, result);
    expect(await offsetOf(path)).toBe(5); // just before the second prompt (0..3 first turn, 4 is the meta entry)
    failFlushFrom = -1;
    const remaining = await extractClaudeCodeTurns(raw);
    expect(remaining.map((t) => t.userMessage)).toEqual([PROMPT_TWO]);
  });

  it("a prompt that has no assistant text yet is not consumed and is delivered once answered", async () => {
    const entries = [user(PROMPT_ONE), assistant(text(SECOND_TEXT)), user(PROMPT_TWO), assistant(thinking, toolUse)];
    const { path, raw } = await transcript(entries);
    const turns = await extractClaudeCodeTurns(raw);
    expect(turns.map((t) => t.userMessage)).toEqual([PROMPT_ONE]);
    await commitClaudeCodeFlush(raw, await flushTurns(turns));
    expect(await offsetOf(path)).toBe(2); // stops before the unanswered prompt
    await writeFile(path, [...entries, assistant(text(THIRD_TEXT))].map((e) => JSON.stringify(e)).join("\n") + "\n", "utf-8");
    const next = await extractClaudeCodeTurns(raw);
    expect(next.map((t) => t.userMessage)).toEqual([PROMPT_TWO]);
    expect(next[0].assistantResponse).toBe(THIRD_TEXT);
  });
});

// ── Offsets advanced past undelivered turns by the old reader ────

describe("legacy offsets (advanced by the previous reader)", () => {
  it("rewinds a marker-less offset across turns the old reader could not see, delivers them, then stops rewinding", async () => {
    const { path, raw } = await transcript(realSession());
    await writeFile(path + ".neuralscape-offset", "8", "utf-8"); // what the old reader left: everything 'processed'
    const turns = await extractClaudeCodeTurns(raw);
    expect(turns.length).toBe(2); // the two lost turns come back
    await commitClaudeCodeFlush(raw, await flushTurns(turns));
    expect(await markerOf(path)).toBe("2");
    expect(await extractClaudeCodeTurns(raw)).toEqual([]); // marker present: no further rewind
  });

  it("keeps the offset when the old reader could read the transcript (flattened shape)", async () => {
    const { path, raw } = await transcript([
      { type: "user", content: "flattened first prompt with plenty of text" },
      { type: "assistant", content: "flattened first answer with plenty of text" },
    ]);
    await writeFile(path + ".neuralscape-offset", "2", "utf-8");
    expect(await extractClaudeCodeTurns(raw)).toEqual([]);
  });

  it("leaves the rewind in place when nothing is delivered, so the next flush tries again", async () => {
    failFlushFrom = 0;
    const { path, raw } = await transcript(realSession());
    await writeFile(path + ".neuralscape-offset", "8", "utf-8");
    await commitClaudeCodeFlush(raw, await flushTurns(await extractClaudeCodeTurns(raw)));
    expect(await offsetOf(path)).toBe(8);
    expect(await markerOf(path)).toBeNull();
    failFlushFrom = -1;
    expect((await extractClaudeCodeTurns(raw)).length).toBe(2);
  });

  it("bounds the rewind: only the most recent entries before the offset are recovered", () => {
    const entries: unknown[] = [];
    for (let i = 0; i < 300; i++) entries.push(user(`prompt ${i} with plenty of text`), assistant(text(`answer ${i} with plenty of text`)));
    const content = entries.map((e) => JSON.stringify(e)).join("\n");
    const rewound = legacyRewindOffset(content, 600);
    expect(rewound).toBeGreaterThanOrEqual(200);   // never further back than 400 entries
    expect(rewound).toBeLessThan(600);
    expect(entries[rewound]).toMatchObject({ type: "user" }); // resumes at a prompt, not mid-turn
    expect(legacyRewindOffset(content, 0)).toBe(0);
  });
});

// ── The three delivery paths, as the built hook bundles ──────────

interface Run { code: number | null; stdout: string }
function runHook(script: string, stdin: string): Promise<Run> {
  return mkdtemp(join(tmpdir(), "ns-shape-data-")).then(
    (dataDir) =>
      new Promise<Run>((resolve, reject) => {
        const child = spawn("node", [join(PLUGIN_ROOT, "scripts", script)], {
          env: { PATH: process.env.PATH ?? "", NEURALSCAPE_USER_ID: "shape-test", NEURALSCAPE_URL: SERVER_URL, CLAUDE_PLUGIN_DATA: dataDir },
          stdio: ["pipe", "pipe", "pipe"],
        });
        let stdout = "";
        child.stdout.on("data", (d) => (stdout += d));
        child.on("error", reject);
        child.on("close", (code) => resolve({ code, stdout }));
        child.stdin.write(stdin);
        child.stdin.end();
      }),
  );
}

const flushes = () => requests.filter((q) => q.path.includes("/conversation-compiler/flush"));

describe("real transcript shape: built hooks deliver", () => {
  it("Stop (session-end.js) flushes both turns and commits the offset", async () => {
    const { path, dir } = await transcript(realSession());
    const r = await runHook("session-end.js", JSON.stringify({ session_id: SESSION, transcript_path: path, cwd: dir, hook_event_name: "Stop" }));
    expect(r.code).toBe(0);
    expect(flushes().length).toBe(2);
    expect(flushes()[0].body.user_message).toBe(PROMPT_ONE);
    expect(flushes()[0].body.assistant_response).toContain(SECOND_TEXT);
    expect(flushes()[1].body.user_message).toBe(PROMPT_TWO);
    expect(await offsetOf(path)).toBe(8);
  });

  it("PreCompact (pre-compact.js) flushes both turns, commits the offset and snapshots the prompts", async () => {
    const { path, dir } = await transcript(realSession());
    const r = await runHook("pre-compact.js", JSON.stringify({ session_id: SESSION, transcript_path: path, cwd: dir, hook_event_name: "PreCompact", trigger: "auto" }));
    expect(r.code).toBe(0);
    expect(flushes().length).toBe(2);
    expect(await offsetOf(path)).toBe(8);
    const snapshot = requests.find((q) => q.path.includes("/v1/memories/raw"));
    expect(String(snapshot?.body.content)).toContain(PROMPT_TWO);
    expect(String(snapshot?.body.content)).toContain("after 2 captured turn(s)");
  });

  it("SessionEnd (session-summary.js) builds its note from the real-shape turns", async () => {
    const { path, dir } = await transcript(realSession());
    const r = await runHook("session-summary.js", JSON.stringify({ session_id: SESSION, transcript_path: path, cwd: dir, hook_event_name: "SessionEnd" }));
    expect(r.code).toBe(0);
    const checkpoint = requests.find((q) => q.path.includes("/v1/checkpoint"));
    expect(checkpoint).toBeDefined();
    expect(JSON.stringify(checkpoint?.body.session_note)).toContain("retry");
    expect(JSON.stringify(checkpoint?.body)).not.toContain("TOOL-RESULT-SENTINEL");
  });
});
