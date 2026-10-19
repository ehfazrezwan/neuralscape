/**
 * Real Claude Code transcript shape: entries are `{type, message: {role, content}, ...}` with `content` a string
 * (user prompts) or a block list (assistant text, tool calls, user tool results). Capture used to read only a
 * top-level `content`, so every real turn looked empty, was dropped as noise, and the flush offset still advanced:
 * the Stop flush, the PreCompact flush and the SessionEnd note delivered nothing.
 *
 * The fixtures below are synthetic: they reproduce the key structure only (never real transcript text).
 */

import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readdir, readFile, writeFile } from "node:fs/promises";
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
/** Refuse the Nth flush request (0-based) with this HTTP status; null = never. */
let rejectFlush: { at: number | "all"; status: number } | null = null;

await new Promise<void>((resolve) => {
  server = createServer((req, res) => {
    let body = "";
    req.on("data", (d) => (body += d));
    req.on("end", () => {
      const parsed = body ? JSON.parse(body) : {};
      requests.push({ path: req.url ?? "", body: parsed });
      if ((req.url ?? "").includes("/conversation-compiler/flush")) {
        const n = flushCount++;
        if (rejectFlush !== null && (rejectFlush.at === "all" || n === rejectFlush.at)) {
          res.writeHead(rejectFlush.status, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "refused" }));
          return;
        }
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
// Direct delivery: the offset tests below need a failed POST to be a failed delivery (a durable outbox, where one
// exists, would accept the turn and count it delivered).
process.env.NEURALSCAPE_DURABLE_OUTBOX = "false";

const { commitClaudeCodeFlush, extractAllTurnPairs, extractClaudeCodeTurns } = await import(
  "../src/adapters/claude-code.js"
);
const { boundText, flushTurns } = await import("../src/core/flush.js");

afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

beforeEach(() => {
  requests = [];
  failFlushFrom = -1;
  flushCount = 0;
  rejectFlush = null;
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
const compactSummary = (t: string) => ({
  type: "user", ...common(), isCompactSummary: true, isVisibleInTranscriptOnly: true, message: { role: "user", content: t },
});
const localCommand = (t: string) => ({ type: "user", ...common(), message: { role: "user", content: t } });
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
  it("a full delivery commits the window", async () => {
    const { path, raw } = await transcript(realSession());
    const turns = await extractClaudeCodeTurns(raw);
    const result = await flushTurns(turns);
    expect(result.flushed).toBe(2);
    await commitClaudeCodeFlush(raw, result);
    expect(await offsetOf(path)).toBe(8); // eight user/assistant entries
    expect(await extractClaudeCodeTurns(raw)).toEqual([]);
  });

  it("a failed delivery leaves the offset where it was", async () => {
    failFlushFrom = 0;
    const { path, raw } = await transcript(realSession());
    const result = await flushTurns(await extractClaudeCodeTurns(raw));
    await commitClaudeCodeFlush(raw, result);
    await expect(readFile(path + ".neuralscape-offset", "utf-8")).rejects.toThrow();
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

describe("an offset is used exactly as it was written", () => {
  it("a cursor at the end of a transcript the previous reader skipped delivers nothing: lost turns stay lost, nothing is rewound", async () => {
    const { path, raw } = await transcript(realSession());
    await writeFile(path + ".neuralscape-offset", "8", "utf-8");
    expect(await extractClaudeCodeTurns(raw)).toEqual([]);
  });

  it("a cursor inside the transcript resumes there, never earlier", async () => {
    const { path, raw } = await transcript(realSession());
    await writeFile(path + ".neuralscape-offset", "5", "utf-8"); // just before the second prompt
    expect((await extractClaudeCodeTurns(raw)).map((t) => t.userMessage)).toEqual([PROMPT_TWO]);
  });

  it("a flattened turn already delivered is never replayed, with or without real-shape entries around it", async () => {
    const entries = [...FLAT(1), user(PROMPT_ONE), assistant(text(SECOND_TEXT)), ...FLAT(2)];
    const { path, raw } = await transcript(entries);
    await writeFile(path + ".neuralscape-offset", String(entries.length), "utf-8");
    expect(await extractClaudeCodeTurns(raw)).toEqual([]);
    await append(path, [{ type: "user", isMeta: true, content: "ignored entry appended after the cursor" }]);
    expect(await extractClaudeCodeTurns(raw)).toEqual([]);
  });

  it("an offset file from an older plugin version (with or without extra text) is read as its number", async () => {
    const { path, raw } = await transcript(realSession());
    await writeFile(path + ".neuralscape-offset", "5 v2", "utf-8");
    expect((await extractClaudeCodeTurns(raw)).map((t) => t.userMessage)).toEqual([PROMPT_TWO]);
  });
});

// ── The three delivery paths, as the built hook bundles ──────────

interface Run { code: number | null; stdout: string }
function runHook(script: string, stdin: string): Promise<Run> {
  return mkdtemp(join(tmpdir(), "ns-shape-data-")).then(
    (dataDir) =>
      new Promise<Run>((resolve, reject) => {
        const child = spawn("node", [join(PLUGIN_ROOT, "scripts", script)], {
          env: { PATH: process.env.PATH ?? "", NEURALSCAPE_USER_ID: "shape-test", NEURALSCAPE_URL: SERVER_URL, CLAUDE_PLUGIN_DATA: dataDir, NEURALSCAPE_DURABLE_OUTBOX: "false" },
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


// ── Compaction summaries and local-command wrappers are not what the user said ──

const SUMMARY_TEXT = "COMPACT-SUMMARY-SENTINEL This session is being continued from a previous conversation that ran out of context.";

describe("entries that are not the user's conversation", () => {
  it("a compaction summary followed by assistant text is never delivered as a prompt", async () => {
    const { raw } = await transcript([
      compactSummary(SUMMARY_TEXT),
      assistant(text("ANSWER-AFTER-COMPACT-SENTINEL continuing the task after the compaction has happened")),
      user(PROMPT_TWO),
      assistant(text(THIRD_TEXT)),
    ]);
    const turns = await extractClaudeCodeTurns(raw);
    expect(turns.map((t) => t.userMessage)).toEqual([PROMPT_TWO]);
    expect(JSON.stringify(turns)).not.toContain("COMPACT-SUMMARY-SENTINEL");
    expect(JSON.stringify(turns)).not.toContain("ANSWER-AFTER-COMPACT-SENTINEL"); // unpaired continuation, dropped
  });

  it.each([["isCompactSummary"], ["isVisibleInTranscriptOnly"]])("%s alone is enough to exclude an entry", async (flag) => {
    const entry = { type: "user", ...common(), [flag]: true, message: { role: "user", content: SUMMARY_TEXT } };
    const { raw } = await transcript([entry, assistant(text(SECOND_TEXT)), user(PROMPT_ONE), assistant(text(THIRD_TEXT))]);
    expect((await extractClaudeCodeTurns(raw)).map((t) => t.userMessage)).toEqual([PROMPT_ONE]);
  });

  it("a summary in the middle of a session neither ends the turn before it nor starts one", async () => {
    const { raw } = await transcript([user(PROMPT_ONE), assistant(text(FIRST_TEXT)), compactSummary(SUMMARY_TEXT), assistant(text(SECOND_TEXT))]);
    const turns = await extractClaudeCodeTurns(raw);
    expect(turns.length).toBe(1);
    expect(turns[0].assistantResponse).toBe(`${FIRST_TEXT}\n${SECOND_TEXT}`);
    expect(JSON.stringify(turns)).not.toContain("COMPACT-SUMMARY-SENTINEL");
  });

  it("local slash-command wrappers are not prompts", async () => {
    const wrappers = [
      "<command-name>/compact</command-name>\n<command-message>compact</command-message>\n<command-args></command-args>",
      "<local-command-stdout>LOCAL-COMMAND-SENTINEL synthetic output</local-command-stdout>",
      "<local-command-caveat>Caveat: synthetic</local-command-caveat>",
    ];
    for (const wrapper of wrappers) {
      const { raw } = await transcript([localCommand(wrapper), assistant(text(SECOND_TEXT)), user(PROMPT_ONE), assistant(text(THIRD_TEXT))]);
      expect((await extractClaudeCodeTurns(raw)).map((t) => t.userMessage)).toEqual([PROMPT_ONE]);
    }
    const { raw } = await transcript([user("how do I use <command-name> in a sentence like this one?"), assistant(text(SECOND_TEXT))]);
    expect((await extractClaudeCodeTurns(raw)).length).toBe(1); // only a wrapper at the start of the entry is dropped
  });

  it("the SessionEnd pair reader excludes them too", async () => {
    const { raw } = await transcript([compactSummary(SUMMARY_TEXT), assistant(text(SECOND_TEXT)), user(PROMPT_ONE), assistant(text(THIRD_TEXT))]);
    expect((await extractAllTurnPairs(raw)).map((p) => p.user)).toEqual([PROMPT_ONE]);
  });

  it("the Stop hook sends the real turn and never the summary", async () => {
    const { path, dir } = await transcript([
      compactSummary(SUMMARY_TEXT),
      assistant(text("ANSWER-AFTER-COMPACT-SENTINEL continuing the task after the compaction has happened")),
      user(PROMPT_TWO),
      assistant(text(THIRD_TEXT)),
    ]);
    const r = await runHook("session-end.js", JSON.stringify({ session_id: SESSION, transcript_path: path, cwd: dir, hook_event_name: "Stop" }));
    expect(r.code).toBe(0);
    expect(flushes().map((q) => q.body.user_message)).toEqual([PROMPT_TWO]);
    expect(JSON.stringify(requests)).not.toContain("COMPACT-SUMMARY-SENTINEL");
  });
});

// ── The offset and its format tag are one write ──────────────────

describe("offset file: one atomic write", () => {
  it("stores the offset alone and leaves no temporary or sidecar file", async () => {
    const { path, dir, raw } = await transcript(realSession());
    await commitClaudeCodeFlush(raw, await flushTurns(await extractClaudeCodeTurns(raw)));
    expect((await readFile(path + ".neuralscape-offset", "utf-8")).trim()).toBe("8");
    expect((await readdir(dir)).sort()).toEqual(["transcript.jsonl", "transcript.jsonl.neuralscape-offset"]);
  });

  it("an older reader of the offset file still gets the number", async () => {
    const { path, raw } = await transcript(realSession());
    await commitClaudeCodeFlush(raw, await flushTurns(await extractClaudeCodeTurns(raw)));
    expect(parseInt(await readFile(path + ".neuralscape-offset", "utf-8"), 10)).toBe(8);
  });

  it("a failed write keeps the previous offset, writes no half state and removes its temporary file", async () => {
    const { path, dir, raw } = await transcript(realSession());
    const turns = await extractClaudeCodeTurns(raw);
    const result = await flushTurns(turns);
    await mkdir(path + ".neuralscape-offset"); // a directory where the offset file belongs: the rename cannot succeed
    await commitClaudeCodeFlush(raw, result);
    expect((await readdir(dir)).sort()).toEqual(["transcript.jsonl", "transcript.jsonl.neuralscape-offset"]); // no .tmp left
  });

  it("a delivered flush is not replayed by the next one", async () => {
    const { raw } = await transcript(realSession());
    await commitClaudeCodeFlush(raw, await flushTurns(await extractClaudeCodeTurns(raw)));
    const before = requests.length;
    expect(await extractClaudeCodeTurns(raw)).toEqual([]);
    expect(requests.length).toBe(before);
  });
});

// ── A compaction in the middle of a turn, and a transcript that mixes both shapes ──

const FLAT = (i: number) => [
  { type: "user", content: `flattened prompt number ${i} with plenty of text to keep` },
  { type: "assistant", content: `flattened answer number ${i} with plenty of text to keep` },
];
const preCompact = (path: string, dir: string) =>
  runHook("pre-compact.js", JSON.stringify({ session_id: SESSION, transcript_path: path, cwd: dir, hook_event_name: "PreCompact", trigger: "auto" }));
const stop = (path: string, dir: string) =>
  runHook("session-end.js", JSON.stringify({ session_id: SESSION, transcript_path: path, cwd: dir, hook_event_name: "Stop" }));
const append = (path: string, entries: unknown[]) =>
  readFile(path, "utf-8").then((old) => writeFile(path, old + entries.map((e) => JSON.stringify(e)).join("\n") + "\n", "utf-8"));

describe("PreCompact in the middle of a turn", () => {
  const MID = (): unknown[] => [user(PROMPT_ONE), assistant(text(FIRST_TEXT), toolUse), toolResultEntry()];

  it("PreCompact → continuation → Stop delivers the whole turn exactly once", async () => {
    const { path, dir } = await transcript(MID());
    await preCompact(path, dir);
    expect(flushes().length).toBe(0); // the running turn is withheld
    await expect(readFile(path + ".neuralscape-offset", "utf-8")).rejects.toThrow(); // and nothing is committed past its prompt
    await append(path, [assistant(text(SECOND_TEXT))]);
    await stop(path, dir);
    expect(flushes().length).toBe(1);
    expect(flushes()[0].body.user_message).toBe(PROMPT_ONE);
    expect(flushes()[0].body.assistant_response).toBe(`${FIRST_TEXT}\n${SECOND_TEXT}`);
    await stop(path, dir);
    expect(flushes().length).toBe(1); // exactly once
  });

  it("delivers the finished turns before it and keeps the offset at the running turn's prompt", async () => {
    const entries = [user(PROMPT_ONE), assistant(text(SECOND_TEXT)), user(PROMPT_TWO), assistant(text(FIRST_TEXT), toolUse), toolResultEntry()];
    const { path, dir } = await transcript(entries);
    await preCompact(path, dir);
    expect(flushes().map((q) => q.body.user_message)).toEqual([PROMPT_ONE]);
    expect(await offsetOf(path)).toBe(2);
    await append(path, [assistant(text(THIRD_TEXT))]);
    await stop(path, dir);
    expect(flushes().map((q) => q.body.user_message)).toEqual([PROMPT_ONE, PROMPT_TWO]);
    expect(flushes()[1].body.assistant_response).toBe(`${FIRST_TEXT}\n${THIRD_TEXT}`);
  });

  it("a turn that ends on an assistant entry calling a tool (result not written yet) is withheld too", async () => {
    const { raw } = await transcript([user(PROMPT_ONE), assistant(text(FIRST_TEXT), toolUse)]);
    expect(await extractClaudeCodeTurns({ ...raw, hook_event_name: "PreCompact" })).toEqual([]);
    const flagged = await transcript([user(PROMPT_ONE), { ...assistant(text(FIRST_TEXT)), message: { role: "assistant", stop_reason: "tool_use", content: [text(FIRST_TEXT)] } }]);
    expect(await extractClaudeCodeTurns({ ...flagged.raw, hook_event_name: "PreCompact" })).toEqual([]);
  });

  it("a finished turn is still delivered by PreCompact (the snapshot keeps its prompt) and not again by Stop", async () => {
    const { path, dir } = await transcript([user(PROMPT_ONE), assistant(text(FIRST_TEXT), toolUse), toolResultEntry(), assistant(text(SECOND_TEXT))]);
    await preCompact(path, dir);
    expect(flushes().length).toBe(1);
    expect(String(requests.find((q) => q.path.includes("/v1/memories/raw"))?.body.content)).toContain(PROMPT_ONE);
    await stop(path, dir);
    expect(flushes().length).toBe(1);
  });

  it("the Stop event always delivers what the transcript holds (the turn has ended)", async () => {
    const { raw } = await transcript(MID());
    expect((await extractClaudeCodeTurns({ ...raw, hook_event_name: "Stop" })).length).toBe(1);
  });
});

// ── The payload is bounded to what the flush route accepts, and a refused turn cannot block the rest ──

describe("flush payload bounds and permanent rejections", () => {
  const LIMIT = 50000;

  it("boundText leaves text at the limit untouched and marks text over it", () => {
    const atLimit = "a".repeat(LIMIT);
    expect(boundText(atLimit)).toBe(atLimit);
    const over = "b".repeat(LIMIT + 1);
    const bounded = boundText(over);
    expect(bounded.length).toBeLessThanOrEqual(LIMIT);
    expect(bounded).toContain("[truncated by the plugin:");
    expect(bounded.startsWith("b".repeat(1000))).toBe(true);
    const marker = /\[truncated by the plugin: (\d+) more characters\]/.exec(bounded);
    expect(marker).not.toBeNull();
    expect(bounded.indexOf("\n[truncated") + Number(marker![1])).toBe(over.length); // the marker counts exactly what was left out
  });

  it("never splits a surrogate pair at the cut and stays within the limit for any digit count", () => {
    for (const filler of ["😀", "a😀", "ab😀"]) {
      const text = filler.repeat(Math.ceil((LIMIT * 2) / filler.length));
      const bounded = boundText(text);
      expect(bounded.length).toBeLessThanOrEqual(LIMIT);
      expect(bounded.slice(0, bounded.indexOf("\n[truncated"))).not.toMatch(/[\ud800-\udbff]$/);
    }
    expect(boundText("x".repeat(LIMIT * 20)).length).toBeLessThanOrEqual(LIMIT);
  });

  it("two assistant entries of 30,000 characters make one turn that the service accepts (bounded, marked)", async () => {
    const big = (c: string) => c.repeat(30000);
    const { raw } = await transcript([user(PROMPT_ONE), assistant(text(big("x"))), assistant(text(big("y")))]);
    const turns = await extractClaudeCodeTurns(raw);
    expect(turns[0].assistantResponse.length).toBe(60001); // joined by a newline: over the limit before flushing
    const result = await flushTurns(turns);
    expect(result.firstFailedIndex).toBeNull();
    const sent = flushes()[0].body;
    expect(sent.assistant_response.length).toBeLessThanOrEqual(LIMIT);
    expect(sent.assistant_response).toContain("[truncated by the plugin:");
    expect(sent.user_message).toBe(PROMPT_ONE);
  });

  it("a user message over the limit is bounded too, and text at the limit goes unchanged", async () => {
    const turns = [{ userMessage: "u".repeat(LIMIT + 10), assistantResponse: "a".repeat(LIMIT), sessionId: "s", channel: "claude-code", timestamp: "t", userId: "shape-test" }];
    await flushTurns(turns);
    const sent = flushes()[0].body;
    expect(sent.user_message.length).toBeLessThanOrEqual(LIMIT);
    expect(sent.user_message).toContain("[truncated by the plugin:");
    expect(sent.assistant_response).toBe("a".repeat(LIMIT));
  });

  it.each([413, 422])("a turn the service refuses with HTTP %i is skipped and the later turns are still delivered", async (status) => {
    rejectFlush = { at: 1, status };
    const { path, raw } = await transcript([...FLAT(1), ...FLAT(2), ...FLAT(3)]);
    const turns = await extractClaudeCodeTurns(raw);
    const result = await flushTurns(turns);
    expect(result.firstFailedIndex).toBeNull();
    expect(flushes().map((q) => q.body.user_message)).toEqual([
      "flattened prompt number 1 with plenty of text to keep",
      "flattened prompt number 2 with plenty of text to keep",
      "flattened prompt number 3 with plenty of text to keep",
    ]);
    await commitClaudeCodeFlush(raw, result);
    expect(await offsetOf(path)).toBe(6); // the cursor passed the refused turn
    expect(await extractClaudeCodeTurns(raw)).toEqual([]);
  });

  it.each([400, 401, 403, 404, 408, 429, 500, 503])("HTTP %i is not a permanent rejection of the turn: delivery stops there and the cursor waits", async (status) => {
    rejectFlush = { at: 1, status };
    const { path, raw } = await transcript([...FLAT(1), ...FLAT(2), ...FLAT(3)]);
    const result = await flushTurns(await extractClaudeCodeTurns(raw));
    expect(result.firstFailedIndex).toBe(1);
    expect(flushes().length).toBe(2); // turn 3 was not attempted
    await commitClaudeCodeFlush(raw, result);
    expect(await offsetOf(path)).toBe(2);
  });

  it.each([413, 422])("every turn refused with HTTP %i: delivery stops after the second and the cursor stays", async (status) => {
    rejectFlush = { at: "all", status };
    const { path, raw } = await transcript([...FLAT(1), ...FLAT(2), ...FLAT(3), ...FLAT(4)]);
    const result = await flushTurns(await extractClaudeCodeTurns(raw));
    expect(result.firstFailedIndex).toBe(0);
    expect(flushes().length).toBe(2); // turns 3 and 4 were not attempted
    await commitClaudeCodeFlush(raw, result);
    expect(await offsetOf(path).catch(() => 0)).toBe(0); // no cursor was written
    expect((await extractClaudeCodeTurns(raw)).length).toBe(4); // every turn is still pending
  });

  it("one refused turn between good ones is skipped, and a later refusal starts a new count", async () => {
    rejectFlush = { at: 2, status: 422 };
    const { path, raw } = await transcript([...FLAT(1), ...FLAT(2), ...FLAT(3), ...FLAT(4)]);
    const result = await flushTurns(await extractClaudeCodeTurns(raw));
    expect(result.firstFailedIndex).toBeNull();
    expect(flushes().length).toBe(4);
    await commitClaudeCodeFlush(raw, result);
    expect(await offsetOf(path)).toBe(8);
  });

  it("two refusals in a row after delivered turns leave the cursor at the first refused turn", async () => {
    rejectFlush = { at: "all", status: 413 };
    const { path, raw } = await transcript([...FLAT(1), ...FLAT(2), ...FLAT(3)]);
    const turns = await extractClaudeCodeTurns(raw);
    rejectFlush = null;
    await flushTurns(turns.slice(0, 1));
    rejectFlush = { at: "all", status: 413 };
    const result = await flushTurns(turns.slice(1));
    expect(result.firstFailedIndex).toBe(0);
    expect(flushes().length).toBe(3);
  });
});
