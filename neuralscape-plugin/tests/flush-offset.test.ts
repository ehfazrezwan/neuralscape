/**
 * Audit 27 #34(b) — the Stop-hook flush must commit the transcript offset
 * ONLY after confirmed delivery: all-failed flushes leave the cursor where
 * it was, and a partial failure advances only past the successfully-POSTed
 * prefix, so the next session re-flushes exactly the undelivered turns.
 */

import { createServer, type Server } from "node:http";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";

// ── Fixture server (must be up BEFORE utils.js resolves its base URL) ──

let server: Server;
let received: string[] = [];
/** Fail the Nth request (0-based) and everything after; -1 = never fail. */
let failFrom = -1;

await new Promise<void>((resolve) => {
  server = createServer((req, res) => {
    let body = "";
    req.on("data", (d) => (body += d));
    req.on("end", () => {
      const n = received.length;
      received.push(body);
      if (failFrom >= 0 && n >= failFrom) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "boom" }));
        return;
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok" }));
    });
  });
  server.listen(0, "127.0.0.1", resolve);
});
const address = server.address();
process.env.NEURALSCAPE_URL = `http://127.0.0.1:${typeof address === "object" && address ? address.port : 0}`;
process.env.NEURALSCAPE_USER_ID = "flush-test";
process.env.NEURALSCAPE_TEST = "1";

const flushMod = await import("../src/core/flush.js");
const { flushTurns } = flushMod;
const ccMod = await import("../src/adapters/claude-code.js");
const { commitClaudeCodeFlush, extractClaudeCodeTurns } = ccMod;
import type { ConversationTurn } from "../src/core/types.js";

afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

beforeEach(() => {
  received = [];
  failFrom = -1;
});

function turn(i: number): ConversationTurn {
  return {
    userMessage: `user message number ${i} with enough substance to keep`,
    assistantResponse: `assistant response number ${i} with enough substance to keep`,
    sessionId: "s-flush",
    channel: "claude-code",
    timestamp: "2026-07-05T10:00:00Z",
    userId: "flush-test",
  };
}

// ── flushTurns result contract ───────────────────────────────────

describe("flushTurns delivery accounting", () => {
  it("reports full success", async () => {
    const result = await flushTurns([turn(0), turn(1)]);
    expect(result.flushed).toBe(2);
    expect(result.firstFailedIndex).toBeNull();
    expect(received.length).toBe(2);
  });

  it("stops at the first failed POST and reports its index", async () => {
    failFrom = 1; // first POST ok, second fails
    const result = await flushTurns([turn(0), turn(1), turn(2)]);
    expect(result.flushed).toBe(1);
    expect(result.firstFailedIndex).toBe(1);
    // No POST attempted past the failure — the tail must stay re-flushable.
    expect(received.length).toBe(2);
  });

  it("reports index 0 when every POST fails", async () => {
    failFrom = 0;
    const result = await flushTurns([turn(0), turn(1)]);
    expect(result.flushed).toBe(0);
    expect(result.firstFailedIndex).toBe(0);
  });

  it("counts noise-skipped turns as delivered (nothing to re-flush)", async () => {
    const noisy: ConversationTurn = { ...turn(0), userMessage: "ping", assistantResponse: "pong" };
    const result = await flushTurns([noisy, turn(1)]);
    expect(result.flushed).toBe(1);
    expect(result.firstFailedIndex).toBeNull();
    expect(received.length).toBe(1);
  });
});

// ── offset commit semantics ──────────────────────────────────────

const MESSAGES = [
  { type: "user", content: "turn zero user prompt with plenty of text" },
  { type: "assistant", content: "turn zero assistant answer with plenty of text" },
  { type: "user", content: "turn one user prompt with plenty of text" },
  { type: "assistant", content: "turn one assistant answer with plenty of text" },
  { type: "user", content: "turn two user prompt with plenty of text" },
  { type: "assistant", content: "turn two assistant answer with plenty of text" },
];

async function makeTranscript(): Promise<{ path: string; raw: Record<string, unknown> }> {
  const dir = await mkdtemp(join(tmpdir(), "ns-flush-"));
  const path = join(dir, "transcript.jsonl");
  await writeFile(path, MESSAGES.map((m) => JSON.stringify(m)).join("\n") + "\n", "utf-8");
  return { path, raw: { transcript_path: path, session_id: "s-offsets", cwd: dir } };
}

async function readOffset(path: string): Promise<number | null> {
  try {
    return parseInt(await readFile(path + ".neuralscape-offset", "utf-8"), 10);
  } catch {
    return null;
  }
}

describe("commitClaudeCodeFlush offset semantics", () => {
  it("full success commits the full offset", async () => {
    const { path, raw } = await makeTranscript();
    const turns = await extractClaudeCodeTurns(raw);
    expect(turns.length).toBe(3);
    const result = await flushTurns(turns);
    await commitClaudeCodeFlush(raw, result);
    expect(await readOffset(path)).toBe(6);
  });

  it("all POSTs failed → offset is NOT advanced (audit 27 #34b)", async () => {
    failFrom = 0;
    const { path, raw } = await makeTranscript();
    const turns = await extractClaudeCodeTurns(raw);
    const result = await flushTurns(turns);
    await commitClaudeCodeFlush(raw, result);
    expect(await readOffset(path)).toBeNull(); // cursor stays at its prior position
  });

  it("partial failure advances only past the successfully-POSTed prefix", async () => {
    failFrom = 2; // turns 0 and 1 delivered, turn 2 failed
    const { path, raw } = await makeTranscript();
    const turns = await extractClaudeCodeTurns(raw);
    const result = await flushTurns(turns);
    await commitClaudeCodeFlush(raw, result);
    expect(await readOffset(path)).toBe(4); // past turn 1's messages only

    // The next extraction resumes exactly at the undelivered turn.
    failFrom = -1;
    received = [];
    const remaining = await extractClaudeCodeTurns(raw);
    expect(remaining.length).toBe(1);
    expect(remaining[0].userMessage).toContain("turn two");
    const second = await flushTurns(remaining);
    await commitClaudeCodeFlush(raw, second);
    expect(await readOffset(path)).toBe(6);
  });
});
