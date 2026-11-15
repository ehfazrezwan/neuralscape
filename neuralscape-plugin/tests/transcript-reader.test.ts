/**
 * The lag wait parses the transcript once and then only what was appended
 * (counted through the adapter's transcriptStats seam), whatever the polling
 * count; half-written lines, shrinking files and JSON-document transcripts
 * are handled.
 */

import { appendFileSync, mkdtempSync, truncateSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeEach, describe, expect, it } from "vitest";

process.env.NEURALSCAPE_USER_ID = "reader-test";
const { extractClaudeCodeTurns, transcriptStats } = await import("../src/adapters/claude-code.js");

const user = (text: string) => JSON.stringify({ type: "user", message: { role: "user", content: text } }) + "\n";
const assistant = (text: string) =>
  JSON.stringify({ type: "assistant", message: { role: "assistant", content: [{ type: "text", text }] } }) + "\n";

function newTranscript(): string {
  return join(mkdtempSync(join(tmpdir(), "ns-reader-")), "t.jsonl");
}

const stopEvent = (path: string, last: string) => ({
  session_id: "s",
  transcript_path: path,
  hook_event_name: "Stop",
  last_assistant_message: last,
});

beforeEach(() => {
  process.env.NEURALSCAPE_TRANSCRIPT_LAG_WAIT_MS = "1500";
  transcriptStats.linesParsed = 0;
  transcriptStats.fullReads = 0;
  transcriptStats.tailReads = 0;
});

describe("lag wait on a large transcript", () => {
  it("parses the transcript once, then only the appended tail, however many polls it takes", async () => {
    const path = newTranscript();
    const PAIRS = 1500; // 3000 lines
    let body = "";
    for (let i = 0; i < PAIRS; i++) body += user(`question ${i}`) + assistant(`answer number ${i} with some words`);
    body += user("the last question");
    writeFileSync(path, body);
    const FINAL = "The final answer arrives late, after about a dozen polls.";
    setTimeout(() => appendFileSync(path, assistant(FINAL)), 1200); // ~12 polls of 100 ms before it lands

    const turns = await extractClaudeCodeTurns(stopEvent(path, FINAL));

    expect(turns.at(-1)?.assistantResponse).toBe(FINAL);
    expect(turns.length).toBe(PAIRS + 1);
    expect(transcriptStats.fullReads).toBe(1);
    expect(transcriptStats.tailReads).toBeGreaterThanOrEqual(1);
    // Before: ~12 polls x 3000 lines re-parsed. Now: the file once, plus the one appended line.
    expect(transcriptStats.linesParsed).toBeLessThanOrEqual(PAIRS * 2 + 1 + 1 + 2);
  });

  it("a poll that sees a half-written line waits for the rest of it", async () => {
    const path = newTranscript();
    writeFileSync(path, user("question") );
    const FINAL = "A final answer that is written in two pieces by the writer.";
    const line = assistant(FINAL);
    const cut = Math.floor(line.length / 2);
    setTimeout(() => appendFileSync(path, line.slice(0, cut)), 150);
    setTimeout(() => appendFileSync(path, line.slice(cut)), 450);
    const turns = await extractClaudeCodeTurns(stopEvent(path, FINAL));
    expect(turns).toHaveLength(1);
    expect(turns[0].assistantResponse).toBe(FINAL);
  });

  it("a multi-byte character split across two appends is read intact", async () => {
    const path = newTranscript();
    writeFileSync(path, user("question"));
    const FINAL = "Résumé of the plan: ünïcödé — and 日本語 together.";
    const buf = Buffer.from(assistant(FINAL), "utf-8");
    const cut = buf.indexOf(Buffer.from("日")) + 1; // inside a 3-byte char
    setTimeout(() => appendFileSync(path, buf.subarray(0, cut)), 150);
    setTimeout(() => appendFileSync(path, buf.subarray(cut)), 450);
    const turns = await extractClaudeCodeTurns(stopEvent(path, FINAL));
    expect(turns[0].assistantResponse).toBe(FINAL);
  });

  it("a transcript that shrank (rewritten) is read in full again", async () => {
    const path = newTranscript();
    writeFileSync(path, user("old question") + assistant("old answer that is long enough") + user("pending"));
    const FINAL = "The rewritten transcript's final answer, long enough.";
    setTimeout(() => {
      truncateSync(path, 0);
      writeFileSync(path, user("pending") + assistant(FINAL));
    }, 250);
    const turns = await extractClaudeCodeTurns(stopEvent(path, FINAL));
    expect(turns.at(-1)?.assistantResponse).toBe(FINAL);
    expect(transcriptStats.fullReads).toBeGreaterThanOrEqual(2);
  });

  it("a transcript stored as one JSON document is re-read in full (it cannot be tailed)", async () => {
    const path = newTranscript();
    const first = [{ type: "user", message: { role: "user", content: "question" } }];
    writeFileSync(path, JSON.stringify(first));
    const FINAL = "The final answer inside a JSON document transcript.";
    setTimeout(
      () => writeFileSync(path, JSON.stringify([...first, { type: "assistant", message: { role: "assistant", content: [{ type: "text", text: FINAL }] } }])),
      250,
    );
    const turns = await extractClaudeCodeTurns(stopEvent(path, FINAL));
    expect(turns[0].assistantResponse).toBe(FINAL);
    expect(transcriptStats.fullReads).toBeGreaterThanOrEqual(2);
    expect(transcriptStats.tailReads).toBe(0);
  });

  it("an unchanged file is not re-read or re-parsed while waiting", async () => {
    const path = newTranscript();
    writeFileSync(path, user("question") + assistant("short"));
    process.env.NEURALSCAPE_TRANSCRIPT_LAG_WAIT_MS = "400";
    const turns = await extractClaudeCodeTurns(stopEvent(path, "an answer that never lands in the transcript"));
    expect(turns).toEqual([]);
    expect(transcriptStats.fullReads).toBe(1);
    expect(transcriptStats.tailReads).toBe(0);
    expect(transcriptStats.linesParsed).toBe(2);
  });
});
