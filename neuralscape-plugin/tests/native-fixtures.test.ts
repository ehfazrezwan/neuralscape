/**
 * Native-client fixtures, checked against reviewer-owned expectations.
 *
 * The inputs are sanitized synthetic native events and transcripts (tests/native-fixtures).  Each
 * is parsed by the plugin's own code; a content-free capture receipt is built from the result and
 * compared, together with the retained text, with the expected files.  The expectations describe
 * what a native event must yield; they are not derived from the implementation.
 */

import { createHash } from "node:crypto";
import { mkdtemp, readFile, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";

const ROOT = new URL("./native-fixtures/", import.meta.url).pathname;
process.env.NEURALSCAPE_TEST = "1";
process.env.NEURALSCAPE_USER_ID = "fixture-user";
process.env.CLAUDE_PLUGIN_DATA = await mkdtemp(join(tmpdir(), "ns-native-fixtures-"));

const hook = await import("../src/hooks/post-tool-use.js");
const adapter = await import("../src/adapters/claude-code.js");
const utils = await import("../src/utils.js");
const { shouldCapture, buildObservationRow } = hook;

type Json = Record<string, any>;
const readJson = async (rel: string): Promise<Json> => JSON.parse(await readFile(join(ROOT, rel), "utf-8"));
const expectedEvents = (await readJson("expected/events.json")).cases as Json;
const expectedTranscripts = (await readJson("expected/transcripts.json")).cases as Json;
const eventNames = (await readdir(join(ROOT, "events"))).map((f) => f.replace(/\.json$/, "")).sort();
const transcriptNames = (await readdir(join(ROOT, "transcripts"))).map((f) => f.replace(/\.jsonl$/, "")).sort();

// ── the fixtures themselves ─────────────────────────────────────────────────

describe("fixture hygiene", () => {
  it("every fixture and every expectation is listed in the manifest with its SHA-256", async () => {
    const manifest = await readJson("MANIFEST.json");
    const found: Record<string, string> = {};
    for (const dir of ["events", "transcripts", "expected"]) {
      for (const name of await readdir(join(ROOT, dir))) {
        found[`${dir}/${name}`] = createHash("sha256").update(await readFile(join(ROOT, dir, name))).digest("hex");
      }
    }
    found["README.md"] = createHash("sha256").update(await readFile(join(ROOT, "README.md"))).digest("hex");
    expect(manifest.files).toEqual(Object.fromEntries(Object.entries(found).sort()));
  });

  it("every event fixture has a provenance tag and no real data", async () => {
    for (const name of eventNames) {
      const fixture = await readJson(`events/${name}.json`);
      expect(fixture.provenance.kind).toBe("synthetic-sanitized");
      expect(fixture.provenance.contains_real_data).toBe(false);
      expect(typeof fixture.provenance.client_version_tag).toBe("string");
    }
  });

  it("no fixture holds a private-looking host, address, e-mail or credential", async () => {
    const forbidden = [
      /\b(?:\d{1,3}\.){3}\d{1,3}\b/,                 // an IPv4 address
      /[A-Za-z0-9._%+-]+@(?!example\.invalid)[A-Za-z0-9.-]+\.[A-Za-z]{2,}/, // a real e-mail
      /\bsk-[A-Za-z0-9]{16,}/, /\bAKIA[0-9A-Z]{12,}/, /BEGIN [A-Z ]*PRIVATE KEY/, /ghp_[A-Za-z0-9]{20,}/,
      /\/Users\/(?!user\b)/, /\/home\/(?!user\b)/,    // a real home directory
      /https?:\/\/(?!example\.invalid)[a-z0-9.-]+\.[a-z]{2,}/i, // a real host
    ];
    for (const dir of ["events", "transcripts", "expected"]) {
      for (const name of await readdir(join(ROOT, dir))) {
        const text = await readFile(join(ROOT, dir, name), "utf-8");
        for (const re of forbidden) expect(text, `${dir}/${name} ${re}`).not.toMatch(re);
      }
    }
  });

  it("every fixture has an expectation and every expectation a fixture", () => {
    expect(Object.keys(expectedEvents).sort()).toEqual(eventNames);
    expect(Object.keys(expectedTranscripts).sort()).toEqual(transcriptNames);
  });
});

// ── PostToolUse events → capture receipts ───────────────────────────────────

function receiptOf(raw: string, row: Json | null): Json {
  if (row === null) return { captured: false };
  const output = String(row.output ?? "");
  return {
    captured: true,
    truncated: output.includes("[truncated]"),
    private_removed: raw.includes("<private>") && !output.includes("<private>"),
    output_empty: output === "",
  };
}

describe("PostToolUse native events", () => {
  it.each(eventNames)("%s", async (name) => {
    const fixture = await readJson(`events/${name}.json`);
    const expected = expectedEvents[name];
    const event = fixture.event;
    const captured = shouldCapture(event);
    expect(captured).toBe(expected.captured);
    const row = captured ? (buildObservationRow(event) as Json) : null;
    expect(receiptOf(JSON.stringify(event), row)).toEqual(expected.receipt);
    if (row === null) return;

    expect(row.tool).toBe(expected.tool);
    const output = String(row.output ?? "");
    for (const text of expected.output_contains ?? []) expect(output, `retained: ${text}`).toContain(text);
    for (const text of expected.output_contains_ci ?? []) expect(output.toLowerCase(), `retained: ${text}`).toContain(text.toLowerCase());
    for (const text of expected.output_not_contains ?? []) expect(output, `must not retain: ${text}`).not.toContain(text);
    if (expected.output_empty) expect(output).toBe("");
    for (const [key, value] of Object.entries(expected.input_contains ?? {})) expect(row.input[key]).toBe(value);
  });
});

describe("local buffer persistence of captured events", () => {
  it("every captured event is appended as one parseable JSON line and reads back unchanged", async () => {
    const rows: Json[] = [];
    for (const name of eventNames) {
      const event = (await readJson(`events/${name}.json`)).event;
      if (shouldCapture(event)) rows.push(buildObservationRow(event) as Json);
    }
    expect(rows.length).toBeGreaterThan(5);
    for (const row of rows) await utils.appendObservation({ ...row, session_id: "buffer-check" });
    const text = await readFile(utils.getBufferPath("buffer-check"), "utf-8");
    const lines = text.split("\n").filter(Boolean);
    expect(lines).toHaveLength(rows.length);
    lines.forEach((line, i) => expect(JSON.parse(line).output).toBe(rows[i].output));
  }, 30_000);

  it("a torn final line (an interrupted write) does not hide the complete rows before it", async () => {
    const { appendFile } = await import("node:fs/promises");
    await utils.appendObservation({ ts: "2026-01-01T00:00:00.000Z", session_id: "torn", tool: "Bash", output: "complete row" });
    await appendFile(utils.getBufferPath("torn"), '{"ts":"2026-01-01T00:00:01.000Z","session_id":"torn","tool":"Ba', "utf-8");
    const stats = await utils.getBufferStats(utils.getBufferPath("torn"));
    expect(stats.lineCount).toBe(2);
    expect(stats.oldestTs).toBe("2026-01-01T00:00:00.000Z");
    const complete = (await readFile(utils.getBufferPath("torn"), "utf-8")).split("\n").filter(Boolean).flatMap((l) => {
      try { return [JSON.parse(l)]; } catch { return []; }
    });
    expect(complete.map((r) => r.output)).toEqual(["complete row"]);
  });
});

// ── native transcripts → conversation turns ─────────────────────────────────

describe("native transcripts", () => {
  it.each(transcriptNames)("%s", async (name) => {
    const expected = expectedTranscripts[name];
    const path = join(ROOT, "transcripts", `${name}.jsonl`);
    const turns = await adapter.extractAllTurnPairs({ transcript_path: path });
    expect(turns.length).toBeGreaterThanOrEqual(expected.turns_min);
    expect(turns.length).toBeLessThanOrEqual(expected.turns_max);
    const users = turns.map((t: Json) => t.user).join("\n");
    const assistants = turns.map((t: Json) => t.assistant).join("\n");
    for (const text of expected.user_contains ?? []) expect(users, `user text retained: ${text}`).toContain(text);
    for (const text of expected.assistant_contains ?? []) expect(assistants, `assistant text retained: ${text}`).toContain(text);
    for (const text of expected.never_in_any_turn ?? []) {
      expect(users, `must not appear: ${text}`).not.toContain(text);
      expect(assistants, `must not appear: ${text}`).not.toContain(text);
    }
    const raw = await readFile(path, "utf-8");
    const tornTail = raw.length > 0 && !raw.endsWith("\n");
    expect(tornTail).toBe(expected.receipt.interrupted_tail_dropped);
  });
});
