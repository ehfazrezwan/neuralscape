/**
 * SessionStart helpers added around the compact/resume and pending-buffer work:
 * the local compact-snapshot fallback, the pending-observations note, and the
 * legacy/index formatters' edge branches. Behaviour is asserted, not just run.
 */

import { mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Must be set before the awaited import: the hook runs main() at load otherwise.
process.env.NEURALSCAPE_TEST = "1";

const listPendingBuffersMock = vi.hoisted(() => ({ fail: false }));
vi.mock("../src/utils.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/utils.js")>();
  return {
    ...actual,
    listPendingBuffers: (...args: Parameters<typeof actual.listPendingBuffers>) => {
      if (listPendingBuffersMock.fail) return Promise.reject(new Error("boom"));
      return actual.listPendingBuffers(...args);
    },
  };
});

const { buildIndexContext, buildPendingNote, formatMemories, localSnapshotMemory, withLocalCompactSnapshot } =
  await import("../src/hooks/session-start.js");
const { writeLocalCompactSnapshot, getObservationDir } = await import("../src/utils.js");

import type { ContextResponse, NeuralscapeMemory } from "../src/utils.js";

const NOW = new Date("2026-07-05T12:00:00.000Z");
let scratch: string;
let savedData: string | undefined;

beforeEach(() => {
  scratch = mkdtempSync(join(tmpdir(), "ns-session-start-"));
  savedData = process.env.CLAUDE_PLUGIN_DATA;
  process.env.CLAUDE_PLUGIN_DATA = scratch;
  listPendingBuffersMock.fail = false;
});
afterEach(() => {
  if (savedData === undefined) delete process.env.CLAUDE_PLUGIN_DATA;
  else process.env.CLAUDE_PLUGIN_DATA = savedData;
  rmSync(scratch, { recursive: true, force: true });
  vi.restoreAllMocks();
});

const SNAP = {
  sessionId: "sess-1",
  projectId: "proj",
  content: "Compact snapshot: session sess-1 in project proj compacted (auto) at x after 2 captured turn(s).",
  createdAt: "2026-07-05T11:55:00.000Z",
};

describe("localSnapshotMemory", () => {
  it("maps a local snapshot to a task_context compact-snapshot memory keyed by session", () => {
    expect(localSnapshotMemory(SNAP)).toEqual({
      id: "local-compact-sess-1",
      memory: SNAP.content,
      category: "task_context",
      project_id: "proj",
      tags: ["compact_snapshot", "session:sess-1"],
      created_at: SNAP.createdAt,
    });
  });
});

describe("withLocalCompactSnapshot", () => {
  const input = { source: "compact", session_id: "sess-1" };

  it("returns the context untouched when no local snapshot file exists", async () => {
    const ctx: ContextResponse = { categories: {} };
    expect(await withLocalCompactSnapshot(ctx, input)).toBe(ctx);
  });

  it("adds the local snapshot under task_context when the service has not returned it yet", async () => {
    await writeLocalCompactSnapshot(SNAP);
    const out = await withLocalCompactSnapshot({} as ContextResponse, input);
    expect(out.categories?.task_context?.map((m: NeuralscapeMemory) => m.id)).toEqual(["local-compact-sess-1"]);
  });

  it("appends after existing task_context memories and keeps other categories", async () => {
    await writeLocalCompactSnapshot(SNAP);
    const other: NeuralscapeMemory = { id: "m1", memory: "unrelated", category: "task_context" };
    const decision: NeuralscapeMemory = { id: "d1", memory: "chose ARQ", category: "decision" };
    const out = await withLocalCompactSnapshot({ categories: { task_context: [other], decision: [decision] } }, input);
    expect(out.categories?.task_context?.map((m: NeuralscapeMemory) => m.id)).toEqual(["m1", "local-compact-sess-1"]);
    expect(out.categories?.decision).toEqual([decision]);
  });

  it("does not duplicate a snapshot the service already returned (whitespace-insensitive)", async () => {
    await writeLocalCompactSnapshot(SNAP);
    const served: NeuralscapeMemory = { id: "srv", memory: `  ${SNAP.content}\n`, category: "task_context" };
    const ctx: ContextResponse = { categories: { task_context: [served], empty: null as unknown as NeuralscapeMemory[] } };
    expect(await withLocalCompactSnapshot(ctx, input)).toBe(ctx);
  });

  it("tolerates a category whose list is null and a memory with no text", async () => {
    await writeLocalCompactSnapshot(SNAP);
    const noText = { id: "n", category: "task_context" } as NeuralscapeMemory;
    const ctx = { categories: { a: null as unknown as NeuralscapeMemory[], task_context: [noText] } };
    const out = await withLocalCompactSnapshot(ctx, input);
    expect(out.categories?.task_context?.map((m: NeuralscapeMemory) => m.id)).toEqual(["n", "local-compact-sess-1"]);
  });
});

describe("buildPendingNote", () => {
  const seed = (name: string, rows: number, ended: boolean) => {
    const dir = getObservationDir();
    mkdirSync(dir, { recursive: true });
    const path = join(dir, `${name}.jsonl`);
    writeFileSync(path, '{"ts":"2026-07-05","tool":"Read"}\n'.repeat(rows), "utf-8");
    const old = new Date(Date.now() - 5 * 60_000);
    utimesSync(path, old, old);
    if (ended) writeFileSync(`${path}.ended`, "x", "utf-8");
    return path;
  };

  it("is empty when there are no pending buffers", async () => {
    expect(await buildPendingNote("current")).toBe("");
  });

  it("names every ended prior buffer with its path and the total row count, and skips the current session", async () => {
    const a = seed("prior-a", 2, true);
    const b = seed("prior-b", 3, true);
    seed("current", 4, true);
    const note = await buildPendingNote("current");
    expect(note).toContain("# Neuralscape — Pending Observations");
    expect(note).toContain("2 buffer file(s) from prior sessions hold 5 unprocessed tool observations");
    expect(note).toContain(`\`${a}\``);
    expect(note).toContain(`\`${b}\``);
    expect(note).not.toContain("current.jsonl");
    expect(note).toContain("compile-observations");
  });

  it("does not offer a live buffer of another concurrent session", async () => {
    seed("other-terminal", 2, false);
    expect(await buildPendingNote("mine")).toBe("");
  });

  it("degrades to an empty note (never throws) when listing fails", async () => {
    listPendingBuffersMock.fail = true;
    await expect(buildPendingNote("x")).resolves.toBe("");
  });
});

describe("formatMemories", () => {
  it("is empty when no category has memories", () => {
    expect(formatMemories({})).toBe("");
    expect(formatMemories({ preference: [] })).toBe("");
  });

  it("labels known categories and falls back to nothing for unknown ones", () => {
    const out = formatMemories({
      preference: [{ id: "1", memory: "likes tabs" }],
      not_a_category: [{ id: "2", memory: "ignored" }],
    });
    expect(out).toContain("# Neuralscape Memory Context");
    expect(out).toContain("- likes tabs");
    expect(out).not.toContain("ignored");
  });

  it("stops at the character budget without emitting a header-only section", () => {
    const big = "x".repeat(7990);
    const out = formatMemories({
      preference: [{ id: "1", memory: big }],
      personal_fact: [{ id: "2", memory: "y".repeat(100) }],
    });
    expect(out).toContain(big);
    expect(out).not.toContain("y".repeat(100));
    // a memory too large for the remaining budget yields no section at all
    expect(formatMemories({ preference: [{ id: "3", memory: "z".repeat(9000) }] })).toBe("");
  });
});

describe("buildIndexContext edge inputs", () => {
  const base = { cards: [], codeGraphAvailable: false, budgetTokens: 1000, now: NOW };

  it("renders nothing for a context without categories", () => {
    expect(buildIndexContext({ ...base, context: {} as ContextResponse })).toBe("");
  });

  it("renders Previously for a session note with no timestamp and keeps the footer", () => {
    const note: NeuralscapeMemory = {
      id: "note-1",
      memory: "Session note:\nRequest: ship the gate\nNext steps: merge it",
      category: "task_context",
      tags: ["session_note"],
    };
    const out = buildIndexContext({ ...base, context: { categories: { task_context: [note] } } });
    expect(out).toContain("## Previously");
    expect(out).toContain("**Next steps:** merge it");
    expect(out).not.toContain("note-1");
  });

  it("ignores a session note whose body is not a parsable note", () => {
    const bad: NeuralscapeMemory = { id: "bad", memory: "free text", category: "task_context", tags: ["session_note"] };
    const out = buildIndexContext({ ...base, context: { categories: { task_context: [bad], empty: null as unknown as NeuralscapeMemory[] } } });
    expect(out).not.toContain("## Previously");
  });

  it("resume with predecessor ids re-anchors only on the predecessor's snapshot", () => {
    const mk = (id: string, sess: string): NeuralscapeMemory => ({
      id,
      memory: `Compact snapshot: session ${sess} in project p compacted (auto) at x after 1 captured turn(s).`,
      category: "task_context",
      tags: ["compact_snapshot", `session:${sess}`],
      created_at: "2026-07-05T11:50:00.000Z",
    });
    const out = buildIndexContext({
      ...base,
      source: "resume",
      sessionId: "new-sess",
      predecessorIds: ["old-sess"],
      context: { categories: { task_context: [mk("a", "old-sess"), mk("b", "unrelated")] } },
    });
    expect(out).toContain("## Resuming after compaction");
    const resumeBlock = out.split("\n\n---\n\n").find((sec) => sec.includes("## Resuming after compaction")) ?? "";
    expect(resumeBlock).toContain("session old-sess");
    expect(resumeBlock).not.toContain("session unrelated");
  });
});
