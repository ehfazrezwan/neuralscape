/**
 * Compact-resilience loop, piece 2 — compact-aware SessionStart.
 *
 * Claude Code sends `source: "compact"` on the SessionStart that follows a
 * compaction (and "resume" on --resume/--continue). On those sources the
 * index-mode injection additionally renders the recent compact-snapshot
 * memories (task_context tagged `compact_snapshot`) under a
 * "## Resuming after compaction" section, full content — and keeps them
 * out of the index table (no double render). Ordinary startups are
 * untouched.
 */

import { describe, expect, it } from "vitest";

// MUST be set at module top level BEFORE the awaited imports below: the
// imported hook modules auto-run main() at evaluation time unless this is
// set, and beforeAll callbacks only fire after module evaluation — setting
// it there would be order-dependent across files/workers (Copilot, PR #131).
process.env.NEURALSCAPE_TEST = "1";

const disclosure = await import("../src/core/disclosure.js");
const { isCompactSnapshot, renderResumeAfterCompact } = disclosure;

const sessionStart = await import("../src/hooks/session-start.js");
const { buildIndexContext } = sessionStart;

import type { NeuralscapeMemory } from "../src/utils.js";

const NOW = new Date("2026-07-05T12:00:00.000Z");

function isoMinutesAgo(min: number): string {
  return new Date(NOW.getTime() - min * 60_000).toISOString();
}

let seq = 0;
function snapshot(over: Partial<NeuralscapeMemory> = {}): NeuralscapeMemory {
  seq++;
  return {
    id: `3f2a91c4-0000-4000-8000-00000000000${seq}`,
    memory:
      `Compact snapshot: session sess-${seq} in project neuralscape compacted (auto) ` +
      `at ${isoMinutesAgo(5)} after 3 captured turn(s).\n` +
      "Last user messages before compaction:\n- keep going on the compact loop",
    category: "task_context",
    tags: ["compact_snapshot"],
    created_at: isoMinutesAgo(5),
    ...over,
  };
}

function ordinary(over: Partial<NeuralscapeMemory> = {}): NeuralscapeMemory {
  seq++;
  return {
    id: `9b1de644-0000-4000-8000-00000000000${seq}`,
    memory: "Chose ARQ over Celery for the background queue.",
    category: "decision",
    observation_type: "decision",
    title: "ARQ over Celery",
    created_at: isoMinutesAgo(120),
    ...over,
  };
}

// ── isCompactSnapshot ────────────────────────────────────────────

describe("isCompactSnapshot", () => {
  it("matches by tag", () => {
    expect(isCompactSnapshot(snapshot())).toBe(true);
  });

  it("matches tag-less rows by the content marker", () => {
    expect(isCompactSnapshot(snapshot({ tags: [] }))).toBe(true);
  });

  it("rejects ordinary memories and session notes", () => {
    expect(isCompactSnapshot(ordinary())).toBe(false);
    expect(
      isCompactSnapshot(
        ordinary({ memory: "Session note:\nRequest: something", tags: ["session_note"] }),
      ),
    ).toBe(false);
  });
});

// ── renderResumeAfterCompact ─────────────────────────────────────

describe("renderResumeAfterCompact", () => {
  it("renders the newest snapshots (full content) under the resume heading", () => {
    const old = snapshot({ created_at: isoMinutesAgo(600), memory: "Compact snapshot: OLD one" });
    const fresh = snapshot({ created_at: isoMinutesAgo(2), memory: "Compact snapshot: FRESH one" });
    const { block, ids } = renderResumeAfterCompact(
      { task_context: [old, fresh], decision: [ordinary()] },
      { now: NOW },
    );
    expect(block).toContain("## Resuming after compaction");
    expect(block).toContain("FRESH one");
    // Newest first.
    expect(block.indexOf("FRESH one")).toBeLessThan(block.indexOf("OLD one"));
    expect(ids).toContain(old.id);
    expect(ids).toContain(fresh.id);
  });

  it("caps how many snapshots render and returns only the rendered ids", () => {
    const snaps = [10, 8, 6, 4, 2].map((m) =>
      snapshot({ created_at: isoMinutesAgo(m), memory: `Compact snapshot: at minute ${m}` }),
    );
    const { block, ids } = renderResumeAfterCompact({ task_context: snaps }, { now: NOW, limit: 3 });
    expect(ids.length).toBe(3);
    expect(block).toContain("at minute 2");
    expect(block).toContain("at minute 6");
    expect(block).not.toContain("at minute 10");
  });

  it("returns an empty block when no snapshots exist", () => {
    const { block, ids } = renderResumeAfterCompact({ decision: [ordinary()] }, { now: NOW });
    expect(block).toBe("");
    expect(ids.length).toBe(0);
  });
});

// ── buildIndexContext wiring ─────────────────────────────────────

function contextWith(memories: NeuralscapeMemory[], category = "task_context") {
  return {
    status: "ok",
    user_id: "u1",
    categories: { [category]: memories } as Record<string, NeuralscapeMemory[]>,
  };
}

describe("buildIndexContext compact awareness", () => {
  const base = {
    cards: [],
    codeGraphAvailable: false,
    budgetTokens: 1500,
    now: NOW,
  };

  it('renders "Resuming after compaction" when source is "compact"', () => {
    const snap = snapshot();
    const out = buildIndexContext({
      ...base,
      context: contextWith([snap, ordinary()]),
      source: "compact",
    });
    expect(out).toContain("## Resuming after compaction");
    expect(out).toContain("keep going on the compact loop");
  });

  it('renders it on "resume" too', () => {
    const out = buildIndexContext({
      ...base,
      context: contextWith([snapshot()]),
      source: "resume",
    });
    expect(out).toContain("## Resuming after compaction");
  });

  it("does NOT render it on ordinary startup/clear or when source is absent", () => {
    for (const source of ["startup", "clear", undefined]) {
      const out = buildIndexContext({
        ...base,
        context: contextWith([snapshot()]),
        source,
      });
      expect(out).not.toContain("## Resuming after compaction");
    }
  });

  it("keeps rendered snapshots out of the index table (no double render)", () => {
    const snap = snapshot({ title: "unmistakable-snapshot-title" });
    const out = buildIndexContext({
      ...base,
      context: contextWith([snap, ordinary()]),
      source: "compact",
    });
    // Full content appears once in the resume section…
    expect(out).toContain("## Resuming after compaction");
    // …and the snapshot's id/title row is absent from the index table.
    const indexSection = out.split("## Memory Index")[1] ?? "";
    expect(indexSection).not.toContain(snap.id.slice(0, 8));
  });

  it("stays a no-op for compact sources when there are no snapshots", () => {
    const out = buildIndexContext({
      ...base,
      context: contextWith([ordinary()], "decision"),
      source: "compact",
    });
    expect(out).not.toContain("## Resuming after compaction");
  });
});
