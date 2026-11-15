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
const { isCompactSnapshot, renderResumeAfterCompact, snapshotSessionId, RESUME_FALLBACK_WINDOW_MS } = disclosure;

const sessionStart = await import("../src/hooks/session-start.js");
const { buildIndexContext, isResumeSource, withLocalCompactSnapshot } = sessionStart;

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

// ── per-session selection ────────────────────────────────────────

describe("snapshotSessionId", () => {
  it("reads the session:<id> tag, else the id in the content line", () => {
    expect(snapshotSessionId(snapshot({ tags: ["compact_snapshot", "session:abc-1"] }))).toBe("abc-1");
    expect(snapshotSessionId(snapshot({ tags: ["compact_snapshot"], memory: "Compact snapshot: session xyz-2 in project p compacted" }))).toBe("xyz-2");
    expect(snapshotSessionId(snapshot({ tags: [], memory: "Compact snapshot: no id here" }))).toBeNull();
  });
});

describe("renderResumeAfterCompact with a session id", () => {
  const mine = () =>
    snapshot({ tags: ["compact_snapshot", "session:mine"], memory: "Compact snapshot: session mine in project p\n- my prompt", created_at: isoMinutesAgo(30) });
  const other = () =>
    snapshot({ tags: ["compact_snapshot", "session:other"], memory: "Compact snapshot: session other in project p\n- other terminal prompt", created_at: isoMinutesAgo(1) });

  it("compact: renders only this session's snapshots, even when another session's is newer", () => {
    const { block } = renderResumeAfterCompact({ task_context: [other(), mine()] }, { now: NOW, sessionId: "mine", source: "compact" });
    expect(block).toContain("my prompt");
    expect(block).not.toContain("other terminal prompt");
  });

  it("compact: renders nothing when this session has no snapshot (never another terminal's)", () => {
    const { block, ids } = renderResumeAfterCompact({ task_context: [other()] }, { now: NOW, sessionId: "mine", source: "compact" });
    expect(block).toBe("");
    expect(ids).toEqual([]);
  });

  it("matches a tag-less legacy snapshot through the id in its content", () => {
    const legacy = snapshot({ tags: [], memory: "Compact snapshot: session mine in project p\n- legacy prompt" });
    const { block } = renderResumeAfterCompact({ task_context: [legacy, other()] }, { now: NOW, sessionId: "mine", source: "compact" });
    expect(block).toContain("legacy prompt");
    expect(block).not.toContain("other terminal prompt");
  });

  it("resume/fork with a new id and no own snapshot: only the newest snapshot inside the window", () => {
    const recent = other();
    const older = snapshot({ tags: ["compact_snapshot", "session:older"], memory: "Compact snapshot: session older in project p\n- older prompt", created_at: isoMinutesAgo(90) });
    const { block, ids } = renderResumeAfterCompact({ task_context: [older, recent] }, { now: NOW, sessionId: "brand-new", source: "resume" });
    expect(ids).toEqual([recent.id]);
    expect(block).toContain("other terminal prompt");
    expect(block).not.toContain("older prompt");
  });

  it("resume/fork: a snapshot older than the window is not pulled in", () => {
    const stale = snapshot({ created_at: new Date(NOW.getTime() - RESUME_FALLBACK_WINDOW_MS - 60_000).toISOString() });
    expect(renderResumeAfterCompact({ task_context: [stale] }, { now: NOW, sessionId: "brand-new", source: "fork" }).block).toBe("");
  });

  it("resume with an own snapshot uses it", () => {
    const { block } = renderResumeAfterCompact({ task_context: [other(), mine()] }, { now: NOW, sessionId: "mine", source: "resume" });
    expect(block).toContain("my prompt");
    expect(block).not.toContain("other terminal prompt");
  });
});

describe("session-start resume wiring", () => {
  it("treats compact, resume and fork as resume sources", () => {
    expect(isResumeSource("compact")).toBe(true);
    expect(isResumeSource("resume")).toBe(true);
    expect(isResumeSource("fork")).toBe(true);
    expect(isResumeSource("startup")).toBe(false);
    expect(isResumeSource("clear")).toBe(false);
    expect(isResumeSource(undefined)).toBe(false);
  });

  it("buildIndexContext on fork renders the snapshot block", () => {
    const out = buildIndexContext({
      context: { status: "ok", user_id: "u1", categories: { task_context: [snapshot()] } },
      cards: [],
      codeGraphAvailable: false,
      budgetTokens: 2000,
      now: NOW,
      source: "fork",
      sessionId: "brand-new",
    });
    expect(out).toContain("## Resuming after compaction");
  });

  it("withLocalCompactSnapshot is a no-op unless the source is compact", async () => {
    const ctx = { status: "ok", user_id: "u1", categories: {} };
    expect(await withLocalCompactSnapshot(ctx, { source: "startup", session_id: "s" })).toBe(ctx);
    expect(await withLocalCompactSnapshot(ctx, { source: "compact" })).toBe(ctx);
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
