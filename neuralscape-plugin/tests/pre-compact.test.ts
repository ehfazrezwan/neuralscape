/**
 * Compact-resilience loop, piece 1 — the PreCompact snapshot builders.
 *
 * When Claude Code compacts a session (auto or /compact), the PreCompact
 * hook flushes undelivered turns through the existing adapter path and then
 * stores ONE small `task_context` "compact snapshot" memory via the existing
 * POST /v1/memories/raw endpoint. These tests pin the pure builders:
 * content shape (marker line + tail of last user messages), <private>
 * redaction, the hard size cap, and the payload contract the server's
 * RawMemoryRequest expects.
 */

import { beforeAll, describe, expect, it } from "vitest";

beforeAll(() => {
  process.env.NEURALSCAPE_TEST = "1";
});

const compact = await import("../src/core/compact.js");
const {
  COMPACT_SNAPSHOT_MARKER,
  COMPACT_SNAPSHOT_TAG,
  MAX_SNAPSHOT_CHARS,
  buildCompactSnapshotContent,
  buildCompactSnapshotPayload,
} = compact;

const WHEN = new Date("2026-07-05T10:00:00.000Z");

function inputs(over: Record<string, unknown> = {}) {
  return {
    sessionId: "sess-abc123",
    projectId: "neuralscape",
    trigger: "auto",
    when: WHEN,
    capturedTurns: 4,
    recentUserMessages: [
      "please fix the flaky flush offset test",
      "now wire the compact snapshot into session start",
      "and rebuild the bundles before pushing",
    ],
    ...over,
  };
}

// ── buildCompactSnapshotContent ──────────────────────────────────

describe("buildCompactSnapshotContent", () => {
  it("opens with the compact marker line naming session, project, trigger, time, turns", () => {
    const content = buildCompactSnapshotContent(inputs());
    const firstLine = content.split("\n")[0];
    expect(firstLine.startsWith(COMPACT_SNAPSHOT_MARKER)).toBe(true);
    expect(firstLine).toContain("sess-abc123");
    expect(firstLine).toContain("neuralscape");
    expect(firstLine).toContain("(auto)");
    expect(firstLine).toContain("2026-07-05T10:00:00.000Z");
    expect(firstLine).toContain("4 captured turn(s)");
  });

  it("appends the tail of the last user messages as bullets", () => {
    const content = buildCompactSnapshotContent(inputs());
    expect(content).toContain("Last user messages before compaction:");
    expect(content).toContain("- please fix the flaky flush offset test");
    expect(content).toContain("- and rebuild the bundles before pushing");
  });

  it("keeps only the last 3 messages when given more", () => {
    const content = buildCompactSnapshotContent(
      inputs({
        recentUserMessages: ["one", "two", "three", "four", "five"].map(
          (w) => `${w} message with enough substance`,
        ),
      }),
    );
    expect(content).not.toContain("one message");
    expect(content).not.toContain("two message");
    expect(content).toContain("- three message with enough substance");
    expect(content).toContain("- five message with enough substance");
  });

  it("skips empty/whitespace messages and omits the tail block when none remain", () => {
    const content = buildCompactSnapshotContent(
      inputs({ recentUserMessages: ["", "   ", "\n"] }),
    );
    expect(content).not.toContain("Last user messages");
    expect(content.split("\n").length).toBe(1);
  });

  it("redacts <private> spans before anything leaves the machine (D4)", () => {
    const content = buildCompactSnapshotContent(
      inputs({
        recentUserMessages: [
          "deploy key is <private>hunter2 the secret</private> ok?",
        ],
      }),
    );
    expect(content).not.toContain("hunter2");
    expect(content).toContain("[redacted]");
  });

  it("squashes newlines inside a message so each stays a single bullet", () => {
    const content = buildCompactSnapshotContent(
      inputs({ recentUserMessages: ["line one\nline two of the same message"] }),
    );
    expect(content).toContain("- line one line two of the same message");
  });

  it("truncates an oversized single message and caps total content", () => {
    const content = buildCompactSnapshotContent(
      inputs({ recentUserMessages: ["x".repeat(5000), "y".repeat(5000), "z".repeat(5000)] }),
    );
    expect(content.length).toBeLessThanOrEqual(MAX_SNAPSHOT_CHARS);
  });

  it("names the missing project explicitly when there is none", () => {
    const content = buildCompactSnapshotContent(inputs({ projectId: undefined }));
    expect(content.split("\n")[0]).toContain("(no project)");
  });
});

// ── buildCompactSnapshotPayload ──────────────────────────────────

describe("buildCompactSnapshotPayload", () => {
  it("targets the raw-memory contract: task_context + compact_snapshot tag", () => {
    const payload = buildCompactSnapshotPayload("Compact snapshot: …", "user-1", "proj-1", WHEN);
    expect(payload.content).toBe("Compact snapshot: …");
    expect(payload.user_id).toBe("user-1");
    expect(payload.category).toBe("task_context");
    expect(payload.tags).toEqual([COMPACT_SNAPSHOT_TAG]);
    expect(payload.source_type).toBe("tool_extraction");
  });

  it("scopes to project when a project id is present", () => {
    const payload = buildCompactSnapshotPayload("c", "user-1", "proj-1", WHEN);
    expect(payload.scope).toBe("project");
    expect(payload.project_id).toBe("proj-1");
  });

  it("falls back to global scope without a project id (no null project_id key)", () => {
    const payload = buildCompactSnapshotPayload("c", "user-1", undefined, WHEN);
    expect(payload.scope).toBe("global");
    expect("project_id" in payload).toBe(false);
  });

  it("sets a finite expiry so snapshots don't pile up forever", () => {
    const payload = buildCompactSnapshotPayload("c", "user-1", "proj-1", WHEN);
    const expires = new Date(payload.expires_at as string).getTime();
    expect(Number.isFinite(expires)).toBe(true);
    expect(expires).toBeGreaterThan(WHEN.getTime());
    // At most ~30 days out — this is a continuity artifact, not a wiki memory.
    expect(expires - WHEN.getTime()).toBeLessThanOrEqual(30 * 24 * 60 * 60 * 1000);
  });
});
