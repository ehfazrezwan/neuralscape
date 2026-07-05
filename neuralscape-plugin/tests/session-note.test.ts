/**
 * Tests for the structured Stop-summary heuristics (roadmap D2) and the
 * <private> redaction hygiene (roadmap D4) they rely on.
 */

import { beforeAll, describe, expect, it } from "vitest";

beforeAll(() => {
  process.env.NEURALSCAPE_TEST = "1";
});

const noteMod = await import("../src/core/session-note.js");
const {
  buildCheckpointPayload,
  buildSessionNote,
  extractCompleted,
  extractInvestigated,
  extractLearned,
  extractNextSteps,
  extractRequest,
} = noteMod;

const utils = await import("../src/utils.js");
const { redactPrivate } = utils;

const TURNS = [
  { user: "ping", assistant: "pong" }, // heartbeat-ish: skipped for request
  {
    user: "Fix the transcript offset bug in the plugin",
    assistant:
      "Looking into it. Turns out the offset was written before flushTurns resolved, so a failed flush advanced the cursor anyway.",
  },
  {
    user: "great, ship it",
    assistant: [
      "Done — the offset now commits after the flush succeeds.",
      "",
      "## Next steps",
      "- Add an integration test for the crash-mid-flush path",
      "- Backport to the OpenClaw adapter",
      "",
      "Let me know if you want the backport now.",
    ].join("\n"),
  },
];

const OBSERVATIONS = [
  { tool: "Edit", input: { file_path: "/repo/src/adapters/claude-code.ts" } },
  { tool: "Edit", input: { file_path: "/repo/src/core/flush.ts" } },
  { tool: "Bash", input: { command: "npm test" } },
  { tool: "Bash", input: { command: "npm run build" } },
  { tool: "WebSearch", input: { query: "esbuild watch api" } },
];

describe("extractRequest", () => {
  it("returns the first substantive user message", () => {
    expect(extractRequest(TURNS)).toBe("Fix the transcript offset bug in the plugin");
  });

  it("skips heartbeats, system messages, and command envelopes", () => {
    const turns = [
      { user: "...", assistant: "x" },
      { user: "[system] boot", assistant: "x" },
      { user: "<command-name>/compact</command-name>", assistant: "x" },
      { user: "real request", assistant: "x" },
    ];
    expect(extractRequest(turns)).toBe("real request");
  });

  it("returns undefined for an all-noise session", () => {
    expect(extractRequest([{ user: "ping", assistant: "" }])).toBeUndefined();
    expect(extractRequest([])).toBeUndefined();
  });

  it("truncates long requests", () => {
    const long = "x".repeat(1000);
    expect(extractRequest([{ user: long, assistant: "y" }])!.length).toBeLessThanOrEqual(300);
  });
});

describe("extractInvestigated", () => {
  it("summarizes files, commands, and web lookups", () => {
    const out = extractInvestigated(OBSERVATIONS)!;
    expect(out).toContain("2 file(s)");
    expect(out).toContain("adapters/claude-code.ts");
    expect(out).toContain("ran 2 command(s)");
    expect(out).toContain("1 web lookup(s)");
  });

  it("caps the file list at 5 names", () => {
    const rows = Array.from({ length: 8 }, (_, i) => ({
      tool: "Edit",
      input: { file_path: `/repo/file-${i}.ts` },
    }));
    const out = extractInvestigated(rows)!;
    expect(out).toContain("8 file(s)");
    expect(out).toContain(", …");
  });

  it("returns undefined with no observations", () => {
    expect(extractInvestigated([])).toBeUndefined();
    expect(extractInvestigated([{ tool: "SomethingElse", input: {} }])).toBeUndefined();
  });
});

describe("extractLearned", () => {
  it("picks discovery-flavored sentences", () => {
    const out = extractLearned(TURNS)!;
    expect(out).toContain("Turns out the offset was written before flushTurns resolved");
  });

  it("returns undefined when nothing reads like a discovery", () => {
    expect(extractLearned([{ user: "u", assistant: "All done." }])).toBeUndefined();
  });
});

describe("extractCompleted", () => {
  it("uses the head of the final assistant message", () => {
    const out = extractCompleted(TURNS)!;
    expect(out).toContain("Done — the offset now commits after the flush succeeds.");
  });

  it("returns undefined with no assistant text", () => {
    expect(extractCompleted([{ user: "u", assistant: "  " }])).toBeUndefined();
  });
});

describe("extractNextSteps", () => {
  it("collects bullets under a Next steps heading", () => {
    const out = extractNextSteps(TURNS)!;
    expect(out).toBe(
      "Add an integration test for the crash-mid-flush path; Backport to the OpenClaw adapter",
    );
  });

  it("handles an inline next-steps label", () => {
    const turns = [{ user: "u", assistant: "Next steps: wire the SessionEnd hook" }];
    expect(extractNextSteps(turns)).toBe("wire the SessionEnd hook");
  });

  it("returns undefined when absent", () => {
    expect(extractNextSteps([{ user: "u", assistant: "all wrapped up" }])).toBeUndefined();
  });
});

describe("buildSessionNote", () => {
  it("assembles all five fields", () => {
    const note = buildSessionNote(TURNS, OBSERVATIONS)!;
    expect(note.request).toContain("offset bug");
    expect(note.investigated).toContain("file(s)");
    expect(note.learned).toContain("Turns out");
    expect(note.completed).toContain("Done");
    expect(note.next_steps).toContain("integration test");
  });

  it("returns null for a trivial session", () => {
    expect(buildSessionNote([], [])).toBeNull();
  });

  it("redacts <private> spans in every field", () => {
    const turns = [
      {
        user: "Rotate the key <private>sk-super-secret</private> please",
        assistant: "Done. Next steps: verify <private>the secret env</private> works",
      },
    ];
    const note = buildSessionNote(turns, [])!;
    expect(JSON.stringify(note)).not.toContain("sk-super-secret");
    expect(JSON.stringify(note)).not.toContain("the secret env");
    expect(note.request).toContain("[redacted]");
  });
});

describe("buildCheckpointPayload", () => {
  it("is ONE checkpoint call: session_note only, zero memories", () => {
    const payload = buildCheckpointPayload({ request: "r" }, "alice", "proj-1");
    expect(payload).toEqual({
      memories: [],
      session_note: { request: "r" },
      user_id: "alice",
      project_id: "proj-1",
    });
  });

  it("omits project_id when global", () => {
    const payload = buildCheckpointPayload({ next_steps: "n" }, "alice");
    expect(payload).not.toHaveProperty("project_id");
  });
});

// ── Edge branches ────────────────────────────────────────────────

describe("edge branches", () => {
  it("extractRequest skips local-command envelopes", () => {
    const turns = [
      { user: "<local-command-stdout>x</local-command-stdout>", assistant: "y" },
      { user: "<command-message>compact</command-message>", assistant: "y" },
      { user: "do the thing", assistant: "y" },
    ];
    expect(extractRequest(turns)).toBe("do the thing");
  });

  it("extractInvestigated counts subagent tasks and notebook paths", () => {
    const out = extractInvestigated([
      { tool: "Task", input: { description: "explore" } },
      { tool: "NotebookEdit", input: { notebook_path: "/repo/nb.ipynb" } },
    ])!;
    expect(out).toContain("1 subagent task(s)");
    expect(out).toContain("nb.ipynb");
  });

  it("extractLearned caps at two sentences", () => {
    const turns = [
      {
        user: "u",
        assistant:
          "Turns out A. Turns out B. Turns out C. Something else entirely here.",
      },
    ];
    const out = extractLearned(turns)!;
    expect(out).toContain("Turns out A.");
    expect(out).toContain("Turns out B.");
    expect(out).not.toContain("Turns out C.");
  });

  it("extractCompleted skips system-message tails", () => {
    const turns = [
      { user: "u", assistant: "the real wrap-up of the work session done here" },
      { user: "u2", assistant: "[system] shutting down" },
    ];
    expect(extractCompleted(turns)).toContain("real wrap-up");
  });

  it("extractNextSteps stops bullet collection at a non-bullet line", () => {
    const turns = [
      {
        user: "u",
        assistant: "Next steps\n- only bullet\nplain paragraph after\n- ignored bullet",
      },
    ];
    expect(extractNextSteps(turns)).toBe("only bullet");
  });

  it("extractNextSteps returns undefined for a bare heading with nothing under it", () => {
    expect(extractNextSteps([{ user: "u", assistant: "## Next steps" }])).toBeUndefined();
  });

  it("buildSessionNote works from observations alone", () => {
    const note = buildSessionNote([], [{ tool: "Bash", input: { command: "make" } }])!;
    expect(note.investigated).toContain("ran 1 command(s)");
    expect(note.request).toBeUndefined();
  });
});

// ── <private> redaction (D4) ─────────────────────────────────────

describe("redactPrivate", () => {
  it("redacts closed spans", () => {
    expect(redactPrivate("a <private>secret</private> b")).toBe("a [redacted] b");
  });

  it("redacts multiple and multi-line spans", () => {
    const out = redactPrivate(
      "one <private>s1</private> two <PRIVATE>s2\nline</PRIVATE> three",
    );
    expect(out).toBe("one [redacted] two [redacted] three");
  });

  it("bounds an unclosed tag to the current line/sentence, never to EOF (audit 27 #34)", () => {
    // A stray literal <private> must not swallow the rest of the transcript.
    const out = redactPrivate(
      "The doc says use <private> to mark secrets.\nSecond line survives.\nThird line survives too.",
    );
    expect(out).toContain("Second line survives.");
    expect(out).toContain("Third line survives too.");
    expect(out).not.toContain("<private>");
  });

  it("unmatched opener redacts through the next sentence boundary with a warning marker", () => {
    const out = redactPrivate("a <private>oops secret here. But this sentence stays.");
    expect(out).toContain("[redacted:unclosed]");
    expect(out).toContain("But this sentence stays.");
    expect(out).not.toContain("oops secret here");
  });

  it("unmatched opener with no boundary redacts to end of that line only", () => {
    expect(redactPrivate("head <private>tail without close")).toBe("head [redacted:unclosed]");
    const multi = redactPrivate("head <private>tail without close\nnext line stays");
    expect(multi).toBe("head [redacted:unclosed]\nnext line stays");
  });

  it("still redacts matched pairs fully even when an unmatched opener follows", () => {
    const out = redactPrivate("a <private>s1</private> b <private>trailing");
    expect(out).toBe("a [redacted] b [redacted:unclosed]");
  });

  it("is idempotent", () => {
    const once = redactPrivate("a <private>s</private> b <private>c. d");
    expect(redactPrivate(once)).toBe(once);
  });

  it("leaves clean text untouched", () => {
    expect(redactPrivate("nothing to hide")).toBe("nothing to hide");
    expect(redactPrivate("")).toBe("");
    // A stray closer without an opener is inert (nothing to redact).
    expect(redactPrivate("just a </private> closer")).toBe("just a </private> closer");
  });
});
