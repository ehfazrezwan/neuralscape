/**
 * Privacy: text that leaves the machine (or is written to disk) is redacted
 * FAIL-CLOSED: an unclosed <private> opener redacts through the end of the
 * text, and a span split across assistant entries is redacted before the
 * final entry is isolated. Sentinels must never appear.
 */

import { beforeAll, describe, expect, it } from "vitest";

beforeAll(() => {
  process.env.NEURALSCAPE_TEST = "1";
});

const { redactPrivate } = await import("../src/utils.js");
const { buildObservationRow } = await import("../src/hooks/post-tool-use.js");
const { buildSessionNote, extractCompleted, redactedFinal } = await import("../src/core/session-note.js");
const { buildCompactSnapshotContent } = await import("../src/core/compact.js");

describe("redactPrivate failClosed", () => {
  it("default mode still bounds an unclosed opener to the line (unchanged)", () => {
    expect(redactPrivate("a <private>b\nnext line stays")).toBe("a [redacted:unclosed]\nnext line stays");
  });
  it("fail-closed redacts an unclosed opener through the end of the text, across lines and sentences", () => {
    expect(redactPrivate("a <private>SECRET1. SECRET2\nSECRET3", { failClosed: true })).toBe("a [redacted:unclosed]");
  });
  it("fail-closed keeps matched pairs exact and only swallows after the unmatched one", () => {
    expect(redactPrivate("x <private>one</private> y <private>two\nthree", { failClosed: true })).toBe(
      "x [redacted] y [redacted:unclosed]",
    );
  });
  it("is idempotent and a no-op without a tag", () => {
    const once = redactPrivate("p <private>q", { failClosed: true });
    expect(redactPrivate(once, { failClosed: true })).toBe(once);
    expect(redactPrivate("plain", { failClosed: true })).toBe("plain");
  });
});

describe("observation rows (PostToolUse) are fail-closed", () => {
  it("output: an unclosed opener hides the following lines", () => {
    const row = buildObservationRow({
      tool_name: "Bash",
      tool_input: { command: "cat notes" },
      tool_response: "before <private>SENTINEL-A first line\nSENTINEL-B second line\nSENTINEL-C",
    });
    const out = row.output as string;
    expect(out).toContain("before");
    expect(out).not.toMatch(/SENTINEL/);
  });
  it("output: a JSON-stringified object response with an unclosed opener", () => {
    const row = buildObservationRow({
      tool_name: "mcp__other__tool",
      tool_input: {},
      tool_response: { stdout: "ok <private>SENTINEL-A.\nSENTINEL-B", exit: 0, more: "SENTINEL-C after" },
    });
    expect(row.output as string).not.toMatch(/SENTINEL/);
    expect(row.output as string).toContain("[redacted:unclosed]");
  });
  it("input: an unclosed opener in a command or edit string", () => {
    const edit = buildObservationRow({
      tool_name: "Edit",
      tool_input: { file_path: "/a/b.ts", old_string: "x", new_string: "key = <private>SENTINEL-A\nSENTINEL-B" },
    });
    expect(JSON.stringify(edit.input)).not.toMatch(/SENTINEL/);
    const bash = buildObservationRow({
      tool_name: "Bash",
      tool_input: { command: "echo hi <private>SENTINEL-A. then SENTINEL-B" },
    });
    expect(JSON.stringify(bash.input)).not.toMatch(/SENTINEL/);
  });
  it("input: a span that the 600-char elision would cut is redacted first", () => {
    const command = "echo " + "A".repeat(590) + " <private>SENTINEL-A" + "z".repeat(300) + "SENTINEL-B</private> tail";
    const row = buildObservationRow({ tool_name: "Bash", tool_input: { command } });
    expect(JSON.stringify(row.input)).not.toMatch(/SENTINEL/);
  });
});

describe("session note: a span split across assistant entries", () => {
  const split = {
    user: "do the thing",
    assistant: "Working on it. <private>SENTINEL-OPEN stays secret\nstill secret\nSENTINEL-CLOSE</private>\nAll done, shipped.",
    final: "SENTINEL-CLOSE</private>\nAll done, shipped.",
  };

  it("completed never contains the span's tail (it lost its opening tag in the final entry alone)", () => {
    const completed = extractCompleted([split]);
    expect(completed).toBeDefined();
    expect(completed).not.toMatch(/SENTINEL/);
    expect(completed).toContain("All done, shipped.");
  });
  it("redactedFinal returns the wrap-up with the span already gone", () => {
    expect(redactedFinal(split)).not.toMatch(/SENTINEL/);
    expect(redactedFinal(split)).toContain("All done, shipped.");
  });
  it("an unclosed opener in an earlier entry swallows the final too (fail closed)", () => {
    const t = { user: "u", assistant: "Progress <private>SENTINEL-OPEN\nmore.\nDone and dusted.", final: "Done and dusted." };
    expect(extractCompleted([t])).toBeUndefined();
  });
  it("a final that is not the tail of assistant is redacted on its own", () => {
    expect(redactedFinal({ assistant: "x", final: "y <private>SENTINEL-A" })).toBe("y [redacted:unclosed]");
  });
  it("without `final` the whole assistant text is used, redacted", () => {
    expect(extractCompleted([{ user: "u", assistant: "ok <private>SENTINEL-A\nB" }])).toBe("ok [redacted:unclosed]");
  });
  it("the whole note is clean: a discovery sentence inside a span, request, next steps", () => {
    const note = buildSessionNote(
      [
        {
          user: "fix it <private>SENTINEL-U",
          assistant:
            "Turns out the root cause was <private>SENTINEL-L1. We found that SENTINEL-L2 leaks.</private> Fine.\nNext steps:\n- ship <private>SENTINEL-N",
        },
      ],
      [],
    );
    expect(JSON.stringify(note)).not.toMatch(/SENTINEL/);
  });
});

describe("compact snapshot", () => {
  it("fail-closed: an unclosed opener in a recent prompt does not leak the rest", () => {
    const content = buildCompactSnapshotContent({
      sessionId: "s",
      trigger: "auto",
      when: new Date(),
      recentUserMessages: ["look <private>SENTINEL-A. and SENTINEL-B"],
    });
    expect(content).not.toMatch(/SENTINEL/);
  });
});
