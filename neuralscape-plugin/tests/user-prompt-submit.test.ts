/**
 * Tests for the UserPromptSubmit hook decision logic.
 *
 * Cover the threshold/age/hard-cap branches of `shouldCompile` plus the
 * additionalContext shape produced by `buildCompileInstruction`.
 */

import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

beforeAll(() => {
  process.env.NEURALSCAPE_TEST = "1";
});

const mod = await import("../src/hooks/user-prompt-submit.js");
const {
  DEFAULT_COMPILE_THRESHOLD,
  DEFAULT_COMPILE_AGE_MIN,
  HARD_CAP,
  shouldCompile,
  buildCompileInstruction,
  getThreshold,
  getAgeMinutes,
} = mod;


describe("constants", () => {
  it("has reasonable defaults", () => {
    expect(DEFAULT_COMPILE_THRESHOLD).toBe(25);
    expect(DEFAULT_COMPILE_AGE_MIN).toBe(30);
    expect(HARD_CAP).toBe(500);
  });
});


describe("shouldCompile", () => {
  const baseStats = (over: Record<string, unknown> = {}) => ({
    path: "/tmp/buf.jsonl",
    sessionId: "s1",
    lineCount: 0,
    oldestTs: null,
    isStale: false,
    ...over,
  }) as any;

  it("returns false when buffer is empty", () => {
    const result = shouldCompile(baseStats({ lineCount: 0 }), 25, 30);
    expect(result.compile).toBe(false);
    expect(result.reason).toBe("empty");
  });

  it("returns true at hard cap regardless of threshold", () => {
    const result = shouldCompile(baseStats({ lineCount: 600 }), 1000, 9999);
    expect(result.compile).toBe(true);
    expect(result.reason).toContain("hard-cap");
  });

  it("returns true when lineCount >= threshold", () => {
    const result = shouldCompile(baseStats({ lineCount: 25 }), 25, 30);
    expect(result.compile).toBe(true);
    expect(result.reason).toContain("threshold");
  });

  it("returns true when buffer is older than age threshold", () => {
    const oldTs = new Date(Date.now() - 31 * 60_000).toISOString();
    const result = shouldCompile(
      baseStats({ lineCount: 5, oldestTs: oldTs }),
      25,
      30,
    );
    expect(result.compile).toBe(true);
    expect(result.reason).toContain("aged");
  });

  it("returns false when below threshold and buffer is fresh", () => {
    const recent = new Date(Date.now() - 60_000).toISOString();
    const result = shouldCompile(
      baseStats({ lineCount: 5, oldestTs: recent }),
      25,
      30,
    );
    expect(result.compile).toBe(false);
    expect(result.reason).toContain("below threshold");
  });

  it("returns false when oldestTs is null and below threshold", () => {
    const result = shouldCompile(baseStats({ lineCount: 5, oldestTs: null }), 25, 30);
    expect(result.compile).toBe(false);
  });

  it("hard-cap takes precedence over threshold and age", () => {
    const ancient = new Date(Date.now() - 10 * 24 * 60 * 60_000).toISOString();
    const result = shouldCompile(
      baseStats({ lineCount: HARD_CAP + 1, oldestTs: ancient }),
      10,
      5,
    );
    expect(result.compile).toBe(true);
    expect(result.reason).toContain("hard-cap");
  });
});


describe("buildCompileInstruction", () => {
  const stats = {
    path: "/tmp/observations/sess-1.jsonl",
    sessionId: "sess-1",
    lineCount: 42,
    oldestTs: "2026-05-09T00:00:00Z",
    isStale: false,
  };

  it("includes the buffer path", () => {
    const text = buildCompileInstruction(stats, "threshold reached (42 ≥ 25)");
    expect(text).toContain("/tmp/observations/sess-1.jsonl");
  });

  it("includes the line count", () => {
    const text = buildCompileInstruction(stats, "threshold reached (42 ≥ 25)");
    expect(text).toContain("42 tool observations");
  });

  it("references the compile-observations skill", () => {
    const text = buildCompileInstruction(stats, "any reason");
    expect(text).toContain("compile-observations");
  });

  it("references the remember MCP tool", () => {
    const text = buildCompileInstruction(stats, "any reason");
    expect(text).toContain("mcp__plugin_neuralscape_neuralscape__remember");
  });

  it("includes the v2 vocab summaries", () => {
    const text = buildCompileInstruction(stats, "any reason");
    expect(text).toContain("domain");
    expect(text).toContain("observation_type");
    expect(text).toContain("concepts");
    expect(text).toContain("source_type");
    expect(text).toContain("tool_extraction");
  });

  it("includes the privacy reminder", () => {
    const text = buildCompileInstruction(stats, "any reason");
    expect(text).toMatch(/never store .*API keys/i);
  });

  it("includes the throughput target", () => {
    const text = buildCompileInstruction(stats, "any reason");
    expect(text.toLowerCase()).toContain("throughput");
  });
});


describe("getThreshold", () => {
  let prevModern: string | undefined;
  let prevLegacy: string | undefined;

  beforeEach(() => {
    prevModern = process.env.CLAUDE_PLUGIN_OPTION_COMPILE_THRESHOLD;
    prevLegacy = process.env.NEURALSCAPE_COMPILE_THRESHOLD;
    delete process.env.CLAUDE_PLUGIN_OPTION_COMPILE_THRESHOLD;
    delete process.env.NEURALSCAPE_COMPILE_THRESHOLD;
  });

  afterEach(() => {
    if (prevModern === undefined) delete process.env.CLAUDE_PLUGIN_OPTION_COMPILE_THRESHOLD;
    else process.env.CLAUDE_PLUGIN_OPTION_COMPILE_THRESHOLD = prevModern;
    if (prevLegacy === undefined) delete process.env.NEURALSCAPE_COMPILE_THRESHOLD;
    else process.env.NEURALSCAPE_COMPILE_THRESHOLD = prevLegacy;
  });

  it("returns default when unset", () => {
    expect(getThreshold()).toBe(DEFAULT_COMPILE_THRESHOLD);
  });

  it("reads CLAUDE_PLUGIN_OPTION_COMPILE_THRESHOLD when set", () => {
    process.env.CLAUDE_PLUGIN_OPTION_COMPILE_THRESHOLD = "50";
    expect(getThreshold()).toBe(50);
  });

  it("falls back to NEURALSCAPE_COMPILE_THRESHOLD", () => {
    process.env.NEURALSCAPE_COMPILE_THRESHOLD = "100";
    expect(getThreshold()).toBe(100);
  });

  it("returns default for non-numeric values", () => {
    process.env.CLAUDE_PLUGIN_OPTION_COMPILE_THRESHOLD = "not-a-number";
    expect(getThreshold()).toBe(DEFAULT_COMPILE_THRESHOLD);
  });

  it("returns default for negative or zero values", () => {
    process.env.CLAUDE_PLUGIN_OPTION_COMPILE_THRESHOLD = "0";
    expect(getThreshold()).toBe(DEFAULT_COMPILE_THRESHOLD);
    process.env.CLAUDE_PLUGIN_OPTION_COMPILE_THRESHOLD = "-5";
    expect(getThreshold()).toBe(DEFAULT_COMPILE_THRESHOLD);
  });
});


describe("getAgeMinutes", () => {
  let prevModern: string | undefined;

  beforeEach(() => {
    prevModern = process.env.CLAUDE_PLUGIN_OPTION_COMPILE_AGE_MIN;
    delete process.env.CLAUDE_PLUGIN_OPTION_COMPILE_AGE_MIN;
  });

  afterEach(() => {
    if (prevModern === undefined) delete process.env.CLAUDE_PLUGIN_OPTION_COMPILE_AGE_MIN;
    else process.env.CLAUDE_PLUGIN_OPTION_COMPILE_AGE_MIN = prevModern;
  });

  it("returns default when unset", () => {
    expect(getAgeMinutes()).toBe(DEFAULT_COMPILE_AGE_MIN);
  });

  it("reads CLAUDE_PLUGIN_OPTION_COMPILE_AGE_MIN when set", () => {
    process.env.CLAUDE_PLUGIN_OPTION_COMPILE_AGE_MIN = "60";
    expect(getAgeMinutes()).toBe(60);
  });

  it("returns default for non-numeric values", () => {
    process.env.CLAUDE_PLUGIN_OPTION_COMPILE_AGE_MIN = "not-numeric";
    expect(getAgeMinutes()).toBe(DEFAULT_COMPILE_AGE_MIN);
  });
});
