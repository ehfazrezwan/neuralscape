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
  shouldInjectNow,
  getCompileMarkerPath,
  readCompileMarker,
  writeCompileMarker,
  clearCompileMarker,
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

  it("matches the compile-observations skill: one checkpoint for 3+, remember for 1-2", () => {
    const text = buildCompileInstruction(stats, "any reason");
    expect(text).toContain("mcp__plugin_neuralscape_neuralscape__checkpoint");
    expect(text).toContain("mcp__plugin_neuralscape_neuralscape__remember");
    expect(text).toMatch(/ONE `mcp__plugin_neuralscape_neuralscape__checkpoint`[^.]*3 or more/);
    expect(text).toMatch(/`mcp__plugin_neuralscape_neuralscape__remember` calls when there are 1-2/);
    // the old text told Claude to call remember once per memory
    expect(text).not.toMatch(/one wiki-quality memory via/i);
  });

  it("points at the skill instead of duplicating its field list", () => {
    const text = buildCompileInstruction(stats, "any reason");
    expect(text).toContain("The skill holds the quality rubric and the field list");
    expect(text).not.toContain("observation_type");
    expect(text).not.toContain("concepts");
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


describe("shouldInjectNow (once per threshold crossing)", () => {
  const MIN = 60_000;
  const now = Date.parse("2026-11-16T12:00:00Z");
  const st = (lineCount: number) => ({ path: "/tmp/b.jsonl", sessionId: "s", lineCount, oldestTs: null }) as any;

  it("injects the first time (no marker)", () => {
    expect(shouldInjectNow(st(25), null, 25, 30, now)).toBe(true);
  });

  it("does not re-inject on the next prompts after an injection", () => {
    const marker = { lineCount: 25, injectedAt: now - 1 * MIN };
    expect(shouldInjectNow(st(25), marker, 25, 30, now)).toBe(false);
    expect(shouldInjectNow(st(30), marker, 25, 30, now)).toBe(false);
    expect(shouldInjectNow(st(49), marker, 25, 30, now)).toBe(false);
  });

  it("re-arms when the buffer grows by another threshold", () => {
    const marker = { lineCount: 25, injectedAt: now - 1 * MIN };
    expect(shouldInjectNow(st(50), marker, 25, 30, now)).toBe(true);
  });

  it("re-arms when the last injection ages out", () => {
    const marker = { lineCount: 25, injectedAt: now - 30 * MIN };
    expect(shouldInjectNow(st(26), marker, 25, 30, now)).toBe(true);
  });

  it("re-arms when the buffer was truncated and refilled PAST the old count between two prompts", () => {
    const marker = { lineCount: 25, injectedAt: now - 1 * MIN, firstTs: "2026-11-16T10:00:00.000Z" };
    const refilled = { ...st(30), oldestTs: "2026-11-16T11:30:00.000Z" };
    expect(shouldInjectNow(refilled, marker, 25, 30, now)).toBe(true);
    // same first row: still the same fill, stays quiet
    expect(shouldInjectNow({ ...st(30), oldestTs: marker.firstTs }, marker, 25, 30, now)).toBe(false);
  });

  it("re-arms when the buffer was truncated and refilled below the old count", () => {
    const marker = { lineCount: 80, injectedAt: now - 1 * MIN };
    expect(shouldInjectNow(st(25), marker, 25, 30, now)).toBe(true);
  });
});


describe("compile marker file", () => {
  it("round-trips, tolerates garbage and absence, and clears", async () => {
    const { mkdtemp, writeFile } = await import("node:fs/promises");
    const { tmpdir } = await import("node:os");
    const { join } = await import("node:path");
    const dir = await mkdtemp(join(tmpdir(), "ns-marker-"));
    const buf = join(dir, "s.jsonl");
    expect(getCompileMarkerPath(buf)).toBe(buf + ".compile-armed");
    expect(await readCompileMarker(buf)).toBeNull();
    await writeCompileMarker(buf, { lineCount: 7, injectedAt: 123 });
    expect(await readCompileMarker(buf)).toEqual({ lineCount: 7, injectedAt: 123 });
    await writeCompileMarker(buf, { lineCount: 7, injectedAt: 123, firstTs: "t0" });
    expect(await readCompileMarker(buf)).toEqual({ lineCount: 7, injectedAt: 123, firstTs: "t0" });
    await writeCompileMarker(buf, { lineCount: 7, injectedAt: 123, firstTs: null });
    expect(await readCompileMarker(buf)).toEqual({ lineCount: 7, injectedAt: 123, firstTs: null });
    await writeFile(getCompileMarkerPath(buf), "{}");
    expect(await readCompileMarker(buf)).toBeNull();
    await clearCompileMarker(buf);
    await clearCompileMarker(buf); // already gone: no throw
    // an unwritable location is swallowed (the hook must never fail a prompt)
    await writeCompileMarker(join(dir, "missing-dir", "s.jsonl"), { lineCount: 1, injectedAt: 1 });
  });
});
