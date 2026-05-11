/**
 * Tests for the v2 helpers we added to src/utils.ts.
 *
 * These cover everything in the "Observation Buffer" and "Tool input/output
 * shaping" sections — the surface that the new PostToolUse and
 * UserPromptSubmit hooks depend on.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  appendObservation,
  getBufferPath,
  getBufferStats,
  getObservationDir,
  getStaleMarkerPath,
  listPendingBuffers,
  markBufferStale,
  pickRelevantInput,
  truncateBuffer,
  truncateOutput,
} from "../src/utils.js";

// Each test gets its own scratch directory and uses CLAUDE_PLUGIN_DATA to
// point the helpers at it. We restore the env after each test.
let scratch: string;
let prevPluginData: string | undefined;

beforeEach(() => {
  scratch = mkdtempSync(join(tmpdir(), "neuralscape-utils-"));
  prevPluginData = process.env.CLAUDE_PLUGIN_DATA;
  process.env.CLAUDE_PLUGIN_DATA = scratch;
});

afterEach(() => {
  if (prevPluginData === undefined) delete process.env.CLAUDE_PLUGIN_DATA;
  else process.env.CLAUDE_PLUGIN_DATA = prevPluginData;
  rmSync(scratch, { recursive: true, force: true });
});


describe("getObservationDir", () => {
  it("uses CLAUDE_PLUGIN_DATA when set", () => {
    process.env.CLAUDE_PLUGIN_DATA = "/tmp/test-claude-plugin";
    expect(getObservationDir()).toBe(join("/tmp/test-claude-plugin", "observations"));
  });

  it("falls back to ~/.neuralscape/observations when CLAUDE_PLUGIN_DATA unset", () => {
    delete process.env.CLAUDE_PLUGIN_DATA;
    const dir = getObservationDir();
    expect(dir.endsWith(join(".neuralscape", "observations"))).toBe(true);
  });

  it("trims whitespace from CLAUDE_PLUGIN_DATA", () => {
    process.env.CLAUDE_PLUGIN_DATA = "   ";
    delete process.env.CLAUDE_PLUGIN_DATA; // empty string also falls back
    const dir = getObservationDir();
    expect(dir.endsWith(join(".neuralscape", "observations"))).toBe(true);
  });
});


describe("getBufferPath", () => {
  it("returns observations/{session_id}.jsonl", () => {
    const path = getBufferPath("abc123");
    expect(path.endsWith(join("observations", "abc123.jsonl"))).toBe(true);
  });

  it("sanitizes unsafe filename characters", () => {
    const path = getBufferPath("foo/../bar:weird*chars");
    // All special chars get replaced with underscore
    expect(path).toMatch(/foo_.._bar_weird_chars\.jsonl$/);
  });

  it("returns 'unknown.jsonl' for empty session id", () => {
    const path = getBufferPath("");
    expect(path.endsWith("unknown.jsonl")).toBe(true);
  });
});


describe("getStaleMarkerPath", () => {
  it("appends .stale to the buffer path", () => {
    const path = getStaleMarkerPath("abc");
    expect(path.endsWith(".jsonl.stale")).toBe(true);
  });
});


describe("appendObservation", () => {
  it("writes one JSON line per call", async () => {
    await appendObservation({ session_id: "s1", tool: "Edit", ts: "2026-05-09" });
    await appendObservation({ session_id: "s1", tool: "Bash", ts: "2026-05-09" });

    const content = readFileSync(getBufferPath("s1"), "utf-8");
    const lines = content.trim().split("\n");
    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[0]).tool).toBe("Edit");
    expect(JSON.parse(lines[1]).tool).toBe("Bash");
  });

  it("creates the observation dir on first call", async () => {
    await appendObservation({ session_id: "s2", tool: "Write" });
    const dir = getObservationDir();
    expect(statSync(dir).isDirectory()).toBe(true);
  });

  it("uses 'unknown' as session id when row has none", async () => {
    await appendObservation({ tool: "Edit" });
    const path = getBufferPath("unknown");
    expect(readFileSync(path, "utf-8")).toContain('"tool":"Edit"');
  });

  it("does not throw when the directory cannot be created", async () => {
    // Point at a path inside a non-writable location: we mock by setting
    // CLAUDE_PLUGIN_DATA to an existing FILE rather than a directory.
    const trapFile = join(scratch, "not-a-dir");
    writeFileSync(trapFile, "blocker");
    process.env.CLAUDE_PLUGIN_DATA = trapFile;
    // Should swallow the error rather than throw — hook must not block.
    await expect(appendObservation({ session_id: "s3", tool: "X" })).resolves.toBeUndefined();
  });
});


describe("getBufferStats", () => {
  it("returns lineCount=0 for missing file", async () => {
    const stats = await getBufferStats(join(scratch, "missing.jsonl"));
    expect(stats.lineCount).toBe(0);
    expect(stats.oldestTs).toBeNull();
    expect(stats.isStale).toBe(false);
  });

  it("counts non-empty lines", async () => {
    const path = join(scratch, "session.jsonl");
    writeFileSync(path, '{"ts":"2026-01-01","tool":"A"}\n{"ts":"2026-01-02","tool":"B"}\n\n', "utf-8");
    const stats = await getBufferStats(path);
    expect(stats.lineCount).toBe(2);
  });

  it("extracts oldestTs from the first row", async () => {
    const path = join(scratch, "session.jsonl");
    writeFileSync(path, '{"ts":"2026-01-01T00:00:00Z","tool":"A"}\n', "utf-8");
    const stats = await getBufferStats(path);
    expect(stats.oldestTs).toBe("2026-01-01T00:00:00Z");
  });

  it("reports oldestTs=null for malformed first line", async () => {
    const path = join(scratch, "bad.jsonl");
    writeFileSync(path, "not json\n", "utf-8");
    const stats = await getBufferStats(path);
    expect(stats.lineCount).toBe(1);
    expect(stats.oldestTs).toBeNull();
  });

  it("detects the .stale marker", async () => {
    const path = join(scratch, "session.jsonl");
    writeFileSync(path, '{"ts":"2026-01-01","tool":"A"}\n', "utf-8");
    writeFileSync(path + ".stale", "2026-05-09", "utf-8");
    const stats = await getBufferStats(path);
    expect(stats.isStale).toBe(true);
  });

  it("oldestTs=null when first row's ts is not a string", async () => {
    const path = join(scratch, "session.jsonl");
    writeFileSync(path, '{"ts":42}\n', "utf-8");
    const stats = await getBufferStats(path);
    expect(stats.oldestTs).toBeNull();
  });
});


describe("listPendingBuffers", () => {
  it("returns [] when the observation dir doesn't exist yet", async () => {
    delete process.env.CLAUDE_PLUGIN_DATA;
    process.env.CLAUDE_PLUGIN_DATA = join(scratch, "nonexistent");
    const pending = await listPendingBuffers();
    expect(pending).toEqual([]);
  });

  it("excludes the current session id", async () => {
    const dir = getObservationDir();
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "current.jsonl"), '{"ts":"2026-01-01","tool":"A"}\n', "utf-8");
    writeFileSync(join(dir, "prior.jsonl"), '{"ts":"2026-01-01","tool":"A"}\n', "utf-8");
    const pending = await listPendingBuffers("current");
    expect(pending.map((b) => b.sessionId)).toEqual(["prior"]);
  });

  it("excludes empty buffers", async () => {
    const dir = getObservationDir();
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "empty.jsonl"), "", "utf-8");
    writeFileSync(join(dir, "full.jsonl"), '{"ts":"2026-01-01","tool":"A"}\n', "utf-8");
    const pending = await listPendingBuffers();
    expect(pending.map((b) => b.sessionId)).toEqual(["full"]);
  });

  it("ignores non-jsonl files", async () => {
    const dir = getObservationDir();
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "session.jsonl"), '{"ts":"2026-01-01","tool":"A"}\n', "utf-8");
    writeFileSync(join(dir, "notes.txt"), "ignored", "utf-8");
    writeFileSync(join(dir, "session.jsonl.stale"), "marker", "utf-8");
    const pending = await listPendingBuffers();
    expect(pending).toHaveLength(1);
    expect(pending[0].sessionId).toBe("session");
  });

  it("returns [] if currentSessionId omitted and only the active buffer exists", async () => {
    const dir = getObservationDir();
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "only.jsonl"), '{"ts":"x"}\n', "utf-8");
    // Without currentSessionId filter, "only" should show up
    const pending = await listPendingBuffers();
    expect(pending.map((b) => b.sessionId)).toEqual(["only"]);
  });
});


describe("truncateBuffer", () => {
  it("empties the buffer file but preserves it", async () => {
    const path = join(scratch, "buf.jsonl");
    writeFileSync(path, '{"a":1}\n{"a":2}\n', "utf-8");
    await truncateBuffer(path);
    expect(readFileSync(path, "utf-8")).toBe("");
    expect(statSync(path).isFile()).toBe(true);
  });

  it("removes the .stale marker if present", async () => {
    const path = join(scratch, "buf.jsonl");
    writeFileSync(path, '{"a":1}\n', "utf-8");
    writeFileSync(path + ".stale", "marker", "utf-8");
    await truncateBuffer(path);
    expect(() => statSync(path + ".stale")).toThrow();
  });

  it("does not throw when the stale marker is missing", async () => {
    const path = join(scratch, "buf.jsonl");
    writeFileSync(path, '{"a":1}\n', "utf-8");
    await expect(truncateBuffer(path)).resolves.toBeUndefined();
  });

  it("swallows write errors instead of throwing", async () => {
    // Path that cannot be written (parent is a file, not a directory)
    const blocker = join(scratch, "block-file");
    writeFileSync(blocker, "x");
    await expect(truncateBuffer(join(blocker, "child.jsonl"))).resolves.toBeUndefined();
  });
});


describe("markBufferStale", () => {
  it("creates a .stale file alongside the buffer path", async () => {
    await markBufferStale("session-x");
    const markerPath = getStaleMarkerPath("session-x");
    expect(statSync(markerPath).isFile()).toBe(true);
    const content = readFileSync(markerPath, "utf-8");
    // Content is an ISO timestamp
    expect(content).toMatch(/^\d{4}-\d{2}-\d{2}T/);
  });

  it("creates the directory on first call", async () => {
    // Use a fresh subdirectory that does not yet exist
    process.env.CLAUDE_PLUGIN_DATA = join(scratch, "fresh");
    await markBufferStale("s1");
    expect(statSync(getStaleMarkerPath("s1")).isFile()).toBe(true);
  });

  it("swallows write errors silently", async () => {
    const blocker = join(scratch, "blocker");
    writeFileSync(blocker, "x");
    process.env.CLAUDE_PLUGIN_DATA = blocker;
    await expect(markBufferStale("x")).resolves.toBeUndefined();
  });
});


describe("truncateOutput", () => {
  it("returns short input unchanged", () => {
    expect(truncateOutput("hello")).toBe("hello");
  });

  it("preserves head + tail with marker for long input", () => {
    const long = "A".repeat(1000) + "B".repeat(500);
    const result = truncateOutput(long, 100, 50);
    expect(result.startsWith("A".repeat(100))).toBe(true);
    expect(result.endsWith("B".repeat(50))).toBe(true);
    expect(result).toContain("[truncated]");
  });

  it("coerces non-string input to string", () => {
    // Numbers, objects, etc. become strings.
    // @ts-expect-error — testing runtime coercion
    expect(truncateOutput(42)).toBe("42");
    // @ts-expect-error — testing runtime coercion
    expect(truncateOutput(null)).toBe("");
    // @ts-expect-error — testing runtime coercion
    expect(truncateOutput(undefined)).toBe("");
  });

  it("respects custom head/tail sizes", () => {
    const text = "x".repeat(2000);
    const result = truncateOutput(text, 50, 20);
    // Roughly head (50) + marker + tail (20)
    expect(result.length).toBeLessThan(text.length);
    expect(result.length).toBeGreaterThan(50 + 20);
  });
});


describe("pickRelevantInput", () => {
  it("returns {} for missing input", () => {
    expect(pickRelevantInput("Edit", undefined)).toEqual({});
  });

  it("keeps Edit's allowlisted fields", () => {
    const result = pickRelevantInput("Edit", {
      file_path: "/tmp/foo.ts",
      old_string: "a",
      new_string: "b",
      junk_field: "drop me",
    });
    expect(result).toEqual({
      file_path: "/tmp/foo.ts",
      old_string: "a",
      new_string: "b",
    });
  });

  it("keeps Bash command + description", () => {
    const result = pickRelevantInput("Bash", {
      command: "ls -la",
      description: "list files",
      noise: "drop",
    });
    expect(result.command).toBe("ls -la");
    expect(result.description).toBe("list files");
    expect(result.noise).toBeUndefined();
  });

  it("truncates large old_string / new_string for Edit", () => {
    const big = "X".repeat(1000);
    const result = pickRelevantInput("Edit", {
      file_path: "x.ts",
      old_string: big,
      new_string: big,
    });
    expect((result.old_string as string).length).toBeLessThanOrEqual(601);
    expect(result.old_string as string).toMatch(/…$/);
  });

  it("truncates Bash command if oversized", () => {
    const big = "echo " + "y".repeat(2000);
    const result = pickRelevantInput("Bash", { command: big });
    expect((result.command as string).length).toBeLessThanOrEqual(601);
  });

  it("falls back to scalar/short string fields for unknown tool", () => {
    const result = pickRelevantInput("UnfamiliarTool", {
      foo: "short string",
      bar: 42,
      baz: true,
      huge: "Z".repeat(1000),  // gets truncated to 400 + ellipsis
      ignored: { nested: "drop" }, // objects dropped
    });
    expect(result.foo).toBe("short string");
    expect(result.bar).toBe(42);
    expect(result.baz).toBe(true);
    expect((result.huge as string).length).toBe(401);
    expect(result.ignored).toBeUndefined();
  });

  it("returns {} for null input object", () => {
    // @ts-expect-error — testing runtime coercion
    expect(pickRelevantInput("Edit", null)).toEqual({});
  });

  it("strips fields not in allowlist for Write", () => {
    const result = pickRelevantInput("Write", {
      file_path: "/tmp/x.txt",
      content: "the entire file contents... " + "x".repeat(10000),  // dropped
    });
    expect(result.file_path).toBe("/tmp/x.txt");
    expect(result.content).toBeUndefined();
  });

  it("keeps WebFetch url and prompt", () => {
    const result = pickRelevantInput("WebFetch", {
      url: "https://example.com",
      prompt: "summarize",
    });
    expect(result).toEqual({ url: "https://example.com", prompt: "summarize" });
  });

  it("keeps WebSearch query", () => {
    const result = pickRelevantInput("WebSearch", { query: "python async" });
    expect(result).toEqual({ query: "python async" });
  });

  it("keeps Task subagent + description + prompt", () => {
    const result = pickRelevantInput("Task", {
      description: "investigate",
      subagent_type: "Explore",
      prompt: "find the bug",
    });
    expect(result.description).toBe("investigate");
    expect(result.subagent_type).toBe("Explore");
    expect(result.prompt).toBe("find the bug");
  });
});
