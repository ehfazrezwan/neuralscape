/**
 * compileForSession: windowed compile bookkeeping (state file), the retry on
 * failure, dates across midnight, the interval setting, and the state-file
 * helpers.
 */

import { existsSync, mkdtempSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const post = vi.hoisted(() => ({ calls: [] as Array<Record<string, unknown>>, fail: false }));
vi.mock("../src/utils.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/utils.js")>();
  return {
    ...actual,
    neuralscapePost: async (_path: string, body: Record<string, unknown>) => {
      post.calls.push(body);
      if (post.fail) throw new Error("boom");
      return {};
    },
  };
});

process.env.NEURALSCAPE_USER_ID = "window-test";
const { compileForSession, getCompileIntervalMin, triggerCompile, DEFAULT_COMPILE_INTERVAL_MIN } = await import(
  "../src/core/compile.js"
);
const { getCompileStatePath, readCompileState, writeCompileState, listPendingBuffers, getObservationDir } = await import(
  "../src/utils.js"
);

let saved: Record<string, string | undefined>;
beforeEach(() => {
  saved = {
    data: process.env.CLAUDE_PLUGIN_DATA,
    modern: process.env.CLAUDE_PLUGIN_OPTION_COMPILE_INTERVAL_MIN,
    legacy: process.env.NEURALSCAPE_COMPILE_INTERVAL_MIN,
  };
  process.env.CLAUDE_PLUGIN_DATA = mkdtempSync(join(tmpdir(), "ns-window-"));
  delete process.env.CLAUDE_PLUGIN_OPTION_COMPILE_INTERVAL_MIN;
  delete process.env.NEURALSCAPE_COMPILE_INTERVAL_MIN;
  post.calls.length = 0;
  post.fail = false;
});
afterEach(() => {
  for (const [k, env] of [
    ["data", "CLAUDE_PLUGIN_DATA"],
    ["modern", "CLAUDE_PLUGIN_OPTION_COMPILE_INTERVAL_MIN"],
    ["legacy", "NEURALSCAPE_COMPILE_INTERVAL_MIN"],
  ] as const) {
    if (saved[k] === undefined) delete process.env[env];
    else process.env[env] = saved[k];
  }
});

const input = (date = "2026-11-16") => ({ date, userId: "u", shouldCompile: true });
const MIN = 60_000;

describe("compileForSession", () => {
  it("compiles on the first delivering Stop, then not again inside the window", async () => {
    const t0 = 1_000_000_000_000;
    expect(await compileForSession("s", input(), { delivered: 1, final: false, now: t0 })).toBe(true);
    expect(await compileForSession("s", input(), { delivered: 2, final: false, now: t0 + 5 * MIN })).toBe(false);
    expect(await compileForSession("s", input(), { delivered: 1, final: false, now: t0 + 14 * MIN })).toBe(false);
    expect(post.calls).toEqual([{ date: "2026-11-16", user_id: "u" }]);
    expect((await readCompileState("s")).dirtyDates).toEqual(["2026-11-16"]);
    // the window has passed: the turns delivered meanwhile are compiled
    expect(await compileForSession("s", input(), { delivered: 0, final: false, now: t0 + 16 * MIN })).toBe(true);
    expect(post.calls).toHaveLength(2);
    expect((await readCompileState("s")).dirtyDates).toEqual([]);
  });

  it("does nothing when nothing was delivered and nothing is dirty", async () => {
    expect(await compileForSession("s", input(), { delivered: 0, final: true })).toBe(false);
    expect(post.calls).toHaveLength(0);
    expect(existsSync(getCompileStatePath("s"))).toBe(false);
  });

  it("final (SessionEnd) compiles whatever is dirty regardless of the window", async () => {
    const t0 = 2_000_000_000_000;
    await compileForSession("s", input(), { delivered: 1, final: false, now: t0 });
    await compileForSession("s", input(), { delivered: 1, final: false, now: t0 + MIN });
    expect(post.calls).toHaveLength(1);
    // dirty date is already recorded; one more compile at the end
    expect(await compileForSession("s", input(), { delivered: 0, final: true, now: t0 + 2 * MIN })).toBe(true);
    expect(post.calls).toHaveLength(2);
    expect(await compileForSession("s", input(), { delivered: 0, final: true, now: t0 + 3 * MIN })).toBe(false);
  });

  it("a failed POST keeps the date dirty and the window open, so the next Stop retries", async () => {
    const t0 = 3_000_000_000_000;
    post.fail = true;
    expect(await compileForSession("s", input(), { delivered: 1, final: false, now: t0 })).toBe(true);
    expect((await readCompileState("s")).dirtyDates).toEqual(["2026-11-16"]);
    expect((await readCompileState("s")).compiledAt).toBe(0);
    post.fail = false;
    expect(await compileForSession("s", input(), { delivered: 0, final: false, now: t0 + MIN })).toBe(true);
    expect((await readCompileState("s")).dirtyDates).toEqual([]);
  });

  it("compiles an older still-dirty date as well (the session crossed midnight)", async () => {
    const t0 = 4_000_000_000_000;
    await compileForSession("s", input("2026-11-15"), { delivered: 1, final: false, now: t0, intervalMin: 60 });
    post.calls.length = 0;
    await compileForSession("s", input("2026-11-16"), { delivered: 1, final: true, now: t0 + MIN });
    // the older date was compiled at the first Stop; dirty now only has the new one
    expect(post.calls.map((c) => c.date)).toEqual(["2026-11-16"]);
    // and when the earlier compile had failed, both go out
    post.calls.length = 0;
    post.fail = true;
    await compileForSession("t", input("2026-11-15"), { delivered: 1, final: false, now: t0 });
    post.fail = false;
    post.calls.length = 0;
    await compileForSession("t", input("2026-11-16"), { delivered: 1, final: true, now: t0 + MIN });
    expect(post.calls.map((c) => c.date)).toEqual(["2026-11-15", "2026-11-16"]);
  });

  it("honours an explicit interval and the COMPILE_INTERVAL_MIN setting", async () => {
    const t0 = 5_000_000_000_000;
    await compileForSession("s", input(), { delivered: 1, final: false, now: t0, intervalMin: 1 });
    expect(await compileForSession("s", input(), { delivered: 1, final: false, now: t0 + 2 * MIN, intervalMin: 1 })).toBe(true);
    expect(getCompileIntervalMin()).toBe(DEFAULT_COMPILE_INTERVAL_MIN);
    process.env.CLAUDE_PLUGIN_OPTION_COMPILE_INTERVAL_MIN = "0";
    expect(getCompileIntervalMin()).toBe(0);
    process.env.CLAUDE_PLUGIN_OPTION_COMPILE_INTERVAL_MIN = "-3";
    expect(getCompileIntervalMin()).toBe(DEFAULT_COMPILE_INTERVAL_MIN);
    process.env.CLAUDE_PLUGIN_OPTION_COMPILE_INTERVAL_MIN = "abc";
    expect(getCompileIntervalMin()).toBe(DEFAULT_COMPILE_INTERVAL_MIN);
  });
});

describe("triggerCompile", () => {
  it("reports success, failure, and a no-op for shouldCompile=false", async () => {
    expect(await triggerCompile(input())).toBe(true);
    post.fail = true;
    expect(await triggerCompile(input())).toBe(false);
    post.calls.length = 0;
    expect(await triggerCompile({ ...input(), shouldCompile: false })).toBe(true);
    expect(post.calls).toHaveLength(0);
  });
});

describe("compile state file", () => {
  it("reads garbage and absence as empty state, and survives an unwritable directory", async () => {
    expect(await readCompileState("nope")).toEqual({ dirtyDates: [], compiledAt: 0 });
    await writeCompileState("g", { dirtyDates: ["d"], compiledAt: 5 });
    writeFileSync(getCompileStatePath("g"), "not json");
    expect(await readCompileState("g")).toEqual({ dirtyDates: [], compiledAt: 0 });
    writeFileSync(getCompileStatePath("g"), JSON.stringify({ dirtyDates: [1, "a", "b", "c", "d"], compiledAt: "x" }));
    expect(await readCompileState("g")).toEqual({ dirtyDates: ["a", "b", "c", "d"].slice(-3), compiledAt: 0 });
    process.env.CLAUDE_PLUGIN_DATA = join(process.env.CLAUDE_PLUGIN_DATA!, "file-not-dir");
    writeFileSync(process.env.CLAUDE_PLUGIN_DATA, "x");
    await writeCompileState("h", { dirtyDates: [], compiledAt: 1 }); // swallowed
  });

  it("is aged out with the other per-session leftovers", async () => {
    await writeCompileState("old", { dirtyDates: [], compiledAt: 1 });
    await writeCompileState("new", { dirtyDates: [], compiledAt: 1 });
    const old = new Date(Date.now() - 15 * 24 * 60 * 60 * 1000);
    utimesSync(getCompileStatePath("old"), old, old);
    await listPendingBuffers("mine");
    expect(existsSync(getCompileStatePath("old"))).toBe(false);
    expect(existsSync(getCompileStatePath("new"))).toBe(true);
    expect(getObservationDir()).toContain("observations");
  });
});
