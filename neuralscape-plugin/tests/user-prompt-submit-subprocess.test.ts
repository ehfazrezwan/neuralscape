/**
 * UserPromptSubmit hook, end to end: the BUILT scripts/user-prompt-submit.js
 * run as a subprocess against a real buffer file. Proves the compile
 * instruction is injected once per threshold crossing (not on every prompt)
 * and is re-armed after truncation, growth, or aging.
 */

import { mkdtemp, mkdir, readdir, writeFile, appendFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { additionalContext, runHook } from "./helpers/fixture.js";

const row = (i: number) => JSON.stringify({ ts: new Date().toISOString(), tool: "Bash", input: `cmd ${i}`, output: "ok" }) + "\n";

async function setup(rows: number) {
  const dataDir = await mkdtemp(join(tmpdir(), "ns-ups-"));
  await mkdir(join(dataDir, "observations"), { recursive: true });
  const buffer = join(dataDir, "observations", "sess-ups.jsonl");
  await writeFile(buffer, Array.from({ length: rows }, (_, i) => row(i)).join(""), "utf-8");
  return { dataDir, buffer };
}

const prompt = (dataDir: string, extra: Record<string, string> = {}) =>
  runHook("user-prompt-submit.js", JSON.stringify({ session_id: "sess-ups", hook_event_name: "UserPromptSubmit", prompt: "hi" }), {
    NEURALSCAPE_USER_ID: "ups-test",
    CLAUDE_PLUGIN_DATA: dataDir,
    CLAUDE_PLUGIN_OPTION_COMPILE_THRESHOLD: "10",
    ...extra,
  });

describe("user-prompt-submit.js cool-down", () => {
  it("injects once, stays quiet on the next prompts, re-arms on growth and after truncation", async () => {
    const { dataDir, buffer } = await setup(10);

    const first = await prompt(dataDir);
    expect(first.code).toBe(0);
    const ctx = additionalContext(first);
    expect(ctx).toContain("# Neuralscape — Compile Observations");
    expect(ctx).toContain("**10 tool observations**");
    expect(ctx).toContain("ONE `mcp__plugin_neuralscape_neuralscape__checkpoint`");
    expect(ctx).toContain(buffer);
    expect(existsSync(buffer + ".compile-armed")).toBe(true);

    // The next prompts (compile ignored or still running) add nothing.
    expect(additionalContext(await prompt(dataDir))).toBe("");
    await appendFile(buffer, row(11).repeat(5));
    expect(additionalContext(await prompt(dataDir))).toBe("");

    // Buffer grew by another threshold: injected again, with the new count.
    await appendFile(buffer, row(12).repeat(5));
    const regrown = additionalContext(await prompt(dataDir));
    expect(regrown).toContain("**20 tool observations**");

    // Compile ran: buffer truncated -> marker cleared on the next prompt ...
    await writeFile(buffer, "", "utf-8");
    expect(additionalContext(await prompt(dataDir))).toBe("");
    expect(existsSync(buffer + ".compile-armed")).toBe(false);
    // ... and a refill crossing the threshold injects again.
    await writeFile(buffer, row(1).repeat(10), "utf-8");
    expect(additionalContext(await prompt(dataDir))).toContain("**10 tool observations**");
  });

  it("re-arms when the last injection is older than the age window", async () => {
    const { dataDir, buffer } = await setup(10);
    expect(additionalContext(await prompt(dataDir))).toContain("Compile Observations");
    expect(additionalContext(await prompt(dataDir))).toBe("");
    const old = new Date(Date.now() - 31 * 60_000);
    // age the marker by rewriting its injectedAt
    await writeFile(buffer + ".compile-armed", JSON.stringify({ lineCount: 10, injectedAt: old.getTime() }));
    expect(additionalContext(await prompt(dataDir))).toContain("Compile Observations");
  });

  it("treats an unreadable marker as no marker", async () => {
    const { dataDir, buffer } = await setup(10);
    await writeFile(buffer + ".compile-armed", "not json");
    expect(additionalContext(await prompt(dataDir))).toContain("Compile Observations");
  });

  it("does not leave the marker among the pending buffers", async () => {
    const { dataDir } = await setup(10);
    await prompt(dataDir);
    const files = await readdir(join(dataDir, "observations"));
    expect(files.filter((f) => f.endsWith(".jsonl"))).toEqual(["sess-ups.jsonl"]);
  });
});

