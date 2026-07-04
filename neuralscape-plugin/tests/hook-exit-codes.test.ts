/**
 * Exit-code taxonomy tests (roadmap D4) — run the BUILT hook bundles as
 * real subprocesses:
 *
 *   transport failure (NS unreachable) → exit 0 + one-line notice
 *   malformed hook stdin (client bug)  → exit 2
 *
 * `npm test` builds first (pretest), so scripts/*.js exist.
 */

import { spawn } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const PLUGIN_ROOT = join(__dirname, "..");

// A port that refuses connections immediately.
const DEAD_URL = "http://127.0.0.1:9";

interface RunResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

function runHook(
  script: string,
  stdin: string,
  env: Record<string, string> = {},
): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    const child = spawn("node", [join(PLUGIN_ROOT, "scripts", script)], {
      env: {
        PATH: process.env.PATH ?? "",
        NEURALSCAPE_URL: DEAD_URL,
        NEURALSCAPE_USER_ID: "exit-code-test",
        ...env,
      },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    child.on("error", reject);
    child.on("close", (code) => resolve({ code, stdout, stderr }));
    child.stdin.write(stdin);
    child.stdin.end();
  });
}

describe("session-start exit taxonomy", () => {
  it("exits 2 on malformed stdin (client bug)", async () => {
    const r = await runHook("session-start.js", "this is not json");
    expect(r.code).toBe(2);
    expect(r.stderr).toContain("malformed hook input");
  });

  it("exits 2 on empty stdin", async () => {
    const r = await runHook("session-start.js", "");
    expect(r.code).toBe(2);
  });

  it("exits 0 with a one-line notice when NS is unreachable", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "ns-exit-test-"));
    const r = await runHook(
      "session-start.js",
      JSON.stringify({ session_id: "s-1", cwd: dataDir, hook_event_name: "SessionStart" }),
      { CLAUDE_PLUGIN_DATA: dataDir },
    );
    expect(r.code).toBe(0);
    const out = JSON.parse(r.stdout);
    expect(out.continue).toBe(true);
    expect(out.hookSpecificOutput.additionalContext).toContain(
      "memory service unreachable",
    );
  }, 15000);
});

describe("session-summary exit taxonomy", () => {
  it("exits 2 on malformed stdin (client bug)", async () => {
    const r = await runHook("session-summary.js", "{broken");
    expect(r.code).toBe(2);
    expect(r.stderr).toContain("malformed hook input");
  });

  it("exits 0 when the checkpoint POST cannot reach NS", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "ns-exit-test-"));
    const transcript = join(dataDir, "transcript.jsonl");
    await writeFile(
      transcript,
      [
        JSON.stringify({ type: "user", content: "please fix the flaky test" }),
        JSON.stringify({
          type: "assistant",
          content: "Done — the fixture now pins the clock so the test is deterministic.",
        }),
      ].join("\n") + "\n",
      "utf-8",
    );
    const r = await runHook(
      "session-summary.js",
      JSON.stringify({
        session_id: "s-2",
        cwd: dataDir,
        transcript_path: transcript,
        hook_event_name: "SessionEnd",
      }),
      { CLAUDE_PLUGIN_DATA: dataDir },
    );
    expect(r.code).toBe(0);
    expect(r.stderr).toContain("checkpoint failed");
  }, 15000);
});

describe("post-tool-use exit taxonomy", () => {
  it("exits 2 on malformed stdin (client bug)", async () => {
    const r = await runHook("post-tool-use.js", "not json either");
    expect(r.code).toBe(2);
  });

  it("exits 0 on a valid captured event", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "ns-exit-test-"));
    const r = await runHook(
      "post-tool-use.js",
      JSON.stringify({
        session_id: "s-3",
        cwd: dataDir,
        tool_name: "Bash",
        tool_input: { command: "make build" },
        tool_output: "ok",
      }),
      { CLAUDE_PLUGIN_DATA: dataDir },
    );
    expect(r.code).toBe(0);
  });
});
