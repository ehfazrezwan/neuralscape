/**
 * CONTEXT_MODE=full, end to end: the BUILT scripts/session-start.js with
 * CLAUDE_PLUGIN_OPTION_CONTEXT_MODE=full against a fixture server. Full mode
 * must keep the continuity blocks (resume after compaction, "Previously…",
 * identity cards, standards) and then render the full memory text within
 * the legacy 8000-char budget. The default stays "index".
 */

import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { additionalContext, type Fixture, PLUGIN_ROOT, runHook, startFixture } from "./helpers/fixture.js";

let fx: Fixture;
beforeAll(async () => {
  fx = await startFixture();
});
afterAll(async () => {
  await fx.close();
});
beforeEach(() => {
  fx.requests.length = 0;
  fx.context = {};
});

const env = (dataDir: string, extra: Record<string, string> = {}) => ({
  NEURALSCAPE_USER_ID: "full-mode-test",
  NEURALSCAPE_URL: fx.url,
  NEURALSCAPE_PROJECT_ID: "full-mode-project",
  CLAUDE_PLUGIN_DATA: dataDir,
  ...extra,
});
const FULL = { CLAUDE_PLUGIN_OPTION_CONTEXT_MODE: "full" };

const start = (sessionId: string, source: string, e: Record<string, string>) =>
  runHook(
    "session-start.js",
    JSON.stringify({ session_id: sessionId, cwd: PLUGIN_ROOT, hook_event_name: "SessionStart", source }),
    e,
  );

const NOTE = [
  "Session note:",
  "Request: wire the retry loop into the worker",
  "Completed: retry loop merged behind a flag",
  "Next steps: add the backoff jitter test",
].join("\n");

async function precompactSnapshot(dataDir: string, sessionId: string): Promise<Record<string, unknown>> {
  const dir = await mkdtemp(join(tmpdir(), "ns-full-transcript-"));
  const transcript = join(dir, `${sessionId}.jsonl`);
  const rows = [
    { type: "user", message: { role: "user", content: "check that full mode keeps my resume block" } },
    { type: "assistant", message: { role: "assistant", content: [{ type: "text", text: "Checking the resume block now, with substance." }] } },
  ];
  await writeFile(transcript, rows.map((r) => JSON.stringify(r)).join("\n") + "\n", "utf-8");
  const before = fx.requests.length;
  const r = await runHook(
    "pre-compact.js",
    JSON.stringify({ session_id: sessionId, transcript_path: transcript, cwd: PLUGIN_ROOT, hook_event_name: "PreCompact", trigger: "auto" }),
    env(dataDir),
  );
  expect(r.code).toBe(0);
  const raw = fx.requests.slice(before).find((q) => q.method === "POST" && q.path.includes("/v1/memories/raw"));
  expect(raw).toBeDefined();
  return {
    id: "snap-1",
    memory: raw!.body.content,
    category: raw!.body.category,
    tags: raw!.body.tags,
    created_at: new Date().toISOString(),
  };
}

describe("CONTEXT_MODE=full", () => {
  it("startup: standards, Previously block, then full memory text (no index table)", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "ns-full-"));
    fx.context = {
      standards: [{ id: "std-1", memory: "Always sign commits (binding standard)" }],
      categories: {
        task_context: [{ id: "note-1", memory: NOTE, tags: ["session_note"], created_at: new Date().toISOString() }],
        convention: [{ id: "m-1", memory: "Use uv for Python dependency management in this repo", category: "convention" }],
      },
    };
    const r = await start("sess-startup", "startup", env(dataDir, FULL));
    expect(r.code).toBe(0);
    const ctx = additionalContext(r);
    expect(ctx).toContain("Always sign commits (binding standard)");
    expect(ctx).toContain("## Previously");
    expect(ctx).toContain("**Next steps:** add the backoff jitter test");
    expect(ctx).toContain("# Neuralscape Memory Context");
    expect(ctx).toContain("- Use uv for Python dependency management in this repo");
    expect(ctx).not.toContain("## Memory Index");
    // the note renders once, as Previously, not again as a bullet
    expect(ctx).not.toContain("- Session note:");
    // order: standards, Previously, memories
    expect(ctx.indexOf("Always sign commits")).toBeLessThan(ctx.indexOf("## Previously"));
    expect(ctx.indexOf("## Previously")).toBeLessThan(ctx.indexOf("# Neuralscape Memory Context"));
  });

  it("compact: renders 'Resuming after compaction' for this session, once, before the memories", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "ns-full-"));
    const snap = await precompactSnapshot(dataDir, "sess-compacted");
    fx.context = {
      categories: {
        task_context: [snap],
        convention: [{ id: "m-1", memory: "Prefer small PRs", category: "convention" }],
      },
    };
    const r = await start("sess-compacted", "compact", env(dataDir, FULL));
    const ctx = additionalContext(r);
    expect(ctx).toContain("## Resuming after compaction");
    expect(ctx).toContain("check that full mode keeps my resume block");
    expect(ctx).toContain("- Prefer small PRs");
    // the snapshot is not repeated as a memory bullet
    expect(ctx.split("Compact snapshot: session sess-compacted").length - 1).toBe(1);
    expect(ctx.indexOf("## Resuming after compaction")).toBeLessThan(ctx.indexOf("# Neuralscape Memory Context"));
  });

  it("a startup session does not get another session's snapshot as a resume block", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "ns-full-"));
    const snap = await precompactSnapshot(dataDir, "other-session");
    fx.context = { categories: { task_context: [snap] } };
    const ctx = additionalContext(await start("sess-new", "startup", env(dataDir, FULL)));
    expect(ctx).not.toContain("## Resuming after compaction");
  });

  it("keeps the full memory text within the 8000-char budget", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "ns-full-"));
    const many = Array.from({ length: 60 }, (_, i) => ({
      id: `m-${i}`,
      memory: `memory number ${i} ${"x".repeat(300)}`,
      category: "convention",
    }));
    fx.context = { categories: { convention: many } };
    const ctx = additionalContext(await start("sess-budget", "startup", env(dataDir, FULL)));
    const body = ctx.slice(ctx.indexOf("# Neuralscape Memory Context"));
    expect(body.length).toBeLessThan(8000 + 400);
    expect(body).toContain("memory number 0 ");
    expect(body).not.toContain("memory number 59 ");
  });

  it("default (no CONTEXT_MODE) is still the index", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "ns-full-"));
    fx.context = { categories: { convention: [{ id: "m-1", memory: "Prefer small PRs", category: "convention", created_at: new Date().toISOString() }] } };
    const ctx = additionalContext(await start("sess-default", "startup", env(dataDir)));
    expect(ctx).toContain("## Memory Index");
    expect(ctx).not.toContain("# Neuralscape Memory Context");
  });
});
