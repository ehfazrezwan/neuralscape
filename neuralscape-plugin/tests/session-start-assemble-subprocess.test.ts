/**
 * CONTEXT_MODE=assemble, end to end: the BUILT scripts/session-start.js
 * against a fixture server. With the setting ON the assembled bundle is
 * injected under its own heading and the route is called with the right body;
 * every failure of the route falls back to the plain index; with the setting
 * OFF (default, index, full) the route is never called.
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

const ASSEMBLE = "/v1/context/assemble";
const NOW = new Date().toISOString();
const BUNDLE_TEXT = [
  "## Identity card\nlikes small PRs",
  "## Session summary (long slot)\nwe were wiring the retry loop",
  "## Relevant memories (index)\n- [m-hit] ◇ Retry loop uses jittered backoff (convention, now, ~12 tok)",
  "## Recent messages\nuser: continue with the retry loop",
].join("\n\n");

beforeEach(() => {
  fx.requests.length = 0;
  fx.assemble = {
    status: 200,
    body: { status: "ok", budget_tokens: 1200, used_tokens: 321, format: "plain", bundle: { text: BUNDLE_TEXT } },
  };
  fx.context = {
    standards: [{ id: "std-1", memory: "Always sign commits (binding standard)" }],
    categories: {
      convention: [
        { id: "m-hit", memory: "Retry loop uses jittered backoff", category: "convention", created_at: NOW },
        { id: "m-other", memory: "Prefer small PRs", category: "convention", created_at: NOW },
      ],
    },
  };
});

const env = (dataDir: string, extra: Record<string, string> = {}) => ({
  NEURALSCAPE_USER_ID: "assemble-test",
  NEURALSCAPE_URL: fx.url,
  NEURALSCAPE_PROJECT_ID: "assemble-project",
  CLAUDE_PLUGIN_DATA: dataDir,
  ...extra,
});
const ON = { CLAUDE_PLUGIN_OPTION_CONTEXT_MODE: "assemble" };

const start = (sessionId: string, source: string, e: Record<string, string>, extra: Record<string, unknown> = {}) =>
  runHook(
    "session-start.js",
    JSON.stringify({ session_id: sessionId, cwd: PLUGIN_ROOT, hook_event_name: "SessionStart", source, ...extra }),
    e,
  );
const assembleCalls = () => fx.requests.filter((r) => r.method === "POST" && r.path === ASSEMBLE);
const newData = () => mkdtemp(join(tmpdir(), "ns-assemble-"));

describe("setting ON", () => {
  it("calls the route with the budget, session, project, user and plain format, and injects the bundle", async () => {
    const dataDir = await newData();
    const r = await start("sess-assemble-1", "startup", env(dataDir, { ...ON, CLAUDE_PLUGIN_OPTION_ASSEMBLE_BUDGET_TOKENS: "1200" }));
    expect(r.code).toBe(0);
    expect(assembleCalls()).toHaveLength(1);
    expect(assembleCalls()[0].body).toEqual({
      budget_tokens: 1200,
      format: "plain",
      user_id: "assemble-test",
      session_id: "sess-assemble-1",
      project_id: "assemble-project",
    });
    const ctx = additionalContext(r);
    expect(ctx).toContain("## Session context (assembled, ~321 of 1200 tokens)");
    expect(ctx).toContain("we were wiring the retry loop");
    expect(ctx).toContain("likes small PRs");
    // standards stay first and binding; the bundle comes before the index
    expect(ctx.indexOf("Always sign commits")).toBeLessThan(ctx.indexOf("## Session context (assembled"));
    expect(ctx.indexOf("## Session context (assembled")).toBeLessThan(ctx.indexOf("## Memory Index"));
  });

  it("does not repeat what the bundle shows: the row it lists is not in the index table, the others are", async () => {
    const dataDir = await newData();
    const ctx = additionalContext(await start("sess-assemble-2", "startup", env(dataDir, ON)));
    const index = ctx.slice(ctx.indexOf("## Memory Index"));
    expect(index).not.toContain("#m-hit");
    expect(index).toContain("#m-other");
  });

  it("sends ASSEMBLE_QUERY when set, and clamps the budget to the route's bounds", async () => {
    const dataDir = await newData();
    await start("s", "startup", env(dataDir, { ...ON, CLAUDE_PLUGIN_OPTION_ASSEMBLE_QUERY: "retry loop", CLAUDE_PLUGIN_OPTION_ASSEMBLE_BUDGET_TOKENS: "5" }));
    expect(assembleCalls()[0].body).toMatchObject({ query: "retry loop", budget_tokens: 100 });
    fx.requests.length = 0;
    await start("s", "startup", env(dataDir, { ...ON, CLAUDE_PLUGIN_OPTION_ASSEMBLE_BUDGET_TOKENS: "99999999" }));
    expect(assembleCalls()[0].body).toMatchObject({ budget_tokens: 1_000_000 });
    expect(assembleCalls()[0].body).not.toHaveProperty("query");
  });

  it("on compact, the last user prompt of the transcript is the query; the resume block and Previously stay", async () => {
    const dataDir = await newData();
    const transcript = join(dataDir, "t.jsonl");
    await writeFile(
      transcript,
      [
        { type: "user", message: { role: "user", content: "earlier question" } },
        { type: "assistant", message: { role: "assistant", content: [{ type: "text", text: "an answer of some length" }] } },
        { type: "user", message: { role: "user", content: "continue with the retry loop jitter" } },
      ]
        .map((x) => JSON.stringify(x))
        .join("\n") + "\n",
    );
    fx.context = {
      categories: {
        task_context: [
          {
            id: "snap-1",
            memory: "Compact snapshot: session sess-c in project p compacted (auto) at 2026-11-16T00:00:00Z.\nlast prompt: do x",
            tags: ["compact_snapshot", "session:sess-c"],
            created_at: NOW,
          },
          { id: "note-1", memory: "Session note:\nRequest: ship it\nNext steps: write tests", tags: ["session_note"], created_at: NOW },
        ],
      },
    };
    const r = await start("sess-c", "compact", env(dataDir, ON), { transcript_path: transcript });
    expect(assembleCalls()[0].body).toMatchObject({ query: "continue with the retry loop jitter", session_id: "sess-c" });
    const ctx = additionalContext(r);
    expect(ctx).toContain("## Session context (assembled");
    expect(ctx).toContain("## Resuming after compaction");
    expect(ctx).toContain("**Next steps:** write tests");
  });

  it("does not render the identity-card block again (the bundle carries it)", async () => {
    const dataDir = await newData();
    const ctx = additionalContext(await start("s", "startup", env(dataDir, ON)));
    expect(ctx).not.toContain("## Identity Card (");
  });
});

describe("setting ON, the route fails: the index is shown", () => {
  const index = (ctx: string) => {
    expect(ctx).toContain("## Memory Index");
    expect(ctx).toContain("#m-hit");
    expect(ctx).not.toContain("## Session context (assembled");
  };

  it("404 (an older server): quietly, no notice", async () => {
    fx.assemble = { status: 404, body: { detail: "Not Found" } };
    const r = await start("s", "startup", env(await newData(), ON));
    index(additionalContext(r));
    expect(additionalContext(r)).not.toContain("assemble");
    expect(r.stderr).not.toContain("context assemble failed"); // quiet: nothing logged either
  });

  it("400 (rejected values): the index plus a one-line notice", async () => {
    fx.assemble = { status: 400, body: { detail: "bad" } };
    const ctx = additionalContext(await start("s", "startup", env(await newData(), ON)));
    index(ctx);
    expect(ctx).toContain("CONTEXT_MODE=assemble was rejected by the service (HTTP 4xx); showing the index instead.");
  });

  it("401: the index plus a notice naming the rejected key", async () => {
    fx.assemble = { status: 401, body: {} };
    const ctx = additionalContext(await start("s", "startup", env(await newData(), ON)));
    index(ctx);
    expect(ctx).toContain("(HTTP 401/403)");
  });

  it("500: the index, no notice (the index fetch reports a failing service)", async () => {
    fx.assemble = { status: 500, body: {} };
    const r = await start("s", "startup", env(await newData(), ON));
    index(additionalContext(r));
    expect(additionalContext(r)).not.toContain("rejected");
    expect(r.stderr).toContain("context assemble failed (server)");
  });

  it("connection dropped (down): the index", async () => {
    fx.assemble = { status: 200, body: {}, destroy: true };
    const r = await start("s", "startup", env(await newData(), ON));
    index(additionalContext(r));
    expect(r.stderr).toContain("context assemble failed (network)");
  });

  it("200 with an empty bundle: the index", async () => {
    fx.assemble = { status: 200, body: { status: "ok", bundle: { text: "  " } } };
    index(additionalContext(await start("s", "startup", env(await newData(), ON))));
  });
});

describe("setting OFF: the route is never called", () => {
  it.each([
    ["default", {}],
    ["index", { CLAUDE_PLUGIN_OPTION_CONTEXT_MODE: "index" }],
    ["full", { CLAUDE_PLUGIN_OPTION_CONTEXT_MODE: "full" }],
    ["garbage value", { CLAUDE_PLUGIN_OPTION_CONTEXT_MODE: "banana" }],
  ])("%s", async (_name, extra) => {
    const r = await start("s", "startup", env(await newData(), { ...extra, CLAUDE_PLUGIN_OPTION_ASSEMBLE_BUDGET_TOKENS: "1200" }));
    expect(r.code).toBe(0);
    expect(assembleCalls()).toHaveLength(0);
    expect(additionalContext(r)).not.toContain("assembled");
  });
});
