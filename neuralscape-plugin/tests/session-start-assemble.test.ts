/**
 * CONTEXT_MODE=assemble helpers: settings, request body, response parsing, the
 * failure kinds of fetchAssembledBundle, the query, and the rendering (no
 * double render of cards or rows the bundle already shows).
 */

import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

process.env.NEURALSCAPE_TEST = "1";

const post = vi.hoisted(() => ({ impl: null as null | ((path: string, body: Record<string, unknown>) => Promise<unknown>), calls: [] as unknown[] }));
vi.mock("../src/utils.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/utils.js")>();
  return {
    ...actual,
    neuralscapePost: (path: string, body: Record<string, unknown>) => {
      post.calls.push({ path, body });
      return post.impl ? post.impl(path, body) : Promise.resolve({});
    },
  };
});

const mod = await import("../src/hooks/session-start.js");
const { NeuralscapeHttpError, NeuralscapeNetworkError } = await import("../src/utils.js");
import type { ContextResponse } from "../src/utils.js";

const KEYS = [
  "CLAUDE_PLUGIN_OPTION_CONTEXT_MODE",
  "CLAUDE_PLUGIN_OPTION_ASSEMBLE_BUDGET_TOKENS",
  "CLAUDE_PLUGIN_OPTION_ASSEMBLE_QUERY",
  "NEURALSCAPE_ASSEMBLE_QUERY",
];
let saved: Record<string, string | undefined>;
beforeEach(() => {
  saved = Object.fromEntries(KEYS.map((k) => [k, process.env[k]]));
  for (const k of KEYS) delete process.env[k];
  post.impl = null;
  post.calls.length = 0;
});
afterEach(() => {
  for (const k of KEYS) {
    if (saved[k] === undefined) delete process.env[k];
    else process.env[k] = saved[k];
  }
});

describe("settings", () => {
  it("getContextMode knows assemble; unknown values stay index", () => {
    process.env.CLAUDE_PLUGIN_OPTION_CONTEXT_MODE = "Assemble";
    expect(mod.getContextMode()).toBe("assemble");
    process.env.CLAUDE_PLUGIN_OPTION_CONTEXT_MODE = "nope";
    expect(mod.getContextMode()).toBe("index");
  });
  it("budget: default, clamped to the route's bounds, garbage ignored", () => {
    expect(mod.getAssembleBudgetTokens()).toBe(1500);
    process.env.CLAUDE_PLUGIN_OPTION_ASSEMBLE_BUDGET_TOKENS = "20";
    expect(mod.getAssembleBudgetTokens()).toBe(100);
    process.env.CLAUDE_PLUGIN_OPTION_ASSEMBLE_BUDGET_TOKENS = "5000000";
    expect(mod.getAssembleBudgetTokens()).toBe(1_000_000);
    process.env.CLAUDE_PLUGIN_OPTION_ASSEMBLE_BUDGET_TOKENS = "2500";
    expect(mod.getAssembleBudgetTokens()).toBe(2500);
    for (const bad of ["abc", "-4", "0"]) {
      process.env.CLAUDE_PLUGIN_OPTION_ASSEMBLE_BUDGET_TOKENS = bad;
      expect(mod.getAssembleBudgetTokens()).toBe(1500);
    }
  });
  it("query: trimmed, capped at the route's 2000 characters", () => {
    expect(mod.getAssembleQuery()).toBeUndefined();
    process.env.CLAUDE_PLUGIN_OPTION_ASSEMBLE_QUERY = "  retry loop  ";
    expect(mod.getAssembleQuery()).toBe("retry loop");
    process.env.CLAUDE_PLUGIN_OPTION_ASSEMBLE_QUERY = "x".repeat(3000);
    expect(mod.getAssembleQuery()).toHaveLength(2000);
  });
});

describe("buildAssembleRequest", () => {
  it("builds the plain request, clamps and truncates", () => {
    expect(mod.buildAssembleRequest({ budgetTokens: 1500, userId: "u" })).toEqual({ budget_tokens: 1500, format: "plain", user_id: "u" });
    const body = mod.buildAssembleRequest({
      budgetTokens: 5.9e9,
      userId: "u",
      sessionId: "s".repeat(150),
      projectId: "p",
      query: "q".repeat(2500),
    });
    expect(body.budget_tokens).toBe(1_000_000);
    expect((body.session_id as string).length).toBe(100);
    expect(body.project_id).toBe("p");
    expect((body.query as string).length).toBe(2000);
    expect(mod.buildAssembleRequest({ budgetTokens: 3, userId: "u", query: "   " })).toEqual({ budget_tokens: 100, format: "plain", user_id: "u" });
  });
});

describe("parseAssembleResponse / assembledMemoryIds", () => {
  it("returns the plain bundle with its accounting, or null without text", () => {
    expect(mod.parseAssembleResponse({ bundle: { text: " hello " }, used_tokens: 5, budget_tokens: 100 })).toEqual({
      text: "hello",
      usedTokens: 5,
      budgetTokens: 100,
    });
    expect(mod.parseAssembleResponse({ bundle: { text: "x" } })).toEqual({ text: "x", usedTokens: 0, budgetTokens: 0 });
    for (const bad of [null, {}, { bundle: {} }, { bundle: { text: "  " } }, { bundle: { text: 4 } }, "x"]) {
      expect(mod.parseAssembleResponse(bad)).toBeNull();
    }
  });
  it("lists the memory ids of the bundle's rows", () => {
    const ids = mod.assembledMemoryIds("## Relevant\n- [a1] x\n- [b-2] y\nnot - [c3] row\n- [] z");
    expect([...ids].sort()).toEqual(["a1", "b-2"]);
  });
});

describe("fetchAssembledBundle", () => {
  const body = { budget_tokens: 1500, format: "plain", user_id: "u" };
  it("success", async () => {
    post.impl = async () => ({ bundle: { text: "T" }, used_tokens: 1, budget_tokens: 1500 });
    const out = await mod.fetchAssembledBundle(body);
    expect(out.bundle?.text).toBe("T");
    expect(out.notice).toBe("");
    expect(post.calls).toEqual([{ path: "/v1/context/assemble", body }]);
  });
  it.each([
    ["404", new NeuralscapeHttpError(404, "Not Found"), ""],
    ["400", new NeuralscapeHttpError(400, "Bad Request"), "HTTP 4xx"],
    ["422", new NeuralscapeHttpError(422, "x"), "HTTP 4xx"],
    ["401", new NeuralscapeHttpError(401, "x"), "HTTP 401/403"],
    ["403", new NeuralscapeHttpError(403, "x"), "HTTP 401/403"],
    ["500", new NeuralscapeHttpError(500, "x"), ""],
    ["network", new NeuralscapeNetworkError("down"), ""],
  ])("%s falls back (null bundle)", async (_n, error, noticePart) => {
    post.impl = async () => {
      throw error;
    };
    const out = await mod.fetchAssembledBundle(body);
    expect(out.bundle).toBeNull();
    if (noticePart) expect(out.notice).toContain(noticePart);
    else expect(out.notice).toBe("");
  });
  it("an empty bundle is null without a notice", async () => {
    post.impl = async () => ({ bundle: { text: "" } });
    expect(await mod.fetchAssembledBundle(body)).toEqual({ bundle: null, notice: "" });
  });
});

describe("resolveAssembleQuery", () => {
  const transcript = () => {
    const path = join(mkdtempSync(join(tmpdir(), "ns-aq-")), "t.jsonl");
    writeFileSync(
      path,
      [
        { type: "user", message: { role: "user", content: "first" } },
        { type: "user", message: { role: "user", content: "the key is <private>SENTINEL-A and more" } },
      ]
        .map((x) => JSON.stringify(x))
        .join("\n") + "\n",
    );
    return path;
  };
  it("the setting wins", async () => {
    process.env.CLAUDE_PLUGIN_OPTION_ASSEMBLE_QUERY = "fixed";
    expect(await mod.resolveAssembleQuery({ source: "compact", transcript_path: transcript() })).toBe("fixed");
  });
  it("on resume/compact/fork it is the last user prompt, redacted fail-closed", async () => {
    for (const source of ["compact", "resume", "fork"]) {
      const q = await mod.resolveAssembleQuery({ source, transcript_path: transcript() });
      expect(q).toContain("the key is");
      expect(q).not.toMatch(/SENTINEL/);
    }
  });
  it("a fresh session, no transcript, or an unreadable one sends none", async () => {
    expect(await mod.resolveAssembleQuery({ source: "startup", transcript_path: transcript() })).toBeUndefined();
    expect(await mod.resolveAssembleQuery({ source: "compact" })).toBeUndefined();
    expect(await mod.resolveAssembleQuery({ source: "compact", transcript_path: "/nonexistent/t.jsonl" })).toBeUndefined();
  });
});

describe("buildIndexContext with an assembled bundle", () => {
  const NOW = new Date("2026-11-16T12:00:00.000Z");
  const context = {
    standards: [{ id: "s1", memory: "Sign commits" }],
    categories: {
      convention: [
        { id: "hit", memory: "Retry jitter", category: "convention", created_at: "2026-11-16T11:00:00.000Z" },
        { id: "other", memory: "Small PRs", category: "convention", created_at: "2026-11-16T10:00:00.000Z" },
      ],
    },
  } as unknown as ContextResponse;
  const cards = [{ label: "user", lines: ["likes tests"] }];
  const base = { context, cards, codeGraphAvailable: false, budgetTokens: 1500, now: NOW };

  it("adds the bundle under its heading, drops the card block and the rows it shows", () => {
    const out = mod.buildIndexContext({
      ...base,
      assembled: { text: "## Identity card\nlikes tests\n\n- [hit] x", usedTokens: 40, budgetTokens: 1000 },
    });
    expect(out).toContain("## Session context (assembled, ~40 of 1000 tokens)");
    expect(out).not.toContain("## Identity Card (user)");
    expect(out).toContain("Sign commits");
    const index = out.slice(out.indexOf("## Memory Index"));
    expect(index).not.toContain("#hit");
    expect(index).toContain("#other");
  });
  it("without a bundle it is the plain index, cards included", () => {
    const out = mod.buildIndexContext(base);
    expect(out).toContain("## Identity Card (user)");
    expect(out).not.toContain("assembled");
    expect(out).toContain("#hit");
  });
});
