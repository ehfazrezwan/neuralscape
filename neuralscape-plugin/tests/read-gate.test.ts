/**
 * Unit tests for the File Read Gate (roadmap D3) and the remaining D4
 * capture-hygiene knobs: excluded-project globs, the PostToolUse skip-tools
 * list, the fail-loud transport-failure counter, and read-gate session state.
 *
 * The gate's decision matrix at the process level (deny JSON shape, exit
 * codes, once-per-session dedup across real invocations) is covered by the
 * subprocess suite in read-gate-subprocess.test.ts.
 */

import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeAll, describe, expect, it } from "vitest";

beforeAll(() => {
  process.env.NEURALSCAPE_TEST = "1";
});

const readGate = await import("../src/core/read-gate.js");
const {
  DEFAULT_READ_GATE_MIN_BYTES,
  DEFAULT_READ_GATE_TIME_BUDGET_MS,
  GATE_BYPASS_EXTENSIONS,
  MODIFY_OBSERVATION_TYPES,
  READ_GATE_FETCH_LIMIT,
  buildSteerOutput,
  distinctFileMentions,
  fileNameOf,
  isBypassedExtension,
  modifiedScore,
  pathTail,
  rankFileMemories,
  referencesFile,
  renderGateRow,
  renderReadGateContext,
  tailMatchRe,
} = readGate;

const preToolUse = await import("../src/hooks/pre-tool-use.js");
const { getReadGateEnabled, getReadGateMinBytes, getReadGateTimeBudgetMs, targetFilePath } = preToolUse;

const postToolUse = await import("../src/hooks/post-tool-use.js");
const { DENY_TOOLS, getSkipTools, shouldCapture } = postToolUse;

const sessionStart = await import("../src/hooks/session-start.js");
const { buildUnreachableNotice } = sessionStart;

const utils = await import("../src/utils.js");
const {
  DEFAULT_FAIL_LOUD_THRESHOLD,
  getExcludedProjectGlobs,
  getFailLoudThreshold,
  getTransportFailureCount,
  globToRegExp,
  isProjectExcluded,
  loadGatedFiles,
  loadReadGateIndexCache,
  recordGatedFile,
  recordTransportFailure,
  resetTransportFailures,
  saveReadGateIndexCache,
} = utils;

import type { NeuralscapeMemory } from "../src/utils.js";

// ── Env helpers ──────────────────────────────────────────────────

const TOUCHED_ENV = [
  "CLAUDE_PLUGIN_DATA",
  "CLAUDE_PLUGIN_OPTION_EXCLUDED_PROJECTS",
  "NEURALSCAPE_EXCLUDED_PROJECTS",
  "NS_EXCLUDED_PROJECTS",
  "CLAUDE_PLUGIN_OPTION_SKIP_TOOLS",
  "NEURALSCAPE_SKIP_TOOLS",
  "CLAUDE_PLUGIN_OPTION_READ_GATE_ENABLED",
  "NEURALSCAPE_READ_GATE_ENABLED",
  "CLAUDE_PLUGIN_OPTION_READ_GATE_MIN_BYTES",
  "NEURALSCAPE_READ_GATE_MIN_BYTES",
  "CLAUDE_PLUGIN_OPTION_FAIL_LOUD_THRESHOLD",
  "NEURALSCAPE_FAIL_LOUD_THRESHOLD",
];
const saved: Record<string, string | undefined> = {};
for (const k of TOUCHED_ENV) saved[k] = process.env[k];

afterEach(() => {
  for (const k of TOUCHED_ENV) {
    if (saved[k] === undefined) delete process.env[k];
    else process.env[k] = saved[k];
  }
});

// ── Fixtures ─────────────────────────────────────────────────────

const TARGET = "/repo/src/gate/target-module.ts";

function mem(overrides: Partial<NeuralscapeMemory>): NeuralscapeMemory {
  return {
    id: "m-x",
    memory: "unrelated content",
    created_at: "2026-07-01T10:00:00Z",
    ...overrides,
  };
}

const MODIFIED_BUGFIX = mem({
  id: "m-bugfix",
  memory: "Fixed the debounce race in src/gate/target-module.ts by pinning the timer to the event loop tick.",
  observation_type: "bugfix",
  title: "Fixed debounce race in target-module.ts",
  token_estimate: 120,
  created_at: "2026-07-02T09:00:00Z",
});

const MODIFIED_VERB_ONLY = mem({
  id: "m-verb",
  memory: "Updated gate/target-module.ts and three siblings (a.ts, b.ts, c.ts) to the new logging API.",
  observation_type: "discovery",
  created_at: "2026-07-03T09:00:00Z",
});

const READ_ONLY_SPECIFIC = mem({
  id: "m-read-specific",
  memory: "gate/target-module.ts exposes the retry queue; consumers must drain it before shutdown.",
  observation_type: "discovery",
  created_at: "2026-07-03T12:00:00Z",
});

const READ_ONLY_BROAD = mem({
  id: "m-read-broad",
  memory:
    "Survey: gate/target-module.ts, alpha.py, beta.go, gamma.rs and delta.java each satisfy the worker interface differently.",
  observation_type: "research_note",
  created_at: "2026-07-03T13:00:00Z",
});

const UNRELATED = mem({
  id: "m-unrelated",
  memory: "The deploy pipeline uses blue/green switchovers on Sundays.",
});

// ── Path helpers ─────────────────────────────────────────────────

describe("fileNameOf / pathTail", () => {
  it("extracts the basename across separators", () => {
    expect(fileNameOf("/a/b/c.ts")).toBe("c.ts");
    expect(fileNameOf("C:\\repo\\src\\c.ts")).toBe("c.ts");
    expect(fileNameOf("")).toBe("");
  });

  it("pathTail keeps the last n segments, empty when the path is shallower", () => {
    expect(pathTail("/repo/src/gate/target-module.ts")).toBe("gate/target-module.ts");
    expect(pathTail("/repo/src/gate/target-module.ts", 3)).toBe("src/gate/target-module.ts");
    // A single-segment path has no dir/basename tail — callers fall back to
    // the basename matcher.
    expect(pathTail("solo.ts")).toBe("");
  });
});

describe("isBypassedExtension", () => {
  it.each(["/x/pic.png", "/x/movie.MP4", "/x/font.woff2", "/x/archive.tar", "/x/doc.pdf", "/x/model.gguf"])(
    "bypasses binary/media: %s",
    (p) => {
      expect(isBypassedExtension(p)).toBe(true);
    },
  );

  it.each(["/x/code.ts", "/x/notes.md", "/x/conf.yaml", "/x/.env", "/x/Makefile", "/x/data.json"])(
    "gates text formats: %s",
    (p) => {
      expect(isBypassedExtension(p)).toBe(false);
    },
  );

  it("exposes a non-trivial extension set", () => {
    expect(GATE_BYPASS_EXTENSIONS.has("png")).toBe(true);
    expect(GATE_BYPASS_EXTENSIONS.has("ts")).toBe(false);
  });
});

// ── Reference + ranking signals ──────────────────────────────────

describe("referencesFile (path-tail matching, audit 27 #32)", () => {
  it("accepts dir/basename tail mentions in content", () => {
    expect(referencesFile(MODIFIED_BUGFIX, TARGET)).toBe(true);
    expect(referencesFile(mem({ memory: "fixed gate/target-module.ts today" }), TARGET)).toBe(true);
  });

  it("REJECTS bare-basename mentions when the target path has directories", () => {
    // The pre-audit basename matcher fired on every same-named file in the
    // repo; a bare `target-module.ts` no longer counts as a reference to
    // /repo/src/gate/target-module.ts.
    expect(referencesFile(mem({ memory: "notes on target-module.ts alone" }), TARGET)).toBe(false);
    expect(referencesFile(mem({ memory: "x", title: "notes on target-module.ts" }), TARGET)).toBe(false);
  });

  it("falls back to the basename for single-segment target paths", () => {
    expect(referencesFile(mem({ memory: "notes on solo.ts here" }), "solo.ts")).toBe(true);
    expect(referencesFile(mem({ memory: "notes on solo.tsx here" }), "solo.ts")).toBe(false);
  });

  it("accepts mentions via title or tags", () => {
    expect(referencesFile(mem({ memory: "x", title: "notes on gate/target-module.ts" }), TARGET)).toBe(true);
    expect(referencesFile(mem({ memory: "x", tags: ["gate/target-module.ts"] }), TARGET)).toBe(true);
  });

  it("is case-insensitive and separator-tolerant", () => {
    expect(referencesFile(mem({ memory: "See GATE/TARGET-MODULE.TS for details" }), TARGET)).toBe(true);
    expect(referencesFile(mem({ memory: "See gate\\target-module.ts on Windows" }), TARGET)).toBe(true);
  });

  it("rejects memories that never mention the file", () => {
    expect(referencesFile(UNRELATED, TARGET)).toBe(false);
  });

  it("rejects substring false positives", () => {
    // tail `repo/a.ts` must NOT match `repo/data.ts`
    expect(referencesFile(mem({ memory: "refactored repo/data.ts today" }), "/repo/a.ts")).toBe(false);
    // tail must NOT match the `.tsx` extension superset
    expect(referencesFile(mem({ memory: "see gate/target-module.tsx for the JSX port" }), TARGET)).toBe(false);
    // ...nor a different dotted file in the same dir
    expect(referencesFile(mem({ memory: "generated gate/x.target-module.ts stub" }), TARGET)).toBe(false);
    // ...nor a partial dir-segment prefix (irrigate/ vs gate/)
    expect(referencesFile(mem({ memory: "see irrigate/target-module.ts" }), TARGET)).toBe(false);
  });

  it("accepts token matches with longer path prefixes and sentence punctuation", () => {
    expect(referencesFile(mem({ memory: "fixed src/gate/target-module.ts." }), TARGET)).toBe(true);
    expect(referencesFile(mem({ memory: "(gate/target-module.ts)" }), TARGET)).toBe(true);
  });

  it("tailMatchRe is exposed for the ranking pass", () => {
    expect(tailMatchRe("gate/target-module.ts").test("in src/gate/target-module.ts today")).toBe(true);
    expect(tailMatchRe("gate/target-module.ts").test("in target-module.ts alone")).toBe(false);
  });
});

describe("modifiedScore", () => {
  it("scores modify-typed observations highest", () => {
    for (const t of MODIFY_OBSERVATION_TYPES) {
      expect(modifiedScore(mem({ observation_type: t, memory: "plain text" }))).toBe(2);
    }
  });

  it("scores modify verbs in content as 1", () => {
    expect(modifiedScore(MODIFIED_VERB_ONLY)).toBe(1);
  });

  it("scores read-only mentions 0", () => {
    expect(modifiedScore(READ_ONLY_SPECIFIC)).toBe(0);
  });
});

describe("distinctFileMentions", () => {
  it("counts distinct file basenames", () => {
    expect(distinctFileMentions("a.ts then src/a.ts again, plus b.py")).toBe(2);
  });

  it("ignores version numbers (no letter-led extension)", () => {
    expect(distinctFileMentions("upgraded to 2.5 and 3.14")).toBe(0);
  });

  it("handles empty content", () => {
    expect(distinctFileMentions("")).toBe(0);
    expect(distinctFileMentions(null)).toBe(0);
  });
});

describe("rankFileMemories", () => {
  it("filters to verified references only", () => {
    const ranked = rankFileMemories([UNRELATED, MODIFIED_BUGFIX], TARGET);
    expect(ranked.map((m) => m.id)).toEqual(["m-bugfix"]);
  });

  it("ranks modified over merely-read, then specificity, newest last tiebreak", () => {
    const ranked = rankFileMemories(
      [READ_ONLY_BROAD, READ_ONLY_SPECIFIC, MODIFIED_VERB_ONLY, MODIFIED_BUGFIX, UNRELATED],
      TARGET,
    );
    expect(ranked.map((m) => m.id)).toEqual([
      "m-bugfix", // observation_type modify (2)
      "m-verb", // verb modify (1)
      "m-read-specific", // read-only but mentions 1 file
      "m-read-broad", // read-only, mentions 5 files
    ]);
  });

  it("caps the result list", () => {
    const many = Array.from({ length: 30 }, (_, i) =>
      mem({ id: `m-${i}`, memory: `note ${i} about gate/target-module.ts` }),
    );
    expect(rankFileMemories(many, TARGET)).toHaveLength(10);
    expect(rankFileMemories(many, TARGET, 3)).toHaveLength(3);
  });

  it("ranks title/tag-referenced memories on the same haystack as verification", () => {
    // References the file ONLY via title — must not be scored as
    // maximally non-specific and pushed below a broad content mention.
    const titleOnly = mem({
      id: "m-title-only",
      memory: "The retry queue must be drained before shutdown.",
      title: "gate/target-module.ts retry-queue contract",
      created_at: "2026-07-01T00:00:00Z",
    });
    const broad = mem({
      id: "m-broad",
      memory:
        "Survey: gate/target-module.ts, alpha.py, beta.go, gamma.rs and delta.java each satisfy the worker interface.",
      created_at: "2026-07-03T00:00:00Z",
    });
    const ranked = rankFileMemories([broad, titleOnly], TARGET);
    expect(ranked.map((m) => m.id)).toEqual(["m-title-only", "m-broad"]);
  });

  it("orders equal scores by recency", () => {
    const older = mem({ id: "m-old", memory: "gate/target-module.ts holds the retry queue.", created_at: "2026-06-01T00:00:00Z" });
    const newer = mem({ id: "m-new", memory: "gate/target-module.ts holds the retry queue.", created_at: "2026-07-01T00:00:00Z" });
    const undated = mem({ id: "m-undated", memory: "gate/target-module.ts holds the retry queue.", created_at: undefined });
    const ranked = rankFileMemories([older, undated, newer], TARGET);
    expect(ranked.map((m) => m.id)).toEqual(["m-new", "m-old", "m-undated"]);
  });
});

// ── Rendering + steering shape (audit 27 #32: steer, never block) ─

describe("renderGateRow", () => {
  it("renders `#id | when | title | ~tokens`", () => {
    const now = new Date("2026-07-04T09:00:00Z");
    const row = renderGateRow(MODIFIED_BUGFIX, now);
    expect(row).toBe("#m-bugfix | 2d | 🐛 Fixed debounce race in target-module.ts | ~120");
  });

  it("falls back to distilled title and estimated tokens", () => {
    const row = renderGateRow(READ_ONLY_SPECIFIC, new Date("2026-07-04T09:00:00Z"));
    expect(row).toContain("#m-read-specific | ");
    expect(row).toMatch(/~\d+$/);
    expect(row).toContain("target-module.ts exposes the retry queue");
  });
});

describe("renderReadGateContext", () => {
  const now = new Date("2026-07-04T09:00:00Z");

  it("includes the path, rows, and escalation menu — and never a skip/override framing", () => {
    const context = renderReadGateContext(TARGET, [MODIFIED_BUGFIX, READ_ONLY_SPECIFIC], now);
    expect(context).toContain("[Neuralscape]");
    expect(context).toContain("`" + TARGET + "`");
    expect(context).toContain("2 stored memories reference");
    expect(context).toContain("`#id | when | title | ~tokens`");
    expect(context).toContain("#m-bugfix | 2d |");
    expect(context).toContain("mcp__plugin_neuralscape_neuralscape__get_memories");
    expect(context).toContain("mcp__plugin_neuralscape_neuralscape__timeline");
    expect(context).toContain("index_only: true");
    // The Read is NOT skipped or substituted — no deny/override language.
    expect(context).not.toContain("Skipped reading");
    expect(context).not.toContain("Read the same path again");
  });

  it("uses singular phrasing for one memory", () => {
    const context = renderReadGateContext(TARGET, [MODIFIED_BUGFIX], now);
    expect(context).toContain("1 stored memory references");
  });

  it("redacts <private> spans coming back from the server", () => {
    const leaky = mem({
      id: "m-leak",
      memory: "gate/target-module.ts stores the key <private>sk-super-secret</private> at boot.",
    });
    const context = renderReadGateContext(TARGET, [leaky], now);
    expect(context).not.toContain("sk-super-secret");
    expect(context).toContain("[redacted]");
  });
});

describe("buildSteerOutput", () => {
  it("emits additionalContext with NO permission decision (steer, never block)", () => {
    const out = buildSteerOutput("ctx") as {
      hookSpecificOutput: Record<string, unknown>;
    };
    expect(out).toEqual({
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        additionalContext: "ctx",
      },
    });
    expect(out.hookSpecificOutput.permissionDecision).toBeUndefined();
  });
});

describe("hot-path budgets (audit 27 #31)", () => {
  it("caps the one session fetch at a sane row count", () => {
    expect(READ_GATE_FETCH_LIMIT).toBeLessThanOrEqual(200);
  });

  it("has a hard time budget with a config override", () => {
    expect(DEFAULT_READ_GATE_TIME_BUDGET_MS).toBe(2000);
    expect(getReadGateTimeBudgetMs()).toBe(2000);
    process.env.NEURALSCAPE_READ_GATE_TIME_BUDGET_MS = "750";
    expect(getReadGateTimeBudgetMs()).toBe(750);
    process.env.NEURALSCAPE_READ_GATE_TIME_BUDGET_MS = "banana";
    expect(getReadGateTimeBudgetMs()).toBe(2000);
    delete process.env.NEURALSCAPE_READ_GATE_TIME_BUDGET_MS;
  });
});

// ── Hook config knobs ────────────────────────────────────────────

describe("read-gate config knobs", () => {
  it("is enabled by default", () => {
    expect(getReadGateEnabled()).toBe(true);
  });

  it.each(["false", "0", "off", "no", "FALSE"])("disables on %s", (v) => {
    process.env.NEURALSCAPE_READ_GATE_ENABLED = v;
    expect(getReadGateEnabled()).toBe(false);
  });

  it("defaults min bytes to 1500", () => {
    expect(DEFAULT_READ_GATE_MIN_BYTES).toBe(1500);
    expect(getReadGateMinBytes()).toBe(1500);
  });

  it("honors READ_GATE_MIN_BYTES overrides and rejects garbage", () => {
    process.env.NEURALSCAPE_READ_GATE_MIN_BYTES = "5000";
    expect(getReadGateMinBytes()).toBe(5000);
    process.env.NEURALSCAPE_READ_GATE_MIN_BYTES = "banana";
    expect(getReadGateMinBytes()).toBe(1500);
  });
});

describe("targetFilePath", () => {
  it("returns the trimmed path for Read events", () => {
    expect(targetFilePath({ tool_name: "Read", tool_input: { file_path: " /a/b.ts " } })).toBe("/a/b.ts");
  });

  it("returns null for other tools or missing paths", () => {
    expect(targetFilePath({ tool_name: "Edit", tool_input: { file_path: "/a/b.ts" } })).toBeNull();
    expect(targetFilePath({ tool_name: "Read", tool_input: {} })).toBeNull();
    expect(targetFilePath({ tool_name: "Read", tool_input: { file_path: "  " } })).toBeNull();
    expect(targetFilePath({})).toBeNull();
  });
});

// ── D4: excluded-project globs ───────────────────────────────────

describe("excluded projects", () => {
  it("globToRegExp anchors and supports * and ?", () => {
    expect(globToRegExp("scratch-*").test("scratch-42")).toBe(true);
    expect(globToRegExp("scratch-*").test("my-scratch-42")).toBe(false);
    expect(globToRegExp("proj?").test("proj1")).toBe(true);
    expect(globToRegExp("proj?").test("proj12")).toBe(false);
    // regex metacharacters in ids are literal
    expect(globToRegExp("a.b").test("a.b")).toBe(true);
    expect(globToRegExp("a.b").test("aXb")).toBe(false);
  });

  it("reads comma-separated globs from config", () => {
    process.env.NEURALSCAPE_EXCLUDED_PROJECTS = "secret-*, scratch , ";
    expect(getExcludedProjectGlobs()).toEqual(["secret-*", "scratch"]);
  });

  it("supports the NS_EXCLUDED_PROJECTS alias", () => {
    process.env.NS_EXCLUDED_PROJECTS = "alias-project";
    expect(isProjectExcluded("alias-project")).toBe(true);
  });

  it("matches case-insensitively and passes non-matches", () => {
    process.env.NEURALSCAPE_EXCLUDED_PROJECTS = "Secret-*";
    expect(isProjectExcluded("secret-lab")).toBe(true);
    expect(isProjectExcluded("public-lab")).toBe(false);
  });

  it("never excludes an unresolved project id", () => {
    process.env.NEURALSCAPE_EXCLUDED_PROJECTS = "*";
    expect(isProjectExcluded(undefined)).toBe(false);
  });

  it("shouldCapture drops events from excluded projects", async () => {
    const dir = await mkdtemp(join(tmpdir(), "ns-excl-"));
    const projectId = dir.split("/").filter(Boolean).pop()!;
    process.env.NEURALSCAPE_EXCLUDED_PROJECTS = projectId;
    expect(shouldCapture({ tool_name: "Edit", tool_input: { file_path: "/x" }, cwd: dir })).toBe(false);
    delete process.env.NEURALSCAPE_EXCLUDED_PROJECTS;
    expect(shouldCapture({ tool_name: "Edit", tool_input: { file_path: "/x" }, cwd: dir })).toBe(true);
  });
});

// ── D4: skip-tools list ──────────────────────────────────────────

describe("skip-tools list", () => {
  it("skips harness plumbing by default (inspected from real buffers)", () => {
    for (const t of ["Skill", "ToolSearch", "AskUserQuestion", "TaskUpdate", "TaskStop", "Workflow", "Monitor", "WaitForMcpServers"]) {
      expect(DENY_TOOLS.has(t)).toBe(true);
      expect(shouldCapture({ tool_name: t })).toBe(false);
    }
  });

  it("SKIP_TOOLS config adds tools without replacing the defaults", () => {
    process.env.NEURALSCAPE_SKIP_TOOLS = "SendMessage , CustomTool";
    const skip = getSkipTools();
    expect(skip.has("SendMessage")).toBe(true);
    expect(skip.has("CustomTool")).toBe(true);
    expect(skip.has("Read")).toBe(true); // defaults preserved
    expect(shouldCapture({ tool_name: "SendMessage" })).toBe(false);
  });

  it("returns the default set when SKIP_TOOLS is empty", () => {
    expect(getSkipTools()).toBe(DENY_TOOLS);
  });
});

// ── D4: fail-loud counter + notice ───────────────────────────────

describe("fail-loud transport-failure counter", () => {
  it("increments consecutively and resets on success", async () => {
    const dir = await mkdtemp(join(tmpdir(), "ns-failloud-"));
    process.env.CLAUDE_PLUGIN_DATA = dir;
    expect(await getTransportFailureCount()).toBe(0);
    expect(await recordTransportFailure()).toBe(1);
    expect(await recordTransportFailure()).toBe(2);
    expect(await getTransportFailureCount()).toBe(2);
    await resetTransportFailures();
    expect(await getTransportFailureCount()).toBe(0);
    // reset on an already-clean state is a no-op
    await resetTransportFailures();
    expect(await getTransportFailureCount()).toBe(0);
  });

  it("treats a corrupt counter file as zero", async () => {
    const dir = await mkdtemp(join(tmpdir(), "ns-failloud-"));
    process.env.CLAUDE_PLUGIN_DATA = dir;
    await recordTransportFailure(); // creates the observations dir + file
    await writeFile(join(dir, "observations", "unreachable.count"), "not-a-number", "utf-8");
    expect(await getTransportFailureCount()).toBe(0);
  });

  it("threshold defaults to 3 and honors config", () => {
    expect(DEFAULT_FAIL_LOUD_THRESHOLD).toBe(3);
    expect(getFailLoudThreshold()).toBe(3);
    process.env.NEURALSCAPE_FAIL_LOUD_THRESHOLD = "5";
    expect(getFailLoudThreshold()).toBe(5);
    process.env.NEURALSCAPE_FAIL_LOUD_THRESHOLD = "zero";
    expect(getFailLoudThreshold()).toBe(3);
  });

  it("buildUnreachableNotice stays generic below the threshold", () => {
    const notice = buildUnreachableNotice(1, 3);
    expect(notice).toContain("memory service unreachable");
    expect(notice).not.toContain("docker compose ps");
  });

  it("buildUnreachableNotice fails loud at the threshold", () => {
    const notice = buildUnreachableNotice(3, 3);
    expect(notice).toContain("unreachable for 3 consecutive events");
    expect(notice).toContain("docker compose ps");
  });
});

// ── D3: read-gate session state (once per file per session) ─────

describe("read-gate session state", () => {
  it("records and reloads gated paths per session", async () => {
    const dir = await mkdtemp(join(tmpdir(), "ns-gatestate-"));
    process.env.CLAUDE_PLUGIN_DATA = dir;
    expect((await loadGatedFiles("sess-a")).size).toBe(0);
    await recordGatedFile("sess-a", "/repo/a.ts");
    await recordGatedFile("sess-a", "/repo/b.ts");
    await recordGatedFile("sess-a", "/repo/a.ts"); // idempotent
    const gated = await loadGatedFiles("sess-a");
    expect(gated.has("/repo/a.ts")).toBe(true);
    expect(gated.has("/repo/b.ts")).toBe(true);
    expect(gated.size).toBe(2);
    // sessions are isolated
    expect((await loadGatedFiles("sess-b")).size).toBe(0);
    // persisted as a JSON array on disk
    const raw = JSON.parse(await readFile(join(dir, "observations", "sess-a.readgate.json"), "utf-8"));
    expect(Array.isArray(raw)).toBe(true);
  });

  it("survives a corrupt state file", async () => {
    const dir = await mkdtemp(join(tmpdir(), "ns-gatestate-"));
    process.env.CLAUDE_PLUGIN_DATA = dir;
    await recordGatedFile("sess-c", "/repo/a.ts");
    await writeFile(join(dir, "observations", "sess-c.readgate.json"), "{corrupt", "utf-8");
    expect((await loadGatedFiles("sess-c")).size).toBe(0);
    // non-array JSON is also treated as empty
    await writeFile(join(dir, "observations", "sess-c.readgate.json"), '{"a":1}', "utf-8");
    expect((await loadGatedFiles("sess-c")).size).toBe(0);
  });

  it("sanitizes hostile session ids in state paths", async () => {
    const dir = await mkdtemp(join(tmpdir(), "ns-gatestate-"));
    process.env.CLAUDE_PLUGIN_DATA = dir;
    await recordGatedFile("../../evil", "/repo/a.ts");
    const gated = await loadGatedFiles("../../evil");
    expect(gated.has("/repo/a.ts")).toBe(true);
  });

  it("index cache is keyed by session AND project (Copilot, PR #126)", async () => {
    const dir = await mkdtemp(join(tmpdir(), "ns-gatecache-"));
    process.env.CLAUDE_PLUGIN_DATA = dir;
    const rows = [mem({ id: "m-proj-a", memory: "gate/target-module.ts note" })];
    await saveReadGateIndexCache("sess-p", "proj-a", true, rows);
    // A different project in the SAME session must not see proj-a's rows —
    // it triggers its own fresh fetch instead of reusing wrong-project cache.
    expect(await loadReadGateIndexCache("sess-p", "proj-b")).toBeNull();
    expect(await loadReadGateIndexCache("sess-p", undefined)).toBeNull();
    const hit = await loadReadGateIndexCache("sess-p", "proj-a");
    expect(hit?.ok).toBe(true);
    expect(hit?.rows[0]?.id).toBe("m-proj-a");
    // The failure sentinel is project-keyed too.
    await saveReadGateIndexCache("sess-p", "proj-b", false, []);
    expect((await loadReadGateIndexCache("sess-p", "proj-b"))?.ok).toBe(false);
    expect((await loadReadGateIndexCache("sess-p", "proj-a"))?.ok).toBe(true);
  });
});
