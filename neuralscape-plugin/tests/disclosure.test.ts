/**
 * Tests for the progressive-disclosure renderers (roadmap D1/D2).
 *
 * Fixture-driven: index rows render day-grouped under a token budget,
 * the savings header reports the measured counterfactual, the escalation
 * footer teaches the 3-layer workflow (+ F2 code-graph deferral), and the
 * session-note round-trip (server body → fields → "Previously…") holds.
 */

import { beforeAll, describe, expect, it } from "vitest";

beforeAll(() => {
  process.env.NEURALSCAPE_TEST = "1";
});

const disclosure = await import("../src/core/disclosure.js");
const {
  DEFAULT_INDEX_BUDGET_TOKENS,
  dayLabel,
  distillTitle,
  estimateTokens,
  findLatestSessionNote,
  glyphFor,
  humanizeAge,
  isSessionNote,
  parseSessionNoteBody,
  renderCardBlock,
  renderEscalationFooter,
  renderIndexTable,
  renderPreviously,
  renderSavingsHeader,
  timeLabel,
  toIndexEntry,
} = disclosure;

const sessionStart = await import("../src/hooks/session-start.js");
const {
  buildIndexContext,
  formatMemories,
  formatStandards,
  getContextMode,
  getIndexBudgetTokens,
  getCodeGraphMode,
} = sessionStart;

// A fixed "now" so day grouping is deterministic (local-time based).
const NOW = new Date(2026, 6, 2, 15, 0, 0); // 2026-07-02 15:00 local

function isoAt(daysAgo: number, hour = 10, minute = 30): string {
  const d = new Date(NOW.getTime() - daysAgo * 24 * 60 * 60 * 1000);
  d.setHours(hour, minute, 0, 0);
  return d.toISOString();
}

function mem(over: Record<string, unknown> = {}) {
  return {
    id: "3f2a91c4-0000-4000-8000-000000000001",
    memory: "Chose ARQ over Celery for the background queue because of asyncio-native workers.",
    category: "decision",
    observation_type: "decision",
    created_at: isoAt(0),
    title: null,
    token_estimate: null,
    tags: [],
    ...over,
  } as any;
}

// ── Primitives ───────────────────────────────────────────────────

describe("estimateTokens", () => {
  it("is ceil(len/4) with floor 1", () => {
    expect(estimateTokens("")).toBe(1);
    expect(estimateTokens("abcd")).toBe(1);
    expect(estimateTokens("abcde")).toBe(2);
    expect(estimateTokens(null)).toBe(1);
  });
});

describe("distillTitle", () => {
  it("takes the first sentence, clipped to 10 words", () => {
    expect(distillTitle("Pin pydantic to >=2.5. The validator changed.")).toBe(
      "Pin pydantic to >=2.5",
    );
  });

  it("clips long sentences with an ellipsis", () => {
    const words = Array.from({ length: 20 }, (_, i) => `word${i}`).join(" ");
    const title = distillTitle(words);
    expect(title.endsWith("…")).toBe(true);
    expect(title.split(/\s+/).length).toBeLessThanOrEqual(11); // 10 words + ellipsis
  });

  it("strips markdown noise", () => {
    expect(distillTitle("## - The fix works")).toBe("The fix works");
  });

  it("handles empty content", () => {
    expect(distillTitle("")).toBe("(untitled)");
    expect(distillTitle(null)).toBe("(untitled)");
  });
});

describe("glyphFor", () => {
  it("maps known observation types and defaults the rest", () => {
    expect(glyphFor("decision")).toBe("⚖");
    expect(glyphFor("gotcha")).toBe("⚠");
    expect(glyphFor("unknown_type")).toBe("·");
    expect(glyphFor(null)).toBe("·");
  });
});

describe("dayLabel / timeLabel", () => {
  it("labels today and yesterday", () => {
    expect(dayLabel(isoAt(0), NOW)).toBe("Today");
    expect(dayLabel(isoAt(1), NOW)).toBe("Yesterday");
  });

  it("labels older days with weekday + date", () => {
    expect(dayLabel(isoAt(3), NOW)).toMatch(/^\w{3} \d{4}-\d{2}-\d{2}$/);
  });

  it("handles missing/garbage timestamps", () => {
    expect(dayLabel(null, NOW)).toBe("Undated");
    expect(dayLabel("not-a-date", NOW)).toBe("Undated");
    expect(timeLabel(null)).toBe("--:--");
    expect(timeLabel("garbage")).toBe("--:--");
  });
});

describe("humanizeAge", () => {
  it("mirrors the service buckets", () => {
    expect(humanizeAge(new Date(NOW.getTime() - 30_000).toISOString(), NOW)).toBe("now");
    expect(humanizeAge(new Date(NOW.getTime() - 5 * 60_000).toISOString(), NOW)).toBe("5m");
    expect(humanizeAge(isoAt(2), NOW)).toBe("2d");
    expect(humanizeAge(null, NOW)).toBe("?");
  });
});

// ── toIndexEntry ─────────────────────────────────────────────────

describe("toIndexEntry", () => {
  it("prefers server-stamped title and token_estimate", () => {
    const entry = toIndexEntry(mem({ title: "Server title", token_estimate: 99 }));
    expect(entry.title).toBe("Server title");
    expect(entry.tokens).toBe(99);
  });

  it("falls back to client distillation for legacy memories", () => {
    const entry = toIndexEntry(mem());
    expect(entry.title).toContain("Chose ARQ over Celery");
    expect(entry.tokens).toBeGreaterThan(1);
    expect(entry.glyph).toBe("⚖");
  });
});

// ── renderIndexTable ─────────────────────────────────────────────

describe("renderIndexTable", () => {
  const entries = [
    toIndexEntry(mem({ id: "id-today", created_at: isoAt(0, 14, 32) })),
    toIndexEntry(
      mem({
        id: "id-yesterday",
        created_at: isoAt(1),
        memory: "The docling container needs 2GB memory or PDF parses die silently.",
        observation_type: "gotcha",
        category: "dependency",
      }),
    ),
    toIndexEntry(mem({ id: "id-older", created_at: isoAt(4) })),
  ];

  it("renders day-grouped rows, newest first", () => {
    const r = renderIndexTable(entries, { now: NOW });
    const text = r.text;
    expect(text.indexOf("### Today")).toBeGreaterThanOrEqual(0);
    expect(text.indexOf("### Today")).toBeLessThan(text.indexOf("### Yesterday"));
    expect(text.indexOf("### Yesterday")).toBeLessThan(text.indexOf("#id-older"));
    expect(text).toContain("#id-today | 14:32 | ⚖ decision |");
    expect(text).toContain("⚠ dependency");
    expect(r.included).toBe(3);
    expect(r.total).toBe(3);
  });

  it("reports served vs baseline tokens", () => {
    const r = renderIndexTable(entries, { now: NOW });
    expect(r.servedTokens).toBeGreaterThan(0);
    expect(r.baselineTokens).toBe(entries.reduce((s, e) => s + e.tokens, 0));
  });

  it("enforces the token budget and points at index-first recall", () => {
    const many = Array.from({ length: 100 }, (_, i) =>
      toIndexEntry(mem({ id: `id-${i}`, created_at: isoAt(0, 9, i % 60) })),
    );
    const r = renderIndexTable(many, { budgetTokens: 200, now: NOW });
    expect(r.included).toBeLessThan(100);
    expect(r.servedTokens).toBeLessThanOrEqual(200 + 40); // omission-line slack
    expect(r.text).toContain("more not shown");
    expect(r.text).toContain("index_only=true");
  });

  it("counts every rendered line — including the omission line — in servedTokens", () => {
    const many = Array.from({ length: 100 }, (_, i) =>
      toIndexEntry(mem({ id: `id-${i}`, created_at: isoAt(0, 9, i % 60) })),
    );
    const r = renderIndexTable(many, { budgetTokens: 200, now: NOW });
    const perLineTokens = r.text
      .split("\n")
      .reduce((s, l) => s + Math.max(1, Math.ceil(l.length / 4)), 0);
    expect(r.servedTokens).toBe(perLineTokens);
  });

  it("always includes at least one row even under a tiny budget", () => {
    const r = renderIndexTable(entries, { budgetTokens: 1, now: NOW });
    expect(r.included).toBe(1);
  });

  it("uses the default budget when none is given", () => {
    expect(DEFAULT_INDEX_BUDGET_TOKENS).toBe(1500);
    const r = renderIndexTable(entries);
    expect(r.included).toBe(3);
  });

  it("groups undated entries last", () => {
    const r = renderIndexTable(
      [toIndexEntry(mem({ id: "id-undated", created_at: null })), ...entries],
      { now: NOW },
    );
    expect(r.text.indexOf("### Undated")).toBeGreaterThan(r.text.indexOf("#id-older"));
  });
});

describe("renderSavingsHeader", () => {
  it("reports N memories, served vs full, and percent saved", () => {
    const header = renderSavingsHeader({
      text: "",
      included: 28,
      total: 28,
      servedTokens: 950,
      baselineTokens: 14200,
    });
    expect(header).toBe("index: 28 memories, ~950 tokens vs ~14200 full (93% saved)");
  });

  it("shows included-of-total when the budget clipped rows", () => {
    const header = renderSavingsHeader({
      text: "",
      included: 10,
      total: 40,
      servedTokens: 500,
      baselineTokens: 5000,
    });
    expect(header).toContain("10 of 40 memories");
  });
});

// ── Escalation footer (+ F2) ─────────────────────────────────────

describe("renderEscalationFooter", () => {
  it("teaches index → search → get_memories → timeline", () => {
    const footer = renderEscalationFooter(false);
    expect(footer).toContain("recall_memories");
    expect(footer).toContain("index_only");
    expect(footer).toContain("get_memories");
    expect(footer).toContain("timeline");
    expect(footer).toContain("mcp__plugin_neuralscape_neuralscape__");
    expect(footer).not.toContain("query_code_graph");
  });

  it("steers with guidance, not absolutes (audit 27 #33)", () => {
    const footer = renderEscalationFooter(false);
    expect(footer).not.toMatch(/never expand/i);
    expect(footer).toContain("titles are lossy");
  });

  it("adds the F2 code-graph deferral policy when a graph is available", () => {
    const footer = renderEscalationFooter(true);
    expect(footer).toContain("query_code_graph");
    expect(footer).toContain("get_code_neighbors");
    expect(footer).toContain("code_path");
    expect(footer).toMatch(/structure rots/i);
    expect(footer).toMatch(/NOT store purely structural/i);
  });
});

// ── Session-note parsing + Previously (D2) ───────────────────────

const NOTE_BODY = [
  "Session note:",
  "Request: Fix the transcript offset bug",
  "Investigated: worked in 3 file(s) (adapters/claude-code.ts, core/flush.ts, utils.ts); ran 9 command(s)",
  "Learned: The offset must be committed after flushTurns resolves.",
  "Completed: Shipped the pendingOffsets staging fix",
  "Next steps: Add an integration test for the crash-mid-flush path",
].join("\n");

describe("parseSessionNoteBody", () => {
  it("round-trips the server-rendered body", () => {
    const note = parseSessionNoteBody(NOTE_BODY)!;
    expect(note.request).toBe("Fix the transcript offset bug");
    expect(note.investigated).toContain("3 file(s)");
    expect(note.learned).toContain("committed after flushTurns");
    expect(note.completed).toBe("Shipped the pendingOffsets staging fix");
    expect(note.next_steps).toBe("Add an integration test for the crash-mid-flush path");
  });

  it("handles multi-line field values", () => {
    const note = parseSessionNoteBody(
      "Session note:\nNext steps: line one\ncontinued line two\nRequest: r",
    )!;
    expect(note.next_steps).toBe("line one\ncontinued line two");
    expect(note.request).toBe("r");
  });

  it("returns null for non-note content", () => {
    expect(parseSessionNoteBody("Just a normal memory.")).toBeNull();
    expect(parseSessionNoteBody("")).toBeNull();
    expect(parseSessionNoteBody(null)).toBeNull();
    expect(parseSessionNoteBody("Session note:\n")).toBeNull();
  });
});

describe("isSessionNote / findLatestSessionNote", () => {
  it("detects by tag or body prefix", () => {
    expect(isSessionNote(mem({ tags: ["session_note"] }))).toBe(true);
    expect(isSessionNote(mem({ memory: "Session note:\nRequest: r" }))).toBe(true);
    expect(isSessionNote(mem())).toBe(false);
  });

  it("picks the newest note across categories", () => {
    const categories = {
      task_context: [
        mem({ id: "old-note", memory: NOTE_BODY, tags: ["session_note"], created_at: isoAt(3) }),
        mem({ id: "new-note", memory: NOTE_BODY, tags: ["session_note"], created_at: isoAt(1) }),
      ],
      decision: [mem({ id: "not-a-note" })],
    };
    expect(findLatestSessionNote(categories)!.id).toBe("new-note");
  });

  it("returns null when there is no note", () => {
    expect(findLatestSessionNote({ decision: [mem()] })).toBeNull();
  });
});

describe("renderPreviously", () => {
  it("puts next_steps first", () => {
    const block = renderPreviously(parseSessionNoteBody(NOTE_BODY)!, "last session, 2d ago");
    expect(block).toContain("## Previously (last session, 2d ago)");
    const nextIdx = block.indexOf("**Next steps:**");
    expect(nextIdx).toBeGreaterThan(0);
    expect(nextIdx).toBeLessThan(block.indexOf("**Request:**"));
    expect(block.indexOf("**Request:**")).toBeLessThan(block.indexOf("**Investigated:**"));
  });

  it("returns empty for an empty note", () => {
    expect(renderPreviously({})).toBe("");
  });
});

describe("renderCardBlock", () => {
  it("renders each card under its label", () => {
    const block = renderCardBlock([
      { label: "project: neuralscape", lines: ["IDENTITY: A memory service", "INSTRUCTION: Never commit to dev"] },
      { label: "user", lines: ["IDENTITY: Ehfaz, builder of Neuralscape"] },
    ]);
    expect(block).toContain("## Identity Card (project: neuralscape)");
    expect(block).toContain("INSTRUCTION: Never commit to dev");
    expect(block).toContain("## Identity Card (user)");
  });

  it("skips empty cards", () => {
    expect(renderCardBlock([{ label: "user", lines: [] }])).toBe("");
    expect(renderCardBlock([])).toBe("");
  });
});

// ── buildIndexContext (full assembly) ────────────────────────────

describe("buildIndexContext", () => {
  const context = {
    status: "ok",
    user_id: "u",
    categories: {
      decision: [mem({ id: "d1" })],
      task_context: [
        mem({
          id: "note-1",
          memory: NOTE_BODY,
          tags: ["session_note"],
          category: "task_context",
          observation_type: "meeting_outcome",
          created_at: isoAt(1),
        }),
      ],
    },
    standards: [mem({ id: "s1", memory: "Always use uv for Python deps." })],
  } as any;

  it("assembles standards, cards, Previously, index and footer in order", () => {
    const body = buildIndexContext({
      context,
      cards: [{ label: "user", lines: ["IDENTITY: Test user"] }],
      codeGraphAvailable: true,
      budgetTokens: 1500,
      now: NOW,
    });
    const order = [
      "AUTHORITATIVE Standards",
      "Identity Card (user)",
      "## Previously",
      "## Memory Index — index:",
      "## Using this index",
      "query_code_graph",
    ];
    let last = -1;
    for (const marker of order) {
      const idx = body.indexOf(marker);
      expect(idx, `expected marker ${marker}`).toBeGreaterThan(last);
      last = idx;
    }
  });

  it("excludes the Previously note from the index table", () => {
    const body = buildIndexContext({
      context,
      cards: [],
      codeGraphAvailable: false,
      budgetTokens: 1500,
      now: NOW,
    });
    expect(body).toContain("## Previously");
    expect(body).not.toContain("#note-1 |");
    expect(body).toContain("#d1 |");
  });

  it("returns empty for an empty context", () => {
    const body = buildIndexContext({
      context: { status: "ok", user_id: "u", categories: {} } as any,
      cards: [],
      codeGraphAvailable: false,
      budgetTokens: 1500,
      now: NOW,
    });
    expect(body).toBe("");
  });

  it("respects the budget for the index section", () => {
    const many = {
      status: "ok",
      user_id: "u",
      categories: {
        decision: Array.from({ length: 200 }, (_, i) =>
          mem({ id: `dd-${i}`, created_at: isoAt(0, 9, i % 60) }),
        ),
      },
    } as any;
    const body = buildIndexContext({
      context: many,
      cards: [],
      codeGraphAvailable: false,
      budgetTokens: 300,
      now: NOW,
    });
    expect(body).toContain("more not shown");
    // Rendered index section stays in the budget's ballpark.
    const indexSection = body.slice(body.indexOf("## Memory Index"), body.indexOf("## Using this index"));
    expect(indexSection.length / 4).toBeLessThan(300 + 150);
  });
});

// ── Edge branches ────────────────────────────────────────────────

describe("edge branches", () => {
  it("distillTitle clips a pathological single word at the char cap", () => {
    const title = distillTitle("x".repeat(300));
    expect(title.length).toBeLessThanOrEqual(82);
    expect(title.endsWith("…")).toBe(true);
  });

  it("distillTitle handles whitespace-only content", () => {
    expect(distillTitle("   \n  ")).toBe("(untitled)");
  });

  it("toIndexEntry defaults a missing category", () => {
    expect(toIndexEntry(mem({ category: null })).category).toBe("memory");
  });

  it("humanizeAge covers all buckets including the future clamp", () => {
    expect(humanizeAge(new Date(NOW.getTime() + 60_000).toISOString(), NOW)).toBe("now");
    expect(humanizeAge(new Date(NOW.getTime() - 3 * 60 * 60_000).toISOString(), NOW)).toBe("3h");
    expect(humanizeAge(isoAt(10), NOW)).toBe("1w");
    expect(humanizeAge(isoAt(60), NOW)).toBe("2mo");
    expect(humanizeAge(isoAt(800), NOW)).toBe("2y");
    expect(humanizeAge("garbage", NOW)).toBe("?");
  });

  it("renderSavingsHeader tolerates a zero baseline", () => {
    const header = renderSavingsHeader({
      text: "",
      included: 0,
      total: 0,
      servedTokens: 0,
      baselineTokens: 0,
    });
    expect(header).toBe("index: 0 memories, ~0 tokens vs ~0 full");
  });

  it("parseSessionNoteBody handles a bare label line", () => {
    const note = parseSessionNoteBody("Session note:\nRequest:\nvalue on next line")!;
    expect(note.request).toBe("value on next line");
  });

  it("renderPreviously works without an age suffix", () => {
    const block = renderPreviously({ next_steps: "do it" });
    expect(block).toContain("## Previously\n");
    expect(block).toContain("**Next steps:** do it");
  });

  it("buildIndexContext renders the footer when only a card exists", () => {
    const body = buildIndexContext({
      context: { status: "ok", user_id: "u", categories: {} } as any,
      cards: [{ label: "user", lines: ["IDENTITY: someone"] }],
      codeGraphAvailable: false,
      budgetTokens: 1500,
      now: NOW,
    });
    expect(body).toContain("Identity Card (user)");
    expect(body).toContain("## Using this index");
  });

  it("buildIndexContext keeps an unparsable tagged note in the index", () => {
    const body = buildIndexContext({
      context: {
        status: "ok",
        user_id: "u",
        categories: {
          task_context: [
            mem({ id: "weird-note", memory: "not a note body", tags: ["session_note"] }),
          ],
        },
      } as any,
      cards: [],
      codeGraphAvailable: false,
      budgetTokens: 1500,
      now: NOW,
    });
    expect(body).not.toContain("## Previously");
    expect(body).toContain("#weird-note |");
  });
});

// ── Legacy full-content mode (CONTEXT_MODE=full) ─────────────────

describe("legacy formatters", () => {
  it("formatStandards renders the binding block untruncated", () => {
    const block = formatStandards([
      mem({ memory: "Always use uv." }),
      mem({ memory: "Never commit to dev." }),
    ]);
    expect(block).toContain("AUTHORITATIVE Standards");
    expect(block).toContain("- Always use uv.");
    expect(block).toContain("- Never commit to dev.");
    expect(formatStandards([])).toBe("");
    expect(formatStandards(undefined)).toBe("");
  });

  it("formatMemories renders category sections in taxonomy order under the char cap", () => {
    const out = formatMemories({
      decision: [mem({ memory: "Decided X." })],
      preference: [mem({ memory: "Prefers tabs." })],
    });
    expect(out).toContain("# Neuralscape Memory Context");
    expect(out.indexOf("## Preferences")).toBeLessThan(out.indexOf("## Decisions"));
    expect(out).toContain("- Prefers tabs.");
    expect(formatMemories({})).toBe("");
  });

  it("formatMemories truncates at ~8000 chars", () => {
    const big = { preference: Array.from({ length: 200 }, (_, i) => mem({ memory: `pref ${i} ${"x".repeat(100)}` })) };
    const out = formatMemories(big);
    expect(out.length).toBeLessThan(9500);
  });
});

// ── Config knobs ─────────────────────────────────────────────────

describe("config knobs", () => {
  it("defaults to index mode, 1500 tokens, auto code graph", () => {
    delete process.env.CLAUDE_PLUGIN_OPTION_CONTEXT_MODE;
    delete process.env.CLAUDE_PLUGIN_OPTION_INDEX_BUDGET_TOKENS;
    delete process.env.CLAUDE_PLUGIN_OPTION_CODE_GRAPH;
    expect(getContextMode()).toBe("index");
    expect(getIndexBudgetTokens()).toBe(1500);
    expect(getCodeGraphMode()).toBe("auto");
  });

  it("honors overrides and rejects garbage", () => {
    process.env.CLAUDE_PLUGIN_OPTION_CONTEXT_MODE = "full";
    process.env.CLAUDE_PLUGIN_OPTION_INDEX_BUDGET_TOKENS = "800";
    process.env.CLAUDE_PLUGIN_OPTION_CODE_GRAPH = "off";
    expect(getContextMode()).toBe("full");
    expect(getIndexBudgetTokens()).toBe(800);
    expect(getCodeGraphMode()).toBe("off");

    process.env.CLAUDE_PLUGIN_OPTION_CONTEXT_MODE = "banana";
    process.env.CLAUDE_PLUGIN_OPTION_INDEX_BUDGET_TOKENS = "-3";
    process.env.CLAUDE_PLUGIN_OPTION_CODE_GRAPH = "banana";
    expect(getContextMode()).toBe("index");
    expect(getIndexBudgetTokens()).toBe(1500);
    expect(getCodeGraphMode()).toBe("auto");

    delete process.env.CLAUDE_PLUGIN_OPTION_CONTEXT_MODE;
    delete process.env.CLAUDE_PLUGIN_OPTION_INDEX_BUDGET_TOKENS;
    delete process.env.CLAUDE_PLUGIN_OPTION_CODE_GRAPH;
  });
});
