/**
 * Progressive-disclosure rendering for SessionStart injection (roadmap D1/D2).
 *
 * Pure functions only — no HTTP, no filesystem — so every renderer is unit
 * testable from fixtures. The hook fetches context/cards and delegates all
 * formatting here.
 *
 * The injected block is "the map, not the path" (C1): a day-grouped index
 * table of `#id | time | glyph category | title | ~tokens` rows instead of
 * full memory payloads, plus an escalation footer teaching the 3-layer
 * workflow (index → filter → get_memories → timeline).
 */

import type { NeuralscapeMemory } from "../utils.js";

// ── Budget ───────────────────────────────────────────────────────

/** Default token budget for the index table (config: INDEX_BUDGET_TOKENS). */
export const DEFAULT_INDEX_BUDGET_TOKENS = 1500;

/** ~4 chars per token — same heuristic the service uses in index_format.py. */
export function estimateTokens(text: string | null | undefined): number {
  if (!text) return 1;
  return Math.max(1, Math.ceil(text.length / 4));
}

// ── Glyphs (mirrors OBSERVATION_GLYPHS in neuralscape-service/index_format.py) ──

export const OBSERVATION_GLYPHS: Record<string, string> = {
  bugfix: "🐛",
  feature: "✨",
  refactor: "♻",
  decision: "⚖",
  discovery: "🔍",
  gotcha: "⚠",
  pattern: "◇",
  trade_off: "⇄",
  research_note: "📝",
  meeting_outcome: "🤝",
  task_plan: "🗺",
  fact: "•",
  reflection: "💭",
};

export const DEFAULT_GLYPH = "·";

export function glyphFor(observationType: string | null | undefined): string {
  if (!observationType) return DEFAULT_GLYPH;
  return OBSERVATION_GLYPHS[observationType] ?? DEFAULT_GLYPH;
}

// ── Title fallback (client-side twin of index_format.distill_title) ─

const TITLE_MAX_WORDS = 10;
const TITLE_MAX_CHARS = 80;

/**
 * Distill a ~10-word title from memory content. Only used for legacy
 * memories the server didn't stamp a `title` on — server titles win.
 */
export function distillTitle(content: string | null | undefined): string {
  if (!content || !content.trim()) return "(untitled)";
  const firstLine =
    content
      .split("\n")
      .map((l) => l.replace(/^[\s#>*+\-`~|:\d.)\]\[]+/, "").trim())
      .find((l) => l.length > 0) ?? "";
  const sentence = firstLine.split(/(?<=[.!?])\s+/)[0]?.replace(/[.!?]+$/, "").trim() ?? "";
  const base = sentence || content.replace(/\s+/g, " ").trim();
  const words = base.split(/\s+/);
  let clipped = words.slice(0, TITLE_MAX_WORDS).join(" ");
  let truncated = words.length > TITLE_MAX_WORDS;
  if (clipped.length > TITLE_MAX_CHARS) {
    clipped = clipped.slice(0, TITLE_MAX_CHARS - 1).trimEnd();
    truncated = true;
  }
  return truncated ? `${clipped} …` : clipped;
}

// ── Index entries ────────────────────────────────────────────────

export interface IndexEntry {
  id: string;
  title: string;
  category: string;
  glyph: string;
  createdAt: string | null;
  /** Estimated full-payload cost of this memory (what expanding it costs). */
  tokens: number;
}

export function toIndexEntry(mem: NeuralscapeMemory): IndexEntry {
  return {
    id: mem.id,
    title: mem.title || distillTitle(mem.memory),
    category: mem.category || "memory",
    glyph: glyphFor(mem.observation_type),
    createdAt: mem.created_at ?? null,
    tokens: mem.token_estimate || estimateTokens(mem.memory),
  };
}

// ── Day grouping ─────────────────────────────────────────────────

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

function localYmd(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/** "Today" / "Yesterday" / "Mon 2026-06-29" / "Undated". Local time. */
export function dayLabel(iso: string | null, now: Date = new Date()): string {
  if (!iso) return "Undated";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "Undated";
  const ymd = localYmd(d);
  if (ymd === localYmd(now)) return "Today";
  const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);
  if (ymd === localYmd(yesterday)) return "Yesterday";
  return `${WEEKDAYS[d.getDay()]} ${ymd}`;
}

/** "14:32" (local) or "--:--" when the timestamp is missing/unparseable. */
export function timeLabel(iso: string | null): string {
  if (!iso) return "--:--";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "--:--";
  const h = String(d.getHours()).padStart(2, "0");
  const m = String(d.getMinutes()).padStart(2, "0");
  return `${h}:${m}`;
}

// ── Index table renderer ─────────────────────────────────────────

export interface RenderedIndex {
  text: string;
  /** Rows that made it under the budget. */
  included: number;
  total: number;
  /** Token cost of the rendered index block itself. */
  servedTokens: number;
  /** Full-payload cost of the included rows — the counterfactual. */
  baselineTokens: number;
}

function renderRow(e: IndexEntry): string {
  return `#${e.id} | ${timeLabel(e.createdAt)} | ${e.glyph} ${e.category} | ${e.title} | ~${e.tokens}`;
}

/**
 * Render the day-grouped index table under a token budget.
 *
 * Entries are sorted newest-first (undated last) and emitted until the
 * running rendered-token total would exceed `budgetTokens`; a trailing
 * "… N more" line points at index-first recall for the rest.
 */
export function renderIndexTable(
  entries: IndexEntry[],
  opts: { budgetTokens?: number; now?: Date } = {},
): RenderedIndex {
  const budget = opts.budgetTokens ?? DEFAULT_INDEX_BUDGET_TOKENS;
  const now = opts.now ?? new Date();

  const sorted = [...entries].sort((a, b) => {
    const ta = a.createdAt ? new Date(a.createdAt).getTime() : Number.NEGATIVE_INFINITY;
    const tb = b.createdAt ? new Date(b.createdAt).getTime() : Number.NEGATIVE_INFINITY;
    return (Number.isNaN(tb) ? Number.NEGATIVE_INFINITY : tb) - (Number.isNaN(ta) ? Number.NEGATIVE_INFINITY : ta);
  });

  const lines: string[] = [];
  let served = 0;
  let baseline = 0;
  let included = 0;
  let currentDay: string | null = null;

  for (const entry of sorted) {
    const day = dayLabel(entry.createdAt, now);
    const pendingLines: string[] = [];
    if (day !== currentDay) pendingLines.push(`### ${day}`);
    pendingLines.push(renderRow(entry));
    const cost = pendingLines.reduce((s, l) => s + estimateTokens(l), 0);
    if (served + cost > budget && included > 0) break;
    if (day !== currentDay) currentDay = day;
    lines.push(...pendingLines);
    served += cost;
    baseline += entry.tokens;
    included++;
    if (served >= budget) break;
  }

  const omitted = sorted.length - included;
  if (omitted > 0) {
    const omissionLine = `… ${omitted} more not shown — recall_memories(query=…, index_only=true) to browse the rest.`;
    lines.push(omissionLine);
    // The omission line is injected too — count it, or the savings header
    // would undercount the served cost whenever rows are clipped.
    served += estimateTokens(omissionLine);
  }

  return {
    text: lines.join("\n"),
    included,
    total: sorted.length,
    servedTokens: served,
    baselineTokens: baseline,
  };
}

/** `index: N memories, ~X tokens vs ~Y full` — the honest-meter header. */
export function renderSavingsHeader(r: RenderedIndex): string {
  const pct =
    r.baselineTokens > 0
      ? Math.max(0, Math.round((1 - r.servedTokens / r.baselineTokens) * 100))
      : 0;
  const scope = r.included < r.total ? `${r.included} of ${r.total}` : `${r.included}`;
  return (
    `index: ${scope} memories, ~${r.servedTokens} tokens vs ~${r.baselineTokens} full` +
    (r.baselineTokens > 0 ? ` (${pct}% saved)` : "")
  );
}

// ── Escalation footer (C1/C2 workflow + F2 code-graph deferral) ──

const MCP = "mcp__plugin_neuralscape_neuralscape__";

export function renderEscalationFooter(codeGraphAvailable: boolean): string {
  const lines = [
    "## Using this index",
    "",
    "The table above is an index, not the memories themselves. Escalate only when needed:",
    "",
    `1. **Scan** the rows — title + glyph + recency (day heading + time) is usually enough to decide relevance.`,
    `2. **Search** for more: \`${MCP}recall_memories\` with \`index_only: true\` returns the same compact rows for any query.`,
    `3. **Expand** the few that matter: \`${MCP}get_memories\` with \`ids: [...]\` (the \`#id\` values above; batch up to 50).`,
    `4. **History** around a moment: \`${MCP}timeline\` with \`anchor\` = a memory id or query.`,
    "",
    "Prefer the index for broad scans, but expand any row whose title looks relevant — titles are lossy ~10-word summaries, so don't rule a memory out from its title alone. Economics: an index row costs ~10 tokens, a full payload ~50-500.",
  ];
  if (codeGraphAvailable) {
    lines.push(
      "",
      "### Code structure — defer to the code graph (F2)",
      "",
      `This project has a Graphify code graph behind Neuralscape. For questions about code ` +
        `*structure* (what calls what, module layout, dependency paths), use ` +
        `\`${MCP}query_code_graph\`, \`${MCP}get_code_neighbors\`, or \`${MCP}code_path\` ` +
        `instead of searching memories — and do NOT store purely structural observations as ` +
        `memories (structure rots with every commit; the graph is the source of truth). ` +
        `Memories are for decisions, gotchas, and rationale *about* the code.`,
    );
  }
  return lines.join("\n");
}

// ── Session-note ("Previously…") parsing + rendering (D2) ────────

export interface SessionNoteFields {
  request?: string;
  investigated?: string;
  learned?: string;
  completed?: string;
  next_steps?: string;
}

const NOTE_LABELS: Array<[keyof SessionNoteFields, string]> = [
  ["request", "Request"],
  ["investigated", "Investigated"],
  ["learned", "Learned"],
  ["completed", "Completed"],
  ["next_steps", "Next steps"],
];

/**
 * Parse the body the server renders for a checkpoint session note
 * ("Session note:\nRequest: …\nNext steps: …") back into fields.
 * Values may span multiple lines. Returns null when the content is not
 * a session note.
 */
export function parseSessionNoteBody(content: string | null | undefined): SessionNoteFields | null {
  if (!content || !content.trimStart().startsWith("Session note:")) return null;
  const fields: SessionNoteFields = {};
  let currentKey: keyof SessionNoteFields | null = null;
  let buf: string[] = [];
  const flush = () => {
    if (currentKey && buf.length > 0) {
      const value = buf.join("\n").trim();
      if (value) fields[currentKey] = value;
    }
    buf = [];
  };
  for (const line of content.split("\n").slice(1)) {
    const match = NOTE_LABELS.find(([, label]) => line.startsWith(`${label}: `) || line === `${label}:`);
    if (match) {
      flush();
      currentKey = match[0];
      buf.push(line.slice(match[1].length + 1).trim());
    } else if (currentKey) {
      buf.push(line);
    }
  }
  flush();
  return Object.keys(fields).length > 0 ? fields : null;
}

/** Is this memory a checkpoint session note? */
export function isSessionNote(mem: NeuralscapeMemory): boolean {
  return (
    (mem.tags ?? []).includes("session_note") ||
    (mem.memory ?? "").trimStart().startsWith("Session note:")
  );
}

/**
 * Newest session note across the context categories (they live in
 * task_context, but scan everything for safety).
 */
export function findLatestSessionNote(
  categories: Record<string, NeuralscapeMemory[]>,
): NeuralscapeMemory | null {
  let latest: NeuralscapeMemory | null = null;
  let latestTs = Number.NEGATIVE_INFINITY;
  for (const memories of Object.values(categories)) {
    for (const mem of memories ?? []) {
      if (!isSessionNote(mem)) continue;
      const ts = mem.created_at ? new Date(mem.created_at).getTime() : Number.NEGATIVE_INFINITY;
      const effective = Number.isNaN(ts) ? Number.NEGATIVE_INFINITY : ts;
      if (latest === null || effective > latestTs) {
        latest = mem;
        latestTs = effective;
      }
    }
  }
  return latest;
}

/** Compact "Previously…" block — next_steps first, then narrative order. */
export function renderPreviously(note: SessionNoteFields, age?: string): string {
  const lines: string[] = [`## Previously${age ? ` (${age})` : ""}`, ""];
  if (note.next_steps) lines.push(`**Next steps:** ${note.next_steps}`);
  for (const [key, label] of NOTE_LABELS) {
    if (key === "next_steps") continue;
    const value = note[key];
    if (value) lines.push(`**${label}:** ${value}`);
  }
  return lines.length > 2 ? lines.join("\n") : "";
}

/** Compact humanized age, mirroring index_format.humanize_age. */
export function humanizeAge(iso: string | null | undefined, now: Date = new Date()): string {
  if (!iso) return "?";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "?";
  let seconds = (now.getTime() - d.getTime()) / 1000;
  if (seconds < 0) seconds = 0;
  if (seconds < 60) return "now";
  const minutes = seconds / 60;
  if (minutes < 60) return `${Math.floor(minutes)}m`;
  const hours = minutes / 60;
  if (hours < 24) return `${Math.floor(hours)}h`;
  const days = hours / 24;
  if (days < 7) return `${Math.floor(days)}d`;
  if (days < 30) return `${Math.floor(days / 7)}w`;
  if (days < 365) return `${Math.floor(days / 30)}mo`;
  return `${Math.floor(days / 365)}y`;
}

// ── Identity card block (B4) ─────────────────────────────────────

export function renderCardBlock(
  cards: Array<{ label: string; lines: string[] }>,
): string {
  const blocks: string[] = [];
  for (const card of cards) {
    if (!card.lines || card.lines.length === 0) continue;
    blocks.push(`## Identity Card (${card.label})\n\n${card.lines.join("\n")}`);
  }
  return blocks.join("\n\n");
}
