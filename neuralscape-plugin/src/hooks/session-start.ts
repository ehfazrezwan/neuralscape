/**
 * SessionStart hook — progressive-disclosure context injection (roadmap D1/D2).
 *
 * Default ("index") mode injects the MAP, not the memories:
 *   1. binding org standards (unchanged contract — never truncated),
 *   2. the identity card(s) from the dreaming sweep (B4, when available),
 *   3. a "Previously…" block from the last session's structured note (D2),
 *   4. a day-grouped, budget-bounded index table of recent memories with a
 *      savings header (D1),
 *   5. an escalation footer teaching index → filter → get_memories →
 *      timeline (+ the F2 code-graph deferral policy when the project has
 *      a Graphify graph behind the NS surface).
 *
 * Legacy full-content injection is preserved behind CONTEXT_MODE=full.
 *
 * Failure taxonomy (D4): NS unreachable → inject a one-line notice and exit
 * 0 (NEVER block session start); malformed hook stdin → exit 2 (client bug,
 * fail loud — SessionStart exit 2 only surfaces stderr, it cannot block).
 */

import {
  type ContextResponse,
  type NeuralscapeMemory,
  MalformedHookInputError,
  getFailLoudThreshold,
  getProjectId,
  getUserId,
  hasUserId,
  isProjectExcluded,
  listPendingBuffers,
  logError,
  neuralscapeGet,
  outputContinue,
  outputWithContext,
  parseStdinStrict,
  readConfig,
  recordTransportFailure,
  resetTransportFailures,
} from "../utils.js";

import {
  DEFAULT_INDEX_BUDGET_TOKENS,
  findLatestSessionNote,
  humanizeAge,
  parseSessionNoteBody,
  renderCardBlock,
  renderEscalationFooter,
  renderIndexTable,
  renderPreviously,
  renderResumeAfterCompact,
  renderSavingsHeader,
  toIndexEntry,
} from "../core/disclosure.js";

import { CATEGORY_LABELS, CATEGORY_ORDER } from "../types.js";

// Target max chars for legacy full-content injection (~4 chars per token)
const MAX_CHARS = 8000;

// How many recent memories to consider for the index (renderer budget
// decides how many actually render).
const INDEX_FETCH_LIMIT = 200;

// ── Config knobs ─────────────────────────────────────────────────

export function getContextMode(): "index" | "full" {
  const raw = readConfig("CONTEXT_MODE", "index").toLowerCase();
  return raw === "full" ? "full" : "index";
}

export function getIndexBudgetTokens(): number {
  const parsed = parseInt(readConfig("INDEX_BUDGET_TOKENS", ""), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_INDEX_BUDGET_TOKENS;
}

export function getCodeGraphMode(): "auto" | "on" | "off" {
  const raw = readConfig("CODE_GRAPH", "auto").toLowerCase();
  if (raw === "on" || raw === "off") return raw;
  return "auto";
}

// ── Fail-loud unreachable notice (roadmap D4) ────────────────────

/**
 * The one-line notice injected when NS is unreachable. Below the fail-loud
 * threshold it's the generic degrade line; at/after the threshold it names
 * the consecutive-failure count and points at `docker compose ps` so a dead
 * stack stops failing silently. Either way the hook exits 0.
 */
export function buildUnreachableNotice(count: number, threshold: number): string {
  if (count >= threshold) {
    return (
      `[neuralscape] memory service unreachable for ${count} consecutive events — ` +
      "check `docker compose ps` (or your service URL). Continuing without memory context."
    );
  }
  return "[neuralscape] memory service unreachable — continuing without memory context.";
}

// ── Legacy full-content formatting (CONTEXT_MODE=full) ───────────

export function formatStandards(standards: NeuralscapeMemory[] | undefined): string {
  if (!standards || standards.length === 0) return "";
  const lines: string[] = [
    "# ⚖️ Neuralscape AUTHORITATIVE Standards (binding)",
    "",
    "These are organization standards set by a Neuralscape dictator. They are " +
      "BINDING directives, not preferences. On any conflict they OVERRIDE " +
      "personal preferences and project conventions. Follow them unless the " +
      "user explicitly overrides them in this session.",
    "",
  ];
  // `standards` here is the always-inject (critical) subset the server selects
  // — BINDING and exempt from the ordinary context budget. Emit them all, never
  // truncating (a header-only block would silently weaken the contract).
  // Non-critical standards aren't dumped here; they surface via recall.
  for (const mem of standards) {
    lines.push(`- ${mem.memory}`);
  }
  return lines.join("\n");
}

export function formatMemories(categories: Record<string, NeuralscapeMemory[]>): string {
  const sections: string[] = [];
  let totalChars = 0;

  for (const cat of CATEGORY_ORDER) {
    const memories = categories[cat];
    if (!memories || memories.length === 0) continue;

    const label = CATEGORY_LABELS[cat] || cat;
    const lines: string[] = [`## ${label}`];

    for (const mem of memories) {
      const line = `- ${mem.memory}`;
      if (totalChars + line.length > MAX_CHARS) break;
      lines.push(line);
      totalChars += line.length;
    }

    if (lines.length > 1) {
      sections.push(lines.join("\n"));
    }

    if (totalChars >= MAX_CHARS) break;
  }

  if (sections.length === 0) return "";

  return `# Neuralscape Memory Context\n\n${sections.join("\n\n")}`;
}

// ── Index-mode assembly (pure — unit tested) ─────────────────────

export interface IndexModeInputs {
  context: ContextResponse;
  cards: Array<{ label: string; lines: string[] }>;
  codeGraphAvailable: boolean;
  budgetTokens: number;
  now?: Date;
  /**
   * SessionStart `source` from Claude Code: "startup" | "resume" | "clear"
   * | "compact". On "compact"/"resume" the recent compact snapshots render
   * full-content under "## Resuming after compaction".
   */
  source?: string;
}

/**
 * Build the full index-mode injection body from already-fetched inputs.
 * Standards stay a binding block outside the budget; the session note that
 * feeds "Previously…" is excluded from the index table (no double render).
 */
export function buildIndexContext(inputs: IndexModeInputs): string {
  const { context, cards, codeGraphAvailable, budgetTokens } = inputs;
  const now = inputs.now ?? new Date();
  const categories = context.categories || {};

  const sections: string[] = [];

  const standardsBlock = formatStandards(context.standards);
  if (standardsBlock) sections.push(standardsBlock);

  const cardBlock = renderCardBlock(cards);
  if (cardBlock) sections.push(cardBlock);

  // Compact-resilience: after a compaction (or --resume), re-anchor the
  // session with the pre-compact snapshots the PreCompact hook stored.
  // Rendered full-content here, excluded from the index table below.
  const resumeIds = new Set<string>();
  if (inputs.source === "compact" || inputs.source === "resume") {
    const resume = renderResumeAfterCompact(categories, { now });
    if (resume.block) {
      sections.push(resume.block);
      for (const id of resume.ids) resumeIds.add(id);
    }
  }

  // D2: "Previously…" from the newest checkpoint session note.
  const noteMemory = findLatestSessionNote(categories);
  let previouslyId: string | null = null;
  if (noteMemory) {
    const note = parseSessionNoteBody(noteMemory.memory);
    if (note) {
      const block = renderPreviously(
        note,
        `last session, ${humanizeAge(noteMemory.created_at ?? null, now)} ago`,
      );
      if (block) {
        sections.push(block);
        previouslyId = noteMemory.id;
      }
    }
  }

  // D1: day-grouped index of everything else.
  const memories: NeuralscapeMemory[] = [];
  for (const cat of Object.keys(categories)) {
    for (const mem of categories[cat] ?? []) {
      if (mem.id === previouslyId) continue;
      if (resumeIds.has(mem.id)) continue;
      memories.push(mem);
    }
  }

  if (memories.length > 0) {
    const rendered = renderIndexTable(memories.map(toIndexEntry), {
      budgetTokens,
      now,
    });
    sections.push(
      [
        `## Memory Index — ${renderSavingsHeader(rendered)}`,
        "",
        "`#id | time | type | title | ~tokens`",
        "",
        rendered.text,
      ].join("\n"),
    );
    sections.push(renderEscalationFooter(codeGraphAvailable));
  } else if (cardBlock || previouslyId || resumeIds.size > 0) {
    sections.push(renderEscalationFooter(codeGraphAvailable));
  }

  if (sections.length === 0) return "";
  return `# Neuralscape Memory\n\n${sections.join("\n\n---\n\n")}`;
}

/* v8 ignore start — I/O paths below are exercised by the E2E dry-run script */

// ── Fetch helpers ────────────────────────────────────────────────

async function fetchCard(pool: string, userId: string): Promise<string[] | null> {
  try {
    const view = (await neuralscapeGet("/v1/extensions/dreaming/card", {
      pool,
      user_id: userId,
    })) as { status?: string; lines?: string[] };
    if (view && view.status === "ok" && Array.isArray(view.lines) && view.lines.length > 0) {
      return view.lines;
    }
    return null;
  } catch {
    // 404 (no card yet), 403, dreaming disabled, older server — all fine.
    return null;
  }
}

async function probeCodeGraph(userId: string): Promise<boolean> {
  const mode = getCodeGraphMode();
  if (mode === "on") return true;
  if (mode === "off") return false;
  try {
    // Cheapest positive probe: a minimal query. 200 → the extra is
    // installed AND a graph resolves for this deployment. Anything else
    // (501 extra missing, 400 not configured, 404, network) → unavailable.
    await neuralscapeGet("/v1/code-graph/query", {
      question: "__ns_session_start_probe__",
      depth: "1",
      token_budget: "100",
      user_id: userId,
    });
    return true;
  } catch {
    return false;
  }
}

// ── Main ─────────────────────────────────────────────────────────

async function main(): Promise<void> {
  let input;
  try {
    input = await parseStdinStrict();
  } catch (error) {
    if (error instanceof MalformedHookInputError) {
      logError(error.message);
      process.exit(2); // client bug — fail loud (cannot block SessionStart)
    }
    logError("session-start stdin read failed", error);
    outputContinue();
    return;
  }

  try {
    if (!hasUserId()) {
      logError(
        "missing user_id — run `/plugin config neuralscape@neuralscape-plugins` to set USER_ID (or set NEURALSCAPE_USER_ID env var as legacy fallback); skipping context injection",
      );
      outputContinue();
      return;
    }

    const userId = getUserId();
    const projectId = getProjectId(input.cwd);

    // Excluded projects (D4): no disclosure fetch, no injection.
    if (isProjectExcluded(projectId)) {
      logError(`project '${projectId}' matches EXCLUDED_PROJECTS — skipping context injection`);
      outputContinue();
      return;
    }

    // Kick everything off in parallel; the context fetch is the only one
    // whose failure degrades the whole injection.
    const contextPromise: Promise<ContextResponse> = projectId
      ? (neuralscapeGet(`/v1/context/${projectId}`, {
          user_id: userId,
          limit: String(INDEX_FETCH_LIMIT),
        }) as Promise<ContextResponse>)
      : (neuralscapeGet("/v1/context/global", {
          user_id: userId,
        }) as Promise<ContextResponse>);

    const cardPromises: Array<Promise<{ label: string; lines: string[] } | null>> = [];
    if (projectId) {
      cardPromises.push(
        fetchCard(`shared--project--${projectId}`, userId).then((lines) =>
          lines ? { label: `project: ${projectId}`, lines } : null,
        ),
      );
    }
    cardPromises.push(
      fetchCard(`user--${userId}`, userId).then((lines) =>
        lines ? { label: "user", lines } : null,
      ),
    );

    const mode = getContextMode();
    const codeGraphPromise = mode === "index" ? probeCodeGraph(userId) : Promise.resolve(false);

    let context: ContextResponse;
    try {
      context = await contextPromise;
    } catch {
      // NS unreachable — never block session start: one-line notice, exit 0.
      // The consecutive-failure counter upgrades the notice to fail-loud
      // once the streak crosses FAIL_LOUD_THRESHOLD (D4).
      const count = await recordTransportFailure();
      outputWithContext(buildUnreachableNotice(count, getFailLoudThreshold()));
      return;
    }
    await resetTransportFailures();

    // Detect any unprocessed observation buffers left behind by prior sessions
    // and ask Claude to compile them before responding to the user.
    let pendingNote = "";
    try {
      const pending = await listPendingBuffers(input.session_id);
      if (pending.length > 0) {
        const totalRows = pending.reduce((s, b) => s + b.lineCount, 0);
        const paths = pending.map((b) => `\`${b.path}\``).join(", ");
        pendingNote =
          `\n\n# Neuralscape — Pending Observations\n\n` +
          `${pending.length} buffer file(s) from prior sessions hold ` +
          `${totalRows} unprocessed tool observations: ${paths}.\n\n` +
          `**Before responding to the user's first prompt**, run the ` +
          `\`compile-observations\` skill on each of these paths to extract ` +
          `significant memories using the v2 memory model. Skip noise ` +
          `(routine reads/edits/searches). After memories submit successfully, ` +
          `truncate each buffer file. Then continue with the user's request.`;
      }
    } catch (error) {
      logError("listPendingBuffers failed (non-critical)", error);
    }

    let contextBody: string;
    if (mode === "full") {
      // Legacy full-content injection, standards prepended unbudgeted.
      const standardsBlock = formatStandards(context.standards);
      const formatted = formatMemories(context.categories || {});
      contextBody = standardsBlock
        ? formatted
          ? `${standardsBlock}\n\n---\n\n${formatted}`
          : standardsBlock
        : formatted;
    } else {
      const settled = await Promise.all(
        cardPromises.map((p) => p.catch(() => null)),
      );
      const cards = settled.filter(
        (c): c is { label: string; lines: string[] } => c !== null,
      );
      const codeGraphAvailable = await codeGraphPromise.catch(() => false);
      contextBody = buildIndexContext({
        context,
        cards,
        codeGraphAvailable,
        budgetTokens: getIndexBudgetTokens(),
        source: input.source,
      });
    }

    const combined = (contextBody || "") + pendingNote;
    if (combined.trim()) {
      outputWithContext(combined);
    } else {
      outputContinue();
    }
  } catch (error) {
    logError("session-start hook failed", error);
    outputContinue();
  }
}

// Skip auto-running main() when imported by the test harness.
if (process.env.NEURALSCAPE_TEST !== "1") {
  main();
}
/* v8 ignore stop */
