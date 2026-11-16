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
  type LocalCompactSnapshot,
  type FailureKind,
  type NeuralscapeMemory,
  MalformedHookInputError,
  getFailLoudThreshold,
  getProjectId,
  getServiceUrl,
  getUserId,
  getUserIdSource,
  hasUserId,
  isProjectExcluded,
  listPendingBuffers,
  logError,
  neuralscapeGet,
  outputContinue,
  outputWithContext,
  parseStdinStrict,
  readConfig,
  readLocalCompactSnapshot,
  readPredecessorSessionIds,
  recordFailure,
  classifyFailure,
  neuralscapePost,
  redactPrivate,
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

import { extractRecentUserPrompts } from "../adapters/claude-code.js";
import { COMPACT_SNAPSHOT_SESSION_TAG_PREFIX, COMPACT_SNAPSHOT_TAG } from "../core/compact.js";
import { CATEGORY_LABELS, CATEGORY_ORDER } from "../types.js";

// Target max chars for legacy full-content injection (~4 chars per token)
const MAX_CHARS = 8000;

// How many recent memories to consider for the index (renderer budget
// decides how many actually render).
const INDEX_FETCH_LIMIT = 200;

// ── Config knobs ─────────────────────────────────────────────────

export function getContextMode(): "index" | "full" | "assemble" {
  const raw = readConfig("CONTEXT_MODE", "index").toLowerCase();
  return raw === "full" || raw === "assemble" ? raw : "index";
}

/** Bounds of `budget_tokens` on `POST /v1/context/assemble` (schemas.AssembleContextRequest). */
export const ASSEMBLE_MIN_BUDGET = 100;
export const ASSEMBLE_MAX_BUDGET = 1_000_000;
export const DEFAULT_ASSEMBLE_BUDGET_TOKENS = 1500;
const ASSEMBLE_QUERY_MAX = 2000;
const ASSEMBLE_SESSION_ID_MAX = 100;

/** Token budget sent to the assemble route (`ASSEMBLE_BUDGET_TOKENS`), clamped to the route's bounds. */
export function getAssembleBudgetTokens(): number {
  const parsed = parseInt(readConfig("ASSEMBLE_BUDGET_TOKENS", ""), 10);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_ASSEMBLE_BUDGET_TOKENS;
  return Math.min(ASSEMBLE_MAX_BUDGET, Math.max(ASSEMBLE_MIN_BUDGET, parsed));
}

/** An optional fixed query for the assembled bundle's relevant-memory rows (`ASSEMBLE_QUERY`). */
export function getAssembleQuery(): string | undefined {
  const raw = readConfig("ASSEMBLE_QUERY", "").trim();
  return raw ? raw.slice(0, ASSEMBLE_QUERY_MAX) : undefined;
}

/** The request body for `POST /v1/context/assemble` (format is always plain: the bundle is injected as text). */
export function buildAssembleRequest(opts: {
  budgetTokens: number;
  userId: string;
  sessionId?: string;
  projectId?: string;
  query?: string;
}): Record<string, unknown> {
  const body: Record<string, unknown> = {
    budget_tokens: Math.min(ASSEMBLE_MAX_BUDGET, Math.max(ASSEMBLE_MIN_BUDGET, Math.trunc(opts.budgetTokens))),
    format: "plain",
    user_id: opts.userId,
  };
  if (opts.sessionId) body.session_id = opts.sessionId.slice(0, ASSEMBLE_SESSION_ID_MAX);
  if (opts.projectId) body.project_id = opts.projectId;
  if (opts.query && opts.query.trim()) body.query = opts.query.slice(0, ASSEMBLE_QUERY_MAX);
  return body;
}

export interface AssembledBundle {
  text: string;
  usedTokens: number;
  budgetTokens: number;
}

/** The plain bundle out of an assemble response, or null when it carries no text. */
export function parseAssembleResponse(resp: unknown): AssembledBundle | null {
  const r = resp as { bundle?: { text?: unknown }; used_tokens?: unknown; budget_tokens?: unknown } | null;
  const text = r?.bundle?.text;
  if (typeof text !== "string" || !text.trim()) return null;
  return {
    text: text.trim(),
    usedTokens: typeof r?.used_tokens === "number" ? r.used_tokens : 0,
    budgetTokens: typeof r?.budget_tokens === "number" ? r.budget_tokens : 0,
  };
}

/** Memory ids the bundle's relevant-memory rows already show (`- [<id>] ...`), so the index table does not repeat them. */
export function assembledMemoryIds(text: string): Set<string> {
  const ids = new Set<string>();
  for (const m of text.matchAll(/^- \[([^\]\s]+)\]/gm)) ids.add(m[1]);
  return ids;
}

/**
 * The assemble query: the `ASSEMBLE_QUERY` setting, else (on resume/compact/fork, where the session is
 * continuing) the last user prompt of the transcript, else none (a fresh session has nothing to rank by).
 */
export async function resolveAssembleQuery(input: {
  source?: string;
  transcript_path?: string;
}): Promise<string | undefined> {
  const fixed = getAssembleQuery();
  if (fixed) return fixed;
  if (!isResumeSource(input.source) || !input.transcript_path) return undefined;
  const [last] = await extractRecentUserPrompts({ transcript_path: input.transcript_path }, 1);
  const clean = last ? redactPrivate(last, { failClosed: true }).trim() : "";
  return clean ? clean.slice(0, ASSEMBLE_QUERY_MAX) : undefined;
}

/**
 * Call the assemble route. Resolves null for "use the index instead": a 404 (an older server without the
 * route) quietly; any other failure with its #500 kind logged. Never throws.
 */
export async function fetchAssembledBundle(body: Record<string, unknown>): Promise<{
  bundle: AssembledBundle | null;
  notice: string;
}> {
  try {
    const bundle = parseAssembleResponse(await neuralscapePost("/v1/context/assemble", body));
    return { bundle, notice: "" };
  } catch (error) {
    const kind = classifyFailure(error);
    if (kind === "not_found") return { bundle: null, notice: "" };
    logError(`context assemble failed (${kind}); using the index`, error);
    // Network and server failures hit the index fetch too, which reports them; a rejected key or request is
    // specific to this call (or its settings) and would otherwise be invisible.
    const notice =
      kind === "auth" || kind === "client"
        ? `[neuralscape] CONTEXT_MODE=assemble was rejected by the service (${kind === "auth" ? "HTTP 401/403" : "HTTP 4xx"}); showing the index instead.`
        : "";
    return { bundle: null, notice };
  }
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

/**
 * The notice for a failed context fetch, by failure kind. Only `network` and
 * `server` are "unreachable" (and counted: see recordFailure); the others mean
 * the service answered and the plugin's setup is wrong, so they say what to fix.
 * Never includes the token or the URL.
 */
export function buildFailureNotice(
  kind: FailureKind,
  count: number | null,
  threshold: number,
): string {
  switch (kind) {
    case "auth":
      return (
        "[neuralscape] the memory service rejected the API key (HTTP 401/403) — " +
        "the token is missing, expired or revoked. Update API_KEY with " +
        "`/plugin config neuralscape@neuralscape-plugins`. Continuing without memory context."
      );
    case "not_found":
      return (
        "[neuralscape] the memory service answered 404 for the context route — " +
        "the service URL is probably wrong or points at an incompatible version. " +
        "Check URL in `/plugin config neuralscape@neuralscape-plugins`. Continuing without memory context."
      );
    case "client":
      return (
        "[neuralscape] the memory service rejected the context request (HTTP 4xx) — " +
        "check the plugin URL, user id and service version. Continuing without memory context."
      );
    default:
      return buildUnreachableNotice(count ?? 1, threshold);
  }
}

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

/** True when the service URL points at this machine (or does not parse). */
export function isLocalServiceUrl(url: string): boolean {
  try {
    return LOCAL_HOSTS.has(new URL(url).hostname.toLowerCase());
  } catch {
    return true;
  }
}

/**
 * One line shown when this install identifies as the OS login name (USER_ID
 * unset) against a non-local service: the fallback is deliberate (see utils.ts)
 * but the user should know which id their memories are filed under. Empty
 * otherwise. Does not name the service host.
 */
export function buildIdentityNotice(
  userId: string,
  source: "config" | "os" | "none",
  serviceUrl: string,
): string {
  if (source !== "os" || !userId || isLocalServiceUrl(serviceUrl)) return "";
  return (
    `[neuralscape] USER_ID is not set, so memories are filed under your OS login name "${userId}" ` +
    "on a remote memory service. Set USER_ID with `/plugin config neuralscape@neuralscape-plugins` to choose the id."
  );
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
   * | "compact" | "fork". On "compact"/"resume"/"fork" the compact snapshots
   * render full-content under "## Resuming after compaction".
   */
  source?: string;
  /** The starting session's id: on compact/resume only its own snapshots render (see renderResumeAfterCompact). */
  sessionId?: string;
  /** Resume/fork: the earlier session id(s) read from the transcript head (see readPredecessorSessionIds). */
  predecessorIds?: string[];
  /**
   * CONTEXT_MODE=assemble: the bundle `POST /v1/context/assemble` returned. It already carries the identity
   * card(s), the session summary / recent messages and (with a query) relevant-memory rows, so the cards are
   * not rendered again and the index table leaves out the rows the bundle shows. Standards, "Resuming after
   * compaction" and "Previously…" come from the context fetch and are unchanged.
   */
  assembled?: AssembledBundle;
}

/**
 * Build the full index-mode injection body from already-fetched inputs.
 * Standards stay a binding block outside the budget; the session note that
 * feeds "Previously…" is excluded from the index table (no double render).
 */
/**
 * The blocks both modes share, in order: binding standards, identity cards,
 * the resume-after-compaction block (compact/resume/fork only) and the
 * "Previously…" block. Also returns the ids rendered so the caller keeps them
 * out of its memory listing (no double render).
 */
function buildContinuitySections(inputs: IndexModeInputs, now: Date): {
  sections: string[];
  cardBlock: string;
  resumeIds: Set<string>;
  previouslyId: string | null;
} {
  const { context, cards } = inputs;
  const categories = context.categories || {};

  const sections: string[] = [];

  const standardsBlock = formatStandards(context.standards);
  if (standardsBlock) sections.push(standardsBlock);

  if (inputs.assembled) {
    const a = inputs.assembled;
    sections.push(
      `## Session context (assembled, ~${a.usedTokens} of ${a.budgetTokens} tokens)\n\n${a.text}`,
    );
  }

  // The assembled bundle already holds the identity card(s).
  const cardBlock = inputs.assembled ? "" : renderCardBlock(cards);
  if (cardBlock) sections.push(cardBlock);

  // Compact-resilience: after a compaction (or --resume), re-anchor the
  // session with the pre-compact snapshots the PreCompact hook stored.
  // Rendered full-content here, excluded from the listing below.
  const resumeIds = new Set<string>();
  if (isResumeSource(inputs.source)) {
    const resume = renderResumeAfterCompact(categories, {
      now,
      sessionId: inputs.sessionId,
      source: inputs.source,
      predecessorIds: inputs.predecessorIds,
    });
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
  return { sections, cardBlock, resumeIds, previouslyId };
}

/**
 * CONTEXT_MODE=full: the same continuity blocks as index mode (standards,
 * identity cards, "Resuming after compaction", "Previously…"), then the full
 * text of the remaining memories under the legacy MAX_CHARS budget. The
 * resume snapshot and the session note are rendered once, above, and left
 * out of the listing.
 */
export function buildFullContext(inputs: IndexModeInputs): string {
  const now = inputs.now ?? new Date();
  const { sections, resumeIds, previouslyId } = buildContinuitySections(inputs, now);

  const remaining: Record<string, NeuralscapeMemory[]> = {};
  for (const [cat, mems] of Object.entries(inputs.context.categories || {})) {
    const kept = (mems ?? []).filter((m) => m.id !== previouslyId && !resumeIds.has(m.id));
    if (kept.length > 0) remaining[cat] = kept;
  }
  const formatted = formatMemories(remaining);
  if (formatted) sections.push(formatted);

  if (sections.length === 0) return "";
  return sections.join("\n\n---\n\n");
}

export function buildIndexContext(inputs: IndexModeInputs): string {
  const { context, codeGraphAvailable, budgetTokens } = inputs;
  const now = inputs.now ?? new Date();
  const categories = context.categories || {};

  const { sections, cardBlock, resumeIds, previouslyId } = buildContinuitySections(inputs, now);

  // D1: day-grouped index of everything else.
  const memories: NeuralscapeMemory[] = [];
  const inBundle = inputs.assembled ? assembledMemoryIds(inputs.assembled.text) : new Set<string>();
  for (const cat of Object.keys(categories)) {
    for (const mem of categories[cat] ?? []) {
      if (mem.id === previouslyId) continue;
      if (resumeIds.has(mem.id)) continue;
      if (inBundle.has(mem.id)) continue;
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
  } else if (cardBlock || previouslyId) {
    // NOTE: resume-after-compact sections alone do NOT pull the footer in —
    // its copy explains "the table above", and a snapshot-only injection
    // renders no index table (Copilot, PR #131).
    sections.push(renderEscalationFooter(codeGraphAvailable));
  }

  if (sections.length === 0) return "";
  return `# Neuralscape Memory\n\n${sections.join("\n\n---\n\n")}`;
}

/** SessionStart sources that continue earlier work and so get the compact snapshots. */
export function isResumeSource(source: string | undefined): boolean {
  return source === "compact" || source === "resume" || source === "fork";
}

/**
 * Turn the local copy of this session's compact snapshot into a memory the
 * disclosure renderer understands.
 */
export function localSnapshotMemory(snap: LocalCompactSnapshot): NeuralscapeMemory {
  return {
    id: `local-compact-${snap.sessionId}`,
    memory: snap.content,
    category: "task_context",
    project_id: snap.projectId,
    tags: [COMPACT_SNAPSHOT_TAG, `${COMPACT_SNAPSHOT_SESSION_TAG_PREFIX}${snap.sessionId}`],
    created_at: snap.createdAt,
  };
}

/**
 * Add this session's local compact snapshot to the fetched context unless the
 * service already returned that very snapshot (same content). Only after a
 * compaction: the session id is unchanged by compaction, so the id names the
 * file; resume/fork get a new id and have no local copy.
 */
export async function withLocalCompactSnapshot(
  context: ContextResponse,
  input: { source?: string; session_id?: string },
): Promise<ContextResponse> {
  if (input.source !== "compact" || !input.session_id) return context;
  const snap = await readLocalCompactSnapshot(input.session_id);
  if (!snap) return context;
  const categories = context.categories || {};
  for (const memories of Object.values(categories)) {
    if ((memories ?? []).some((m) => (m.memory ?? "").trim() === snap.content.trim())) return context;
  }
  return {
    ...context,
    categories: {
      ...categories,
      task_context: [...(categories.task_context ?? []), localSnapshotMemory(snap)],
    },
  };
}

/** The "Resuming after compaction" block from the local snapshot alone (service unreachable). */
async function localResumeBlock(input: { source?: string; session_id?: string }): Promise<string> {
  const context = await withLocalCompactSnapshot({ categories: {} } as ContextResponse, input);
  return renderResumeAfterCompact(context.categories || {}, {
    sessionId: input.session_id,
    source: input.source,
  }).block;
}

/**
 * The "compile these buffers first" note for buffers left by sessions that
 * are over (see listPendingBuffers). Empty string when there are none.
 * Independent of the service: it only reads local files, so it is also
 * appended to the unreachable notice.
 */
export async function buildPendingNote(currentSessionId?: string): Promise<string> {
  try {
    const pending = await listPendingBuffers(currentSessionId);
    if (pending.length === 0) return "";
    const totalRows = pending.reduce((s, b) => s + b.lineCount, 0);
    const paths = pending.map((b) => `\`${b.path}\``).join(", ");
    return (
      `\n\n# Neuralscape — Pending Observations\n\n` +
      `${pending.length} buffer file(s) from prior sessions hold ` +
      `${totalRows} unprocessed tool observations: ${paths}.\n\n` +
      `**Before responding to the user's first prompt**, run the ` +
      `\`compile-observations\` skill on each of these paths to extract ` +
      `significant memories using the v2 memory model. Skip noise ` +
      `(routine reads/edits/searches). After memories submit successfully, ` +
      `truncate each buffer file. Then continue with the user's request.`
    );
  } catch (error) {
    logError("listPendingBuffers failed (non-critical)", error);
    return "";
  }
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
    const codeGraphPromise = mode !== "full" ? probeCodeGraph(userId) : Promise.resolve(false);

    // CONTEXT_MODE=assemble: the budgeted bundle is fetched alongside the context; any failure leaves null and
    // the index below is rendered as usual. The route is never called in the other modes.
    const assemblePromise: Promise<{ bundle: AssembledBundle | null; notice: string }> =
      mode === "assemble"
        ? resolveAssembleQuery(input).then((query) =>
            fetchAssembledBundle(
              buildAssembleRequest({
                budgetTokens: getAssembleBudgetTokens(),
                userId,
                sessionId: input.session_id,
                projectId,
                query,
              }),
            ),
          )
        : Promise.resolve({ bundle: null, notice: "" });

    let context: ContextResponse;
    try {
      context = await contextPromise;
    } catch (error) {
      // Never block session start: one-line notice, exit 0. Only a down/failing
      // service (network, 5xx) counts toward the consecutive-failure counter,
      // which upgrades the notice to fail-loud at FAIL_LOUD_THRESHOLD (D4);
      // auth / 404 / other 4xx get a notice naming what to fix instead.
      const { kind, count } = await recordFailure(error);
      logError(`context fetch failed (${kind})`, error);
      // Local-only parts still apply: buffers left by finished sessions keep
      // piling up during an outage unless this session is told about them,
      // and the local copy of the compact snapshot needs no service.
      const localResume = await localResumeBlock(input);
      const pendingDuringOutage = await buildPendingNote(input.session_id);
      outputWithContext(
        buildFailureNotice(kind, count, getFailLoudThreshold()) +
          (localResume ? `\n\n${localResume}` : "") +
          pendingDuringOutage,
      );
      return;
    }
    await resetTransportFailures();

    // Detect any unprocessed observation buffers left behind by prior sessions
    // and ask Claude to compile them before responding to the user.
    const pendingNote = await buildPendingNote(input.session_id);

    // After a compaction the stored snapshot may not be searchable yet (the raw
    // write is queued): merge the local copy the PreCompact hook left behind.
    context = await withLocalCompactSnapshot(context, input);

    let contextBody: string;
    let assembleNotice = "";
    if (mode === "full") {
      // Legacy full-content injection plus the continuity blocks (resume,
      // Previously, cards): see buildFullContext.
      const settled = await Promise.all(cardPromises.map((p) => p.catch(() => null)));
      contextBody = buildFullContext({
        context,
        cards: settled.filter((c): c is { label: string; lines: string[] } => c !== null),
        codeGraphAvailable: false,
        budgetTokens: getIndexBudgetTokens(),
        source: input.source,
        sessionId: input.session_id,
        predecessorIds:
          input.source === "resume" || input.source === "fork"
            ? await readPredecessorSessionIds(input.transcript_path, input.session_id)
            : undefined,
      });
    } else {
      const settled = await Promise.all(
        cardPromises.map((p) => p.catch(() => null)),
      );
      const cards = settled.filter(
        (c): c is { label: string; lines: string[] } => c !== null,
      );
      const codeGraphAvailable = await codeGraphPromise.catch(() => false);
      const assembled = await assemblePromise.catch(() => ({ bundle: null, notice: "" }));
      assembleNotice = assembled.notice;
      contextBody = buildIndexContext({
        context,
        cards,
        codeGraphAvailable,
        assembled: assembled.bundle ?? undefined,
        budgetTokens: getIndexBudgetTokens(),
        source: input.source,
        sessionId: input.session_id,
        predecessorIds:
          input.source === "resume" || input.source === "fork"
            ? await readPredecessorSessionIds(input.transcript_path, input.session_id)
            : undefined,
      });
    }

    const identityNotice = buildIdentityNotice(userId, getUserIdSource(), getServiceUrl());
    const combined =
      (contextBody || "") +
      pendingNote +
      (identityNotice ? `\n\n${identityNotice}` : "") +
      (assembleNotice ? `\n\n${assembleNotice}` : "");
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
