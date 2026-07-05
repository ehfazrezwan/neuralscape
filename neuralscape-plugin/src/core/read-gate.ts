/**
 * File Read Gate — ranking + rendering (roadmap D3, reworked per audit 27
 * #31/#32).
 *
 * Pure functions only (no HTTP, no filesystem) so the whole decision
 * surface is unit-testable from fixtures. The PreToolUse hook stats the
 * file, loads/fetches candidate memories, and delegates everything else
 * here.
 *
 * ## Steer, never block (audit 27 #32)
 *
 * The gate used to DENY the Read and substitute a memory index for the file
 * contents. Titles are lossy and matching is heuristic, so denials both
 * blocked legitimate reads and served stale memory in place of real code.
 * The gate now emits `additionalContext` ("NS has N memories about this
 * file: …") while the Read ALWAYS proceeds.
 *
 * ## The file-reference signal
 *
 * NS memories carry NO structured `files_read` / `files_modified` metadata —
 * paths survive in prose (`memory` content, sometimes `title`/`tags`). The
 * verification filter matches on PATH TAILS: at least `dir/basename` when
 * the target path has directories (a bare basename like `utils.ts` is too
 * ambiguous across a repo — the pre-audit basename matcher was the false-
 * positive source), falling back to the basename only for single-segment
 * paths. Deeper tails (3+ segments) rank higher.
 *
 * "Modified vs merely read" stays heuristic: `observation_type` in
 * {bugfix, feature, refactor} is the strongest modify signal, with
 * modification verbs in the content as a weaker fallback. Specificity =
 * fewer distinct file mentions in the memory.
 */

import type { NeuralscapeMemory } from "../utils.js";
import { redactPrivate } from "../utils.js";
import {
  distillTitle,
  estimateTokens,
  glyphFor,
  humanizeAge,
} from "./disclosure.js";

// ── Config defaults ──────────────────────────────────────────────

/** Files at or below this size are never gated (config: READ_GATE_MIN_BYTES). */
export const DEFAULT_READ_GATE_MIN_BYTES = 1500;

/** Max rows in the steering block. */
export const READ_GATE_MAX_ROWS = 10;

/** How many recent index rows to fetch before the verification filter
 *  (audit 27 #31: was 500 FULL payloads; now a capped index-level fetch —
 *  `GET /v1/memories?fields=index` — once per session). */
export const READ_GATE_FETCH_LIMIT = 150;

/** Hard time budget for the one NS fetch (config: READ_GATE_TIME_BUDGET_MS).
 *  On timeout the hook allows the Read with no output — never block. */
export const DEFAULT_READ_GATE_TIME_BUDGET_MS = 2000;

// ── Binary / media bypass ────────────────────────────────────────

/** Extensions the gate never fires on — Read renders these visually or
 *  they're binary blobs no memory meaningfully summarizes. */
export const GATE_BYPASS_EXTENSIONS = new Set<string>([
  // images
  "png", "jpg", "jpeg", "gif", "webp", "bmp", "ico", "icns", "tiff", "tif", "heic", "svg",
  // documents rendered specially
  "pdf",
  // audio / video
  "mp3", "wav", "flac", "ogg", "m4a", "aac", "mp4", "mov", "avi", "mkv", "webm",
  // archives
  "zip", "tar", "gz", "tgz", "bz2", "xz", "7z", "rar", "jar",
  // fonts
  "woff", "woff2", "ttf", "otf", "eot",
  // compiled / opaque binaries
  "exe", "dll", "so", "dylib", "bin", "wasm", "class", "pyc", "o", "a",
  // databases / large opaque data
  "sqlite", "sqlite3", "db", "parquet", "pkl", "pt", "onnx", "gguf",
]);

/** Path segments, tolerant of both separators. */
function pathSegments(filePath: string): string[] {
  return filePath.replace(/\\/g, "/").split("/").filter(Boolean);
}

/** Basename of a path, tolerant of both separators. */
export function fileNameOf(filePath: string): string {
  return pathSegments(filePath).pop() ?? "";
}

/** Last `n` path segments joined with "/" — a higher-confidence needle
 *  than the bare basename (e.g. `src/utils.ts` vs `utils.ts`). Returns ""
 *  when the path has fewer than `n` segments. */
export function pathTail(filePath: string, n = 2): string {
  const segments = pathSegments(filePath);
  if (segments.length < n) return "";
  return segments.slice(-n).join("/");
}

/** True when the extension marks a binary/media file the gate must bypass. */
export function isBypassedExtension(filePath: string): boolean {
  const name = fileNameOf(filePath);
  const dot = name.lastIndexOf(".");
  if (dot <= 0) return false; // no extension (or dotfile) — gateable text
  return GATE_BYPASS_EXTENSIONS.has(name.slice(dot + 1).toLowerCase());
}

// ── Reference + ranking signals ──────────────────────────────────

/** observation_types stamped on memories that changed code (strong signal). */
export const MODIFY_OBSERVATION_TYPES = new Set<string>(["bugfix", "feature", "refactor"]);

/** Weak modify signal: modification verbs in the memory content. */
export const MODIFY_VERB_RE =
  /\b(edit(?:ed|ing)?|modif(?:y|ied|ying)|fix(?:ed|ing|es)?|refactor(?:ed|ing)?|rewr(?:ote|itten|ite)|implement(?:ed|ing)?|updat(?:ed|ing)|chang(?:ed|ing)|creat(?:ed|ing)|add(?:ed|ing)|renam(?:ed|ing)|delet(?:ed|ing)|remov(?:ed|ing)|wrote|patch(?:ed|ing|es)?)\b/i;

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Token-boundary matcher for a basename: `utils.ts` must not match
 * `data.ts`-style suffixes (no filename char before) or `utils.tsx`-style
 * extensions (no word char / dash after; a trailing `.` — sentence
 * punctuation — is fine). Case-insensitive.
 */
export function basenameMatchRe(basename: string): RegExp {
  return new RegExp(`(?<![\\w.-])${escapeRegExp(basename)}(?![\\w-])`, "i");
}

/**
 * Token-boundary matcher for a multi-segment path tail like
 * `gate/target-module.ts`: segments may be joined by either separator in
 * the haystack, a longer path prefix before the tail is fine (`src/gate/…`
 * still contains `gate/…`), but partial-segment prefixes (`irrigate/…`) and
 * extension extensions (`….tsx`) are not matches. Case-insensitive.
 */
export function tailMatchRe(tail: string): RegExp {
  const joined = tail.split("/").map(escapeRegExp).join("[/\\\\]");
  return new RegExp(`(?<![\\w.-])${joined}(?![\\w-])`, "i");
}

/** Everything a memory can reference a file through: content, title, tags. */
export function memoryHaystack(mem: NeuralscapeMemory): string {
  return `${mem.memory ?? ""}\n${mem.title ?? ""}\n${(mem.tags ?? []).join("\n")}`;
}

/**
 * Verification filter (audit 27 #32): does this memory reference the file
 * via a PATH TAIL? Requires at least `dir/basename` when the target path
 * has directories — a bare basename mention is deliberately NOT enough
 * (same-named files exist all over a repo). Single-segment targets fall
 * back to the token-bounded basename. Checks content, title, and tags,
 * case-insensitively.
 */
export function referencesFile(mem: NeuralscapeMemory, filePath: string): boolean {
  const hay = memoryHaystack(mem);
  const tail = pathTail(filePath, 2);
  if (tail) return tailMatchRe(tail).test(hay);
  const base = fileNameOf(filePath);
  if (!base) return false;
  return basenameMatchRe(base).test(hay);
}

/** 2 = modify-typed observation, 1 = modify verbs in content, 0 = read-only. */
export function modifiedScore(mem: NeuralscapeMemory): number {
  if (mem.observation_type && MODIFY_OBSERVATION_TYPES.has(mem.observation_type)) return 2;
  if (MODIFY_VERB_RE.test(mem.memory ?? "")) return 1;
  return 0;
}

// Path-like tokens with a letter-led extension ("src/x.ts", "utils.py") —
// the letter requirement excludes version numbers like "2.5".
const FILE_MENTION_RE = /[\w./\\-]*[\w-]+\.[A-Za-z][A-Za-z0-9]{0,7}\b/g;

/** Distinct files a memory mentions — fewer = more specific to this file. */
export function distinctFileMentions(content: string | null | undefined): number {
  if (!content) return 0;
  const seen = new Set<string>();
  for (const match of content.match(FILE_MENTION_RE) ?? []) {
    seen.add(fileNameOf(match).toLowerCase());
  }
  return seen.size;
}

/**
 * Filter candidates down to verified path-tail references and rank them:
 * modified-the-file first, then specificity (fewer distinct files), then
 * deeper-tail confidence (3+ segments), then recency. Capped at `cap` rows.
 */
export function rankFileMemories(
  memories: NeuralscapeMemory[],
  filePath: string,
  cap = READ_GATE_MAX_ROWS,
): NeuralscapeMemory[] {
  const deepTail = pathTail(filePath, 3);
  const deepRe = deepTail ? tailMatchRe(deepTail) : null;
  const scored = memories
    .filter((m) => referencesFile(m, filePath))
    .map((m) => {
      const ts = m.created_at ? new Date(m.created_at).getTime() : Number.NEGATIVE_INFINITY;
      // Rank over the SAME haystack the verification filter used
      // (content + title + tags) so a memory referencing the file via its
      // title isn't scored as maximally non-specific.
      const hay = memoryHaystack(m);
      return {
        m,
        mod: modifiedScore(m),
        files: distinctFileMentions(hay) || Number.MAX_SAFE_INTEGER,
        tailHit: deepRe && deepRe.test(hay) ? 1 : 0,
        ts: Number.isNaN(ts) ? Number.NEGATIVE_INFINITY : ts,
      };
    });
  scored.sort(
    (a, b) => b.mod - a.mod || a.files - b.files || b.tailHit - a.tailHit || b.ts - a.ts,
  );
  return scored.slice(0, cap).map((s) => s.m);
}

// ── Steering-context rendering (audit 27 #32: steer, never block) ─

const MCP = "mcp__plugin_neuralscape_neuralscape__";

/** One `#id | when | title | ~tokens` row (index format from disclosure). */
export function renderGateRow(mem: NeuralscapeMemory, now: Date = new Date()): string {
  const title = mem.title || distillTitle(mem.memory);
  const tokens = mem.token_estimate || estimateTokens(mem.memory);
  return `#${mem.id} | ${humanizeAge(mem.created_at ?? null, now)} | ${glyphFor(mem.observation_type)} ${title} | ~${tokens}`;
}

/**
 * The steering context injected ALONGSIDE the Read (never in place of it):
 * how many memories reference this file, their ranked index rows, and the
 * escalation menu. Short on purpose — it rides into Claude's context in
 * addition to the file contents.
 */
export function renderReadGateContext(
  filePath: string,
  ranked: NeuralscapeMemory[],
  now: Date = new Date(),
): string {
  const lines: string[] = [
    `[Neuralscape] ${ranked.length} stored ${ranked.length === 1 ? "memory references" : "memories reference"} ` +
      `\`${filePath}\` — prior context that may complement the file you are reading:`,
    "",
    "`#id | when | title | ~tokens`",
    ...ranked.map((m) => renderGateRow(m, now)),
    "",
    `Details: \`${MCP}get_memories\` with \`ids: [...]\` (the #ids above) for full payloads; ` +
      `\`${MCP}timeline\` with \`anchor\` = an #id for surrounding history; ` +
      `\`${MCP}recall_memories\` with \`index_only: true\` to browse further.`,
  ];
  // Server-sourced titles/content could carry <private> spans (e.g. from
  // ingested docs) — never re-emit them into the transcript.
  return redactPrivate(lines.join("\n"));
}

/**
 * The PreToolUse steering output: additionalContext only — NO permission
 * decision, so the Read proceeds through the normal permission flow
 * untouched (audit 27 #32: the gate must never deny-and-substitute).
 */
export function buildSteerOutput(context: string): Record<string, unknown> {
  return {
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      additionalContext: context,
    },
  };
}
