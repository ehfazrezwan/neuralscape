/**
 * File Read Gate — ranking + rendering (roadmap D3).
 *
 * Pure functions only (no HTTP, no filesystem) so the whole decision
 * surface is unit-testable from fixtures. The PreToolUse hook stats the
 * file, fetches candidate memories, and delegates everything else here.
 *
 * ## The file-reference signal (documented per D3)
 *
 * NS memories carry NO structured `files_read` / `files_modified` metadata —
 * the plugin's PostToolUse observations record `file_path` per row, but
 * compile-observations distills them into prose memories where the paths
 * survive only in `memory` content (and sometimes `title`/`tags`). The
 * strongest available signal is therefore:
 *
 *   1. fetch the recency-bounded memory list (`GET /v1/memories`, newest
 *      READ_GATE_FETCH_LIMIT project-scoped memories — a fast list, ~100ms;
 *      the hybrid `POST /v1/search` also runs a Graphiti pass whose latency
 *      routinely exceeds the PreToolUse hook budget), then
 *   2. a hard verification filter: keep only memories whose content/title/
 *      tags literally contain the file's basename (case-insensitive).
 *
 * "Modified vs merely read" is likewise heuristic: `observation_type` in
 * {bugfix, feature, refactor} is the strongest modify signal (stamped by the
 * compile-observations rubric), with modification verbs in the content as a
 * weaker fallback. Specificity = fewer distinct file mentions in the memory.
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

/** Max rows in the deny block. */
export const READ_GATE_MAX_ROWS = 10;

/** How many recent memories to fetch before the verification filter
 *  (the `GET /v1/memories` endpoint caps `limit` at 500). */
export const READ_GATE_FETCH_LIMIT = 500;

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

/** Basename of a path, tolerant of both separators. */
export function fileNameOf(filePath: string): string {
  return filePath.replace(/\\/g, "/").split("/").filter(Boolean).pop() ?? "";
}

/** Last `n` path segments joined with "/" — a higher-confidence needle
 *  than the bare basename (e.g. `src/utils.ts` vs `utils.ts`). */
export function pathTail(filePath: string, n = 2): string {
  const segments = filePath.replace(/\\/g, "/").split("/").filter(Boolean);
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

/** Everything a memory can reference a file through: content, title, tags. */
export function memoryHaystack(mem: NeuralscapeMemory): string {
  return `${mem.memory ?? ""}\n${mem.title ?? ""}\n${(mem.tags ?? []).join("\n")}`;
}

/**
 * Verification filter: does this memory actually reference the file?
 * Content, title, and tags are checked for the basename as a whole token
 * (case-insensitive) — substring hits like `a.ts` inside `data.ts` don't
 * count.
 */
export function referencesFile(mem: NeuralscapeMemory, filePath: string): boolean {
  const base = fileNameOf(filePath);
  if (!base) return false;
  return basenameMatchRe(base).test(memoryHaystack(mem));
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
 * Filter search hits down to verified references and rank them:
 * modified-the-file first, then specificity (fewer distinct files),
 * then tail-match confidence, then recency. Capped at `cap` rows.
 */
export function rankFileMemories(
  memories: NeuralscapeMemory[],
  filePath: string,
  cap = READ_GATE_MAX_ROWS,
): NeuralscapeMemory[] {
  const tail = pathTail(filePath).toLowerCase();
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
        tailHit: tail && hay.toLowerCase().includes(tail) ? 1 : 0,
        ts: Number.isNaN(ts) ? Number.NEGATIVE_INFINITY : ts,
      };
    });
  scored.sort(
    (a, b) => b.mod - a.mod || a.files - b.files || b.tailHit - a.tailHit || b.ts - a.ts,
  );
  return scored.slice(0, cap).map((s) => s.m);
}

// ── Deny-block rendering ─────────────────────────────────────────

const MCP = "mcp__plugin_neuralscape_neuralscape__";

/** One `#id | when | title | ~tokens` row (index format from disclosure). */
export function renderGateRow(mem: NeuralscapeMemory, now: Date = new Date()): string {
  const title = mem.title || distillTitle(mem.memory);
  const tokens = mem.token_estimate || estimateTokens(mem.memory);
  return `#${mem.id} | ${humanizeAge(mem.created_at ?? null, now)} | ${glyphFor(mem.observation_type)} ${title} | ~${tokens}`;
}

/**
 * The full deny reason: what happened, the ranked per-file timeline, the
 * escalation menu, and the exact override instruction. Short on purpose —
 * this whole block lands in Claude's context in place of the file.
 */
export function renderReadGateReason(
  filePath: string,
  sizeBytes: number,
  ranked: NeuralscapeMemory[],
  now: Date = new Date(),
): string {
  const kb = (sizeBytes / 1024).toFixed(1);
  const lines: string[] = [
    `[Neuralscape Read Gate] Skipped reading \`${filePath}\` (${kb} KB) — ` +
      `${ranked.length} stored ${ranked.length === 1 ? "memory references" : "memories reference"} this file. ` +
      `Check whether they already answer your question:`,
    "",
    "`#id | when | title | ~tokens`",
    ...ranked.map((m) => renderGateRow(m, now)),
    "",
    `Details: \`${MCP}get_memories\` with \`ids: [...]\` (the #ids above) for full payloads; ` +
      `\`${MCP}timeline\` with \`anchor\` = an #id for surrounding history; ` +
      `\`${MCP}recall_memories\` with \`index_only: true\` to browse further.`,
    `Override: if you still need the raw contents, just Read the same path again — ` +
      `the retry is always allowed and this gate stays quiet for this file for the rest of the session.`,
  ];
  // Server-sourced titles/content could carry <private> spans (e.g. from
  // ingested docs) — never re-emit them into the transcript.
  return redactPrivate(lines.join("\n"));
}

/** The PreToolUse deny decision in the shape the hooks API expects. */
export function buildDenyOutput(reason: string): Record<string, unknown> {
  return {
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: reason,
    },
  };
}
