"""
Agent tools for the token-savings benchmark.

Two tool families, both presented to the Gemini agent as function declarations:

  * File tools (baseline + with-memory): ``list_dir``, ``grep``, ``read_file`` —
    sandboxed to the corpus checkout. This is the ONLY way the baseline agent can
    find anything, so their token cost IS the "reads a bunch of files" cost.
  * Memory tool (with-memory only): ``code_memory`` — routes to NS
    ``/v1/code-graph/{locate,query,neighbors}`` and returns a compact, ranked
    ``file:line — symbol`` answer. A first-hop hit is when this tool's first call
    returns the gold location, so no file read was ever needed.

Every tool returns a :class:`ToolResult` carrying both the text observation fed
back to the LLM and the structured ``raw`` payload the scorer inspects (e.g. the
ranked results a ``code_memory`` call produced, for first-hop-hit accounting).
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Keep observations bounded so a single tool call can't blow up token counts in
# an unrealistic way (real agents face context limits too). These caps apply to
# BOTH conditions identically, so they don't bias the with-vs-without comparison.
MAX_GREP_MATCHES = 40
MAX_LIST_ENTRIES = 100
MAX_READ_LINES = 200
MEMORY_TOP_K = 5

_SOURCE_EXTS = {".py", ".pyi", ".go", ".ts", ".tsx", ".js", ".jsx", ".rs", ".java"}


@dataclass
class ToolResult:
    """The outcome of one tool call."""

    observation: str  # text fed back to the LLM
    raw: dict = field(default_factory=dict)  # structured payload for the scorer
    ok: bool = True


# --------------------------------------------------------------------------- #
# File tools (sandboxed to the corpus checkout)
# --------------------------------------------------------------------------- #
class FileTools:
    """List / grep / read over a single corpus checkout, path-sandboxed."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"corpus root is not a directory: {self.root}")

    def _resolve(self, rel: str) -> Path | None:
        """Resolve a user-supplied relative path inside the sandbox, or None."""
        rel = (rel or "").strip().lstrip("/")
        target = (self.root / rel).resolve()
        try:
            target.relative_to(self.root)
        except ValueError:
            return None  # escape attempt
        return target

    def list_dir(self, path: str = ".") -> ToolResult:
        target = self._resolve(path)
        if target is None or not target.exists():
            return ToolResult(f"list_dir: path not found: {path}", ok=False)
        if target.is_file():
            return ToolResult(f"list_dir: {path} is a file, not a directory", ok=False)
        entries = []
        for p in sorted(target.iterdir()):
            if p.name == ".git":
                continue
            rel = p.relative_to(self.root).as_posix()
            entries.append(f"{rel}/" if p.is_dir() else rel)
        truncated = len(entries) > MAX_LIST_ENTRIES
        shown = entries[:MAX_LIST_ENTRIES]
        body = "\n".join(shown) if shown else "(empty)"
        if truncated:
            body += f"\n… ({len(entries) - MAX_LIST_ENTRIES} more entries omitted)"
        return ToolResult(body, raw={"count": len(entries)})

    def grep(self, pattern: str, path: str = ".") -> ToolResult:
        """Regex search over source files; returns file:line: text matches."""
        base = self._resolve(path)
        if base is None or not base.exists():
            return ToolResult(f"grep: path not found: {path}", ok=False)
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return ToolResult(f"grep: invalid regex: {e}", ok=False)

        files: list[Path]
        if base.is_file():
            files = [base]
        else:
            files = [
                p for p in sorted(base.rglob("*"))
                if p.is_file() and p.suffix in _SOURCE_EXTS and ".git" not in p.parts
            ]

        matches: list[str] = []
        total = 0
        for fp in files:
            try:
                lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            rel = fp.relative_to(self.root).as_posix()
            for i, line in enumerate(lines, start=1):
                if rx.search(line):
                    total += 1
                    if len(matches) < MAX_GREP_MATCHES:
                        matches.append(f"{rel}:{i}: {line.strip()[:200]}")
        if not matches:
            return ToolResult(f"grep: no matches for /{pattern}/", raw={"matches": 0})
        body = "\n".join(matches)
        if total > MAX_GREP_MATCHES:
            body += f"\n… ({total - MAX_GREP_MATCHES} more matches; refine the pattern)"
        return ToolResult(body, raw={"matches": total})

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> ToolResult:
        target = self._resolve(path)
        if target is None or not target.exists() or not target.is_file():
            return ToolResult(f"read_file: file not found: {path}", ok=False)
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            return ToolResult(f"read_file: {e}", ok=False)
        start = max(1, int(start_line or 1))
        if end_line is None:
            end = min(len(lines), start + MAX_READ_LINES - 1)
        else:
            end = min(len(lines), int(end_line), start + MAX_READ_LINES - 1)
        window = lines[start - 1:end]
        numbered = "\n".join(f"{start + i}\t{ln}" for i, ln in enumerate(window))
        header = f"{path} (lines {start}-{end} of {len(lines)}):"
        return ToolResult(f"{header}\n{numbered}", raw={"lines": len(window)})


# --------------------------------------------------------------------------- #
# Memory tool (routed NS code-graph)
# --------------------------------------------------------------------------- #
def _code_space(corpus_name: str) -> str:
    """Match the ns-native adapter's code_space convention exactly."""
    return f"code--ice-bench--{corpus_name}"


class MemoryTool:
    """
    ``code_memory`` — routed NS ``/v1/code-graph/*`` navigation.

    Arm selection is entirely server-side (the running stack's embedding config),
    exactly like the nl_locate A/B. This tool is arm-agnostic: it just queries and
    formats. ``knowledge_system`` defaults to the native KnowledgeSystem so the
    routed dispatch is used uniformly.
    """

    def __init__(
        self,
        api_url: str,
        corpus_name: str,
        knowledge_system: str = "code-native",
        timeout_s: float = 120.0,
    ):
        self.api_url = api_url.rstrip("/")
        self.corpus_name = corpus_name
        self.graph_id = _code_space(corpus_name)
        self.knowledge_system = knowledge_system
        self.client = httpx.Client(timeout=timeout_s)

    # ---- REST calls ------------------------------------------------------- #
    def _get(self, endpoint: str, params: dict) -> tuple[bool, str]:
        params = {**params, "knowledge_system": self.knowledge_system, "graph_id": self.graph_id}
        try:
            r = self.client.get(f"{self.api_url}/v1/code-graph/{endpoint}", params=params)
        except httpx.RequestError as e:
            return False, f"request error: {e}"
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:300]}"
        return True, r.text

    def code_memory(self, op: str, symbol: str = "", query: str = "") -> ToolResult:
        """
        Query the code memory.

        Args:
            op: "locate" (NL -> location), "symbol_lookup" (symbol -> definition),
                or "neighbors" (who calls a symbol).
            symbol: symbol name (for symbol_lookup / neighbors).
            query: natural-language description (for locate).
        """
        op = (op or "").strip()
        if op == "locate":
            if not query:
                return ToolResult("code_memory: 'query' is required for op=locate", ok=False)
            ok, text = self._get("locate", {"query": query})
        elif op == "symbol_lookup":
            if not symbol:
                return ToolResult("code_memory: 'symbol' is required for op=symbol_lookup", ok=False)
            ok, text = self._get("query", {"question": symbol})
        elif op == "neighbors":
            if not symbol:
                return ToolResult("code_memory: 'symbol' is required for op=neighbors", ok=False)
            ok, text = self._get("neighbors", {"label": symbol})
        else:
            return ToolResult(
                f"code_memory: unknown op '{op}' (use locate|symbol_lookup|neighbors)",
                ok=False,
            )

        if not ok:
            return ToolResult(f"code_memory[{op}] error: {text}", raw={"op": op}, ok=False)

        # Parse + compactly present. The parsed ranked results are stashed on raw
        # so the scorer can compute first-hop hit against gold.
        return self._format(op, text)

    # ---- formatting (compact, token-efficient) ---------------------------- #
    @staticmethod
    def _extract_items(text: str) -> list[dict]:
        """Best-effort structured extraction of {file, line, symbol} from a
        routed code-graph response so we can present `file:line` directly (the
        whole point — the agent should not need to read a file for the line)."""
        import json as _json

        items: list[dict] = []
        try:
            parsed = _json.loads(text)
        except (ValueError, TypeError):
            return items
        seq = None
        if isinstance(parsed, dict) and "results" in parsed:
            seq = parsed.get("results")
        elif isinstance(parsed, list):
            seq = parsed
        for it in seq or []:
            if not isinstance(it, dict):
                continue
            file = it.get("file", "")
            sym = it.get("fqn", "") or it.get("symbol", "") or it.get("name", "")
            line = it.get("line") or it.get("start_line") or it.get("lineno")
            snippet = (it.get("snippet") or it.get("preview") or "").strip()
            if file or sym:
                items.append({"file": file, "symbol": sym, "line": line, "snippet": snippet})
        return items

    def _format(self, op: str, text: str) -> ToolResult:
        # Lazy import so the module has no hard dependency on the scorer at import
        # time (keeps unit tests that only exercise FileTools lightweight).
        from icebench.trackq.score import _parse_ranked_results, _parse_symbol_set

        if op in ("locate", "symbol_lookup"):
            ranked = _parse_ranked_results(text, system="ns-native")
            items = self._extract_items(text)
            top_items = items[:MEMORY_TOP_K]
            if not ranked and not top_items:
                return ToolResult(
                    f"code_memory[{op}]: no location found.",
                    raw={"op": op, "ranked": [], "text": text[:2000]},
                )
            lines = []
            for i, it in enumerate(top_items):
                loc = it["file"] + (f":{it['line']}" if it.get("line") else "")
                tail = f"  # {it['snippet'][:80]}" if it.get("snippet") else ""
                lines.append(f"{i+1}. {loc} — {it['symbol']}{tail}")
            if not lines:  # ranked parsed but no structured items
                lines = [f"{i+1}. {f} — {s}" for i, (f, s) in enumerate(ranked[:MEMORY_TOP_K])]
            body = f"code_memory[{op}] top {len(lines)} match(es):\n" + "\n".join(lines)
            return ToolResult(
                body,
                raw={"op": op, "ranked": [list(r) for r in ranked],
                     "items": top_items, "text": text[:2000]},
            )

        # neighbors
        callers = sorted(_parse_symbol_set(text, system="ns-native"))
        if not callers:
            return ToolResult(
                f"code_memory[neighbors]: no callers returned.",
                raw={"op": op, "callers": [], "text": text[:2000]},
            )
        body = "code_memory[neighbors] callers:\n" + "\n".join(callers[:50])
        return ToolResult(body, raw={"op": op, "callers": callers, "text": text[:2000]})


# --------------------------------------------------------------------------- #
# Gemini function declarations (JSON-schema-ish, as google-genai expects)
# --------------------------------------------------------------------------- #
FILE_TOOL_DECLS = [
    {
        "name": "list_dir",
        "description": "List files and subdirectories at a path within the repository (relative to repo root).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path relative to repo root. Default '.'."}
            },
        },
    },
    {
        "name": "grep",
        "description": "Regex-search source files and return matching `file:line: text` locations. Use this to find where things are.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "A Python regular expression."},
                "path": {"type": "string", "description": "Directory/file to search under, relative to repo root. Default '.'."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a window of lines from a file (line-numbered). Prefer a narrow window.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to repo root."},
                "start_line": {"type": "integer", "description": "First line (1-indexed)."},
                "end_line": {"type": "integer", "description": "Last line (inclusive)."},
            },
            "required": ["path"],
        },
    },
]

MEMORY_TOOL_DECL = {
    "name": "code_memory",
    "description": (
        "Query the code memory index for a location WITHOUT reading files. "
        "op='locate' finds code by a natural-language description (pass `query`); "
        "op='symbol_lookup' finds where a symbol is defined (pass `symbol`); "
        "op='neighbors' finds which functions call a symbol (pass `symbol`). "
        "Prefer this tool first — it is far cheaper than reading files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["locate", "symbol_lookup", "neighbors"]},
            "symbol": {"type": "string", "description": "Symbol name for symbol_lookup/neighbors."},
            "query": {"type": "string", "description": "Natural-language description for locate."},
        },
        "required": ["op"],
    },
}

SUBMIT_TOOL_DECL = {
    "name": "submit_answer",
    "description": (
        "Submit your final answer and stop. For locate/symbol_lookup provide "
        "`location` as `path/to/file.py:LINE`. For a callers question provide "
        "`callers` as a list of caller names."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "file:line for locate/symbol_lookup."},
            "callers": {"type": "array", "items": {"type": "string"}, "description": "Caller names for neighbors."},
            "gave_up": {"type": "boolean", "description": "Set true if you cannot determine the answer."},
        },
    },
}
