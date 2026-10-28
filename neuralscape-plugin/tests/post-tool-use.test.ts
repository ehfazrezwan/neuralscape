/**
 * Tests for the PostToolUse hook decision logic and row builder.
 *
 * The hook's main() runs at module load — we set NEURALSCAPE_TEST=1 in the
 * setup hook to suppress that, then exercise the exported pure functions.
 */

import { beforeAll, describe, expect, it } from "vitest";

beforeAll(() => {
  process.env.NEURALSCAPE_TEST = "1";
});

const mod = await import("../src/hooks/post-tool-use.js");
const { DENY_TOOLS, BORING_BASH_RE, NEURALSCAPE_TOOL_PREFIX, asString, shouldCapture, buildObservationRow } = mod;


describe("DENY_TOOLS", () => {
  it("contains the expected read-only and managed tools", () => {
    for (const t of [
      "Read", "Glob", "Grep", "NotebookRead", "TodoWrite",
      "ListMcpResourcesTool", "ReadMcpResourceTool",
      "ExitPlanMode", "EnterPlanMode", "EnterWorktree", "ExitWorktree",
    ]) {
      expect(DENY_TOOLS.has(t)).toBe(true);
    }
  });

  it("does NOT contain the mutating tools we capture", () => {
    for (const t of ["Edit", "Write", "Bash", "Task", "WebFetch", "WebSearch", "NotebookEdit"]) {
      expect(DENY_TOOLS.has(t)).toBe(false);
    }
  });
});


describe("BORING_BASH_RE", () => {
  it.each([
    "ls -la",
    "  pwd",
    "cd /tmp",
    "cat foo.txt",
    "head -n 10 file",
    "tail -f log.txt",
    "less file",
    "more file",
    "stat file",
    "file foo.bin",
    "which python",
    "where ls",
    "tree -L 2",
    "echo hello",
    "date",
    "whoami",
    "uname -a",
    "hostname",
    "env",
    "set -e",
    "unset FOO",
    "history 5",
  ])("matches boring command: %s", (cmd) => {
    expect(BORING_BASH_RE.test(cmd)).toBe(true);
  });

  it.each([
    "git status",
    "npm install",
    "python -m pytest",
    "docker compose up",
    "rg pattern",
    "make build",
  ])("does NOT match meaningful command: %s", (cmd) => {
    expect(BORING_BASH_RE.test(cmd)).toBe(false);
  });
});


describe("asString", () => {
  it("returns string unchanged", () => {
    expect(asString("hello")).toBe("hello");
  });

  it("returns empty string for null/undefined", () => {
    expect(asString(null)).toBe("");
    expect(asString(undefined)).toBe("");
  });

  it("JSON-stringifies objects", () => {
    expect(asString({ a: 1 })).toBe('{"a":1}');
  });

  it("falls back to String() for circular references", () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    // Just verify it doesn't throw — exact output depends on String() coercion
    expect(() => asString(circular)).not.toThrow();
  });
});


describe("shouldCapture", () => {
  it("returns false when tool_name is missing", () => {
    expect(shouldCapture({})).toBe(false);
  });

  it("returns false for any tool in DENY_TOOLS", () => {
    expect(shouldCapture({ tool_name: "Read" })).toBe(false);
    expect(shouldCapture({ tool_name: "Grep" })).toBe(false);
    expect(shouldCapture({ tool_name: "TodoWrite" })).toBe(false);
  });

  it("returns true for mutating tools", () => {
    expect(shouldCapture({ tool_name: "Edit", tool_input: { file_path: "/x" } })).toBe(true);
    expect(shouldCapture({ tool_name: "Write", tool_input: { file_path: "/x" } })).toBe(true);
    expect(shouldCapture({ tool_name: "Task", tool_input: {} })).toBe(true);
  });

  it("returns false for boring Bash commands", () => {
    expect(shouldCapture({ tool_name: "Bash", tool_input: { command: "ls -la" } })).toBe(false);
    expect(shouldCapture({ tool_name: "Bash", tool_input: { command: "pwd" } })).toBe(false);
  });

  it("returns true for non-trivial Bash commands", () => {
    expect(shouldCapture({ tool_name: "Bash", tool_input: { command: "git commit -am wip" } })).toBe(true);
    expect(shouldCapture({ tool_name: "Bash", tool_input: { command: "npm install x" } })).toBe(true);
  });

  it("handles Bash without command field gracefully", () => {
    // Empty command string fails the boring regex (regex requires actual content)
    // → falls through to capture
    expect(shouldCapture({ tool_name: "Bash", tool_input: {} })).toBe(true);
  });

  it("skips Neuralscape's own MCP tools (prevents feedback loops)", () => {
    expect(shouldCapture({ tool_name: "mcp__plugin_neuralscape_neuralscape__remember" })).toBe(false);
    expect(shouldCapture({ tool_name: "mcp__plugin_neuralscape_neuralscape__recall_memories" })).toBe(false);
    expect(shouldCapture({ tool_name: "mcp__plugin_neuralscape_neuralscape__list_memories" })).toBe(false);
  });

  it("exposes the prefix constant", () => {
    expect(NEURALSCAPE_TOOL_PREFIX).toBe("mcp__plugin_neuralscape_");
  });

  it("does NOT skip MCP tools from other plugins", () => {
    expect(shouldCapture({ tool_name: "mcp__some_other_plugin__tool" })).toBe(true);
  });

  it("skips Write events that target a file inside the observation buffer dir", () => {
    // Configure CLAUDE_PLUGIN_DATA so the observation dir resolves predictably
    const prev = process.env.CLAUDE_PLUGIN_DATA;
    process.env.CLAUDE_PLUGIN_DATA = "C:\\Users\\test\\.claude\\plugins\\data\\neuralscape";
    try {
      const inside = "C:\\Users\\test\\.claude\\plugins\\data\\neuralscape\\observations\\abc.jsonl";
      expect(
        shouldCapture({ tool_name: "Write", tool_input: { file_path: inside } }),
      ).toBe(false);
      const outside = "C:\\Users\\test\\projects\\my-project\\src\\main.ts";
      expect(
        shouldCapture({ tool_name: "Write", tool_input: { file_path: outside } }),
      ).toBe(true);
    } finally {
      if (prev === undefined) delete process.env.CLAUDE_PLUGIN_DATA;
      else process.env.CLAUDE_PLUGIN_DATA = prev;
    }
  });

  it("skips Edit events that target a file inside the observation buffer dir", () => {
    const prev = process.env.CLAUDE_PLUGIN_DATA;
    process.env.CLAUDE_PLUGIN_DATA = "/var/lib/neuralscape";
    try {
      const inside = "/var/lib/neuralscape/observations/sess.jsonl";
      expect(
        shouldCapture({ tool_name: "Edit", tool_input: { file_path: inside, old_string: "a", new_string: "b" } }),
      ).toBe(false);
    } finally {
      if (prev === undefined) delete process.env.CLAUDE_PLUGIN_DATA;
      else process.env.CLAUDE_PLUGIN_DATA = prev;
    }
  });

  it("handles forward-slash vs back-slash paths uniformly", () => {
    const prev = process.env.CLAUDE_PLUGIN_DATA;
    // Set data dir with forward slashes; tool input gives back-slashes
    process.env.CLAUDE_PLUGIN_DATA = "C:/Users/test/.claude/plugins/data/neuralscape";
    try {
      const back = "C:\\Users\\test\\.claude\\plugins\\data\\neuralscape\\observations\\x.jsonl";
      expect(
        shouldCapture({ tool_name: "Write", tool_input: { file_path: back } }),
      ).toBe(false);
    } finally {
      if (prev === undefined) delete process.env.CLAUDE_PLUGIN_DATA;
      else process.env.CLAUDE_PLUGIN_DATA = prev;
    }
  });

  it("skips NotebookEdit when target notebook is in observation dir", () => {
    const prev = process.env.CLAUDE_PLUGIN_DATA;
    process.env.CLAUDE_PLUGIN_DATA = "/var/ns";
    try {
      expect(
        shouldCapture({
          tool_name: "NotebookEdit",
          tool_input: { notebook_path: "/var/ns/observations/sess.ipynb", cell_id: "1", edit_mode: "replace" },
        }),
      ).toBe(false);
    } finally {
      if (prev === undefined) delete process.env.CLAUDE_PLUGIN_DATA;
      else process.env.CLAUDE_PLUGIN_DATA = prev;
    }
  });

  it("captures Write to files outside the observation dir", () => {
    const prev = process.env.CLAUDE_PLUGIN_DATA;
    process.env.CLAUDE_PLUGIN_DATA = "/var/ns";
    try {
      expect(
        shouldCapture({ tool_name: "Write", tool_input: { file_path: "/home/u/main.py" } }),
      ).toBe(true);
    } finally {
      if (prev === undefined) delete process.env.CLAUDE_PLUGIN_DATA;
      else process.env.CLAUDE_PLUGIN_DATA = prev;
    }
  });

  it("handles Write without a file_path field", () => {
    // Defensive: missing file_path falls through to capture
    expect(shouldCapture({ tool_name: "Write", tool_input: {} })).toBe(true);
  });
});


describe("buildObservationRow", () => {
  it("includes ts, session_id, tool, input, output", () => {
    const row = buildObservationRow({
      session_id: "sess-1",
      cwd: "/home/u/project",
      tool_name: "Edit",
      tool_input: { file_path: "/x.ts", old_string: "a", new_string: "b" },
      tool_response: "edit ok",
    });
    expect(row.ts).toBeTypeOf("string");
    expect(row.session_id).toBe("sess-1");
    expect(row.tool).toBe("Edit");
    expect(row.input).toEqual({ file_path: "/x.ts", old_string: "a", new_string: "b" });
    expect(row.output).toBe("edit ok");
    expect(row.cwd).toBe("/home/u/project");
    expect(row.project_id).toBe("project");
  });

  it("uses 'unknown' as session_id when missing", () => {
    const row = buildObservationRow({ tool_name: "Edit" });
    expect(row.session_id).toBe("unknown");
  });

  it("truncates large outputs", () => {
    const huge = "X".repeat(5000);
    const row = buildObservationRow({
      tool_name: "Bash",
      tool_input: { command: "do stuff" },
      tool_response: huge,
    });
    expect((row.output as string).length).toBeLessThan(huge.length);
    expect(row.output as string).toContain("[truncated]");
  });

  it("strips disallowed input fields per tool", () => {
    const row = buildObservationRow({
      tool_name: "Edit",
      tool_input: {
        file_path: "/x.ts",
        old_string: "a",
        new_string: "b",
        replace_all: true,  // not in allowlist
      },
    });
    expect(row.input).not.toHaveProperty("replace_all");
  });

  it("serializes object output via JSON.stringify", () => {
    // Pass an actual object so asString's JSON.stringify branch is exercised
    // (a JSON string would just round-trip as-is and miss the coercion path).
    const row = buildObservationRow({
      tool_name: "Bash",
      tool_input: { command: "x" },
      tool_response: { stdout: "hello", exit: 0 },
    });
    // asString → JSON.stringify produces canonical key order from object literal
    expect(row.output).toBe('{"stdout":"hello","exit":0}');
  });

  // Claude Code sends the tool result as `tool_response` (never
  // `tool_output`): a string for some tools, an object for most.
  it("captures a string tool_response", () => {
    const row = buildObservationRow({
      tool_name: "WebFetch",
      tool_input: { url: "https://example.com" },
      tool_response: "page summary text",
    });
    expect(row.output).toBe("page summary text");
  });

  it("captures an object tool_response as JSON (never [object Object])", () => {
    const resp = { stdout: "built 3 files", stderr: "", interrupted: false, isImage: false };
    const row = buildObservationRow({
      tool_name: "Bash",
      tool_input: { command: "make build" },
      tool_response: resp,
    });
    expect(row.output).toBe(JSON.stringify(resp));
    expect(row.output).not.toContain("[object Object]");
  });

  it("still accepts the legacy tool_output field", () => {
    const row = buildObservationRow({
      tool_name: "Bash",
      tool_input: { command: "make build" },
      tool_output: "legacy ok",
    });
    expect(row.output).toBe("legacy ok");
  });

  it("prefers tool_response over tool_output when both are present", () => {
    const row = buildObservationRow({
      tool_name: "Bash",
      tool_input: { command: "make build" },
      tool_response: "new",
      tool_output: "old",
    });
    expect(row.output).toBe("new");
  });

  it("truncates a large object tool_response to the same size cap", () => {
    const row = buildObservationRow({
      tool_name: "Bash",
      tool_input: { command: "make build" },
      tool_response: { stdout: "Y".repeat(5000), stderr: "" },
    });
    const out = row.output as string;
    expect(out).toContain("[truncated]");
    expect(out.length).toBeLessThan(1100);
  });

  it("redacts a <private> span even when truncation would split it", () => {
    // Opener lands in the elided middle, closer in the kept tail: redacting
    // AFTER truncation would leak the tail half of the secret.
    const text = "A".repeat(900) + "<private>" + "S".repeat(500) + "TAILSECRET</private> done";
    const row = buildObservationRow({
      tool_name: "Bash",
      tool_input: { command: "x" },
      tool_response: text,
    });
    expect(row.output as string).not.toContain("TAILSECRET");
    expect(row.output as string).not.toContain("SSS");
  });

  it("ts is ISO 8601", () => {
    const row = buildObservationRow({ tool_name: "Edit" });
    expect(row.ts).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
  });
});
