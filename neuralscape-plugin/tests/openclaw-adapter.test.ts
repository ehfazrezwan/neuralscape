/**
 * OpenClaw adapter: detection, turn extraction (both user-message spellings),
 * session-end threshold, and project resolution.
 */

import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

// The user id is read once at module load: set it before importing the adapter.
process.env.NEURALSCAPE_USER_ID = "oc-user";
const { detectClient, getSessionEndExtractor, getTurnExtractor } = await import("../src/adapters/detect.js");
const { extractOpenClawSessionEnd, extractOpenClawTurns, resolveOpenClawProjectId } = await import(
  "../src/adapters/openclaw.js"
);

const KEYS = ["CLAUDE_PLUGIN_OPTION_PROJECT_ID", "NEURALSCAPE_PROJECT_ID"];
let saved: Record<string, string | undefined>;
beforeEach(() => {
  saved = Object.fromEntries(KEYS.map((k) => [k, process.env[k]]));
  for (const k of KEYS) delete process.env[k];
});
afterEach(() => {
  for (const k of KEYS) {
    if (saved[k] === undefined) delete process.env[k];
    else process.env[k] = saved[k];
  }
});

const sent = (context: Record<string, unknown>, extra: Record<string, unknown> = {}) => ({
  type: "message",
  action: "sent",
  sessionKey: "agent:main:main",
  timestamp: "2026-11-16T10:00:00.000Z",
  context,
  ...extra,
});

describe("detectClient", () => {
  it("recognises type+action events as openclaw", () => {
    expect(detectClient(sent({}))).toBe("openclaw");
  });
  it("prefers claude-code when transcript_path or hook_event_name is present", () => {
    expect(detectClient({ ...sent({}), hook_event_name: "Stop" })).toBe("claude-code");
    expect(detectClient({ ...sent({}), transcript_path: "/x" })).toBe("claude-code");
  });
  it("falls back to generic", () => {
    expect(detectClient({ type: "message" })).toBe("generic");
  });
  it("routes the openclaw extractors", () => {
    expect(getTurnExtractor("openclaw")).toBe(extractOpenClawTurns);
    expect(getSessionEndExtractor("openclaw")).toBe(extractOpenClawSessionEnd);
  });
});

describe("extractOpenClawTurns", () => {
  it("maps content, channel, session key, timestamp and user", async () => {
    const [turn] = await extractOpenClawTurns(
      sent({ content: "the answer", channelId: "telegram", userMessage: "the question" }),
    );
    expect(turn).toMatchObject({
      userMessage: "the question",
      assistantResponse: "the answer",
      sessionId: "agent:main:main",
      channel: "telegram",
      timestamp: "2026-11-16T10:00:00.000Z",
      userId: "oc-user",
      projectId: undefined,
    });
  });
  it("accepts the snake_case user_message spelling, userMessage winning when both exist", async () => {
    expect((await extractOpenClawTurns(sent({ user_message: "snake", content: "a" })))[0].userMessage).toBe("snake");
    expect((await extractOpenClawTurns(sent({ user_message: "snake", userMessage: "camel", content: "a" })))[0].userMessage).toBe("camel");
  });
  it("defaults missing fields", async () => {
    const [turn] = await extractOpenClawTurns({ type: "message", action: "sent" });
    expect(turn.userMessage).toBe("");
    expect(turn.assistantResponse).toBe("");
    expect(turn.sessionId).toBe("unknown");
    expect(turn.channel).toBe("unknown");
    expect(Number.isNaN(Date.parse(turn.timestamp))).toBe(false);
  });
});

describe("pairing message:received with message:sent", () => {
  let dataDir: string;
  let prevData: string | undefined;
  beforeEach(() => {
    prevData = process.env.CLAUDE_PLUGIN_DATA;
    dataDir = mkdtempSync(join(tmpdir(), "oc-pair-"));
    process.env.CLAUDE_PLUGIN_DATA = dataDir;
  });
  afterEach(() => {
    if (prevData === undefined) delete process.env.CLAUDE_PLUGIN_DATA;
    else process.env.CLAUDE_PLUGIN_DATA = prevData;
  });
  const received = (content: string, sessionKey = "agent:main:telegram") => ({
    type: "message",
    action: "received",
    sessionKey,
    context: { from: "u1", content, channelId: "telegram" },
  });

  it("a received message yields no turn and is used as the user message of the next sent", async () => {
    expect(await extractOpenClawTurns(received("what is the plan?"))).toEqual([]);
    const [turn] = await extractOpenClawTurns({ ...sent({ content: "the plan is X", channelId: "telegram" }), sessionKey: "agent:main:telegram" });
    expect(turn.userMessage).toBe("what is the plan?");
  });
  it("is consumed once and scoped to the session", async () => {
    await extractOpenClawTurns(received("first question"));
    const other = await extractOpenClawTurns({ ...sent({ content: "a" }), sessionKey: "agent:other" });
    expect(other[0].userMessage).toBe("");
    const first = await extractOpenClawTurns({ ...sent({ content: "a" }), sessionKey: "agent:main:telegram" });
    expect(first[0].userMessage).toBe("first question");
    const again = await extractOpenClawTurns({ ...sent({ content: "b" }), sessionKey: "agent:main:telegram" });
    expect(again[0].userMessage).toBe("");
  });
  it("a payload-carried user message wins and leaves the remembered one alone", async () => {
    await extractOpenClawTurns(received("remembered"));
    const [turn] = await extractOpenClawTurns({ ...sent({ content: "a", userMessage: "inline" }), sessionKey: "agent:main:telegram" });
    expect(turn.userMessage).toBe("inline");
  });
  it("keys on the conversation or channel when there is no session key, and survives an unwritable data dir", async () => {
    await extractOpenClawTurns({ type: "message", action: "received", context: { content: "by conv", conversationId: "c-1" } });
    const [turn] = await extractOpenClawTurns({ type: "message", action: "sent", context: { content: "r", conversationId: "c-1" } });
    expect(turn.userMessage).toBe("by conv");
    process.env.CLAUDE_PLUGIN_DATA = join(dataDir, "not-a-dir-file");
    writeFileSync(join(dataDir, "not-a-dir-file"), "x");
    expect(await extractOpenClawTurns(received("lost"))).toEqual([]);
  });
});

describe("extractOpenClawSessionEnd", () => {
  const end = (context: Record<string, unknown>, ts?: string) =>
    extractOpenClawSessionEnd({ type: "session", action: "end", ...(ts ? { timestamp: ts } : {}), context });

  it("compiles when the message count is unknown or at least 2", () => {
    expect(end({}).shouldCompile).toBe(true);
    expect(end({ messageCount: 2 }).shouldCompile).toBe(true);
    expect(end({ messageCount: 40 }).shouldCompile).toBe(true);
  });
  it("skips compilation for a session with fewer than 2 messages", () => {
    expect(end({ messageCount: 1 }).shouldCompile).toBe(false);
    expect(end({ messageCount: 0 }).shouldCompile).toBe(false);
  });
  it("dates from the event timestamp, falling back to today on a missing or invalid one", () => {
    expect(end({}, "2026-03-04T23:59:00.000Z").date).toBe("2026-03-04");
    const today = new Date().toISOString().split("T")[0];
    expect(end({}).date).toBe(today);
    expect(end({}, "not a date").date).toBe(today);
    expect(end({}).userId).toBe("oc-user");
  });
});

describe("project resolution", () => {
  it("is undefined (global scope) when the payload carries no directory and no override is set", async () => {
    expect(resolveOpenClawProjectId(sent({ content: "x" }))).toBeUndefined();
    expect((await extractOpenClawTurns(sent({ content: "x" })))[0].projectId).toBeUndefined();
  });
  it("honours the PROJECT_ID override setting", async () => {
    process.env.CLAUDE_PLUGIN_OPTION_PROJECT_ID = "pinned";
    expect(resolveOpenClawProjectId(sent({ content: "x" }))).toBe("pinned");
    expect((await extractOpenClawTurns(sent({ content: "x" })))[0].projectId).toBe("pinned");
  });
  it("resolves a workspace directory from the payload like the other hooks do", async () => {
    const dir = mkdtempSync(join(tmpdir(), "oc-ws-"));
    mkdirSync(join(dir, ".git"));
    const sub = join(dir, "pkg");
    mkdirSync(sub);
    writeFileSync(join(dir, ".neuralscape-project"), "marked-project\n");
    expect(resolveOpenClawProjectId(sent({ workspaceDir: sub }))).toBe("marked-project");
    expect(resolveOpenClawProjectId(sent({ cwd: sub }))).toBe("marked-project");
    expect(resolveOpenClawProjectId(sent({}, { workspaceDir: sub }))).toBe("marked-project");
    expect(resolveOpenClawProjectId(sent({}, { cwd: sub }))).toBe("marked-project");
    expect((await extractOpenClawTurns(sent({ content: "x", workspaceDir: sub })))[0].projectId).toBe("marked-project");
  });
  it("the override wins over a directory signal", () => {
    process.env.CLAUDE_PLUGIN_OPTION_PROJECT_ID = "pinned";
    expect(resolveOpenClawProjectId(sent({ workspaceDir: tmpdir() }))).toBe("pinned");
  });
  it("ignores blank or non-string directory values", () => {
    expect(resolveOpenClawProjectId(sent({ workspaceDir: "  ", cwd: 5 }))).toBeUndefined();
  });
});

describe("hooks/openclaw-hooks.json", () => {
  it("is valid, relies on no unverified plugin-root variable, and points at shipped scripts", async () => {
    const { existsSync, readFileSync } = await import("node:fs");
    const path = join(__dirname, "..", "hooks", "openclaw-hooks.json");
    const manifest = JSON.parse(readFileSync(path, "utf-8"));
    expect(Object.keys(manifest.hooks).sort()).toEqual(["message:received", "message:sent", "session:end"]);
    for (const entries of Object.values(manifest.hooks) as Array<Array<{ hooks: Array<{ command: string }> }>>) {
      for (const { command } of entries.flatMap((e) => e.hooks)) {
        expect(command).not.toContain("${PLUGIN_ROOT}");
        const script = /scripts\/([a-z-]+\.js)/.exec(command)?.[1];
        expect(script, command).toBeDefined();
        expect(existsSync(join(__dirname, "..", "scripts", script!))).toBe(true);
      }
    }
  });
});
