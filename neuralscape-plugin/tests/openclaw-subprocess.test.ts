/**
 * OpenClaw adapter, end to end (the flag-on proof for a manual-install
 * feature): adapter-shaped `message:sent` and `session:end` payloads fed to
 * the BUILT scripts/conversation-turn.js and scripts/session-end.js, against a
 * fixture HTTP server. Asserts the flush and compile POST bodies.
 */

import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { type Fixture, runHook, startFixture, waitFor } from "./helpers/fixture.js";

let fx: Fixture;
beforeAll(async () => {
  fx = await startFixture();
});
afterAll(async () => {
  await fx.close();
});
beforeEach(() => {
  fx.requests.length = 0;
});

const env = (extra: Record<string, string> = {}) => ({
  NEURALSCAPE_USER_ID: "oc-e2e-user",
  NEURALSCAPE_URL: fx.url,
  CLAUDE_PLUGIN_DATA: mkdtempSync(join(tmpdir(), "ns-oc-")),
  ...extra,
});

const messageSent = (context: Record<string, unknown>, extra: Record<string, unknown> = {}) =>
  JSON.stringify({
    type: "message",
    action: "sent",
    sessionKey: "agent:main:telegram",
    timestamp: "2026-11-16T10:00:00.000Z",
    context,
    ...extra,
  });

const posts = (path: string) => fx.requests.filter((r) => r.method === "POST" && r.path === path);
const FLUSH = "/v1/extensions/conversation-compiler/flush";
const COMPILE = "/v1/extensions/conversation-compiler/compile";

describe("conversation-turn.js with OpenClaw message:sent payloads", () => {
  it("flushes the turn (userMessage spelling) with global scope when nothing names a project", async () => {
    const r = await runHook(
      "conversation-turn.js",
      messageSent({ content: "Here is a long enough assistant answer for capture.", channelId: "telegram", userMessage: "what is the plan?" }),
      env(),
    );
    expect(r.code).toBe(0);
    await waitFor(() => posts(FLUSH).length === 1);
    expect(posts(FLUSH)[0].body).toEqual({
      user_message: "what is the plan?",
      assistant_response: "Here is a long enough assistant answer for capture.",
      session_id: "agent:main:telegram",
      channel: "telegram",
      timestamp: "2026-11-16T10:00:00.000Z",
      project_id: null,
      user_id: "oc-e2e-user",
    });
  });

  it("accepts user_message and honours the PROJECT_ID override", async () => {
    await runHook(
      "conversation-turn.js",
      messageSent({ content: "Another sufficiently long assistant reply here.", channelId: "discord", user_message: "snake case question" }),
      env({ CLAUDE_PLUGIN_OPTION_PROJECT_ID: "pinned-project" }),
    );
    await waitFor(() => posts(FLUSH).length === 1);
    expect(posts(FLUSH)[0].body).toMatchObject({
      user_message: "snake case question",
      channel: "discord",
      project_id: "pinned-project",
    });
  });

  it("resolves the project from a workspace directory in the payload", async () => {
    const ws = mkdtempSync(join(tmpdir(), "oc-workspace-"));
    mkdirSync(join(ws, ".git"));
    writeFileSync(join(ws, ".neuralscape-project"), "from-workspace\n");
    await runHook(
      "conversation-turn.js",
      messageSent({ content: "A reply long enough to be captured, with workspace.", channelId: "slack", userMessage: "workspace question", workspaceDir: ws }),
      env(),
    );
    await waitFor(() => posts(FLUSH).length === 1);
    expect(posts(FLUSH)[0].body.project_id).toBe("from-workspace");
  });

  it("pairs the documented payloads: message:received (user text) then message:sent (assistant text, no user message)", async () => {
    const e = env();
    const received = JSON.stringify({
      type: "message",
      action: "received",
      sessionKey: "agent:main:telegram",
      context: { from: "user-1", content: "remind me what we decided about retries", channelId: "telegram" },
    });
    const sent = messageSent({ to: "user-1", content: "We decided on three retries with jitter.", success: true, channelId: "telegram" });
    const r1 = await runHook("conversation-turn.js", received, e);
    expect(r1.code).toBe(0);
    await new Promise((r) => setTimeout(r, 300));
    expect(posts(FLUSH)).toHaveLength(0); // inbound alone is not a turn
    await runHook("conversation-turn.js", sent, e);
    await waitFor(() => posts(FLUSH).length === 1);
    expect(posts(FLUSH)[0].body).toMatchObject({
      user_message: "remind me what we decided about retries",
      assistant_response: "We decided on three retries with jitter.",
      channel: "telegram",
    });
  });

  it("skips noise: a too-short reply posts nothing", async () => {
    await runHook("conversation-turn.js", messageSent({ content: "ok", channelId: "telegram" }), env());
    await new Promise((r) => setTimeout(r, 300));
    expect(posts(FLUSH)).toHaveLength(0);
  });
});

describe("session-end.js with OpenClaw session:end payloads", () => {
  const sessionEnd = (context: Record<string, unknown>, timestamp = "2026-11-16T22:30:00.000Z") =>
    JSON.stringify({ type: "session", action: "end", sessionKey: "agent:main:telegram", timestamp, context });

  it("triggers the compile for the event date and user when the session had messages", async () => {
    const r = await runHook("session-end.js", sessionEnd({ messageCount: 6 }), env());
    expect(r.code).toBe(0);
    await waitFor(() => posts(COMPILE).length === 1);
    expect(posts(COMPILE)[0].body).toEqual({ date: "2026-11-16", user_id: "oc-e2e-user" });
  });

  it("does not compile a session shorter than 2 messages", async () => {
    await runHook("session-end.js", sessionEnd({ messageCount: 1 }), env());
    await new Promise((r) => setTimeout(r, 300));
    expect(posts(COMPILE)).toHaveLength(0);
  });

  it("compiles when the message count is not reported", async () => {
    await runHook("session-end.js", sessionEnd({}), env());
    await waitFor(() => posts(COMPILE).length === 1);
  });
});


describe("excluded projects apply to the project the adapter resolves", () => {
  const wsNamed = (name: string) => {
    const root = mkdtempSync(join(tmpdir(), "oc-excl-"));
    const ws = join(root, name);
    mkdirSync(join(ws, ".git"), { recursive: true });
    return ws;
  };
  const excl = { CLAUDE_PLUGIN_OPTION_EXCLUDED_PROJECTS: "secret-*" };

  it("captures nothing from an excluded workspace (message:sent), but still captures another", async () => {
    const e = env(excl);
    await runHook(
      "conversation-turn.js",
      messageSent({ content: "A reply long enough to be captured, secret one.", userMessage: "q", workspaceDir: wsNamed("secret-app") }),
      e,
    );
    await runHook(
      "conversation-turn.js",
      messageSent({ content: "A reply long enough to be captured, open one.", userMessage: "q", workspaceDir: wsNamed("open-app") }),
      e,
    );
    await waitFor(() => posts(FLUSH).length >= 1);
    await new Promise((r) => setTimeout(r, 300));
    expect(posts(FLUSH)).toHaveLength(1);
    expect(posts(FLUSH)[0].body.project_id).toBe("open-app");
  });

  it("does not even remember the inbound message of an excluded workspace", async () => {
    const e = env(excl);
    const ws = wsNamed("secret-app");
    await runHook(
      "conversation-turn.js",
      JSON.stringify({ type: "message", action: "received", sessionKey: "k1", context: { content: "private words", workspaceDir: ws } }),
      e,
    );
    const { readdirSync, existsSync } = await import("node:fs");
    const obs = join(e.CLAUDE_PLUGIN_DATA, "observations");
    expect(existsSync(obs) ? readdirSync(obs).filter((f) => f.startsWith("openclaw-")) : []).toEqual([]);
  });

  it("the PROJECT_ID override that is excluded stops capture too, and session-end skips the compile", async () => {
    const e = env({ ...excl, CLAUDE_PLUGIN_OPTION_PROJECT_ID: "secret-pinned" });
    await runHook("conversation-turn.js", messageSent({ content: "A reply long enough to be captured, pinned.", userMessage: "q" }), e);
    await runHook(
      "session-end.js",
      JSON.stringify({ type: "session", action: "end", sessionKey: "k", timestamp: "2026-11-16T22:30:00.000Z", context: { messageCount: 5, workspaceDir: wsNamed("secret-app") } }),
      e,
    );
    await new Promise((r) => setTimeout(r, 400));
    expect(posts(FLUSH)).toHaveLength(0);
    expect(posts(COMPILE)).toHaveLength(0);
  });
});
