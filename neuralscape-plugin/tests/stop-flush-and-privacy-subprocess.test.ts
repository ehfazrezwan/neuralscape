/**
 * Stop / SessionEnd hooks end to end (the BUILT scripts against a fixture
 * server): privacy of what is POSTed, the windowed compile, the SessionEnd
 * delivery of a final turn whose transcript entry lagged, and skipped
 * unanswered prompts.
 */

import { existsSync, readFileSync } from "node:fs";
import { mkdtemp, writeFile, appendFile } from "node:fs/promises";
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
  fx.failPosts.clear();
});

const FLUSH = "/v1/extensions/conversation-compiler/flush";
const COMPILE = "/v1/extensions/conversation-compiler/compile";
const CHECKPOINT = "/v1/checkpoint";
const posts = (path: string) => fx.requests.filter((r) => r.method === "POST" && r.path === path);

const user = (text: string) => JSON.stringify({ type: "user", message: { role: "user", content: text } }) + "\n";
const assistant = (text: string) =>
  JSON.stringify({ type: "assistant", message: { role: "assistant", content: [{ type: "text", text }] } }) + "\n";

async function session() {
  const dir = await mkdtemp(join(tmpdir(), "ns-stop-"));
  const transcript = join(dir, "t.jsonl");
  await writeFile(transcript, "", "utf-8");
  const env = (extra: Record<string, string> = {}) => ({
    NEURALSCAPE_USER_ID: "stop-test",
    NEURALSCAPE_URL: fx.url,
    NEURALSCAPE_PROJECT_ID: "stop-project",
    CLAUDE_PLUGIN_DATA: join(dir, "data"),
    NEURALSCAPE_TRANSCRIPT_LAG_WAIT_MS: "0",
    ...extra,
  });
  const stop = (last?: string, extra: Record<string, string> = {}) =>
    runHook(
      "session-end.js",
      JSON.stringify({
        session_id: "sess-stop",
        transcript_path: transcript,
        cwd: dir,
        hook_event_name: "Stop",
        ...(last ? { last_assistant_message: last } : {}),
      }),
      env(extra),
    );
  const end = (extra: Record<string, string> = {}) =>
    runHook(
      "session-summary.js",
      JSON.stringify({ session_id: "sess-stop", transcript_path: transcript, cwd: dir, hook_event_name: "SessionEnd", reason: "exit" }),
      env(extra),
    );
  return { dir, transcript, stop, end, env };
}

const LONG = (s: string) => `${s} — with enough words to be captured as a turn.`;

describe("privacy of what Stop and SessionEnd POST", () => {
  it("SessionEnd: a span opened in an earlier assistant entry and closed in the final one never reaches /v1/checkpoint or the flush", async () => {
    const { transcript, end } = await session();
    await appendFile(transcript, user("please do the thing"));
    await appendFile(transcript, assistant("Working on it. <private>SENTINEL-OPEN stays secret\nstill secret"));
    await appendFile(transcript, assistant("SENTINEL-CLOSE</private>\nAll done, shipped to staging."));
    const r = await end();
    expect(r.code).toBe(0);
    await waitFor(() => posts(CHECKPOINT).length === 1);
    expect(JSON.stringify(posts(CHECKPOINT)[0].body)).not.toMatch(/SENTINEL/);
    expect(JSON.stringify(posts(CHECKPOINT)[0].body)).toContain("All done, shipped to staging.");
    expect(JSON.stringify(fx.requests.map((q) => q.body))).not.toMatch(/SENTINEL/);
  });

  it("SessionEnd: a NESTED span split across entries, and a span opened in the prompt and closed in the answer, never reach any POST", async () => {
    const { transcript, end } = await session();
    await appendFile(transcript, user("my token is <private>SENTINEL-USER and more"));
    await appendFile(transcript, assistant("Noted SENTINEL-ANSWER</private> and fine. Start <private>outer <private>inner SENTINEL-1"));
    await appendFile(transcript, assistant("SENTINEL-2</private> SENTINEL-MID</private>\nAll done, shipped to staging."));
    await end();
    await waitFor(() => posts(CHECKPOINT).length === 1);
    expect(JSON.stringify(fx.requests.map((q) => q.body))).not.toMatch(/SENTINEL/);
    expect(JSON.stringify(posts(CHECKPOINT)[0].body)).toContain("All done, shipped to staging.");
  });

  it("Stop: an unclosed opener in a turn hides the rest of that turn (multi-line) from the flush", async () => {
    const { transcript, stop } = await session();
    await appendFile(transcript, user("what is the key?"));
    await appendFile(transcript, assistant(LONG("The key is <private>SENTINEL-A\nSENTINEL-B on the next line")));
    await stop();
    await waitFor(() => posts(FLUSH).length === 1);
    expect(JSON.stringify(posts(FLUSH)[0].body)).not.toMatch(/SENTINEL/);
  });
});

describe("windowed compile", () => {
  it("one compile per window, not one per Stop; SessionEnd compiles what is left; nothing delivered, nothing compiled", async () => {
    const { transcript, stop, end } = await session();
    for (const n of [1, 2, 3]) {
      await appendFile(transcript, user(`question ${n}`));
      await appendFile(transcript, assistant(LONG(`answer ${n}`)));
      await stop();
    }
    await waitFor(() => posts(FLUSH).length === 3);
    expect(posts(COMPILE)).toHaveLength(1); // first Stop only; the next two are inside the window
    expect(posts(COMPILE)[0].body).toMatchObject({ user_id: "stop-test" });

    await end();
    await waitFor(() => posts(COMPILE).length === 2); // SessionEnd: turns 2 and 3 were still uncompiled
    await end();
    await new Promise((r) => setTimeout(r, 300));
    expect(posts(COMPILE)).toHaveLength(2); // nothing new: no third compile

    await stop(); // a Stop that delivers nothing never compiles
    await new Promise((r) => setTimeout(r, 300));
    expect(posts(COMPILE)).toHaveLength(2);
  });

  it("COMPILE_INTERVAL_MIN=0 compiles after every Stop that delivered a turn", async () => {
    const { transcript, stop } = await session();
    for (const n of [1, 2]) {
      await appendFile(transcript, user(`question ${n}`));
      await appendFile(transcript, assistant(LONG(`answer ${n}`)));
      await stop(undefined, { CLAUDE_PLUGIN_OPTION_COMPILE_INTERVAL_MIN: "0" });
    }
    await waitFor(() => posts(COMPILE).length === 2);
  });

  it("a failed compile leaves the date dirty: the next Stop retries even inside the window", async () => {
    const { transcript, stop } = await session();
    fx.failPosts.add(COMPILE);
    await appendFile(transcript, user("q1"));
    await appendFile(transcript, assistant(LONG("a1")));
    await stop();
    await waitFor(() => posts(COMPILE).length === 1);
    fx.failPosts.clear();
    await appendFile(transcript, user("q2"));
    await appendFile(transcript, assistant(LONG("a2")));
    await stop();
    await waitFor(() => posts(COMPILE).length === 2); // retried, not held back by the window
  });

  it("other clients keep their single trigger at session end (framework session:end payload)", async () => {
    const dir = await mkdtemp(join(tmpdir(), "ns-stop-oc-"));
    await runHook(
      "session-end.js",
      JSON.stringify({ type: "session", action: "end", timestamp: "2026-11-16T10:00:00.000Z", context: { messageCount: 4 } }),
      { NEURALSCAPE_USER_ID: "stop-test", NEURALSCAPE_URL: fx.url, CLAUDE_PLUGIN_DATA: dir },
    );
    await waitFor(() => posts(COMPILE).length === 1);
    expect(posts(COMPILE)[0].body).toEqual({ date: "2026-11-16", user_id: "stop-test" });
  });
});

describe("a final turn whose transcript entry lagged", () => {
  it("is left pending by its Stop and delivered, then compiled, by SessionEnd", async () => {
    const { transcript, stop, end } = await session();
    await appendFile(transcript, user("the last question"));
    // The Stop event knows the final answer; the transcript has not caught up and the wait is 0.
    await stop("The very last answer, long enough to deliver.");
    await new Promise((r) => setTimeout(r, 300));
    expect(posts(FLUSH)).toHaveLength(0);
    expect(existsSync(transcript + ".neuralscape-offset")).toBe(false);

    await appendFile(transcript, assistant("The very last answer, long enough to deliver."));
    await end(); // the session ends: the transcript is complete now
    await waitFor(() => posts(FLUSH).length === 1);
    expect(posts(FLUSH)[0].body).toMatchObject({
      user_message: "the last question",
      assistant_response: "The very last answer, long enough to deliver.",
    });
    await waitFor(() => posts(COMPILE).length === 1);
    expect(readFileSync(transcript + ".neuralscape-offset", "utf-8").trim()).toBe("2");
  });
});

describe("a middle prompt with no assistant text", () => {
  it("is skipped (nothing to compile from an unanswered prompt) and its neighbours are delivered", async () => {
    const { transcript, stop } = await session();
    await appendFile(transcript, user("first question"));
    await appendFile(transcript, assistant(LONG("first answer")));
    await appendFile(transcript, user("interrupted question")); // the user pressed Esc and asked again
    await appendFile(transcript, user("third question"));
    await appendFile(transcript, assistant(LONG("third answer")));
    await stop();
    await waitFor(() => posts(FLUSH).length === 2);
    expect(posts(FLUSH).map((p) => p.body.user_message)).toEqual(["first question", "third question"]);
    expect(readFileSync(transcript + ".neuralscape-offset", "utf-8").trim()).toBe("5");
  });
});
