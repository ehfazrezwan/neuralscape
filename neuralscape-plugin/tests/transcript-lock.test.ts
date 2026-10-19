/** The per-transcript lock: mutual exclusion, recovery of abandoned locks that never disturbs a live one, release. */

import { mkdtemp, readdir, readFile, utimes, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { recoverAbandoned, withTranscriptLock } from "../src/core/transcript-lock.js";
import { flushTurns } from "../src/core/flush.js";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const fresh = async () => join(await mkdtemp(join(tmpdir(), "ns-lock-")), "transcript.jsonl");
const lockFile = (path: string) => path + ".neuralscape-lock";
const gone = (path: string) => expect(readFile(lockFile(path), "utf-8")).rejects.toThrow();
const DEAD_PID = 2 ** 22 + 4242; // above the kernel's pid limit: no such process
const record = (over: Record<string, unknown> = {}) => JSON.stringify({ pid: process.pid, token: "other", at: Date.now(), ...over });
const leftovers = async (path: string) => (await readdir(dirname(path))).filter((n) => /\.(tmp|stale\.|recover)/.test(n));

describe("withTranscriptLock", () => {
  it("never runs two bodies for the same transcript at once, and runs them all", async () => {
    const path = await fresh();
    let inside = 0;
    let overlapped = false;
    let done = 0;
    const body = async () => {
      inside++;
      if (inside > 1) overlapped = true;
      await sleep(30);
      inside--;
      done++;
    };
    const results = await Promise.all([1, 2, 3, 4].map(() => withTranscriptLock(path, body, { waitMs: 5000 })));
    expect(overlapped).toBe(false);
    expect(done).toBe(4);
    expect(results.every((r) => r.ran)).toBe(true);
    await gone(path);
    expect(await leftovers(path)).toEqual([]);
  });

  it("different transcripts do not wait for each other", async () => {
    const [a, b] = [await fresh(), await fresh()];
    let both = 0;
    let inside = 0;
    const body = async () => {
      inside++;
      both = Math.max(both, inside);
      await sleep(40);
      inside--;
    };
    await Promise.all([withTranscriptLock(a, body), withTranscriptLock(b, body)]);
    expect(both).toBe(2);
  });

  it("releases the lock when the body throws, and passes the error on", async () => {
    const path = await fresh();
    await expect(withTranscriptLock(path, async () => { throw new Error("boom"); })).rejects.toThrow("boom");
    await gone(path);
    expect((await withTranscriptLock(path, async () => 7))).toEqual({ ran: true, value: 7 });
  });

  it("the lock is complete the moment it is visible (record written before the link)", async () => {
    const path = await fresh();
    await withTranscriptLock(path, async (handle) => {
      const text = await readFile(lockFile(path), "utf-8");
      expect(JSON.parse(text).pid).toBe(process.pid);
      expect(await handle.held()).toBe(true);
    });
  });
});

describe("abandoned locks", () => {
  it("recovers a lock whose owner is gone AND that is older than the bound", async () => {
    const path = await fresh();
    await writeFile(lockFile(path), record({ pid: DEAD_PID, token: "dead", at: Date.now() - 60_000 }));
    expect(await withTranscriptLock(path, async () => "ran", { staleMs: 1000 })).toEqual({ ran: true, value: "ran" });
    await gone(path);
    expect(await leftovers(path)).toEqual([]);
  });

  it("does not recover a young lock even when its owner is gone (it waits, then leaves the turns)", async () => {
    const path = await fresh();
    const text = record({ pid: DEAD_PID, token: "dead", at: Date.now() });
    await writeFile(lockFile(path), text);
    let ran = false;
    const result = await withTranscriptLock(path, async () => { ran = true; }, { waitMs: 200, staleMs: 60_000 });
    expect(result.ran).toBe(false);
    expect(ran).toBe(false);
    expect(await readFile(lockFile(path), "utf-8")).toBe(text);
  });

  it("never recovers a live owner's lock, however old, and leaves it untouched", async () => {
    const path = await fresh();
    const text = record({ token: "live", at: Date.now() - 3_600_000 });
    await writeFile(lockFile(path), text);
    let ran = false;
    const result = await withTranscriptLock(path, async () => { ran = true; }, { waitMs: 300, staleMs: 1000 });
    expect(result.ran).toBe(false);
    expect(ran).toBe(false);
    expect(await readFile(lockFile(path), "utf-8")).toBe(text);
  });

  it("recovers a lock file that is not a record once it is older than the bound", async () => {
    const path = await fresh();
    await writeFile(lockFile(path), "{garbage");
    const old = new Date(Date.now() - 60_000);
    await utimes(lockFile(path), old, old);
    expect((await withTranscriptLock(path, async () => 1, { staleMs: 1000, waitMs: 3000 })).ran).toBe(true);
  });

  it("many simultaneous recoverers of one abandoned lock: never two deliverers at once, all accounted for", async () => {
    const path = await fresh();
    await writeFile(lockFile(path), record({ pid: DEAD_PID, token: "dead", at: Date.now() - 60_000 }));
    let inside = 0;
    let overlapped = false;
    let ran = 0;
    const body = async () => {
      inside++;
      if (inside > 1) overlapped = true;
      await sleep(15);
      inside--;
      ran++;
    };
    const results = await Promise.all(Array.from({ length: 16 }, () => withTranscriptLock(path, body, { waitMs: 8000, staleMs: 1000 })));
    expect(overlapped).toBe(false);
    expect(ran).toBe(results.filter((r) => r.ran).length);
    expect(ran).toBeGreaterThan(0);
    await gone(path);
    expect(await leftovers(path)).toEqual([]);
  });

  it("recovery of a record that is no longer on disk removes nothing (the lock changed hands)", async () => {
    const path = await fresh();
    const live = record({ token: "successor" });
    await writeFile(lockFile(path), live);
    const outcome = await recoverAbandoned(lockFile(path), { pid: DEAD_PID, token: "dead", at: 1 });
    expect(outcome).toBe("retry");
    expect(await readFile(lockFile(path), "utf-8")).toBe(live);
    expect(await leftovers(path)).toEqual([]);
  });

  it("a successor that appears between the check and the rename is NOT put back; the recoverer aborts and the owner notices", async () => {
    const path = await fresh();
    const lock = lockFile(path);
    await writeFile(lock, record({ pid: DEAD_PID, token: "dead", at: 1 }));
    // the live owner, as the body of a real acquisition would be: take the lock after the abandoned one is removed
    let handle: { held(): Promise<boolean> } | undefined;
    const outcome = await recoverAbandoned(lock, { pid: DEAD_PID, token: "dead", at: 1 }, {
      afterVerify: async () => {
        // the abandoned record is replaced under the recoverer's feet by a live owner's lock
        const { rename } = await import("node:fs/promises");
        await rename(lock, lock + ".gone");
        await writeFile(lock, record({ token: "live-owner" }));
        handle = { held: async () => (await readFile(lock, "utf-8").catch(() => "")).includes("live-owner") };
      },
    });
    expect(outcome).toBe("aborted");
    expect(await handle!.held()).toBe(false); // nothing was restored: the owner's own check sees its lock gone
    await expect(readFile(lock, "utf-8")).rejects.toThrow();
  });
});

describe("the owner verifies its lock", () => {
  it("held() turns false when the lock file is gone or carries another token, and release then removes nothing of another's", async () => {
    const path = await fresh();
    await withTranscriptLock(path, async (handle) => {
      expect(await handle.held()).toBe(true);
      await writeFile(lockFile(path), record({ token: "successor" }));
      expect(await handle.held()).toBe(false);
    });
    expect(JSON.parse(await readFile(lockFile(path), "utf-8")).token).toBe("successor"); // another's lock survives our release
  });

  it("a flush stops before the next POST when its lock was lost, and posts nothing after", async () => {
    const path = await fresh();
    const posted: string[] = [];
    // flushTurns posts through the network; the proceed hook is consulted first, so a lost lock never reaches it
    const turns = [1, 2, 3].map((i) => ({
      userMessage: `prompt ${i} long enough`, assistantResponse: `an answer number ${i} with plenty of text`,
      sessionId: "s", channel: "claude-code", timestamp: "t", userId: "u",
    }));
    let calls = 0;
    const result = await flushTurns(turns, { proceed: async () => { calls++; posted.push(`check${calls}`); return false; } });
    expect(result.firstFailedIndex).toBe(0);
    expect(result.flushed).toBe(0);
    expect(posted).toEqual(["check1"]);
    void path;
  });
});
