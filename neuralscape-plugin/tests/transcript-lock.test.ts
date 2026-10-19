/** The per-transcript lock: mutual exclusion, take-over of abandoned locks, release on every exit. */

import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { withTranscriptLock } from "../src/core/transcript-lock.js";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const fresh = async () => join(await mkdtemp(join(tmpdir(), "ns-lock-")), "transcript.jsonl");
const lockFile = (path: string) => path + ".neuralscape-lock";
const gone = (path: string) => expect(readFile(lockFile(path), "utf-8")).rejects.toThrow();

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

  it("takes over a lock whose owner process is gone", async () => {
    const path = await fresh();
    await writeFile(lockFile(path), JSON.stringify({ pid: 2 ** 22 + 4242, token: "dead", at: Date.now() }));
    expect(await withTranscriptLock(path, async () => "ran")).toEqual({ ran: true, value: "ran" });
    await gone(path);
  });

  it("takes over a lock older than the stale bound even when its pid looks alive", async () => {
    const path = await fresh();
    await writeFile(lockFile(path), JSON.stringify({ pid: process.pid, token: "old", at: Date.now() - 60_000 }));
    expect((await withTranscriptLock(path, async () => 1, { staleMs: 1000 })).ran).toBe(true);
  });

  it("takes over a lock file that stayed empty (an owner that died before writing it)", async () => {
    const path = await fresh();
    await writeFile(lockFile(path), "");
    const { utimes } = await import("node:fs/promises");
    const old = new Date(Date.now() - 60_000);
    await utimes(lockFile(path), old, old);
    expect((await withTranscriptLock(path, async () => 1, { waitMs: 3000 })).ran).toBe(true);
  });

  it("does not run the body, and leaves the other owner's lock alone, when a live owner holds it past the wait", async () => {
    const path = await fresh();
    const record = JSON.stringify({ pid: process.pid, token: "other", at: Date.now() });
    await writeFile(lockFile(path), record);
    let ran = false;
    const result = await withTranscriptLock(path, async () => { ran = true; }, { waitMs: 200, staleMs: 60_000 });
    expect(result.ran).toBe(false);
    expect(ran).toBe(false);
    expect(await readFile(lockFile(path), "utf-8")).toBe(record);
  });

  it("releases only its own lock: a lock taken over meanwhile is not removed", async () => {
    const path = await fresh();
    const theirs = JSON.stringify({ pid: process.pid, token: "takeover", at: Date.now() });
    await withTranscriptLock(path, async () => {
      await writeFile(lockFile(path), theirs); // someone judged this lock abandoned and replaced it
    });
    expect(await readFile(lockFile(path), "utf-8")).toBe(theirs);
  });
});
