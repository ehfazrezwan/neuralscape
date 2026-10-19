/** The per-transcript lock: mutual exclusion, recovery of abandoned locks that never disturbs a live one, release. */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
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

const procTests = existsSync("/proc/self/stat") ? describe : describe.skip;
const ownStart = async () => {
  const stat = await readFile("/proc/self/stat", "utf-8");
  return stat.slice(stat.lastIndexOf(")") + 2).split(" ")[19];
};

describe("the recovery mutex follows the lock's rule", () => {
  it("a recoverer that stalls inside the mutex keeps it: it is not cleared by age while its recorder is alive", async () => {
    const path = await fresh();
    const lock = lockFile(path);
    const judged = { pid: DEAD_PID, token: "dead", at: 1 };
    await writeFile(lock, record(judged));
    const slow = recoverAbandoned(lock, judged, {
      mutexStaleMs: 50,
      afterVerify: async () => { await sleep(600); }, // stalled well past the mutex's age bound
    });
    await sleep(250);
    // a second recoverer sees a mutex older than the bound whose recorder (this process) is alive: it must not clear it
    expect(await recoverAbandoned(lock, judged, { mutexStaleMs: 50 })).toBe("retry");
    expect(await slow).toBe("removed");
    await gone(path);
    expect(await leftovers(path)).toEqual([]);
  });

  it("a successor that took the lock while a recoverer was stalled is never moved by a second recoverer", async () => {
    const path = await fresh();
    const lock = lockFile(path);
    const judged = { pid: DEAD_PID, token: "dead", at: 1 };
    await writeFile(lock, record(judged));
    let reached = false;
    const slow = recoverAbandoned(lock, judged, { mutexStaleMs: 50, afterVerify: async () => { reached = true; await sleep(500); } });
    while (!reached) await sleep(10);
    await sleep(120);
    const second = await recoverAbandoned(lock, judged, { mutexStaleMs: 50 });
    expect(second).toBe("retry");
    expect(await slow).toBe("removed");
  });

  it("a mutex whose recorder is gone, and that is old, is cleared (by moving it aside)", async () => {
    const path = await fresh();
    const lock = lockFile(path);
    const judged = { pid: DEAD_PID, token: "dead", at: 1 };
    await writeFile(lock, record(judged));
    await writeFile(lock + ".recover", record({ pid: DEAD_PID, token: "dead-recoverer", at: Date.now() - 60_000 }));
    expect(await recoverAbandoned(lock, judged, { mutexStaleMs: 50 })).toBe("retry"); // clears it
    expect(await recoverAbandoned(lock, judged, { mutexStaleMs: 50 })).toBe("removed"); // and the next one recovers
    expect(await leftovers(path)).toEqual([]);
  });
});

procTests("process identity (pid reuse, reboots, no /proc)", () => {
  it("a lock record carries the boot id and the owner's start time", async () => {
    const path = await fresh();
    await withTranscriptLock(path, async () => {
      const text = JSON.parse(await readFile(lockFile(path), "utf-8"));
      expect(text.boot).toBe((await readFile("/proc/sys/kernel/random/boot_id", "utf-8")).trim());
      expect(text.start).toBe(await ownStart());
    });
  });

  it("a week-old lock whose pid now belongs to an unrelated live process (different start time) is recovered", async () => {
    const path = await fresh();
    const unrelated = spawn("sleep", ["30"]);
    try {
      await writeFile(lockFile(path), record({ pid: unrelated.pid, token: "reused", at: Date.now() - 7 * 86_400_000, start: "1", boot: undefined }));
      expect((await withTranscriptLock(path, async () => 1, { staleMs: 1000, waitMs: 3000 })).ran).toBe(true);
    } finally {
      unrelated.kill();
    }
  });

  it("the same pid with the recorded start time is the real owner and is never recovered, however old", async () => {
    const path = await fresh();
    const text = record({ token: "live", at: Date.now() - 24 * 3_600_000, start: await ownStart() });
    await writeFile(lockFile(path), text);
    expect((await withTranscriptLock(path, async () => 1, { staleMs: 1000, waitMs: 300 })).ran).toBe(false);
    expect(await readFile(lockFile(path), "utf-8")).toBe(text);
  });

  it("a lock written in another boot is recovered even though its pid is alive now", async () => {
    const path = await fresh();
    await writeFile(lockFile(path), record({ token: "old-boot", at: Date.now() - 60_000, boot: "00000000-0000-0000-0000-000000000000", start: await ownStart() }));
    expect((await withTranscriptLock(path, async () => 1, { staleMs: 1000, waitMs: 3000 })).ran).toBe(true);
  });

  it("a recovery mutex whose recorder's pid was reused (different start time) is cleared; one whose recorder is alive is not", async () => {
    const path = await fresh();
    const lock = lockFile(path);
    const judged = { pid: DEAD_PID, token: "dead", at: 1 };
    await writeFile(lock, record(judged));
    const unrelated = spawn("sleep", ["30"]);
    try {
      await writeFile(lock + ".recover", record({ pid: unrelated.pid, token: "reused-recoverer", at: Date.now() - 60_000, start: "1" }));
      expect(await recoverAbandoned(lock, judged, { mutexStaleMs: 50 })).toBe("retry"); // clears the reused-pid mutex
      expect(await recoverAbandoned(lock, judged, { mutexStaleMs: 50 })).toBe("removed");
      await writeFile(lock, record(judged));
      await writeFile(lock + ".recover", record({ token: "live-recoverer", at: Date.now() - 60_000, start: await ownStart() }));
      expect(await recoverAbandoned(lock, judged, { mutexStaleMs: 50 })).toBe("retry");
      expect(await recoverAbandoned(lock, judged, { mutexStaleMs: 50 })).toBe("retry"); // still held by a live recoverer
    } finally {
      unrelated.kill();
    }
  });

  it("a young lock is never recovered, whatever its identity says", async () => {
    const path = await fresh();
    await writeFile(lockFile(path), record({ pid: DEAD_PID, token: "young", at: Date.now(), boot: "other" }));
    expect((await withTranscriptLock(path, async () => 1, { staleMs: 60_000, waitMs: 200 })).ran).toBe(false);
  });
});

describe("where /proc cannot confirm a live pid: no age fallback", () => {
  const captureStderr = async (run: () => Promise<unknown>): Promise<string> => {
    const original = process.stderr.write.bind(process.stderr);
    let seen = "";
    process.stderr.write = ((chunk: string | Uint8Array) => { seen += String(chunk); return true; }) as typeof process.stderr.write;
    try { await run(); } finally { process.stderr.write = original; }
    return seen;
  };

  it("a record without a start time whose pid is alive is never reclaimed, however old; the skip says why", async () => {
    const path = await fresh();
    const text = record({ token: "no-start", at: Date.now() - 30 * 86_400_000 });
    await writeFile(lockFile(path), text);
    let result: { ran: boolean } | undefined;
    const log = await captureStderr(async () => { result = await withTranscriptLock(path, async () => 1, { staleMs: 1000, waitMs: 300 }); });
    expect(result?.ran).toBe(false);
    expect(await readFile(lockFile(path), "utf-8")).toBe(text);
    expect(log).toContain("may still be running");
    expect(log).toContain("not reclaimed");
  });
});

/** Bundle the lock module for a real child process (the tests cannot import TypeScript in a child). */
async function bundledLock(): Promise<string> {
  const { build } = await import("esbuild");
  const dir = await mkdtemp(join(tmpdir(), "ns-lock-bundle-"));
  const out = join(dir, "lock.mjs");
  await build({ entryPoints: [join(__dirname, "..", "src", "core", "transcript-lock.ts")], bundle: true, platform: "node", format: "esm", outfile: out, logLevel: "silent" });
  await writeFile(join(dir, "worker.mjs"), `
    const { withTranscriptLock } = await import(${JSON.stringify(out)});
    const [transcript, ms] = process.argv.slice(2);
    await withTranscriptLock(transcript, async (h) => {
      console.log("held");
      await new Promise((r) => setTimeout(r, Number(ms)));
      console.log("done " + (await h.held()));
    }, { waitMs: 1000 });
  `);
  return join(dir, "worker.mjs");
}

describe("a suspended live owner", () => {
  it("is not reclaimed after the stale bound: SIGSTOP the owner, wait past the bound, a waiter skips; SIGCONT and it finishes still holding the lock", async () => {
    const path = await fresh();
    const worker = await bundledLock();
    const child = spawn("node", [worker, path, "1500"], { stdio: ["ignore", "pipe", "inherit"] });
    let out = "";
    child.stdout!.on("data", (d) => (out += d));
    const closed = new Promise<void>((resolve) => child.on("close", () => resolve()));
    try {
      while (!out.includes("held")) await sleep(20);
      child.kill("SIGSTOP");
      const text = await readFile(lockFile(path), "utf-8");
      await sleep(900); // well past the 300 ms stale bound used below
      const waiter = await withTranscriptLock(path, async () => "stole it", { staleMs: 300, waitMs: 500 });
      expect(waiter.ran).toBe(false); // a live owner is never reclaimed, even suspended past the bound
      expect(await readFile(lockFile(path), "utf-8")).toBe(text);
      child.kill("SIGCONT");
      await closed;
      expect(out).toContain("done true"); // it resumed still holding its lock: no successor could have posted
    } finally {
      child.kill("SIGCONT");
      child.kill("SIGKILL");
    }
    await gone(path);
    expect((await withTranscriptLock(path, async () => 1)).ran).toBe(true);
  }, 20000);
});
