/** The per-transcript lock: mutual exclusion, recovery of abandoned locks that never disturbs a live one, release. */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { chmod, mkdir, mkdtemp, readdir, readFile, unlink, utimes, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it, vi } from "vitest";
import { probeHooks, recoverAbandoned, recoveryMutexPath, withTranscriptLock } from "../src/core/transcript-lock.js";
import { flushTurns } from "../src/core/flush.js";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const fresh = async () => join(await mkdtemp(join(tmpdir(), "ns-lock-")), "transcript.jsonl");
const lockFile = (path: string) => path + ".neuralscape-lock";
const gone = (path: string) => expect(readFile(lockFile(path), "utf-8")).rejects.toThrow();
const DEAD_PID = 2 ** 22 + 4242; // above the kernel's pid limit: no such process
const record = (over: Record<string, unknown> = {}) => JSON.stringify({ pid: process.pid, token: "other", at: Date.now(), ...over });
const leftovers = async (path: string) => (await readdir(dirname(path))).filter((n) => /\.(tmp|stale\.|recover)/.test(n));

/**
 * A live process that may be a lock's owner: it started before the records written for it (``at: Date.now() - 1100``
 * with a 1000 ms stale bound).  A process that started AFTER a record was written cannot be its owner (a reused pid),
 * and without ``/proc`` (macOS) that is how the lock tells the two apart, so "a live owner" must be an older process.
 */
let ownerProcess: ReturnType<typeof spawn> | undefined;
let ownerSpawned = 0;
async function olderLiveProcess(): Promise<number> {
  if (!ownerProcess) {
    ownerProcess = spawn("sleep", ["120"]);
    ownerSpawned = Date.now();
  }
  const ready = ownerSpawned + 3500; // older than the whole seconds of ``ps`` elapsed time plus its margin
  if (Date.now() < ready) await sleep(ready - Date.now());
  return ownerProcess.pid!;
}
afterAll(() => { ownerProcess?.kill(); });

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

  it("never recovers a live owner's lock once it is older than the bound, and leaves it untouched", async () => {
    const path = await fresh();
    // the owner started before its record (a recorded /proc start time, tested below, confirms it however old the record is)
    const text = record({ pid: await olderLiveProcess(), token: "live", at: Date.now() - 1100 });
    await writeFile(lockFile(path), text);
    let ran = false;
    const result = await withTranscriptLock(path, async () => { ran = true; }, { waitMs: 300, staleMs: 1000 });
    expect(result.ran).toBe(false);
    expect(ran).toBe(false);
    expect(await readFile(lockFile(path), "utf-8")).toBe(text);
  }, 10000);

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

  it("a mutex whose recorder is gone, and that is old, is passed over (never deleted by a contender), so the same call recovers", async () => {
    const path = await fresh();
    const lock = lockFile(path);
    const judged = { pid: DEAD_PID, token: "dead", at: 1 };
    await writeFile(lock, record(judged));
    const deadMutex = record({ pid: DEAD_PID, token: "dead-recoverer", at: Date.now() - 60_000 });
    await writeFile(recoveryMutexPath(lock, judged, 0), deadMutex);
    expect(await recoverAbandoned(lock, judged, { mutexStaleMs: 50 })).toBe("removed");
    expect(await leftovers(path)).toEqual([]); // the winner deleted the dead generation after the record was gone
  });

  it("a dead recoverer's mutex is left as it was while the record is still there (a contender that gives up deletes nothing)", async () => {
    const path = await fresh();
    const lock = lockFile(path);
    const judged = { pid: DEAD_PID, token: "dead", at: 1 };
    await writeFile(lock, record(judged));
    const deadMutex = record({ pid: DEAD_PID, token: "dead-recoverer", at: Date.now() - 60_000 });
    await writeFile(recoveryMutexPath(lock, judged, 0), deadMutex);
    // a contender whose own mutex vanishes mid-way cannot proceed, and must not have touched the dead generation
    const outcome = await recoverAbandoned(lock, judged, {
      mutexStaleMs: 50,
      afterVerify: async () => { await unlink(recoveryMutexPath(lock, judged, 1)); },
    });
    expect(outcome).toBe("retry");
    expect(await readFile(recoveryMutexPath(lock, judged, 0), "utf-8")).toBe(deadMutex);
    expect(JSON.parse(await readFile(lock, "utf-8")).token).toBe("dead");
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

  it("a recovery mutex whose recorder's pid was reused (different start time) is passed over; one whose recorder is alive is not", async () => {
    const path = await fresh();
    const lock = lockFile(path);
    const judged = { pid: DEAD_PID, token: "dead", at: 1 };
    await writeFile(lock, record(judged));
    const unrelated = spawn("sleep", ["30"]);
    try {
      await writeFile(recoveryMutexPath(lock, judged), record({ pid: unrelated.pid, token: "reused-recoverer", at: Date.now() - 60_000, start: "1" }));
      expect(await recoverAbandoned(lock, judged, { mutexStaleMs: 50 })).toBe("removed"); // passes over the reused-pid mutex
      await writeFile(lock, record(judged));
      await writeFile(recoveryMutexPath(lock, judged), record({ token: "live-recoverer", at: Date.now() - 60_000, start: await ownStart() }));
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

  it("a record without a start time whose pid is alive and may be its owner is never reclaimed; the skip says why", async () => {
    const path = await fresh();
    // its process started before the record was written, so it may be the owner
    const text = record({ pid: await olderLiveProcess(), token: "no-start", at: Date.now() - 1100 });
    await writeFile(lockFile(path), text);
    let result: { ran: boolean } | undefined;
    const log = await captureStderr(async () => { result = await withTranscriptLock(path, async () => 1, { staleMs: 1000, waitMs: 300 }); });
    expect(result?.ran).toBe(false);
    expect(await readFile(lockFile(path), "utf-8")).toBe(text);
    expect(log).toContain("may still be running");
    expect(log).toContain("not reclaimed");
  }, 10000);

  it("a record without a start time whose pid now belongs to a process started after it was written is recovered (macOS pid reuse)", async () => {
    const path = await fresh();
    const unrelated = spawn("sleep", ["30"]);
    try {
      await writeFile(lockFile(path), record({ pid: unrelated.pid, token: "reused-no-proc", at: Date.now() - 7 * 86_400_000 }));
      expect((await withTranscriptLock(path, async () => 1, { staleMs: 1000, waitMs: 3000 })).ran).toBe(true);
      await gone(path);
    } finally {
      unrelated.kill();
    }
  });

  it("a pre-upgrade recovery mutex (no start time) whose recorder's pid was reused by a live process is passed over, not waited on forever", async () => {
    const path = await fresh();
    const lock = lockFile(path);
    const judged = { pid: DEAD_PID, token: "dead", at: 1 };
    await writeFile(lock, record(judged));
    const unrelated = spawn("sleep", ["30"]);
    try {
      // the mutex was written 60 s ago by a process that is gone; its pid belongs to `sleep`, started just now
      await writeFile(recoveryMutexPath(lock, judged), record({ pid: unrelated.pid, token: "reused-recoverer", at: Date.now() - 60_000 }));
      expect(await recoverAbandoned(lock, judged, { mutexStaleMs: 50 })).toBe("removed");
      expect(await leftovers(path)).toEqual([]);
    } finally {
      unrelated.kill();
    }
  });
});

describe("a lost recovery mutex authorizes nothing, and cleaners never delete a live recoverer's mutex", () => {
  it("a recoverer whose mutex was taken from it does not rename the lock (it re-checks its mutex before it acts)", async () => {
    const path = await fresh();
    const lock = lockFile(path);
    const judged = { pid: DEAD_PID, token: "dead", at: 1 };
    await writeFile(lock, record(judged));
    const text = await readFile(lock, "utf-8");
    const outcome = await recoverAbandoned(lock, judged, {
      afterVerify: async () => {
        // what the old cleaner did: the live recoverer's mutex is moved away and another recoverer holds a new one
        const mutex = recoveryMutexPath(lock, judged, 0);
        await unlink(mutex);
        await writeFile(mutex, record({ token: "second-recoverer" }));
      },
    });
    expect(outcome).toBe("retry");
    expect(await readFile(lock, "utf-8")).toBe(text); // the abandoned lock was not touched by the recoverer that lost its mutex
    expect(JSON.parse(await readFile(recoveryMutexPath(lock, judged, 0), "utf-8")).token).toBe("second-recoverer"); // nor the new mutex
  });

  it("two cleaners judge the same dead mutex; one finishes and a live recoverer takes over: the late one deletes nothing of the live recoverer's", async () => {
    const path = await fresh();
    const lock = lockFile(path);
    const first = { pid: DEAD_PID, token: "dead-1", at: 1 };
    await writeFile(lock, record(first));
    const deadMutex = record({ pid: DEAD_PID, token: "dead-recoverer", at: Date.now() - 60_000 });
    await writeFile(recoveryMutexPath(lock, first, 0), deadMutex);

    // B finds the dead mutex and stalls right after judging it abandoned (the old code renamed it aside next)
    let bJudged!: () => void;
    const bReached = new Promise<void>((r) => { bJudged = r; });
    let releaseB!: () => void;
    const bGate = new Promise<void>((r) => { releaseB = r; });
    const b = recoverAbandoned(lock, first, { mutexStaleMs: 50, afterMutexJudged: async () => { bJudged(); await bGate; } });
    await bReached;

    // A passes over the same dead mutex and recovers the lock; then the lock is abandoned again and R starts recovering it
    expect(await recoverAbandoned(lock, first, { mutexStaleMs: 50 })).toBe("removed");
    const second = { pid: DEAD_PID, token: "dead-2", at: 2 };
    await writeFile(lock, record(second));
    let rInside!: () => void;
    const rReached = new Promise<void>((r) => { rInside = r; });
    let releaseR!: () => void;
    const rGate = new Promise<void>((r) => { releaseR = r; });
    const r = recoverAbandoned(lock, second, { afterVerify: async () => { rInside(); await rGate; } });
    await rReached;
    const rMutex = recoveryMutexPath(lock, second, 0);
    const rMutexText = await readFile(rMutex, "utf-8");

    releaseB(); // B resumes with its stale view
    const bOutcome = await b;
    expect(bOutcome).toBe("retry");
    expect(await readFile(rMutex, "utf-8")).toBe(rMutexText); // R still holds its mutex
    expect(JSON.parse(await readFile(lock, "utf-8")).token).toBe("dead-2"); // and B did not touch R's record

    releaseR();
    expect(await r).toBe("removed");
    await gone(path);
    expect(await leftovers(path)).toEqual([]);
  });

  it("eight recoverers of one record behind a dead recoverer's mutex: exactly one removes it and never two act at once", async () => {
    const path = await fresh();
    const lock = lockFile(path);
    const judged = { pid: DEAD_PID, token: "dead", at: 1 };
    await writeFile(lock, record(judged));
    await writeFile(recoveryMutexPath(lock, judged, 0), record({ pid: DEAD_PID, token: "dead-recoverer", at: Date.now() - 60_000 }));
    let inside = 0;
    let overlapped = false;
    const outcomes = await Promise.all(Array.from({ length: 8 }, () => recoverAbandoned(lock, judged, {
      mutexStaleMs: 50,
      afterVerify: async () => { inside++; if (inside > 1) overlapped = true; await sleep(20); inside--; },
    })));
    expect(overlapped).toBe(false);
    expect(outcomes.filter((o) => o === "removed")).toHaveLength(1);
    expect(outcomes.filter((o) => o === "aborted")).toHaveLength(0);
    await gone(path);
    expect(await leftovers(path)).toEqual([]);
  });

  it("a chain of dead recoverers is passed over generation by generation", async () => {
    const path = await fresh();
    const lock = lockFile(path);
    const judged = { pid: DEAD_PID, token: "dead", at: 1 };
    await writeFile(lock, record(judged));
    for (let g = 0; g < 3; g++) await writeFile(recoveryMutexPath(lock, judged, g), record({ pid: DEAD_PID, token: `dead-recoverer-${g}`, at: 5 }));
    expect(await recoverAbandoned(lock, judged, { mutexStaleMs: 50 })).toBe("removed");
    expect(await leftovers(path)).toEqual([]);
  });
});

describe("a lock that exists but cannot be read", () => {
  const captureStderr = async (run: () => Promise<unknown>): Promise<string> => {
    const original = process.stderr.write.bind(process.stderr);
    let seen = "";
    process.stderr.write = ((chunk: string | Uint8Array) => { seen += String(chunk); return true; }) as typeof process.stderr.write;
    try { await run(); } finally { process.stderr.write = original; }
    return seen;
  };

  it("a directory at the lock path: the call returns at once, well inside the wait bound, logs why, and removes nothing", async () => {
    const path = await fresh();
    await mkdir(lockFile(path));
    await writeFile(join(lockFile(path), "inside"), "x");
    const started = Date.now();
    let result: { ran: boolean } | undefined;
    let ran = false;
    const log = await captureStderr(async () => { result = await withTranscriptLock(path, async () => { ran = true; }, { waitMs: 2000, staleMs: 1 }); });
    expect(result?.ran).toBe(false);
    expect(ran).toBe(false);
    expect(Date.now() - started).toBeLessThan(1500);
    expect(log).toContain("cannot be read");
    expect(log).toContain("not reclaimed");
    expect(existsSync(join(lockFile(path), "inside"))).toBe(true);
    expect(await leftovers(path)).toEqual([]);
  }, 5000);

  it("a file without read permission: same, and the file is left alone", async () => {
    const path = await fresh();
    const text = record({ pid: DEAD_PID, token: "dead", at: 1 });
    await writeFile(lockFile(path), text);
    await chmod(lockFile(path), 0o000);
    try {
      const readable = await readFile(lockFile(path), "utf-8").then(() => true, () => false);
      if (readable) return; // running as a user that ignores permissions (root): nothing to test
      const started = Date.now();
      const result = await withTranscriptLock(path, async () => 1, { waitMs: 2000, staleMs: 1 });
      expect(result.ran).toBe(false);
      expect(Date.now() - started).toBeLessThan(1500);
      expect(existsSync(lockFile(path))).toBe(true);
    } finally {
      await chmod(lockFile(path), 0o600);
    }
    expect(await readFile(lockFile(path), "utf-8")).toBe(text);
  }, 5000);
});

describe("which liveness-probe errors prove the owner is gone", () => {
  const withKillStub = async (code: string, run: () => Promise<void>) => {
    const original = process.kill.bind(process);
    const spy = vi.spyOn(process, "kill").mockImplementation(((pid: number, signal?: string | number) => {
      if (pid === DEAD_PID) throw Object.assign(new Error(code), { code });
      return original(pid, signal as never);
    }) as typeof process.kill);
    try { await run(); } finally { spy.mockRestore(); }
  };

  for (const code of ["EINVAL", "EPERM", "EMFILE"]) {
    it(`${code} from the probe is not proof of death: an old lock is not reclaimed`, async () => {
      const path = await fresh();
      const text = record({ pid: DEAD_PID, token: "maybe-alive", at: Date.now() - 60_000 });
      await writeFile(lockFile(path), text);
      await withKillStub(code, async () => {
        const result = await withTranscriptLock(path, async () => 1, { waitMs: 300, staleMs: 1000 });
        expect(result.ran).toBe(false);
      });
      expect(await readFile(lockFile(path), "utf-8")).toBe(text);
    });
  }

  it("ESRCH is proof of death: the same old lock is reclaimed", async () => {
    const path = await fresh();
    await writeFile(lockFile(path), record({ pid: DEAD_PID, token: "dead", at: Date.now() - 60_000 }));
    await withKillStub("ESRCH", async () => {
      expect((await withTranscriptLock(path, async () => 1, { waitMs: 3000, staleMs: 1000 })).ran).toBe(true);
    });
  });
});

describe("waiting on a held lock does not spawn ps on every poll", () => {
  it("one ps call for the whole wait (the verdict is cached per pid and record), and the lock is still not reclaimed", async () => {
    const path = await fresh();
    const text = record({ pid: await olderLiveProcess(), token: "slow-owner", at: Date.now() - 1100 });
    await writeFile(lockFile(path), text);
    const real = probeHooks.etime;
    let calls = 0;
    probeHooks.etime = (pid) => { calls++; return real(pid); };
    try {
      const started = Date.now();
      const result = await withTranscriptLock(path, async () => 1, { waitMs: 800, staleMs: 1000 });
      expect(result.ran).toBe(false);
      expect(Date.now() - started).toBeGreaterThanOrEqual(700); // about sixteen polls happened
    } finally {
      probeHooks.etime = real;
    }
    expect(calls).toBe(1);
    expect(await readFile(lockFile(path), "utf-8")).toBe(text);
  }, 10000);
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
  await writeFile(join(dir, "overlap.mjs"), `
    import { appendFile } from "node:fs/promises";
    const { withTranscriptLock } = await import(${JSON.stringify(out)});
    const [transcript, log] = process.argv.slice(2);
    const result = await withTranscriptLock(transcript, async () => {
      await appendFile(log, "in " + process.pid + "\\n");
      await new Promise((r) => setTimeout(r, 40));
      await appendFile(log, "out " + process.pid + "\\n");
    }, { waitMs: 20000, staleMs: 1000 });
    await appendFile(log, (result.ran ? "ran " : "skipped ") + process.pid + "\\n");
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

describe("overlapping hook processes on one abandoned lock (real processes, the built lock module)", () => {
  it("six processes race to recover a dead owner's lock behind a dead recoverer's mutex: every one delivers, never two at once", async () => {
    const path = await fresh();
    const worker = (await bundledLock()).replace("worker.mjs", "overlap.mjs");
    const log = join(dirname(path), "events.log");
    await writeFile(log, "");
    const judged = { pid: DEAD_PID, token: "dead-owner", at: Date.now() - 60_000 };
    await writeFile(lockFile(path), record(judged));
    await writeFile(recoveryMutexPath(lockFile(path), judged, 0), record({ pid: DEAD_PID, token: "dead-recoverer", at: Date.now() - 60_000 }));
    const children = Array.from({ length: 6 }, () => spawn("node", [worker, path, log], { stdio: ["ignore", "ignore", "inherit"] }));
    await Promise.all(children.map((c) => new Promise<void>((resolve) => c.on("close", () => resolve()))));
    const events = (await readFile(log, "utf-8")).split("\n").filter((l) => /^(in|out) /.test(l));
    expect(events).toHaveLength(12);
    for (let i = 0; i < events.length; i += 2) {
      const [a, b] = [events[i].split(" "), events[i + 1].split(" ")];
      expect(a[0]).toBe("in");
      expect(b).toEqual(["out", a[1]]); // each process leaves before the next one enters
    }
    expect(new Set(events.map((l) => l.split(" ")[1])).size).toBe(6);
    await gone(path);
    expect(await leftovers(path)).toEqual([]);
  }, 40000);
});
