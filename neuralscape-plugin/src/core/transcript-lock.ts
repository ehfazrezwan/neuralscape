/**
 * A per-transcript lock shared by every process that delivers turns from the same transcript.
 *
 * Two hooks can run at the same time (a Stop hook still running when the next one starts), and each reads the
 * cursor, posts the turns after it and writes the new cursor.  Without a lock both read the same cursor and
 * post the same turns, and the one that finishes last decides where the cursor ends up.  The lock makes the
 * whole read-deliver-commit sequence of one process finish before the next one reads the cursor.
 *
 * Node has no portable operating-system file lock without a native module, so this is a lock file:
 *
 * - **Acquire:** the owner record (pid, token, time) is written to a private temporary file and hard-linked to
 *   the lock path.  ``link`` is atomic and fails when the path exists, and the record is complete the moment the
 *   lock becomes visible.
 * - **Abandoned locks:** a lock is recovered only when its owner is dead AND the lock is older than ``staleMs``.
 *   The owner is dead when its pid is not alive on this host, when the record comes from another boot
 *   (``/proc/sys/kernel/random/boot_id``), or when the pid's process start time (``/proc/<pid>/stat``) differs
 *   from the one recorded (the pid was reused).  A live owner is never recovered, however old, and there is no
 *   age fallback: a suspended or slow owner can resume and post while a successor posts, so a lock whose owner
 *   might be alive is never reclaimed.  Where ``/proc`` cannot confirm who a live pid is (macOS, or a record
 *   without a start time) the waiter waits its bound and skips its flush with a log line that says so; the turns
 *   wait for the next flush (a lock whose pid is truly gone is still recovered, since ``kill(pid, 0)`` says so).
 * - **Recovery:** recoverers are serialized by a short-lived recovery mutex that follows the same rule: its
 *   record carries the same identity, and it is cleared only when its recorder is dead and it is old, by moving
 *   it aside and checking the token, never by deleting it by age.  Under the mutex the lock is re-read and, only
 *   if it is still the exact abandoned record, renamed to a unique name and deleted.  If what was moved turns
 *   out not to be that record, it is NOT put back (putting it back is the race that lets a third hook acquire
 *   the vacant path): the recoverer abandons its own flush instead.
 * - **The owner verifies:** the body receives ``held()``, which is true only while the lock file still carries
 *   the owner's token.  The delivery code calls it before every POST and before it writes the cursor, and
 *   stops cleanly when the lock is gone.
 *
 * Residual window (it requires an owner or recoverer that is declared dead while it is alive, for example
 * because it runs in another pid namespace, and a lock older than ``staleMs``): a recoverer's rename can move a
 * successor's lock in the gap between its re-read and its rename.  The owner then notices at its next ``held()`` check, so at
 * most the one POST already past that check can overlap another hook (one turn delivered twice), and the cursor
 * is only ever written forward (see ``commitClaudeCodeFlush``).  Hooks of different Claude Code instances that
 * run in different containers but share a transcript file cannot see each other's pids.
 */

import { link, readFile, rename, stat, unlink, writeFile } from "node:fs/promises";
import { randomBytes } from "node:crypto";
import { logError } from "../utils.js";

const LOCK_SUFFIX = ".neuralscape-lock";

/** How long to wait for another process to finish, and how old a lock must be to count as abandoned. */
export const LOCK_WAIT_MS = 20_000;
export const LOCK_STALE_MS = 90_000;
const POLL_MS = 50;
/** A recovery mutex must be older than this, and its recorder dead, before it is cleared. */
const RECOVERY_MUTEX_STALE_MS = 10_000;
export interface LockOptions {
  waitMs?: number;
  staleMs?: number;
}

/** What the body can ask about the lock it holds. */
export interface LockHandle {
  /** True only while the lock file still carries this owner's token. */
  held(): Promise<boolean>;
}

interface LockOwner {
  pid: number;
  token: string;
  at: number;
  /** The boot the record was written in, and the owner process's start time (both Linux ``/proc``; absent elsewhere). */
  boot?: string;
  start?: string;
}

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM"; // exists, owned by someone else
  }
}

let cachedBoot: string | null | undefined;

/** The kernel's boot id, or null where ``/proc`` has none. */
async function bootId(): Promise<string | null> {
  if (cachedBoot === undefined) {
    cachedBoot = await readFile("/proc/sys/kernel/random/boot_id", "utf-8").then((t) => t.trim() || null, () => null);
  }
  return cachedBoot;
}

/** The start time (clock ticks since boot, field 22 of ``/proc/<pid>/stat``) of a process, or null if unreadable. */
async function startTimeOf(pid: number): Promise<string | null> {
  try {
    const stat = await readFile(`/proc/${pid}/stat`, "utf-8");
    // the command name (field 2) may contain spaces and parentheses: fields resume after the LAST ")"
    const fields = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
    return fields[19] ?? null; // field 22 overall; field 3 is the first one after the name
  } catch {
    return null;
  }
}

/** This process's identity as recorded in the records it writes. */
async function identity(): Promise<Pick<LockOwner, "boot" | "start">> {
  const [boot, start] = await Promise.all([bootId(), startTimeOf(process.pid)]);
  return { ...(boot ? { boot } : {}), ...(start ? { start } : {}) };
}

/**
 * Whether the process that wrote ``record`` is gone and the record is older than ``staleMs``.  Another boot, a
 * dead pid, or a pid whose start time differs from the recorded one (reuse) is dead; a pid that is alive and
 * confirmed by its start time never is.  Where the identity cannot be confirmed (no recorded start time, or no
 * ``/proc``), a live pid is believed alive, however old the record: there is no age fallback.
 */
async function ownerGone(record: LockOwner, staleMs: number): Promise<boolean> {
  const age = Date.now() - record.at;
  if (age <= staleMs) return false;
  const boot = await bootId();
  if (record.boot && boot && record.boot !== boot) return true; // written before the last reboot
  if (!alive(record.pid)) return true;
  const start = await startTimeOf(record.pid);
  if (record.start && start) return record.start !== start; // the pid belongs to another process now
  return false; // alive, and /proc cannot tell whose: believed alive
}

/** Whether a live pid holds ``record`` without anything to confirm it is the owner (no start time to compare). */
async function unconfirmedLive(record: LockOwner): Promise<boolean> {
  return alive(record.pid) && !(record.start && (await startTimeOf(record.pid)));
}

/** The lock's record; ``"invalid"`` is a file that is not a record (never written by this module). */
async function readOwner(path: string): Promise<LockOwner | "invalid" | "gone"> {
  let text: string;
  try {
    text = await readFile(path, "utf-8");
  } catch {
    return "gone";
  }
  try {
    const parsed = JSON.parse(text) as Partial<LockOwner>;
    if (typeof parsed.pid === "number" && typeof parsed.token === "string" && typeof parsed.at === "number") {
      return {
        pid: parsed.pid, token: parsed.token, at: parsed.at,
        ...(typeof parsed.boot === "string" ? { boot: parsed.boot } : {}),
        ...(typeof parsed.start === "string" ? { start: parsed.start } : {}),
      };
    }
  } catch {
    // not a lock record
  }
  return "invalid";
}

/** Create ``path`` holding ``owner`` atomically and exclusively: write a private file, then ``link`` it into place. */
async function linkInto(path: string, owner: LockOwner): Promise<boolean> {
  const temporary = `${path}.${owner.token}.tmp`;
  await writeFile(temporary, JSON.stringify(owner), { flag: "wx" });
  try {
    await link(temporary, path);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EEXIST") return false;
    throw error;
  } finally {
    await unlink(temporary).catch(() => undefined);
  }
}

const sameRecord = (a: LockOwner | "invalid" | "gone", b: LockOwner | "invalid" | "gone"): boolean =>
  typeof a === "object" && typeof b === "object" ? a.token === b.token : a === b;

export type Recovery = "removed" | "retry" | "aborted";

/**
 * Remove an abandoned lock, never a live one.  ``judged`` is the record the caller saw and found abandoned.
 * "retry": nothing was removed (the lock changed hands, or another recoverer is working); "aborted": the file
 * that was moved aside was not the judged record, so a live lock may have been disturbed and the caller must
 * give up its own flush.  Exported for the tests, which inject ``afterVerify`` to open the race window.
 */
export async function recoverAbandoned(
  path: string,
  judged: LockOwner | "invalid",
  hooks: { afterVerify?: () => Promise<void>; mutexStaleMs?: number } = {}
): Promise<Recovery> {
  const mutexPath = path + ".recover";
  const mutexStaleMs = hooks.mutexStaleMs ?? RECOVERY_MUTEX_STALE_MS;
  const me: LockOwner = { pid: process.pid, token: randomBytes(12).toString("hex"), at: Date.now(), ...(await identity()) };
  let have: boolean;
  try {
    have = await linkInto(mutexPath, me);
  } catch {
    return "retry";
  }
  if (!have) {
    await clearAbandonedMutex(mutexPath, mutexStaleMs);
    return "retry";
  }
  try {
    if (!sameRecord(await readOwner(path), judged)) return "retry"; // it changed hands: not ours to remove
    await hooks.afterVerify?.();
    const aside = `${path}.stale.${me.token}`;
    try {
      await rename(path, aside);
    } catch {
      return "retry";
    }
    const moved = await readOwner(aside);
    await unlink(aside).catch(() => undefined);
    return sameRecord(moved, judged) ? "removed" : "aborted"; // never put a moved lock back
  } finally {
    const mine = await readOwner(mutexPath);
    if (typeof mine === "object" && mine.token === me.token) await unlink(mutexPath).catch(() => undefined);
  }
}

/**
 * Clear a recovery mutex whose recorder is gone (and that is old), by the lock's own rule: move it aside and
 * delete it only if what was moved is the record that was judged.  A mutex held by a live (if slow) recoverer is
 * never cleared.  Something else that was moved by mistake is not put back.
 */
async function clearAbandonedMutex(mutexPath: string, staleMs: number): Promise<void> {
  const judged = await readOwner(mutexPath);
  if (judged === "gone") return;
  if (judged === "invalid") {
    const info = await stat(mutexPath).catch(() => null);
    if (info === null || Date.now() - info.mtimeMs <= Math.max(staleMs, 60_000)) return;
  } else if (!(await ownerGone(judged, staleMs))) {
    return;
  }
  const aside = `${mutexPath}.stale.${randomBytes(6).toString("hex")}`;
  try {
    await rename(mutexPath, aside);
  } catch {
    return;
  }
  await unlink(aside).catch(() => undefined);
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Run ``body`` while holding the lock of ``transcriptPath``.  Returns ``{ ran: false }`` without running it when
 * the lock could not be taken within ``waitMs`` (or a recovery disturbed a lock that was not the abandoned
 * one): the caller leaves its turns for the next flush, which is always safe because the cursor was not
 * touched.  ``body`` gets a handle whose ``held()`` it must check before every POST and before writing the cursor.
 */
export async function withTranscriptLock<T>(
  transcriptPath: string,
  body: (handle: LockHandle) => Promise<T>,
  options: LockOptions = {}
): Promise<{ ran: true; value: T } | { ran: false }> {
  const waitMs = options.waitMs ?? LOCK_WAIT_MS;
  const staleMs = options.staleMs ?? LOCK_STALE_MS;
  const path = transcriptPath + LOCK_SUFFIX;
  const owner: LockOwner = { pid: process.pid, token: randomBytes(12).toString("hex"), at: Date.now(), ...(await identity()) };
  const deadline = Date.now() + waitMs;

  for (;;) {
    let created = false;
    try {
      created = await linkInto(path, owner);
    } catch (error) {
      logError("Could not create the transcript lock; leaving the turns for the next flush", error);
      return { ran: false };
    }
    if (created) break;

    const holder = await readOwner(path);
    if (holder === "gone") continue; // released between the attempt and the read
    let abandoned = false;
    if (holder === "invalid") {
      const info = await stat(path).catch(() => null);
      abandoned = info !== null && Date.now() - info.mtimeMs > staleMs;
    } else {
      abandoned = await ownerGone(holder, staleMs); // a confirmed live owner's lock is never recovered
    }
    if (abandoned) {
      const outcome = await recoverAbandoned(path, holder as LockOwner | "invalid");
      if (outcome === "aborted") {
        logError("The transcript lock changed while it was being recovered; leaving the turns for the next flush");
        return { ran: false };
      }
      if (outcome === "removed") continue;
    }
    if (Date.now() >= deadline) {
      logError(
        typeof holder === "object" && Date.now() - holder.at > staleMs && (await unconfirmedLive(holder))
          ? `The transcript lock is held by process ${holder.pid}, which may still be running and whose identity this ` +
              "platform cannot confirm (no /proc start time); it is not reclaimed. Capture skipped; the turns wait for the next flush"
          : "Another hook is delivering this transcript; leaving the turns for the next flush"
      );
      return { ran: false };
    }
    await sleep(POLL_MS);
  }

  const handle: LockHandle = {
    held: async () => {
      const current = await readOwner(path);
      return typeof current === "object" && current.token === owner.token;
    },
  };
  try {
    return { ran: true, value: await body(handle) };
  } finally {
    if (await handle.held()) await unlink(path).catch(() => undefined);
  }
}
