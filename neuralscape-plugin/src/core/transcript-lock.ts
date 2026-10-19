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
 * - **Abandoned locks:** a lock is recovered only when its recorded owner pid is not alive on this host AND the
 *   lock is older than ``staleMs``.  A live owner's lock is never recovered, however old.
 * - **Recovery:** recoverers are serialized by a short-lived recovery mutex (also a ``link``).  Under it the
 *   lock is re-read and, only if it is still the exact abandoned record, renamed to a unique name and deleted.
 *   If what was moved turns out not to be that record, it is NOT put back (putting it back is the race that
 *   lets a third hook acquire the vacant path): the recoverer abandons its own flush instead.
 * - **The owner verifies:** the body receives ``held()``, which is true only while the lock file still carries
 *   the owner's token.  The delivery code calls it before every POST and before it writes the cursor, and
 *   stops cleanly when the lock is gone.
 *
 * Residual window (all of it requires an owner that is declared dead while it is alive, for example because it
 * runs in another pid namespace, and is older than ``staleMs``): a recoverer's rename can move a successor's
 * lock in the gap between its re-read and its rename.  The owner then notices at its next ``held()`` check, so at
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
/** A recovery mutex older than this was left by a recoverer that died; recoverers compete only among themselves. */
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
}

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM"; // exists, owned by someone else
  }
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
      return { pid: parsed.pid, token: parsed.token, at: parsed.at };
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
  hooks: { afterVerify?: () => Promise<void> } = {}
): Promise<Recovery> {
  const mutexPath = path + ".recover";
  const me: LockOwner = { pid: process.pid, token: randomBytes(12).toString("hex"), at: Date.now() };
  let have: boolean;
  try {
    have = await linkInto(mutexPath, me);
  } catch {
    return "retry";
  }
  if (!have) {
    const info = await stat(mutexPath).catch(() => null);
    if (info !== null && Date.now() - info.mtimeMs > RECOVERY_MUTEX_STALE_MS) await unlink(mutexPath).catch(() => undefined);
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
  const owner: LockOwner = { pid: process.pid, token: randomBytes(12).toString("hex"), at: Date.now() };
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
      abandoned = !alive(holder.pid) && Date.now() - holder.at > staleMs; // a live owner's lock is never recovered
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
      logError("Another hook is delivering this transcript; leaving the turns for the next flush");
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
