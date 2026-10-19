/**
 * A per-transcript lock shared by every process that delivers turns from the same transcript.
 *
 * Two hooks can run at the same time (a Stop hook still running when the next one starts), and each reads the
 * cursor, posts the turns after it and writes the new cursor.  Without a lock both read the same cursor and
 * post the same turns, and the one that finishes last decides where the cursor ends up.  The lock makes the
 * whole read-deliver-commit sequence of one process finish before the next one reads the cursor.
 *
 * The lock is a file next to the offset file, created with ``O_EXCL`` and holding the owner's pid, a token and
 * the time it was taken.  A lock whose owner process is gone, or that is older than ``staleMs``, is taken over.
 * The owner removes it in a ``finally``, and only while the file still carries its own token.
 */

import { link, open, readFile, rename, stat, unlink } from "node:fs/promises";
import { randomBytes } from "node:crypto";
import { logError } from "../utils.js";

const LOCK_SUFFIX = ".neuralscape-lock";

/** How long to wait for another process to finish, and how old a lock must be to count as abandoned. */
export const LOCK_WAIT_MS = 20_000;
export const LOCK_STALE_MS = 90_000;
const POLL_MS = 50;
/** A lock file that is still empty (its owner has not written it yet) is not treated as abandoned before this. */
const EMPTY_GRACE_MS = 2_000;

export interface LockOptions {
  waitMs?: number;
  staleMs?: number;
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

async function readOwner(path: string): Promise<LockOwner | "empty" | "gone"> {
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
    // not (yet) a complete lock record
  }
  return "empty";
}

/** Try to create the lock file; the owner record is written before anyone can rely on it. */
async function tryCreate(path: string, owner: LockOwner): Promise<boolean> {
  let handle;
  try {
    handle = await open(path, "wx");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EEXIST") return false;
    throw error;
  }
  try {
    await handle.writeFile(JSON.stringify(owner));
  } finally {
    await handle.close();
  }
  return true;
}

/**
 * Remove an abandoned lock without ever removing a fresh one: the file is moved aside first, and if what was
 * moved is not the record that was judged abandoned (someone took the lock in between) it is put back.
 */
async function takeOver(path: string, judged: LockOwner | "empty"): Promise<void> {
  const aside = `${path}.stale.${randomBytes(6).toString("hex")}`;
  try {
    await rename(path, aside);
  } catch {
    return; // someone else already took it over
  }
  const moved = await readOwner(aside);
  const same =
    judged === "empty"
      ? moved === "empty"
      : typeof moved === "object" && moved.token === judged.token;
  if (!same) {
    // This was a fresh lock: give it back (link fails if yet another process created one meanwhile).
    await link(aside, path).catch(() => undefined);
  }
  await unlink(aside).catch(() => undefined);
}

async function release(path: string, token: string): Promise<void> {
  const owner = await readOwner(path);
  if (typeof owner === "object" && owner.token === token) {
    await unlink(path).catch(() => undefined);
  }
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Run ``body`` while holding the lock of ``transcriptPath``.  Returns ``{ ran: false }`` without running it when
 * the lock could not be taken within ``waitMs``: the caller leaves its turns for the next flush, which is
 * always safe because the cursor was not touched.
 */
export async function withTranscriptLock<T>(
  transcriptPath: string,
  body: () => Promise<T>,
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
      created = await tryCreate(path, owner);
    } catch (error) {
      logError("Could not create the transcript lock; leaving the turns for the next flush", error);
      return { ran: false };
    }
    if (created) break;

    const holder = await readOwner(path);
    if (holder === "gone") continue; // released between the attempt and the read
    if (holder === "empty") {
      const info = await stat(path).catch(() => null);
      if (info !== null && Date.now() - info.mtimeMs > EMPTY_GRACE_MS) await takeOver(path, "empty");
    } else if (!alive(holder.pid) || Date.now() - holder.at > staleMs) {
      await takeOver(path, holder);
      continue;
    }
    if (Date.now() >= deadline) {
      logError("Another hook is delivering this transcript; leaving the turns for the next flush");
      return { ran: false };
    }
    await sleep(POLL_MS);
  }

  try {
    return { ran: true, value: await body() };
  } finally {
    await release(path, owner.token);
  }
}
