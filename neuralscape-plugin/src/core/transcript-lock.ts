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
 *   might be alive is never reclaimed.  Where ``/proc`` cannot compare start times (macOS, or a record without a
 *   start time) a live pid whose process ``ps`` shows started after the record was written is not the owner
 *   (the pid was reused, or the record predates a reboot), so a stale lock cannot wedge capture behind an
 *   unrelated process; a live pid that may be the owner is waited for, then the waiter skips its flush with a
 *   log line that says so and the turns wait for the next flush.
 * - **Recovery:** recoverers of one abandoned record are serialized by a recovery mutex that is named after the
 *   record being recovered and a generation number (``<lock>.recover.<hash of the record's token>.<n>``).  A mutex
 *   file is never removed by anyone but its own recoverer: a recoverer that finds the generation taken by a dead
 *   recoverer moves on to the next generation instead of clearing it, so a mutex can never be taken out from under
 *   a live recoverer (the lost-mutex race of earlier versions).  The recoverer re-reads the lock under the mutex
 *   and, only if it is still the exact abandoned record, checks once more that it still holds its mutex, then
 *   renames the lock to a unique name and deletes it.  If what was moved turns out not to be that record, it is NOT
 *   put back (putting it back is the race that lets a third hook acquire the vacant path): the recoverer abandons
 *   its own flush instead.  Once the record is gone it can never come back (tokens are unique), so the dead
 *   generations are deleted then, and a late contender that judged the same record finds the lock changed.
 * - **The owner verifies:** the body receives ``held()``, which is true only while the lock file still carries
 *   the owner's token.  The delivery code calls it before every POST and before it writes the cursor, and
 *   stops cleanly when the lock is gone.
 *
 * Residual window (it requires an owner or recoverer that is declared dead while it is alive, for example
 * because it runs in another pid namespace, and a lock older than ``staleMs``): a recoverer's rename can move a
 * successor's lock in the gap between its re-read and its rename.  With owners and recoverers judged correctly the
 * window does not exist: recoverers of one record exclude each other and nobody else removes the lock.  The owner then notices at its next ``held()`` check, so at
 * most the one POST already past that check can overlap another hook (one turn delivered twice), and the cursor
 * is only ever written forward (see ``commitClaudeCodeFlush``).  Hooks of different Claude Code instances that
 * run in different containers but share a transcript file cannot see each other's pids.
 */

import { execFile } from "node:child_process";
import { realpathSync } from "node:fs";
import { link, readFile, rename, stat, unlink, writeFile } from "node:fs/promises";
import { createHash, randomBytes } from "node:crypto";
import { resolve } from "node:path";
import { logError } from "../utils.js";

const LOCK_SUFFIX = ".neuralscape-lock";

/** How long to wait for another process to finish, and how old a lock must be to count as abandoned. */
export const LOCK_WAIT_MS = 20_000;
export const LOCK_STALE_MS = 90_000;
const POLL_MS = 50;
/** A recovery mutex generation is passed over (never deleted) only when it is older than this and its recorder dead. */
const RECOVERY_MUTEX_STALE_MS = 10_000;
/** How many dead recoverers one record's mutex chain tolerates before the waiter gives up for this flush. */
const MAX_MUTEX_GENERATIONS = 16;
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
    // only ESRCH proves the process is gone.  EPERM means it exists (owned by someone else) and any other
    // failure (an OS resource error, an unsupported probe) proves nothing, so the owner is believed alive
    return (error as NodeJS.ErrnoException).code !== "ESRCH";
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

const PS_MARGIN_MS = 2_000;

/** Seams for the tests: how the elapsed time of a process is asked for (the default runs ``ps``). */
export const probeHooks = {
  etime: (pid: number): Promise<string | null> =>
    new Promise((resolve) =>
      execFile("ps", ["-o", "etime=", "-p", String(pid)], { timeout: 2000 }, (error, stdout) => resolve(error ? null : stdout))
    ),
};

/** ``ps`` verdicts of this process, kept for the length of a wait: a record never changes, so ask once, not per poll. */
const PS_VERDICT_TTL_MS = 60_000;
const psVerdicts = new Map<string, { value: boolean | null; until: number }>();

/**
 * Whether a live ``pid`` provably started after ``at`` (so it cannot have written a record at ``at``), from the
 * elapsed time ``ps -o etime=`` reports ("[[dd-]hh:]mm:ss", the same on macOS and Linux).  This is how a reused
 * pid, or a record from before a reboot, is told apart where there is no ``/proc`` (macOS).  Null when ``ps``
 * cannot say; the elapsed time is whole seconds, so the comparison allows a margin before calling the pid reused.
 * The verdict is cached per (pid, record time) so a waiter polling every 50 ms spawns ``ps`` once, not per poll.
 */
async function startedAfter(pid: number, at: number): Promise<boolean | null> {
  const key = `${pid}:${at}`;
  const cached = psVerdicts.get(key);
  if (cached && cached.until > Date.now()) return cached.value;
  const text = await probeHooks.etime(pid);
  const m = text?.trim().match(/^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$/);
  let value: boolean | null = null;
  if (m) {
    const elapsed = ((Number(m[1] ?? 0) * 24 + Number(m[2] ?? 0)) * 60 + Number(m[3])) * 60 + Number(m[4]);
    value = Date.now() - (elapsed + 1) * 1000 - PS_MARGIN_MS > at;
  }
  if (psVerdicts.size > 256) psVerdicts.clear();
  psVerdicts.set(key, { value, until: Date.now() + PS_VERDICT_TTL_MS });
  return value;
}

/** This process's identity as recorded in the records it writes. */
async function identity(): Promise<Pick<LockOwner, "boot" | "start">> {
  const [boot, start] = await Promise.all([bootId(), startTimeOf(process.pid)]);
  return { ...(boot ? { boot } : {}), ...(start ? { start } : {}) };
}

/**
 * Whether the process that wrote ``record`` is gone and the record is older than ``staleMs``.  Another boot, a
 * dead pid, or a pid whose start time differs from the recorded one (reuse) is dead; a pid that is alive and
 * confirmed by its start time never is.  Where ``/proc`` cannot compare start times (macOS, or no recorded start
 * time), a live pid that ``ps`` shows started after the record was written is dead (reused); any other live pid
 * is believed alive, however old the record: there is no age fallback.
 */
async function ownerGone(record: LockOwner, staleMs: number): Promise<boolean> {
  const age = Date.now() - record.at;
  if (age <= staleMs) return false;
  const boot = await bootId();
  if (record.boot && boot && record.boot !== boot) return true; // written before the last reboot
  if (!alive(record.pid)) return true;
  const start = await startTimeOf(record.pid);
  if (record.start && start) return record.start !== start; // the pid belongs to another process now
  // no /proc to compare (macOS, or a record without a start time): a pid that started after the record was
  // written is not its owner (reused, or the record predates a reboot).  Otherwise it is believed alive.
  return (await startedAfter(record.pid, record.at)) === true;
}

/** Whether a live pid holds ``record`` without ``/proc`` to confirm it is the owner (no start time to compare). */
async function unconfirmedLive(record: LockOwner): Promise<boolean> {
  return alive(record.pid) && !(record.start && (await startTimeOf(record.pid)));
}

type Read = LockOwner | "invalid" | "gone" | "unreadable";

/**
 * The lock's record.  ``"invalid"`` is a file that is not a record (never written by this module); ``"gone"`` is a
 * path that does not exist (ENOENT, and only that); ``"unreadable"`` is something there that cannot be read (EACCES,
 * a directory, ...): it proves nothing about the owner, so it is never reclaimed and never treated as absent.
 */
async function readOwner(path: string): Promise<Read> {
  let text: string;
  try {
    text = await readFile(path, "utf-8");
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "ENOENT" ? "gone" : "unreadable";
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

const sameRecord = (a: Read, b: Read): boolean => (typeof a === "object" && typeof b === "object" ? a.token === b.token : a === b);

/** The recovery mutex of ``judged``, generation ``generation``.  Named after the record so recoveries of different records never share one. */
export function recoveryMutexPath(path: string, judged: LockOwner | "invalid", generation = 0): string {
  const key = judged === "invalid" ? "invalid" : createHash("sha256").update(judged.token).digest("hex").slice(0, 16);
  return `${path}.recover.${key}.${generation}`;
}

export type Recovery = "removed" | "retry" | "aborted";

interface RecoveryHooks {
  /** Called under the mutex after the lock was re-read and before the check that the mutex is still held. */
  afterVerify?: () => Promise<void>;
  /** Called when a mutex generation was judged abandoned, before the next generation is tried. */
  afterMutexJudged?: () => Promise<void>;
  mutexStaleMs?: number;
}

/**
 * Take the first free generation of ``judged``'s recovery mutex.  A generation held by a recoverer that is gone
 * (and old) is passed over, never deleted: only a mutex's own recoverer removes it, so a live recoverer's mutex
 * cannot be lost to a cleaner.  Null when a live recoverer holds one, or the chain is exhausted or unreadable.
 */
async function takeMutex(
  path: string, judged: LockOwner | "invalid", me: LockOwner, staleMs: number, hooks: RecoveryHooks
): Promise<number | null> {
  for (let generation = 0, flaps = 0; generation < MAX_MUTEX_GENERATIONS; ) {
    const mutexPath = recoveryMutexPath(path, judged, generation);
    try {
      if (await linkInto(mutexPath, me)) return generation;
    } catch {
      return null;
    }
    const holder = await readOwner(mutexPath);
    if (holder === "gone") {
      if (++flaps > MAX_MUTEX_GENERATIONS) return null; // released and retaken over and over: give up for now
      continue;
    }
    if (holder === "unreadable") return null;
    let gone: boolean;
    if (holder === "invalid") {
      const info = await stat(mutexPath).catch(() => null);
      gone = info !== null && Date.now() - info.mtimeMs > Math.max(staleMs, 60_000);
    } else {
      gone = await ownerGone(holder, staleMs);
    }
    if (!gone) return null; // a live (if slow) recoverer has it
    await hooks.afterMutexJudged?.();
    generation++;
  }
  return null;
}

/**
 * Remove an abandoned lock, never a live one.  ``judged`` is the record the caller saw and found abandoned.
 * "retry": nothing was removed (the lock changed hands, or another recoverer is working); "aborted": the file
 * that was moved aside was not the judged record, so a live lock may have been disturbed and the caller must
 * give up its own flush.  Exported for the tests, which inject hooks to open the race windows.
 */
export async function recoverAbandoned(path: string, judged: LockOwner | "invalid", hooks: RecoveryHooks = {}): Promise<Recovery> {
  const me: LockOwner = { pid: process.pid, token: randomBytes(12).toString("hex"), at: Date.now(), ...(await identity()) };
  const generation = await takeMutex(path, judged, me, hooks.mutexStaleMs ?? RECOVERY_MUTEX_STALE_MS, hooks);
  if (generation === null) return "retry";
  const mutexPath = recoveryMutexPath(path, judged, generation);
  // once the judged record is gone it cannot return (tokens are unique), so the dead generations below ours are
  // garbage; until then they stay, so no later contender can take a lower generation while we work
  let settled = false;
  try {
    if (!sameRecord(await readOwner(path), judged)) {
      settled = true;
      return "retry"; // it changed hands: not ours to remove
    }
    await hooks.afterVerify?.();
    const mine = await readOwner(mutexPath);
    if (typeof mine !== "object" || mine.token !== me.token) return "retry"; // the mutex was lost: it authorizes nothing
    const aside = `${path}.stale.${me.token}`;
    try {
      await rename(path, aside);
    } catch (error) {
      settled = (error as NodeJS.ErrnoException).code === "ENOENT";
      return "retry";
    }
    settled = true;
    const moved = await readOwner(aside);
    await unlink(aside).catch(() => undefined);
    return sameRecord(moved, judged) ? "removed" : "aborted"; // never put a moved lock back
  } finally {
    const mine = await readOwner(mutexPath);
    if (typeof mine === "object" && mine.token === me.token) await unlink(mutexPath).catch(() => undefined);
    if (settled) {
      for (let g = 0; g < generation; g++) await unlink(recoveryMutexPath(path, judged, g)).catch(() => undefined);
    }
  }
}

/**
 * One transcript, one lock file: the same file reached through a relative path, ``..`` segments or a symlink gets the
 * same lock (the operating system's canonical spelling).  A transcript that does not exist yet uses its absolute path.
 */
export function canonicalTranscriptPath(transcriptPath: string): string {
  try {
    return (realpathSync.native ?? realpathSync)(transcriptPath);
  } catch {
    return resolve(transcriptPath);
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
  const path = canonicalTranscriptPath(transcriptPath) + LOCK_SUFFIX;
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
    if (holder === "unreadable") {
      // something is there that cannot be read (permissions, a directory): that says nothing about its owner, so
      // it is neither waited on (it will not go away) nor reclaimed
      logError("The transcript lock exists but cannot be read; it is not reclaimed. Capture skipped; the turns wait for the next flush");
      return { ran: false };
    }
    if (holder === "gone") {
      // released between the attempt and the read
      if (Date.now() >= deadline) {
        logError("Another hook is delivering this transcript; leaving the turns for the next flush");
        return { ran: false };
      }
      continue;
    }
    let abandoned = false;
    if (holder === "invalid") {
      const info = await stat(path).catch(() => null);
      abandoned = info !== null && Date.now() - info.mtimeMs > staleMs;
    } else {
      abandoned = await ownerGone(holder, staleMs); // a confirmed live owner's lock is never recovered
    }
    if (abandoned) {
      const outcome = await recoverAbandoned(path, holder);
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
