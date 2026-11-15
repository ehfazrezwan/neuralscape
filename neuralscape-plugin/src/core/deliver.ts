/**
 * Deliver the not-yet-delivered turns of a Claude Code transcript.
 *
 * Shared by the Stop hook (after every assistant turn) and the SessionEnd
 * hook (once, as the last chance to deliver a final turn whose transcript
 * entry had not landed when its Stop ran). Reads the cursor, delivers and
 * writes the cursor as one step under the per-transcript lock.
 */

import { commitClaudeCodeFlush, extractClaudeCodeTurns } from "../adapters/claude-code.js";
import { flushTurns } from "./flush.js";
import { type LockHandle, withTranscriptLock } from "./transcript-lock.js";

/** Number of turns POSTed successfully (0 when nothing was pending or the lock was not obtained). */
export async function deliverClaudeCodeTurns(raw: Record<string, unknown>): Promise<number> {
  let delivered = 0;
  const deliver = async (handle?: LockHandle): Promise<void> => {
    const turns = await extractClaudeCodeTurns(raw);
    const result = await flushTurns(turns, { proceed: handle?.held });
    delivered = result.flushed;
    await commitClaudeCodeFlush(raw, result, handle?.held);
  };
  const transcriptPath = raw.transcript_path as string | undefined;
  if (transcriptPath) await withTranscriptLock(transcriptPath, deliver);
  else await deliver();
  return delivered;
}
