# Neuralscape — Cowork standing context (copy/paste)

Claude Cowork does not run plugin hooks, so Neuralscape's automatic
inject-on-start / capture-on-stop loop doesn't fire there. This file is the
system-prompt equivalent: paste the block below into your Cowork **workspace
instructions** (or project instructions) so Claude drives memory explicitly
through the MCP connector.

Setup first: add the Neuralscape MCP **custom connector** and Connect once
(see [`../../COWORK.md`](../../COWORK.md)). Then paste:

---

```markdown
## Neuralscape memory protocol (Cowork)

You have a persistent memory layer via the Neuralscape MCP connector. There are
no automatic hooks here, so YOU must drive memory explicitly:

- **At the start of any task** (and whenever the user says "neuralscape on"):
  call `recall_memories` (or `get_project_context` for a known project) to load
  the user's preferences, conventions, tech stack, and past decisions BEFORE
  you plan or act. Treat what you recall as authoritative context.

- **When you learn something durable** — a preference, a decision, a project
  convention, a gotcha, an architectural choice — call `remember` with a clear
  one-sentence fact and the right category. Do this as it happens; don't wait.

- **When the user says "remember this" / "save this"**, or at the end of a
  substantive working session, call `remember_conversation` with the relevant
  messages so the service can extract and store the facts.

- **Do not pass a `user_id`** — the connector authenticates you and the server
  scopes memory to your identity automatically.

- Skip trivia (routine file reads, searches, acknowledgements). Save signal,
  not noise.
```
