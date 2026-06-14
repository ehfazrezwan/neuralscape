# Neuralscape — Cowork standing context (copy/paste)

Claude Cowork does not run plugin hooks, so Neuralscape's automatic
inject-on-start / capture-on-stop loop doesn't fire there. This file is the
system-prompt equivalent: paste the block below into your Cowork **workspace
instructions** (or project instructions) so Claude drives memory explicitly
through the MCP connector.

> **If you've installed the Neuralscape plugin in Cowork**, you can instead
> just invoke the cross-platform skills directly — they do exactly what this
> block instructs, on demand:
>
> | Protocol bullet | Equivalent skill |
> |---|---|
> | Recall context at task start | `/neuralscape:recall` |
> | Save a durable fact as you learn it | `/neuralscape:remember` |
> | Save the conversation at session end | `/neuralscape:save-session` |
> | Pick which project to scope to | `/neuralscape:project` |
>
> This paste-block remains the supported path for workspaces **without** the
> plugin installed (skills aren't available there, but the MCP connector is).

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

- **Identity is from the connector token, not `user_id`.** The server scopes
  memory to whoever authenticated the connector and ignores any `user_id` you
  pass. Some tools (`recall_memories`, `remember`, `remember_conversation`)
  mark `user_id` as required — when one does, pass a harmless placeholder like
  `"cowork"` to satisfy the schema; the token still determines your real
  identity. Never block on not knowing the `user_id`.

- Skip trivia (routine file reads, searches, acknowledgements). Save signal,
  not noise.
```
