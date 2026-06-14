---
name: project
description: List the user's Neuralscape projects and pick (or create) the one to scope memory to for this session. Use when the user says "switch project", "what projects do I have", "set project to X", or when a recall/remember needs a project and none is active. Especially useful in Claude Cowork, which has no working directory to infer a project from. MCP-driven — works in both platforms.
---

# Neuralscape — Project

Choose the project that `recall`, `remember`, and `save-session` should scope to. In Claude Code the working directory usually determines this automatically; in Claude Cowork there's no working directory, so this skill is how you select a project.

**Projects are implicit in Neuralscape** — a `project_id` is just a scoping label. There is no separate "project" entity to create or delete: a project comes into existence the moment you `remember` the first fact under its `project_id`, and is removed by deleting its memories. So picking a brand-new name is a valid way to start a new project.

## What to do

1. **Resolve `user_id`** — see the Identity block below.
2. **List existing projects** — call the MCP `list_projects(user_id=<resolved>)` tool. It returns a sorted list of the `project_id`s the user already has memories under.
3. **Present the choices** to the user:
   - The existing projects from step 2.
   - A **(global)** option — no project scope; memory operations hit global scope only.
   - "…or type a new name" — to start a new project. Remind them a new project is created implicitly on the first `remember`; nothing else is needed.
4. **Near-duplicate guard (do this before accepting a *typed* new name).** If the user types a name rather than picking one from the list, normalize both the typed name and every existing project from step 2 — lowercase and strip everything that isn't a letter or digit (so `Neuralscape`, `neural-scape`, `neural_scape`, and `neural scape` all reduce to `neuralscape`). If the normalized typed name matches an existing project's normalized form but the **raw spelling differs**, don't silently create a variant — ask:

   > You typed `Neural-Scape`, but `neuralscape` already exists. Use the existing one, or create `Neural-Scape` as a separate project?

   Default to the existing canonical spelling unless the user confirms they really want a new, distinct project. (A new name with no near-match needs no confirmation — just proceed.)
5. **Record the selection as the active project for this session.** State it back clearly (e.g. "Active project: `neuralscape`. I'll scope recall/remember/save to it until you switch."). For the rest of the conversation, pass this `project_id` to `recall`, `remember`, and `save-session`.
6. If the user picked **(global)**, treat the active project as unset (omit `project_id` on subsequent calls).
7. **Offer to persist the selection to disk** (when a writable working directory is available — always in Claude Code; in Cowork only if the session has the repo mounted). Pinning the choice to a `.neuralscape-project` marker makes it stick across future sessions and lets the SessionStart hook (and every subdirectory) pick it up automatically — turning a one-session choice into a durable, repo-wide default. Offer it:

   > Want me to pin this by writing `.neuralscape-project` at the repo root? Future sessions will use `<id>` automatically.

   If the user accepts:
   - **Find the repo root** — run `git rev-parse --show-toplevel`; fall back to the current working directory if it isn't a git repo.
   - **Write at the repo root, never a subdirectory.** A marker in a subfolder would pin only that subfolder and reintroduce the inconsistency this avoids. Write a single line containing the chosen `project_id` to `<repo-root>/.neuralscape-project` (use the `Write` tool).
   - **Suggest committing it** so teammates and every subdirectory share the same id.
   - **Skip the offer** when the selection is **(global)** (no id to pin), or when a `.neuralscape-project` with the same id already exists at the repo root.

   If there's no writable working directory (e.g. a Cowork session without the repo mounted), you can't persist to disk — keep the selection in conversation context for this session and suggest the user commit a one-line `.neuralscape-project` to their repo through their normal workflow if they want it to stick.

## Identity block (how to resolve `user_id`)

`list_projects` accepts `user_id` but does **not** require it; under token auth the server scopes by the authenticated token identity regardless.

- If `CLAUDE_PLUGIN_OPTION_USER_ID` (or `NEURALSCAPE_USER_ID`) is set → pass that value.
- If neither is set (likely Claude Cowork) → you may omit `user_id`, or pass a placeholder like `"cowork"`; the OAuth token determines the real identity.
- **Never** block or error solely because `user_id` is unknown.

## Notes

- If you persisted the selection (step 7), it's durable: a committed `.neuralscape-project` pins the id for every future session and subdirectory. If you didn't (or couldn't), the active project lives only in this conversation's context — re-select at the start of a new session. In Cowork without the repo mounted, in-context selection is the only option.
- To act on the selected project, follow up with `/neuralscape:recall`, `/neuralscape:remember`, or `/neuralscape:save-session`.
