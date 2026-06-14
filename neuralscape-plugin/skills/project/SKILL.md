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

## Identity block (how to resolve `user_id`)

`list_projects` accepts `user_id` but does **not** require it; under token auth the server scopes by the authenticated token identity regardless.

- If `CLAUDE_PLUGIN_OPTION_USER_ID` (or `NEURALSCAPE_USER_ID`) is set → pass that value.
- If neither is set (likely Claude Cowork) → you may omit `user_id`, or pass a placeholder like `"cowork"`; the OAuth token determines the real identity.
- **Never** block or error solely because `user_id` is unknown.

## Notes

- The active project lives only in this conversation's context. There is no cross-session persistence in Claude Cowork (that's the job hooks do, and Cowork runs none) — re-select at the start of a new session if needed.
- To act on the selected project, follow up with `/neuralscape:recall`, `/neuralscape:remember`, or `/neuralscape:save-session`.
