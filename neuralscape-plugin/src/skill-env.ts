/**
 * Side-effect module: must be the FIRST import of src/hooks/sync.ts.
 *
 * A skill's Bash tool has neither CLAUDE_PLUGIN_ROOT nor CLAUDE_PLUGIN_OPTION_*
 * in its environment (they are exported to hook processes only). utils.ts reads
 * its configuration when it is first imported, so the environment has to be
 * prepared before that import runs, and ES modules evaluate imports in order.
 *
 *  1. CLAUDE_PLUGIN_ROOT is derived from this bundle's own location
 *     (<root>/scripts/sync.js) when it is unset, so the baked `.mcp.json` URL is
 *     found.
 *  2. `--opt KEY=VALUE` arguments (which a skill fills with the documented
 *     `${user_config.KEY}` substitution) become CLAUDE_PLUGIN_OPTION_<KEY>, unless
 *     already set in the environment. Empty values and values that still contain
 *     a `${` template (the substitution did not happen) are ignored. Sensitive
 *     options (API_KEY) are never substituted into skill text, so they cannot be
 *     passed this way; the helper reads the key only from the environment.
 *     The `--opt` arguments are removed from process.argv.
 */

import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

if (!process.env.CLAUDE_PLUGIN_ROOT?.trim()) {
  const root = join(dirname(fileURLToPath(import.meta.url)), "..");
  if (existsSync(join(root, ".claude-plugin", "plugin.json")) || existsSync(join(root, ".mcp.json"))) {
    process.env.CLAUDE_PLUGIN_ROOT = root;
  }
}

const OPT_KEY = /^[A-Z][A-Z0-9_]*$/;
const rest: string[] = [];
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "--opt" && i + 1 < argv.length) {
    const pair = argv[++i];
    const eq = pair.indexOf("=");
    const key = eq > 0 ? pair.slice(0, eq) : "";
    const value = eq > 0 ? pair.slice(eq + 1).trim() : "";
    const name = `CLAUDE_PLUGIN_OPTION_${key}`;
    if (OPT_KEY.test(key) && key !== "API_KEY" && value && !value.includes("${") && !process.env[name]?.trim()) {
      process.env[name] = value;
    }
  } else rest.push(argv[i]);
}
process.argv = [...process.argv.slice(0, 2), ...rest];
