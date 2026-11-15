#!/usr/bin/env node
/**
 * Bakes a distribution channel's configuration into the committed plugin files.
 *
 * Neuralscape is self-hosted: every deployment has its own base URL, and Cowork
 * bundles the MCP connector from `.mcp.json` read-only (it cannot interpolate
 * `${user_config.*}`). So the connector URL is a literal committed per channel.
 * A fork / vendor / self-hoster runs this once to stamp their URL (and, if they
 * publish their own marketplace, its name/owner) instead of hand-editing JSON.
 *
 * Because the hooks derive their API base from this same `.mcp.json`
 * (readBakedUrl in src/utils.ts), the URL has exactly one source of truth — this
 * script only needs to touch `.mcp.json`.
 *
 * Usage:
 *   node tools/bake.mjs --url https://neuralscape.dev
 *   node tools/bake.mjs --url https://memory.acme.internal/mcp/ \
 *     --marketplace-name acme-plugins --owner "Acme" --owner-email ops@acme.com
 *   node tools/bake.mjs --url https://… --dry-run
 *   node tools/bake.mjs --check                      # report the baked URL
 *   node tools/bake.mjs --check --expect https://…   # exit 1 if it differs
 *
 * Options:
 *   --url <url>              Required. Service base or full /mcp/ URL. Must be
 *                           https (loopback http allowed for local testing).
 *                           Normalized to end in exactly one `/mcp/`.
 *   --marketplace-name <s>  Optional. Set marketplace.json top-level `name`.
 *   --owner <s>             Optional. Set marketplace.json owner.name.
 *   --owner-email <s>       Optional. Set marketplace.json owner.email.
 *   --dry-run               Print what would change; write nothing.
 *   --check                 Report which connector URL is baked into .mcp.json
 *                           (and the API base the hooks derive from it); write
 *                           nothing. A fork should run this before publishing: the
 *                           URL shipped in this repo is the upstream channel's.
 *                           Exits 0 unless --expect is given and differs.
 *   --expect <url>          With --check: exit 1 when the baked URL is not this one
 *                           (same normalization as --url). For a fork's own CI.
 *   --help                  Show this help.
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const pluginRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(pluginRoot, "..");
const mcpPath = join(pluginRoot, ".mcp.json");
const marketplacePath = join(repoRoot, ".claude-plugin", "marketplace.json");

function parseArgs(argv) {
  const opts = { dryRun: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    // Consume the next token as this flag's value, failing fast if it's missing
    // or is itself another flag (e.g. `--owner --dry-run` or a trailing `--owner`).
    const takeValue = () => {
      const v = argv[i + 1];
      if (v === undefined || v.startsWith("--")) fail(`${a} requires a value`);
      i++;
      return v;
    };
    switch (a) {
      case "--help":
      case "-h":
        opts.help = true;
        break;
      case "--dry-run":
        opts.dryRun = true;
        break;
      case "--check":
        opts.check = true;
        break;
      case "--expect":
        opts.expect = takeValue();
        break;
      case "--url":
        opts.url = takeValue();
        break;
      case "--marketplace-name":
        opts.marketplaceName = takeValue();
        break;
      case "--owner":
        opts.owner = takeValue();
        break;
      case "--owner-email":
        opts.ownerEmail = takeValue();
        break;
      default:
        fail(`Unknown argument: ${a}  (try --help)`);
    }
  }
  return opts;
}

function fail(msg) {
  console.error(`\n❌ ${msg}\n`);
  process.exit(1);
}

// Render --help from this file's own JSDoc block: keep ` *` body lines, drop the
// ` */` terminator, strip the leading ` * ` marker.
const HELP = readFileSync(fileURLToPath(import.meta.url), "utf8")
  .split("\n")
  .filter((l) => l.startsWith(" *") && !l.startsWith(" */"))
  .map((l) => l.replace(/^ \* ?/, ""))
  .join("\n");

// Accept a base URL or a full /mcp/ URL; return the canonical connector URL that
// ends in exactly one `/mcp/`. Rejects templates, non-http(s), and bad input.
function normalizeMcpUrl(raw) {
  if (!raw) fail("--url is required. Example: --url https://neuralscape.dev");
  if (raw.includes("${")) fail(`--url must be a literal URL, not a template: ${raw}`);
  let parsed;
  try {
    parsed = new URL(raw);
  } catch {
    fail(`--url is not a valid URL: ${raw}`);
  }
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
    fail(`--url must be http(s): ${raw}`);
  }
  const isLoopback = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(parsed.hostname);
  if (parsed.protocol === "http:" && !isLoopback) {
    fail(`--url must use https (http is only allowed for loopback hosts): ${raw}`);
  }
  if (parsed.search || parsed.hash) {
    fail(`--url must not contain a query string or fragment: ${raw}`);
  }
  // Strip a trailing /mcp or /mcp/ and any trailing slash, then re-append /mcp/.
  const path = parsed.pathname.replace(/\/mcp\/?$/, "").replace(/\/$/, "");
  return `${parsed.origin}${path}/mcp/`;
}

function readJson(path) {
  if (!existsSync(path)) fail(`File not found: ${path}`);
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (e) {
    fail(`Could not parse ${path}: ${e.message}`);
  }
}

function writeJson(path, obj) {
  writeFileSync(path, JSON.stringify(obj, null, 2) + "\n");
}

// Report what is baked. Writes nothing; exits 0 unless --expect differs, so it is
// safe as a report step in any CI (including the upstream repo's own).
function runCheck(opts) {
  const mcp = readJson(mcpPath);
  const raw = mcp?.mcpServers?.neuralscape?.url;
  console.log("\n🔎 Baked distribution channel\n");
  if (typeof raw !== "string" || raw === "") {
    console.log("   connector URL: (none)\n   The hooks fall back to http://localhost:8199.\n");
    process.exit(opts.expect ? 1 : 0);
  }
  console.log(`   connector URL: ${raw}`);
  if (raw.includes("${")) {
    console.log("   This is an un-baked template; the hooks ignore it and use http://localhost:8199.\n");
    process.exit(opts.expect ? 1 : 0);
  }
  let parsed;
  try {
    parsed = new URL(raw);
  } catch {
    console.log("   This is not a valid URL; the hooks ignore it and use http://localhost:8199.\n");
    process.exit(opts.expect ? 1 : 0);
  }
  const base = parsed.pathname.replace(/\/mcp\/?$/, "").replace(/\/$/, "");
  console.log(`   hooks API base: ${parsed.origin}${base}   (overridden by the URL userConfig / NEURALSCAPE_URL)`);
  console.log(
    "   Self-hosting? This is the URL every install of your fork talks to by default.\n" +
      "   Point it at your own service with:  npm run bake -- --url https://your-host\n",
  );
  if (opts.expect) {
    const want = normalizeMcpUrl(opts.expect);
    if (normalizeMcpUrl(raw) !== want) {
      console.error(`❌ Baked URL differs from --expect (${want}).\n`);
      process.exit(1);
    }
    console.log("✅ Baked URL matches --expect.\n");
  }
  process.exit(0);
}

const opts = parseArgs(process.argv.slice(2));
if (opts.help) {
  console.log(HELP);
  process.exit(0);
}

if (opts.check) {
  runCheck(opts);
}
if (opts.expect) fail("--expect only works with --check");

const url = normalizeMcpUrl(opts.url);
const changes = [];

// 1. Stamp the connector URL into .mcp.json (the single source of truth).
const mcp = readJson(mcpPath);
const server = mcp?.mcpServers?.neuralscape;
if (!server || typeof server !== "object") {
  fail(`${mcpPath} has no mcpServers.neuralscape entry to bake.`);
}
if (server.url !== url) {
  changes.push(`.mcp.json: url  ${server.url ?? "(unset)"}  →  ${url}`);
  server.url = url;
}

// 2. Optionally stamp marketplace identity (only if any flag is given).
const touchMarketplace =
  opts.marketplaceName || opts.owner || opts.ownerEmail;
let marketplace;
if (touchMarketplace) {
  marketplace = readJson(marketplacePath);
  marketplace.owner = marketplace.owner || {};
  if (opts.marketplaceName && marketplace.name !== opts.marketplaceName) {
    changes.push(`marketplace.json: name  ${marketplace.name}  →  ${opts.marketplaceName}`);
    marketplace.name = opts.marketplaceName;
  }
  if (opts.owner && marketplace.owner.name !== opts.owner) {
    changes.push(`marketplace.json: owner.name  ${marketplace.owner.name ?? "(unset)"}  →  ${opts.owner}`);
    marketplace.owner.name = opts.owner;
  }
  if (opts.ownerEmail && marketplace.owner.email !== opts.ownerEmail) {
    changes.push(`marketplace.json: owner.email  ${marketplace.owner.email ?? "(unset)"}  →  ${opts.ownerEmail}`);
    marketplace.owner.email = opts.ownerEmail;
  }
}

console.log(`\n🔧 Bake distribution channel\n   connector URL → ${url}\n`);
if (changes.length === 0) {
  console.log("✅ Already baked — nothing to change.\n");
  process.exit(0);
}
for (const c of changes) console.log(`   ${c}`);

if (opts.dryRun) {
  console.log("\n(dry run — no files written)\n");
  process.exit(0);
}

writeJson(mcpPath, mcp);
if (marketplace) writeJson(marketplacePath, marketplace);
console.log(`\n✅ Baked ${changes.length} change(s). Commit the result to your channel's repo.\n`);
