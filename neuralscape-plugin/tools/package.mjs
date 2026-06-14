#!/usr/bin/env node
/**
 * Packages the Neuralscape plugin into a clean, installable zip under dist/.
 *
 * What it does:
 *   1. Runs the esbuild hook build (scripts/*.js) so the bundle is fresh.
 *   2. Stages ONLY the runtime files an installed plugin needs into
 *      dist/staging/ (no node_modules, no src/tests, no dev config).
 *   3. Zips the staging dir into dist/neuralscape-plugin-<version>.zip, with a
 *      single top-level <plugin-name>/ directory (Cowork's uploader requires
 *      `<plugin-name>/.claude-plugin/plugin.json`, not files at the zip root).
 *
 * Why a staging copy: macOS "Compress" and a bare `zip` of the working tree
 * preserve node_modules/.bin/* symlinks, which Cowork rejects ("Zip file
 * contains a symbolic link"). Excluding node_modules entirely removes the only
 * symlinks, so the artifact installs cleanly on both Claude Code and Cowork.
 *
 * Usage: npm run package   (alias: npm run dist)
 */
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, cpSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const pluginRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const distDir = join(pluginRoot, "dist");

// The plugin manifest name is the required top-level directory inside the zip.
// Cowork's uploader expects `<plugin-name>/.claude-plugin/plugin.json`, NOT the
// files at the archive root — a flat zip extracts fine but then fails manifest
// discovery ("Plugin validation failed").
const pluginName = JSON.parse(
  readFileSync(join(pluginRoot, ".claude-plugin", "plugin.json"), "utf8"),
).name;
const stageRoot = join(distDir, "staging");
const stagingDir = join(stageRoot, pluginName);

// Runtime allowlist: exactly what the installed plugin loads at runtime.
// Anything not listed here is intentionally excluded from the artifact.
const INCLUDE = [
  ".claude-plugin", // plugin.json (manifest)
  ".mcp.json", // MCP connector definition
  "hooks", // hooks.json + openclaw-hooks.json
  "scripts", // built, bundled hook entrypoints (*.js)
  "skills", // all SKILL.md cross-platform skills
  "cowork", // STANDING_CONTEXT.md paste-block fallback
  "README.md",
  "CHANGELOG.md",
  "LICENSE",
];

function run(cmd, args, opts = {}) {
  execFileSync(cmd, args, { stdio: "inherit", cwd: pluginRoot, ...opts });
}

const { version } = JSON.parse(
  readFileSync(join(pluginRoot, "package.json"), "utf8"),
);

console.log(`\n📦 Packaging neuralscape-plugin v${version}\n`);

// 1. Fresh build of the hook bundle.
console.log("→ Building hook bundle (esbuild)...");
run("node", ["esbuild.config.js"]);

// 2. Clean + restage under the single top-level <plugin-name>/ directory.
console.log("→ Staging runtime files...");
rmSync(stageRoot, { recursive: true, force: true });
mkdirSync(stagingDir, { recursive: true });

for (const entry of INCLUDE) {
  const from = join(pluginRoot, entry);
  if (!existsSync(from)) {
    console.warn(`  ! skipped (missing): ${entry}`);
    continue;
  }
  // dereference: false keeps regular files as-is; the allowlist contains no
  // symlinks, so nothing symbolic ever enters the staging tree.
  cpSync(from, join(stagingDir, entry), {
    recursive: true,
    dereference: false,
  });
}

// Drop any stray build noise that may live inside an included dir.
for (const junk of ["scripts/.DS_Store", ".DS_Store"]) {
  rmSync(join(stagingDir, junk), { force: true });
}

// 3. Zip with files at the archive root (matches local-install expectations).
const zipName = `neuralscape-plugin-${version}.zip`;
const zipPath = join(distDir, zipName);
rmSync(zipPath, { force: true });

console.log("→ Creating archive...");
// Zip from the staging ROOT so the archive contains the `<plugin-name>/` dir.
run("zip", ["-rqX", zipPath, pluginName], { cwd: stageRoot });

// Sanity check: fail loudly if any symlink made it into the artifact.
// zipinfo long format begins each line with the unix mode; symlinks start 'l'.
const zipinfo = execFileSync("zipinfo", [zipPath], { encoding: "utf8" });
const entryLines = zipinfo
  .trim()
  .split("\n")
  .filter((l) => /^[-dl]/.test(l));
const symlinks = entryLines.filter((l) => l.startsWith("l"));

if (symlinks.length > 0) {
  console.error("\n❌ Archive contains symlinks (Cowork will reject it):");
  for (const l of symlinks) console.error(`   ${l}`);
  process.exit(1);
}

console.log(`\n✅ ${zipName}`);
console.log(`   ${entryLines.length} entries, 0 symlinks`);
console.log(`   dist/${zipName}\n`);
