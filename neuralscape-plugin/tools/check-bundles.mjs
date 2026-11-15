#!/usr/bin/env node
/**
 * CI guard: every hook bundle the plugin needs must be a tracked file.
 *
 * Marketplace installs are plain git clones (no install-time build), so a hook
 * bundle that exists only after `npm run build` 404s for users. The expected
 * set is derived, never hand-listed:
 *   - every entry point in esbuild.config.js  (src/hooks/<n>.ts -> scripts/<n>.js)
 *   - every scripts/*.js referenced by a hook manifest in hooks/*.json
 *
 * Usage: node tools/check-bundles.mjs [pluginRoot]
 * Exit 0 when all are tracked and present, 1 otherwise. Dependency-free.
 */
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export function expectedBundles(root) {
  const expected = new Map(); // scripts/<n>.js -> where it was required from

  const cfg = readFileSync(join(root, "esbuild.config.js"), "utf8");
  const block = /entryPoints\s*=\s*\[([\s\S]*?)\]/.exec(cfg);
  if (!block) throw new Error("could not find the entryPoints array in esbuild.config.js");
  const entries = [...block[1].matchAll(/["']src\/hooks\/([\w.-]+)\.ts["']/g)];
  if (entries.length === 0) throw new Error("esbuild.config.js entryPoints lists no src/hooks/*.ts files");
  for (const m of entries) expected.set(`scripts/${m[1]}.js`, "esbuild.config.js");

  const hooksDir = join(root, "hooks");
  const manifests = readdirSync(hooksDir).filter((f) => f.endsWith(".json")).sort();
  if (manifests.length === 0) throw new Error("no hook manifests found in hooks/");
  for (const file of manifests) {
    const text = readFileSync(join(hooksDir, file), "utf8");
    JSON.parse(text); // fail loudly on a malformed manifest
    for (const m of text.matchAll(/scripts\/([\w.-]+\.js)/g)) {
      const key = `scripts/${m[1]}`;
      if (!expected.has(key)) expected.set(key, `hooks/${file}`);
    }
  }
  return expected;
}

export function checkBundles(root) {
  const errors = [];
  const tracked = new Set(
    execFileSync("git", ["ls-files", "--", "scripts"], { cwd: root, encoding: "utf8" })
      .split("\n")
      .filter(Boolean),
  );
  for (const [rel, source] of expectedBundles(root)) {
    if (!tracked.has(rel)) errors.push(`${rel} (required by ${source}) is not committed`);
    else if (!existsSync(join(root, rel))) errors.push(`${rel} (required by ${source}) is tracked but missing on disk`);
  }
  return errors;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = resolve(process.argv[2] ?? join(dirname(fileURLToPath(import.meta.url)), ".."));
  const errors = checkBundles(root);
  for (const e of errors) console.error(`::error::${e} -- hooks would 404 on marketplace installs.`);
  if (errors.length) process.exit(1);
  console.log(`bundle guard ok: ${expectedBundles(root).size} bundles tracked`);
}
