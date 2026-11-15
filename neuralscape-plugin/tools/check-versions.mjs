#!/usr/bin/env node
/**
 * CI guard: the plugin version must be identical everywhere it is declared.
 *   .claude-plugin/plugin.json, package.json, package-lock.json (top level and
 *   packages[""]), and the plugin's entry in the repo-root marketplace.json.
 * A stale marketplace entry ships users an old version string (the 2.7.1 drift).
 *
 * Usage: node tools/check-versions.mjs [pluginRoot]
 * Exit 0 when all agree, 1 otherwise. Dependency-free.
 */
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const readJson = (p) => JSON.parse(readFileSync(p, "utf8"));

export function collectVersions(root) {
  const plugin = readJson(join(root, ".claude-plugin", "plugin.json"));
  const pkg = readJson(join(root, "package.json"));
  const lock = readJson(join(root, "package-lock.json"));
  const market = readJson(join(root, "..", ".claude-plugin", "marketplace.json"));
  const entry = (market.plugins ?? []).find((p) => p.name === plugin.name);
  return {
    ".claude-plugin/plugin.json": plugin.version,
    "package.json": pkg.version,
    "package-lock.json (top level)": lock.version,
    'package-lock.json (packages[""])': lock.packages?.[""]?.version,
    [`marketplace.json (plugin "${plugin.name}")`]: entry?.version,
  };
}

export function checkVersions(root) {
  const versions = collectVersions(root);
  const values = Object.values(versions);
  if (values.every((v) => typeof v === "string" && v === values[0])) return [];
  return Object.entries(versions).map(([where, v]) => `${where}: ${v ?? "<missing>"}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = resolve(process.argv[2] ?? join(dirname(fileURLToPath(import.meta.url)), ".."));
  const mismatches = checkVersions(root);
  if (mismatches.length) {
    console.error("::error::plugin version differs between manifests:");
    for (const m of mismatches) console.error(`  ${m}`);
    process.exit(1);
  }
  console.log(`version guard ok: ${Object.values(collectVersions(root))[0]}`);
}
