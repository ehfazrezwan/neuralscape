/**
 * Guard: the REST endpoints the plugin calls are exactly the ones its README
 * documents. Scans src/**\/*.ts for neuralscapeGet(/neuralscapePost( call
 * sites (and skills/*\/SKILL.md for the REST paths the skills name) and fails
 * when the set differs from the README section "Server endpoints the plugin
 * calls" — so adding or dropping a call forces the doc (and a look at what
 * the server serves) to move with it.
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const ROOT = join(__dirname, "..");

function walk(dir: string, accept: (p: string) => boolean): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...walk(p, accept));
    else if (accept(p)) out.push(p);
  }
  return out;
}

/** `GET /v1/context/{project_id}` style, from a call-site path literal. */
function normalize(path: string): string {
  return path.replace(/\$\{[^}]*\}/g, "{project_id}");
}

export function scanSourceEndpoints(): string[] {
  const found = new Set<string>();
  const call = /neuralscape(Get|Post)\(\s*([`"'])([^`"']+)\2/g;
  for (const file of walk(join(ROOT, "src"), (p) => p.endsWith(".ts"))) {
    const text = readFileSync(file, "utf-8");
    for (const m of text.matchAll(call)) {
      found.add(`${m[1].toUpperCase()} ${normalize(m[3])}`);
    }
  }
  return [...found].sort();
}

function readmeSection(heading: string): string {
  const readme = readFileSync(join(ROOT, "README.md"), "utf-8");
  const start = readme.indexOf(`\n### ${heading}\n`);
  expect(start, `README is missing "### ${heading}"`).toBeGreaterThan(-1);
  const rest = readme.slice(start + 1);
  const next = rest.slice(4).search(/\n#{2,3} /);
  return next === -1 ? rest : rest.slice(0, next + 4);
}

function documented(section: string): string[] {
  return [...section.matchAll(/`(GET|POST|DELETE|PUT) (\/v1\/[^`\s]+)`/g)].map((m) => `${m[1]} ${m[2]}`).sort();
}

describe("server endpoints the plugin calls", () => {
  it("the scanner sees the hook call sites", () => {
    expect(scanSourceEndpoints().length).toBeGreaterThanOrEqual(9);
  });

  it("README lists exactly the endpoints src/ calls", () => {
    const section = readmeSection("From the hooks (src/)");
    expect(documented(section)).toEqual(scanSourceEndpoints());
  });

  it("README lists exactly the REST paths the skills name", () => {
    const named = new Set<string>();
    for (const file of walk(join(ROOT, "skills"), (p) => p.endsWith("SKILL.md"))) {
      const text = readFileSync(file, "utf-8");
      for (const m of text.matchAll(/\/v1\/[A-Za-z0-9_/{}.-]*[A-Za-z0-9_}]/g)) named.add(m[0]);
    }
    const section = readmeSection("From the skills (REST, via the plugin's helper)");
    const paths = documented(section).map((e) => e.split(" ")[1]).sort();
    expect(paths).toEqual([...named].sort());
  });
});
