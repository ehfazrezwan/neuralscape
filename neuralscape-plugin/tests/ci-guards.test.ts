import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
// @ts-expect-error plain ESM tool scripts, no type declarations
import { checkBundles, expectedBundles } from "../tools/check-bundles.mjs";
// @ts-expect-error plain ESM tool scripts, no type declarations
import { checkVersions } from "../tools/check-versions.mjs";

let repo: string;
let root: string;

const write = (rel: string, body: string) => {
  const p = join(repo, rel);
  mkdirSync(join(p, ".."), { recursive: true });
  writeFileSync(p, body);
};
const git = (...args: string[]) => execFileSync("git", args, { cwd: repo, stdio: "pipe" });

beforeEach(() => {
  repo = mkdtempSync(join(tmpdir(), "ci-guards-"));
  root = join(repo, "neuralscape-plugin");
  git("init", "-q");
  write(
    "neuralscape-plugin/esbuild.config.js",
    'const entryPoints = [\n  "src/hooks/a.ts",\n  "src/hooks/b.ts",\n];\n',
  );
  write("neuralscape-plugin/hooks/hooks.json", '{"c":"node \\"${R}/scripts/c.js\\""}');
  write("neuralscape-plugin/hooks/other.json", '{"a":"node scripts/a.js"}');
  for (const n of ["a", "b", "c"]) write(`neuralscape-plugin/scripts/${n}.js`, "//");
  git("add", "-A");
});
afterEach(() => rmSync(repo, { recursive: true, force: true }));

describe("check-bundles", () => {
  it("derives the union of esbuild entries and manifest references", () => {
    expect([...expectedBundles(root).keys()].sort()).toEqual(["scripts/a.js", "scripts/b.js", "scripts/c.js"]);
  });
  it("passes when every bundle is tracked", () => {
    expect(checkBundles(root)).toEqual([]);
  });
  it("fails for an esbuild entry whose bundle is untracked", () => {
    git("rm", "-q", "--cached", "neuralscape-plugin/scripts/b.js");
    expect(checkBundles(root)).toEqual([expect.stringContaining("scripts/b.js (required by esbuild.config.js)")]);
  });
  it("fails for a manifest-only bundle that is missing", () => {
    git("rm", "-q", "-f", "neuralscape-plugin/scripts/c.js");
    expect(checkBundles(root)).toEqual([expect.stringContaining("scripts/c.js (required by hooks/hooks.json)")]);
  });
});

describe("check-versions", () => {
  const setup = (market: string, lock = "1.0.0") => {
    write("neuralscape-plugin/.claude-plugin/plugin.json", '{"name":"p","version":"1.0.0"}');
    write("neuralscape-plugin/package.json", '{"version":"1.0.0"}');
    write("neuralscape-plugin/package-lock.json", `{"version":"${lock}","packages":{"":{"version":"${lock}"}}}`);
    write(".claude-plugin/marketplace.json", `{"plugins":[{"name":"p","version":"${market}"}]}`);
  };
  it("passes when all agree", () => {
    setup("1.0.0");
    expect(checkVersions(root)).toEqual([]);
  });
  it("fails on a stale marketplace entry", () => {
    setup("0.9.0");
    expect(checkVersions(root).join("\n")).toContain("marketplace.json");
  });
  it("fails on a stale lockfile", () => {
    setup("1.0.0", "0.9.0");
    expect(checkVersions(root).join("\n")).toContain("package-lock.json");
  });
});
