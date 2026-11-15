/**
 * `tools/bake.mjs --check`: a fork can see which connector URL is baked in.
 * Runs a copy of the script inside a scratch plugin tree so nothing real is
 * read or written.
 */

import { spawnSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const BAKE = join(__dirname, "..", "tools", "bake.mjs");
const FIXTURE_URL = "https://memory.fixture.test/mcp/";

function scratch(mcpUrl: string | undefined | null): { run: (...args: string[]) => ReturnType<typeof spawnSync>; mcp: string } {
  const root = mkdtempSync(join(tmpdir(), "ns-bake-"));
  const plugin = join(root, "neuralscape-plugin");
  mkdirSync(join(plugin, "tools"), { recursive: true });
  copyFileSync(BAKE, join(plugin, "tools", "bake.mjs"));
  const mcp = join(plugin, ".mcp.json");
  const server = mcpUrl === null ? {} : { url: mcpUrl };
  writeFileSync(mcp, JSON.stringify({ mcpServers: { neuralscape: server } }, null, 2));
  return {
    mcp,
    run: (...args) =>
      spawnSync("node", [join(plugin, "tools", "bake.mjs"), ...args], { encoding: "utf-8" }),
  };
}

describe("bake --check", () => {
  it("reports the baked URL and the hooks' API base, writes nothing, exits 0", () => {
    const { run, mcp } = scratch(FIXTURE_URL);
    const before = readFileSync(mcp, "utf-8");
    const r = run("--check");
    expect(r.status).toBe(0);
    expect(r.stdout).toContain(`connector URL: ${FIXTURE_URL}`);
    expect(r.stdout).toContain("hooks API base: https://memory.fixture.test ");
    expect(r.stdout).toContain("npm run bake -- --url");
    expect(readFileSync(mcp, "utf-8")).toBe(before);
  });

  it("--expect passes when the baked URL matches (any equivalent spelling)", () => {
    const { run } = scratch(FIXTURE_URL);
    expect(run("--check", "--expect", "https://memory.fixture.test").status).toBe(0);
  });

  it("--expect fails with exit 1 when it differs", () => {
    const { run } = scratch(FIXTURE_URL);
    const r = run("--check", "--expect", "https://other.fixture.test");
    expect(r.status).toBe(1);
    expect(r.stderr).toContain("differs");
  });

  it("reports an un-baked template, an invalid URL and a missing URL as such", () => {
    expect(scratch("${MCP_URL}").run("--check").stdout).toContain("un-baked template");
    expect(scratch("not a url").run("--check").stdout).toContain("not a valid URL");
    expect(scratch(null).run("--check").stdout).toContain("(none)");
    // with --expect those are failures
    expect(scratch("${MCP_URL}").run("--check", "--expect", "https://a.fixture.test").status).toBe(1);
    expect(scratch(null).run("--check", "--expect", "https://a.fixture.test").status).toBe(1);
  });

  it("--expect without --check is an error", () => {
    const { run } = scratch(FIXTURE_URL);
    const r = run("--url", "https://a.fixture.test", "--expect", "https://a.fixture.test");
    expect(r.status).toBe(1);
    expect(r.stderr).toContain("--expect only works with --check");
  });

  it("a normal bake still works", () => {
    const { run, mcp } = scratch(FIXTURE_URL);
    expect(run("--url", "https://new.fixture.test").status).toBe(0);
    expect(JSON.parse(readFileSync(mcp, "utf-8")).mcpServers.neuralscape.url).toBe("https://new.fixture.test/mcp/");
  });
});
