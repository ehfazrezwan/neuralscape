/**
 * Shared harness for subprocess tests: a fixture HTTP server standing in for
 * Neuralscape and a runner for the BUILT hook bundles in scripts/.
 */

import { spawn } from "node:child_process";
import { createServer, type Server } from "node:http";
import { join } from "node:path";

export const PLUGIN_ROOT = join(__dirname, "..", "..");

export interface Recorded {
  method: string;
  path: string;
  body: Record<string, unknown>;
}

export interface Fixture {
  url: string;
  requests: Recorded[];
  /** Body returned for GET /v1/context/<anything>. */
  context: Record<string, unknown>;
  close(): Promise<void>;
}

export async function startFixture(): Promise<Fixture> {
  const fx = { url: "", requests: [] as Recorded[], context: {} as Record<string, unknown> } as Fixture;
  const server: Server = createServer((req, res) => {
    let body = "";
    req.on("data", (d) => (body += d));
    req.on("end", () => {
      const path = req.url ?? "";
      fx.requests.push({ method: req.method ?? "", path, body: body ? JSON.parse(body) : {} });
      if (req.method === "GET" && path.startsWith("/v1/context/")) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ status: "ok", user_id: "fixture-user", categories: {}, ...fx.context }));
        return;
      }
      if (req.method === "GET") {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end("{}");
        return;
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok" }));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const address = server.address();
  fx.url = `http://127.0.0.1:${typeof address === "object" && address ? address.port : 0}`;
  fx.close = () => new Promise<void>((resolve) => server.close(() => resolve()));
  return fx;
}

export interface RunResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

export function runHook(
  script: string,
  stdin: string,
  env: Record<string, string>,
): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    const child = spawn("node", [join(PLUGIN_ROOT, "scripts", script)], {
      env: { PATH: process.env.PATH ?? "", ...env },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    child.on("error", reject);
    child.on("close", (code) => resolve({ code, stdout, stderr }));
    child.stdin.write(stdin);
    child.stdin.end();
  });
}

export function additionalContext(r: RunResult): string {
  return JSON.parse(r.stdout).hookSpecificOutput?.additionalContext ?? "";
}

/** Poll until `predicate` holds (the async hooks answer before they finish their HTTP calls). */
export async function waitFor(predicate: () => boolean, timeoutMs = 5000): Promise<void> {
  const start = Date.now();
  while (!predicate()) {
    if (Date.now() - start > timeoutMs) throw new Error("waitFor timed out");
    await new Promise((r) => setTimeout(r, 25));
  }
}
