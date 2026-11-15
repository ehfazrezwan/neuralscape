/**
 * Failure classification (plg-02): a bad token, a wrong URL and a down
 * service used to all read "memory service unreachable". The HTTP client now
 * throws typed errors, one classifier decides what a failure means, and
 * SessionStart words the notice per kind. Also: the OS-login identity notice.
 */

import { mkdtemp, readFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeEach, describe, expect, it, vi } from "vitest";

process.env.NEURALSCAPE_TEST = "1";
process.env.NEURALSCAPE_USER_ID = "unit-user";

let server: Server;
let status = 200;
let mode: "status" | "badjson" = "status";
const sockets = new Set<import("node:net").Socket>();

await (async () => {
  server = createServer((_req, res) => {
    if (mode === "badjson") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end("not json");
      return;
    }
    res.writeHead(status, { "Content-Type": "application/json" });
    res.end(status === 200 ? '{"ok":true}' : '{"detail":"x"}');
  });
  server.on("connection", (s) => {
    sockets.add(s);
    s.on("close", () => sockets.delete(s));
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const address = server.address();
  process.env.NEURALSCAPE_URL = `http://127.0.0.1:${typeof address === "object" && address ? address.port : 0}`;
})();

afterAll(async () => {
  for (const s of sockets) s.destroy();
  await new Promise<void>((r) => server.close(() => r()));
});

beforeEach(() => {
  status = 200;
  mode = "status";
});

const utils = await import("../src/utils.js");
const {
  NeuralscapeHttpError,
  NeuralscapeNetworkError,
  classifyFailure,
  getTransportFailureCount,
  getUserIdSource,
  isUnreachableKind,
  neuralscapeGet,
  neuralscapePost,
  recordFailure,
} = utils;
const { buildFailureNotice, buildIdentityNotice, isLocalServiceUrl } = await import("../src/hooks/session-start.js");

describe("classifyFailure", () => {
  it.each([
    [401, "auth"],
    [403, "auth"],
    [404, "not_found"],
    [400, "client"],
    [422, "client"],
    [429, "client"],
    [500, "server"],
    [503, "server"],
  ] as const)("HTTP %i is %s", (code, kind) => {
    expect(classifyFailure(new NeuralscapeHttpError(code, "x"))).toBe(kind);
  });

  it("anything that is not an HTTP answer is a network failure", () => {
    expect(classifyFailure(new NeuralscapeNetworkError("could not connect"))).toBe("network");
    expect(classifyFailure(new TypeError("fetch failed"))).toBe("network");
    expect(classifyFailure("boom")).toBe("network");
  });

  it("only network and server failures are 'unreachable'", () => {
    expect(isUnreachableKind("network")).toBe(true);
    expect(isUnreachableKind("server")).toBe(true);
    for (const k of ["auth", "not_found", "client"] as const) expect(isUnreachableKind(k)).toBe(false);
  });

  it("keeps the legacy message shape", () => {
    expect(new NeuralscapeHttpError(401, "Unauthorized").message).toBe("HTTP 401: Unauthorized");
  });
});

describe("the HTTP client throws typed errors", () => {
  it("GET: non-2xx → NeuralscapeHttpError with the status", async () => {
    status = 401;
    const err = await neuralscapeGet("/v1/x").catch((e) => e);
    expect(err).toBeInstanceOf(NeuralscapeHttpError);
    expect(err.status).toBe(401);
  });

  it("POST: non-2xx → NeuralscapeHttpError with the status", async () => {
    status = 500;
    const err = await neuralscapePost("/v1/x", {}).catch((e) => e);
    expect(err).toBeInstanceOf(NeuralscapeHttpError);
    expect(err.status).toBe(500);
  });

  it("2xx still resolves with the body", async () => {
    expect(await neuralscapeGet("/v1/x", { a: "b" })).toEqual({ ok: true });
    expect(await neuralscapePost("/v1/x", { a: 1 })).toEqual({ ok: true });
  });

  it("a refused connection → NeuralscapeNetworkError", async () => {
    const closed = createServer();
    await new Promise<void>((r) => closed.listen(0, "127.0.0.1", r));
    const addr = closed.address();
    const port = typeof addr === "object" && addr ? addr.port : 0;
    await new Promise<void>((r) => closed.close(() => r()));
    const saved = process.env.NEURALSCAPE_URL;
    process.env.NEURALSCAPE_URL = `http://127.0.0.1:${port}`;
    try {
      vi.resetModules(); // the service URL is read once, at import
      const fresh = await import("../src/utils.js");
      const err = await fresh.neuralscapeGet("/v1/x").catch((e: unknown) => e);
      expect(err).toBeInstanceOf(fresh.NeuralscapeNetworkError);
      expect(fresh.classifyFailure(err)).toBe("network");
    } finally {
      process.env.NEURALSCAPE_URL = saved;
    }
  });

  it("the request timeout (AbortError) → NeuralscapeNetworkError saying so", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockRejectedValue(Object.assign(new Error("aborted"), { name: "AbortError" }));
    try {
      const err = await neuralscapeGet("/v1/x").catch((e) => e);
      expect(err).toBeInstanceOf(NeuralscapeNetworkError);
      expect(err.message).toContain("timed out");
    } finally {
      spy.mockRestore();
    }
  });

  it("a 2xx with a non-JSON body is passed through and still counts as transport", async () => {
    mode = "badjson";
    const err = await neuralscapeGet("/v1/x").catch((e) => e);
    expect(err).toBeInstanceOf(Error);
    expect(err).not.toBeInstanceOf(NeuralscapeHttpError);
    expect(classifyFailure(err)).toBe("network");
  });
});

describe("recordFailure only counts a down or failing service", () => {
  let dir: string;
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "ns-failkind-"));
    process.env.CLAUDE_PLUGIN_DATA = dir;
  });

  it("network and 5xx advance the streak", async () => {
    expect(await recordFailure(new NeuralscapeNetworkError("x"))).toEqual({ kind: "network", count: 1 });
    expect(await recordFailure(new NeuralscapeHttpError(502, "Bad Gateway"))).toEqual({ kind: "server", count: 2 });
    expect(await getTransportFailureCount()).toBe(2);
  });

  it("401 / 403 / 404 / 400 leave the streak alone", async () => {
    await recordFailure(new NeuralscapeNetworkError("x"));
    for (const code of [401, 403, 404, 400]) {
      const r = await recordFailure(new NeuralscapeHttpError(code, "x"));
      expect(r.count).toBeNull();
    }
    expect(await getTransportFailureCount()).toBe(1);
    await expect(readFile(join(dir, "observations", "unreachable.count"), "utf-8")).resolves.toBe("1");
  });
});

describe("buildFailureNotice", () => {
  it("auth: says the key was rejected and how to fix it, never unreachable", () => {
    const n = buildFailureNotice("auth", null, 3);
    expect(n).toContain("rejected the API key");
    expect(n).toContain("/plugin config neuralscape@neuralscape-plugins");
    expect(n).not.toContain("unreachable");
    expect(n).not.toContain("docker compose");
  });

  it("not_found: points at the URL", () => {
    const n = buildFailureNotice("not_found", null, 3);
    expect(n).toContain("404");
    expect(n).toContain("URL");
    expect(n).not.toContain("unreachable");
  });

  it("client: generic 4xx wording", () => {
    const n = buildFailureNotice("client", null, 3);
    expect(n).toContain("rejected the context request");
    expect(n).not.toContain("unreachable");
  });

  it("server / network keep the unreachable wording and the fail-loud upgrade", () => {
    expect(buildFailureNotice("network", 1, 3)).toContain("memory service unreachable —");
    expect(buildFailureNotice("server", 3, 3)).toContain("unreachable for 3 consecutive events");
    expect(buildFailureNotice("network", null, 3)).toContain("unreachable");
  });
});

describe("identity notice (USER_ID unset, falling back to the OS login name)", () => {
  const REMOTE = "https://memory.fixture.test";

  it("fires for the OS fallback against a non-local URL, naming the id but not the host", () => {
    const n = buildIdentityNotice("alice", "os", REMOTE);
    expect(n).toContain('"alice"');
    expect(n).toContain("USER_ID");
    expect(n).toContain("/plugin config neuralscape@neuralscape-plugins");
    expect(n).not.toContain("fixture.test");
  });

  it("is silent for a configured id, no id, or a local service", () => {
    expect(buildIdentityNotice("alice", "config", REMOTE)).toBe("");
    expect(buildIdentityNotice("", "none", REMOTE)).toBe("");
    expect(buildIdentityNotice("", "os", REMOTE)).toBe("");
    for (const local of ["http://localhost:8199", "http://127.0.0.1:8199", "http://[::1]:8199"]) {
      expect(buildIdentityNotice("alice", "os", local)).toBe("");
    }
  });

  it("isLocalServiceUrl: unparsable URLs count as local (no notice noise)", () => {
    expect(isLocalServiceUrl("not a url")).toBe(true);
    expect(isLocalServiceUrl(REMOTE)).toBe(false);
  });

  it("getUserIdSource reports config when USER_ID is configured", () => {
    // NEURALSCAPE_USER_ID was set before the module loaded.
    expect(getUserIdSource()).toBe("config");
  });
});
