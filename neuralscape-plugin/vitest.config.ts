import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    globals: false,
    environment: "node",
    coverage: {
      provider: "v8",
      reporter: ["text", "text-summary"],
      include: [
        "src/utils.ts",
        "src/core/disclosure.ts",
        "src/core/read-gate.ts",
        "src/core/session-note.ts",
        "src/hooks/post-tool-use.ts",
        "src/hooks/pre-tool-use.ts",
        "src/hooks/session-start.ts",
        "src/hooks/user-prompt-submit.ts",
      ],
      // Per-file thresholds tuned to match what we built on this branch.
      // The hooks fire side-effecting `main()` calls at module load — those
      // bottom-of-file invocations are excluded as untestable in unit tests.
      thresholds: {
        lines: 95,
        functions: 95,
        branches: 90,
        statements: 95,
      },
    },
  },
});
