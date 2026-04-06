import * as esbuild from "esbuild";

const entryPoints = [
  "src/session-start.ts",
  "src/post-tool-use.ts",
  "src/stop.ts",
  "src/conversation-turn.ts",
  "src/session-summary.ts",
];

const isWatch = process.argv.includes("--watch");

const buildOptions = {
  entryPoints,
  bundle: true,
  platform: "node",
  target: "node18",
  format: "esm",
  outdir: "scripts",
  banner: {
    js: '#!/usr/bin/env node',
  },
};

if (isWatch) {
  const ctx = await esbuild.context(buildOptions);
  await ctx.watch();
  console.log("Watching for changes...");
} else {
  await esbuild.build(buildOptions);
  console.log("Build complete.");
}
