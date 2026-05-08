import * as esbuild from "esbuild";

const entryPoints = [
  "src/hooks/session-start.ts",
  "src/hooks/conversation-turn.ts",
  "src/hooks/session-end.ts",
];

const isWatch = process.argv.includes("--watch");

const buildOptions = {
  entryPoints,
  bundle: true,
  platform: "node",
  target: "node18",
  format: "esm",
  outdir: "scripts",
  outbase: "src/hooks",
  minify: true,
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
