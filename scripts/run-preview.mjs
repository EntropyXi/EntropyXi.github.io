import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";

const require = createRequire(import.meta.url);
const astroPackage = require.resolve("astro/package.json");
const astroCli = resolve(dirname(astroPackage), "bin/astro.mjs");

const child = spawn(
  process.execPath,
  [astroCli, "preview", "--host", "127.0.0.1", "--port", "4321"],
  {
    env: { ...process.env, ASTRO_PREVIEW_BACKGROUND: "0" },
    stdio: "inherit",
  },
);

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    if (!child.killed) child.kill(signal);
  });
}

child.on("error", (error) => {
  console.error("Failed to start the Astro preview server.", error);
  process.exitCode = 1;
});

child.on("exit", (code) => {
  process.exitCode = code ?? 1;
});
