import { execFileSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { chromium } from "@playwright/test";

const BASE_URL = "http://127.0.0.1:4321";
const OUTPUT_DIR = path.resolve("audit-screenshots/phase-0");

const pages = [
  { id: "home", pathname: "/" },
  {
    id: "article-ddim",
    pathname:
      "/2026/05/10/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%B5%81%E5%8C%B9%E9%85%8D%E4%B8%8E%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B/DDIM/",
  },
  { id: "archives", pathname: "/archives/" },
  { id: "categories", pathname: "/categories/" },
  { id: "tags", pathname: "/tags/" },
  { id: "search", pathname: "/search/" },
  { id: "about", pathname: "/about/" },
  { id: "404", pathname: "/404.html" },
] as const;

const viewports = [
  { id: "1440x900", width: 1440, height: 900 },
  { id: "390x844", width: 390, height: 844 },
] as const;

const themes = ["dark", "light"] as const;

async function main(): Promise<void> {
  await mkdir(OUTPUT_DIR, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const files: Array<{
    file: string;
    pathname: string;
    theme: (typeof themes)[number];
    viewport: (typeof viewports)[number];
  }> = [];

  try {
    for (const viewport of viewports) {
      for (const theme of themes) {
        const context = await browser.newContext({
          colorScheme: theme,
          deviceScaleFactor: 1,
          viewport,
        });
        await context.addInitScript((selectedTheme) => {
          localStorage.setItem("entropyxi-theme", selectedTheme);
        }, theme);

        const page = await context.newPage();

        for (const pageSpec of pages) {
          const response = await page.goto(
            new URL(pageSpec.pathname, BASE_URL).href,
            { waitUntil: "networkidle" },
          );
          if (!response?.ok()) {
            throw new Error(
              `${pageSpec.pathname} returned ${response?.status() ?? "no response"}`,
            );
          }

          const settledTheme = await page.locator("html").getAttribute("data-theme");
          if (settledTheme !== theme) {
            throw new Error(
              `${pageSpec.pathname} expected ${theme} theme, received ${settledTheme}`,
            );
          }

          const file = `${pageSpec.id}-${viewport.id}-${theme}.png`;
          await page.screenshot({
            animations: "disabled",
            fullPage: false,
            path: path.join(OUTPUT_DIR, file),
          });
          files.push({ file, pathname: pageSpec.pathname, theme, viewport });
        }

        await context.close();
      }
    }

    const commit = execFileSync("git", ["rev-parse", "HEAD"], {
      encoding: "utf8",
    }).trim();
    const index = {
      browser: `Chromium ${browser.version()}`,
      commit,
      generatedAt: new Date().toISOString(),
      baseUrl: BASE_URL,
      command: "npm run capture:ui-baseline",
      motionPreference: "no-preference",
      files,
    };
    await writeFile(
      path.join(OUTPUT_DIR, "index.json"),
      `${JSON.stringify(index, null, 2)}\n`,
      "utf8",
    );
    console.log(`Captured ${files.length} baseline screenshots in ${OUTPUT_DIR}`);
  } finally {
    await browser.close();
  }
}

void main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
