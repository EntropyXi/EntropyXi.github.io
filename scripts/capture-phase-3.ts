import { execFileSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { chromium, type Page } from "@playwright/test";

const BASE_URL = "http://127.0.0.1:4321";
const OUTPUT_DIR = path.resolve("audit-screenshots/phase-3");

interface CaptureCase {
  height: number;
  id: string;
  reducedMotion?: boolean;
  theme: "dark" | "light";
  width: number;
}

const cases = [
  { id: "home-dark-desktop-1440", width: 1440, height: 900, theme: "dark" },
  { id: "home-light-desktop-1440", width: 1440, height: 900, theme: "light" },
  { id: "home-dark-mobile-390", width: 390, height: 844, theme: "dark" },
  { id: "home-light-mobile-390", width: 390, height: 844, theme: "light" },
  {
    id: "home-dark-reduced-motion",
    width: 1440,
    height: 900,
    theme: "dark",
    reducedMotion: true,
  },
] as const satisfies readonly CaptureCase[];

async function inspectPage(page: Page) {
  return page.evaluate(() => {
    const layerState = [...document.querySelectorAll<HTMLElement>("[data-layer]")].map(
      (element) => ({
        animationName: getComputedStyle(element).animationName,
        display: getComputedStyle(element).display,
        layer: element.dataset.layer,
        pointerEvents: getComputedStyle(element).pointerEvents,
      }),
    );
    const ambient = document.querySelector<HTMLElement>(".ambient-background");

    return {
      ambientZIndex: ambient ? getComputedStyle(ambient).zIndex : null,
      bodyBackground: getComputedStyle(document.body).backgroundColor,
      clientWidth: document.documentElement.clientWidth,
      headerState:
        document.querySelector<HTMLElement>("[data-site-header]")?.dataset.headerState ??
        null,
      horizontalOverflow:
        document.documentElement.scrollWidth > document.documentElement.clientWidth,
      layerState,
      motionPreference: document.documentElement.dataset.motionPreference ?? null,
      scrollWidth: document.documentElement.scrollWidth,
      theme: document.documentElement.dataset.theme ?? null,
    };
  });
}

async function main(): Promise<void> {
  await mkdir(OUTPUT_DIR, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const evidence = [];

  try {
    for (const testCase of cases) {
      const reducedMotion =
        "reducedMotion" in testCase && testCase.reducedMotion;
      const context = await browser.newContext({
        colorScheme: testCase.theme,
        deviceScaleFactor: 1,
        reducedMotion: reducedMotion ? "reduce" : "no-preference",
        viewport: { width: testCase.width, height: testCase.height },
      });
      await context.addInitScript((theme) => {
        localStorage.setItem("entropyxi-theme", theme);
      }, testCase.theme);

      const page = await context.newPage();
      const response = await page.goto(BASE_URL, { waitUntil: "networkidle" });
      if (!response?.ok()) {
        throw new Error(`Home page returned ${response?.status() ?? "no response"}`);
      }

      const state = await inspectPage(page);
      if (state.theme !== testCase.theme) {
        throw new Error(`${testCase.id} resolved theme ${state.theme}`);
      }
      if (state.horizontalOverflow) {
        throw new Error(`${testCase.id} has horizontal overflow`);
      }
      if (state.layerState.length !== 4) {
        throw new Error(`${testCase.id} exposes ${state.layerState.length} ambient layers`);
      }
      if (state.layerState.some((layer) => layer.pointerEvents !== "none")) {
        throw new Error(`${testCase.id} has an interactive ambient layer`);
      }
      if (
        reducedMotion &&
        state.layerState.some((layer) => layer.animationName !== "none")
      ) {
        throw new Error(`${testCase.id} retains an ambient animation`);
      }

      const file = `${testCase.id}.png`;
      await page.screenshot({
        animations: "disabled",
        fullPage: false,
        path: path.join(OUTPUT_DIR, file),
      });
      evidence.push({ ...testCase, file, ...state });
      await context.close();
    }

    const commit = execFileSync("git", ["rev-parse", "HEAD"], {
      encoding: "utf8",
    }).trim();
    const index = {
      allCasesPass: true,
      baseUrl: BASE_URL,
      browser: `Chromium ${browser.version()}`,
      command: "npm run capture:phase-3",
      commit,
      generatedAt: new Date().toISOString(),
      note: "Captured from the reviewed Stage 3 worktree before its independent commit.",
      cases: evidence,
    };
    await writeFile(
      path.join(OUTPUT_DIR, "index.json"),
      `${JSON.stringify(index, null, 2)}\n`,
      "utf8",
    );
    console.log(`Captured ${evidence.length} Stage 3 screenshots in ${OUTPUT_DIR}`);
  } finally {
    await browser.close();
  }
}

void main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
