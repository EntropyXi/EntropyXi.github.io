import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { chromium, type Page } from "@playwright/test";

const BASE_URL = "http://127.0.0.1:4321";
const OUTPUT_DIR = path.resolve("audit-screenshots/phase-7");

interface CaptureCase {
  description: string;
  featureAmbient?: boolean;
  featureMagnetic?: boolean;
  featureReveal?: boolean;
  hasTouch?: boolean;
  height: number;
  hoverSelector?: string;
  id: string;
  isMobile?: boolean;
  reducedMotion?: boolean;
  theme: "dark" | "light";
  width: number;
}

const cases: CaptureCase[] = [
  {
    id: "01-fine-pointer-desktop-dark",
    width: 1440,
    height: 900,
    theme: "dark",
    description: "Fine pointer enabled desktop default dark theme",
  },
  {
    id: "02-fine-pointer-desktop-light",
    width: 1440,
    height: 900,
    theme: "light",
    description: "Fine pointer enabled desktop light high-contrast theme",
  },
  {
    id: "03-fine-pointer-magnetic-hover-cta",
    width: 1440,
    height: 900,
    theme: "dark",
    hoverSelector: ".hero-btn-primary[data-magnetic]",
    description: "Fine pointer magnetic hover state on primary hero CTA",
  },
  {
    id: "04-fine-pointer-magnetic-hover-card",
    width: 1440,
    height: 900,
    theme: "dark",
    hoverSelector: ".post-card[data-magnetic]",
    description: "Fine pointer magnetic hover state on post card",
  },
  {
    id: "05-touch-disabled-mobile-390-dark",
    width: 390,
    height: 844,
    theme: "dark",
    hasTouch: true,
    isMobile: true,
    description:
      "Touch device mobile 390px (magnetic disabled, scanline/flow disabled)",
  },
  {
    id: "06-touch-disabled-mobile-390-light",
    width: 390,
    height: 844,
    theme: "light",
    hasTouch: true,
    isMobile: true,
    description: "Touch device mobile 390px light theme (magnetic disabled)",
  },
  {
    id: "07-reduced-motion-desktop-dark",
    width: 1440,
    height: 900,
    theme: "dark",
    reducedMotion: true,
    description:
      "Reduced motion preference enabled (zero unnecessary motion, instant reveal)",
  },
  {
    id: "08-fallback-reveal-disabled",
    width: 1440,
    height: 900,
    theme: "dark",
    featureReveal: false,
    description: "Independent feature-flag fallback: data-feature-reveal=false",
  },
  {
    id: "09-fallback-ambient-disabled",
    width: 1440,
    height: 900,
    theme: "dark",
    featureAmbient: false,
    description: "Independent feature-flag fallback: data-feature-ambient=false",
  },
  {
    id: "10-fallback-magnetic-disabled",
    width: 1440,
    height: 900,
    theme: "dark",
    featureMagnetic: false,
    description:
      "Independent feature-flag fallback: data-feature-magnetic=false",
  },
  {
    id: "11-fallback-all-flags-disabled",
    width: 1440,
    height: 900,
    theme: "dark",
    featureReveal: false,
    featureAmbient: false,
    featureMagnetic: false,
    description:
      "All motion feature flags disabled simultaneously (safe clean baseline)",
  },
];

async function isServerReady(url: string): Promise<boolean> {
  try {
    const res = await fetch(url);
    return res.ok;
  } catch {
    return false;
  }
}

async function ensureServer(): Promise<{ child?: ChildProcess }> {
  if (await isServerReady(BASE_URL)) {
    return {};
  }
  console.log("Building site for preview server...");
  execFileSync("npm", ["run", "build"], { stdio: "inherit", shell: true });
  console.log("Starting preview server at 127.0.0.1:4321...");
  const child = spawn("npm", ["run", "preview:test"], {
    shell: true,
    stdio: "pipe",
  });
  for (let i = 0; i < 30; i++) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    if (await isServerReady(BASE_URL)) {
      console.log("Preview server is ready.");
      return { child };
    }
  }
  throw new Error("Preview server timed out after 30s.");
}

async function inspectPage(page: Page) {
  return page.evaluate(() => {
    const root = document.documentElement;
    const magneticElements = document.querySelectorAll("[data-magnetic]");
    const revealElements = document.querySelectorAll("[data-reveal]");
    const ambientLayers = document.querySelectorAll(".ambient-layer");
    return {
      theme: root.dataset.theme,
      motionPreference: root.dataset.motionPreference,
      motionReady: root.dataset.motion === "ready",
      featureReveal: root.getAttribute("data-feature-reveal"),
      featureAmbient: root.getAttribute("data-feature-ambient"),
      featureMagnetic: root.getAttribute("data-feature-magnetic"),
      magneticElementsCount: magneticElements.length,
      revealElementsCount: revealElements.length,
      ambientLayersCount: ambientLayers.length,
      horizontalOverflow: root.scrollWidth > root.clientWidth,
    };
  });
}

async function main(): Promise<void> {
  await mkdir(OUTPUT_DIR, { recursive: true });
  const server = await ensureServer();
  const browser = await chromium.launch({ headless: true });
  const evidenceList = [];
  try {
    for (const testCase of cases) {
      const context = await browser.newContext({
        colorScheme: testCase.theme,
        deviceScaleFactor: 1,
        reducedMotion: testCase.reducedMotion ? "reduce" : "no-preference",
        hasTouch: Boolean(testCase.hasTouch),
        isMobile: Boolean(testCase.isMobile),
        viewport: { width: testCase.width, height: testCase.height },
      });
      await context.addInitScript(
        ({ theme, featureReveal, featureAmbient, featureMagnetic }) => {
          localStorage.setItem("entropyxi-theme", theme);
          if (featureReveal !== undefined) {
            document.documentElement.setAttribute(
              "data-feature-reveal",
              String(featureReveal),
            );
          }
          if (featureAmbient !== undefined) {
            document.documentElement.setAttribute(
              "data-feature-ambient",
              String(featureAmbient),
            );
          }
          if (featureMagnetic !== undefined) {
            document.documentElement.setAttribute(
              "data-feature-magnetic",
              String(featureMagnetic),
            );
          }
        },
        {
          theme: testCase.theme,
          featureReveal: testCase.featureReveal,
          featureAmbient: testCase.featureAmbient,
          featureMagnetic: testCase.featureMagnetic,
        },
      );
      const page = await context.newPage();
      const response = await page.goto(BASE_URL, { waitUntil: "networkidle" });
      if (!response?.ok()) {
        throw new Error(
          `Failed to load ${BASE_URL} (status: ${response?.status()})`,
        );
      }
      if (testCase.hoverSelector) {
        const target = page.locator(testCase.hoverSelector).first();
        if (await target.isVisible()) {
          const box = await target.boundingBox();
          if (box) {
            await page.mouse.move(
              box.x + box.width / 2 + 10,
              box.y + box.height / 2 + 10,
            );
            await page.waitForTimeout(200);
          }
        }
      }
      const state = await inspectPage(page);
      const filename = `${testCase.id}.png`;
      const filepath = path.join(OUTPUT_DIR, filename);
      await page.screenshot({ path: filepath, fullPage: false });
      evidenceList.push({
        file: filename,
        id: testCase.id,
        description: testCase.description,
        width: testCase.width,
        height: testCase.height,
        theme: testCase.theme,
        reducedMotion: Boolean(testCase.reducedMotion),
        hasTouch: Boolean(testCase.hasTouch),
        featureFlags: {
          reveal: testCase.featureReveal ?? true,
          ambient: testCase.featureAmbient ?? true,
          magnetic: testCase.featureMagnetic ?? true,
        },
        state,
      });
      console.log(`Captured: ${filename}`);
      await context.close();
    }
    let commit = "HEAD";
    try {
      commit = execFileSync("git", ["rev-parse", "HEAD"], {
        encoding: "utf8",
      }).trim();
    } catch {
      // ignore
    }
    const report = {
      phase: 7,
      name: "Phase 7 Visual Evidence - Advanced Motion, Reveal and Magnetic Dynamics",
      browser: `Chromium ${browser.version()}`,
      commit,
      generatedAt: new Date().toISOString(),
      baseUrl: BASE_URL,
      evidence: evidenceList,
    };
    await writeFile(
      path.join(OUTPUT_DIR, "evidence.json"),
      `${JSON.stringify(report, null, 2)}\n`,
      "utf8",
    );
    await writeFile(
      path.join(OUTPUT_DIR, "index.json"),
      `${JSON.stringify(report, null, 2)}\n`,
      "utf8",
    );
    console.log(
      `Successfully generated ${evidenceList.length} Phase 7 evidence artifacts in ${OUTPUT_DIR}`,
    );
  } finally {
    await browser.close();
    if (server.child && !server.child.killed) {
      server.child.kill();
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
