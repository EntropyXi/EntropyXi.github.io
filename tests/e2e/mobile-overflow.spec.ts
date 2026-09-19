import path from "node:path";
import { expect, test } from "@playwright/test";
import { buildPostPath } from "../../src/lib/routing/post-path";
import { readPostSources } from "../../scripts/lib/content-manifest";
import baseline from "../fixtures/legacy-baseline.json";

const repositoryRoot = path.resolve(__dirname, "../..");

const legacyArticlePaths = baseline.html
  .filter((entry) => entry.kind === "article")
  .map((entry) => entry.pathname);

// 迁移后新增的文章不在冻结基线里（基线禁止追加新文章），但同样要过移动端
// 溢出闸门——新文章带图片，而图片正是最容易撑破 360px 视口的内容。
const addedArticlePaths = readPostSources(
  path.join(repositoryRoot, "src", "content", "blog"),
)
  .filter((post) => post.data.draft !== true)
  .map((post) => post.data.permalink)
  .filter(
    (permalink): permalink is string =>
      typeof permalink === "string" && permalink.trim() !== "",
  )
  .map((permalink) => buildPostPath(permalink));

const SAMPLE_PAGES = [
  ...new Set([...legacyArticlePaths, ...addedArticlePaths]),
  "/about/",
  "/categories/",
  "/tags/",
  "/archives/",
];

const VIEWPORTS = [
  { name: "360", width: 360, height: 800 },
  { name: "320", width: 320, height: 568 },
];

interface Offender {
  tag: string;
  cls: string;
  width: number;
  right: number;
  snippet: string;
}

// Rendering is deterministic here (static build-time MathJax SVG, local
// fonts only), so a short stabilization wait plus a single measurement is
// reliable; the offender detail rides along in the assertion message.
async function measure(page: import("@playwright/test").Page) {
  await page.waitForTimeout(400);
  return page.evaluate(() => {
    const clientWidth = document.documentElement.clientWidth;
    const offenders: Offender[] = [];
    document.querySelectorAll("body *").forEach((el) => {
      // Fixed decoration layers (wallpaper, ambient) follow the viewport and
      // only mirror document width when something else overflows — skip them
      // so the report points at the real culprit.
      if (getComputedStyle(el).position === "fixed") return;
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.right > clientWidth + 2) {
        offenders.push({
          tag: el.tagName,
          cls: el.getAttribute("class") ?? "",
          width: Math.round(rect.width),
          right: Math.round(rect.right),
          snippet: (el.textContent ?? "").trim().slice(0, 40),
        });
      }
    });
    return {
      overflow: document.documentElement.scrollWidth > clientWidth + 1,
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth,
      offenders: offenders.slice(0, 12),
    };
  });
}

VIEWPORTS.forEach((viewport) => {
  test.describe(`mobile overflow gate @${viewport.name}`, () => {
    test.use({
      viewport: { width: viewport.width, height: viewport.height },
      hasTouch: true,
      isMobile: true,
    });

    for (const path of SAMPLE_PAGES) {
      test(`no page-level horizontal overflow: ${path}`, async ({ page }) => {
        // `load` (not domcontentloaded): pages with images (e.g. the CNN
        // article) reflow when they finish, and measuring mid-reflow flakes.
        await page.goto(path, { waitUntil: "load" });
        const result = await measure(page);
        const detail = JSON.stringify(
          {
            scrollWidth: result.scrollWidth,
            clientWidth: result.clientWidth,
            offenders: result.offenders,
          },
          null,
          1,
        );
        expect(
          result.overflow,
          `page overflows at ${viewport.name}: ${detail}`,
        ).toBe(false);
      });
    }
  });
});
