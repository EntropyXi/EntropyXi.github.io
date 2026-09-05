import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const pages = [
  "/",
  "/archives/",
  "/search/",
  "/2026/05/10/深度学习/流匹配与扩散模型/DDIM/",
] as const;

for (const path of pages) {
  test(`has no serious or critical axe violations: ${path}`, async ({
    page,
  }) => {
    await page.goto(path, { waitUntil: "networkidle" });
    // Wait for motion to reach its static final state: the hero entrance
    // writes data-hero-ready when done, pages without a hero resolve at once.
    await page
      .waitForFunction(
        () =>
          document.documentElement.getAttribute("data-hero-ready") === "true" ||
          document.querySelector(".hero-fullscreen") === null,
        undefined,
        { timeout: 8000 },
      )
      .catch(() => {
        // The 1.6s inline safety timeout guarantees a visible hero even if
        // the choreography never ran; scan whatever state we reached.
      });
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
      .analyze();
    const blocking = results.violations.filter(
      ({ impact }) => impact === "serious" || impact === "critical",
    );

    expect(blocking).toEqual([]);
  });
}
