import { expect, test } from "@playwright/test";

const complexMathPosts = [
  "/2026/03/15/深度学习/流匹配与扩散模型/DDPM/",
  "/2026/05/17/深度学习/流匹配与扩散模型/体系/1. 从SDE开始/",
  "/2026/05/17/深度学习/流匹配与扩散模型/超分辨率/SR3/",
  "/2026/05/10/深度学习/流匹配与扩散模型/DDIM/",
  "/2026/05/17/深度学习/流匹配与扩散模型/超分辨率/ResShift/",
];

for (const path of complexMathPosts) {
  test(`article layout is stable for ${path}`, async ({ page }) => {
    for (const viewport of [
      { width: 1280, height: 900 },
      { width: 390, height: 844 },
    ]) {
      await page.setViewportSize(viewport);
      await page.goto(path);
      await expect(page.locator("mjx-container").first()).toBeVisible();

      const formulaCount = await page.locator("mjx-container").count();
      await expect(
        page.locator('mjx-container svg[role="img"][aria-label^="数学公式："]'),
      ).toHaveCount(formulaCount);

      const layout = await page.evaluate(() => ({
        viewportWidth: window.innerWidth,
        documentWidth: document.documentElement.scrollWidth,
      }));
      expect(layout.documentWidth).toBeLessThanOrEqual(layout.viewportWidth);
    }
  });
}
