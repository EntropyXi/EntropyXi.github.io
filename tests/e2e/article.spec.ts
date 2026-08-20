import { expect, test } from "@playwright/test";

test("article page renders content and MathJax formulas", async ({ page }) => {
  await page.goto("/2026/03/15/深度学习/流匹配与扩散模型/DDPM/");
  await expect(
    page.getByRole("heading", { name: "DDPM", exact: true }),
  ).toBeVisible();
  await expect(page.locator("mjx-container").first()).toBeVisible();

  const mathText = await page.locator(".prose").innerText();
  expect(mathText).not.toContain("$$");
  expect(mathText).not.toContain("\\begin");

  const layout = await page.evaluate(() => ({
    viewportWidth: window.innerWidth,
    documentWidth: document.documentElement.scrollWidth,
  }));
  expect(layout.documentWidth).toBeLessThanOrEqual(layout.viewportWidth);
});
