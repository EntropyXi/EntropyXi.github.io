import { expect, test } from "@playwright/test";

test("article page renders content and MathJax formulas", async ({ page }) => {
  await page.goto("/2026/03/15/深度学习/流匹配与扩散模型/DDPM/");
  await expect(
    page.getByRole("heading", { name: "DDPM", exact: true }),
  ).toBeVisible();
  await expect(page.locator("mjx-container").first()).toBeVisible();

  const formulaCount = await page.locator("mjx-container").count();
  const accessibleFormulaCount = await page
    .locator('mjx-container svg[role="img"][aria-label^="数学公式："]')
    .count();
  expect(accessibleFormulaCount).toBe(formulaCount);

  const mathText = await page.locator(".prose").innerText();
  expect(mathText).not.toContain("$$");
  expect(mathText).not.toContain("\\begin");

  const layout = await page.evaluate(() => ({
    viewportWidth: window.innerWidth,
    documentWidth: document.documentElement.scrollWidth,
  }));
  expect(layout.documentWidth).toBeLessThanOrEqual(layout.viewportWidth);
});

test("article page renders inline and display MathJax formulas with correct inline layout", async ({
  page,
}) => {
  await page.goto("/2026/02/08/数值分析/共轭梯度法中参数alpha和beta的推导/");
  await expect(
    page.getByRole("heading", {
      name: "共轭梯度法中参数alpha和beta的推导",
      exact: true,
    }),
  ).toBeVisible();

  // Verify inline math containers and svgs are not display: block
  const inlineContainers = page.locator('mjx-container:not([display="true"])');
  const inlineCount = await inlineContainers.count();
  expect(inlineCount).toBeGreaterThan(0);

  const inlineStyles = await inlineContainers.first().evaluate((el) => {
    const svg = el.querySelector("svg");
    return {
      containerDisplay: window.getComputedStyle(el).display,
      svgDisplay: svg ? window.getComputedStyle(svg).display : null,
    };
  });
  expect(inlineStyles.containerDisplay).toBe("inline-block");
  expect(inlineStyles.svgDisplay).toBe("inline-block");

  // Verify display math containers are display: block
  const displayContainers = page.locator('mjx-container[display="true"]');
  const displayCount = await displayContainers.count();
  expect(displayCount).toBeGreaterThan(0);

  const displayStyles = await displayContainers.first().evaluate((el) => {
    return {
      display: window.getComputedStyle(el).display,
    };
  });
  expect(displayStyles.display).toBe("block");
});
