import { expect, test } from "@playwright/test";

test.describe("Hero Component Accessibility and Contrast Verification", () => {
  test("hero renders all 6 topic chips with valid hrefs", async ({ page }) => {
    await page.goto("/");
    const chips = page.locator(".hero-topic-chip");
    await expect(chips).toHaveCount(6);

    for (let i = 0; i < 6; i++) {
      const href = await chips.nth(i).getAttribute("href");
      expect(href).toMatch(/^\/tags\/.+/);
    }
  });

  test("hero renders 4 CTA action buttons", async ({ page }) => {
    await page.goto("/");
    const primaryCta = page.locator(".hero-btn-primary");
    await expect(primaryCta).toBeVisible();
    await expect(primaryCta).toHaveAttribute("href", "#latest-posts");

    const secondaryCtas = page.locator(".hero-btn-secondary");
    await expect(secondaryCtas).toHaveCount(3);
  });

  test("scroll indicator has accessible label and 44px min touch target", async ({
    page,
  }) => {
    await page.goto("/");
    const scrollBtn = page.locator("#hero-scroll-indicator");
    await expect(scrollBtn).toBeVisible();
    await expect(scrollBtn).toHaveAttribute("aria-label", "向下滚动至最新文章");

    const box = await scrollBtn.boundingBox();
    expect(box).not.toBeNull();
    if (box) {
      expect(box.width).toBeGreaterThanOrEqual(44);
      expect(box.height).toBeGreaterThanOrEqual(44);
    }
  });

  test("hero maintains dark-theme scope styling regardless of page theme", async ({
    page,
  }) => {
    await page.goto("/");
    // Switch to light theme
    await page.evaluate(() => {
      document.documentElement.dataset.theme = "light";
    });

    const hero = page.locator(".hero-fullscreen");
    await expect(hero).toHaveAttribute("data-theme-scope", "dark");

    // Title line 1 color should be high contrast white
    const titleColor = await page
      .locator(".welcome-line-1")
      .evaluate((el) => window.getComputedStyle(el).color);
    expect(titleColor).toBe("rgb(255, 255, 255)");
  });
});
