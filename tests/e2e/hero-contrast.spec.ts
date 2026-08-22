import { expect, test } from "@playwright/test";

test.describe("Hero Component Accessibility, Typography and Contrast Verification", () => {
  test("hero renders condensed welcome title on exactly two non-wrapping lines", async ({
    page,
  }) => {
    await page.goto("/");
    const line1 = page.locator(".welcome-line-1");
    const line2 = page.locator(".welcome-line-2");

    await expect(line1).toBeVisible();
    await expect(line2).toBeVisible();
    await expect(line1).toHaveText("WELCOME TO");
    await expect(line2).toHaveText("ENTROPYXI BLOG !");

    // Verify title lines do not wrap/overflow horizontally
    const line1NoWrap = await line1.evaluate((el) => {
      const style = window.getComputedStyle(el);
      return (
        style.whiteSpace === "nowrap" && el.scrollWidth <= el.clientWidth + 2
      );
    });
    expect(line1NoWrap).toBe(true);

    const line2NoWrap = await line2.evaluate((el) => {
      const style = window.getComputedStyle(el);
      return (
        style.whiteSpace === "nowrap" && el.scrollWidth <= el.clientWidth + 2
      );
    });
    expect(line2NoWrap).toBe(true);
  });

  test("hero narrative block has clean transparent background and updated text", async ({
    page,
  }) => {
    await page.goto("/");
    const narrativeBlock = page.locator(".hero-narrative-block");
    await expect(narrativeBlock).toBeVisible();

    // Verify no black box background
    const bg = await narrativeBlock.evaluate((el) => {
      const style = window.getComputedStyle(el);
      return (
        style.backgroundColor === "rgba(0, 0, 0, 0)" ||
        style.backgroundColor === "transparent"
      );
    });
    expect(bg).toBe(true);

    // Verify text content: contains core domains and excludes "收敛性证明"
    const text = await narrativeBlock.innerText();
    expect(text).toContain("这里是 EntropyXi 的技术笔记");
    expect(text).toContain("很高兴与你相遇！");
    expect(text).toContain("深度学习");
    expect(text).toContain("扩散模型 (Diffusion Models)");
    expect(text).toContain("流匹配 (Flow Matching)");
    expect(text).toContain("数值分析");
    expect(text).not.toContain("收敛性证明");
  });

  test("vertical bar scroll indicator has accessible label and 44px min touch target", async ({
    page,
  }) => {
    await page.goto("/");
    const scrollBtn = page.locator(".hero-scroll-indicator");
    await expect(scrollBtn).toBeVisible();
    await expect(scrollBtn).toHaveAttribute("aria-label", "向下滚动至最新文章");
    await expect(scrollBtn).toHaveAttribute("href", "#latest-posts");

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

    // Title line 1 color should remain high contrast pure white
    const titleColor = await page
      .locator(".welcome-line-1")
      .evaluate((el) => window.getComputedStyle(el).color);
    expect(titleColor).toBe("rgb(255, 255, 255)");
  });
});
