import { expect, test, type Page } from "@playwright/test";

async function waitForHeroSettled(page: Page): Promise<void> {
  // The entrance choreography writes data-hero-ready when it reaches the
  // final layout; pages without a hero resolve immediately. The 8s cap only
  // guards against a stalled run — the inline 1.6s timeout already
  // guarantees a visible hero.
  await page
    .waitForFunction(
      () =>
        document.documentElement.getAttribute("data-hero-ready") === "true" ||
        document.querySelector(".hero-fullscreen") === null,
      undefined,
      { timeout: 8000 },
    )
    .catch(() => undefined);
}

test.describe("Hero Component Accessibility, Typography and Contrast Verification", () => {
  test("hero renders condensed welcome title on exactly two non-wrapping lines", async ({
    page,
  }) => {
    await page.goto("/");
    await waitForHeroSettled(page);
    const line1 = page.locator(".welcome-line-1");
    const line2 = page.locator(".welcome-line-2");

    await expect(line1).toBeVisible();
    await expect(line2).toBeVisible();
    await expect(line1).toHaveText("WELCOME TO");
    await expect(line2).toHaveText("ENTROPYXI BLOG !");

    // The h1 keeps an accessible name after the line split and revert.
    await expect(
      page.getByRole("heading", { name: "WELCOME TO ENTROPYXI BLOG !" }),
    ).toBeVisible();

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
    await waitForHeroSettled(page);
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
    expect(text).toContain("这里是 EntropyXi 的个人博客");
    expect(text).toContain("很高兴与你相遇！");
    expect(text).toContain("深度学习");
    expect(text).toContain("扩散模型");
    expect(text).toContain("流匹配");
    expect(text).toContain("数值分析");
    expect(text).not.toContain("收敛性证明");

    // Verify navbar brand avatar logo
    await expect(page.locator(".site-logo-avatar")).toBeVisible();
  });

  test("vertical bar scroll indicator has accessible label and 44px min touch target", async ({
    page,
  }) => {
    await page.goto("/");
    await waitForHeroSettled(page);
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

  test("hero maintains dark-theme scope styling and high contrast white title", async ({
    page,
  }) => {
    await page.goto("/");
    await waitForHeroSettled(page);
    const hero = page.locator(".hero-fullscreen");
    await expect(hero).toHaveAttribute("data-theme-scope", "dark");

    // Title line 1 color should remain high contrast pure white
    const titleColor = await page
      .locator(".welcome-line-1")
      .evaluate((el) => window.getComputedStyle(el).color);
    expect(titleColor).toBe("rgb(255, 255, 255)");
  });

  test("hero entrance settles without clipping on short viewports", async ({
    page,
  }) => {
    for (const viewport of [
      { width: 1280, height: 700 },
      { width: 390, height: 600 },
    ]) {
      await page.setViewportSize(viewport);
      await page.goto("/");
      await waitForHeroSettled(page);

      for (const line of [".welcome-line-1", ".welcome-line-2"]) {
        const noWrap = await page.locator(line).evaluate((el) => {
          const style = window.getComputedStyle(el);
          return (
            style.whiteSpace === "nowrap" &&
            el.scrollWidth <= el.clientWidth + 2
          );
        });
        expect(noWrap, `${line} at ${viewport.width}x${viewport.height}`).toBe(
          true,
        );
      }

      const box = await page.locator(".hero-scroll-indicator").boundingBox();
      expect(box?.height).toBeGreaterThanOrEqual(44);
    }
  });
});
