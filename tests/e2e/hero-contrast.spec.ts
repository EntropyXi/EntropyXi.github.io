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

    // The fit-title feature must have measured the title and left its
    // decision marker; the value is environment-dependent (native when
    // Impact resolves locally, scaled on font-poor CI Linux), so only the
    // attribute presence is asserted here.
    await expect(page.locator(".hero-welcome-title")).toHaveAttribute(
      "data-hero-fit",
      /^(native|scaled)$/,
    );

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

  test("hero title scales to fit when Android-style font fallback applies", async ({
    page,
  }) => {
    // Android resolves none of the condensed fonts in the title stack, so
    // the fallback renders ~38% wider and was silently clipped. Force the
    // sans-serif fallback to reproduce that environment (the audit probe
    // measured 350px vs a 350px container at 390, 62px overflow at 320).
    // The override must exist BEFORE the deferred module scripts run their
    // first fit measurement, and documentElement may not exist at
    // init-script time (client-matrix.spec.ts:105 lesson) — attach it the
    // moment the element appears. DOMContentLoaded is too late.
    await page.addInitScript(() => {
      const observer = new MutationObserver(() => {
        if (document.documentElement) {
          const style = document.createElement("style");
          style.textContent =
            ".hero-welcome-title { font-family: sans-serif !important; }";
          document.documentElement.appendChild(style);
          observer.disconnect();
        }
      });
      observer.observe(document, { childList: true, subtree: true });
    });

    for (const viewport of [
      { width: 390, height: 844 },
      { width: 360, height: 800 },
      { width: 320, height: 568 },
    ]) {
      await page.setViewportSize(viewport);
      await page.goto("/");
      await waitForHeroSettled(page);

      // Carrier assertion: the rendered lines must stay inside the layout
      // container. (line scrollWidth alone is tautological here — the
      // fit-content parent tracks the text width.)
      const fit = await page.evaluate(() => {
        const container = document.querySelector(
          ".hero-content-container",
        ) as HTMLElement | null;
        const title = document.querySelector(
          ".hero-welcome-title",
        ) as HTMLElement | null;
        if (!container || !title) return null;
        const containerRight = container.getBoundingClientRect().right;
        const widestLineRight = Math.max(
          ...Array.from(title.querySelectorAll(":scope > span")).map(
            (line) => line.getBoundingClientRect().right,
          ),
        );
        return {
          marker: title.dataset.heroFit ?? null,
          widestLineRight: Math.round(widestLineRight),
          containerRight: Math.round(containerRight),
        };
      });
      expect(fit, `fit state at ${viewport.width}`).not.toBeNull();
      expect(
        fit!.widestLineRight,
        `title must not exceed the container at ${viewport.width}`,
      ).toBeLessThanOrEqual(fit!.containerRight + 2);
      expect(fit!.marker, `fit marker must exist at ${viewport.width}`).toMatch(
        /^(native|scaled)$/,
      );
      if (viewport.width < 390) {
        // 390 is the algorithm's equality boundary (350px text vs 350px
        // container) where native is acceptable; narrower viewports must
        // actively scale.
        expect(fit!.marker, `must scale at ${viewport.width}`).toBe("scaled");
      }
    }
  });
});
