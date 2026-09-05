import { expect, test } from "@playwright/test";

test("client foundation matches its configured environment", async ({
  page,
}, testInfo) => {
  if (testInfo.project.name === "reduced-motion") {
    await page.emulateMedia({ reducedMotion: "reduce" });
  }
  await page.goto("/");

  const root = page.locator("html");
  const heading = page.getByRole("heading", {
    name: /WELCOME TO.*ENTROPYXI BLOG/i,
  });
  await expect(heading).toBeVisible();

  if (testInfo.project.name === "javascript-disabled") {
    await expect(root).not.toHaveAttribute("data-motion", "ready");
    await expect(page.locator(".post-card").first()).toBeVisible();
    await page.goto("/search/");
    await expect(page.locator(".search-noscript-text")).toContainText(
      "实时站内搜索功能不可用",
    );
    await expect(page.getByRole("link", { name: "全站归档" })).toHaveAttribute(
      "href",
      "/archives/",
    );
    await expect(page.getByRole("link", { name: "文章分类" })).toHaveAttribute(
      "href",
      "/categories/",
    );
    await expect(page.getByRole("link", { name: "标签列表" })).toHaveAttribute(
      "href",
      "/tags/",
    );
    return;
  }

  await expect(root).toHaveAttribute("data-motion", "ready");

  if (testInfo.project.name === "reduced-motion") {
    await expect(root).toHaveAttribute("data-motion-preference", "reduced");
    await expect(page.locator(".ambient-scanline-layer")).toHaveCSS(
      "animation-name",
      "none",
    );
    await expect(page.locator(".status-pulse")).toHaveCSS(
      "animation-name",
      "none",
    );
    return;
  }

  if (
    testInfo.project.name === "mobile-390" ||
    testInfo.project.name === "mobile-360"
  ) {
    await expect
      .poll(() =>
        page.evaluate(
          () =>
            document.documentElement.scrollWidth <=
            document.documentElement.clientWidth,
        ),
      )
      .toBe(true);
    await expect(page.locator(".ambient-scanline-layer")).toHaveCSS(
      "display",
      "none",
    );
    await expect(page.locator(".ambient-flow-layer")).toHaveCSS(
      "display",
      "none",
    );
    await page.getByRole("button", { name: "打开导航菜单" }).click();
    await expect(page.getByRole("button", { name: "关闭菜单" })).toBeFocused();
    return;
  }

  if (testInfo.project.name === "zoom-200") {
    await page.evaluate(() => {
      document.documentElement.style.zoom = "200%";
    });
    await expect(heading).toBeVisible();
    await expect
      .poll(() =>
        page.evaluate(
          () =>
            document.documentElement.scrollWidth <=
            document.documentElement.clientWidth,
        ),
      )
      .toBe(true);
  }
});

test("motion runtime matches its capability gate", async ({
  page,
}, testInfo) => {
  if (testInfo.project.name === "reduced-motion") {
    await page.emulateMedia({ reducedMotion: "reduce" });
  }
  if (testInfo.project.name === "zoom-200") {
    // Set CSS zoom before the idle-time gate runs; documentElement may not
    // exist at init-script time, so fall back to DOMContentLoaded.
    await page.addInitScript(() => {
      const apply = (): void => {
        document.documentElement.style.zoom = "200%";
      };
      if (document.documentElement) {
        apply();
      } else {
        document.addEventListener("DOMContentLoaded", apply, { once: true });
      }
    });
  }
  await page.goto("/");
  const root = page.locator("html");

  if (testInfo.project.name === "javascript-disabled") {
    await expect(root).not.toHaveAttribute("data-motion-runtime-init");
    return;
  }

  if (testInfo.project.name === "reduced-motion") {
    // Reduced motion never loads the stack and GSAP must not touch
    // inline transforms — CSS animation-name assertions alone cannot see
    // GSAP-driven movement (plan §4.0 marker contract + R2-5).
    await expect
      .poll(() =>
        page.evaluate(() => ({
          lenisActive: document.documentElement.dataset.lenisActive ?? null,
          lenisClass: document.documentElement.classList.contains("lenis"),
          gsapActive: document.documentElement.dataset.gsapActive ?? null,
        })),
      )
      .toEqual({ lenisActive: null, lenisClass: false, gsapActive: null });
    await expect(page.locator(".welcome-line-1")).toHaveCSS(
      "transform",
      "none",
    );
    await expect(page.locator(".post-card").first()).toHaveCSS(
      "transform",
      "none",
    );
    await expect
      .poll(() =>
        page.evaluate(async () => {
          const line = document.querySelector(".welcome-line-1");
          const readTop = () =>
            new Promise<number | undefined>((resolve) => {
              requestAnimationFrame(() =>
                resolve(line?.getBoundingClientRect().top),
              );
            });
          const first = await readTop();
          const second = await readTop();
          return first === second;
        }),
      )
      .toBe(true);
    return;
  }

  if (testInfo.project.name === "zoom-200") {
    // CSS zoom disables Lenis only — GSAP scroll work stays on.
    await expect
      .poll(() =>
        page.evaluate(() => ({
          lenisActive: document.documentElement.dataset.lenisActive ?? null,
          lenisClass: document.documentElement.classList.contains("lenis"),
          gsapActive: document.documentElement.dataset.gsapActive ?? null,
        })),
      )
      .toEqual({ lenisActive: null, lenisClass: false, gsapActive: "true" });
    return;
  }

  if (
    testInfo.project.name === "mobile-390" ||
    testInfo.project.name === "mobile-360" ||
    testInfo.project.name === "mobile-safari"
  ) {
    // Touch keeps GSAP-driven scroll work but never Lenis.
    await expect
      .poll(() =>
        page.evaluate(() => ({
          lenisActive: document.documentElement.dataset.lenisActive ?? null,
          lenisClass: document.documentElement.classList.contains("lenis"),
          gsapActive: document.documentElement.dataset.gsapActive ?? null,
        })),
      )
      .toEqual({ lenisActive: null, lenisClass: false, gsapActive: "true" });
    return;
  }

  // Desktop with full motion: Lenis drives real scroll positions.
  await expect
    .poll(() =>
      page.evaluate(() => ({
        lenisActive: document.documentElement.dataset.lenisActive ?? null,
        lenisClass: document.documentElement.classList.contains("lenis"),
        gsapActive: document.documentElement.dataset.gsapActive ?? null,
      })),
    )
    .toEqual({
      lenisActive: "true",
      lenisClass: true,
      gsapActive: "true",
    });

  await page.getByRole("link", { name: "向下滚动至最新文章" }).click();
  await expect
    .poll(() => page.evaluate(() => window.scrollY))
    .toBeGreaterThan(300);

  // The anchor must land below the sticky header, honoring scroll-padding.
  await expect
    .poll(() =>
      page.evaluate(() => {
        const section = document.querySelector("#latest-posts");
        if (!section) return Number.POSITIVE_INFINITY;
        return section.getBoundingClientRect().top;
      }),
    )
    .toBeGreaterThanOrEqual(56);
});
