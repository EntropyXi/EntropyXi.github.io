import { expect, test } from "@playwright/test";

interface WallpaperElement extends HTMLElement {
  __motionOverhaulCheck?: string;
}

test.describe("view transitions preserve the persistent wallpaper across navigation", () => {
  // Post cards are magnetic (they follow the pointer), so Playwright's
  // click-stability check never settles on them; below-fold cards are also
  // visibility:hidden until scrolled into view. Navigate via the header nav
  // with the magnetic feature disabled — node-identity persistence is
  // orthogonal to both effects.
  test.beforeEach(async ({ context }) => {
    await context.addInitScript(() => {
      localStorage.setItem("entropyxi-feature-magnetic", "false");
    });
  });

  async function markWallpaper(page: import("@playwright/test").Page): Promise<void> {
    await page.evaluate(() => {
      const wallpaper = document.querySelector(
        ".ambient-background",
      ) as WallpaperElement | null;
      if (!wallpaper) throw new Error("ambient background not found");
      wallpaper.__motionOverhaulCheck = "persistent-node";
    });
  }

  async function navigateToArchives(page: import("@playwright/test").Page): Promise<void> {
    await page.click('.desktop-nav .nav-link[href="/archives/"]');
    await page.waitForURL(/\/archives\/$/);
    await page.waitForLoadState("load");
  }

  test("ambient background node identity survives client-side navigation", async ({
    page,
  }) => {
    await page.goto("/");
    await markWallpaper(page);
    await navigateToArchives(page);

    const persistedMark = await page.evaluate(() => {
      const wallpaper = document.querySelector(
        ".ambient-background",
      ) as WallpaperElement | null;
      return wallpaper?.__motionOverhaulCheck ?? null;
    });
    expect(persistedMark).toBe("persistent-node");
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await expect(page.locator(".section-title")).toContainText("全站归档");
  });

  test("navigation works identically with the gsap feature flag disabled", async ({
    page,
  }) => {
    await page.addInitScript(() => {
      localStorage.setItem("entropyxi-feature-gsap", "false");
    });
    await page.goto("/");
    await markWallpaper(page);
    await navigateToArchives(page);

    const persistedMark = await page.evaluate(() => {
      const wallpaper = document.querySelector(
        ".ambient-background",
      ) as WallpaperElement | null;
      return wallpaper?.__motionOverhaulCheck ?? null;
    });
    expect(persistedMark).toBe("persistent-node");
    await expect(page.locator(".ambient-background")).toBeAttached();
  });
});
