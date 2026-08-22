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
    name: "EntropyXi 的技术笔记",
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
