import { expect, test } from "@playwright/test";

test.describe("theme behavior contract", () => {
  test.beforeEach(async ({ page }) => {
    await page.emulateMedia({ colorScheme: "light" });
  });

  test("first visit follows the system theme without duplicate ids", async ({
    page,
  }) => {
    await page.goto("/");

    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
    await expect(page.locator("[id]")).toHaveCount(
      await page.locator("[id]").evaluateAll((elements) => {
        return new Set(elements.map((element) => element.id)).size;
      }),
    );
  });

  test("the visible theme control toggles and persists the explicit choice", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");

    const toggle = page.locator(".theme-toggle-btn:visible");
    await expect(toggle).toHaveCount(1);
    await toggle.click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  });

  test("storage failure falls back to the system theme", async ({ page }) => {
    await page.addInitScript(() => {
      Storage.prototype.getItem = () => {
        throw new DOMException("Storage access denied", "SecurityError");
      };
    });
    await page.goto("/");

    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  });

  test("repeated lifecycle events never duplicate the theme listener", async ({
    page,
  }) => {
    await page.goto("/");
    await page.evaluate(() => {
      document.dispatchEvent(new Event("astro:page-load"));
      document.dispatchEvent(new Event("astro:page-load"));
    });

    await page.locator(".theme-toggle-btn:visible").click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  });

  test("system changes apply only until the visitor makes a choice", async ({
    page,
  }) => {
    await page.goto("/");
    await page.emulateMedia({ colorScheme: "dark" });
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

    await page.locator(".theme-toggle-btn:visible").click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
    await page.emulateMedia({ colorScheme: "light" });
    await page.emulateMedia({ colorScheme: "dark" });
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  });

  test("before-swap removes listeners and page-load restores them", async ({
    page,
  }) => {
    await page.goto("/");
    const toggle = page.locator(".theme-toggle-btn:visible");

    await page.evaluate(() => {
      document.dispatchEvent(new Event("astro:before-swap"));
    });
    await toggle.click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");

    await page.evaluate(() => {
      document.dispatchEvent(new Event("astro:page-load"));
    });
    await toggle.click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  });
});

test("mobile drawer traps focus and restores it after dismissal", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  const trigger = page.getByRole("button", { name: "打开导航菜单" });
  const drawer = page.locator("#mobile-drawer");
  const close = page.getByRole("button", { name: "关闭菜单" });

  await expect(drawer).toHaveAttribute("role", "dialog");
  await expect(drawer).toHaveAttribute("aria-modal", "true");
  await expect(drawer).toHaveAttribute("aria-hidden", "true");
  await trigger.click();
  await expect(page.getByRole("dialog", { name: "导航目录" })).toBeVisible();
  await expect(drawer).toHaveAttribute("aria-hidden", "false");
  await expect(close).toBeFocused();

  const lastLink = page.locator("#mobile-drawer a[href]").last();
  await lastLink.focus();
  await page.keyboard.press("Tab");
  await expect(close).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(lastLink).toBeFocused();

  await page.keyboard.press("Escape");
  await expect(trigger).toBeFocused();
  await expect(trigger).toHaveAttribute("aria-expanded", "false");

  await trigger.click();
  await page
    .locator("#mobile-drawer-backdrop")
    .click({ position: { x: 20, y: 20 } });
  await expect(trigger).toBeFocused();
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
});

test("skip link is the first keyboard target and focuses main content", async ({
  page,
}) => {
  await page.goto("/");

  await page.keyboard.press("Tab");
  const skipLink = page.getByRole("link", { name: "跳至主要内容" });
  await expect(skipLink).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#main-content")).toBeFocused();
});

test("site header switches between expanded and compact states", async ({
  page,
}) => {
  await page.goto("/");
  const header = page.locator("[data-site-header]");

  await expect(header).toHaveAttribute("data-header-state", "expanded");
  await page.evaluate(() => window.scrollTo(0, 200));
  await expect(header).toHaveAttribute("data-header-state", "compact");
  await page.evaluate(() => window.scrollTo(0, 0));
  await expect(header).toHaveAttribute("data-header-state", "expanded");
});

test("ambient background exposes isolated non-interactive layers", async ({
  page,
}) => {
  await page.goto("/");
  const background = page.locator(".ambient-background");

  await expect(background).toHaveAttribute("aria-hidden", "true");
  await expect(background).toHaveCSS("pointer-events", "none");
  await expect(background.locator("[data-layer]")).toHaveCount(4);
  await expect(background.locator('[data-layer="grid"]')).toHaveCount(1);
  await expect(background.locator('[data-layer="scanline"]')).toHaveCount(1);
  await expect(background.locator('[data-layer="glow"]')).toHaveCount(1);
  const flow = background.locator('[data-layer="flow"]');
  await expect(flow).toHaveCount(1);
  await expect(flow.locator("path")).toHaveCount(3);
  const strokes = await flow
    .locator("path")
    .evaluateAll((paths) => paths.map((path) => getComputedStyle(path).stroke));
  expect(strokes).not.toContain("none");
});

test("search returns indexed results and clears stale results", async ({
  page,
}) => {
  await page.goto("/search/");
  const input = page.getByRole("searchbox", { name: "搜索文章" });
  const status = page.locator("#search-status");

  await input.fill("SDE");
  await expect(status).toContainText(/找到 \d+ 篇相关文章/);
  await expect(page.locator("#search-results li").first()).toBeVisible();

  await input.fill("");
  await expect(status).toHaveText("");
  await expect(page.locator("#search-results li")).toHaveCount(0);
});

test("code copy reports clipboard success and failure", async ({ page }) => {
  await page.goto(
    "/2026/02/08/深度学习/线性回归/整理一下softmax回归实现中训练部分代码的思路/",
  );
  const copyButton = page.getByRole("button", { name: "复制代码" }).first();
  await expect(copyButton).toBeVisible();

  await page.evaluate(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: () => Promise.resolve() },
    });
  });
  await copyButton.click();
  await expect(copyButton).toHaveText("已复制");

  await page.reload();
  const reloadedButton = page.getByRole("button", { name: "复制代码" }).first();
  await page.evaluate(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: () => Promise.reject(new Error("denied")) },
    });
  });
  await reloadedButton.click();
  await expect(reloadedButton).toHaveText("复制失败");
});

test.describe("progressive enhancement contract", () => {
  test.use({ javaScriptEnabled: false });

  test("desktop keeps the primary navigation without a duplicate fallback", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(
      page.getByRole("navigation", { name: "主导航" }),
    ).toBeVisible();
    await expect(
      page.getByRole("navigation", { name: "无脚本主导航" }),
    ).toBeHidden();
  });

  test("home and article content remain visible without JavaScript", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "EntropyXi 的技术笔记" }),
    ).toBeVisible();
    await expect(page.locator(".post-card").first()).toBeVisible();
    await expect(
      page.getByRole("navigation", { name: "无脚本主导航" }),
    ).toBeVisible();
    await expect(
      page.getByRole("link", { name: "首页", exact: true }),
    ).toHaveAttribute("aria-current", "page");
    await expect(
      page.getByRole("button", { name: "打开导航菜单" }),
    ).toBeHidden();
    await expect(page.locator(".theme-toggle-btn")).toHaveCount(2);
    await expect(page.locator(".theme-toggle-btn").first()).toBeHidden();
    await expect(page.locator(".theme-toggle-btn").last()).toBeHidden();

    await page.goto(
      "/2026/05/17/深度学习/流匹配与扩散模型/体系/1.%20从SDE开始/",
    );
    await expect(page.locator("article")).toBeVisible();
    await expect(page.locator("mjx-container").first()).toBeVisible();
  });
});
