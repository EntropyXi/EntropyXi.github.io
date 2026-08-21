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

test("mobile drawer manages focus and restores it after Escape", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  const trigger = page.getByRole("button", { name: "打开导航菜单" });
  const close = page.getByRole("button", { name: "关闭菜单" });

  await trigger.click();
  await expect(close).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(trigger).toBeFocused();
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
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

  test("home and article content remain visible without JavaScript", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "EntropyXi 的技术笔记" }),
    ).toBeVisible();
    await expect(page.locator(".post-card").first()).toBeVisible();

    await page.goto(
      "/2026/05/17/深度学习/流匹配与扩散模型/体系/1.%20从SDE开始/",
    );
    await expect(page.locator("article")).toBeVisible();
    await expect(page.locator("mjx-container").first()).toBeVisible();
  });
});
