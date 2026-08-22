import { expect, test } from "@playwright/test";

test("home page renders the site title and welcome heading", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page).toHaveTitle(/EntropyXi/);
  await expect(
    page.getByRole("heading", { name: /WELCOME TO.*ENTROPYXI BLOG/i }),
  ).toBeVisible();

  // On full-screen hero landing page, scroll down to reveal latest posts section
  await page.getByRole("link", { name: "向下滚动至最新文章" }).click();
  await expect(page.locator(".post-card").first()).toBeVisible();
});

test("404 page renders without redirecting to a broken layout", async ({
  page,
}) => {
  await page.goto("/not-found/");
  await expect(page.getByRole("heading", { name: "页面未找到" })).toBeVisible();
  await expect(page.getByRole("link", { name: "返回首页" })).toBeVisible();
});
