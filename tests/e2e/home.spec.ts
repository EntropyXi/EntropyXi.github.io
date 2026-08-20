import { expect, test } from '@playwright/test';

test('home page renders the site title and welcome heading', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/EntropyXi/);
  await expect(page.getByRole('heading', { name: '欢迎来到 EntropyXi 的博客' })).toBeVisible();
});

test('404 page renders without redirecting to a broken layout', async ({ page }) => {
  await page.goto('/not-found/');
  await expect(page.getByRole('heading', { name: '页面未找到' })).toBeVisible();
  await expect(page.getByRole('link', { name: '返回首页' })).toBeVisible();
});
