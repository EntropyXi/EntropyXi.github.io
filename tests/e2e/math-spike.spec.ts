import { expect, test } from '@playwright/test';

test('math spike page renders LaTeX with self-hosted MathJax', async ({ page }) => {
  await page.goto('/dev/math-spike/');

  await page.waitForFunction(() => {
    const mathJax = (window as unknown as { MathJax?: { startup?: { promise?: Promise<unknown> } } })
      .MathJax;
    return mathJax?.startup?.promise !== undefined;
  });

  await page.waitForFunction(() => {
    const containers = document.querySelectorAll('mjx-container');
    return containers.length > 0;
  });

  await expect(page.locator('mjx-container').first()).toBeVisible();

  const mathText = await page.locator('.spike-body').innerText();
  expect(mathText).not.toContain('$$');
  expect(mathText).not.toContain('\\begin');

  const layout = await page.evaluate(() => ({
    viewportWidth: window.innerWidth,
    documentWidth: document.documentElement.scrollWidth,
  }));
  expect(layout.documentWidth).toBeLessThanOrEqual(layout.viewportWidth);
});
