import { test } from "@playwright/test";

test("probe: live computed hero font across viewport heights", async ({
  browser,
}) => {
  const contexts = [
    { name: "390x844 (no toolbar)", width: 390, height: 844 },
    { name: "390x660 (chrome toolbar)", width: 390, height: 660 },
    { name: "360x640 (chrome toolbar)", width: 360, height: 640 },
    { name: "320x568 (small)", width: 320, height: 568 },
  ];
  for (const vp of contexts) {
    const context = await browser.newContext({
      viewport: { width: vp.width, height: vp.height },
      hasTouch: true,
      isMobile: true,
    });
    const page = await context.newPage();
    await page.goto("https://entropyxi.github.io/", { waitUntil: "load" });
    await page.waitForTimeout(1500);
    const state = await page.evaluate(() => {
      const title = document.querySelector(
        ".hero-welcome-title",
      ) as HTMLElement | null;
      const line = document.querySelector(".welcome-line-2");
      return {
        computedFontSize: title
          ? getComputedStyle(title).fontSize
          : null,
        inlineFontSize: title?.style.fontSize ?? null,
        heroFit: title?.dataset.heroFit ?? null,
        lineWidth: Math.round(line?.getBoundingClientRect().width ?? 0),
        containerWidth: Math.round(
          document.querySelector(".hero-content-container")?.getBoundingClientRect()
            .width ?? 0,
        ),
      };
    });
    console.log(`FONT[${vp.name}]:`, JSON.stringify(state));
    await context.close();
  }
});
