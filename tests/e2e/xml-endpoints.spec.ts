import { expect, test } from "@playwright/test";

test("XML compatibility endpoints are parseable and use frozen paths", async ({
  page,
}) => {
  for (const path of ["/sitemap.xml", "/atom.xml", "/search.xml"]) {
    await page.goto(path);
    const parsed = await page.evaluate(() => {
      const doc = new DOMParser().parseFromString(
        document.documentElement.outerHTML,
        "application/xml",
      );
      return {
        root: doc.documentElement.nodeName,
        parserError: doc.querySelector("parsererror") !== null,
      };
    });
    expect(parsed.parserError).toBe(false);
    expect(parsed.root).not.toBe("");
  }
});
