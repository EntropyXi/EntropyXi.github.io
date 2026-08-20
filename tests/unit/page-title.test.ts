import { describe, expect, it } from "vitest";
import { SITE_TITLE } from "@/data/site";
import { buildPageTitle } from "@/lib/seo/page-title";

describe("buildPageTitle", () => {
  it("uses the site title when the page title is empty", () => {
    expect(buildPageTitle("")).toBe(SITE_TITLE);
  });

  it("combines a page title with the site title", () => {
    expect(buildPageTitle("关于")).toBe(`关于 | ${SITE_TITLE}`);
  });

  it("trims surrounding whitespace before combining", () => {
    expect(buildPageTitle("  关于  ")).toBe(`关于 | ${SITE_TITLE}`);
  });
});
