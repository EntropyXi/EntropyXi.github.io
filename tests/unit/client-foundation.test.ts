import { describe, expect, it } from "vitest";
import { resolveTheme } from "@/lib/client/theme";
import { calculateReadingProgress } from "@/lib/client/reading-progress";
import { resolveSiteHeaderState } from "@/lib/client/site-header";

describe("resolveTheme", () => {
  it("always returns dark theme as the fixed mode", () => {
    expect(resolveTheme("dark", false)).toBe("dark");
    expect(resolveTheme("light", true)).toBe("dark");
    expect(resolveTheme(null, false)).toBe("dark");
    expect(resolveTheme(undefined, true)).toBe("dark");
  });
});

describe("calculateReadingProgress", () => {
  it("returns 0 on a page that cannot scroll", () => {
    expect(calculateReadingProgress(0, 800, 800)).toBe(0);
    expect(calculateReadingProgress(500, 600, 800)).toBe(0);
  });

  it("returns 0 at the top of a scrollable page", () => {
    expect(calculateReadingProgress(0, 1200, 800)).toBe(0);
  });

  it("returns 100 at the bottom of a scrollable page", () => {
    expect(calculateReadingProgress(400, 1200, 800)).toBe(100);
  });

  it("computes the proportion of scrollable height", () => {
    expect(calculateReadingProgress(200, 1200, 800)).toBe(50);
    expect(calculateReadingProgress(100, 1200, 800)).toBe(25);
  });

  it("clamps negative scroll positions to 0", () => {
    expect(calculateReadingProgress(-50, 1200, 800)).toBe(0);
  });

  it("clamps scroll positions past the end to 100", () => {
    expect(calculateReadingProgress(800, 1200, 800)).toBe(100);
  });
});

describe("resolveSiteHeaderState", () => {
  it("keeps the header expanded at and above the document origin", () => {
    expect(resolveSiteHeaderState(-10)).toBe("expanded");
    expect(resolveSiteHeaderState(0)).toBe("expanded");
  });

  it("keeps the header expanded through the scroll threshold", () => {
    expect(resolveSiteHeaderState(24)).toBe("expanded");
  });

  it("compacts the header after the scroll threshold", () => {
    expect(resolveSiteHeaderState(25)).toBe("compact");
    expect(resolveSiteHeaderState(400)).toBe("compact");
  });
});
