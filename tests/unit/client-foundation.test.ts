import { describe, expect, it } from "vitest";
import { resolveTheme } from "@/lib/client/theme";
import { calculateReadingProgress } from "@/lib/client/reading-progress";

describe("resolveTheme", () => {
  it("prefers a stored dark theme over a light system preference", () => {
    expect(resolveTheme("dark", false)).toBe("dark");
  });

  it("prefers a stored light theme over a dark system preference", () => {
    expect(resolveTheme("light", true)).toBe("light");
  });

  it("falls back to the dark system preference when nothing is stored", () => {
    expect(resolveTheme(null, true)).toBe("dark");
    expect(resolveTheme(undefined, true)).toBe("dark");
  });

  it("falls back to the light system preference when nothing is stored", () => {
    expect(resolveTheme(null, false)).toBe("light");
    expect(resolveTheme(undefined, false)).toBe("light");
  });

  it("treats unknown stored values as invalid and falls back to the system", () => {
    expect(resolveTheme("system", true)).toBe("dark");
    expect(resolveTheme("system", false)).toBe("light");
  });

  it("treats empty strings and non-string values as invalid", () => {
    expect(resolveTheme("", true)).toBe("dark");
    expect(resolveTheme(42, false)).toBe("light");
  });

  it("is case-sensitive when matching stored themes", () => {
    expect(resolveTheme("DARK", false)).toBe("light");
    expect(resolveTheme("Light", true)).toBe("dark");
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
