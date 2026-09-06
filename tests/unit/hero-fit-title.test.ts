import { describe, expect, it } from "vitest";
import { computeHeroFitFontSize } from "@/lib/client/hero-fit-title";
import type { HeroFitInput } from "@/lib/client/hero-fit-title";

const baseInput: HeroFitInput = {
  baseFontSize: 35.2,
  availableWidth: 288,
  measuredWidth: 350,
  minScale: 0.55,
};

describe("computeHeroFitFontSize", () => {
  it("returns an empty string when the text already fits", () => {
    expect(computeHeroFitFontSize({ ...baseInput, measuredWidth: 280 })).toBe(
      "",
    );
  });

  it("returns an empty string at the exact equality boundary", () => {
    expect(computeHeroFitFontSize({ ...baseInput, measuredWidth: 288 })).toBe(
      "",
    );
  });

  it("scales down proportionally and rounds to two decimals", () => {
    expect(computeHeroFitFontSize(baseInput)).toBe("28.96px");
  });

  it("clamps at the readability floor", () => {
    const next = computeHeroFitFontSize({ ...baseInput, measuredWidth: 1000 });
    expect(next).toBe("19.36px");
  });

  it("returns the exact base size when minScale is 1", () => {
    expect(
      computeHeroFitFontSize({
        ...baseInput,
        measuredWidth: 1000,
        minScale: 1,
      }),
    ).toBe("35.2px");
  });

  it("returns an empty string for degenerate measurements", () => {
    expect(computeHeroFitFontSize({ ...baseInput, measuredWidth: 0 })).toBe("");
    expect(computeHeroFitFontSize({ ...baseInput, availableWidth: 0 })).toBe(
      "",
    );
    expect(computeHeroFitFontSize({ ...baseInput, baseFontSize: 0 })).toBe("");
  });
});
