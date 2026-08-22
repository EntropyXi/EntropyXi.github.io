import { describe, expect, it } from "vitest";
import { resolveSiteHeaderState } from "@/lib/client/site-header";

describe("SiteHeader State Resolver", () => {
  it("returns expanded when scroll position is at top (0)", () => {
    expect(resolveSiteHeaderState(0)).toBe("expanded");
  });

  it("returns expanded when scroll position is within threshold (<= 24)", () => {
    expect(resolveSiteHeaderState(12)).toBe("expanded");
    expect(resolveSiteHeaderState(24)).toBe("expanded");
  });

  it("returns compact when scroll position exceeds threshold (> 24)", () => {
    expect(resolveSiteHeaderState(25)).toBe("compact");
    expect(resolveSiteHeaderState(100)).toBe("compact");
  });

  it("handles negative scroll positions gracefully (elastic bounce)", () => {
    expect(resolveSiteHeaderState(-10)).toBe("expanded");
  });
});
