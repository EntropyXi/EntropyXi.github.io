import { describe, expect, it } from "vitest";
import { resolveMotionGate } from "@/lib/client/motion/gsap-gate";
import type { MotionGateInput } from "@/lib/client/motion/gsap-gate";

const baseInput: MotionGateInput = {
  motionPreference: "full",
  hasFinePointer: true,
  gsapFlag: "true",
  zoom: 1,
};

describe("resolveMotionGate", () => {
  it("enables the full stack for a fine pointer with full motion", () => {
    expect(resolveMotionGate(baseInput)).toEqual({ gsap: true, lenis: true });
  });

  it("disables everything under reduced motion", () => {
    expect(
      resolveMotionGate({ ...baseInput, motionPreference: "reduced" }),
    ).toEqual({ gsap: false, lenis: false });
  });

  it("disables everything when the feature flag is off", () => {
    expect(resolveMotionGate({ ...baseInput, gsapFlag: "false" })).toEqual({
      gsap: false,
      lenis: false,
    });
  });

  it("keeps gsap but disables lenis for touch environments", () => {
    expect(resolveMotionGate({ ...baseInput, hasFinePointer: false })).toEqual({
      gsap: true,
      lenis: false,
    });
  });

  it("keeps gsap but disables lenis under css zoom", () => {
    expect(resolveMotionGate({ ...baseInput, zoom: 2 })).toEqual({
      gsap: true,
      lenis: false,
    });
  });

  it("treats a missing flag attribute as enabled", () => {
    expect(resolveMotionGate({ ...baseInput, gsapFlag: null })).toEqual({
      gsap: true,
      lenis: true,
    });
  });
});
