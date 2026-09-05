export interface MotionGateInput {
  motionPreference: string | undefined;
  hasFinePointer: boolean;
  gsapFlag: string | null;
  zoom: number;
}

export interface MotionGateDecision {
  /** GSAP choreography (hero, parallax, batch reveal) may run. */
  gsap: boolean;
  /** Lenis smooth scroll may run; requires full motion and a fine pointer. */
  lenis: boolean;
}

/**
 * Pure capability gate for the motion stack (plan §4.0). Keyboard-only and
 * touch environments keep GSAP-driven scroll work but never get Lenis or
 * magnetic updates; reduced-motion and the feature flag disable everything.
 * A CSS zoom on the root element (the zoom-200 e2e scenario) disables Lenis
 * because its wheel multipliers misbehave under zoom.
 */
export function resolveMotionGate(input: MotionGateInput): MotionGateDecision {
  const gsap =
    input.gsapFlag !== "false" && input.motionPreference !== "reduced";
  const lenis = gsap && input.hasFinePointer && input.zoom === 1;
  return { gsap, lenis };
}

export function readMotionGateInput(root: HTMLElement): MotionGateInput {
  const zoomValue = Number.parseFloat(getComputedStyle(root).zoom);
  return {
    motionPreference: root.dataset.motionPreference,
    hasFinePointer: window.matchMedia("(hover: hover) and (pointer: fine)")
      .matches,
    gsapFlag: root.getAttribute("data-feature-gsap"),
    zoom: Number.isFinite(zoomValue) ? zoomValue : 1,
  };
}
