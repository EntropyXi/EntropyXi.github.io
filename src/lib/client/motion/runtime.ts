import { readMotionGateInput, resolveMotionGate } from "./gsap-gate";

function scheduleIdle(callback: () => void): void {
  if (typeof window.requestIdleCallback === "function") {
    window.requestIdleCallback(() => callback());
  } else {
    window.setTimeout(callback, 0);
  }
}

/**
 * Boots the motion stack behind the capability gate (plan §4.0). Runs once
 * per document: the init marker lives on <html> and is intentionally not
 * relayed across view transitions, so each new page re-evaluates the gate.
 * The returned cleanup is a no-op on purpose — Lenis is a module singleton
 * that survives swaps, and per-page GSAP work owns its own gsap.context.
 */
export function initializeMotionRuntime(): () => void {
  const root = document.documentElement;
  if (root.dataset.motionRuntimeInit === "true") return () => undefined;
  root.dataset.motionRuntimeInit = "true";

  const decision = resolveMotionGate(readMotionGateInput(root));
  if (!decision.gsap) return () => undefined;

  scheduleIdle(() => {
    void boot(decision.lenis);
  });

  return () => undefined;
}

async function boot(lenisEnabled: boolean): Promise<void> {
  if (lenisEnabled) {
    const { initializeLenis } = await import("./lenis-controller");
    initializeLenis();
  }

  if (document.documentElement.hasAttribute("data-hero-pending")) {
    const { runHeroChoreography } = await import("./hero-choreography");
    runHeroChoreography();
  }
  // Phase 4 mounts scroll narrative (parallax + batch reveal) here.
}
