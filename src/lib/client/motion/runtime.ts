import { readMotionGateInput, resolveMotionGate } from "./gsap-gate";

function scheduleIdle(callback: () => void): void {
  if (typeof window.requestIdleCallback === "function") {
    window.requestIdleCallback(() => callback());
  } else {
    window.setTimeout(callback, 0);
  }
}

let narrativeCleanup: (() => void) | null = null;

/**
 * Boots the motion stack behind the capability gate (plan §4.0). Runs once
 * per document: the init marker lives on <html> and is intentionally not
 * relayed across view transitions, so each new page re-evaluates the gate.
 * The cleanup reverts the per-document gsap.context; Lenis itself is a
 * module singleton that survives swaps.
 */
export function initializeMotionRuntime(): () => void {
  const root = document.documentElement;
  if (root.dataset.motionRuntimeInit === "true") {
    return () => undefined;
  }
  root.dataset.motionRuntimeInit = "true";

  scheduleIdle(() => {
    void boot(root);
  });

  return () => {
    narrativeCleanup?.();
    narrativeCleanup = null;
    root.removeAttribute("data-motion-runtime-init");
  };
}

async function boot(root: HTMLElement): Promise<void> {
  try {
    // The gate is evaluated at idle time, not at registration: styles may
    // still be settling (e.g. a zoom applied on DOMContentLoaded) and idle
    // callbacks run after either has taken effect.
    const decision = resolveMotionGate(readMotionGateInput(root));
    if (!decision.gsap) return;

    if (decision.lenis) {
      const { initializeLenis } = await import("./lenis-controller");
      initializeLenis();
    }

    root.dataset.gsapActive = "true";

    if (root.hasAttribute("data-hero-pending")) {
      const { runHeroChoreography } = await import("./hero-choreography");
      runHeroChoreography();
    }

    const { initializeScrollNarrative } = await import("./scroll-narrative");
    narrativeCleanup = initializeScrollNarrative() ?? null;
  } catch {
    // Chunk loading failed: restore the CSS-only visibility contract so no
    // content stays hidden behind the GSAP takeover.
    root.removeAttribute("data-gsap-active");
    root.removeAttribute("data-hero-pending");
    document
      .querySelectorAll("[data-reveal]")
      .forEach((element) =>
        element.setAttribute("data-reveal-state", "visible"),
      );
  }
}
