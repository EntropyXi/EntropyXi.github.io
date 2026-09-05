import type { ClientCleanup } from "./lifecycle";
import { readMotionGateInput, resolveMotionGate } from "./motion/gsap-gate";

function setupCssMagnetic(events: AbortController): ClientCleanup {
  const root = document.documentElement;
  let rAfId: number | null = null;
  let currentX = 0;
  let currentY = 0;
  let isMoving = false;
  let activeMagneticElement: HTMLElement | null = null;
  let activeMagneticCenterX = 0;
  let activeMagneticCenterY = 0;

  const updatePointer = () => {
    root.style.setProperty("--pointer-x", `${currentX}px`);
    root.style.setProperty("--pointer-y", `${currentY}px`);

    if (activeMagneticElement) {
      const deltaX = currentX - activeMagneticCenterX;
      const deltaY = currentY - activeMagneticCenterY;
      activeMagneticElement.style.setProperty("--magnetic-x", `${deltaX}px`);
      activeMagneticElement.style.setProperty("--magnetic-y", `${deltaY}px`);
    }

    isMoving = false;
  };

  const onPointerMove = (e: PointerEvent) => {
    // Abort on touch interactions
    if (e.pointerType === "touch" || e.pointerType === "pen") {
      return;
    }

    currentX = e.clientX;
    currentY = e.clientY;

    if (!isMoving) {
      isMoving = true;
      rAfId = requestAnimationFrame(updatePointer);
    }
  };

  document.addEventListener("pointermove", onPointerMove, {
    signal: events.signal,
    passive: true,
  });

  const magneticElements = document.querySelectorAll("[data-magnetic]");

  magneticElements.forEach((el) => {
    const htmlEl = el as HTMLElement;
    htmlEl.addEventListener(
      "pointerenter",
      () => {
        const rect = htmlEl.getBoundingClientRect();
        activeMagneticCenterX = rect.left + rect.width / 2;
        activeMagneticCenterY = rect.top + rect.height / 2;
        activeMagneticElement = htmlEl;
        htmlEl.setAttribute("data-magnetic-state", "active");
      },
      { signal: events.signal },
    );

    htmlEl.addEventListener(
      "pointerleave",
      () => {
        if (activeMagneticElement === htmlEl) {
          activeMagneticElement = null;
        }
        htmlEl.setAttribute("data-magnetic-state", "inactive");
        htmlEl.style.removeProperty("--magnetic-x");
        htmlEl.style.removeProperty("--magnetic-y");
      },
      { signal: events.signal },
    );
  });

  return () => {
    events.abort();
    if (rAfId !== null) {
      cancelAnimationFrame(rAfId);
    }
    root.style.removeProperty("--pointer-x");
    root.style.removeProperty("--pointer-y");
    magneticElements.forEach((el) => {
      const htmlEl = el as HTMLElement;
      htmlEl.removeAttribute("data-magnetic-state");
      htmlEl.style.removeProperty("--magnetic-x");
      htmlEl.style.removeProperty("--magnetic-y");
    });
  };
}

export function initializePointerController(): ClientCleanup | void {
  const root = document.documentElement;
  const isEnabled = root.getAttribute("data-feature-magnetic") !== "false";
  if (!isEnabled) return;

  const motionPref = root.dataset.motionPreference;
  if (motionPref === "reduced") return;

  // Initialize magnetic behavior only for fine pointers and abort on touch
  const isFinePointer = window.matchMedia(
    "(hover: hover) and (pointer: fine)",
  ).matches;
  if (!isFinePointer) return;

  const events = new AbortController();

  // The CSS path installs immediately so the effect works during the chunk
  // fetch; when the spring path takes over it owns the transform inline and
  // the CSS transition is disabled via data-motion-gsap.
  let cleanup: ClientCleanup = setupCssMagnetic(events);

  if (resolveMotionGate(readMotionGateInput(root)).gsap) {
    // Springy GSAP path; falls back to the CSS variables path if the
    // chunk cannot be loaded.
    void import("./motion/magnetic-spring")
      .then(({ initializeMagneticSpring }) => {
        const teardown = initializeMagneticSpring(events);
        const previousCleanup = cleanup;
        cleanup = () => {
          teardown();
          previousCleanup();
        };
      })
      .catch(() => {
        // Chunk failure: the CSS path above keeps the effect working.
      });
  }

  return () => {
    cleanup();
  };
}
