import { gsap } from "gsap";

const FOLLOW_DURATION = 0.4;
const RELEASE_DURATION = 0.8;

function readMagneticFactor(element: HTMLElement): number {
  const value = Number.parseFloat(
    getComputedStyle(element).getPropertyValue("--magnetic-factor"),
  );
  return Number.isFinite(value) && value > 0 ? value : 0.2;
}

/**
 * Magnetic hover with a springy follow (plan §4.6), loaded only when the
 * motion stack is gated on. GSAP owns the transform inline while motion.css
 * disables the CSS transitions for [data-magnetic] under
 * html[data-motion-gsap], so per-frame quickTo updates never fight a
 * transition. Returns a teardown that restores the CSS path's namespace.
 */
export function initializeMagneticSpring(events: AbortController): () => void {
  const root = document.documentElement;
  root.dataset.motionGsap = "true";

  document
    .querySelectorAll<HTMLElement>("[data-magnetic]")
    .forEach((element) => {
      const followX = gsap.quickTo(element, "x", {
        duration: FOLLOW_DURATION,
        ease: "power3.out",
      });
      const followY = gsap.quickTo(element, "y", {
        duration: FOLLOW_DURATION,
        ease: "power3.out",
      });
      const releaseX = gsap.quickTo(element, "x", {
        duration: RELEASE_DURATION,
        ease: "elastic.out(1, 0.5)",
      });
      const releaseY = gsap.quickTo(element, "y", {
        duration: RELEASE_DURATION,
        ease: "elastic.out(1, 0.5)",
      });

      let moveEvents: AbortController | null = null;

      element.addEventListener(
        "pointerenter",
        () => {
          element.setAttribute("data-magnetic-state", "active");
          const factor = readMagneticFactor(element);
          moveEvents = new AbortController();
          element.addEventListener(
            "pointermove",
            (moveEvent: PointerEvent) => {
              const rect = element.getBoundingClientRect();
              const deltaX = moveEvent.clientX - (rect.left + rect.width / 2);
              const deltaY = moveEvent.clientY - (rect.top + rect.height / 2);
              followX(deltaX * factor);
              followY(deltaY * factor);
            },
            { signal: moveEvents.signal },
          );
        },
        { signal: events.signal },
      );

      element.addEventListener(
        "pointerleave",
        () => {
          moveEvents?.abort();
          moveEvents = null;
          releaseX();
          releaseY();
          element.setAttribute("data-magnetic-state", "inactive");
        },
        { signal: events.signal },
      );
    });

  return () => {
    root.removeAttribute("data-motion-gsap");
  };
}
