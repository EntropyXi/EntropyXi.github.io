import type { ClientCleanup } from "./lifecycle";

export function initializeRevealController(): ClientCleanup | void {
  const isEnabled =
    document.documentElement.getAttribute("data-feature-reveal") !== "false";
  if (!isEnabled) return;

  const motionPref = document.documentElement.dataset.motionPreference;
  if (motionPref === "reduced") {
    document.querySelectorAll("[data-reveal]").forEach((el) => {
      el.setAttribute("data-reveal-state", "visible");
    });
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.setAttribute("data-reveal-state", "visible");
          observer.unobserve(entry.target);
        }
      }
    },
    {
      root: null,
      rootMargin: "0px 0px -10% 0px",
      threshold: 0,
    },
  );

  const revealElements = document.querySelectorAll("[data-reveal]");
  revealElements.forEach((el) => {
    if (el.getAttribute("data-reveal-state") !== "visible") {
      observer.observe(el);
    }
  });

  return () => {
    observer.disconnect();
  };
}
