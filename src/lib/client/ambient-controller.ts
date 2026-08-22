import type { ClientCleanup } from "./lifecycle";

export function initializeAmbientController(): ClientCleanup | void {
  const isEnabled =
    document.documentElement.getAttribute("data-feature-ambient") !== "false";
  if (!isEnabled) return;

  const events = new AbortController();
  const ambientElements = document.querySelectorAll(
    ".ambient-background, [data-ambient]",
  );

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.setAttribute("data-ambient-visible", "true");
      } else {
        entry.target.setAttribute("data-ambient-visible", "false");
      }
    }
  });

  ambientElements.forEach((el) => observer.observe(el));

  return () => {
    events.abort();
    observer.disconnect();
    ambientElements.forEach((el) => el.removeAttribute("data-ambient-visible"));
  };
}
