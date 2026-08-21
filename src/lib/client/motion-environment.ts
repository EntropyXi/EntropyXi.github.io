export type MotionPreference = "full" | "reduced";

export function initializeMotionEnvironment(): () => void {
  const root = document.documentElement;
  const media = window.matchMedia("(prefers-reduced-motion: reduce)");
  const events = new AbortController();

  const syncPreference = (): void => {
    root.dataset.motionPreference = media.matches ? "reduced" : "full";
  };
  const syncVisibility = (): void => {
    root.dataset.pageVisibility = document.hidden ? "hidden" : "visible";
  };

  syncPreference();
  syncVisibility();
  root.dataset.motion = "ready";

  media.addEventListener("change", syncPreference, {
    signal: events.signal,
  });
  document.addEventListener("visibilitychange", syncVisibility, {
    signal: events.signal,
  });

  return () => {
    events.abort();
    root.removeAttribute("data-motion");
    root.removeAttribute("data-motion-preference");
    root.removeAttribute("data-page-visibility");
  };
}
