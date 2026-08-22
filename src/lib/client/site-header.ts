export type SiteHeaderState = "expanded" | "compact";

const COMPACT_SCROLL_THRESHOLD = 24;

/**
 * Resolves the visual header state without reading from the DOM.
 * Keeping this decision pure makes the scroll boundary deterministic and
 * straightforward to verify independently from browser events.
 */
export function resolveSiteHeaderState(scrollY: number): SiteHeaderState {
  return Math.max(0, scrollY) > COMPACT_SCROLL_THRESHOLD
    ? "compact"
    : "expanded";
}

/**
 * Synchronizes the site-header state at most once per animation frame.
 * The returned cleanup removes listeners and cancels pending work so Astro
 * view transitions can safely initialize the feature again.
 */
export function initializeSiteHeaderState(): () => void {
  const header = document.querySelector<HTMLElement>("[data-site-header]");
  if (!header) return () => undefined;

  const events = new AbortController();
  let frameId: number | undefined;

  const render = (): void => {
    frameId = undefined;
    header.dataset.headerState = resolveSiteHeaderState(window.scrollY);
  };

  const scheduleRender = (): void => {
    if (frameId !== undefined) return;
    frameId = window.requestAnimationFrame(render);
  };

  window.addEventListener("scroll", scheduleRender, {
    passive: true,
    signal: events.signal,
  });
  window.addEventListener("resize", scheduleRender, {
    passive: true,
    signal: events.signal,
  });

  render();

  return () => {
    events.abort();
    if (frameId !== undefined) window.cancelAnimationFrame(frameId);
  };
}
