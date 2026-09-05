/**
 * Marks horizontally-scrollable formulas with data-scroll-hint so CSS can
 * fade the clipped edge(s) (mobile fixes plan §3.2). Pure progressive
 * enhancement: no JS, no hint; measurements re-run once after fonts load
 * and on resize.
 */
export function initializeMathScrollHints(): () => void {
  const containers = Array.from(
    document.querySelectorAll<HTMLElement>("mjx-container"),
  );
  if (containers.length === 0) return () => undefined;

  const events = new AbortController();
  let frame = 0;

  const classify = (container: HTMLElement): void => {
    const maxScroll = container.scrollWidth - container.clientWidth;
    if (maxScroll <= 1) {
      container.removeAttribute("data-scroll-hint");
      return;
    }
    const atStart = container.scrollLeft <= 1;
    const atEnd = container.scrollLeft >= maxScroll - 1;
    container.setAttribute(
      "data-scroll-hint",
      atStart ? "right" : atEnd ? "left" : "both",
    );
  };

  const measureAll = (): void => {
    frame = 0;
    containers.forEach(classify);
  };

  const scheduleMeasure = (): void => {
    if (frame !== 0) return;
    frame = window.requestAnimationFrame(measureAll);
  };

  measureAll();
  document.fonts?.ready.then(scheduleMeasure).catch(() => undefined);

  containers.forEach((container) => {
    container.addEventListener("scroll", scheduleMeasure, {
      passive: true,
      signal: events.signal,
    });
  });
  window.addEventListener("resize", scheduleMeasure, {
    passive: true,
    signal: events.signal,
  });

  return () => {
    events.abort();
    if (frame !== 0) window.cancelAnimationFrame(frame);
  };
}
