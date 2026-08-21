export function calculateReadingProgress(
  scrollPosition: number,
  documentHeight: number,
  viewportHeight: number,
): number {
  const scrollableHeight = documentHeight - viewportHeight;
  if (scrollableHeight <= 0) return 0;
  return Math.min(100, Math.max(0, (scrollPosition / scrollableHeight) * 100));
}

export function initializeReadingProgress(): () => void {
  const bar = document.getElementById("reading-progress");
  if (!(bar instanceof HTMLElement)) return () => undefined;

  const events = new AbortController();
  let animationFrame = 0;

  const render = (): void => {
    animationFrame = 0;
    const progress = calculateReadingProgress(
      window.scrollY,
      document.documentElement.scrollHeight,
      window.innerHeight,
    );
    bar.style.width = `${progress}%`;
    bar.setAttribute("aria-valuenow", String(Math.round(progress)));
  };

  const schedule = (): void => {
    if (animationFrame !== 0 || document.hidden) return;
    animationFrame = window.requestAnimationFrame(render);
  };

  window.addEventListener("scroll", schedule, {
    passive: true,
    signal: events.signal,
  });
  window.addEventListener("resize", schedule, {
    passive: true,
    signal: events.signal,
  });
  document.addEventListener(
    "visibilitychange",
    () => {
      if (!document.hidden) schedule();
    },
    { signal: events.signal },
  );
  render();

  return () => {
    events.abort();
    if (animationFrame !== 0) window.cancelAnimationFrame(animationFrame);
  };
}
