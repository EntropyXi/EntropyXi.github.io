export function initializeBackToTop(): () => void {
  const button = document.getElementById("back-to-top");
  if (!(button instanceof HTMLAnchorElement)) return () => undefined;

  const events = new AbortController();
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let animationFrame = 0;

  const render = (): void => {
    animationFrame = 0;
    button.classList.toggle("visible", window.scrollY > 300);
  };

  const schedule = (): void => {
    if (animationFrame !== 0) return;
    animationFrame = window.requestAnimationFrame(render);
  };

  button.addEventListener(
    "click",
    (event) => {
      event.preventDefault();
      window.scrollTo({
        top: 0,
        behavior: reducedMotion.matches ? "auto" : "smooth",
      });
    },
    { signal: events.signal },
  );
  window.addEventListener("scroll", schedule, {
    passive: true,
    signal: events.signal,
  });
  render();

  return () => {
    events.abort();
    if (animationFrame !== 0) window.cancelAnimationFrame(animationFrame);
  };
}
