const MIN_SCALE = 0.55;

export interface HeroFitInput {
  /** CSS base font-size of the title (inline styles cleared). */
  baseFontSize: number;
  /** Width the title must fit into (its layout container). */
  availableWidth: number;
  /** Widest rendered line width under the current font stack. */
  measuredWidth: number;
  /** Readability floor as a fraction of the base font-size. */
  minScale: number;
}

/**
 * Font-size that fits the measured title into the available width, or ""
 * when the CSS base already fits. Pure and platform-agnostic: the measured
 * width comes from whatever font the runtime actually resolved, so Android
 * fallbacks (Roboto ≈ 350px) shrink exactly where desktop Impact (254px)
 * would never need to.
 */
export function computeHeroFitFontSize(input: HeroFitInput): string {
  const { baseFontSize, availableWidth, measuredWidth, minScale } = input;
  if (measuredWidth <= availableWidth) return "";
  if (baseFontSize <= 0 || availableWidth <= 0 || measuredWidth <= 0) {
    return "";
  }
  const scale = Math.max(availableWidth / measuredWidth, minScale);
  const next = Math.round(baseFontSize * scale * 100) / 100;
  return `${next}px`;
}

/**
 * Fits the hero welcome title into its layout container on every font
 * stack (mobile fixes follow-up: Android has none of the condensed fonts
 * in the title stack, so `nowrap` lines overflowed into the
 * `overflow: hidden` clip). Re-entrant by contract — every activate
 * re-queries and re-binds; the previous listeners die with the shared
 * AbortController, so there is intentionally no mount guard (a guard would
 * kill the resize/fonts.ready re-measure after the second activate).
 */
export function initializeHeroFitTitle(): () => void {
  const section = document.querySelector<HTMLElement>(".hero-fullscreen");
  const title = section?.querySelector<HTMLElement>(".hero-welcome-title");
  const container = section?.querySelector<HTMLElement>(
    ".hero-content-container",
  );
  // `.hero-content-container` is the only valid width reference: it has an
  // explicit width and does not shrink-wrap the title. The brand block is
  // fit-content and would make measured/available identically 1.
  if (!section || !title || !container) return () => undefined;

  const events = new AbortController();
  let frame = 0;

  const fit = (): void => {
    frame = 0;
    const previous = title.style.fontSize;
    // Reset to the CSS base before measuring so scales never compound.
    title.style.fontSize = "";
    let measuredWidth = 0;
    title.querySelectorAll<HTMLElement>(":scope > span").forEach((line) => {
      measuredWidth = Math.max(measuredWidth, line.scrollWidth);
    });
    const baseFontSize = parseFloat(getComputedStyle(title).fontSize);
    const next = computeHeroFitFontSize({
      baseFontSize,
      availableWidth: container.clientWidth,
      measuredWidth,
      minScale: MIN_SCALE,
    });
    // Same-task clear/write produces no intermediate paint; writing an
    // identical value back is a visual no-op while keeping the reflow
    // count at one per fit.
    if (next) title.style.fontSize = next;
    title.dataset.heroFit = next ? "scaled" : "native";
    void previous;
  };

  const schedule = (): void => {
    if (frame !== 0) return;
    frame = window.requestAnimationFrame(() => fit());
  };

  fit();
  // Re-measure once fonts settle (cheap insurance; this site uses local
  // fonts only, so the value is expected to be unchanged — if a webfont is
  // ever introduced, re-measuring must move behind data-hero-ready instead
  // to avoid resizing mid-SplitText).
  if (document.fonts?.ready) {
    document.fonts.ready.then(schedule).catch(() => undefined);
  }
  window.addEventListener("resize", schedule, {
    passive: true,
    signal: events.signal,
  });

  return () => {
    events.abort();
    if (frame !== 0) window.cancelAnimationFrame(frame);
  };
}
