import { gsap } from "gsap";
import { SplitText } from "gsap/SplitText";

gsap.registerPlugin(SplitText);

const HERO_TITLE_LABEL = "WELCOME TO ENTROPYXI BLOG !";

/**
 * Hero entrance choreography (plan §4.4). Runs only when the inline script
 * gated the hero with data-hero-pending — i.e. home + full motion + gsap
 * flag on — so no-JS and reduced-motion never pay for or see this module.
 *
 * Line-level SplitText only: textContent stays intact for the
 * hero-contrast text assertions, and the h1 keeps an accessible name via
 * aria-label while the line clones are aria-hidden.
 */
export function runHeroChoreography(): void {
  const root = document.documentElement;
  const section = document.querySelector<HTMLElement>(".hero-fullscreen");
  const title = section?.querySelector<HTMLElement>(".hero-welcome-title");
  if (!section || !title) return;
  if (section.dataset.heroChoreographed === "true") return;
  section.dataset.heroChoreographed = "true";

  const markReady = (): void => {
    root.removeAttribute("data-hero-pending");
    root.setAttribute("data-hero-ready", "true");
  };

  if (root.getAttribute("data-hero-pending") !== "true") {
    // The inline safety timeout already restored visibility; skip the intro.
    markReady();
    return;
  }

  const narrativeItems = section.querySelectorAll<HTMLElement>(
    ".hero-narrative-block > *",
  );
  const bracket = section.querySelector<HTMLElement>(".narrative-tech-bracket");
  const indicator = section.querySelector<HTMLElement>(".hero-scroll-wrapper");

  const fontsReady = Promise.race([
    document.fonts?.ready ?? Promise.resolve(),
    new Promise<void>((resolve) => {
      window.setTimeout(resolve, 800);
    }),
  ]);

  void fontsReady.then(() => {
    if (root.getAttribute("data-hero-pending") !== "true") {
      markReady();
      return;
    }

    const split = new SplitText(title, { type: "lines", mask: "lines" });
    title.setAttribute("aria-label", HERO_TITLE_LABEL);

    const finish = (): void => {
      split.revert();
      title.removeAttribute("aria-label");
      markReady();
    };

    const timeline = gsap.timeline({
      onStart: () => root.removeAttribute("data-hero-pending"),
      onComplete: finish,
    });

    timeline
      .from(split.lines, {
        yPercent: 115,
        duration: 0.9,
        ease: "power4.out",
        stagger: 0.09,
      })
      .from(
        narrativeItems,
        {
          autoAlpha: 0,
          y: 14,
          duration: 0.7,
          ease: "power3.out",
          stagger: 0.08,
        },
        "-=0.45",
      )
      .from(
        bracket,
        {
          scaleX: 0,
          transformOrigin: "left bottom",
          duration: 0.6,
          ease: "power3.out",
        },
        "-=0.35",
      )
      .from(
        indicator,
        { autoAlpha: 0, y: 6, duration: 0.6, ease: "power2.out" },
        "-=0.3",
      );

    // A resize mid-intro (viewport change, devtools) would leave the line
    // masks stale — bail straight to the final visible layout instead.
    const bailToFinal = (): void => {
      timeline.kill();
      finish();
    };
    window.addEventListener("resize", bailToFinal, { once: true });
    timeline.eventCallback("onComplete", () => {
      window.removeEventListener("resize", bailToFinal);
      finish();
    });
  });
}
