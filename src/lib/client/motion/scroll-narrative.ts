import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { getLenis } from "./lenis-bridge";

gsap.registerPlugin(ScrollTrigger);

/**
 * Scroll narrative (plan §4.5): wallpaper parallax, hero scroll fade, and
 * the batch reveal that replaces the nth-child stagger. Owns one
 * gsap.context per document; runtime reverts it on view transitions.
 *
 * Reveal mutex: when this module runs, reveal-controller stays idle and
 * batch takes over [data-reveal] — entering elements get their CSS hidden
 * state released (data-reveal-state=visible) and are tweened via autoAlpha
 * in the same frame, so there is no flash and no double animation.
 */
export function initializeScrollNarrative(): (() => void) | void {
  const ctx = gsap.context(() => {
    const lenis = getLenis();
    if (lenis) {
      lenis.on("scroll", ScrollTrigger.update);
      gsap.ticker.lagSmoothing(0);
    }

    // Wallpaper parallax. The trigger must be a document-flow element —
    // the wallpaper itself is position:fixed and cannot be a trigger.
    const wallpaper = document.querySelector(".ambient-wallpaper-img");
    const main = document.getElementById("main-content");
    if (wallpaper && main) {
      gsap.fromTo(
        wallpaper,
        { scale: 1 },
        {
          scale: 1.06,
          ease: "none",
          scrollTrigger: {
            trigger: main,
            start: "top top",
            end: "bottom bottom",
            scrub: true,
            invalidateOnRefresh: true,
          },
        },
      );
    }

    // Hero content eases out while the first screen scrolls away.
    const hero = document.querySelector(".hero-fullscreen");
    const heroContent = document.querySelector(".hero-content-container");
    if (hero && heroContent) {
      gsap.to(heroContent, {
        yPercent: -8,
        autoAlpha: 0.25,
        ease: "none",
        scrollTrigger: {
          trigger: hero,
          start: "top top",
          end: "bottom top",
          scrub: true,
        },
      });
    }

    if (
      document.documentElement.getAttribute("data-feature-reveal") !== "false"
    ) {
      const cards = gsap.utils.toArray<HTMLElement>("[data-reveal]");
      if (cards.length > 0) {
        ScrollTrigger.batch(cards, {
          start: "top 85%",
          once: true,
          onEnter: (batch) => {
            batch.forEach((element) => {
              element.setAttribute("data-reveal-state", "visible");
            });
            gsap.set(batch, { autoAlpha: 0, y: 24 });
            gsap.to(batch, {
              autoAlpha: 1,
              y: 0,
              duration: 0.7,
              ease: "power3.out",
              stagger: 0.08,
              clearProps: "all",
            });
          },
        });
      }
    }
  });

  return () => ctx.revert();
}
