export function initializeTocHighlight(): () => void {
  const headings = Array.from(
    document.querySelectorAll<HTMLElement>(
      ".prose h2, .prose h3, .prose h4, .prose h5, .prose h6",
    ),
  ).filter((h) => h.id);

  const desktopLinks = Array.from(
    document.querySelectorAll<HTMLAnchorElement>(".post-toc-link"),
  );
  const mobileLinks = Array.from(
    document.querySelectorAll<HTMLAnchorElement>(".post-toc-mobile-link"),
  );

  if (
    headings.length === 0 ||
    (desktopLinks.length === 0 && mobileLinks.length === 0)
  ) {
    return () => undefined;
  }

  const events = new AbortController();
  let animationFrame = 0;
  let activeId: string | null = null;

  // Sliding TOC indicator (plan §4.6): created only when the motion stack is
  // live; no-JS and reduced-motion keep the class-toggle-only behavior.
  const desktopList = document.querySelector<HTMLElement>(".post-toc-list");
  let indicator: HTMLElement | null = null;

  const syncIndicator = (): void => {
    if (!indicator || !desktopList) return;
    const activeLink = desktopLinks.find(
      (link) => link.getAttribute("href") === `#${activeId}`,
    );
    if (!activeLink) return;
    const listTop = desktopList.getBoundingClientRect().top;
    const linkTop = activeLink.getBoundingClientRect().top;
    indicator.style.transform = `translateY(${(linkTop - listTop).toString()}px)`;
    indicator.dataset.indicatorVisible = "true";
  };

  const attachIndicator = (): void => {
    if (indicator || !desktopList) return;
    indicator = document.createElement("span");
    indicator.className = "post-toc-indicator";
    indicator.setAttribute("aria-hidden", "true");
    desktopList.prepend(indicator);
    syncIndicator();
  };

  if (document.documentElement.dataset.gsapActive === "true") {
    attachIndicator();
  } else {
    document.addEventListener("motion:narrative-ready", attachIndicator, {
      signal: events.signal,
      once: true,
    });
  }

  const render = (): void => {
    animationFrame = 0;

    const scrollY = window.scrollY;
    const offset = 120; // Accounts for sticky header and spacing

    for (let i = headings.length - 1; i >= 0; i--) {
      const heading = headings[i];
      if (heading && heading.offsetTop <= scrollY + offset) {
        activeId = heading.id;
        break;
      }
    }

    if (!activeId && headings.length > 0) {
      activeId = headings[0]?.id ?? null;
    }

    if (activeId) {
      syncIndicator();
      desktopLinks.forEach((link) => {
        link.classList.toggle(
          "active",
          link.getAttribute("href") === `#${activeId}`,
        );
      });
      mobileLinks.forEach((link) => {
        link.classList.toggle(
          "active",
          link.getAttribute("href") === `#${activeId}`,
        );
      });
    }
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
