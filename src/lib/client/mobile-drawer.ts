const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(",");

export function initializeMobileDrawer(): () => void {
  const trigger = document.getElementById("mobile-menu-btn");
  const drawer = document.getElementById("mobile-drawer");
  const backdrop = document.getElementById("mobile-drawer-backdrop");
  const closeButton = document.getElementById("mobile-drawer-close");

  if (
    !(trigger instanceof HTMLButtonElement) ||
    !(drawer instanceof HTMLElement) ||
    !(backdrop instanceof HTMLElement) ||
    !(closeButton instanceof HTMLButtonElement)
  ) {
    return () => undefined;
  }

  const events = new AbortController();
  const desktopMedia = window.matchMedia("(width > 48rem)");
  let previousBodyOverflow = "";

  const isOpen = (): boolean => drawer.classList.contains("open");

  const close = (restoreFocus = true): void => {
    if (!isOpen()) return;
    drawer.classList.remove("open");
    drawer.setAttribute("aria-hidden", "true");
    trigger.setAttribute("aria-expanded", "false");
    document.body.style.overflow = previousBodyOverflow;
    if (restoreFocus) trigger.focus();
  };

  const open = (): void => {
    if (isOpen()) return;
    previousBodyOverflow = document.body.style.overflow;
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
    trigger.setAttribute("aria-expanded", "true");
    document.body.style.overflow = "hidden";
    closeButton.focus();
  };

  trigger.addEventListener("click", open, { signal: events.signal });
  backdrop.addEventListener("click", () => close(), {
    signal: events.signal,
  });
  closeButton.addEventListener("click", () => close(), {
    signal: events.signal,
  });
  drawer.querySelectorAll<HTMLAnchorElement>("a[href]").forEach((link) => {
    link.addEventListener("click", () => close(false), {
      signal: events.signal,
    });
  });

  window.addEventListener(
    "keydown",
    (event) => {
      if (!isOpen()) return;
      if (event.key === "Escape") {
        event.preventDefault();
        close();
        return;
      }
      if (event.key !== "Tab") return;

      const focusable = Array.from(
        drawer.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
      ).filter((element) => !element.hasAttribute("disabled"));
      const first = focusable.at(0);
      const last = focusable.at(-1);
      if (!first || !last) return;

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    },
    { signal: events.signal },
  );

  desktopMedia.addEventListener(
    "change",
    (event) => {
      if (event.matches) close(false);
    },
    { signal: events.signal },
  );

  return () => {
    events.abort();
    close(false);
  };
}
