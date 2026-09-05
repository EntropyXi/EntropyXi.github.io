import Lenis from "lenis";
import { setLenisInstance } from "./lenis-bridge";

let instance: Lenis | null = null;

function readScrollPaddingTop(root: HTMLElement): number {
  const value = Number.parseFloat(getComputedStyle(root).scrollPaddingTop);
  return Number.isFinite(value) ? value : 0;
}

function isFocusableTarget(target: Element): target is HTMLElement {
  return target.matches(
    'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
  );
}

/**
 * Creates the Lenis singleton (plan §4.3). The instance deliberately
 * survives view transitions: re-invocation on a new document only resizes.
 * Anchor handling stays here instead of Lenis' built-in `anchors` option so
 * the scroll-padding offset, hash update, and focus contract are explicit;
 * the skip link is excluded and keeps its native instant jump + focus.
 */
export function initializeLenis(): void {
  const root = document.documentElement;
  if (instance) {
    instance.resize();
    return;
  }

  instance = new Lenis({ autoRaf: true });
  setLenisInstance(instance);
  root.dataset.lenisActive = "true";

  document.addEventListener("click", (event) => {
    if (event.defaultPrevented || event.button !== 0) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
      return;
    }
    const anchor = (event.target as Element | null)?.closest("a[href^='#']");
    if (!(anchor instanceof HTMLAnchorElement)) return;
    if (anchor.classList.contains("skip-link")) return;

    const hash = anchor.getAttribute("href");
    if (!hash || hash === "#") return;

    let target: Element | null = null;
    try {
      target = document.querySelector(hash);
    } catch {
      return;
    }
    if (!target) return;

    event.preventDefault();
    instance?.scrollTo(target as HTMLElement, {
      offset: -readScrollPaddingTop(root),
    });
    history.pushState(null, "", hash);
    if (isFocusableTarget(target)) {
      target.focus({ preventScroll: true });
    }
  });
}
