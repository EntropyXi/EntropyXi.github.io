export const THEME_STORAGE_KEY = "entropyxi-theme";

export type Theme = "dark" | "light";

export function isTheme(value: unknown): value is Theme {
  return value === "dark" || value === "light";
}

export function resolveTheme(
  storedTheme: unknown,
  systemPrefersDark: boolean,
): Theme {
  if (isTheme(storedTheme)) return storedTheme;
  return systemPrefersDark ? "dark" : "light";
}

function readStoredTheme(): Theme | null {
  try {
    const value = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isTheme(value) ? value : null;
  } catch {
    return null;
  }
}

function persistTheme(theme: Theme): void {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Storage can be unavailable in private or restricted browsing contexts.
  }
}

function applyTheme(
  theme: Theme,
  controls: readonly HTMLButtonElement[],
): void {
  document.documentElement.dataset.theme = theme;
  const nextThemeLabel =
    theme === "light" ? "切换至深色主题" : "切换至浅色主题";
  for (const control of controls) {
    control.setAttribute("aria-label", nextThemeLabel);
    control.setAttribute("title", nextThemeLabel);
    control.setAttribute("aria-pressed", String(theme === "dark"));
  }
}

export function initializeThemeControls(): () => void {
  const controls = Array.from(
    document.querySelectorAll<HTMLButtonElement>("[data-theme-toggle]"),
  );
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const events = new AbortController();
  let explicitTheme = readStoredTheme();

  applyTheme(resolveTheme(explicitTheme, media.matches), controls);

  for (const control of controls) {
    control.addEventListener(
      "click",
      () => {
        const current = isTheme(document.documentElement.dataset.theme)
          ? document.documentElement.dataset.theme
          : resolveTheme(explicitTheme, media.matches);
        explicitTheme = current === "light" ? "dark" : "light";
        persistTheme(explicitTheme);
        applyTheme(explicitTheme, controls);
      },
      { signal: events.signal },
    );
  }

  media.addEventListener(
    "change",
    (event) => {
      if (explicitTheme === null) {
        applyTheme(event.matches ? "dark" : "light", controls);
      }
    },
    { signal: events.signal },
  );

  window.addEventListener(
    "storage",
    (event) => {
      if (event.key !== THEME_STORAGE_KEY) return;
      explicitTheme = isTheme(event.newValue) ? event.newValue : null;
      applyTheme(resolveTheme(explicitTheme, media.matches), controls);
    },
    { signal: events.signal },
  );

  return () => events.abort();
}
