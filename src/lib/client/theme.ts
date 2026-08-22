export const THEME_STORAGE_KEY = "entropyxi-theme";

export type Theme = "dark";

export function isTheme(value: unknown): value is Theme {
  return value === "dark";
}

export function resolveTheme(..._args: unknown[]): Theme {
  void _args;
  return "dark";
}

export function initializeThemeControls(): () => void {
  document.documentElement.dataset.theme = "dark";
  return () => {};
}
