export const THEME_STORAGE_KEY = "entropyxi-theme";

export type Theme = "dark";

/**
 * The site permanently locked the dark theme on 2026-08-23; there is no
 * light variant and no toggle. Kept as the single pure contract for tests
 * and any future code that needs to name the active theme.
 */
export function resolveTheme(..._args: unknown[]): Theme {
  void _args;
  return "dark";
}
