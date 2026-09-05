import type Lenis from "lenis";

let instance: Lenis | null = null;

export function setLenisInstance(value: Lenis | null): void {
  instance = value;
}

/**
 * Returns the active Lenis instance without importing the lenis package:
 * only lenis-controller (dynamically loaded) touches the dependency, so
 * statically-imported modules such as back-to-top can probe it for free.
 */
export function getLenis(): Lenis | null {
  return instance;
}
