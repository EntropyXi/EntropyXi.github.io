export type ClientCleanup = () => void;
export type ClientInitializer = () => ClientCleanup | void;

interface Registration {
  dispose: ClientCleanup;
}

const registrations = new Map<string, Registration>();

/**
 * Registers an idempotent client feature for the current Astro document.
 * Re-registering the same key first removes every listener and pending effect
 * owned by the previous registration.
 */
export function registerClientFeature(
  key: string,
  initialize: ClientInitializer,
): void {
  registrations.get(key)?.dispose();

  const lifecycle = new AbortController();
  let cleanup: ClientCleanup = () => undefined;

  const deactivate = (): void => {
    cleanup();
    cleanup = () => undefined;
  };

  const activate = (): void => {
    deactivate();
    try {
      cleanup = initialize() ?? (() => undefined);
    } catch (error) {
      document.documentElement.removeAttribute("data-motion");
      console.error(`Client feature "${key}" failed to initialize.`, error);
    }
  };

  document.addEventListener("astro:page-load", activate, {
    signal: lifecycle.signal,
  });
  document.addEventListener("astro:before-swap", deactivate, {
    signal: lifecycle.signal,
  });
  window.addEventListener("pagehide", deactivate, {
    signal: lifecycle.signal,
  });
  window.addEventListener("pageshow", activate, {
    signal: lifecycle.signal,
  });

  const dispose = (): void => {
    lifecycle.abort();
    deactivate();
    registrations.delete(key);
  };

  registrations.set(key, { dispose });
  activate();
}
