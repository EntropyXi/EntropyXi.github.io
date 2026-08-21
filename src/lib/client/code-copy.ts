const RESET_DELAY_MS = 2_000;

function setButtonState(
  button: HTMLButtonElement,
  state: "idle" | "success" | "error",
): void {
  button.classList.toggle("copied", state === "success");
  button.classList.toggle("copy-failed", state === "error");
  button.textContent =
    state === "success" ? "已复制" : state === "error" ? "复制失败" : "复制";
}

async function writeClipboard(text: string): Promise<void> {
  if (!navigator.clipboard?.writeText) {
    throw new Error("Clipboard API is unavailable.");
  }
  await navigator.clipboard.writeText(text);
}

export function initializeCodeCopyButtons(): () => void {
  const events = new AbortController();
  const timers = new Set<number>();

  document.querySelectorAll<HTMLElement>("pre.astro-code").forEach((pre) => {
    let button = pre.querySelector<HTMLButtonElement>(".code-copy-button");
    if (!button) {
      button = document.createElement("button");
      button.type = "button";
      button.className = "code-copy-button";
      button.setAttribute("aria-label", "复制代码");
      setButtonState(button, "idle");
      pre.appendChild(button);
    }

    const language = pre.dataset.language ?? "";
    if (language && !pre.querySelector(".code-language-label")) {
      const label = document.createElement("span");
      label.className = "code-language-label";
      label.textContent = language;
      pre.insertBefore(label, button);
    }

    button.addEventListener(
      "click",
      async () => {
        const code = pre.querySelector("code")?.textContent ?? "";
        try {
          await writeClipboard(code);
          setButtonState(button, "success");
        } catch {
          setButtonState(button, "error");
        }

        const timer = window.setTimeout(() => {
          timers.delete(timer);
          setButtonState(button, "idle");
        }, RESET_DELAY_MS);
        timers.add(timer);
      },
      { signal: events.signal },
    );
  });

  return () => {
    events.abort();
    for (const timer of timers) window.clearTimeout(timer);
    timers.clear();
  };
}
