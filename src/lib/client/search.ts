interface PagefindResultData {
  excerpt?: string;
  meta?: { title?: string };
  url: string;
}

interface PagefindResult {
  data: () => Promise<PagefindResultData>;
}

interface PagefindModule {
  search: (query: string) => Promise<{ results: PagefindResult[] }>;
}

const SEARCH_DEBOUNCE_MS = 250;

function appendExcerptNode(
  source: Node,
  target: DocumentFragment | Element,
): void {
  if (source.nodeType === Node.TEXT_NODE) {
    target.append(document.createTextNode(source.textContent ?? ""));
    return;
  }
  if (!(source instanceof HTMLElement)) return;

  const nextTarget =
    source.tagName === "MARK" ? document.createElement("mark") : target;
  for (const child of source.childNodes) appendExcerptNode(child, nextTarget);
  if (nextTarget !== target) target.append(nextTarget);
}

export function renderSafeExcerpt(
  container: HTMLElement,
  excerpt: string,
): void {
  const template = document.createElement("template");
  template.innerHTML = excerpt;
  const fragment = document.createDocumentFragment();
  for (const child of template.content.childNodes) {
    appendExcerptNode(child, fragment);
  }
  container.replaceChildren(fragment);
}

function createResultCard(data: PagefindResultData): HTMLLIElement {
  const card = document.createElement("li");
  card.className = "search-result-card";

  const title = document.createElement("h2");
  title.className = "search-result-title";
  const link = document.createElement("a");
  link.href = data.url;
  link.textContent = data.meta?.title ?? data.url;
  title.appendChild(link);

  const excerpt = document.createElement("div");
  excerpt.className = "search-result-excerpt";
  renderSafeExcerpt(excerpt, data.excerpt ?? "");

  card.append(title, excerpt);
  return card;
}

export function initializeSearch(): () => void {
  const input = document.getElementById("search-input");
  const results = document.getElementById("search-results");
  const status = document.getElementById("search-status");
  if (
    !(input instanceof HTMLInputElement) ||
    !(results instanceof HTMLUListElement) ||
    !(status instanceof HTMLDivElement)
  ) {
    return () => undefined;
  }

  const events = new AbortController();
  let debounceTimer = 0;
  let requestVersion = 0;

  const executeSearch = async (query: string): Promise<void> => {
    const version = ++requestVersion;
    results.replaceChildren();
    const trimmed = query.trim();
    if (!trimmed) {
      status.textContent = "";
      return;
    }

    status.textContent = "正在搜索...";
    try {
      const pagefindUrl = "/pagefind/pagefind.js";
      const pagefind = (await import(
        /* @vite-ignore */ pagefindUrl
      )) as PagefindModule;
      const search = await pagefind.search(trimmed);
      if (version !== requestVersion) return;

      if (search.results.length === 0) {
        status.textContent = `未找到与 "${trimmed}" 相关的文章。`;
        return;
      }

      status.textContent = `找到 ${search.results.length} 篇相关文章：`;
      const cards = await Promise.all(
        search.results.map(async (result) =>
          createResultCard(await result.data()),
        ),
      );
      if (version === requestVersion) results.replaceChildren(...cards);
    } catch {
      if (version === requestVersion) {
        status.textContent = "搜索索引加载失败，请刷新页面重试。";
      }
    }
  };

  const form = input.closest("form");
  form?.addEventListener(
    "submit",
    (event) => {
      event.preventDefault();
    },
    { signal: events.signal },
  );

  window.addEventListener(
    "keydown",
    (event) => {
      if (
        event.key === "/" &&
        document.activeElement !== input &&
        document.activeElement?.tagName !== "INPUT" &&
        document.activeElement?.tagName !== "TEXTAREA"
      ) {
        event.preventDefault();
        input.focus();
      }
    },
    { signal: events.signal },
  );
  input.addEventListener(
    "input",
    () => {
      window.clearTimeout(debounceTimer);
      debounceTimer = window.setTimeout(
        () => void executeSearch(input.value),
        SEARCH_DEBOUNCE_MS,
      );
    },
    { signal: events.signal },
  );

  return () => {
    events.abort();
    window.clearTimeout(debounceTimer);
    requestVersion += 1;
  };
}
