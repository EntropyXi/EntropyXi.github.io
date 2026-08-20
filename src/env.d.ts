/// <reference types="astro/client" />

declare module "/pagefind/pagefind.js" {
  interface PagefindResultData {
    url: string;
    meta?: { title?: string };
    excerpt?: string;
  }

  interface PagefindResult {
    data(): Promise<PagefindResultData>;
  }

  interface PagefindSearchResponse {
    results: PagefindResult[];
  }

  interface PagefindApi {
    search(query: string): Promise<PagefindSearchResponse>;
  }

  const pagefind: PagefindApi;
  export default pagefind;
}
