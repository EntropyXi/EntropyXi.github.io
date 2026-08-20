import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://entropyxi.github.io",
  trailingSlash: "always",
  output: "static",
  // 与 Hexo 的 public/ 完全隔离，避免共存期互相覆盖。
  publicDir: "astro-public",
});
