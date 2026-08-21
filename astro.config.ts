import { unified } from "@astrojs/markdown-remark";
import { defineConfig } from "astro/config";
import rehypeMathjax from "rehype-mathjax";
import remarkMath from "remark-math";

import {
  captureMathAccessibilitySources,
  labelMathJaxSvg,
} from "./src/lib/markdown/math-accessibility";

export default defineConfig({
  site: "https://entropyxi.github.io",
  trailingSlash: "always",
  output: "static",
  // 与 Hexo 的 public/ 完全隔离，避免共存期互相覆盖。
  publicDir: "astro-public",
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        captureMathAccessibilitySources,
        rehypeMathjax,
        labelMathJaxSvg,
      ],
    }),
  },
});
