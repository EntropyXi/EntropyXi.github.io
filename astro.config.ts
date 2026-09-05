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
  // 静态资源根目录
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
