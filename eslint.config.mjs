// @ts-check
import eslintPluginAstro from "eslint-plugin-astro";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: [
      "dist/",
      "node_modules/",
      ".astro/",
      ".deploy_git/",
      "public/",
      "astro-public/",
      "playwright-report/",
      "test-results/",
      // Legacy Hexo CommonJS code and generated baselines.
      "scripts/**/*.js",
      "tests/fixtures/",
      "tests/legacy-baseline/",
      "update_categories.js",
    ],
  },
  // Order matters: tseslint's unscoped parser config must come first so that
  // eslint-plugin-astro can override the parser for .astro files.
  ...tseslint.configs.recommended,
  ...eslintPluginAstro.configs["flat/recommended"],
);
