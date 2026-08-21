// @ts-check
/** @type {import('stylelint').Config} */
export default {
  extends: ["stylelint-config-standard"],
  overrides: [
    {
      files: ["**/*.astro"],
      customSyntax: "postcss-html",
      rules: {
        "selector-pseudo-class-no-unknown": [
          true,
          {
            ignorePseudoClasses: ["global"],
          },
        ],
      },
    },
  ],
};
