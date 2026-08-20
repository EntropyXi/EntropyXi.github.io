// @ts-check
import eslintPluginAstro from 'eslint-plugin-astro';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    ignores: [
      'dist/',
      'node_modules/',
      '.astro/',
      'public/',
      'astro-public/',
      'playwright-report/',
      'test-results/',
      // Legacy Hexo CommonJS scripts; TypeScript scripts live in scripts/**/*.ts
      'scripts/*.js',
    ],
  },
  ...eslintPluginAstro.configs['flat/recommended'],
  ...tseslint.configs.recommended,
);
