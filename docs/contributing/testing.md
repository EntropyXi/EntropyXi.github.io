# 测试指南

本文说明本项目的自动化测试分层与质量门禁，来源为
`docs/superpowers/plans/2026-08-20-astro-migration-plan.md` §11 与
`docs/refactor/quality-gates.md`。

## 统一门禁

在工作区根目录运行：

```bash
npm run check
```

按序执行：`format:check` → `lint`（ESLint + Stylelint）→ `check:types`（astro check）→
`audit:content` → `audit:assets` → `test:unit` → `test:e2e` → `audit:bundle` → `audit:output`。
任何一步失败即非零退出。

## 单元测试（Vitest）

- 目录：`tests/unit/**/*.test.ts`；运行：`npm run test:unit -- --run`。
- 覆盖纯函数与组件输出：分页、路径、标题、主题锁定（`resolveTheme` 固定返回 dark）、
  阅读进度、页头状态、数学可访问性与标题唯一性。

## 端到端测试（Playwright）

- 目录：`tests/e2e/`；运行：`npm run test:e2e`。
- 在 `npm run build` + `npm run preview:test` 的生产预览上运行，覆盖 7 种项目：
  desktop-chromium、mobile-390、mobile-360、mobile-safari（WebKit，iOS 替代）、
  reduced-motion、javascript-disabled、zoom-200。
- CI 需安装对应浏览器：`npx playwright install --with-deps chromium webkit`
  （见 `.github/workflows/quality.yml` 与 `.github/workflows/deploy.yml`）。
- axe 检查首页、归档、搜索与 DDIM 文章，`serious`/`critical` 必须为 0。

## 内容与产物审计

- `scripts/audit-content.ts`：校验 30 篇源文的 frontmatter、permalink、图片引用、
  禁止模式（`<!-- more -->`、Obsidian 语法、正文 H1、`$$` 邻接 CJK、空公式 `$$$$`）。
- `scripts/audit-output.ts`：校验 `dist/` 每篇产物页面恰有一个 `<h1>`、无 TeX 分隔符泄漏、
  无 MathJax 错误标记、全部 `mjx-container` SVG 具有可访问名称。
- `scripts/audit-bundle.ts`：校验第一方客户端脚本 gzip 总量上限（80 KiB，单文件 32 KiB；ADR 0003）。

## 质量门禁阈值

- 公式可访问名称计数与渲染数量一致（当前 1351/1351）。
- 30 篇内容、产物文章页、兼容页面与关键资源（404、search、about、sitemap、atom、search.xml、pagefind）齐全。
- Lighthouse 四分类（home/search/article-normal/article-ddim）均 ≥90，Accessibility ≥95。
