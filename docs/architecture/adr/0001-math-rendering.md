# ADR 0001：数学渲染采用按页客户端 MathJax 3

- 状态：已接受
- 日期：2026-08-20
- 关联计划：`docs/superpowers/plans/2026-08-20-astro-migration-plan.md` §9

## 背景

旧站由 NexT 在 `mathjax: true` 的文章页按需加载 MathJax，Pandoc 以
`--mathjax` 保留 TeX 源文本。阶段 0 基线扫描显示：

- 30 篇文章中 29 篇含数学。
- 使用 `$...$` 854 对、`$$...$$` 468 对。
- `\(...\)` 与 `\[...\]` 使用数为 0。
- 存在 `align`、`aligned`、`equation`、`cases`、矩阵、`\tag`、
  `\mathbb`、`\boldsymbol`、`\underbrace` 等高级语法。

迁移必须保证全部历史公式在浏览器渲染后无原始分隔符泄漏，并且不改变文章正文。

## 决策

1. 数学渲染使用**客户端 MathJax 3**，仅在 `math: true` 的文章页加载。
2. 分隔符保留旧站事实：`inlineMath: [['$', '$']]`、
   `displayMath: [['$$', '$$']]`；不引入 `\(...\)`、`\[...\]`，
   不把现有正文归一化为其他分隔符。
3. MathJax 资源**自托管**：`mathjax@3.2.2` 的 `es5` 目录由
   `scripts/copy-vendor-assets.ts` 复制到 `astro-public/vendor/mathjax/`，
   页面引用 `/vendor/mathjax/tex-mml-chtml.js`。
4. 配置 `processEscapes: true`，跳过 `script`、`noscript`、`style`、
   `textarea`、`pre`、`code`，关闭自动编号（`tags: 'none'`）。
5. 非数学页不加载任何 MathJax 脚本；构建产物审计按客户端方案验收：
   静态断言 `math: false` 页面无 TeX，`math: true` 页面由 Playwright
   等待 MathJax 渲染完成后再断言无可见原始分隔符。

## 备选方案与理由

- 构建期 MathJax：HTML 体积和构建复杂度显著增加，且复杂 LaTeX 的
  CHTML 字体/样式在静态审计中更脆弱；对 29 篇数学文章的首屏体积
  也没有客户端按需加载可控。不采用。
- 纯 KaTeX：旧文含大量 AMS 高级环境与自定义宏，直接切换需要批量
  改写公式，违反“正文不变”约束。不采用。
- CDN 加载：可用，但不符合本项目的第三方脚本自托管优先规则。
  不采用。

## 验证

- `src/pages/dev/math-spike.astro` 覆盖行内/显示公式、`equation`、
  `aligned`、`align`、`cases`、`bmatrix`、`pmatrix`、`\tag`、
  `\mathbb`、`\boldsymbol`、`\underbrace`、中文 `\text` 与长公式。
- `tests/e2e/math-spike.spec.ts` 等待 MathJax startup promise 完成后
  断言存在 `mjx-container`、正文无 `$$`/`\begin` 泄漏、页面无横向溢出。
- 该 spike 在 Chromium 中通过，MathJax 扩展按需自托管加载成功。

## 后果

- 文章迁移脚本不得改写 TeX 分隔符；内容哈希差异报告不得把数学正文
  变化归为可接受差异。
- 视觉回归截图必须等待 MathJax 渲染完成后再采集。
- 阶段 7 清理时，`astro-public/vendor/mathjax` 是构建生成物，仍由
  copy 脚本生成并保持 git 忽略。
