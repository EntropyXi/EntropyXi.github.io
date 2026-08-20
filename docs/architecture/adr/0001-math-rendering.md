# ADR 0001：数学渲染采用构建期 MathJax 3

- 状态：已接受
- 日期：2026-08-20
- 修订：2026-08-20 阶段 3 实证后由“按页客户端”改为“构建期”。
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

1. 文章页使用**构建期 MathJax**：`@astrojs/markdown-remark` 的 `unified()`
   处理器 + `remark-math` + `rehype-mathjax`，在 Astro 构建时输出
   `mjx-container`，浏览器端无需加载 MathJax 脚本。
2. 分隔符保留旧站事实：`$...$` 与 `$$...$$`；不引入 `\(...\)`、
   `\[...\]`，不把现有正文归一化为其他分隔符。
3. 非数学页不加载任何 MathJax 脚本。
4. 客户端 MathJax 仅保留在 `src/pages/dev/math-spike.astro` 作为参考实现，
   对应 `src/components/content/MathJax.astro` 与自托管 vendor 资源。

## 修订原因

Astro 7 默认 Sätteri Markdown 处理器会把 LaTeX 中的 `_` 当作 Markdown
强调，破坏下标；启用 Sätteri 的 math 特性后，它只输出剥离 `$` 分隔符的
`<code class="language-math">`，客户端 MathJax 无法按旧站语义直接渲染。
因此文章页改用构建期 MathJax，从解析层保护数学节点。

## 备选方案与理由

- 按页客户端 MathJax：受 Markdown 解析器限制，需要对 29 篇数学文章做
  易错的正交预处理，且审计链路更脆弱。不采用。
- 纯 KaTeX：旧文含大量 AMS 高级环境与自定义宏，直接切换需要批量改写
  公式，违反“正文不变”约束。不采用。
- CDN 加载：不符合本项目第三方脚本自托管优先规则。不采用。

## 验证

- `src/pages/dev/math-spike.astro` 覆盖行内/显示公式、`equation`、
  `aligned`、`align`、`cases`、`bmatrix`、`pmatrix`、`\tag`、
  `\mathbb`、`\boldsymbol`、`\underbrace`、中文 `\text` 与长公式。
- `tests/e2e/math-spike.spec.ts` 验证客户端参考实现。
- `tests/e2e/article.spec.ts` 验证真实文章页存在 `mjx-container`，
  正文无 `$$`/`\begin` 泄漏，页面无横向溢出。

## 后果

- 文章迁移脚本不得改写 TeX 分隔符；内容哈希差异报告不得把数学正文
  变化归为可接受差异。
- 输出审计按构建期方案执行：静态扫描构建 HTML 断言无 TeX 分隔符泄漏、
  无 MathJax 错误标记，并统计 `mjx-container` 数量。
- `astro-public/vendor/mathjax` 仅用于 dev spike，由 copy 脚本生成并保持
  git 忽略；阶段 7 可评估是否移除客户端 vendor。
