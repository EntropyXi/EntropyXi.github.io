# ADR 0001：数学渲染采用构建期 MathJax 3

- 状态：已接受
- 日期：2026-08-20
- 修订：2026-08-20 阶段 3 实证后由“按页客户端”改为“构建期”；2026-08-21 移除迁移期客户端参考页并补齐 SVG 可访问名称。
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
4. 每个构建期 MathJax SVG 使用同序原始 TeX 生成以“数学公式：”开头的
   `aria-label`；捕获数量与渲染数量不一致时构建失败。
5. 删除迁移期 `/dev/math-spike/`、客户端 MathJax 组件、自托管 vendor 复制
   脚本与直接 `mathjax` 依赖，避免开发资产进入生产产物。

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

- 5 篇真实复杂公式文章覆盖行内/显示公式、AMS 环境、矩阵、`\tag`、
  `\mathbb`、`\boldsymbol`、`\underbrace`、中文 `\text` 与长公式。
- `tests/unit/math-accessibility.test.ts` 验证源顺序、空白归一化、SVG 标注
  与数量不一致的失败路径。
- `tests/e2e/article.spec.ts` 与 `article-layout.spec.ts` 验证真实文章页存在
  `mjx-container`，每个公式 SVG 都有可访问名称，正文无原始 TeX 泄漏，
  桌面与移动页面无横向溢出。

## 后果

- 文章迁移脚本不得改写 TeX 分隔符；内容哈希差异报告不得把数学正文
  变化归为可接受差异。
- 输出审计按构建期方案执行：静态扫描构建 HTML 断言无 TeX 分隔符泄漏、
  无 MathJax 错误标记，并要求全部 `mjx-container` 的 SVG 具有可访问名称。
- 生产产物审计禁止 `/dev/math-spike/` 和 `vendor/mathjax` 再次进入 `dist/`。
- 可访问名称保留原 TeX 而不是生成自然语言朗读；这是无需引入客户端运行时
  的确定性基线。若未来引入专业数学语音规则，必须另开 ADR 并比较产物体积。
