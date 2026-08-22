/**
 * 每页单 H1 断言工具。
 *
 * 依据 `docs/superpowers/plans/2026-08-20-astro-migration-plan.md` §6.4：
 * “正文从 `##` 开始，文章标题由 layout 输出”“每个页面只有 layout 输出的
 * 一个 `h1`”。`PostLayout` 已输出唯一 `h1.post-title`，因此正文不得再出现
 * 一级标题，产物页面必须且只能有一个 `<h1>`。
 */

/** 统计渲染后 HTML 中顶层 `<h1>` 元素个数（仅匹配开始标签）。 */
export function countH1Elements(html: string): number {
  return html.match(/<h1[\s>]/gu)?.length ?? 0;
}

/**
 * 检测 Markdown 正文（已剥离 frontmatter）中是否存在一级 ATX 标题。
 * 匹配行首单个 `#` 且其后不是 `#` 的情况（`# Title` 与 `#Title`），
 * 不匹配 `## Title` 等更高级标题。
 */
export function hasTopLevelMarkdownHeading(body: string): boolean {
  return /^#(?!#)/m.test(body);
}
