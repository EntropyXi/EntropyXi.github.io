import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { countH1Elements } from "../src/lib/audit/heading-uniqueness";
import { readPostSources } from "./lib/content-manifest";

/**
 * 产物审计。
 *
 * 检查范围 = 冻结的迁移基线（全部历史页面必须仍然产出）+ 迁移后新增的文章
 * （由 `src/content/blog` 的 frontmatter 推导 pathname）。新增文章同样要满足
 * “每页恰好一个 h1”和“MathJax SVG 均有可访问名称”，否则门禁只覆盖历史文章。
 */

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const distRoot = path.join(root, "dist");
const contentRoot = path.join(root, "src", "content", "blog");
const baselineFile = path.join(root, "tests", "fixtures", "legacy-baseline.json");

interface LegacyHtmlRecord {
  pathname: string;
  kind: string;
}

interface LegacyBaseline {
  html: LegacyHtmlRecord[];
}

const errors: string[] = [];

/**
 * 解码后的站点 pathname → dist 中的 HTML 文件路径。
 *
 * 全脚本统一以**解码后**的 pathname 作为键：基线里的 pathname 保留百分号编码
 * （来自 canonical URL），而 permalink 是未编码的原始中文，两者必须先归一到
 * 同一形式，否则同一篇文章会被算成两个页面。
 */
function htmlFileFor(decodedPathname: string): string {
  return path.join(distRoot, `${decodedPathname.replace(/^\/+/, "")}index.html`);
}

/** 需要校验的文章 pathname（解码形式）：历史文章 ∪ 非草稿的新文章。 */
function collectArticlePathnames(baseline: LegacyBaseline): Set<string> {
  const pathnames = new Set<string>();
  for (const record of baseline.html) {
    if (record.kind === "article") pathnames.add(decodeURIComponent(record.pathname));
  }
  for (const post of readPostSources(contentRoot)) {
    if (post.data.draft === true) continue;
    const permalink = post.data.permalink;
    // 非法 permalink 由 audit-content.ts 报告，这里不重复报错。
    if (typeof permalink !== "string" || permalink.trim() === "") continue;
    pathnames.add(`/${permalink}/`);
  }
  return pathnames;
}

function main(): void {
  if (!existsSync(distRoot)) {
    console.error("OUTPUT: dist directory is missing");
    process.exit(1);
  }

  const baseline = JSON.parse(readFileSync(baselineFile, "utf8")) as LegacyBaseline;
  const articlePathnames = collectArticlePathnames(baseline);
  let mathContainerCount = 0;
  let accessibleMathSvgCount = 0;

  const missing = new Set<string>();
  for (const record of baseline.html) {
    const pathname = decodeURIComponent(record.pathname);
    if (!existsSync(htmlFileFor(pathname))) {
      missing.add(pathname);
      errors.push(`missing output for ${record.pathname}`);
    }
  }

  for (const pathname of articlePathnames) {
    if (missing.has(pathname)) continue;
    const file = htmlFileFor(pathname);
    if (!existsSync(file)) {
      errors.push(`missing article output for ${pathname}`);
      continue;
    }

    const html = readFileSync(file, "utf8");

    const h1Count = countH1Elements(html);
    if (h1Count !== 1) {
      errors.push(`${pathname}: expected exactly 1 <h1>, found ${h1Count}`);
    }

    const htmlWithoutMathLabels = html.replace(/aria-label="数学公式：[^"]*"/gu, "");
    if (htmlWithoutMathLabels.includes("$$")) {
      errors.push(`${pathname}: raw $$ delimiter leaked outside an accessible name`);
    }
    if (htmlWithoutMathLabels.includes("\\begin{")) {
      errors.push(`${pathname}: raw \\begin leaked outside an accessible name`);
    }
    if (html.includes("mathjax-error")) errors.push(`${pathname}: MathJax error marker found`);

    const articleMathContainers = html.match(/<mjx-container\b/gu)?.length ?? 0;
    const articleAccessibleSvg = html.match(/aria-label="数学公式：/gu)?.length ?? 0;
    mathContainerCount += articleMathContainers;
    accessibleMathSvgCount += articleAccessibleSvg;
    if (articleAccessibleSvg !== articleMathContainers) {
      errors.push(
        `${pathname}: ${articleAccessibleSvg}/${articleMathContainers} MathJax SVGs have accessible names`,
      );
    }
  }

  for (const extra of [
    "404.html",
    "search/index.html",
    "about/index.html",
    "sitemap.xml",
    "atom.xml",
    "search.xml",
    "pagefind/pagefind.js",
  ]) {
    if (!existsSync(path.join(distRoot, extra))) {
      errors.push(`missing output: ${extra}`);
    }
  }

  for (const removedProductionArtifact of ["dev/math-spike/index.html", "vendor/mathjax"]) {
    if (existsSync(path.join(distRoot, removedProductionArtifact))) {
      errors.push(`development-only artifact leaked into production: ${removedProductionArtifact}`);
    }
  }

  if (errors.length > 0) {
    for (const error of errors) console.error(`OUTPUT: ${error}`);
    process.exit(1);
  }

  console.log(
    `OUTPUT: ${baseline.html.length} legacy pages, ${articlePathnames.size} articles, and ${accessibleMathSvgCount}/${mathContainerCount} accessible formulas passed`,
  );
}

main();
