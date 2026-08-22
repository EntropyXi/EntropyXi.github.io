import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { countH1Elements } from "../src/lib/audit/heading-uniqueness";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const distRoot = path.join(root, "dist");
const baselineFile = path.join(root, "tests", "fixtures", "legacy-baseline.json");

interface LegacyHtmlRecord {
  pathname: string;
  kind: string;
}

interface LegacyBaseline {
  html: LegacyHtmlRecord[];
}

const errors: string[] = [];

function main(): void {
  if (!existsSync(distRoot)) {
    console.error("OUTPUT: dist directory is missing");
    process.exit(1);
  }

  const baseline = JSON.parse(readFileSync(baselineFile, "utf8")) as LegacyBaseline;
  const articleCount = baseline.html.filter((record) => record.kind === "article").length;
  let articleHtmlCount = 0;
  let mathContainerCount = 0;
  let accessibleMathSvgCount = 0;

  for (const record of baseline.html) {
    const decodedPathname = decodeURIComponent(record.pathname);
    const relativeFile = `${decodedPathname.replace(/^\/+/, "")}index.html`.replace(/^index\.html$/, "index.html");
    const file = path.join(distRoot, relativeFile);
    if (!existsSync(file)) {
      errors.push(`missing output for ${record.pathname}`);
      continue;
    }
    if (record.kind === "article") {
      articleHtmlCount += 1;
      const html = readFileSync(file, "utf8");

      const h1Count = countH1Elements(html);
      if (h1Count !== 1) {
        errors.push(`${record.pathname}: expected exactly 1 <h1>, found ${h1Count}`);
      }

      const htmlWithoutMathLabels = html.replace(/aria-label="数学公式：[^"]*"/gu, "");
      if (htmlWithoutMathLabels.includes("$$")) {
        errors.push(`${record.pathname}: raw $$ delimiter leaked outside an accessible name`);
      }
      if (htmlWithoutMathLabels.includes("\\begin{")) {
        errors.push(`${record.pathname}: raw \\begin leaked outside an accessible name`);
      }
      if (html.includes("mathjax-error")) errors.push(`${record.pathname}: MathJax error marker found`);

      const articleMathContainers = html.match(/<mjx-container\b/gu)?.length ?? 0;
      const articleAccessibleSvg = html.match(/aria-label="数学公式：/gu)?.length ?? 0;
      mathContainerCount += articleMathContainers;
      accessibleMathSvgCount += articleAccessibleSvg;
      if (articleAccessibleSvg !== articleMathContainers) {
        errors.push(
          `${record.pathname}: ${articleAccessibleSvg}/${articleMathContainers} MathJax SVGs have accessible names`,
        );
      }
    }
  }

  if (articleHtmlCount !== articleCount) {
    errors.push(`expected ${articleCount} article HTML files, found ${articleHtmlCount}`);
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
    `OUTPUT: ${baseline.html.length} legacy pages, ${articleCount} articles, and ${accessibleMathSvgCount}/${mathContainerCount} accessible formulas passed`,
  );
}

main();
