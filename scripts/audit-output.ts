import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

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
      if (html.includes("$$")) errors.push(`${record.pathname}: raw $$ delimiter leaked`);
      if (html.includes("\\begin{")) errors.push(`${record.pathname}: raw \\begin leaked`);
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

  if (errors.length > 0) {
    for (const error of errors) console.error(`OUTPUT: ${error}`);
    process.exit(1);
  }

  console.log(`OUTPUT: ${baseline.html.length} legacy pages and ${articleCount} articles passed`);
}

main();
