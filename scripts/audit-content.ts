import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { hasTopLevelMarkdownHeading } from "../src/lib/audit/heading-uniqueness";
import { readPostSources } from "./lib/content-manifest";

/**
 * 内容审计。
 *
 * `tests/fixtures/legacy-baseline.json` 记录的是迁移前冻结的历史 URL，禁止
 * 手工追加新文章；因此本脚本对新增文章采取增量语义：
 *
 * - 结构规则（frontmatter 字段、正文禁 H1、禁 Obsidian 图片语法、图片必须存在）
 *   对**所有**文章生效；
 * - 基线中每一条历史文章 URL 必须仍然由某篇文章产出（防止历史 URL 静默丢失）；
 * - 文章总数不得少于基线 `postCount`。
 */

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const migratedRoot = path.join(root, "src", "content", "blog");
const publicRoot = path.join(root, "astro-public");
const baselineFile = path.join(root, "tests", "fixtures", "legacy-baseline.json");

interface LegacyHtmlRecord {
  pathname: string;
  kind: string;
}

interface LegacyBaseline {
  html: LegacyHtmlRecord[];
  summary: { postCount: number };
}

const errors: string[] = [];

function main(): void {
  const baseline = JSON.parse(readFileSync(baselineFile, "utf8")) as LegacyBaseline;
  const legacyPermalinks = new Map<string, string>();
  for (const record of baseline.html) {
    if (record.kind !== "article") continue;
    const pathname = decodeURIComponent(record.pathname);
    legacyPermalinks.set(pathname, pathname.replace(/^\/+/, "").replace(/\/+$/, ""));
  }

  const posts = readPostSources(migratedRoot);
  if (posts.length < baseline.summary.postCount) {
    errors.push(
      `expected at least ${baseline.summary.postCount} posts (frozen legacy baseline), found ${posts.length}`,
    );
  }

  const seenPermalinks = new Set<string>();
  for (const { file, data, body } of posts) {
    const title = data.title;
    const description = data.description;
    const date = data.date;
    const updated = data.updated;
    const tags = data.tags;
    const categories = data.categories;
    const permalink = data.permalink;
    const math = data.math;
    const draft = data.draft;

    if (typeof title !== "string" || title.trim() === "") errors.push(`${file}: title must be a non-empty string`);
    if (typeof description !== "string" || description.trim() === "") errors.push(`${file}: description must be a non-empty string`);
    if (typeof date !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00$/.test(date)) {
      errors.push(`${file}: date must be ISO 8601 with +08:00`);
    }
    if (updated !== date) errors.push(`${file}: updated must equal date`);
    if (!Array.isArray(tags) || tags.length === 0) errors.push(`${file}: tags must be a non-empty array`);
    if (!Array.isArray(categories) || categories.length === 0) errors.push(`${file}: categories must be a non-empty array`);
    if (typeof permalink !== "string" || permalink.trim() === "") errors.push(`${file}: permalink must be a non-empty string`);
    if (typeof math !== "boolean") errors.push(`${file}: math must be boolean`);
    if (typeof draft !== "boolean") errors.push(`${file}: draft must be boolean`);

    if (typeof permalink === "string" && permalink.trim() !== "") {
      if (seenPermalinks.has(permalink)) errors.push(`${file}: duplicate permalink ${permalink}`);
      seenPermalinks.add(permalink);
    }

    if (body.includes("<!-- more -->")) errors.push(`${file}: contains <!-- more -->`);
    if (/!\[\[[^\]]+\]\]/.test(body)) errors.push(`${file}: contains Obsidian image syntax`);
    if (hasTopLevelMarkdownHeading(body)) {
      errors.push(`${file}: body must not contain a top-level H1 heading (single H1 is rendered by PostLayout)`);
    }

    // Math delimiter sanity: protect against body text swallowed by display math
    // ($$ directly adjacent to CJK text) and empty display formulas ($$$$).
    if (body.includes("$$")) {
      if (/\$\$[\u4e00-\u9fff]|[\u4e00-\u9fff]\$\$/.test(body)) {
        errors.push(`${file}: display math delimiter ($$) directly adjacent to CJK body text`);
      }
      if (/\$\$\$\$/.test(body)) {
        errors.push(`${file}: empty display math formula ($$$$)`);
      }
    }

    const refs = [
      ...Array.from(body.matchAll(/!\[[^\]]*\]\(([^)]+)\)/g), (m) => m[1] ?? ""),
      ...Array.from(body.matchAll(/<img\b[^>]*src=["']([^"']+)["'][^>]*>/g), (m) => m[1] ?? ""),
    ].filter((ref): ref is string => ref !== "");
    for (const ref of refs) {
      const target = (ref.split("#")[0] ?? "").split("?")[0] ?? "";
      if (target.startsWith("http://") || target.startsWith("https://")) continue;
      const resolved = target.startsWith("/")
        ? path.join(publicRoot, target.slice(1))
        : path.resolve(path.dirname(file), target);
      if (!existsSync(resolved)) errors.push(`${file}: image reference not found: ${ref}`);
    }
  }

  // 迁移基线是冻结的历史 URL 清单：迁移后新增文章属于增量，但历史 permalink
  // 一个都不能消失（不得因改标题、移动文件或改分类而改变）。
  let preservedLegacy = 0;
  for (const [pathname, legacyPermalink] of legacyPermalinks) {
    if (seenPermalinks.has(legacyPermalink)) preservedLegacy += 1;
    else errors.push(`legacy URL ${pathname} is no longer produced by any post (expected permalink ${legacyPermalink})`);
  }

  if (errors.length > 0) {
    for (const error of errors) console.error(`CONTENT: ${error}`);
    process.exit(1);
  }

  console.log(
    `CONTENT: ${posts.length} posts passed (${preservedLegacy}/${legacyPermalinks.size} legacy URLs preserved, ${posts.length - preservedLegacy} added after migration)`,
  );
}

main();
