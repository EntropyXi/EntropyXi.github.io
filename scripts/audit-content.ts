import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { JSON_SCHEMA, load as yamlLoad } from "js-yaml";
import { hasTopLevelMarkdownHeading } from "../src/lib/audit/heading-uniqueness";

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

function walkMarkdown(dir: string, files: string[] = []): string[] {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith(".")) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walkMarkdown(full, files);
    else if (entry.name.endsWith(".md")) files.push(full);
  }
  return files;
}

function parseFrontmatter(file: string): { data: Record<string, unknown>; body: string } {
  const raw = readFileSync(file, "utf8");
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!match || match[1] === undefined) throw new Error(`Missing frontmatter block: ${file}`);
  return {
    data: yamlLoad(match[1], { schema: JSON_SCHEMA }) as Record<string, unknown>,
    body: raw.slice(match[0].length),
  };
}

function main(): void {
  const baseline = JSON.parse(readFileSync(baselineFile, "utf8")) as LegacyBaseline;
  const legacyPathnames = new Set(
    baseline.html.filter((record) => record.kind === "article").map((record) => decodeURIComponent(record.pathname)),
  );

  const migratedFiles = walkMarkdown(migratedRoot);
  if (migratedFiles.length !== baseline.summary.postCount) {
    errors.push(`expected ${baseline.summary.postCount} migrated posts, found ${migratedFiles.length}`);
  }

  const seenPermalinks = new Set<string>();
  for (const file of migratedFiles) {
    const { data, body } = parseFrontmatter(file);
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
      const expectedPathname = `/${permalink}/`;
      if (!legacyPathnames.has(expectedPathname)) {
        errors.push(`${file}: permalink pathname ${expectedPathname} not found in legacy URL manifest`);
      }
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

  if (errors.length > 0) {
    for (const error of errors) console.error(`CONTENT: ${error}`);
    process.exit(1);
  }

  console.log(`CONTENT: ${migratedFiles.length} posts passed`);
}

main();
