import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { dump, JSON_SCHEMA, load as yamlLoad } from 'js-yaml';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const postsRoot = path.join(root, 'source', '_posts');
const targetRoot = path.join(root, 'src', 'content', 'blog');
const publicRoot = path.join(root, 'astro-public');

interface LegacyData {
  title: string;
  description: string;
  date: string;
  tags: string[];
  categories: string[];
  mathjax?: unknown;
}

interface MigratedData {
  title: string;
  description: string;
  date: string;
  updated: string;
  tags: string[];
  categories: string[];
  permalink: string;
  math: boolean;
  draft: boolean;
}

interface MigrationReport {
  posts: number;
  removedObsidianImages: Array<{ file: string; lines: string[] }>;
  copiedAssets: string[];
  skippedWrites: number;
}

function walkMarkdown(dir: string, files: string[] = []): string[] {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walkMarkdown(full, files);
    else if (entry.name.endsWith('.md')) files.push(full);
  }
  return files;
}

function parsePost(file: string): { data: LegacyData; body: string } {
  const raw = readFileSync(file, 'utf8');
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!match || match[1] === undefined) throw new Error(`Missing frontmatter block: ${file}`);
  const data = parseYamlAsLegacyData(match[1], file);
  return { data, body: raw.slice(match[0].length) };
}

function parseYamlAsLegacyData(yamlText: string, file: string): LegacyData {
  // 本迁移脚本只依赖 js-yaml 解析 YAML，不依赖 gray-matter，保证 TS 可类型化。
  const loaded = yamlLoad(yamlText, { schema: JSON_SCHEMA }) as Record<string, unknown>;
  const str = (value: unknown, name: string): string => {
    if (typeof value !== 'string') throw new Error(`${file}: ${name} must be a string`);
    return value;
  };
  const arr = (value: unknown, name: string): string[] => {
    if (!Array.isArray(value)) throw new Error(`${file}: ${name} must be an array`);
    return value.map((item) => {
      if (typeof item !== 'string') throw new Error(`${file}: ${name} items must be strings`);
      return item;
    });
  };
  return {
    title: str(loaded.title, 'title'),
    description: str(loaded.description, 'description'),
    date: str(loaded.date, 'date'),
    tags: arr(loaded.tags, 'tags'),
    categories: arr(loaded.categories, 'categories'),
    mathjax: loaded.mathjax,
  };
}

function toShanghaiIso(date: string): string {
  const match = date.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})$/);
  if (!match) throw new Error(`Unsupported legacy date format: ${date}`);
  return `${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:${match[6]}+08:00`;
}

function toMathBoolean(value: unknown): boolean {
  return value === true || value === 'true';
}

function legacyPermalink(date: string, relativePathNoExt: string): string {
  const match = date.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!match) throw new Error(`Cannot derive permalink from date: ${date}`);
  return `${match[1]}/${match[2]}/${match[3]}/${relativePathNoExt}`;
}

function yamlString(value: string): string {
  return JSON.stringify(value);
}

function renderFrontmatter(data: MigratedData): string {
  const lines = [
    '---',
    `title: ${yamlString(data.title)}`,
    `description: ${yamlString(data.description)}`,
    `date: ${yamlString(data.date)}`,
    `updated: ${yamlString(data.updated)}`,
    'tags:',
    ...data.tags.map((tag) => `  - ${yamlString(tag)}`),
    'categories:',
    ...data.categories.map((category) => `  - ${yamlString(category)}`),
    `permalink: ${yamlString(data.permalink)}`,
    `math: ${data.math}`,
    `draft: ${data.draft}`,
    '---',
  ];
  return `${lines.join('\n')}\n`;
}

function stripMoreMarker(body: string): string {
  return body.replace(/^<!-- more -->[ \t]*\r?\n?/gm, '');
}

function stripObsidianImages(body: string): { body: string; removed: string[] } {
  const removed: string[] = [];
  const lines = body.split(/\r?\n/);
  const kept = lines.filter((line) => {
    const trimmed = line.trim();
    if (trimmed.startsWith('![[[') || (trimmed.startsWith('![[') && trimmed.endsWith(']]'))) {
      removed.push(trimmed);
      return false;
    }
    return true;
  });
  return { body: kept.join('\n'), removed };
}

function normalizeLf(body: string): string {
  return body.replace(/\r\n/g, '\n');
}

export function migrateContent(): MigrationReport {
  const files = walkMarkdown(postsRoot);
  const report: MigrationReport = {
    posts: files.length,
    removedObsidianImages: [],
    copiedAssets: [],
    skippedWrites: 0,
  };

  rmSync(targetRoot, { recursive: true, force: true });
  mkdirSync(targetRoot, { recursive: true });

  for (const file of files) {
    const relative = path.relative(postsRoot, file).replaceAll('\\', '/');
    const relativePathNoExt = relative.replace(/\.md$/, '');
    const { data, body: rawBody } = parsePost(file);

    const date = toShanghaiIso(data.date);
    const permalink = legacyPermalink(data.date, relativePathNoExt);
    const math = toMathBoolean(data.mathjax);
    const stripped = stripMoreMarker(normalizeLf(rawBody));
    const obsidian = stripObsidianImages(stripped);

    if (obsidian.removed.length > 0) {
      report.removedObsidianImages.push({
        file: relative,
        lines: obsidian.removed,
      });
    }

    const migrated: MigratedData = {
      title: data.title,
      description: data.description,
      date,
      updated: date,
      tags: [...new Set(data.tags)],
      categories: [...data.categories],
      permalink,
      math,
      draft: false,
    };

    const targetFile = path.join(targetRoot, relativePathNoExt + '.md');
    const targetDir = path.dirname(targetFile);
    mkdirSync(targetDir, { recursive: true });

    const output = `${renderFrontmatter(migrated)}${obsidian.body.trimEnd()}\n`;
    if (existsSync(targetFile) && readFileSync(targetFile, 'utf8') === output) {
      report.skippedWrites += 1;
      continue;
    }
    writeFileSync(targetFile, output, 'utf8');
  }

  // 有效图片资源迁移到旧 URL pathname；根相对引用无需改写。
  const imageSource = path.join(root, 'source', 'images', 'eigenvalue-error.png');
  const imageTargetDir = path.join(publicRoot, 'images');
  const imageTarget = path.join(imageTargetDir, 'eigenvalue-error.png');
  if (existsSync(imageSource)) {
    mkdirSync(imageTargetDir, { recursive: true });
    cpSync(imageSource, imageTarget);
    report.copiedAssets.push('images/eigenvalue-error.png');
  }

  return report;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const report = migrateContent();
  console.log(dump(report, { lineWidth: -1, noCompatMode: true }));
}
