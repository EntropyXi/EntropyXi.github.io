import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSON_SCHEMA, load as yamlLoad } from 'js-yaml';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const legacyPostsRoot = path.join(root, 'source', '_posts');
const migratedRoot = path.join(root, 'src', 'content', 'blog');
const publicRoot = path.join(root, 'astro-public');
const baselineFile = path.join(root, 'tests', 'fixtures', 'legacy-baseline.json');

interface LegacyHtmlRecord {
  pathname: string;
  kind: string;
  title: string;
}

interface LegacyBaseline {
  html: LegacyHtmlRecord[];
  summary: { postCount: number };
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

const errors: string[] = [];

function walkMarkdown(dir: string, files: string[] = []): string[] {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walkMarkdown(full, files);
    else if (entry.name.endsWith('.md')) files.push(full);
  }
  return files;
}

function parseFrontmatter(
  file: string,
): { data: Record<string, unknown>; body: string } {
  const raw = readFileSync(file, 'utf8');
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!match || match[1] === undefined) throw new Error(`Missing frontmatter block: ${file}`);
  return {
    data: yamlLoad(match[1], { schema: JSON_SCHEMA }) as Record<string, unknown>,
    body: raw.slice(match[0].length),
  };
}

function normalizeBodyForHash(body: string): string {
  return body
    .replace(/\r\n/g, '\n')
    .replace(/^<!-- more -->[ \t]*\r?\n?/gm, '')
    .split('\n')
    .filter((line) => {
      const trimmed = line.trim();
      return !(trimmed.startsWith('![[[') || (trimmed.startsWith('![[') && trimmed.endsWith(']]')));
    })
    .join('\n')
    .trimEnd();
}

function sha256(text: string): string {
  return createHash('sha256').update(text).digest('hex');
}

function asMigratedData(data: Record<string, unknown>, file: string): MigratedData {
  const str = (name: string): string => {
    const value = data[name];
    if (typeof value !== 'string' || value.trim() === '') {
      errors.push(`${file}: ${name} must be a non-empty string`);
      return '';
    }
    return value;
  };
  const arr = (name: string): string[] => {
    const value = data[name];
    if (!Array.isArray(value) || value.length === 0) {
      errors.push(`${file}: ${name} must be a non-empty array`);
      return [];
    }
    for (const item of value) {
      if (typeof item !== 'string') errors.push(`${file}: ${name} items must be strings`);
    }
    return value.filter((item) => typeof item === 'string');
  };
  return {
    title: str('title'),
    description: str('description'),
    date: str('date'),
    updated: str('updated'),
    tags: arr('tags'),
    categories: arr('categories'),
    permalink: str('permalink'),
    math: data.math === true,
    draft: data.draft === false,
  };
}

function main(): void {
  const baseline = JSON.parse(readFileSync(baselineFile, 'utf8')) as LegacyBaseline;
  const legacyArticles = baseline.html.filter((record) => record.kind === 'article');
  const legacyPathnames = new Set(
    legacyArticles.map((record) => decodeURIComponent(record.pathname)),
  );

  const migratedFiles = walkMarkdown(migratedRoot);
  if (migratedFiles.length !== baseline.summary.postCount) {
    errors.push(
      `expected ${baseline.summary.postCount} migrated posts, found ${migratedFiles.length}`,
    );
  }

  const legacyFiles = walkMarkdown(legacyPostsRoot);
  const legacyByRelative = new Map<string, string>();
  for (const file of legacyFiles) {
    const relative = path.relative(legacyPostsRoot, file).replaceAll('\\', '/');
    legacyByRelative.set(relative, file);
  }

  const seenPermalinks = new Set<string>();
  const seenDates: Array<{ file: string; oldDate: string; newDate: string }> = [];

  for (const file of migratedFiles) {
    const relative = path.relative(migratedRoot, file).replaceAll('\\', '/');
    const relativePathNoExt = relative.replace(/\.md$/, '');
    const { data, body } = parseFrontmatter(file);
    const migrated = asMigratedData(data, file);

    if (migrated.date !== migrated.updated) {
      errors.push(`${file}: updated must equal date during migration`);
    }
    if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00$/.test(migrated.date)) {
      errors.push(`${file}: date must be ISO 8601 with +08:00`);
    }

    const expectedPathname = `/${migrated.permalink}/`;
    if (!legacyPathnames.has(expectedPathname)) {
      errors.push(`${file}: permalink pathname ${expectedPathname} not found in legacy URL manifest`);
    }

    if (seenPermalinks.has(migrated.permalink)) {
      errors.push(`${file}: duplicate permalink ${migrated.permalink}`);
    }
    seenPermalinks.add(migrated.permalink);

    const migratedHash = sha256(normalizeBodyForHash(body));
    const legacyRelative = relativePathNoExt + '.md';
    const legacyFile = legacyByRelative.get(legacyRelative);
    if (!legacyFile) {
      errors.push(`${file}: no legacy source file for ${legacyRelative}`);
      continue;
    }
    const legacy = parseFrontmatter(legacyFile);
    const legacyHash = sha256(normalizeBodyForHash(legacy.body));
    if (legacyHash !== migratedHash) {
      errors.push(`${file}: normalized body hash mismatch`);
    }

    seenDates.push({
      file: relative,
      oldDate: String(legacy.data.date ?? ''),
      newDate: migrated.date,
    });
  }

  // 检查迁移后正文不残留 Obsidian 图片语法与 more 标记。
  for (const file of migratedFiles) {
    const raw = readFileSync(file, 'utf8');
    if (raw.includes('<!-- more -->')) errors.push(`${file}: contains <!-- more -->`);
    if (/!\[\[[^\]]+\]\]/.test(raw)) errors.push(`${file}: contains Obsidian image syntax`);
  }

  // 检查本地图片引用都有对应资源文件。
  for (const file of migratedFiles) {
    const { body } = parseFrontmatter(file);
    const refs: string[] = [
      ...Array.from(body.matchAll(/!\[[^\]]*\]\(([^)]+)\)/g), (m) => m[1] ?? ''),
      ...Array.from(body.matchAll(/<img\b[^>]*src=["']([^"']+)["'][^>]*>/g), (m) => m[1] ?? ''),
    ].filter((ref): ref is string => ref !== '');
    for (const ref of refs) {
      const target = (ref.split('#')[0] ?? '').split('?')[0] ?? '';
      if (target.startsWith('http://') || target.startsWith('https://')) continue;
      const resolved = target.startsWith('/')
        ? path.join(publicRoot, target.slice(1))
        : path.resolve(path.dirname(file), target);
      if (!existsSync(resolved)) {
        errors.push(`${file}: image reference not found: ${ref}`);
      }
    }
  }

  const dateComparison = seenDates.map((entry) => ({
    ...entry,
    oldToUtc: new Date(`${entry.oldDate.replace(' ', 'T')}+08:00`).getTime(),
    newToUtc: new Date(entry.newDate).getTime(),
  }));

  if (process.argv.includes('--write-report')) {
    const reportPath = path.join(root, 'docs', 'migration', 'phase-2-date-comparison.md');
    const rows = dateComparison
      .map((entry) => {
        const status = entry.oldToUtc === entry.newToUtc ? '通过' : '不通过';
        return `| ${entry.file} | ${entry.oldDate} | ${entry.newDate} | ${status} |`;
      })
      .join('\n');
    const report = [
      '# 阶段 2 日期逐篇对比报告',
      '',
      '本报告由 `scripts/audit-content.ts --write-report` 生成。',
      '旧站日期解释为 Asia/Shanghai（+08:00）后与迁移后的 ISO 8601 时间戳逐篇比较。',
      '',
      '| 文章 | 旧站 frontmatter 日期 | 迁移后日期 | 结论 |',
      '| --- | --- | --- | --- |',
      rows,
      '',
    ].join('\n');
    mkdirSync(path.dirname(reportPath), { recursive: true });
    writeFileSync(reportPath, `${report}\n`, 'utf8');
  }

  console.log(
    JSON.stringify(
      {
        migratedPosts: migratedFiles.length,
        legacyPosts: legacyFiles.length,
        permalinksChecked: seenPermalinks.size,
        dateComparison: dateComparison.map((entry) => ({
          file: entry.file,
          oldDate: entry.oldDate,
          newDate: entry.newDate,
          utcEqual: entry.oldToUtc === entry.newToUtc,
        })),
      },
      null,
      2,
    ),
  );

  if (errors.length > 0) {
    for (const error of errors) console.error(`CONTENT: ${error}`);
    process.exit(1);
  }
}

main();
