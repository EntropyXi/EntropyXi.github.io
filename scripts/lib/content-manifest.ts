import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { JSON_SCHEMA, load as yamlLoad } from "js-yaml";

/**
 * 文章源清单读取工具。
 *
 * 迁移后的内容既有冻结的历史文章，也有迁移后新增的文章，两个内容审计脚本
 * （`audit-content.ts`、`audit-output.ts`）都需要同一份源清单，因此把文件
 * 系统读取集中在这里，避免两处各写一遍 frontmatter 解析。
 */

export interface PostSource {
  /** 文章 Markdown 的绝对路径。 */
  file: string;
  /** frontmatter 解析结果。 */
  data: Record<string, unknown>;
  /** 去除 frontmatter 之后的正文。 */
  body: string;
}

/** 递归收集目录下的 Markdown 文件，跳过点开头的目录。 */
export function walkMarkdown(dir: string, files: string[] = []): string[] {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith(".")) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walkMarkdown(full, files);
    else if (entry.name.endsWith(".md")) files.push(full);
  }
  return files;
}

/** 拆分 frontmatter 与正文；缺少 frontmatter 块时抛错。 */
export function parseFrontmatter(file: string): {
  data: Record<string, unknown>;
  body: string;
} {
  const raw = readFileSync(file, "utf8");
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!match || match[1] === undefined) {
    throw new Error(`Missing frontmatter block: ${file}`);
  }
  return {
    data: yamlLoad(match[1], { schema: JSON_SCHEMA }) as Record<string, unknown>,
    body: raw.slice(match[0].length),
  };
}

/** 读取全部文章源文件，按路径字典序排序，保证多次运行结果一致。 */
export function readPostSources(root: string): PostSource[] {
  return walkMarkdown(root)
    .sort()
    .map((file) => ({ file, ...parseFrontmatter(file) }));
}
