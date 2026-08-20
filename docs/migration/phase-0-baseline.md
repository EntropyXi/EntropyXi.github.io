# 阶段 0：Hexo 基线捕获

本文档说明如何生成、理解以及维护 `tests/fixtures/legacy-baseline.json`。

## 生成命令

先确保 Hexo 已生成最新 `public/`：

```bash
npm run build
```

然后捕获基线：

```bash
node scripts/capture-baseline.js
```

该命令会检查 `public/index.html` 是否存在，并将结果写入：

```text
tests/fixtures/legacy-baseline.json
```

## 验证命令

```bash
node --test tests/legacy-baseline/*.test.js
```

重复运行捕获命令应产生完全相同的文件：

```bash
node scripts/capture-baseline.js
node scripts/capture-baseline.js
git diff --exit-code tests/fixtures/legacy-baseline.json
```

## 输出结构

顶层字段：

| 字段 | 说明 |
| --- | --- |
| `schemaVersion` | 基线结构版本，当前为 `1` |
| `generatedFromCommit` | 生成时 `git rev-parse HEAD` 的结果；不是 Git 仓库时返回 `null` |
| `html` | `public/` 下所有 HTML 的元数据数组，按 `relativeFile` 排序 |
| `assets` | `public/` 下非 HTML 文件清单（含 URL、相对路径、字节大小与 SHA-256），按 `relativeFile` 排序 |
| `posts` | `source/_posts/` 下所有 Markdown 文章的清单，按 `sourceRelative` 排序 |
| `summary` | 数量与分类汇总 |

### HTML 记录字段

每个 HTML 记录包含：

- `pathname`：从 canonical URL 解析出的路径，保留百分号编码
- `relativeFile`：相对 `public/` 的 HTML 文件路径
- `kind`：`home` / `article` / `archive` / `category` / `tag` / `about` / `page` / `other`
- `title`：优先取 `og:title`，否则取 `<title>` 文本
- `description`：`<meta name="description">` 内容（已解码基础 HTML 实体）
- `canonical`：`<link rel="canonical">` 的完整 URL
- `publishedTime`：文章页 `article:published_time`；非文章页为空字符串
- `tags`：文章页所有 `article:tag` meta 的顺序数组

### 静态资源记录字段

- `pathname`：逐路径段 URL 编码后的站点绝对路径
- `relativeFile`：相对 `public/` 的文件路径
- `size`：文件字节数
- `sha256`：资源内容 SHA-256，用于迁移后等价检查

### 文章记录字段

每个文章记录包含：

- `sourceRelative`：相对 `source/` 的 Markdown 路径（例如 `_posts/深度学习/demo.md`）
- `title`、`description`：frontmatter 字符串
- `date`：frontmatter 中原始 `date` 字符串的稳定表示
- `tags`、`categories`：frontmatter 数组
- `mathjaxRaw`：frontmatter 中 `mathjax` 的原始 JSON 可表达值；缺失为 `null`
- `mathEnabled`：原值为布尔 `true` 或字符串 `"true"` 时为 `true`
- `normalizedBodySha256`：正文 SHA-256，只做 CRLF/CR→LF 并移除 `<!-- more -->` 标记
- `math`：忽略 fenced/inline code 后的数学扫描结果，包含 `$...$`、`$$...$$`、`\(...\)`、`\[...\]` 成对公式数、未闭合数、LaTeX environment 与 command 计数

## 确定性与维护约定

- 所有数组均按相对路径字典序排序，重复运行结果一致。
- 输出使用 UTF-8、LF 换行，且不写入当前时间。
- `tests/fixtures/legacy-baseline.json` 是迁移基线，**禁止手工编辑**。
- 如需更新基线，应重新生成 Hexo `public/` 后运行捕获命令，并审查 diff。

## 与生产站交叉核对

本地基线生成后运行：

```bash
node scripts/check-production-baseline.js --output docs/migration/baseline/production-verification.json
```

该命令以基线 canonical 推导生产站根地址，并检查全部 HTML 的 HTTP 状态、最终 pathname 和 canonical；同时检查静态资源状态与 SHA-256。报告不写当前时间，结果不变时可确定性重跑。任何失败都必须人工判断是本地构建与线上版本差异、旧站既有问题还是迁移阻塞项，不得静默忽略。

当前生产核验结论记录在 `baseline/production-verification.json`：81/81 个 HTML 页面均通过状态码、最终路径与 canonical 检查；44/48 个静态资源内容完全一致。4 个内容哈希差异来自 `atom.xml`、`search.xml`、`sitemap.xml` 与 NexT 的 `css/main.css`，表示当前工作树构建产物与线上已部署版本存在内容差异。它们不影响旧 URL 存活性，但 Astro 迁移验收时必须用新的静态契约替代，不能把差异静默当作成功。

## 视觉基线

桌面端、移动端和长数学文章截图位于 `baseline/hexo-next/`，清单与使用边界见该目录的 `README.md`。这些截图只用于核对内容层级、可读性和响应式完整性，不是 NexT 动画或像素级复刻目标。
