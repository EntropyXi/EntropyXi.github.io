# 内容模型

本文定义文章 Content Collection 的 frontmatter schema、URL 生成与 Markdown 规则，
来源为 `docs/superpowers/plans/2026-08-20-astro-migration-plan.md` §6。文章存放于
`src/content/blog/`，按分类目录组织，生产构建排除草稿。

## Frontmatter Schema

| 字段          | 类型     | 约束                                                   |
| ------------- | -------- | ------------------------------------------------------ |
| `title`       | string   | 必填非空字符串                                         |
| `description` | string   | 必填，去首尾空白后非空，用于卡片、SEO 和 RSS           |
| `date`        | string   | 必填，ISO 8601 并显式带 `+08:00`，避免 CI 时区漂移     |
| `updated`     | string   | 可选；缺失时等于 `date`，不得从文件 mtime 推导         |
| `tags`        | string[] | 必填，去重后至少一项                                   |
| `categories`  | string[] | 必填，保留从大类到子类的顺序                           |
| `permalink`   | string   | 必填且全站唯一，不含首尾斜杠，发布后冻结               |
| `math`        | boolean  | 严格布尔，默认 `false`；`true` 时由构建期 MathJax 渲染 |
| `draft`       | boolean  | 严格布尔，默认 `false`；生产构建排除草稿               |
| `cover`       | string   | 可选；只允许站内资源或 HTTPS URL                       |

未在 schema 定义的字段默认报错，避免拼写错误静默进入生产。字段约束由
`scripts/audit-content.ts` 在源码层校验（title/description/date/updated/tags/categories/permalink/math/draft）。

## URL 生成规则

- 文章路由通过 catch-all 静态路由读取 `permalink`，输出 pathname 恒为 `/${permalink}/`。
- `permalink` 一旦发布，不得因改标题、移动文件或调整分类而改变。
- 分类和标签显示名使用原始中文；路由 slug 经过统一、可逆或显式映射。
- 历史 URL 冻结于 `tests/fixtures/legacy-baseline.json`；每篇 `permalink` 的 pathname
  必须出现在旧站 manifest 中（由 `scripts/audit-content.ts` 校验）。

## Markdown 规则

- 使用 UTF-8 和 LF；仓库用 `.gitattributes` 固定行尾。
- 只使用 ATX 标题（`#`）；正文从 `##` 开始，文章标题由 `PostLayout` 输出。
  标题层级连续，每个页面只有 layout 输出的一个 `h1`。
- 代码块必须使用 fenced code block 并声明语言。
- 优先 Markdown，避免内联 HTML；确需 HTML 时只允许白名单标签，不含 `style` 或脚本事件属性。
- 图片使用标准 Markdown 或受控组件生成的语义 HTML，不使用 Obsidian Wiki 图片语法。
- 数学块前后保留空行；禁止把公式放进 HTML 标签属性。
- 行内公式 `$...$`，块级公式 `$$...$$`；`math: true` 的文章由构建期 MathJax 渲染。

## 摘要与排序

- 首页卡片摘要只使用 `description`，不从正文截断；`<!-- more -->` 无运行时意义。
- 文章按 `date` 倒序；日期相同则按 `permalink` 稳定排序。
- 归档按年月分组；分类按 `categories` 每级生成聚合；标签按精确字符串聚合。
