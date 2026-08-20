# 阶段 0 执行记录

本文记录 `Hexo → Astro 迁移计划` 阶段 0 的执行证据。计划原文见
`docs/superpowers/plans/2026-08-20-astro-migration-plan.md` 第 13 节。

## 起始状态

- 分支：`codex/astro-migration`（此前已创建）。
- 阶段 0 起始 commit：`0780179a4e3054eb532dc5868c6929306fb0cfe5`
  （`test: freeze Hexo migration baseline`）。
- 阶段 0 有若干产出在此前已经完成；本记录同时核对既有产物与本轮新增证据。

## 任务执行状态

- [x] 确认工作树干净并记录当前 commit。（本轮补记）
- [x] 运行现有 `npm run check`，保存通过结果。（本轮复核）
- [x] 生成旧站 manifest。（此前已完成：`tests/fixtures/legacy-baseline.json`）
- [x] 用生产站爬取结果与本地 Hexo manifest 交叉核对。（此前已完成：
      `docs/migration/baseline/production-verification.json`）
- [x] 审计 30 篇 frontmatter。（此前已建立 `scripts/audit-frontmatter.js`，本轮复核通过）
- [x] 扫描全部数学分隔符和环境。（此前已完成，结果固化在 legacy baseline 的 `summary.math`）
- [x] 保存首页、归档、分类、标签、关于和 5 篇复杂公式文章基准截图。（此前已完成：
      `docs/migration/baseline/hexo-next/`）
- [x] 记录当前 Giscus 配置、搜索、RSS、Sitemap 与部署行为。
      （本轮完成：`docs/migration/baseline/legacy-behavior-freeze.md`）
- [x] 标记 Hexo 可回滚点 `pre-astro-migration`。
      （本轮完成：annotated tag 指向 `0780179a4e3054eb532dc5868c6929306fb0cfe5`）
- [x] 本阶段不得改变生产部署。（未修改 `.github/workflows/deploy.yml` 与 `source` 分支）

## 此前已有产物的核对结果

- `tests/fixtures/legacy-baseline.json`
  - schemaVersion `1`，generatedFromCommit `6a608104915fd163a857d064ac7ffe03f8d52f20`。
  - summary：81 HTML、48 静态资源、30 文章、29 数学文章。
  - 数学：行内 `$...$` 854 对，显示 `$$...$$` 468 对，未闭合 0；
    `\(...\)` 与 `\[...\]` 使用数为 0。
- `docs/migration/baseline/production-verification.json`
  - 81/81 HTML 状态码、最终 pathname、canonical 全部通过。
  - 44/48 静态资源内容 SHA-256 完全一致；4 个差异为
    `atom.xml`、`search.xml`、`sitemap.xml`、NexT `css/main.css`，
    代表本地工作树与线上版本内容不同，需在 Astro 验收时用新的静态契约替代。
- `docs/migration/baseline/hexo-next/`
  - 已包含首页、归档、分类、标签、关于的桌面截图，首页移动截图，
    以及 DDPM、SDE、SR3、DDIM、ResShift 5 篇复杂公式文章截图与移动数学页截图。
- `scripts/audit-frontmatter.js`
  - 本轮 `npm run check` 输出 `FRONTMATTER: all posts passed`。

## 本轮新增证据

### 旧站构建检查

- 命令：`npm ci`（按已提交 `package-lock.json` 确定性安装，265 packages）。
- 命令：`npm run check`。
- 结果：退出码 `0`。
  - `FRONTMATTER: all posts passed`
  - `129 files generated in 2.34 s`
  - `VERIFY: 81 HTML files passed`
- 结论：Hexo 基线在阶段 0 起始 commit 上可重复构建并通过既有验证。
