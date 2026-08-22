# EntropyXi 的个人博客

基于 Astro 7 的静态技术博客，部署在 GitHub Pages。

**线上地址**：https://entropyxi.github.io

## 技术栈

| 组件     | 说明                                           |
| -------- | ---------------------------------------------- |
| 静态生成 | Astro 7 SSG                                    |
| 内容     | Astro Content Collections                      |
| 数学     | 构建期 MathJax（remark-math + rehype-mathjax） |
| 搜索     | Pagefind 中文静态搜索                          |
| 评论     | 暂不加载第三方评论脚本                         |
| CI/CD    | GitHub Actions → GitHub Pages                  |

## 本地开发

```bash
npm ci
npm run dev
```

打开 `http://localhost:4321`。

## 质量门禁

```bash
npm run check
```

`npm run check` 依次执行：格式检查、Lint、类型检查、内容审计、单元测试、
E2E（内部会先完整构建并生成 Pagefind 索引）、输出审计。

## 构建与预览

```bash
npm run build
npm run preview
```

- 构建输出：`dist/`
- Astro 静态源目录：`astro-public/`（旧站兼容图片等静态资源）

## 写一篇新笔记

在 `src/content/blog/` 对应分类目录下创建 `.md` 文件，frontmatter 使用：

```yaml
---
title: 文章标题
description: 摘要
date: 2026-08-20T10:00:00+08:00
updated: 2026-08-20T10:00:00+08:00
tags:
  - 深度学习
categories:
  - 深度学习
permalink: 2026/08/20/深度学习/文章标题
math: true
draft: false
---
```

规则：`permalink` 发布后冻结；行内公式 `$...$`，块级公式 `$$...$$`；
`math: true` 的文章由构建期 MathJax 渲染。

## 部署

推送 `source` 分支会触发 `.github/workflows/deploy.yml` 构建并部署 `dist/`。

## 目录结构

```text
Blog_file/
├── astro-public/        # 静态资源，构建时原样复制到 dist/
├── scripts/             # 内容审计、输出审计、vendor 复制脚本
├── src/
│   ├── components/      # 组件
│   ├── content/blog/    # 文章（Content Collections）
│   ├── data/            # 站点配置
│   ├── layouts/         # BaseLayout、PostLayout
│   ├── lib/             # 无 UI 领域逻辑
│   ├── pages/           # 路由页面与 XML endpoint
│   └── styles/          # 全局与排版样式
├── tests/               # E2E 与单元测试
└── astro.config.ts
```

## 迁移记录

- 计划：`docs/superpowers/plans/2026-08-20-astro-migration-plan.md`
- ADR：`docs/architecture/adr/`
- 回滚：`docs/migration/rollback.md`（回滚 tag `pre-astro-migration`）

## 架构与贡献文档

- 架构概览：`docs/architecture/overview.md`
- 内容模型：`docs/architecture/content-model.md`
- 渲染与交互：`docs/architecture/rendering-and-interactivity.md`
- 编码风格：`docs/contributing/coding-style.md`
- 内容写作：`docs/contributing/writing-content.md`
- 测试：`docs/contributing/testing.md`

## Phase 8 重构记录

- 视觉与动效：全站深色赛博风格已重构完毕，引入磁吸光标、分层环境背景、全屏 Hero，满足 prefers-reduced-motion 降级要求。
- 最新部署状态：本地已完成 100% Phase 8 质量门禁与浏览器矩阵验收 (无 JS, 触屏, Zoom 等)。
