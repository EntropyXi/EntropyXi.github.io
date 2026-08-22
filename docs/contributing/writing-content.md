# 内容写作指南

本文说明如何新增或修改文章并保持内容/URL/数学不变，来源为
`docs/superpowers/plans/2026-08-20-astro-migration-plan.md` §6 与
`docs/architecture/content-model.md`。

## 新建文章

在 `src/content/blog/<分类>/` 下创建 `.md` 文件，frontmatter 使用：

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

`permalink` 全站唯一、不含首尾斜杠，发布后冻结（不得因改标题、移动文件或改分类而改变）。

## 标题规则（每页单 H1）

- 文章标题由 `PostLayout` 输出为页面唯一的 `h1`；正文不得再写一级标题。
- 正文从 `##` 开始，层级连续（`##` → `###` → `####`），不跳级。
- 此规则由 `scripts/audit-content.ts`（源码层禁止正文 H1）与
  `scripts/audit-output.ts`（产物层断言每篇页面恰有一个 `<h1>`）双重校验。

## 数学

- 行内公式 `$...$`，块级公式 `$$...$$`；`math: true` 的文章由构建期 MathJax 渲染。
- 数学块前后保留空行；禁止把公式放进 HTML 标签属性。
- 不要改动历史文章的 TeX 分隔符；不要引入 `\(...\)`/`\[...\]`（ADR 0001）。
- 不要在 `$$` 与 CJK 正文之间直接相邻（正文会被吞入公式），也不要出现空公式 `$$$$`；
  二者均由 `scripts/audit-content.ts` 检测。

## 图片与链接

- 图片用标准 Markdown `![](path)` 或受控组件语义 HTML，不用 Obsidian Wiki 图片语法 `![[...]]`。
- 图片引用必须指向已存在的站内资源（由 `scripts/audit-content.ts` 校验）。
- 链接文本必须说明目标，不用孤立的「点击这里」。

## 发布前自检

在工作区根目录运行：

```bash
npm run check
```

预期结果：格式、Lint、类型、内容审计、单元测试、E2E、bundle 与产物审计全部通过，
无报错退出（见 `docs/contributing/testing.md`）。
