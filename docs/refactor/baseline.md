# AyeezBlog 风格重构：阶段 0 基线

## 1. 记录范围

- 日期：2026-08-21（Asia/Shanghai）
- 分支：`codex/ayeez-ui-motion-refactor`
- 起点 commit：`1462a23`
- Node.js：`v24.13.0`
- npm：`11.12.1`
- Astro：`7.2.4`
- 输出模式：Astro SSG，`trailingSlash: "always"`
- 静态资源目录：`astro-public/`
- 内容：30 篇 Markdown 文章

## 2. 全量质量门禁

执行：`npm run check`

结果：通过。

- Prettier：通过；
- ESLint + Stylelint：通过；
- Astro Check：56 个文件，0 error / 0 warning / 0 hint；
- 内容审计：30/30；
- Vitest：2 个文件、5 个测试通过；
- Playwright：10/10 通过；
- 产物审计：81 个历史页面、30 篇文章通过。

公式基线：生产文章使用 `remark-math` + `rehype-mathjax` 构建期输出；`dist/` 中共找到 1289 个 `mjx-container`。真实 DDIM 文章 E2E 和 `/dev/math-spike/` 参考页 E2E 均通过。

## 3. 构建产物与体积

- `dist/`：约 33 MiB、243 个文件、84 个 HTML 文件；
- 全部 CSS gzip 合计：18,013 bytes；
- 全部 JS gzip 合计：5,508,153 bytes；
- JS 总量主要来自 `/dev/math-spike/` 所需的客户端 MathJax vendor 和 Pagefind，不代表普通生产页面实际下载；
- 最大文章 HTML：ResShift 约 800 KiB，DDIM 约 794 KiB；体积主要来自构建期 SVG 公式。

阶段 0.5 必须评估是否从生产产物移除仅用于开发验证的 `/dev/math-spike/` 和客户端 MathJax vendor。

## 4. Lighthouse 移动基线

工具：Lighthouse 13.4.1，默认移动审计；本地 Astro 静态预览。原始 JSON 位于 `docs/refactor/lighthouse/`。

| 页面          | Performance | Accessibility | Best Practices | SEO |     LCP | CLS |  TBT |
| ------------- | ----------: | ------------: | -------------: | --: | ------: | --: | ---: |
| 首页          |         100 |           100 |             96 | 100 |  907 ms |   0 | 0 ms |
| 搜索          |         100 |           100 |             96 | 100 |  905 ms |   0 | 0 ms |
| 归档          |         100 |           100 |             96 | 100 |  905 ms |   0 | 0 ms |
| DDIM 公式文章 |         100 |            98 |             96 | 100 | 1354 ms |   0 | 0 ms |

已知审计项：

- 全页面缺少 `/favicon.ico`，产生控制台 404，Best Practices 为 96；
- DDIM 原文标题从页面 `h1` 跳到正文 `h3/h5`，Lighthouse 报 heading order；这是历史内容语义问题，不能用 CSS 隐藏；
- 构建期 MathJax SVG 使用 `role="img"` 但没有可访问名称，axe 标记为 serious；必须在阶段 0.5 修复并增加自动测试。

第一次 Lighthouse 运行在清理临时目录时出现 Windows `EPERM`，但报告已经完整生成；其余运行成功。该环境噪声不影响分数，最终 CI 以 Ubuntu GitHub Actions 为准。

## 5. 浏览器视觉基线

- 脚本：`npm run capture:ui-baseline`；
- 页面：首页、DDIM 文章、归档、分类、标签、搜索、关于、404；
- 矩阵：1440×900 / 390×844 × 暗色 / 亮色，共 32 张；
- 输出：`audit-screenshots/phase-0/`；
- 元数据：`audit-screenshots/phase-0/index.json`。

Chrome 人工抽查确认当前首页和 DDIM 公式文章在桌面/移动均可读，未发现页面级横向溢出。

## 6. 阶段 0 发现的基线缺陷

1. `SiteHeader` 同时渲染两个 `id="theme-toggle"`；脚本只绑定第一个元素，移动端可见主题按钮无法切换。
2. 当前没有 reduced-motion、JavaScript disabled、390 touch、360 overflow、200% zoom 的独立 Playwright 项目。
3. 搜索无 JavaScript 时没有解释或替代入口。
4. MathJax SVG 缺少可访问名称。
5. `/favicon.ico` 404。
6. `src/pages/dev/math-spike.astro` 和整套客户端 MathJax vendor 进入生产产物。
7. 顶层 `audit-screenshots/` 有 4 张上一轮截图；保留原位作为 legacy 证据，不与 `phase-0/` 混用。
8. `.deploy_git/` 约 16 MiB、未被 Git 跟踪；本轮不自动删除用户数据。

这些问题均已登记，不允许由后续视觉改动静默掩盖。

## 7. 阶段 0 结论

当前 Astro 构建、内容、路由、Pagefind、构建期数学和 GitHub Pages 工作流基线可用。阶段 0 允许完成；阶段 1 之前必须先完成阶段 0.5 的数学/生产产物决策，并在阶段 2 修复主题生命周期与自动化矩阵。
