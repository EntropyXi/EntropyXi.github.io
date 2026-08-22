# 架构概览

本文概述 EntropyXi 个人博客的架构边界与依赖方向，作为 `docs/architecture/` 的入口。
详细决策见各 ADR，编码与内容规则见 `docs/contributing/`，实施背景见
`docs/superpowers/plans/2026-08-20-astro-migration-plan.md` §5–§7 与
`docs/superpowers/plans/2026-08-21-ayeez-ui-motion-refactor-plan.md` §6。

## 技术栈

| 组件     | 说明                                             |
| -------- | ------------------------------------------------ |
| 静态生成 | Astro 7 SSG（`.astro` 服务端渲染输出静态 HTML）  |
| 内容     | Astro Content Collections（`src/content/blog/`） |
| 数学     | 构建期 MathJax（remark-math + rehype-mathjax）   |
| 搜索     | Pagefind 中文静态搜索（`pagefind --site dist`）  |
| 部署     | GitHub Actions → GitHub Pages                    |

本项目明确不引入 React、Vue、Svelte 等客户端 UI 框架；仅用原生 Astro 模板 +
纯 TypeScript 模块 + 标准语义 CSS（见 `docs/superpowers/plans/2026-08-21-ayeez-ui-motion-refactor-plan.md` §6.2）。

## 分层与依赖方向

```text
pages → layouts → components → lib/data
             └──→ styles/tokens
tests → public contracts / rendered behavior
scripts → content/schema/build output
```

- `src/pages/` 只负责编排路由数据和页面组合。
- `src/layouts/` 负责跨页面结构、metadata 和 slot（`BaseLayout`、`PostLayout`）。
- `src/components/` 按职责拆分；不得反向导入 page 或 layout。
- `src/lib/` 保持 UI 无关，不得导入 `.astro`、全局 DOM 或组件 CSS。
- `src/data/` 只存站点配置常量（如 `src/data/site.ts` 的 URL、标题、语言）。
- `src/styles/` 负责全局令牌、排版与跨组件层。
- `scripts/` 只做构建/审计任务，不进入浏览器运行时。
- 禁止循环依赖；跨层例外必须先写 ADR。

## 关键架构决策

- 文章页使用构建期 MathJax，产物输出 `mjx-container`，浏览器不加载 MathJax 脚本（ADR 0001）。
- 文章图片暂不引入灯箱（ADR 0002）。
- 每篇文章页面只有 layout 输出的一个 `h1`，正文从 `##` 开始。

## ADR 触发条件

出现以下任一情况必须先写 ADR 并经技术负责人放行（`docs/superpowers/plans/2026-08-21-ayeez-ui-motion-refactor-plan.md` §6.6）：

- 引入新的运行时依赖或客户端框架；
- 修改 permalink、分页、聚合路由、Content Collection schema、Astro SSG、Markdown/数学渲染、搜索或部署链路；
- 引入 Canvas/WebGL、大型动画库或第三方统计/评论脚本；
- 超过任一性能预算；改变主题存储键、页面初始化顺序、部署或 GitHub Pages 行为。

## 相关文档

- 内容模型：`docs/architecture/content-model.md`
- 渲染与交互：`docs/architecture/rendering-and-interactivity.md`
- 决策记录：`docs/architecture/adr/`
- 编码风格：`docs/contributing/coding-style.md`
- 内容写作：`docs/contributing/writing-content.md`
- 测试：`docs/contributing/testing.md`
