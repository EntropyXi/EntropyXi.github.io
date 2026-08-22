# 编码风格

本文汇总本项目的编码规则，参考 Google 风格并针对 Astro 静态博客本地化，来源为
`docs/superpowers/plans/2026-08-20-astro-migration-plan.md` §7 与
`docs/superpowers/plans/2026-08-21-ayeez-ui-motion-refactor-plan.md` §7。

## TypeScript

- 使用 `astro/tsconfigs/strictest`；第三方类型阻塞时可先用 `strict` 并记录回到 `strictest` 的条件。
- 使用 ESM，禁止 CommonJS 新代码；默认 `const`，确需重新赋值用 `let`，禁止 `var`。
- 使用显式 `import type` 与 `export type`；公共函数边界、组件 props、配置对象和领域模型必须有明确类型。
- 用 `unknown` 接收未经验证的外部数据，经 schema 或类型守卫收窄；禁止非空断言作为常规手段。
- 禁止 `enum`（用 `as const` 对象或字面量联合）、namespace、装饰器、动态求值与修改全局对象。
- 异步函数必须处理失败路径，不允许悬空 Promise。
- 注释解释「为什么」和约束，不复述代码；一个文件只承载一个主要领域概念。

## 命名

- Astro 组件 `PascalCase.astro`；TypeScript 模块 `kebab-case.ts`，测试同名 `.test.ts`。
- 变量与函数 `camelCase`；类型与接口 `PascalCase` 且不用 `I` 前缀。
- 布尔值用 `is`、`has`、`can`、`should` 前缀；事件处理函数 `handleX`，回调 prop `onX`。
- CSS 类名小写连字符并表达结构/角色（如 `.post-card`），不用 `.green-box`、`.left2` 等表现型命名。
- `data-*` 属性用于脚本钩子时以领域命名（如 `data-search-dialog`），禁止用 CSS 类作脚本契约。

## Astro 组件

- frontmatter 按「类型导入、值导入、Props、解构、派生数据」排序；所有非平凡组件定义 `Props` 接口。
- 不在模板表达式写复杂排序/过滤/多层条件，先在 frontmatter 计算。
- 组件 scoped 样式只处理局部结构；token、排版和通用状态放全局层。
- 客户端 `<script>` 必须幂等；不使用 `set:html` 渲染不可信字符串。
- 页面和组件不得硬编码站点域名、作者信息或社交地址。

## HTML

- HTML5 doctype、`lang="zh-CN"`、正确 viewport 与 UTF-8 声明；语义化，不用废弃元素。
- 属性双引号；表单控件有 label，图标按钮有可访问名称；不使用正 tabindex（仅 `0`/`-1`）。
- DOM 顺序与阅读/键盘顺序一致；装饰 SVG 不被辅助技术重复朗读；外部资源全 HTTPS。

## CSS

- 使用原生 CSS，不引入 Sass/Less/CSS-in-JS/utility-first 框架。
- `tokens.css` 是颜色、间距、圆角、阴影、层级、动效时间与排版尺度的唯一 token 来源。
- 选择器低特异性，优先单类选择器与 `:where()`；禁止 ID 选择器做样式；嵌套最多两层。
- 移动优先；断点来自 token；light/dark 必须成对；关键动画提供 reduced-motion 静态替代。
- `!important` 默认禁止，仅第三方嵌入覆盖层允许并注释原因；不使用远程 CSS `@import`。

## JavaScript/浏览器脚本

- 新脚本用 TypeScript；事件监听具备清理或幂等重复绑定策略。
- 滚动监听用 passive listener；高频更新用 `requestAnimationFrame` 合并。
- 优先 IntersectionObserver/ResizeObserver，不在滚动事件中遍历大量元素。
- localStorage 只保存非敏感偏好，键名带站点前缀和版本；查询 DOM 后处理元素不存在的情况。
- 功能检测优先于浏览器识别；动态生成 HTML 用 DOM API 或模板，不拼接不可信字符串。

## 配置与文档

- JSON key 用 `camelCase`；配置必须有 schema 或 TypeScript 类型。
- `package-lock.json` 必须提交，CI 用 `npm ci`；依赖升级单独提交并记录构建/视觉回归。
- 文档规则见 `docs/superpowers/plans/2026-08-20-astro-migration-plan.md` §7.8（每篇单 H1、命令可复制并说明目录与预期结果等）。
