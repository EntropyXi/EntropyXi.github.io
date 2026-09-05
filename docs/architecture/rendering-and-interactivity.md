# 渲染与交互

本文说明页面的静态渲染链路与渐进增强的客户端交互边界，来源为
`docs/superpowers/plans/2026-08-21-ayeez-ui-motion-refactor-plan.md` §6 与
`docs/architecture/adr/0001-math-rendering.md`。

## 静态渲染

- 默认使用 `.astro` 服务端渲染并输出静态 HTML，构建命令为 `npm run build`
  （`astro build` + `pagefind --site dist`），产物位于 `dist/`。
- 静态资源目录为 `astro-public/`（`astro.config.ts` 的 `publicDir`），构建时原样复制到 `dist/`。
- 文章页在构建期经 `remark-math` + `rehype-mathjax` 渲染数学，输出 `mjx-container`；
  浏览器端不加载 MathJax 脚本（ADR 0001）。每个构建期 MathJax SVG 使用以
  「数学公式：」开头的 `aria-label`，捕获数量与渲染数量不一致时构建失败。
- 非数学页不加载任何 MathJax 脚本。

## 服务端与客户端边界

- 客户端模块只处理：菜单、主题、搜索、复制、进度、reveal、指针和必要状态反馈。
- 核心信息不能只由 CSS `content`、Canvas 或客户端脚本生成。
- 所有核心链接必须是真实 `<a href>`；不使用 click handler 模拟导航。
- 按钮只处理动作，链接只处理导航，禁止混用语义。

## 客户端模块

`src/lib/client/` 保持 UI 无关，每个模块只承担一个职责并通过 `registerClientFeature`
幂等初始化（页面恢复或视图过渡后重复初始化安全）：

- 主题（2026-08-23 起永久锁定暗色，无切换按钮）：`theme.ts`；移动抽屉：`mobile-drawer.ts`（抽屉 DOM 在 `MobileDrawer.astro`，作为 body 直接子元素渲染——header 的 backdrop-filter 会劫持 fixed 后代的包含块）；页头状态：`site-header.ts`；公式横向裁切提示：`math-scroll-hint.ts`。
- 搜索：`search.ts`；代码复制：`code-copy.ts`；目录高亮：`post-toc.ts`。
- 阅读进度：`reading-progress.ts`；运动偏好：`motion-environment.ts`。
- 背景/显现/指针动效：`ambient-controller.ts`、`reveal-controller.ts`、`pointer-controller.ts`。
- 动效运行时（ADR 0003）：`motion/runtime.ts` 经 `motion/gsap-gate.ts`
  能力门控在 idle 时动态加载 Lenis 与 GSAP（ScrollTrigger/SplitText），
  运行时状态以 `data-gsap-active`/`data-lenis-active`/`data-hero-ready`
  写于 `<html>` 供 E2E 断言；页面动画由 `gsap.context` 承载并在视图过渡
  前 revert。

## 降级与可访问性

- 无 JavaScript 时，首页、文章正文与构建期公式不得隐藏；核心导航与内容可访问。
- `prefers-reduced-motion: reduce` 下关闭非必要运动与渐入，保留静态终态。
- 触控设备禁用磁吸光晕等 fine-pointer 专属效果。
- 视觉装饰 SVG 必须 `aria-hidden="true"`；有信息含义的图形必须提供文本等价物。

## CSS 架构

层次为 `tokens.css → global.css → motion.css → prose.css → 组件内部样式`。颜色、间距、
圆角、阴影、z-index 和时长必须优先使用 `tokens.css` 中的 token；`z-index` 来自层级 token
（background、content、sticky、overlay、modal）。详细规则见
`docs/superpowers/plans/2026-08-21-ayeez-ui-motion-refactor-plan.md` §6.4。
