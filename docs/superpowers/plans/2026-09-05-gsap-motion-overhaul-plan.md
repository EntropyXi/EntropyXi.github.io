# GSAP 动效体系升级计划（Lenis + GSAP + ScrollTrigger + SplitText + ClientRouter）R2 修订版

- 状态：R1/R2 双高精度审查完成，29 项意见全部闭合（见 §12），可执行
- 日期：2026-09-05
- 关联：ADR 0003（本计划 Phase 0 产出）；`docs/refactor/quality-gates.md`；
  `docs/architecture/rendering-and-interactivity.md`
- 用户决策记录：站点所有者已明确解除第一方 JS 体积预算（原 24 KB），允许引入
  动效库与框架；「引入大型动画库」按 `docs/architecture/overview.md` 仍需 ADR。

## 1. 背景与问题诊断

站点当前动效「僵硬」的七个具体来源（均已对照源码确认）：

| #   | 来源                                                                                                  | 证据位置                                          |
| --- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| D1  | Hero 零入场编排，加载即静态终态                                                                       | `src/components/visual/Hero.astro` 无任何入场动画 |
| D2  | Reveal 单一公式（opacity+translateY 16px、450ms、单缓动），stagger 写死 nth-child(1-6)，第 7 项起同帧 | `src/styles/motion.css` §1                        |
| D3  | 无任何滚动链接动画：壁纸 `position: fixed` 惰性、页头 24px 阈值二值切换                               | `AmbientBackground.astro`、`site-header.ts`       |
| D4  | 氛围层静止：扫描线 12s 线性循环、光晕球无 keyframes、流线静态虚线                                     | `AmbientBackground.astro`                         |
| D5  | 悬停语言单调：全站 translateY(-2/-3px)；磁吸仅三处白名单（logo 0.22 / 导航 0.2 / 卡片 0.05）          | `motion.css:129-173`、`pointer-controller.ts`     |
| D6  | 页面导航硬切，无视图过渡                                                                              | `BaseLayout.astro` 无 ClientRouter                |
| D7  | 缓动扁平：3 条 cubic-bezier、7 档时长（50–12000ms），但无弹簧/过冲，微交互层时长同质化                | `tokens.css:175-184`                              |

## 2. 用户需求

- REQ-1：主页面与全站动效摆脱「僵硬」，达到现代滚动叙事站点的质感；
- REQ-2：允许引入动效库/框架，JS 预算不设 24 KB 上限；
- REQ-3：保持既有底线不回退——无 JS 可读、reduced-motion 降级、旧 URL 冻结、
  公式可访问名 100%、现有 7 项目 Playwright 矩阵继续全绿。

## 3. 选型与实测依据（2026-09-05 实测，gzip -9）

| 组件               | 版本       | gzip 体积              | 角色                                          |
| ------------------ | ---------- | ---------------------- | --------------------------------------------- |
| gsap core          | 3.15.0     | 28,268 B               | 动画引擎、时间线、quickTo 弹簧                |
| ScrollTrigger      | 3.15.0     | 17,998 B               | 滚动编排：视差、batch reveal                  |
| SplitText          | 3.15.0     | 3,658 B                | Hero 标题行遮罩入场（仅行级）                 |
| lenis              | 1.3.26     | 5,431 B                | 惯性平滑滚动（仅 fine pointer + full motion） |
| Astro ClientRouter | 内置 7.3.1 | +5,642 B（spike 实测） | 跨页视图过渡                                  |
| 现有第一方脚本     | —          | 5,928 B                | 不变                                          |

- 拒绝 ScrollSmoother（需 wrapper div，侵入 fixed 壁纸结构）；拒绝 Motion
  （与 GSAP 能力重叠，避免双引擎）；拒绝 three.js/tsparticles（阅读站不需要
  WebGL，违反克制原则）。

## 4. 架构设计

### 4.0 能力门控（新增纯函数 + 单测）

`src/lib/client/motion/gsap-gate.ts` 导出纯函数
`shouldRunGsap(input): boolean`，输入为
`{ motionPreference, hasFinePointer, gsapFlag, zoom }`：

- `motionPreference === "reduced"` → false；
- `gsapFlag === "false"`（`entropyxi-feature-gsap`，localStorage/URL 双通道）→ false；
- `hasFinePointer` 为 false（触屏）→ Lenis 与磁吸不启用，但视差/batch reveal
  保留 → 返回 `"partial"`（用 `gsap` 而无 `lenis`）；
- `zoom !== 1`（`getComputedStyle(html).zoom`，zoom-200 场景）→ Lenis 禁用；
- 判定通过才 `await import()` GSAP 全家桶（**满足才加载**，R1-12 措辞修正）。

激活成功后由 controller 在 `<html>` 写运行时标记：
`data-gsap-active="true"`、`data-lenis-active="true"`（可测契约，R2-4）、
`data-motion-gsap="true"`（CSS 门控钩子）、`data-hero-ready="true"`（编排完成）。

### 4.1 模块组织（延续 lib/client 单职责 + registerClientFeature 惯例）

```text
src/lib/client/motion/
├── gsap-gate.ts          # 纯函数能力门控（Vitest 覆盖）
├── gsap-register.ts      # 分块动态 import：core 与插件分开（见下）
├── lenis-controller.ts   # Lenis 单例（见 4.3）
├── hero-choreography.ts  # SplitText 行遮罩入场（仅首页、仅 full motion）
├── scroll-narrative.ts   # 壁纸视差、hero 滚动淡出、卡片 batch reveal
└── micro-interaction.ts  # 磁吸弹簧化（quickTo）、TOC FLIP 指示条、404 glitch
```

- **分 chunk 策略（R2-3）**：core 与插件分开动态 import——
  `const { gsap } = await import("gsap")` 与
  `await import("gsap/ScrollTrigger")`、`await import("gsap/SplitText")`
  分处两个模块函数，Vite 生成共享 core chunk（≈28 KB）+ 两个插件 chunk，
  单文件均低于 32 KB 单文件预算；首次 `audit:bundle` 验证**提前到 Phase 3**
  （GSAP 首次入站即验预算，不堆到收口）。
- 入口保持在 `BaseLayout.astro` 的 `<script>`（motion-environment 之后）；
  Hero 编排由 `Hero.astro` 自己引入，Vite 自动去重共享 chunk。
- `await import()` 本身放进 `requestIdleCallback`（兜底 setTimeout 0），
  避免解析/编译挤占首屏 TBT（R2-13）。
- **同一 DOM 只编排一次守卫（R1-3）**：注册即 activate + `pageshow` +
  （Phase 1 起）`astro:page-load` 会对同一文档多次触发 activate；编排类
  feature 初始化前检查 `document.documentElement` 上的代次标记（swap 后
  新文档无标记则重新编排），防止入场动画重放。
- 每页动画用 `gsap.context()` 收集，`astro:before-swap` 时 `ctx.revert()`
  （已核实：before-swap 同步派发先于 DOM swap，无竞态）。
- **Lenis 单例（R1-10）**：Lenis 首次激活创建，仅在 feature-flag 关闭或页面
  卸载时销毁，**不**随 before-swap 销毁重建（避免与 Astro 滚动位置恢复竞态）；
  per-page teardown 只属于 `gsap.context()`。

### 4.2 视图过渡（ClientRouter）

- `BaseLayout.astro` head 加 `<ClientRouter />`；
- `<AmbientBackground>` 根元素加 `transition:persist="ambient-background"`
  （**显式 key**，R1-6：persist 匹配基于属性值而非 transition:name）；
- **属性接力（R1-4）**：Astro swap 会清空旧 `<html>` 全部属性并以新页 SSR
  属性覆盖（`swapRootAttributes`）。新增 `astro:before-swap` 监听，把当前
  `<html>` 的 `data-motion`、`data-motion-preference`、`data-page-visibility`、
  `data-feature-reveal/ambient/magnetic/gsap` 显式复制到
  `event.newDocument.documentElement`；每页 SSR 的 `data-theme="dark"` 天然保持；
- 防 FOUC inline 脚本加 `data-astro-rerun` 作二次兜底（swap 后、page-load 前
  重放 feature-flag）；
- 搜索页 Pagefind 动态 import 有模块缓存，`astro:page-load` 重新绑定即可
  （`registerClientFeature` 已幂等）。

### 4.3 Lenis 集成契约

- 初始化条件（gsap-gate 判定）：fine pointer + full motion + flag 开 +
  zoom===1；触屏不初始化；
- **锚点（R1-1 / R2-2 / R2-7）**：`anchors` 用对象配置
  `{ offset: -(读取 computed scroll-padding-top) }`（Lenis 不读 CSS
  scroll-padding，需手动注入；监听 resize 更新）；锚点接管时必须补两步——
  更新 `location.hash` + 对目标元素 `focus()`（`#main-content` 已有
  tabindex="-1"）；**skip-link 排除在平滑接管外**（保持原生即时跳转 + 焦点
  迁移），`client-behavior.spec.ts:60-70` 的焦点契约原样保留为门禁；
- **双平滑隔离三层关系（R1-8）**：lenis.css 的
  `.lenis.lenis-smooth { scroll-behavior: auto !important }` 为硬保险；
  reduced-motion 下全局 `scroll-behavior: auto !important`（motion.css）恒胜出
  且 Lenis 根本不初始化；`html:not(.lenis) { scroll-behavior: smooth }`
  仅作可选加固。Lenis 用真实 scrollTop 驱动，`site-header.ts` /
  `reading-progress.ts` 的 window scroll 监听零改动；
- **偏好运行时变更（R2-14）**：`data-motion-preference` 变为 reduced 时执行
  `lenis.stop()` + `gsap.globalTimeline.pause()`（最小闭环），完全卸载留待刷新；
- 每次视图过渡后（`astro:page-load`）调用 `ScrollTrigger.refresh()` 与
  `lenis.resize()`。

### 4.4 Hero 入场编排（SplitText）

- **只做行级分割**（`type: "lines"`，SplitText 3.13+ mask），保留
  `hero-contrast.spec.ts` 对 `.welcome-line-1/2` 的文本断言（textContent 不变，
  已核实）；
- **SplitText 根节点 = `h1.hero-welcome-title`（R2-6）**：heading 角色支持
  aria-label；分割后显式 `h1.setAttribute("aria-label", "WELCOME TO ENTROPYXI BLOG !")`
  并对行克隆 aria-hidden；hero-contrast.spec 增加按 role 的可访问名断言；
- **初始隐藏门控（R1-5）**：防 FOUC inline 脚本（加 `data-astro-rerun`）在
  「首页 + full motion + gsap flag 开」时给 `<html>` 打 `data-hero-pending`，
  CSS 据此隐藏两行标题；编排成功后移除并写 `data-hero-ready`；**1.5s 超时
  兜底**：未 ready 则移除 `data-hero-pending` 强制可见（覆盖 chunk 加载失败）；
- **mask 裁切防护（R2-15）**：mask wrapper 对 text-shadow/drop-shadow 光晕与
  `line-height: 0.95` 的字形溢出做 padding 补偿，动画完成后移除 overflow；
  resize 时 `SplitText.revert()` + re-split（或启用 autoSplit）；
- 时序：`data-hero-pending` 置位 → `document.fonts.ready`（800ms 兜底）→
  SplitText → 行遮罩上滑（错拍 90ms）→ 叙述三行 stagger 淡入 → 金色角标
  scaleX 画出 → 滚动指示器浮现（**指示器入场仅 opacity/translateY，
  终态 scale=1**，保证 44px 触控区断言稳定，R1-11）；
- 滚动指示器断言前加 settle 等待（等 `data-hero-ready`）。

### 4.5 滚动叙事（ScrollTrigger）

- **与 motion.css 隐藏契约的互斥（R1-1，P0）**：GSAP 激活路径初始化时对全部
  `[data-reveal]` 先置 `data-reveal-state="visible"`（中和
  `motion.css:99-104` 的 `visibility:hidden` 规则），改用 GSAP `autoAlpha`
  管理显隐；同时 `reveal-controller.ts` 增加 flag 门控——
  `data-gsap-active` 存在则跳过（两控制器互斥）；
  `entropyxi-feature-gsap=false` 时 reveal-controller 照常工作；
- 壁纸视差：**trigger 挂 `#main-content`（文档流元素）**，动画对象为壁纸 img
  （fixed 元素不能做默认 trigger，R1-13），`#0 → 视口底部` 映射
  `scale 1 → 1.06`（transform-only）；
- Hero 内容随首屏滚动 `y -8% / opacity → 0.25`；
- 卡片 reveal 改 `ScrollTrigger.batch`（`once: true`，interval 80ms，
  进入视口 15% 触发），替换 nth-child 写死 stagger（仅 GSAP 激活路径）；
  once 后元素保持终态，视图过渡返回时由「同一 DOM 只编排一次」守卫保证不重放；
- 光晕漂移：**drift keyframes 只挂 primary/cyan 两个 orb（R2-10），排除
  `.glow-orb-pointer`**（其 transform 被指针跟随占用）；既有
  reduced-motion / feature-ambient / page-visibility 暂停规则自动覆盖。

### 4.6 微交互

- 磁吸：`pointer-controller.ts` 升级为 `gsap.quickTo`（x/y 弹簧、回弹
  `elastic.out(1, 0.5)`）；**GSAP 激活时通过 `html[data-motion-gsap]` 门控在
  motion.css 禁用 `[data-magnetic]` 的 CSS transition**（R1-9：避免每帧
  inline transform 与 CSS transition 迟滞打架）；无 GSAP 时保留现有路径；
- 卡片 hover：现有 translateY 基础上加光泽扫过（`::after` 渐变位移）与
  微倾斜（rotateX/rotateY ≤3°，fine pointer only）；
- TOC 高亮：指示条用 FLIP 补间滑动到当前项（`post-toc.ts`）；
- 404：`error-singularity-code` 数字加一次性 glitch 入场（steps() 关键帧）。

## 5. 硬约束（不可回退项）

- 无 JS：Hero、卡片、公式、导航全部可见（GSAP 全部走 `await import`，
  无 JS 时零加载，CSS 终态契约不动）；
- `prefers-reduced-motion: reduce`：不加载 GSAP/Lenis，不运行编排，
  光晕漂移/扫描线由现有全局 reduced-motion 规则冻结；
- 触屏（hover:none）：不加载 Lenis 与磁吸；视差与 batch reveal 保留；
- permalink、旧 URL、单 H1、公式可访问名、frontmatter schema：零改动；
- 7 个 Playwright 项目全部保持绿（新增断言见 §8）。

## 6. 分阶段实施（每阶段独立 commit + 门禁，可单独 revert）

- [ ] **Phase 0 治理**：写 ADR 0003（选型/降级矩阵/预算新值/废除 ≤15 KiB
      首屏 JS 子目标/CSS 体积豁免口径）；`audit-bundle.ts` 总预算 24,576 →
      81,920 B、单文件 8,192 → 32,768 B；**同一 commit** 同步
      `quality-gates.md`（资源预算节 + ≤15 KiB 行 + 浏览器矩阵表）与
      `testing.md:41`（R2-9，消除门禁真空期）；`npm i gsap lenis`（锁 `~3.15` /
      `~1.3`）；dependabot 增加 gsap/lenis 分组；**flag 接线**：
      `entropyxi-feature-gsap` 三处接线（SSR `<html>` 默认属性 + inline 脚本
      读取段 + 模块门控，R1-12）。
- [ ] **Phase 1 视图过渡**：ClientRouter + `transition:persist="ambient-background"`
  - `data-astro-rerun` + before-swap 属性接力；flag=false 时视图过渡仍工作的
    复验；**playwright.config.ts testMatch 扩展**（R2-1）：新增
    `view-transitions.spec.ts` 并入 desktop-chromium；逐项目断言写入
    `client-matrix.spec.ts`。
- [ ] **Phase 2 惯性滚动**：`gsap-gate.ts`（纯函数 + Vitest）+
      `lenis-controller.ts`（单例 + anchors 对象 + offset + hash/focus + zoom
      门控 + `data-lenis-active`）+ Lenis 官方 CSS + 返回顶部接管；
      `client-matrix.spec.ts` 增加逐项目断言（mobile：expect.poll 三件套——
      无 `.lenis`、无 `data-lenis-active`、**有 `data-gsap-active`** + 滚动后
      卡片可见，R2-12）；skip-link/TOC/指示器三类锚点落点断言；
      zoom-200 专项（Lenis 禁用 + 滚动行为正常）。
- [ ] **Phase 3 Hero 编排**：`hero-choreography.ts` + SplitText 行遮罩 +
      初始隐藏门控 + 超时兜底 + `data-hero-ready`；`accessibility.spec.ts` 的
      500ms 固定等待改等 `data-hero-ready`（R2-16）；`hero-contrast.spec` 复核
      （含 320–430px、1280px 不折行 + **height ≤44rem 短视口矩阵（1280×700、
      390×600）** R1-11 + h1 可访问名断言 + mask 裁切断言）；**首次 audit:bundle
      实测**。
- [ ] **Phase 4 滚动叙事**：`scroll-narrative.ts`（视差 + batch reveal）+
      光晕漂移 CSS（排除 pointer orb）；`reveal-controller` flag 互斥；无 JS /
      reduced-motion 复验（reduced-motion 断言升级：hero/卡片
      `transform === "none"`、无 `data-gsap-active`/`data-lenis-active`、两帧
      位置采样不变，R2-5）。
- [ ] **Phase 5 微交互**：磁吸弹簧化（`data-motion-gsap` transition 门控）、
      卡片光泽/微倾斜、TOC FLIP、404 glitch。
- [ ] **Phase 6 收口**：E2E 收尾 + `quality-gates.md` /
      `rendering-and-interactivity.md` / `coding-style.md` / `testing.md` 全量
      同步 + `audit-bundle` 实测值回填本计划与文档 + Lighthouse 四分类复测 ≥90
      **（含 mobile profile，R2-13）** + 完整 `npm run check` +
      `npm run smoke:production`。

## 7. 性能预算与度量

- `audit-bundle.ts`：总 gzip ≤ 80 KB（81,920 B），单文件 ≤ 32 KB（32,768 B；
  分 chunk 后 gsap core ≈28 KB 为最大单文件，有余量）；实施完成后把实测值
  回填本节与 quality-gates.md / testing.md；
- Lighthouse 四分类（home/search/article-normal/article-ddim）仍 ≥90、
  Accessibility ≥95、CLS ≤0.1，另采 mobile profile；GSAP 经 rIC + 动态
  import 不进关键路径；SplitText 在 `fonts.ready` 后执行且只 transform，
  零 CLS；
- 本站脚本不得制造 >50ms 重复 Long Task：初始化拆帧；
- 新增 CSS（Lenis 官方段、glitch、drift）不在 JS 预算内，豁免口径记入
  ADR 0003（R2-13）。

## 8. 测试与验收矩阵（含断言归属文件，R2-1）

| 项目                    | 断言                                                                                                                               | 归属文件                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| desktop-chromium        | Hero 入场终态与现断言一致；batch 后卡片可见；TOC 指示条高亮正确；视图过渡壁纸 identity                                             | hero-contrast / client-behavior / **view-transitions.spec.ts（新）** |
| mobile-390 / mobile-360 | 无横向溢出不变；`expect.poll`：无 `.lenis`、无 `data-lenis-active`、有 `data-gsap-active`、滚动后卡片可见                          | **client-matrix.spec.ts**（唯一在非桌面项目执行的 spec）             |
| mobile-safari (WebKit)  | 视图过渡导航后内容与公式仍可见                                                                                                     | client-matrix.spec.ts                                                |
| reduced-motion          | `transform === "none"`（hero 标题+首屏卡片）、无 `data-gsap-active`/`data-lenis-active`/`.lenis`、两帧位置不变、内容即时可见       | client-matrix.spec.ts                                                |
| javascript-disabled     | 全部现断言不变（GSAP 零加载）                                                                                                      | client-matrix.spec.ts                                                |
| zoom-200                | Lenis 禁用（无 `data-lenis-active`）、滚动后页头状态/阅读进度正常、无横向溢出                                                      | client-matrix.spec.ts                                                |
| 全项目                  | GSAP 资源加载断言：`performance.getEntriesByType("resource")` 中 gsap/ScrollTrigger/lenis chunk 的有无与标记一致（chunk 名含库名） | client-matrix.spec.ts                                                |

`client-behavior.spec.ts` 的动效 feature-flag 契约扩展
`entropyxi-feature-gsap`（localStorage + URL 参数双通道，与现有 flag 同构）。

**E2E 点击约定（Phase 1 实测新增）**：磁吸元素（卡片/导航/logo）会跟随指针
移动，Playwright 的 click 稳定性检查永远无法通过；视口外的卡片又处于
`visibility:hidden`（reveal 未触发）构成二重死锁。涉及点击磁吸目标的用例须
`entropyxi-feature-magnetic=false`（view-transitions.spec 已采用），或点击
视口内非磁吸目标；不得用 `force: true` 绕过。

## 9. 风险登记表

| 风险                                                   | 概率         | 影响 | 缓解                                               | 阻塞上线 |
| ------------------------------------------------------ | ------------ | ---- | -------------------------------------------------- | -------- |
| SplitText 破坏 hero-contrast 文本断言                  | 中           | 中   | 仅行级分割 + Phase 3 全量复核该 spec               | 是       |
| motion.css visibility:hidden 与 batch 互斥致卡片不可见 | 高（已证实） | 高   | §4.5 互斥机制（置 visible 态 + flag 互斥）         | 是       |
| 新断言不在矩阵 testMatch 内执行（假绿）                | 高（已证实） | 高   | §8 归属文件 + testMatch 扩展                       | 是       |
| skip-link 焦点契约被 Lenis 破坏                        | 高（已证实） | 高   | skip-link 排除接管 + hash/focus 补齐               | 是       |
| Lenis 与 200% 缩放滚轮倍率异常                         | 中           | 中   | computed zoom 门控禁用 + zoom-200 专项             | 是       |
| ClientRouter 与 Pagefind/磁吸/抽屉重复初始化           | 中           | 中   | registerClientFeature 幂等 + 编排一次守卫          | 是       |
| GSAP core 单 chunk 超单文件预算                        | 中           | 中   | 分 chunk 策略 + Phase 3 提前实测                   | 是       |
| persist 后壁纸 isHome eager/lazy 属性不恢复            | 高           | 低   | 接受现状（壁纸在视口内必加载），风险表留档（R1-7） | 否       |
| 视图过渡后壁纸 srcset 断点错位                         | 低           | 低   | resize 时 refresh；可接受                          | 否       |
| GSAP 初始化 Long Task / TBT                            | 低           | 中   | rIC 内 import + 拆帧；Lighthouse 复测              | 否       |
| 双平滑（CSS smooth vs Lenis）打架                      | 中           | 低   | lenis.css !important 硬保险（§4.3 三层）           | 是       |
| 动效偏好运行时变更不生效                               | 低           | 中   | §4.3 最小 teardown（stop/pause）                   | 否       |
| 320px mask 裁切光晕/字形                               | 中           | 低   | mask padding 补偿 + 动画后移除 overflow            | 否       |
| dependabot 单独升级 GSAP 破坏兼容                      | 低           | 中   | dependabot 分组 + 锁 minor（`~3.15`）              | 否       |
| 三方库样式污染                                         | 低           | 中   | 仅引入 Lenis 官方小段 CSS；GSAP 无样式             | 否       |

## 10. 回滚策略

- 每阶段独立 commit，`git revert` 单阶段即可；
- 总开关：`entropyxi-feature-gsap=false`（localStorage/URL 参数）一键回到
  纯 CSS 降级路径，不用回滚代码；
- 极端回滚点：Phase 0 前的 commit `b6863e7`。

## 11. 审查与治理

- R1（视觉编排/组件架构/视图过渡生命周期/Lenis 正确性）：结论「修订后可执行」，
  1 P0 / 4 P1 / 8 P2；
- R2（无障碍/性能预算/响应式矩阵/测试覆盖）：结论「修订后可执行」，
  2 P0 / 8 P1 / 6 P2；
- 两份意见 29 条全部闭合于 §12，P0 均已转化为 §4/§6 的硬性设计条款。

## 12. 审查对账闭环表

| #   | 来源 | 级别 | 修订项                                       | 处置                                                          | 状态 |
| --- | ---- | ---- | -------------------------------------------- | ------------------------------------------------------------- | ---- |
| 1   | R1   | P0   | batch 与 motion.css 隐藏契约互斥             | §4.5：置 visible 态 + autoAlpha + reveal-controller flag 互斥 | ✅   |
| 2   | R1   | P1   | anchors 需带 offset、skip-link 即时跳转      | §4.3 + Phase 2 锚点落点断言                                   | ✅   |
| 3   | R1   | P1   | 同一 DOM 只编排一次守卫                      | §4.1 代次标记                                                 | ✅   |
| 4   | R1   | P1   | before-swap 属性接力（swap 清空旧属性）      | §4.2 重写 + 订正表述                                          | ✅   |
| 5   | R1   | P1   | Hero 初始隐藏门控矛盾                        | §4.4 data-hero-pending + 1.5s 兜底                            | ✅   |
| 6   | R1   | P2   | persist 显式 key                             | §4.2 `transition:persist="ambient-background"`                | ✅   |
| 7   | R1   | P2   | persist 后 isHome 属性不恢复                 | §9 风险表留档                                                 | ✅   |
| 8   | R1   | P2   | scroll-behavior 三层优先级                   | §4.3 写死                                                     | ✅   |
| 9   | R1   | P2   | 磁吸 CSS transition 门控                     | §4.6 `html[data-motion-gsap]`                                 | ✅   |
| 10  | R1   | P2   | Lenis 单例化                                 | §4.1/§4.3                                                     | ✅   |
| 11  | R1   | P2   | 短视口矩阵 + 指示器动画稳定性                | Phase 3 + §4.4                                                | ✅   |
| 12  | R1   | P2   | flag 三处接线 + 措辞订正                     | Phase 0 + §4.0（满足才加载）                                  | ✅   |
| 13  | R1   | P2   | D5/D7 数字订正 + 视差 trigger                | §1 + §4.5 trigger=#main-content                               | ✅   |
| 14  | R2   | P0   | 断言归属/testMatch 盲区                      | §8 表格 + Phase 1/2 配置修改                                  | ✅   |
| 15  | R2   | P0   | skip-link 焦点契约                           | §4.3 hash+focus + 排除接管                                    | ✅   |
| 16  | R2   | P1   | 单文件预算 vs 全家桶单 chunk                 | §4.1 分 chunk + Phase 3 提前实测                              | ✅   |
| 17  | R2   | P1   | 「chunk 未加载」断言不可实现                 | §4.0 运行时标记 + resource 断言                               | ✅   |
| 18  | R2   | P1   | reduced-motion 对 GSAP inline 盲区           | Phase 4 断言升级                                              | ✅   |
| 19  | R2   | P1   | SplitText h1 可访问名                        | §4.4 根节点=h1 + 断言                                         | ✅   |
| 20  | R2   | P1   | scroll-padding offset 落点                   | §4.3 + Phase 2 断言                                           | ✅   |
| 21  | R2   | P1   | 「不满足才 import」写反                      | §4.0 修正 + gate 纯函数单测                                   | ✅   |
| 22  | R2   | P1   | Phase 0 同步文档消除门禁真空                 | Phase 0 同 commit + ≤15KiB 废除                               | ✅   |
| 23  | R2   | P2   | 漂移排除 pointer orb                         | §4.5                                                          | ✅   |
| 24  | R2   | P2   | zoom 检测门控                                | §4.0 + Phase 2                                                | ✅   |
| 25  | R2   | P2   | mobile poll 三件套                           | Phase 2 + §8                                                  | ✅   |
| 26  | R2   | P2   | import 进 rIC + Lighthouse mobile + CSS 豁免 | §4.1/§6/§7 + ADR                                              | ✅   |
| 27  | R2   | P2   | 偏好变更最小 teardown                        | §4.3                                                          | ✅   |
| 28  | R2   | P2   | mask 裁切 + autoSplit + 44rem                | §4.4 + Phase 3                                                | ✅   |
| 29  | R2   | P2   | axe 等待改 data-hero-ready                   | Phase 3                                                       | ✅   |

## 13. DoD

- [ ] `npm run check` 九步全绿；
- [ ] 新预算下 `audit:bundle` 通过且实测值回填文档；
- [ ] Lighthouse 四分类 ≥90 / A11y ≥95（含 mobile profile）；
- [ ] 7 项目 Playwright 矩阵全绿 + 新增 view-transitions spec；
- [ ] reduced-motion / 无 JS / 触屏三路降级人工抽查截图
      （`audit-screenshots/motion-overhaul/`）；
- [ ] ADR 0003 合入，quality-gates.md / rendering-and-interactivity.md /
      coding-style.md / testing.md 同步；
- [ ] 生产 smoke（`smoke:production`）81 页全过。
