# 移动端适配修复计划（抽屉包含块劫持 / 公式裁切提示 / 触控热区 / 320px 档）

- 状态：高精度审查完成（0 P0 / 4 P1 / 9 P2，13 条全部闭合），可执行
- 日期：2026-09-05
- 证据：`audit-screenshots/mobile-audit/`（截图 + diagnostics.json）；本文所有
  结论均已在审计中用构建产物、computed style、getBoundingClientRect 与
  pre-motion worktree 对照实验验证。

## 1. 背景与根因（已实证）

### F1 移动端抽屉完全损坏（P1，历史遗留）

- 现象：打开抽屉后面板无深色底、无毛玻璃、无全屏遮罩，壁纸穿透；
  `.mobile-drawer-backdrop` 实测仅 390×63px（应为全屏）。
- 根因链：`.site-header` 声明了 `backdrop-filter`（非 none）→ 按 CSS 规范
  成为 **fixed 后代的包含块** → `#mobile-drawer`（`position: fixed; inset: 0`）
  作为 `<header>` 的子元素，inset 相对 header 盒子（390×63，63 = 64 − 1px
  边框）解析 → 抽屉塌缩进 header。
- 证据：computed style 全部"正确"但 rect/绘制错误；隐藏壁纸容器后一切正常；
  git worktree 检出 `ecd9687`（动效改造前）复现同一问题 → 自 2026-08-23
  全站壁纸上线即存在；E2E 只断言抽屉焦点行为、无视觉断言，因此漏检。
- 真机影响：结构性 DOM 几何错误，与仿真/截图无关。

### F2 宽公式横向裁切无提示（P2）

- `mjx-container` 以 `overflow: auto hidden` 兜底（页面级零溢出因此保住），
  但超宽公式在右缘被静默裁断（审计实测 svg right=435 > vw=390），无任何
  可滑动视觉暗示。

### F3 页脚内联链接触控目标 19px（P3）

- 页脚「Astro 7」「GitHub」内联链接高度 19px（< 24px WCAG 2.5.8 最低值；
  因句内 inline 豁免不构成 AA 违规，axe 放过），手机难点中。

### F4 测试盲区（P1 的放大器）

- 抽屉无视觉断言；矩阵最小宽度仅 360px（无 320px 档）；移动端卡片滚动
  reveal 未被断言（本次审计中只能靠手工验证）。

## 2. 修复项

- REQ-1：抽屉移出 `<header>`，根治包含块劫持；不得改变既有 id、焦点行为、
  aria 契约与 noscript 回退。
- REQ-2：可横滑公式给出右缘渐隐提示（滚动到尽头后消失）；纯 JS 渐进增强，
  无 JS 时维持现状；不得产生 CLS。
- REQ-3：页脚链接触控热区 ≥24px 高，且不改变现有视觉布局。
- REQ-4：新增「抽屉打开 → 遮罩覆盖全视口」视觉断言；playwright 矩阵新增
  `mobile-320` 项目；矩阵分支按 `mobile-` 前缀匹配，未来加档位零改动。
- REQ-5：全部改动保持既有门禁语义（九步 check、7→8 项目矩阵、无 JS /
  reduced-motion 契约）。

## 3. 技术方案

### 3.1 REQ-1 抽屉重构（唯一结构性改动）

- 新建 `src/components/chrome/MobileDrawer.astro`：承载现 `#mobile-drawer`
  标记与相关 scoped 样式（`.mobile-drawer`、`.mobile-drawer-backdrop`、
  `.mobile-drawer-content`、`.mobile-drawer-header`、`.mobile-drawer-title`、
  `.mobile-drawer-close`、`.mobile-nav*`），由 `BaseLayout` 在
  `<SiteHeader />` 之后作为 body 直接子元素渲染（脱离 header 的包含块）。
- `SiteHeader.astro`：删除抽屉标记与样式，保留汉堡按钮（`aria-controls`
  仍指向 `#mobile-drawer`）与 noscript 回退（正常流元素，不受劫持影响）。
- `mobile-drawer.ts` 按 id 查找，零改动；z 顺序变为 body 层 z-drawer=100 >
  header z=50，比修复前（被困在 header 上下文内）更符合设计意图。
- 打开抽屉时 `document.body.style.overflow = "hidden"` 契约不变；Lenis
  单例在触屏不激活，桌面宽 >48rem 抽屉不出现，`data-lenis-prevent` 保留。

### 3.2 REQ-2 公式滑动提示（渐进增强）

- 新增客户端 feature `math-scroll-hint`（注册进 `BaseLayout` 脚本，
  `registerClientFeature` 幂等）：对每个 `mjx-container` 在
  `scrollWidth > clientWidth + 1` 时加 `data-scroll-hint="right"`，滚动
  更新为 `both`/`left`/移除；监听 scroll（passive）+ resize，AbortController
  清理。
- CSS（global.css 的 mjx 段）：
  `[data-scroll-hint="right"] { mask-image: linear-gradient(to right, #000 calc(100% - 28px), transparent) }`，
  `left`/`both` 同理；仅作用于已确认可滑的容器，非可滑公式不受影响；
  mask 不产生 CLS；no-JS 无提示（与现状一致）；reduced-motion 无动画语义，
  纯静态遮罩不受影响。

### 3.3 REQ-3 页脚热区

- `.site-footer a` 改 `display: inline-block; padding: 0.3rem 0.2rem;
  margin: -0.3rem -0.2rem;`——命中区高约 19+9.6≈28px ≥24px，负 margin
  抵消布局位移，视觉零变化。

### 3.4 REQ-4 测试补强

- `client-behavior.spec.ts` 抽屉组新增断言：打开后
  `.mobile-drawer-backdrop` 的 `getBoundingClientRect()` 覆盖视口
  （height ≥ clientHeight − 1）且 `.mobile-drawer-content` 背景非透明；
  面板宽度 = 288px（18rem）。
- `playwright.config.ts` 新增 `mobile-320` 项目（320×568, touch+isMobile）；
  `client-matrix.spec.ts` 的项目分支从枚举改为 `startsWith("mobile-")`；
  矩阵补「滚动到 #latest-posts 后首卡可见」断言（mobile 分支）。

## 4. 验收标准

- [ ] 全量 `npm run check` 绿（矩阵从 7 → 8 项目）；
- [ ] 抽屉截图复验（390 视口）与 `backdropRect` = 全屏；
- [ ] view-transitions / client-behavior 既有抽屉焦点断言不放宽且全绿；
- [ ] 公式提示：DDIM 页存在 `data-scroll-hint="right"` 的容器且首个完全
      可见公式无该属性；
- [ ] 页脚链接高度 ≥24px；
- [ ] `mobile-320` 项目全绿。

## 5. 风险

| 风险 | 缓解 |
| --- | --- |
| 抽屉移出后 scoped 样式丢失导致样式回归 | 样式随组件整体迁移；e2e 视觉断言 + 截图复验 |
| mask-image 兼容性（旧 Android） | `@supports (mask-image)` 包裹，不支持则维持现状 |
| 负 margin 页脚在窄屏折行错位 | 320px 项目 + 截图复验 |
| mobile-320 暴露既有 320px 溢出 | 属审计目标本身；发现即修（hero clamp 已支持，预计安全） |

## 6. 回滚

各修复独立 commit，可单独 revert；REQ-2/REQ-3 纯增量可随时还原。

## 7. 审查对账闭环表

| # | 级别 | 修订项 | 处置 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | P1 | mobile-320 补 `testMatch: /client-matrix\.spec\.ts/` | §3.4 | ✅ |
| 2 | P1 | `registerClientFeature("mobile-drawer")` 单点注册 | 保留在 SiteHeader（与汉堡按钮同交互对），不随抽屉迁移 | ✅ |
| 3 | P1 | 滚动断言置于开抽屉之前、用 `toBeVisible()` 自动等待 reveal | §3.4 | ✅ |
| 4 | P1 | mask 前缀策略 | `-webkit-mask-image` + `mask-image`，前者配 stylelint-disable 行注释；`@supports` 包裹 | ✅ |
| 5 | P2 | `startsWith("mobile-")` 使 mobile-safari 获得溢出/遮罩/抽屉断言（扩围声明） | 接受扩围，WebKit 上验证 | ✅ |
| 6 | P2 | navItems/isActive 两组件重复 | 提取 `src/lib/navigation.ts` 共享模块 | ✅ |
| 7 | P2 | 面板宽度断言仅对 390 视口成立（max-width 85vw） | 断言写死 288px 并注明仅 390 视口 | ✅ |
| 8 | P2 | §4 措辞：抽屉焦点断言在 client-behavior + client-matrix（无 view-transitions） | 已改 | ✅ |
| 9 | P2 | data-scroll-hint 覆盖 display 与 inline 容器；订正 svg `!important` 转述 | 属性选择器不限 display；JS 对全部 mjx-container 测量 | ✅ |
| 10 | P2 | 页脚聚焦描边随 padding 外移约 5px | 验收豁免留档 | ✅ |
| 11 | P2 | clean.ts 会清掉新审计截图 | clean.ts 豁免 `mobile-audit` 目录 | ✅ |
| 12 | P2 | 文档同步：testing.md / quality-gates.md 7→8 项目；以 e2e computed-style 断言替代 dist CSS 断言 | §3.4/§6 | ✅ |
| 13 | P2 | hint 测量加 fonts.ready 一次性重测 | 采纳（廉价保险） | ✅ |

审查结论：修订后可执行（0 P0 / 4 P1 / 9 P2），13 条全部闭合。验收中的抽屉焦点断言归属为 client-behavior.spec.ts + client-matrix.spec.ts。

## 8. DoD

- [ ] 审查意见全部闭合；
- [ ] §4 验收全过；
- [ ] 计划勾选 + 审计证据归档。
