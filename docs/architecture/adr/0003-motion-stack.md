# ADR 0003：动效体系采用 Lenis + GSAP（ScrollTrigger / SplitText）+ ClientRouter

- 状态：已接受
- 日期：2026-09-05
- 关联计划：`docs/superpowers/plans/2026-09-05-gsap-motion-overhaul-plan.md`（R1/R2 双审查 29 项意见闭合）

## 背景

站点动效被诊断为「僵硬」（计划 §1 的 D1–D7：Hero 零入场编排、reveal 单一公式
且第 7 项起同帧、无滚动链接动画、氛围层静止、悬停语言单调、导航硬切、缓动
扁平）。站点所有者明确解除第一方 JS 体积预算（原 24 KiB）并允许引入动效库。

## 决策

1. 引入 **GSAP 3.15**（core + ScrollTrigger + SplitText，全插件自 3.13 起免费）
   与 **Lenis 1.3** 作为运行时依赖；启用 **Astro ClientRouter** 视图过渡。
2. 分 chunk 策略：core 与插件分开动态 `import()`，避免单文件超预算。
3. 能力门控纯函数 `shouldRunGsap`：reduced-motion / flag 关闭 → 不加载；
   触屏 → 仅视差与 batch reveal（无 Lenis/磁吸）；`zoom !== 1` → Lenis 禁用。
4. 运行时标记契约：`data-gsap-active` / `data-lenis-active` /
   `data-motion-gsap` / `data-hero-ready` 写于 `<html>`，作为 E2E 可测锚点。
5. 预算调整：`audit-bundle.ts` 总量 24 KiB → **80 KiB**，单文件 8 KiB →
   **32 KiB**（分 chunk 后最大单文件为 gsap core ≈28 KiB）。
6. **废除** quality-gates.md 中「新增首屏 JS ≤15 KiB」子目标，由 80 KiB 总预算
   与 Lighthouse TBT/Long Task 门禁替代。
7. 新增 CSS（Lenis 官方段、glitch、drift keyframes）不在 JS 预算内，
   豁免口径以本 ADR 为准。
8. 回滚：`entropyxi-feature-gsap=false`（localStorage/URL 参数）一键降级回
   纯 CSS 路径；各 Phase 独立 commit 可单独 revert。

## 备选方案与理由

- **Motion 13**（mini ≈2 KiB / vanilla ≈10–17 KiB）：spring API 优秀，但缺
  ScrollTrigger 级滚动编排与 SplitText 级文字编排，引入即双引擎。不采用。
- **GSAP ScrollSmoother**（5.5 KiB，已免费）：要求 wrapper div 结构，与
  `AmbientBackground` 的 `position: fixed` 壁纸层侵入冲突。Lenis 零结构改动，
  采用 Lenis。
- **纯 CSS scroll-driven 动画**：Chrome/Edge/Safari 26 已支持、Firefox 2026
  收尾中，作为零成本增强吸收（光晕漂移等），但不满足入场编排与 batch 需求，
  不能独立成方案。
- **three.js / tsparticles**：WebGL 对阅读型站点过重且违反克制原则。不采用。

## 降级矩阵

| 环境                           | GSAP | Lenis | 视差/batch                   | 磁吸弹簧                  | 光晕漂移   |
| ------------------------------ | ---- | ----- | ---------------------------- | ------------------------- | ---------- |
| full motion + fine pointer     | ✅   | ✅    | ✅                           | ✅                        | ✅         |
| 触屏（hover:none）             | ✅   | ❌    | ✅                           | ❌（保留 CSS 静态 hover） | ✅         |
| reduced-motion                 | ❌   | ❌    | ❌（CSS 终态）               | ❌（CSS 路径）            | ❌（冻结） |
| `entropyxi-feature-gsap=false` | ❌   | ❌    | ❌（reveal-controller 接管） | ❌（CSS 路径）            | ✅         |
| `zoom !== 1`                   | ✅   | ❌    | ✅                           | ✅                        | ✅         |
| 无 JS                          | ❌   | ❌    | ❌（CSS 默认可见）           | ❌                        | ❌         |

## 后果

- E2E 断言归属必须与 playwright.config.ts 的 testMatch 对齐（非桌面项目仅
  执行 client-matrix.spec.ts），防止假绿。
- Lenis 接管锚点时必须补 `location.hash` 更新与目标 `focus()`；skip-link
  保持原生路径，`client-behavior.spec.ts` 焦点契约原样保留。
- SplitText 仅行级、根节点为 `h1` 并显式写 aria-label，可访问名契约与公式同级。
- Lenis 为单例（不随视图过渡销毁）；per-page teardown 仅限 `gsap.context()`。
- 动效偏好运行时变更为最小 teardown（`lenis.stop()` + timeline pause）。
