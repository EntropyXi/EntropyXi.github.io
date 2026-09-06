# Hero 标题自适应缩放计划（方案 A：fit-to-width，根治 Android 字体回退裁切）

- 状态：双高精度审查完成（R1 测量/时序：2 P0 / 2 P1 / 5 P2；R2 可测性/门禁：
  1 P0 / 2 P1 / 7 P2；18 条全部闭合于 §7），可执行
- 日期：2026-09-06
- 证据：`audit-screenshots/mobile-audit/hero-androidsim-{390,360,320}.png`
  （注入 sans-serif 模拟 Android 字体回退）+ 探针实测
  （`audit-screenshots/mobile-audit/` 诊断流程）

## 1. 背景与根因（已实证）

Hero 标题字体栈 `Impact, "Arial Narrow", Haettenschweiler, "DIN Alternate",
"Roboto Condensed", "Franklin Gothic Bold", -apple-system, sans-serif` 在
**Android 上全部缺失**（除 sans-serif=Roboto），回退后 "ENTROPYXI BLOG !"
宽度从 Impact 的 254px 膨胀到 **350px**（2.2rem 实测，探针数据）：

| 视口 | 容器宽 | Roboto 宽度 | 结果                        |
| ---- | ------ | ----------- | --------------------------- |
| 390  | 350    | 350         | 临界（亚像素波动即裁切）    |
| 360  | 328    | 350         | 溢出 22px                   |
| 320  | 288    | 350         | **溢出 62px，"LOG !" 被裁** |

放大因素：`white-space: nowrap` + `.hero-fullscreen { overflow: hidden }`
→ 超宽即静默裁切；移动端 `clamp(2.2rem, 9vw, 3.2rem)` 的下限 2.2rem 在
≤391px 触底，视口缩小时字号不再缩小。

**测试盲区**：E2E 跑在 Windows 本地 Chromium（Impact 存在 → 恒 254px），
恒满足 `scrollWidth <= clientWidth + 2` 断言；iOS 有 Impact 同样不受影响；
**只有 Android 真机暴露**。

## 2. 修复项（方案 A：纯 JS fit-to-width，不引入 webfont）

- REQ-1：新增 `src/lib/client/hero-fit-title.ts`——测量标题行在**当前字体栈**
  下的实际宽度，超出可用宽度时按比例缩小标题 `font-size`（带下限），使
  `white-space: nowrap` 的行在任意平台字体下都恰好容纳。**可用宽度的测量
  对象是 `.hero-content-container`（`width:100%` + `max-width`，宽度与标题
  字号无关，R1-F1）**——`.hero-brand-block` 是 fit-content 的 flex item，
  宽度恒等于文本宽，作分母会使比值恒为 1、fit 永不生效；
- REQ-2：缩放系数计算抽为纯函数 `computeHeroFitFontSize`（Vitest 覆盖）；
- REQ-3：E2E 关闭盲区——新增「注入 `sans-serif !important` 强制走 Android
  回退 → 断言两行不裁切」用例；
- REQ-4：与既有机制共存——SplitText 入场编排（在缩放后的最终字号上分割）、
  resize 重算、视图过渡、reduced-motion（本修复无动效语义，reduced-motion
  与 gsap-flag=false 下照常生效）、no-JS（无法缩放，维持现状不退化）。

## 3. 技术方案

### 3.1 缩放算法（纯函数，R2-可测）

```ts
export interface HeroFitInput {
  baseFontSize: number; // 清除 inline 后的 CSS 计算字号（随媒体查询变）
  availableWidth: number; // 标题父容器 clientWidth
  measuredWidth: number; // 两行中较宽者的 scrollWidth
  minScale: number; // 可读性下限（0.55）
}
export function computeHeroFitFontSize(input: HeroFitInput): string;
```

- `measuredWidth <= availableWidth` → 返回 `""`（清除 inline，用 CSS 原值）；
- 否则返回 `max(round(base * available/measured * 100)/100, base*minScale)`
  的 px 值；触底仍溢出则接受（320 极端 + 未来字体场景的最终兜底）。

### 3.2 运行时行为（initializeHeroFitTitle）

- 挂载点 `.hero-fullscreen`；非首页直接返回 noop cleanup（null 检查，同
  choreography 模式）。**无 dataset 单次守卫（双审查 P0）**：`astro:page-load`
  首载即触发、bfcache `pageshow` 会重激活，守卫会让第二次 activate 在
  cleanup 已 abort 旧监听后空转、resize 重算永久失效——**initialize 必须可
  重入**：每次 activate 重新查询 DOM、重新绑定（旧监听由 AbortController
  统一 abort，与 `math-scroll-hint.ts` 同构，而非 choreography 的一次性守卫）。
- 测量：遍历 `title.querySelectorAll(":scope > span")` 取最大 `scrollWidth`
  （不硬编码两行；block+nowrap = 文本真实宽度；letter-spacing 计入——em 基
  随字号等比、比例式成立；text-shadow/drop-shadow 为 ink overflow 不计入）；
  可用宽取 `.hero-content-container.clientWidth`（见 REQ-1）；**先清 inline
  font-size 再读 CSS 基准与测量**（避免复合缩放）。
- 重算触发：注册即同步 fit 一次 + `document.fonts.ready` 一次性重测 +
  `resize`（rAF 节流）。重测纪律：同任务内「清 → 测 → 写」，写前与原 inline
  值比较、相同则还原等值（无中间绘制）；本站标题无 webfont、重测预期等值，
  若未来引入 webfont 须改为 `data-hero-ready` 后重测（避免 SplitText mask
  中途失配）。`hero-pending` 的隐藏是 `opacity: 0`（motion.css:38-45，非
  visibility），保留布局，测量不受影响。
- 注册位置：BaseLayout 脚本内置于 motion-runtime **之前**（先布局修正后动效，
  注册顺序不影响时序正确性——rIC 必然晚于同步注册，但语义清晰）。
- 标记：`title.dataset.heroFit = "scaled" | "native"`，供 E2E 与诊断。

### 3.3 E2E 关闭盲区（hero-contrast.spec 新增用例）

- **注入机制（双审查 P1）**：`addInitScript` 内用 `MutationObserver` 监听
  `document`（childList+subtree），`documentElement` 出现的微任务时刻立即把
  `<style>.hero-welcome-title{font-family:sans-serif !important}</style>` 挂到
  `documentElement` 上——早于 deferred module 脚本的首次 fit 测量。**禁止
  DOMContentLoaded 回退**（模块脚本先于 DCL 执行，注入就太晚了；
  client-matrix.spec.ts:105 的 documentElement-null 教训）。
- **承载断言（R1-F4）**：行 span 的 `scrollWidth <= clientWidth + 2` 在
  fit-content 父容器下是同义反复，仅作辅助；承载断言为
  `行 rect.right <= .hero-content-container rect.right + 2`。
- **390 等值边界（R2-P1）**：350/350 恰为算法等值点，`scaled|native` 双态
  均合法——390 只断言「不裁切 + `data-hero-fit` 存在」；360/320 断言
  `data-hero-fit === "scaled"`（回退字体必然超宽，确定态）。
- Windows/Linux 回归路径（R2-7）：既有首用例末尾补
  `toHaveAttribute("data-hero-fit", /^(native|scaled)$/)`（本地 Windows 命中
  Impact 为 native、CI Ubuntu 无 Impact 可能为 scaled，环境相关不做等值断言）；
  原不折行断言在缩放后恒成立，照常保留。

## 4. 验收标准

- [ ] 全量 `npm run check` 绿；
- [ ] 新 E2E：sans-serif 回退模拟下 360/320 断言 `scaled`、390 断言不裁切
      且标记存在（等值边界双态）；
- [ ] Vitest（`tests/unit/hero-fit-title.test.ts`，baseInput fixture 风格）：
      不溢出返回空串 / 恰好等于可用宽返回空串 / 触底钳制 / 两位小数取整 /
      minScale=1 返回恰为 base 的 px 串；
- [ ] 不影响 SplitText 编排与既有 hero-contrast 断言（Windows 路径回归）；
- [ ] bundle 增量 ≤1KB gzip（以实现当次 `audit:bundle` 输出为准，实测快照
      68019/81920——数字随无关提交漂移，不引用历史绝对值）；

## 5. 风险

| 风险                                      | 缓解                                                                                 |
| ----------------------------------------- | ------------------------------------------------------------------------------------ |
| 缩放与 SplitText 编排时序竞争             | fit 注册即同步执行，早于 rIC 编排；fonts.ready 重测同值不写 DOM；无 webfont 前提留档 |
| resize 循环（改 font-size 触发 resize？） | font-size 变化不触发 window resize；rAF 节流                                         |
| 无 JS 的 Android 仍裁切                   | 现状不变、无退化；documented limitation                                              |
| 触底（0.55）后仍溢出                      | 320+Roboto 实需 ≈0.80–0.82，远离下限；极端字体场景接受                               |
| 390 等值边界 scaled/native 漂移           | 确定                                                                                 | 低  | 验收按双态断言（§3.3）                   | 否  |
| CI Linux 无 Impact（与本地 Windows 不同） | 确定                                                                                 | 低  | 归属断言用双态正则；缩放路径本就平台无关 | 否  |
| desktop Impact 路径回归                   | 纯函数单测 + 既有断言 + data-hero-fit="native" 断言                                  |

## 6. 回滚

单 commit；功能含 `hero-fit-title` 注册行，revert 即完全移除；无数据/资产变更。

## 7. 审查对账闭环表

| #   | 来源 | 级别 | 修订项                                                                     | 处置                                                     | 状态 |
| --- | ---- | ---- | -------------------------------------------------------------------------- | -------------------------------------------------------- | ---- |
| 1   | R1   | P0   | availableWidth 测 `.hero-brand-block`（fit-content≡文本宽）使 fit 恒 no-op | 改测 `.hero-content-container.clientWidth`（REQ-1 重写） | ✅   |
| 2   | R1   | P0   | `heroFitBound` 守卫致第二次 activate 后监听永久丢失                        | 删除守卫，可重入契约（math-scroll-hint 同构）            | ✅   |
| 3   | R2   | P0   | 同上（独立命中，指出 bfcache 路径）                                        | 同上                                                     | ✅   |
| 4   | R1   | P1   | E2E 注入时机：init 时 documentElement 为 null、DCL 回退太晚                | MutationObserver 解析期挂载，禁 DCL 回退                 | ✅   |
| 5   | R2   | P1   | 同上（独立命中，禁止 DCL 回退）                                            | 同上                                                     | ✅   |
| 6   | R1   | P1   | 行 scrollWidth 断言是同义反复（fit-content 父）                            | 承载断言改 `.hero-content-container` rect 对照           | ✅   |
| 7   | R2   | P1   | 390 等值边界断言非确定                                                     | 360/320 断 scaled、390 断双态                            | ✅   |
| 8   | R1   | P2   | hero-pending 是 opacity 非 visibility                                      | 表述更正（结论不变）                                     | ✅   |
| 9   | R1   | P2   | 容器宽数据偏差（328/288 → 应为 320/280）                                   | 实施后按新测量口径复测回填                               | ✅   |
| 10  | R1   | P2   | fonts.ready 重测中途改字号的竞争窗口                                       | 同值不写 DOM + 无 webfont 前提留档                       | ✅   |
| 11  | R1   | P2   | 注册位置/非首页早退/letter-spacing 计入未写明                              | §3.2 补齐                                                | ✅   |
| 12  | R1   | P2   | SplitText 缩放稳定性论据                                                   | §3.2/§5 补（nowrap 行数恒 2、textContent 不变）          | ✅   |
| 13  | R2   | P2   | hero-pending 机制名错误                                                    | 同 #8                                                    | ✅   |
| 14  | R2   | P2   | fonts.ready 重测纪律 + 无 webfont 前提                                     | 同 #10                                                   | ✅   |
| 15  | R2   | P2   | 单测文件命名与风格 + 补 minScale=1/格式用例                                | `tests/unit/hero-fit-title.test.ts`                      | ✅   |
| 16  | R2   | P2   | native 断言无落点                                                          | 既有首用例补双态正则断言                                 | ✅   |
| 17  | R2   | P2   | 两行硬编码                                                                 | 遍历 `:scope > span` 取最大值                            | ✅   |
| 18  | R2   | P2   | bundle 数字漂移 + quality-gates 无需改                                     | §4 以当次实测为准                                        | ✅   |
| 19  | R2   | P2   | 0.55 下限补依据 + testing.md 单测清单回填                                  | §3.1 + 实施时回填                                        | ✅   |

## 8. DoD

- [ ] 双审查意见全部闭合；
- [ ] §4 验收全过；
- [ ] 审查执行情况（实现 vs 计划逐项核对）并记录；
- [ ] `docs/contributing/testing.md` 单测覆盖清单补 hero-fit 条目；
- [ ] 提交 + push + Actions 部署成功 + 线上抽检。
