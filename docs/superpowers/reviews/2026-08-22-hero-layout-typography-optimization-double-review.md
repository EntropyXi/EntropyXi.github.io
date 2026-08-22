# 博客首屏优化双高精度审查报告 (DeepSeek-V4-Flash Max Thinking)

## 第一轮高精度审查：视觉保真度、排版系统与组件架构

第一轮高精度审查已完成，报告已落盘至 `docs/superpowers/reviews/2026-08-22-hero-layout-typography-r1-review-visual-typography-architecture.md`。以下为结论摘要：

---

## 审查结论：**不建议按 R1 直接执行**（2 × P0 / 6 × P1 / 5 × P2 / 4 × P3）

### 🔴 P0 — 阻断门禁 / 破坏设计契约

**T2｜clamp 数学矛盾，移动端标题必折行。** `clamp(3.8rem, 8.5vw, 6.8rem)` 的下限 60.8px 大于 390px 视口可容纳的 ≈45px（第二行 16 字符，即使 Impact 窄字也需 ≤2.8rem）。方案删除了现状全部三组媒体查询（`≤48rem / ≤25rem / ≤44rem` 高），自己却把"360~430px 防溢出"列为审查重点——无解自洽。

**C1｜测试适配遗漏两个 spec 文件。** 计划 §3 只覆盖 `hero-contrast.spec.ts`，但 `home.spec.ts` L13 点击的「探索文章」主 CTA 与 `client-behavior.spec.ts` L302 的 `.hero-btn-primary[data-magnetic]` 磁吸测试都会被 REQ-3 删除 → `npm run test:e2e` 必红。

### 🟠 P1 — 必须修订

- **T1｜字体栈是 Windows/macOS 专属**：iOS/Android/Linux 无 Impact/DIN/Franklin Gothic → 回退到宽体 SF/Roboto/DejaVu（CI Linux 上 6.8rem 时行宽 ≈1046px > 容器 742px，桌面也折行）。需补 `Roboto Condensed / Oswald` 等回退，且现有测试对折行全部"假绿"（只查页面级 scrollWidth）。
- **T3｜`scroll-breathe` 无 Keyframes 定义**，且动画 transform 会覆盖指示器自身的 `translateX(-50%)` 居中——现状实现正是为此用 wrapper 分层，方案把两层合并了。
- **T4｜forced-colors 规则被删**：`-webkit-text-fill-color: transparent` 渐变字在 Windows 高对比度模式下会不可见。
- **V1｜REQ-5 去黑框与 WCAG AA 自相矛盾**：文本左移后恰落在 scrim 最弱（45% 暗）的左上区域，最坏对比度仅 2.8:1（正文需 4.5:1），且 axe 只回溯祖先背景色，测不出图片上的真实对比度——现有门禁对这个问题"假绿"。建议局部径向 scrim（62% 暗可将对比度抬至 10:1）。
- **V2｜短视口裁切**：1280×720 下内容顶满视口与指示器重叠；横屏直接裁掉叙述区。
- **C2｜新指示器上的 `data-magnetic` 是死属性**（motion.css 磁吸白名单无此选择器）。

### 🟡 P2 — 应当修订（摘选）

- **V3｜REQ-2 表格与 CSS 数值不一致**：`min(46rem,58vw)` vs `min(48rem,58vw)`；`clamp(2rem,7vw,7rem)` vs `clamp(1.5rem,6vw,6rem)`。
- **C3｜作用域令牌块与 `color-scheme: dark` 被删** → 亮色主题下 Hero 内焦点环回退为 `#c2410c`（3.1:1 贴线）；建议保留精简令牌块并把硬编码色值改令牌引用。
- **C5｜补"标题不折行"断言**：`welcome-line-2` 相对容器 `scrollWidth ≤ clientWidth`，把 T1/T2 变成可回归的门禁。

### ✅ 通过项

REQ-3 删除范围与 markup 一致；REQ-5 文案落实；锚点跳转链路完整（全局 `scroll-padding-top`）；heading 正则与暗色作用域测试对新样式仍成立；指示器 44px 触控目标与 aria 属性达标。

**审查限制**：用户参考截图不在仓库内且当前模型无图像输入能力，视觉保真度结论基于 CSS 数值推导（报告附录含完整行宽/对比度模型，R2 可用公式复算），建议实施后以 `npm run capture:*` 截图做像素复核。

---

## 第二轮高精度审查：无障碍可达性、响应式防溢出、测试覆盖与质量保障

第二轮高精度审查已完成并落盘。总结如下：

## 交付物

**`docs/superpowers/reviews/2026-08-22-hero-layout-typography-optimization-double-review.md`** —— 在仓库预留的"第二轮高精度审查"章节中写入了完整报告（第一轮章节补了指回 R1 报告的摘要指针）。

## 审查方法

全部结论建立在仓库实际代码与数值推导上（非纸面推演）：逐行核实了 `Hero.astro`、`motion.css`、`playwright.config.ts`、5 个 E2E spec、`AmbientBackground.astro` scrim、`tokens.css` 与 capture 脚本，并复核了 R1 报告的每一条关键结论。

## 核心发现（P0×2 / P1×9 / P2×9 / P3×4）

**P0（门禁必红）**

- **B-1**：标题两行契约在 **320~~430px 与 1024~~1280px 全矩阵折行**——补上了 R1 漏掉的桌面端证据：1280px（CI 默认视口）容器仅 742px vs 文本 766px，**R1 建议的 `clamp(2.4rem, 10.5vw, 6.8rem)` 修不了桌面端**；1440px 仅余 1.7px。给出全矩阵约束表与推荐式 `clamp(2rem, 7.5vw, 5.9rem)` + 自托管窄长 webfont。
- **C-1**：`home.spec.ts` L13、`client-behavior.spec.ts` L302、`hero-contrast.spec.ts` 三处断言必然失败，计划 §3.1 覆盖面不足。

**P1 亮点（R1 未覆盖的新发现）**

- **A-1**：`data-magnetic` 触发 motion.css 四条 `transform: none !important` 规则，**触屏/减弱动效/特性关闭时指示器失去 `translateX(-50%)` 居中、右偏 22px**——R1 只判"死属性"，实际是交互回归，必须删除。
- **A-4**：200% 缩放下标题被 `overflow: hidden` 裁切，现有 zoom-200 断言**三重假绿**（`style.zoom` 布局不变式 + 折行无横向溢出 + 裁切不改 scrollWidth）。
- **A-3**：渐变标题暗端 `#ea580c` 最坏对比度仅 **0.94:1**（R1 只算了白字灰字）；scrim ≥65%（推荐 70%）才达标；计划 §3.1 的 background-color 断言与 scrim 方案存在口径张力。
- **C-2**：**视口矩阵结构性盲区**——6 个非桌面 project 的 `testMatch` 只跑 `client-matrix.spec.ts`，hero 断言（44px、对比度、溢出）在移动端/减弱动效/缩放场景**根本不执行**。

**配套修复**：320px 与横屏 844×390 两个新 project、元素级两行契约断言集、forced-colors/reduced-motion 仿真断言、像素级对比度门禁（canvas 采样，零新依赖，补齐 axe 对图片背景的盲区）、键盘 Enter 焦点断言、capture 脚本更新、`npm run check` 门禁入口。

报告含 4 张可复核的数值推导表（字号约束/对比度/垂直预算/缩放模型）、与 R1 的逐条对账表，以及合并后的 12 条 R2 修订清单。

**建议后续**：按报告第 6 节清单将方案修订为 R2 版本后再进入实施（按方案 §4.3 流程）；如需，我可以直接产出修订后的 R2 方案文件。
