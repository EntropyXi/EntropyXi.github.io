# 第一轮高精度审查报告：视觉保真度、排版系统与组件架构

**审查对象**：`docs/superpowers/plans/2026-08-22-hero-layout-typography-optimization-plan.md`（R1 版）
**审查范围**：视觉保真度（REQ-1~REQ-5 与目标截图的还原度）、排版系统（字体栈/字阶/行高/字距/响应式）、组件架构（Hero 重构、样式体系一致性、测试与门禁适配）
**审查日期**：2026-08-22
**审查人**：前端视觉架构与现代排版系统资深专家（视觉保真度 / 排版系统 / 组件架构）
**证据基线**：

- `src/components/visual/Hero.astro`（当前实现，563 行）
- `src/styles/tokens.css`、`src/styles/global.css`、`src/styles/motion.css`、`src/styles/home.css`
- `src/components/visual/AmbientBackground.astro`（壁纸/水仙插画层）
- `src/pages/index.astro`、`src/layouts/BaseLayout.astro`（经 grep 确认锚点与头部）
- `tests/e2e/hero-contrast.spec.ts`、`home.spec.ts`、`client-matrix.spec.ts`、`client-behavior.spec.ts`、`accessibility.spec.ts`
- `playwright.config.ts`、`scripts/capture-phase-7.ts`、`scripts/capture-phase-8.ts`、`package.json`

**审查限制声明**：用户参考截图（`屏幕截图 2026-08-22 *.png`）不在仓库内且本模型无图像输入能力，视觉保真度结论基于 CSS 数值推导（行宽/字宽/对比度计算）与代码事实，不包含像素级比对。建议实施后在 R2 阶段以 `npm run capture:*` 截图基线做像素复核。

---

## 0. 审查判定总览

| 级别               | 数量 | 含义                                                                                   |
| ------------------ | ---- | -------------------------------------------------------------------------------------- |
| **P0（阻断门禁）** | 2    | 执行后 `npm run test:e2e` 必然失败；移动端标题必然折行破坏海报结构                     |
| **P1（必须修订）** | 6    | 动画定位冲突、Keyframes 缺失、Forced-Colors 回归、短视口裁切、磁吸失效、对比度方案缺失 |
| **P2（应当修订）** | 5    | 需求表与 CSS 规格不一致、设计令牌体系回归、测试断言补强、焦点环回归、死代码            |
| **P3（建议打磨）** | 4    | 渐变语法、文案阴影值不一致、截屏脚本、字号权重说明                                     |

**结论**：方案方向正确（REQ-1~5 的 markup 级设计均与目标一致），但**排版系统数学上自相矛盾**（clamp 下限大于移动端可容纳字宽 → 双行海报必然折行）且**测试适配面遗漏 2 个 spec 文件**。**不建议按 R1 直接执行，须按第 5 节 R2 修订清单回炉后实施。**

---

## 1. 排版系统审查（核心）

### T1【P0】字体栈缺少 Linux / Android / iOS 的窄长回退 → 标题必然折行

计划 §2.2.2 的字体栈为 `Impact, "Arial Narrow", "DIN Alternate", "Franklin Gothic Heavy", -apple-system, sans-serif`：

- **Impact / DIN Alternate / Franklin Gothic Heavy 仅 Windows/macOS 自带**；iOS/Android/Linux 全部缺失，直接落到 `-apple-system`/`sans-serif`（SF / Roboto / DejaVu Sans，大写字宽 ≈ 0.53–0.62em，远宽于 Impact 的 ≈ 0.40–0.42em）。
- 而 **Playwright CI 为 Linux Chromium**（`playwright.config.ts`，无 Impact/Arial Narrow）→ 桌面端（1280px，6.8rem 上限）第二行 `ENTROPYXI BLOG !`（16 字符，含 0.02em 字距）≈ 16 × 0.60 × 6.8rem ≈ **65rem ≈ 1046px**，容器仅 `min(48rem, 58vw)` ≈ 742px → **CI 与真实 Linux/移动端上两行海报折成三行，且现有测试全部无法捕获**（详见 C5）。
- 计划 §4.2 第 1 条把"跨设备回退兼容性"列为审查重点，但方案本身**未提供任何解**。

**R2 要求**：

1. 栈中增加移动端/开源窄长字体：`"Arial Narrow", "Roboto Condensed", "Oswald", Impact, "DIN Alternate", "Franklin Gothic Heavy", "Liberation Sans Narrow", -apple-system, "PingFang SC", sans-serif`（Roboto Condensed 为 Android 系统字体；Oswald/Liberation Sans Narrow 覆盖常见桌面发行版）。
2. 在 `tokens.css` 定义 `--font-display-condensed` 令牌，组件引用令牌而非硬编码栈（见 C3）。
3. 标题行强制 `white-space: nowrap` + `text-wrap: balance`，把"两行即契约"变为可测状态（配合 C5 的新断言）。
4. 增加移动端字号覆盖（见 T2 的数学约束）。

### T2【P0】`clamp(3.8rem, 8.5vw, 6.8rem)` 下限与移动端可容纳宽度矛盾 → 必然折行

行宽推导（第二行 16 字符，平均字宽 `w`em/字符，含 0.02em 字距 → 每字符 ≈ w+0.02）：

- 390px 视口：内容区 ≈ 390 − 1.5rem×2（hero 内边距）− 1.5rem（容器 margin-left）≈ **318px**；
- 360px 视口：≈ **288px**。
- 即使字体为 Impact（w≈0.42）：3.8rem = 60.8px → 16 × 0.44 × 60.8 ≈ **428px > 318px（390px 视口）** → 必折行。
- 反推约束：390px 视口需字号 ≤ 318 ÷ (16 × 0.44) ≈ **45px ≈ 2.8rem**；360px 视口需 ≤ 41px。
- 即 clamp 的 **min 值（3.8rem）大于空间可容纳的字号**，`8.5vw` 分支在移动端（≤ 8.5% × 390 = 33px）又被 min 钳住——**数学上无解**，必须引入媒体查询或调低下限并提高 vw 权重（如 `clamp(2.4rem, 10.5vw, 6.8rem)` + `@media (width <= 48rem)` 覆盖）。

计划 §4.2 第 3 条自述"移动端 360~430px 换行与防溢出"为审查重点，但 **CSS 章节没有任何媒体查询**——现状实现的 `@media (width <= 48rem)`、`(width <= 25rem)`、`(height <= 44rem)` 三组规则（Hero.astro L472-511）在计划中全部被删除。

### T3【P1】`scroll-breathe` 动画 Keyframes 未定义，且与 `translateX(-50%)` 定位冲突

- 计划 §2.2.4 引入 `animation: scroll-breathe 2.4s ease-in-out infinite;`，但**仓库内（motion.css / Hero.astro / 计划 CSS）均无 `@keyframes scroll-breathe` 定义** → 动画不生效，REQ-1"居中呼吸跳动"落空。
- 更深层缺陷：`.hero-scroll-indicator` 自身用 `transform: translateX(-50%)` 居中（计划 L175），若 Keyframes 按惯例只写 `translateY(...)`，**动画期间会覆盖 translateX 导致指示器水平跳动**。现状实现正是为此把定位 transform 放在 `.hero-scroll-wrapper` 父层、动画放在子层（Hero.astro L411-438）。
- 另需遵守 motion.css 头部契约"**Keyframes 仅 transform 与 opacity**"。

**R2 要求**：Keyframes 须为 `transform: translateX(-50%) translateY(0) ↔ translateX(-50%) translateY(6px)`（或恢复 wrapper 结构），并补 `@media (prefers-reduced-motion: reduce)`（全局 motion.css L244-259 的 `*` 规则可兜底，但组件内显式处理与本仓库现状一致更稳）。

### T4【P1】Forced-Colors 与 Reduced-Motion 规则被整体删除 → 无障碍回归

- 现状 `.welcome-line-2` 在 `@media (forced-colors: active)` 下恢复 `-webkit-text-fill-color: CanvasText; background: none`（Hero.astro L557-562）。计划删除了该规则，而新方案保留 `-webkit-text-fill-color: transparent` + `background-clip: text` → **Windows 高对比度模式下第二行渐变字可能完全不可见**。
- 计划删除组件内 `prefers-reduced-motion` 与 `:focus-visible` 处理（现状 L442-447、L543-554）。全局 `:focus-visible`（global.css L44-48）可覆盖焦点环，但 forced-colors 必须恢复。

### T5【P2】`line-height: 0.95` + `font-weight: 900` 的回退风险未声明

- `font-weight: 900` 对仅有一档字重的 Impact 是**浏览器合成加粗**（faux bold），各平台渲染宽度/粗细不一致——可接受但应在方案中声明，或对回退字体降为 700。
- `line-height: 0.95` 对全大写两行文本（无下伸部字形冲突）数学上安全（y 的下伸部与下行大写顶缘间距 ≈ 0.75em），但依赖 `overflow: visible`；若 R2 引入 `nowrap`+裁切需复查。
- `letter-spacing: 0.02em` 在窄长字体上偏保守，与截图"凝聚"感一致，OK；但**移动端 0.02em 会吃掉约 5px 行宽**，建议 ≤48rem 降至 0.01em。

### T6【P2】渐变文本语法缺标准属性

计划 §2.2.2 仅有 `-webkit-background-clip: text; -webkit-text-fill-color: transparent`，现状实现（Hero.astro L234-243）还含标准 `background-clip: text`。建议补标准属性保证非 WebKit 引擎一致性。

### T7【P2】字号/行高体系与 `--font-display` 令牌脱钩

现状标题用 `var(--font-display)`（tokens.css L238，等宽栈）。计划改为硬编码窄长栈——方向符合 REQ-4，但应落为 `--font-display-condensed` 令牌（含字重、行高、字距的完整契约），否则后续组件无法复用、令牌体系断裂。

### T8【P3】文案阴影值不一致

REQ-5 表（L18）写 `text-shadow: 0 2px 10px rgba(0,0,0,0.75)`，CSS 章节（L157/L165）写 `0.85`——两处不一致，需统一（建议以 0.85 为准，逆光场景更强）。

---

## 2. 视觉保真度审查

### V1【P1】REQ-5 去黑框 与 WCAG 2.2 AA 自相矛盾，且无兜底方案

- 壁纸为全幅 `object-fit: cover` 图 + 全局 scrim 渐变（AmbientBackground.astro L134-145：顶部仅 45% 暗、底部 78% 暗），且 `object-position: center 25%`。
- REQ-2 将文本移至**左上**——恰是 scrim 最弱（45%）区域，水仙亮部花瓣（白/黄）可能直接位于文本后方。
- 最坏情况（白底花瓣 + 45% 暗幕 ≈ #91919b，亮度 ≈ 0.29）：`narrative-line #f1f5f9`（亮度 0.906）对比度 ≈ **2.8:1**，`narrative-lead #cbd5e1` ≈ **1.9:1**——均远低于正文 4.5:1 要求。`text-shadow` 不参与 WCAG 对比度计算。
- 同时注意：axe（accessibility.spec.ts）的对比度算法只回溯祖先背景（最终落到 html 的 `--color-bg-canvas #0a0c16`），**测不出图片上的真实对比度**——即现有门禁对这个问题"假绿"。
- **R2 要求**：二选一并写明取舍——(a) 保留"无卡片"视觉：在 `.hero-narrative-block` 后加**局部柔和径向 scrim**（如 `background: radial-gradient(ellipse at left top, rgb(10 12 22 / 62%), transparent 70%)`，视觉上非"卡片"，但保证文本区最坏 ≥4.5:1）；(b) 文档化豁免（大字号标题 ≥3:1 可满足，正文不行）。推荐 (a)。

### V2【P1】短视口/横屏下内容被 `overflow: hidden` 裁切

新方案 `.hero-fullscreen` 保留 `overflow: hidden` + `justify-content: flex-start`，但删除了现状 `@media (height <= 44rem)` 压缩规则（Hero.astro L514-540：缩小标题、隐藏 narrative-lead）。推演 1280×720：padding-top 7.5rem + margin-top 4vh + 标题 ~207px + 叙述 ~160px ≈ **719px 顶满视口**，底部指示器（bottom 2rem）与叙述区重叠；844×390 横屏直接裁掉叙述与指示器。**R2 必须恢复高度感知媒体查询**（≤44rem 时收紧 padding、压缩字号、可选隐藏 lead 行）。

### V3【P2】需求表与 CSS 规格数值不一致（REQ-2）

| 项             | 需求表（L15）            | CSS 章节（L100-101）       |
| -------------- | ------------------------ | -------------------------- |
| 容器 max-width | `min(46rem, 58vw)`       | `min(48rem, 58vw)`         |
| margin-left    | `clamp(2rem, 7vw, 7rem)` | `clamp(1.5rem, 6vw, 6rem)` |

同一方案的"目标设计"与"落地规格"必须唯一。另外 REQ-2 表格将 `padding-top` 挂在"容器"上，CSS 挂在 `.hero-fullscreen` 上（语义等价，但需统一表述）。

### V4【P2】REQ-1 指示器定位可行，但焦点/悬停反馈需显式化

新 `.hero-scroll-indicator` 无 `:hover`/`:focus-visible` 样式（现状 L441-447 有）。全局焦点环（global.css L44）可兜底，但本项目组件惯例是显式声明；建议补 `outline` + 悬停时胶囊条亮度提升（纯 opacity/transform 实现）。

### V5【P3】渐变配色"夕阳琥珀金"与目标一致

`#f59e0b → #ea580c`（135deg）对比度在暗底上 ≈ 8.5:1 / 5.2:1，满足大字号 ≥3:1。相比现状四段渐变（奶油→黄→琥珀→珊瑚）更"窄长海报"化，与 REQ-4 描述一致，保留通过。

### V6【通过】REQ-2 左移空间诉求与壁纸层结构兼容

壁纸为 fixed 全幅层（z-index: -1，AmbientBackground L96-107），Hero 内容 z-index 2 之上；左对齐后右侧 45%+ 区域仅剩插画与装饰层，无重叠遮挡——结构上支持"把右侧留给水仙插画"。`object-position: center 25%` 下人物位于画面右侧的推断与用户描述一致。

---

## 3. 组件架构与测试门禁审查

### C1【P0】测试适配遗漏两个 spec 文件 → 门禁必红

计划 §3 仅覆盖 `hero-contrast.spec.ts`，但删除 6 chips + 4 CTA + 文字胶囊后，以下断言**必然失败**：

- `tests/e2e/home.spec.ts` L13：`getByRole("link", { name: "探索文章" })`（主 CTA 被删）→ 应改为点击新指示器（`aria-label="向下滚动至最新文章"`）。
- `tests/e2e/client-behavior.spec.ts` L302-305：`.hero-btn-primary[data-magnetic]` 磁吸特性测试 → 元素不存在；须改用剩余磁吸元素（如 `.site-logo-link[data-magnetic]` 或 `.post-card[data-magnetic]`）。
- `tests/e2e/hero-contrast.spec.ts` L29：`#hero-scroll-indicator` 定位符——新 markup **没有 id**（计划 §2.1 L54-74）→ 建议保留 `id="hero-scroll-indicator"` 减少改动面，或同步更新定位符。

### C2【P1】新指示器上的 `data-magnetic` 是死属性

- pointer-controller.ts L60 会对所有 `[data-magnetic]` 元素写入状态属性，但 motion.css 的磁吸变换白名单仅有 `.site-logo-link / .nav-link / .hero-btn-primary / .post-card`（L129-173）——新指示器**不会获得任何磁吸变换**，属性无效且误导。
- **R2 要求**：要么删除该属性（推荐——下滚指示器做磁吸无 UX 价值），要么在 motion.css 增加 `.hero-scroll-indicator` 白名单规则。

### C3【P2】样式体系回归：作用域令牌块与 `color-scheme: dark` 被删除

- 现状 `.hero-fullscreen` 在作用域内重定义整套暗色令牌并声明 `color-scheme: dark`（Hero.astro L132-144）。计划 CSS 仅剩布局属性：
  - 亮色主题下，Hero 内 `--color-focus` 回退为亮色主题的 `#c2410c`（tokens.css L160），在暗壁纸上焦点环对比 ≈ 3.1:1，**贴近 3:1 下限**；
  - 原生控件/滚动条配色随亮色 scheme 变化。
- 新 markup 自身只用硬编码色值（`#ffffff`/`#f59e0b`/`#f1f5f9`），功能上不依赖令牌，但**架构上建议**：保留精简版作用域令牌块（暗色 `--color-focus` 等）+ `color-scheme: dark`，并将硬编码色值替换为 `var(--color-accent-primary)` 等令牌（与 `frontend-ui-engineering` 语义色令牌规范一致）。

### C4【P2】删除的组件遗留物未清理（死代码面）

- `researchTopics` 数组与 `buildTagPath` 导入随 chips 删除而移除——计划 §2.1 的 frontmatter 已空，OK；
- `motion.css` L264-265 遗留 `.hero-title-segment/.segment-brand` 选择器（历史遗留，非本计划引入，可顺手清理）；
- `scripts/capture-phase-7.ts` / `capture-phase-8.ts` L44 `hoverSelector: ".hero-btn-primary[data-magnetic]"` 与 `audit-screenshots/phase-8/*.json` 引用——CTA 删除后截图脚本报错，需改为新指示器或移除（不入 9 步门禁，但属仓库健康度）。

### C5【P2】测试无法捕获"折行破坏海报"——需补强断言

现有 mobile-390/360、zoom-200 项目只断言 `scrollWidth <= clientWidth`（client-matrix.spec.ts L58-66、L85-93）——**折行不产生横向溢出，测试全绿**。R2 应新增：

- `welcome-line-2` 的 `scrollWidth <= clientWidth`（相对其容器）→ 折行即红；
- 或断言 h1 内 span 数 = 2 且均未 wrap（`getClientRects()` 高度单行）。
- 该断言同时把 T1/T2 的修复变成可回归验证的门禁，直接服务计划 §4.2 第 3 条。

### C6【P2】9 步门禁链本身 OK，但 hero-contrast 新增断言需明确写法

§3.1"无黑框背景文本（`background-color: rgba(0,0,0,0)`）"断言成立（`background: none` 计算值为 transparent）。建议同时断言 `backdrop-filter: none`，防止未来引入玻璃底又"看起来没框"。

### C7【通过】锚点跳转链路完整

`html { scroll-behavior: smooth; scroll-padding-top: calc(var(--header-height) + 1.5rem) }`（global.css L12-13）→ 新指示器 `href="#latest-posts"`（index.astro L30 有 `tabindex="-1"` 可聚焦目标）→ 跳转落点正确；`prefers-reduced-motion` 下全局 `scroll-behavior: auto`（motion.css L248-250）兜底。heading 正则 `/WELCOME TO.*ENTROPYXI BLOG/i` 对新文案（span 换行 → 可访问名含空格）持续匹配。hero-contrast 暗色作用域断言（L41-58，`welcome-line-1` = `rgb(255,255,255)`）对新样式仍成立。

### C8【通过】可访问性基础属性齐备

新指示器为原生 `<a>` + `aria-label`，装饰元素 `aria-hidden`，44px 最小触控目标（`min-width/min-height: 44px` 与 bounding box 断言一致）；section 保留 `aria-label` 与 `data-theme-scope="dark"`；axe 门禁（accessibility.spec.ts）不受删除影响。

---

## 4. 校对为"通过"的项（Positive Findings）

1. REQ-3 删除范围与 markup 一致：badge / topics（含 `aria-label="核心研究主题"`）/ actions 在 §2.1 中完整移除，无残留。
2. REQ-4 的栈、字号、行高、字距、双行配色（白 + 琥珀金渐变）与 CSS 章节一致；行 1 白字高光（text-shadow 双层）实现正确。
3. REQ-5 文案变更（删除"、收敛性证明"）在 markup 中落实，`audit:content`（scripts/audit-content.ts）不涉及 Hero 文案，无审计冲突。
4. `min-height: 100vh + 100svh` 双声明、`position: relative`、z-index 分层（内容 2 / 指示器 10 / 壁纸 -1）结构正确。
5. 指示器 anchor 的键盘可达性、`aria-label`、装饰 `aria-hidden` 达标；全局焦点环兜底。
6. 移除 6 个 chip 链接后无其他页面引用 `buildTagPath` 的 Hero 场景（grep 确认仅 Hero 使用）——删除安全。

---

## 5. R2 修订清单（按优先级）

| #   | 级别 | 修订项                                                                                                                                                    | 对应发现   |
| --- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| 1   | P0   | 补移动端字号媒体查询，重算 clamp（建议 `clamp(2.4rem, 10.5vw, 6.8rem)` + ≤48rem 覆盖），并给出 ≤360/390/430px 的字号约束表                                | T2         |
| 2   | P0   | 更新 `home.spec.ts`（点击新指示器）、`client-behavior.spec.ts`（磁吸测试改目标元素）、`hero-contrast.spec.ts`（保留/更新 `#hero-scroll-indicator`）       | C1         |
| 3   | P1   | 字体栈补 `Roboto Condensed / Oswald / Liberation Sans Narrow` 并落为 `--font-display-condensed` 令牌；标题行 `white-space: nowrap` + `text-wrap: balance` | T1, T7     |
| 4   | P1   | 定义 `@keyframes scroll-breathe`（含 `translateX(-50%)` 基线，仅 transform/opacity），或恢复 wrapper 分层结构                                             | T3         |
| 5   | P1   | 恢复 `forced-colors` 规则（渐变字回退 CanvasText）与 `prefers-reduced-motion` 处理                                                                        | T4         |
| 6   | P1   | 为叙述区补局部径向 scrim 或书面化对比度豁免，并加"最坏区域对比度 ≥4.5:1"的验证条目                                                                        | V1         |
| 7   | P1   | 恢复短视口（`height <= 44rem`）压缩媒体查询                                                                                                               | V2         |
| 8   | P2   | 统一 REQ-2 表格与 CSS 的 max-width / margin-left 数值                                                                                                     | V3         |
| 9   | P2   | 删除新指示器的 `data-magnetic`（或补 motion.css 白名单）                                                                                                  | C2         |
| 10  | P2   | 保留精简暗色作用域令牌块 + `color-scheme: dark`；组件内色值改令牌引用                                                                                     | C3         |
| 11  | P2   | 新增"标题不折行"回归断言（`welcome-line-2` 相对容器 scrollWidth ≤ clientWidth）                                                                           | C5         |
| 12  | P2   | 指示器补 `:hover`/`:focus-visible` 显式样式                                                                                                               | V4         |
| 13  | P3   | 补 `background-clip: text` 标准属性；统一 text-shadow 透明度（0.85）；更新 capture-phase-7/8 脚本 hoverSelector；声明 faux-bold 策略                      | T6, T8, C4 |

---

## 6. 附录：关键数值推导

**行宽模型**：`width = n × (w + tracking) × font-size`，n=16（`ENTROPYXI BLOG !`），tracking=0.02em。

- Impact（w≈0.42）：6.8rem → ≈ 48rem（≈768px）；3.8rem → ≈ 27rem（≈428px）。
- SF / Roboto（w≈0.53）：6.8rem → ≈ 60rem（≈957px）；3.8rem → ≈ 33rem（≈534px）。
- DejaVu（w≈0.58）：6.8rem → ≈ 65rem（≈1046px）。
- 可用宽度：桌面 ≤ `min(48rem, 58vw)`；390px 视口 ≈ 318px；360px 视口 ≈ 288px（含 1.5rem×2 内边距 + 1.5rem 左边距）。

**对比度模型**（最坏区域 = 白花瓣 + 45% 暗幕 ≈ #91919b，L≈0.29）：

- `#f1f5f9`（L≈0.906）：(0.906+0.05)/(0.29+0.05) ≈ **2.8:1**（需 ≥4.5:1）；
- `#cbd5e1`（L≈0.593）：≈ **1.9:1**；
- 标题白字（≥3:1 大字号目标）：(1.05)/(0.34) ≈ **3.1:1**，贴线。
- 若按 R2 建议 scrim 抬至 62% 暗：#91919b → ≈ #3a3a44（L≈0.045），`#f1f5f9` ≈ (0.956)/(0.095) ≈ **10:1** ✓，`#cbd5e1` ≈ 6.8:1 ✓。
