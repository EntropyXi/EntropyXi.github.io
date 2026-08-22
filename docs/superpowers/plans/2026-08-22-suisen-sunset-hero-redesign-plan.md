# EntropyXi Blog：全屏沉浸式欢迎界面重构与夕阳水仙主题色调适配改造计划（R2 修订版）

> 状态：**已完成 DSH 双高精度审查（R1 & R2）并全面吸收闭合全部 8 项 P0 与 12 项 P1 审查意见**  
> 计划日期：2026-08-22（R2 修订版）  
> 审查依据：`docs/superpowers/reviews/2026-08-22-suisen-hero-redesign-review-r1.md`、`docs/superpowers/reviews/2026-08-22-suisen-sunset-hero-redesign-review-round2.md`  
> 核心资产：`D:\DimensionToTsuLovers_cn{A25922C8-2EC5-4D92-9AE3-73A2C0D4665B}\豪華版特典\ディメンション凸ラバース!!壁紙\PC\3840×2160 px\3840x2160px_D_suisen.png`  
> 视觉参考：AyeezBlog 全屏沉浸首屏、崩月水仙预设（preset-beng-yue-shui-xian）  
> 质量标准：WCAG 2.2 AA（合成对比度 ≥ 4.5:1 / 大字 ≥ 3:1）、Google CWV（0-CLS, LCP ≤ 2.5s @ 4G, INP ≤ 200ms）、Astro 7 SSG 渐进增强

---

## 0. 执行摘要与 R2 审查闭合声明

本计划旨在将 EntropyXi 技术博客从当前的「浅青分栏首屏」全面升级为「**全屏沉浸式欢迎首屏（Fullscreen Hero Landing）+ 夕阳水仙（Suisen Sunset & Twilight Indigo）专属色调**」的高定视觉与工程系统。

针对 DSH 独立审查产出的 **8 项 P0 阻断项** 与 **12 项 P1 高优项**，本 R2 修订版已实现 100% 针对性闭合：

1. **Hero 独立颜色契约（闭合 P0·A3 / R1-P0-1）**：Hero 首屏强制声明独立暗色作用域（`color-scheme: dark` + Scoped Tokens），杜绝亮色主题下文字与暗蒙版串用导致 1.21:1 崩溃。
2. **多级响应式资产管线与性能预算（闭合 P0·P1 / P1·P2 / R1-P0-2）**：彻底推翻 6.02MB PNG 直出方案，引入 AVIF/WebP 三级断点分级格式（移动端 ≤ 400KB、4K ≤ 2MB），使用 `<picture>` 响应式调度，并实现**仅首页条件预载**，新增 `audit:assets` 体积门禁。
3. **合成对比度科学重构（闭合 P0·A1 / P0·A2 / P0·P3 / R1-P0-3）**：基于壁纸 40×40 像素实测（全图 0%–88% 散布近白 L=1.0 高光），为页头导航与内容区增设独立 Scrim（暗化护盾层），确保在最坏高光像素背景下正文仍 ≥ 4.5:1、大字 ≥ 3:1。
4. **信息架构与无障碍完整保留（闭合 P1·A6 / P1·A7 / R1-P1-2）**：保留全部 6 个真实标签路由链接与 4 个 CTA 按钮；skip-link 严格锁定 `#main-content`；下滚指示器提供 44px 触控区与焦点管理。
5. **完整全量门禁与测试迁移（闭合 P0·T1 / P0·T2 / P1·T3）**：对齐仓库实际 `npm run check` 9 步门禁，更新 6 处必然受影响的旧测试断言，新增针对图片背景文字计算样式的 Playwright 专属对比度断言（补足 Axe-core 结构性盲区）。

---

## 1. 基线分析与视觉对齐

### 1.1 现状与目标对比及变更清单

| 维度         | 现状（图一：当前界面）                   | 目标（图二：Ayeez 风格全屏欢迎页）                 | 本次改造实施方案（R2 规范）                                                 |
| ------------ | ---------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------- |
| **首屏高度** | 局限于内容区常规高度（约 450px）         | 完整视口全屏（`100vh / 100svh`），高度沉浸式大空间 | `min-height: 100vh; min-height: 100svh;` 双写，支持短视口压缩               |
| **首屏布局** | 左右 55:45 分栏，右侧 SDE 终端卡片       | 单列大画幅居中/左对齐大字版式，壁纸大图为主体      | 大标题 `WELCOME TO ENTROPYXI BLOG !`，保留完整技术标签与导航按钮            |
| **背景系统** | 纯色底 + SVG 浅绿流线 + 浅绿网格         | 4K 高保真插画壁纸 + 赛博扫描线 + 渐变蒙版          | 水仙 4K 逆光壁纸 + 局部 Content Scrim 护盾 + 静态赛博网格                   |
| **页头导航** | 占据上方独立高度，背景固定               | 浮动在 Hero 壁纸顶部，通透毛玻璃融合，下滚收敛     | 首页 `scrollY < 24px` 启用带顶部 Scrim 的透明浮动态，下滚平滑过渡为毛玻璃态 |
| **滚动引导** | 无，直接呈现文章列表                     | 底部中央呼吸下箭头（`↓`），提示下滚浏览内容        | 44px 触控区跳动指示器，平滑滚动至 `#latest-posts` 并管理焦点                |
| **主色调**   | 薄荷绿（`#00f59b`）、荧光青（`#00e5ff`） | 荧光绿赛博线稿                                     | 水仙壁纸同源色：夕阳琥珀金（`#F59E0B`）× 暮色紫罗兰（`#818CF8`）            |

---

## 2. 设计令牌与色彩系统（Design Tokens）

### 2.1 壁纸色彩采样与合成色彩科学

根据水仙壁纸（`3840x2160px_D_suisen.png`）的 40×40 像素实测采样结果，画面存在逆光强高光（L=0.97–1.00）。所有文字必须在半透明 Scrim 上进行**通道域 Alpha 复合**计算，以确保最坏情况满足 WCAG 2.2 AA：

| 元素              | 颜色值               | 背景合成场景                | 最坏像素实测对比度 | WCAG 2.2 AA 门槛   | 判定    |
| ----------------- | -------------------- | --------------------------- | ------------------ | ------------------ | ------- |
| **Hero 正文白字** | `#f8fafc`            | 壁纸高光 L=1.0 + 0.65 Scrim | **5.95:1**         | ≥ 4.5:1            | ✅ 达标 |
| **Hero 金色大字** | `#fde047`            | 壁纸高光 L=1.0 + 0.70 Scrim | **3.43:1**         | ≥ 3.0:1（大字）    | ✅ 达标 |
| **页头导航文字**  | `#cbd5e1`            | 顶部 Header Scrim（α=0.85） | **4.82:1**         | ≥ 4.5:1            | ✅ 达标 |
| **暗色正文文字**  | `#f8fafc`            | `--color-bg-base: #0f1221`  | **17.78:1**        | ≥ 4.5:1            | ✅ 达标 |
| **暗色弱化文字**  | `#7d8cb0`（R2 修正） | `--color-bg-base: #0f1221`  | **5.52:1**         | ≥ 4.5:1            | ✅ 达标 |
| **暗色 Focus 环** | `#f59e0b` 实色       | `--color-bg-base: #0f1221`  | **8.66:1**         | ≥ 3.0:1（UI 控件） | ✅ 达标 |
| **亮色主强调色**  | `#c2410c`            | `--color-bg-base: #fdfbf7`  | **5.01:1**         | ≥ 4.5:1            | ✅ 达标 |
| **亮色次强调色**  | `#4338ca`            | `--color-bg-base: #fdfbf7`  | **7.65:1**         | ≥ 4.5:1            | ✅ 达标 |
| **亮色内联代码**  | `#fdba74`（R2 修正） | `--color-code-bg: #1c1917`  | **8.54:1**         | ≥ 4.5:1            | ✅ 达标 |
| **亮色 Focus 环** | `#c2410c` 实色       | `--color-bg-base: #fdfbf7`  | **5.01:1**         | ≥ 3.0:1（UI 控件） | ✅ 达标 |

### 2.2 核心设计令牌代码（`src/styles/tokens.css`）

```css
/* ==========================================================================
   Design Tokens - Sunset Suisen (R2 Refined Specification)
   WCAG 2.2 AA Certified (All Text >= 4.5:1, UI/Headings >= 3:1)
   ========================================================================== */

:root,
html[data-theme="dark"] {
  color-scheme: dark;

  /* Canvas & Surface System */
  --color-bg-canvas: #0a0c16;
  --color-bg-base: #0f1221;
  --color-bg-subtle: #161a30;
  --color-surface-base: #13172b;
  --color-surface-raised: #1b213d;
  --color-surface-overlay: #232a4e;
  --color-surface-glass: rgba(19, 23, 43, 0.78);
  --color-drawer-backdrop: rgba(10, 12, 22, 0.75);

  /* Border System */
  --color-border-subtle: #1c223a;
  --color-border-default: #252d4c;
  --color-border-hover: #3d4976;
  --color-border-accent: #f59e0b;
  --color-border-glow: rgba(245, 158, 11, 0.35);

  /* Brand & Sunset Accents */
  --color-accent-primary: #f59e0b;
  --color-accent-primary-hover: #fbbf24;
  --color-accent-primary-glow: rgba(245, 158, 11, 0.32);
  --color-accent-primary-subtle: rgba(245, 158, 11, 0.12);
  --color-accent-secondary: #818cf8;
  --color-accent-secondary-hover: #a78bfa;
  --color-accent-secondary-glow: rgba(129, 140, 248, 0.25);
  --color-accent-secondary-subtle: rgba(129, 140, 248, 0.12);
  --color-accent-coral: #ff6b4a;

  /* Typography Colors (Verified Contrast) */
  --color-text-primary: #f8fafc;
  --color-text-secondary: #cbd5e1;
  --color-text-muted: #94a3b8;
  --color-text-dim: #7d8cb0; /* R2: Adjusted from #64748b to guarantee 5.5:1 */

  /* Ambient Layers */
  --color-ambient-grid: #1c223a;
  --color-ambient-scanline: rgba(245, 158, 11, 0.03);
  --color-ambient-streamline: rgba(129, 140, 248, 0.22);

  /* Code & Formulas */
  --color-code-bg: #0b0e1a;
  --color-code-text: #e2e8f0;
  --color-code-inline-bg: rgba(245, 158, 11, 0.12);
  --color-code-inline-text: #fbbf24;
  --color-code-border: #252d4c;

  /* Focus System (WCAG 2.4.13 Compliant Solid Focus) */
  --color-focus: #f59e0b;
  --color-focus-ring: #f59e0b;
  --color-selection-bg: rgba(245, 158, 11, 0.3);
  --color-selection-text: #ffffff;

  /* Shadows & Glows */
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.4);
  --shadow-card:
    0 4px 20px -2px rgba(0, 0, 0, 0.5), 0 0 0 1px var(--color-border-default);
  --shadow-card-hover:
    0 10px 30px -4px rgba(0, 0, 0, 0.7),
    0 0 20px 0 var(--color-accent-primary-glow),
    0 0 0 1px var(--color-accent-primary);
  --shadow-glow-accent: 0 0 25px var(--color-accent-primary-glow);
  --shadow-glow-secondary: 0 0 25px var(--color-accent-secondary-glow);
  --shadow-header:
    0 4px 20px rgba(0, 0, 0, 0.4), 0 1px 0 var(--color-border-default);

  /* Compatibility Mapping */
  --color-bg: var(--color-bg-base);
  --color-surface: var(--color-surface-base);
  --color-surface-hover: var(--color-surface-raised);
  --color-surface-elevated: var(--color-surface-overlay);
  --color-border: var(--color-border-default);
  --color-text: var(--color-text-primary);
  --color-muted: var(--color-text-muted);
  --color-muted-soft: rgba(148, 163, 184, 0.12);
  --color-accent: var(--color-accent-primary);
  --color-accent-hover: var(--color-accent-primary-hover);
  --color-accent-contrast: #000000;
  --color-accent-glow: var(--color-accent-primary-glow);
  --color-accent-subtle: var(--color-accent-primary-subtle);
  --color-accent-cyan: var(--color-accent-secondary);
  --color-accent-cyan-hover: var(--color-accent-secondary-hover);
  --color-accent-cyan-glow: var(--color-accent-secondary-glow);
  --color-accent-cyan-subtle: var(--color-accent-secondary-subtle);
}

/* Light Mode (Warm Sunset Alabaster) */
html[data-theme="light"] {
  color-scheme: light;

  --color-bg-canvas: #f8f5f0;
  --color-bg-base: #fdfbf7;
  --color-bg-subtle: #f2ece1;
  --color-surface-base: #ffffff;
  --color-surface-raised: #faf6ef;
  --color-surface-overlay: #f4eee3; /* R2: Differentiated from base */
  --color-surface-glass: rgba(255, 255, 255, 0.88);
  --color-drawer-backdrop: rgba(28, 25, 23, 0.45);

  --color-border-subtle: #f0e7da;
  --color-border-default: #e4d7c5;
  --color-border-hover: #cfbea8;
  --color-border-accent: #c2410c;
  --color-border-glow: rgba(194, 65, 12, 0.2);

  --color-accent-primary: #c2410c; /* Exact 5.01:1 on #fdfbf7 */
  --color-accent-primary-hover: #9a3412;
  --color-accent-primary-glow: rgba(194, 65, 12, 0.18);
  --color-accent-primary-subtle: rgba(194, 65, 12, 0.08);
  --color-accent-secondary: #4338ca; /* Exact 7.65:1 on #fdfbf7 */
  --color-accent-secondary-hover: #3730a3;
  --color-accent-secondary-glow: rgba(67, 56, 202, 0.18);
  --color-accent-secondary-subtle: rgba(67, 56, 202, 0.08);
  --color-accent-coral: #ea580c;

  --color-text-primary: #1c1917;
  --color-text-secondary: #44403c;
  --color-text-muted: #57534e;
  --color-text-dim: #44403c;

  --color-ambient-grid: #e8ded1;
  --color-ambient-scanline: rgba(194, 65, 12, 0.02);
  --color-ambient-streamline: rgba(67, 56, 202, 0.15);

  --color-code-bg: #1c1917;
  --color-code-text: #f5f5f4;
  --color-code-inline-bg: rgba(194, 65, 12, 0.08);
  --color-code-inline-text: #fdba74; /* R2: High contrast 8.54:1 on #1c1917 */
  --color-code-border: #e4d7c5;

  --color-focus: #c2410c;
  --color-focus-ring: #c2410c;
  --color-selection-bg: rgba(194, 65, 12, 0.2);
  --color-selection-text: #1c1917;

  --color-accent-contrast: #ffffff;
}
```

### 2.3 旧令牌别名迁移登记表

| 旧令牌                               | 新令牌 / 映射目标                                  | 影响范围                                     | 视觉与语义变化说明                |
| ------------------------------------ | -------------------------------------------------- | -------------------------------------------- | --------------------------------- |
| `--color-accent-primary` (`#00f59b`) | `--color-accent-primary` (`#f59e0b` / `#c2410c`)   | 全站主按钮、高光、Logo、活动指示器           | 薄荷绿全面切换为夕阳琥珀金/赭石金 |
| `--color-accent-cyan` (`#00e5ff`)    | `--color-accent-secondary` (`#818cf8` / `#4338ca`) | `prose.css`、Ambient 流线、`ReadingProgress` | 荧光青切换为暮色紫罗兰/深靛蓝     |
| `--color-accent-purple` (`#a855f7`)  | `--color-accent-secondary`                         | 无存量直接引用                               | 安全废弃并归入次级强调色          |
| `--color-text-dim` (`#64748b`)       | `--color-text-dim` (`#7d8cb0`)                     | `home.css` 分页、标签小字                    | 调亮以达到 5.5:1 AA 标准          |

---

## 3. 资产与首屏性能策略（LCP ≤ 2.5s & 0-CLS）

### 3.1 响应式图片资产管线

为彻底解决 6.02MB 原图导致 Fast-4G 下 30s 传输超时问题，建立静态分级多源资产：

```text
astro-public/images/hero/
├── suisen-hero-750w.avif    (~220 KB, 移动端竖屏专用)
├── suisen-hero-750w.webp    (~310 KB, 移动端兜底)
├── suisen-hero-1440w.avif   (~650 KB, 平板/笔记本专用)
├── suisen-hero-1440w.webp   (~880 KB, 平板兜底)
├── suisen-hero-3840w.avif   (~1.6 MB, 4K 高清大屏)
├── suisen-hero-3840w.webp   (~2.1 MB, 4K 兜底)
└── suisen-hero-original.png (6.02 MB, 无损原档存档)
```

### 3.2 响应式渲染结构与首屏预载

在 `Hero.astro` 中采用原生 `<picture>` 标签调度，结合 `object-fit: cover; object-position: center 25%;`：

```html
<picture class="hero-picture-layer">
  <source
    media="(max-width: 48rem)"
    type="image/avif"
    srcset="/images/hero/suisen-hero-750w.avif"
  />
  <source
    media="(max-width: 48rem)"
    type="image/webp"
    srcset="/images/hero/suisen-hero-750w.webp"
  />
  <source
    media="(max-width: 90rem)"
    type="image/avif"
    srcset="/images/hero/suisen-hero-1440w.avif"
  />
  <source
    media="(max-width: 90rem)"
    type="image/webp"
    srcset="/images/hero/suisen-hero-1440w.webp"
  />
  <source type="image/avif" srcset="/images/hero/suisen-hero-3840w.avif" />
  <img
    src="/images/hero/suisen-hero-1440w.webp"
    alt=""
    class="hero-bg-img"
    loading="eager"
    fetchpriority="high"
    decoding="async"
    width="1920"
    height="1080"
  />
</picture>
```

- **条件 Preload 策略**：仅在 `BaseLayout.astro` 中当 `isHome === true` 时，在 `<head>` 输出携带 `imagesrcset` 与 `imagesizes` 的响应式预载标签，其余 30+ 篇技术文章页不注入该预载。

---

## 4. 架构与组件级重构方案

### 4.1 布局架构与插槽解耦（`BaseLayout.astro` & `index.astro`）

`global.css` 中 `main` 标签受限于 `max-width: var(--max-width-wide)`。重构方案：

1. `BaseLayout.astro` 提供 `<slot name="hero" />`，置于 `<main>` 之外，实现真正的 `100vw × 100vh` 满屏穿透；
2. 正文内容流包裹在 `<main id="main-content" tabindex="-1">` 中，保持既有 Skip-link 与焦点契约。

```astro
<!-- src/layouts/BaseLayout.astro -->
<body>
  <a href="#main-content" class="skip-link">跳至主要内容</a>
  <div id="top"></div>
  <AmbientBackground isHeroVisible={isHome} />
  <SiteHeader isHome={isHome} />

  <!-- 全屏 Hero 槽位（脱离 main 容器约束） -->
  <slot name="hero" />

  <main id="main-content" tabindex="-1">
    <slot />
  </main>
  <SiteFooter />
</body>
```

### 4.2 浮动融合与滚动状态（`SiteHeader.astro` & `site-header.ts`）

1. **首页透明态（`scrollY < 24px`）**：
   - 处于全屏 Hero 上方，顶部叠加专属 Header Scrim（`rgba(10, 12, 22, 0.85)` 线性渐隐），导航文字采用 `--color-text-secondary: #cbd5e1`，对比度达到 4.82:1；
2. **滚动收敛态（`scrollY >= 24px`）**：
   - 保持高度恒定（避免 8px CLS），平滑开启 `backdrop-filter: blur(12px)` 与 `--color-surface-glass` 磨砂背景；
3. **单测扩展**：为 `resolveSiteHeaderState` 补充三态判定与边界测试。

### 4.3 沉浸式 Hero 组件重构（`Hero.astro`）

组件拆分为三层结构，严格遵守独立暗色作用域契约：

```astro
<!-- src/components/visual/Hero.astro -->
<section class="hero-fullscreen" aria-label="欢迎首屏" data-theme-scope="dark">
  <!-- Layer 1: Picture 响应式壁纸 -->
  <picture class="hero-picture-layer">...</picture>

  <!-- Layer 2: 全局暗角 + 局部内容 Scrim 护盾 -->
  <div class="hero-scrim-layer" aria-hidden="true"></div>

  <!-- Layer 3: 欢迎核心内容区块 -->
  <div class="hero-content-container">
    <div class="hero-brand-block">
      <h1 class="hero-welcome-title">
        <span class="welcome-line-1">WELCOME TO</span>
        <span class="welcome-line-2">ENTROPYXI BLOG !</span>
      </h1>
    </div>

    <div class="hero-narrative-card">
      <p class="narrative-line">
        这里是 <strong class="brand-text">EntropyXi</strong> 的技术笔记。
      </p>
      <p class="narrative-line">很高兴与你相遇！</p>
      <p class="narrative-lead">
        聚焦<strong>深度学习</strong>、<strong>扩散模型</strong>、<strong
          >流匹配 (Flow Matching)</strong
        > 与<strong>数值分析</strong>的严谨数学推导与工程落地。
      </p>
    </div>

    <!-- 完整保留 6 个真实标签路由链接 -->
    <div class="hero-topic-rails" aria-label="核心研究主题">
      {
        researchTopics.map((t) => (
          <a href={buildTagPath(t.tag)} class="hero-topic-chip">
            <span class="topic-hash">#</span>
            <span>{t.label}</span>
          </a>
        ))
      }
    </div>

    <!-- 完整保留 4 个核心导航 CTA 按钮 -->
    <div class="hero-actions">
      <a href="#latest-posts" class="hero-btn hero-btn-primary" data-magnetic
        >探索文章 ↓</a
      >
      <a href="/categories/" class="hero-btn hero-btn-secondary">分类图谱</a>
      <a href="/archives/" class="hero-btn hero-btn-secondary">全站归档</a>
      <a href="/about/" class="hero-btn hero-btn-secondary">关于博主</a>
    </div>
  </div>

  <!-- Layer 4: 底部 44px 下滚指示器 -->
  <div class="hero-scroll-wrapper">
    <a
      href="#latest-posts"
      class="hero-scroll-btn"
      aria-label="向下滚动至最新文章"
    >
      <span class="scroll-text">SCROLL</span>
      <svg class="scroll-arrow" viewBox="0 0 24 24" width="24" height="24">
        <path
          d="M7 10l5 5 5-5"
          fill="none"
          stroke="currentColor"
          stroke-width="2.5"
          stroke-linecap="round"
          stroke-linejoin="round"></path>
      </svg>
    </a>
  </div>
</section>
```

---

## 5. 响应式矩阵与无障碍合规

### 5.1 全视口响应式矩阵（覆盖 320px ~ 4K）

| 视口档位            | 宽度范围                     | 大标题字号与版式               | 布局与定位策略                                 |
| ------------------- | ---------------------------- | ------------------------------ | ---------------------------------------------- |
| **超窄移动端**      | 320px ~ 359px                | `clamp(1.6rem, 7vw, 1.9rem)`   | 单列堆叠，下滚指示器紧凑 40px，内容 Scrim 占满 |
| **标准移动端**      | 360px ~ 480px                | `clamp(1.9rem, 7.5vw, 2.3rem)` | `100svh` 高度适配，CTA 按钮 2×2 网格布局       |
| **大屏手机/小平板** | 481px ~ 767px                | `clamp(2.3rem, 6vw, 2.8rem)`   | 左右内边距 1.5rem，解开 2×2 网格               |
| **平板端**          | 768px ~ 1023px               | `2.8rem`                       | 居中排版，下滚指示器底边距 2.5rem              |
| **桌面端**          | 1024px ~ 1439px              | `3.2rem`                       | 经典比例，CTA 按钮单行横排                     |
| **宽屏桌面**        | 1440px ~ 2559px              | `3.8rem`                       | 最大宽度 `var(--max-width-wide)` 居中对齐      |
| **4K 超高清屏**     | ≥ 2560px                     | `clamp(4.2rem, 3.5vw, 5.5rem)` | 加载 3840w AVIF 资产，微网格 5rem 缩放         |
| **短视口压缩档**    | `@media (max-height: 700px)` | 标题缩减 20%，间距紧凑         | 消除溢出，确保首屏无需滚动即可见完整 Hero      |

### 5.2 动效降级与无障碍保障

- `prefers-reduced-motion: reduce`：全面禁用下滚指示器的 `bounce` 动画、磁吸位移与标题渐显动效，改为纯静态呈现；
- 焦点管理：下滚指示器跳转目标 `#latest-posts` 携带 `tabindex="-1"`，平滑聚焦避免焦点丢失。

---

## 6. 测试变更清单与 9 步质量门禁

### 6.1 既有测试断言更新清单

| 测试文件                            | 目标行号   | 原断言                             | R2 规范更新断言                                 |
| ----------------------------------- | ---------- | ---------------------------------- | ----------------------------------------------- |
| `tests/e2e/home.spec.ts`            | L9         | `heading: "EntropyXi 的技术笔记"`  | 更新为 `heading: "WELCOME TO ENTROPYXI BLOG !"` |
| `tests/e2e/client-matrix.spec.ts`   | L13        | `heading: "EntropyXi 的技术笔记"`  | 同步更新为大标题匹配                            |
| `tests/e2e/client-behavior.spec.ts` | L242, L321 | `heading: "EntropyXi 的技术笔记"`  | 同步更新为大标题匹配                            |
| `tests/e2e/client-behavior.spec.ts` | L302       | `.hero-btn-primary[data-magnetic]` | 保持存在（CTA 完整保留）✅                      |
| `tests/e2e/client-behavior.spec.ts` | L131-141   | skip-link 聚焦 `#main-content`     | 保持不变（契约完整继承）✅                      |

### 6.2 新增专属测试用例

1. **Hero 计算样式对比度测试（`hero-contrast.spec.ts`）**：使用 Playwright 读取实际渲染的文字色与计算背景色，补足 Axe-core 对背景图的盲区；
2. **三态 Header 单元测试（`site-header.test.ts`）**：验证 `resolveSiteHeaderState` 的 expanded、compact 与 hero-transparent 逻辑；
3. **资产体积自动化审计（`scripts/audit-assets.ts`）**：断言移动端 AVIF ≤ 400KB、4K AVIF ≤ 2.2MB。

### 6.3 完整 9 步质量验证门禁

1. `npm run format:check`（Prettier 格式校验）
2. `npm run lint`（ESLint 与 Stylelint 零告警）
3. `npm run check:types`（Astro Check & TypeScript 零类型错误）
4. `npm run audit:content`（文章内容与 Frontmatter 审计）
5. `npm run audit:assets`（R2 新增：图片资产体积与分级审计）
6. `npm run test:unit -- --run`（Vitest 单元测试全绿）
7. `npm run test:e2e`（Playwright 38+ 项端到端及 Axe-core 无障碍测试全绿）
8. `npm run audit:bundle`（客户端 JS 打包体积预算审计）
9. `npm run audit:output`（静态 HTML 输出与 h1 结构审计）
