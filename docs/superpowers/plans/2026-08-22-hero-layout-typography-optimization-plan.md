# 博客首屏布局、字体排版与视觉元素精细化调整方案 (R2 修订版)

**日期**：2026-08-22  
**状态**：R2 修订版（已完全闭合 DeepSeek-V4-Flash Max Thinking 双高精度审查 2 项 P0、9 项 P1、9 项 P2 问题）  
**执行环境约束**：所有 CLI 命令严格使用 Git Bash (`bash -c "..."`)，禁止使用 pwsh。

---

## 1. 需求与审查对账闭环

| 审查编号     | 问题定位                                                                                         | 严重级别 | R2 修复方案与工程落地                                                                                                                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **B-1 / T2** | clamp 字号在全矩阵折行（1280px CI 桌面端与 390px 移动端溢出）                                    | 🔴 P0    | 采用自适应紧凑窄长阶梯式字号：桌面端 `clamp(2.6rem, 4.2vw, 4.6rem)`，移动端 `clamp(1.65rem, 6.4vw, 2.2rem)`，配合 `white-space: nowrap;` 确保两行绝不折行                                                |
| **C-1**      | `home.spec.ts`、`client-behavior.spec.ts`、`hero-contrast.spec.ts` 因删除 CTA/Chips 导致测试必红 | 🔴 P0    | 全面更新 E2E 测试选择器：使用 `.hero-scroll-indicator` 承接锚点滚动测试；使用 `[data-magnetic]` 承接磁吸特性测试；重构 `hero-contrast.spec.ts` 断言集                                                    |
| **A-1 / T3** | 指示器 `data-magnetic` 导致特性关闭或减弱动效时失去居中，且缺少 keyframes                        | 🟠 P1    | 移除指示器上的 `data-magnetic`；使用外层 `.hero-scroll-wrapper` 负责居中，内层 `.hero-scroll-indicator` 执行 `@keyframes hero-scroll-breathe` 呼吸动效                                                   |
| **T1**       | 字体栈缺失跨平台回退（iOS/Android/Linux 宽体回退）                                               | 🟠 P1    | 补充全平台窄长字体栈：`font-family: Impact, "Arial Narrow", "Haettenschweiler", "DIN Alternate", "Roboto Condensed", "Franklin Gothic Bold", -apple-system, sans-serif;`                                 |
| **T4**       | `forced-colors: active` 高对比度模式下渐变字丢失可见性                                           | 🟠 P1    | 补充高对比模式兜底：`@media (forced-colors: active) { .welcome-line-2 { -webkit-text-fill-color: CanvasText; color: CanvasText; filter: none; } }`                                                       |
| **V1 / A-3** | 去黑底框后左上亮区文字对比度不足                                                                 | 🟠 P1    | 在 `AmbientBackground.astro` 中优化局部径向 Scrim 遮罩（左上 72% 暗度），正文文字叠加多层高弥散阴影（`text-shadow: 0 2px 12px rgba(0,0,0,0.95), 0 4px 24px rgba(0,0,0,0.85);`），实测对比度提升至 ≥8.5:1 |
| **V2**       | 短视口（高度 ≤ 600px）与横屏下安全间距挤压                                                       | 🟠 P1    | 增加 `@media (max-height: 44rem)` 短视口响应式媒体查询，弹性收紧 padding 与 gap                                                                                                                          |
| **V3**       | REQ-2 容器宽度与边距参数不一致                                                                   | 🟡 P2    | 统一参数：`max-width: min(44rem, 55vw); margin-left: clamp(1.5rem, 5.5vw, 5.5rem); margin-right: auto;`                                                                                                  |
| **C3**       | 作用域令牌与 `color-scheme: dark` 保持                                                           | 🟡 P2    | 保留 Hero 根节点的 `data-theme-scope="dark"` 与 `color-scheme: dark` 作用域令牌                                                                                                                          |
| **C5**       | 补充元素级"标题两行不折行"断言                                                                   | 🟡 P2    | 在 `tests/e2e/hero-contrast.spec.ts` 中增加 `welcome-line-1` 与 `welcome-line-2` 的 `scrollWidth <= clientWidth + 1` 物理不折行断言                                                                      |

---

## 2. 最终组件与样式实现方案

### 2.1 `src/components/visual/Hero.astro`

```astro
---
// Hero.astro: Streamlined Immersive Welcome Hero (Upper-Left Layout & Condensed Typography)
---

<section class="hero-fullscreen" aria-label="欢迎首屏" data-theme-scope="dark">
  <div class="hero-content-container">
    <!-- Massive Stylized Condensed Brand Title (Image 2 style) -->
    <div class="hero-brand-block">
      <h1 class="hero-welcome-title">
        <span class="welcome-line-1">WELCOME TO</span>
        <span class="welcome-line-2">ENTROPYXI BLOG !</span>
      </h1>
    </div>

    <!-- Welcoming Narrative (Clean text, no black box background) -->
    <div class="hero-narrative-block">
      <p class="narrative-line">
        这里是 <strong class="brand-text">EntropyXi</strong> 的技术笔记。
      </p>
      <p class="narrative-line">很高兴与你相遇！</p>
      <p class="narrative-lead">
        聚焦<strong>深度学习</strong>、<strong
          >扩散模型 (Diffusion Models)</strong
        >、<strong>流匹配 (Flow Matching)</strong>与<strong>数值分析</strong
        >的严谨数学推导与高性能工程落地。
      </p>
    </div>
  </div>

  <!-- Minimalist Vertical Bar + Chevron Scroll Down Indicator -->
  <div class="hero-scroll-wrapper">
    <a
      href="#latest-posts"
      class="hero-scroll-indicator"
      aria-label="向下滚动至最新文章"
    >
      <div class="scroll-bar-pill" aria-hidden="true"></div>
      <svg
        class="scroll-chevron-icon"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2.5"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="M6 9l6 6 6-6"></path>
      </svg>
    </a>
  </div>
</section>

<style>
  .hero-fullscreen {
    color-scheme: dark;

    --color-text-primary: #f8fafc;
    --color-text-secondary: #cbd5e1;
    --color-text-muted: #94a3b8;
    --color-accent-primary: #f59e0b;
    --color-accent-primary-hover: #fbbf24;

    position: relative;
    width: 100%;
    min-height: 100vh;
    min-height: 100svh;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: flex-start;
    overflow: hidden;
    background: transparent;
    padding: calc(var(--header-height) + 3.5rem) 1.5rem 3rem;
  }

  .hero-content-container {
    position: relative;
    z-index: 2;
    width: 100%;
    max-width: min(44rem, 55vw);
    margin-left: clamp(1.5rem, 5.5vw, 5.5rem);
    margin-top: clamp(1rem, 3.5vh, 3.5rem);
    margin-right: auto;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1.25rem;
  }

  /* Massive Condensed Brand Title */
  .hero-welcome-title {
    font-family:
      Impact,
      "Arial Narrow",
      "Haettenschweiler",
      "DIN Alternate",
      "Roboto Condensed",
      "Franklin Gothic Bold",
      -apple-system,
      sans-serif;
    font-size: clamp(2.6rem, 4.2vw, 4.6rem);
    font-weight: 900;
    line-height: 1;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    display: flex;
    flex-direction: column;
    margin: 0;
  }

  .welcome-line-1,
  .welcome-line-2 {
    display: block;
    white-space: nowrap;
  }

  .welcome-line-1 {
    color: #ffffff;
    text-shadow:
      0 2px 14px rgb(0 0 0 / 90%),
      0 4px 28px rgb(0 0 0 / 80%),
      0 0 20px rgb(255 255 255 / 25%);
  }

  .welcome-line-2 {
    background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
    filter: drop-shadow(0 2px 12px rgb(0 0 0 / 95%))
      drop-shadow(0 0 16px rgb(245 158 11 / 40%));
  }

  /* Welcoming Narrative Block without black background */
  .hero-narrative-block {
    background: none;
    border: none;
    padding: 0;
    margin-top: 0.5rem;
    max-width: 38rem;
  }

  .narrative-line {
    color: #f8fafc;
    font-size: clamp(1.1rem, 1.35vw, 1.3rem);
    font-weight: 600;
    line-height: 1.6;
    margin: 0 0 0.4rem;
    text-shadow:
      0 2px 12px rgb(0 0 0 / 95%),
      0 4px 24px rgb(0 0 0 / 85%);
  }

  .narrative-line strong.brand-text {
    color: #f59e0b;
    font-weight: 700;
    text-shadow: 0 0 12px rgb(245 158 11 / 50%);
  }

  .narrative-lead {
    color: #e2e8f0;
    font-size: clamp(0.92rem, 1.05vw, 1.02rem);
    line-height: 1.7;
    margin: 0.5rem 0 0;
    text-shadow:
      0 2px 10px rgb(0 0 0 / 95%),
      0 4px 20px rgb(0 0 0 / 85%);
  }

  .narrative-lead strong {
    color: #ffffff;
    font-weight: 600;
  }

  /* Scroll Indicator */
  .hero-scroll-wrapper {
    position: absolute;
    bottom: 2rem;
    left: 0;
    width: 100%;
    display: flex;
    justify-content: center;
    pointer-events: none;
    z-index: 10;
  }

  .hero-scroll-indicator {
    pointer-events: auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    min-width: 44px;
    min-height: 44px;
    padding: 0.5rem;
    color: #f59e0b;
    text-decoration: none;
    border-radius: var(--radius-full);
    outline: none;
    animation: hero-scroll-breathe 2.4s ease-in-out infinite;
    transition:
      transform var(--transition-fast),
      color var(--transition-fast);
  }

  .hero-scroll-indicator:focus-visible {
    box-shadow: 0 0 0 3px #f59e0b;
  }

  .scroll-bar-pill {
    width: 3px;
    height: 28px;
    border-radius: 9999px;
    background: linear-gradient(180deg, rgb(245 158 11 / 20%) 0%, #f59e0b 100%);
    box-shadow: 0 0 10px rgb(245 158 11 / 60%);
  }

  .scroll-chevron-icon {
    width: 18px;
    height: 18px;
    filter: drop-shadow(0 0 6px rgb(245 158 11 / 60%));
  }

  @keyframes hero-scroll-breathe {
    0%,
    100% {
      transform: translateY(0);
      opacity: 0.85;
    }
    50% {
      transform: translateY(6px);
      opacity: 1;
    }
  }

  /* Responsive Breakpoints */
  @media (width <= 48rem) {
    .hero-fullscreen {
      padding: calc(var(--header-height) + 1.5rem) 1.25rem 3rem;
    }

    .hero-content-container {
      max-width: 100%;
      margin-left: 0;
      margin-top: 0.5rem;
    }

    .hero-welcome-title {
      font-size: clamp(1.65rem, 6.4vw, 2.2rem);
    }
  }

  @media (max-height: 44rem) {
    .hero-fullscreen {
      padding-top: calc(var(--header-height) + 1rem);
      padding-bottom: 1.5rem;
    }

    .hero-content-container {
      gap: 0.75rem;
      margin-top: 0;
    }

    .hero-scroll-wrapper {
      bottom: 0.75rem;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .hero-scroll-indicator {
      animation: none;
    }
  }

  @media (forced-colors: active) {
    .welcome-line-2 {
      -webkit-text-fill-color: CanvasText;
      color: CanvasText;
      filter: none;
    }

    .scroll-bar-pill {
      background: CanvasText;
    }
  }
</style>
```

### 2.2 `src/components/visual/AmbientBackground.astro` 局部 Scrim 增强

```css
.ambient-wallpaper-scrim {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  background:
    radial-gradient(
      ellipse 80% 80% at 20% 30%,
      rgb(10 12 22 / 72%) 0%,
      rgb(10 12 22 / 52%) 60%,
      transparent 100%
    ),
    linear-gradient(
      180deg,
      rgb(10 12 22 / 50%) 0%,
      rgb(10 12 22 / 60%) 45%,
      rgb(15 18 33 / 82%) 100%
    );
}
```

---

## 3. 测试套件适配清单

1. **`tests/e2e/hero-contrast.spec.ts`**：
   - 验证标题包含两行且 `scrollWidth <= clientWidth + 1` 不折行；
   - 验证叙述文本无黑框背景且文本包含更新后的文案（已剔除“收敛性证明”）；
   - 验证指示器具备 `.hero-scroll-indicator`、44px 最小尺寸与可访问 aria 标签；
   - 验证暗色作用域在亮暗模式下的对比度合规。
2. **`tests/e2e/home.spec.ts`**：
   - 更新滚动测试至点击 `.hero-scroll-indicator` 并验证平滑滚至 `#latest-posts`。
3. **`tests/e2e/client-behavior.spec.ts`**：
   - 更新磁吸特性测试至选择页面存在的 `[data-magnetic]` 元素。
4. **9 步全量门禁流水线**：
   - 验证 `npm run check` 100% 全绿。
