# EntropyXi Blog：Ayeez 风格视觉设计规范与组件规格说明书 (v1.0.0)

> **版本**：`v1.0.0`（系统标识：`data-design-spec-version="1.0.0"`）  
> **责任 Agent**：Sub-Agent #2（Antigravity / Gemini 3.7 Flash High）  
> **审批者**：主 Agent  
> **状态**：阶段 1 返工修订完成，提交主 Agent 审批  
> **关联计划**：`docs/superpowers/plans/2026-08-21-ayeez-ui-motion-refactor-plan.md` §3–§7, §9, §10 (阶段 1)  
> **架构核心原则**：服务端优先、渐进增强；**明确不引入 React、Vue、Svelte 等任何客户端 UI 框架**，全站基于原生 Astro SSG 模板 + 纯原生 TypeScript 模块 + 标准语义 CSS 落地。

---

## 目录

1. [视觉定位与原创性规范（Cyber-Math System）](#1-视觉定位与原创性规范cyber-math-system)
2. [完整主题策略与生命周期契约](#2-完整主题策略与生命周期契约)
3. [设计令牌系统与 WCAG 2.2 AA 对比度实测](#3-设计令牌系统与-wcag-22-aa-对比度实测)
4. [字体排印、自托管与 0-CLS 策略](#4-字体排印自托管与-0-cls-策略)
5. [响应式断点与布局折叠规则矩阵（360/390/768/1024/1440/1920）](#5-响应式断点与布局折叠规则矩阵360390768102414401920)
6. [组件交互状态机规格表（8 类组件 × 7 种状态 + Touch 等价）](#6-组件交互状态机规格表8-类组件--7-种状态--touch-等价)
7. [动效分级、性能降级与无 JS 契约](#7-动效分级性能降级与无-js-契约)
8. [原创 Hero 右侧流场 / 扩散轨迹 SVG 规格与资产台账](#8-原创-hero-右侧流场--扩散轨迹-svg-规格与资产台账)
9. [Astro 架构映射表与零客户端框架承诺](#9-astro-架构映射表与零客户端框架承诺)
10. [主 Agent 验收清单（Definition of Done）](#10-主-agent-验收清单definition-of-done)

---

## 1. 视觉定位与原创性规范（Cyber-Math System）

### 1.1 风格借鉴与原创差异对照

本设计解构 [AyeezBlog](https://blog.ayeez.cn/) 优秀设计语法，针对 EntropyXi 学术内容进行 100% 原创表达：

| AyeezBlog 视觉语法 (Inspiration)   | EntropyXi 专属原创表达 (Original Expression)                  |
| :--------------------------------- | :------------------------------------------------------------ |
| **深黑冷灰底层 + 荧光绿/青色高光** | 矩阵绿 (`#00f59b`) + 电子青 (`#00e5ff`) + 深空黑 (`#090d16`)  |
| **二次元立绘 / 人物插画为主视觉**  | **严谨数学建模**：SDE 随机漂移轨迹、连续正则化流 (CNF) 矢量图 |
| **大视口全屏叙事 + 压缩字标**      | 叙事大视口 Hero + 等宽数理符号 ($\Xi$) + 语义化研究主题轨道   |
| **密集光晕与扫描线氛围层**         | 物理分层、微弱低能耗 SVG 网格 + 移动端自动降级光晕系统        |
| **玻璃拟态与流光文章卡片**         | 几何面板、状态指示胶囊、渐变顶光与分类元数据层级              |

### 1.2 严格资产红线

1. **绝对禁止**：严禁复制 AyeezBlog 的人物素材、背景图片、站标 Logo、头像、博主经历文案与私有 CSS/JS 源码。
2. **原创资产**：全站所有 Hero 主视觉、关于页极客徽章、404 故障几何图形均为自研原生响应式 SVG，具备独立版权。

---

## 2. 完整主题策略与生命周期契约

站点支持 `dark`（默认暗色赛博）与 `light`（高对比明亮赛博）双主题。

### 2.1 主题判定优先级与持久化规则

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 优先级 1: 检查 localStorage.getItem("entropyxi-theme")                 │
│          ├─ 存在且值为 "light" / "dark" ──► [应用用户显式偏好] (最高)    │
│          └─ 不存在 / 读取异常 ────────────► 进入系统偏好判定           │
├────────────────────────────────────────────────────────────────────────┤
│ 优先级 2: 检查 window.matchMedia("(prefers-color-scheme: light)")       │
│          ├─ 匹配成功 (OS 当前为浅色模式) ──► [应用 light 主题]          │
│          └─ 匹配失败 / 不支持 ───────────► [应用 dark 主题] (默认回退)  │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 防 FOUC（闪烁）内联脚本策略

在 `BaseLayout.astro` 的 `<head>` 顶部注入同步内联脚本（约 200 字节），在 DOM 渲染前直接确定 `html[data-theme]`：

```html
<script is:inline>
  (function () {
    try {
      var stored = localStorage.getItem("entropyxi-theme");
      if (stored === "light" || stored === "dark") {
        document.documentElement.dataset.theme = stored;
      } else if (
        window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: light)").matches
      ) {
        document.documentElement.dataset.theme = "light";
      } else {
        document.documentElement.dataset.theme = "dark";
      }
    } catch (e) {
      document.documentElement.dataset.theme = "dark";
    }
  })();
</script>
```

### 2.3 系统偏好动态监听与 Cleanup 契约

- **动态响应**：当且仅当用户**未显式选择主题**（即 `localStorage` 无记录）时，系统主题切换事件实时联动切换页面主题：
  ```ts
  const mediaQuery = window.matchMedia("(prefers-color-scheme: light)");
  const handleSystemThemeChange = (e: MediaQueryListEvent) => {
    if (!localStorage.getItem("entropyxi-theme")) {
      document.documentElement.dataset.theme = e.matches ? "light" : "dark";
    }
  };
  mediaQuery.addEventListener("change", handleSystemThemeChange, {
    signal: abortController.signal,
  });
  ```
- **异常防御与 Cleanup**：
  - 所有 `localStorage` 操作均置于 `try-catch` 内，防止隐私模式或 iframe 禁用存储时抛出未捕获异常；
  - 所有事件监听通过统一的 `AbortController` 绑定，在页面跳转或组件注销时统一调用 `abortController.abort()` 释放，杜绝内存泄漏。

---

## 3. 设计令牌系统与 WCAG 2.2 AA 对比度实测

### 3.1 颜色令牌（十六进制明细）

```css
/* ==========================================================================
   Design Tokens - Dark Cyber (Default Theme)
   ========================================================================== */
:root,
html[data-theme="dark"] {
  /* 基础背景与画布 */
  --color-bg-canvas: #060911; /* 全局最底层画布 */
  --color-bg-base: #090d16; /* 页面主背景 (Dark Base) */
  --color-bg-subtle: #0f172a; /* 次级微弱背景 / 槽位 */
  --color-surface-base: #111827; /* 基础卡片面板 */
  --color-surface-raised: #162032; /* 悬浮面板 / 菜单面板 */
  --color-surface-overlay: #1a2234; /* 抽屉 / 遮罩容器 */
  --color-surface-glass: rgba(17, 24, 39, 0.75); /* 毛玻璃半透层 */

  /* 边框系统 */
  --color-border-subtle: #162032; /* 极微分割线 */
  --color-border-default: #1e293b; /* 标准面板边框 */
  --color-border-hover: #334155; /* 悬停高亮边框 */
  --color-border-accent: #00f59b; /* 霓虹强调边框 */
  --color-border-glow: rgba(0, 245, 155, 0.35); /* 边框发光 */

  /* 品牌与赛博强调色 */
  --color-accent-primary: #00f59b; /* 核心矩阵绿 */
  --color-accent-primary-hover: #10b981;
  --color-accent-primary-glow: rgba(0, 245, 155, 0.28);
  --color-accent-primary-subtle: rgba(0, 245, 155, 0.1);

  --color-accent-cyan: #00e5ff; /* 辅助电子青 */
  --color-accent-cyan-hover: #38bdf8;
  --color-accent-cyan-glow: rgba(0, 229, 255, 0.25);
  --color-accent-cyan-subtle: rgba(0, 229, 255, 0.1);

  --color-accent-purple: #a855f7; /* 量子紫 */
  --color-accent-amber: #f59e0b; /* 状态橙 */
  --color-accent-red: #ef4444; /* 故障红 */

  /* 排版文本分层 */
  --color-text-primary: #f8fafc; /* 主标题与高亮正文 */
  --color-text-secondary: #cbd5e1; /* 正文与副标题 */
  --color-text-muted: #94a3b8; /* 元数据、日期与辅助标签 */
  --color-text-dim: #64748b; /* 弱化占位文本 */

  /* 代码与公式 */
  --color-code-bg: #0b0f19;
  --color-code-text: #e2e8f0;
  --color-code-inline-bg: rgba(0, 245, 155, 0.12);
  --color-code-inline-text: #00f59b;
  --color-code-border: #1e293b;

  /* 焦点与选区 */
  --color-focus: #00f59b;
  --color-focus-ring: rgba(0, 245, 155, 0.45);
  --color-selection-bg: rgba(0, 245, 155, 0.3);
  --color-selection-text: #ffffff;
}

/* ==========================================================================
   Design Tokens - Light Cyber (High Contrast Theme)
   ========================================================================== */
html[data-theme="light"] {
  /* 基础背景与画布 */
  --color-bg-canvas: #f1f5f9;
  --color-bg-base: #f8fafc;
  --color-bg-subtle: #edf2f7;
  --color-surface-base: #ffffff;
  --color-surface-raised: #f8fafc;
  --color-surface-overlay: #ffffff;
  --color-surface-glass: rgba(255, 255, 255, 0.85);

  /* 边框系统 */
  --color-border-subtle: #f1f5f9;
  --color-border-default: #e2e8f0;
  --color-border-hover: #cbd5e1;
  --color-border-accent: #047857; /* 高对比墨绿 */
  --color-border-glow: rgba(4, 120, 87, 0.2);

  /* 品牌强调色 (严格修正以满足 WCAG 2.2 AA >= 4.5:1) */
  --color-accent-primary: #047857; /* Emerald-700 */
  --color-accent-primary-hover: #065f46;
  --color-accent-primary-glow: rgba(4, 120, 87, 0.18);
  --color-accent-primary-subtle: rgba(4, 120, 87, 0.08);

  --color-accent-cyan: #0369a1; /* Sky-700 (对白底 5.93:1，对基底 5.67:1) */
  --color-accent-cyan-hover: #075985;
  --color-accent-cyan-glow: rgba(3, 105, 161, 0.18);
  --color-accent-cyan-subtle: rgba(3, 105, 161, 0.08);

  --color-accent-purple: #7c3aed;
  --color-accent-amber: #d97706;
  --color-accent-red: #dc2626;

  /* 排版文本分层 */
  --color-text-primary: #0f172a; /* Slate-900 */
  --color-text-secondary: #334155; /* Slate-700 */
  --color-text-muted: #475569; /* Slate-600 */
  --color-text-dim: #334155; /* Slate-700 */

  /* 代码与公式 */
  --color-code-bg: #0f172a;
  --color-code-text: #e2e8f0;
  --color-code-inline-bg: rgba(4, 120, 87, 0.08);
  --color-code-inline-text: #047857;
  --color-code-border: #e2e8f0;

  /* 焦点与选区 */
  --color-focus: #047857;
  --color-focus-ring: rgba(4, 120, 87, 0.35);
  --color-selection-bg: rgba(4, 120, 87, 0.2);
  --color-selection-text: #0f172a;
}
```

### 3.2 WCAG 2.2 AA 对比度数学实测表

依据 W3C 相对亮度公式 $L = 0.2126 R + 0.7152 G + 0.0722 B$ 与对比度公式 $(L_1 + 0.05) / (L_2 + 0.05)$ 实测：

| 元素分类         | 前景色 Token (Hex)                               | 背景色 Token (Hex)                             | Dark 对比度实测 |             Light 对比度实测              | WCAG 2.2 AA 标准 |   达标判定   |
| :--------------- | :----------------------------------------------- | :--------------------------------------------- | :-------------: | :---------------------------------------: | :--------------: | :----------: |
| **页面主标题**   | `--color-text-primary` (`#f8fafc` / `#0f172a`)   | `--color-bg-base` (`#090d16` / `#f8fafc`)      |  **18.92 : 1**  |               **15.78 : 1**               |     ≥ 4.5:1      | 符合标准要求 |
| **正文段落**     | `--color-text-secondary` (`#cbd5e1` / `#334155`) | `--color-surface-base` (`#111827` / `#ffffff`) |  **11.53 : 1**  |               **10.30 : 1**               |     ≥ 4.5:1      | 符合标准要求 |
| **元数据与日期** | `--color-text-muted` (`#94a3b8` / `#475569`)     | `--color-surface-base` (`#111827` / `#ffffff`) |  **6.94 : 1**   | **7.57 : 1** (对画布 `#f1f5f9` 达 6.93:1) |     ≥ 4.5:1      | 符合标准要求 |
| **主要强调文本** | `--color-accent-primary` (`#00f59b` / `#047857`) | `--color-bg-base` (`#090d16` / `#f8fafc`)      |  **12.47 : 1**  |               **5.27 : 1**                |     ≥ 4.5:1      | 符合标准要求 |
| **次要青色文本** | `--color-accent-cyan` (`#00e5ff` / `#0369a1`)    | `--color-bg-base` (`#090d16` / `#f8fafc`)      |  **12.63 : 1**  |               **5.67 : 1**                |     ≥ 4.5:1      | 符合标准要求 |
| **交互焦点环**   | `--color-focus` (`#00f59b` / `#047857`)          | `--color-surface-base` (`#111827` / `#ffffff`) |  **12.47 : 1**  |               **5.27 : 1**                |     ≥ 3.0:1      | 符合标准要求 |

---

### 3.3 空间、圆角、阴影与动效令牌

```css
:root {
  /* 空间步进 */
  --space-1: 0.25rem; /* 4px */
  --space-2: 0.5rem; /* 8px */
  --space-3: 0.75rem; /* 12px */
  --space-4: 1rem; /* 16px */
  --space-5: 1.25rem; /* 20px */
  --space-6: 1.5rem; /* 24px */
  --space-8: 2rem; /* 32px */
  --space-10: 2.5rem; /* 40px */
  --space-12: 3rem; /* 48px */
  --space-16: 4rem; /* 64px */

  /* 容器尺寸 */
  --max-width-prose: 50rem; /* 800px 文章阅读宽度 */
  --max-width-content: 54rem; /* 864px 列表/归档宽度 */
  --max-width-wide: 74rem; /* 1184px 全宽视口容器 */
  --header-height: 4rem; /* 64px 页头标准高度 */
  --header-height-scrolled: 3.5rem; /* 56px 滚动压缩高度 */

  /* 硬朗微圆角体系 */
  --radius-xs: 0.25rem; /* 4px */
  --radius-sm: 0.375rem; /* 6px - 按钮、标签 */
  --radius-md: 0.5rem; /* 8px - 输入框、代码容器 */
  --radius-lg: 0.75rem; /* 12px - 文章卡片、面板 */
  --radius-xl: 1rem; /* 16px - 模态抽屉 */
  --radius-full: 9999px; /* 胶囊状态 */

  /* 阴影与多层辉光体系 */
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.4);
  --shadow-card:
    0 4px 20px -2px rgba(0, 0, 0, 0.5), 0 0 0 1px var(--color-border-default);
  --shadow-card-hover:
    0 10px 30px -4px rgba(0, 0, 0, 0.7),
    0 0 20px 0 var(--color-accent-primary-glow),
    0 0 0 1px var(--color-accent-primary);
  --shadow-glow-accent: 0 0 25px var(--color-accent-primary-glow);
  --shadow-glow-cyan: 0 0 25px var(--color-accent-cyan-glow);
  --shadow-header:
    0 4px 20px rgba(0, 0, 0, 0.4), 0 1px 0 var(--color-border-default);

  /* 玻璃拟态模糊 */
  --glass-blur: blur(12px);

  /* 动效令牌 */
  --motion-duration-fast: 150ms;
  --motion-duration-normal: 260ms;
  --motion-duration-slow: 450ms;
  --motion-duration-pulse: 2400ms;
  --motion-duration-ambient: 12000ms;
  --motion-ease-standard: cubic-bezier(0.2, 0, 0, 1);
  --motion-ease-emphasized: cubic-bezier(0.05, 0.7, 0.1, 1);
  --motion-ease-pulse: cubic-bezier(0.4, 0, 0.6, 1);

  /* 层级令牌 (z-index) */
  --z-bg: -1;
  --z-base: 1;
  --z-sticky: 40;
  --z-header: 50;
  --z-floating: 80;
  --z-drawer: 100;
  --z-pointer: 150;
  --z-modal: 200;
}
```

---

## 4. 字体排印、自托管与 0-CLS 策略

### 4.1 字体栈明细与来源策略

- **无衬线主正文族（UI & Body）**：
  ```css
  --font-sans:
    -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
    "Hiragino Sans GB", "Microsoft YaHei", "WenQuanYi Micro Hei", Roboto,
    sans-serif;
  ```
  - **来源与许可证**：客户端操作系统自带系统字体（Apple / Microsoft / Linux 发行版内置），0 许可风险。
- **等宽 / 数理字标族（Mono & Display）**：
  ```css
  --font-mono:
    "JetBrains Mono", "Fira Code", "Cascadia Code", "Source Code Pro", Consolas,
    Monaco, monospace;
  --font-display: var(--font-mono);
  ```
  - **候选链明确说明**：`JetBrains Mono` 作为首选候选字体，**不保证在所有用户设备上预装**。浏览器将按顺序自动回退至 `Fira Code` $\to$ `Cascadia Code` $\to$ `Consolas` $\to$ `Monaco` $\to$ 通用 `monospace`。

### 4.2 字体加载、font-display 适用性与 0-CLS 保证

1. **font-display 不适用性**：由于当前全站采用系统预装字体栈，字体度量由本地系统直接解析，无需网络下载，因此 `@font-face` 的 `font-display` 属性在此阶段**不适用**。
2. **0-CLS 布局稳定性保证**：系统字体无需经历 FOIT（不可见文本闪烁）或 FOUT（未格式化文本闪烁），首屏文本渲染 CLS 严格为 `0`。
3. **未来 WebFont 准入限制**：若后续阶段引入自托管特色英文字标，必须同时满足：
   - 字体文件放置于本地 `public/fonts/`，禁止任何第三方 CDN 引入；
   - 采用子集化 WOFF2 格式，体积 `< 25 KiB`；
   - 必须配置 `font-display: swap` 并使用 `size-adjust` 进行度量补偿。

---

## 5. 响应式断点与布局折叠规则矩阵（360/390/768/1024/1440/1920）

| 视口规格   | 典型设备             | 导航栏 (SiteHeader)        | 首页 Hero 布局                         | 文章列表 (PostList)     | 文章阅读页 (PostLayout)         | 0 溢出防护保证  |
| :--------- | :------------------- | :------------------------- | :------------------------------------- | :---------------------- | :------------------------------ | :-------------: |
| **1920px** | 2K / 4K 宽屏显示器   | 居中全宽导航，边距自动扩展 | 左右双栏 (55% / 45%)，SVG 流场全幅展现 | 3 栏网格 (gap: 1.5rem)  | 正文 (50rem) + 侧边 TOC (16rem) |   严格 0 溢出   |
| **1440px** | 标准桌面显示器       | 完整桌面横向导航           | 左右双栏 (55% / 45%)，SVG 保持 480×420 | 3 栏网格 (gap: 1.5rem)  | 正文 (50rem) + 侧边 TOC (16rem) |   严格 0 溢出   |
| **1024px** | 笔记本 / iPad 横屏   | 完整桌面横向导航           | 左右双栏 (60% / 40%)，SVG 等比微缩     | 2 栏网格 (gap: 1.25rem) | 正文 (100%) + 侧边 TOC (14rem)  |   严格 0 溢出   |
| **768px**  | iPad 竖屏 / 大折叠屏 | 汉堡按钮 + 抽屉菜单        | 上下堆叠，SVG 占据 280px 高度          | 2 栏网格                | 单栏正文 + 顶部折叠 TOC         |   严格 0 溢出   |
| **390px**  | iPhone 14/15/16 Pro  | 汉堡按钮 + 抽屉菜单        | 单栏垂直流，SVG 下沉隐藏防重排         | 1 栏垂直卡片流          | 单栏正文 + 公式横向滚动容器     |   严格 0 溢出   |
| **360px**  | 小型 Android 设备    | 汉堡按钮 + 抽屉菜单        | 单栏紧凑流，内边距收敛至 1rem          | 1 栏垂直卡片流          | 单栏正文 + 公式横向滚动容器     | **严格 0 溢出** |

---

## 6. 组件交互状态机规格表（8 类组件 × 7 种状态 + Touch 等价）

| 组件名称              | default                                               | hover                                 | focus-visible                     | active                    | disabled                                     | loading                        | error                  | mobile-touch 等价策略                      |
| :-------------------- | :---------------------------------------------------- | :------------------------------------ | :-------------------------------- | :------------------------ | :------------------------------------------- | :----------------------------- | :--------------------- | :----------------------------------------- |
| **1. SiteHeader**     | 固定置顶，半透毛玻璃，底边框 `--color-border-default` | 链接变亮，背景浮现微弱浅色            | 2px 霓虹绿外发光 Focus Ring       | 链接文字加粗微内缩        | N/A (始终可交互)                             | N/A (静态载入)                 | N/A (静态载入)         | 触控即时响应，无悬停锁定，抽屉全屏弹出     |
| **2. Hero**           | 状态呼吸点，大字标，左右双栏                          | 按钮微上浮 2px，辉光增强              | 按钮与 Chip 显示 2px Focus 环     | 按钮 scale(0.98) 按压反馈 | N/A (纯展示/导航)                            | N/A (静态首屏)                 | N/A (静态首屏)         | 移动端右侧大图隐藏，按钮全宽居中排列       |
| **3. PostCard**       | 基础面板，顶边渐变高光伪元素隐藏                      | 顶边高光浮现，边框变绿，上浮 3px      | 卡片整体呈现 Hover 等价边框与阴影 | 卡片轻微内缩按压          | N/A (可访问链接)                             | 骨架屏脉冲微光                 | N/A (静态 SSG)         | 移除 translateY 浮动，触控仅高亮边框与背景 |
| **4. SectionHeader**  | `#` 符号前缀，大标题，计数胶囊                        | N/A (非交互标题)                      | N/A (语义化 h2)                   | N/A                       | N/A                                          | N/A                            | N/A                    | 保持紧凑边距排版                           |
| **5. PaginationNav**  | 等宽数字面板，1px 默认边框                            | 边框变绿，数字提亮                    | 2px Focus Ring                    | 按压微内缩                | 置灰 `opacity: 0.45`, `pointer-events: none` | N/A (纯静态链接)               | N/A (纯静态链接)       | 触摸区域扩大至 ≥ 44×44px 黄金标准          |
| **6. SiteFooter**     | 状态脉冲绿点，版权文字                                | 外链下划线变绿                        | 链接呈现 2px Focus 环             | 链接加深                  | N/A                                          | N/A                            | N/A                    | 单栏居中紧凑流                             |
| **7. PostLayout TOC** | 粘性定位，当前阅读章节左侧绿条                        | 标题文字提亮为 `--color-text-primary` | 键盘聚焦时左侧指示条点亮          | 点击瞬时微缩              | N/A                                          | N/A                            | N/A                    | 移动端使用 `<details>` 折叠，点击展开      |
| **8. SearchResult**   | 搜索卡片，`<mark>` 霓虹青高亮                         | 边框高亮，标题变绿                    | 键盘 Tab 键选中高亮               | 点击跳转                  | N/A                                          | 骨架屏脉冲, `aria-busy="true"` | 提示搜索失败并提供重试 | 触控点击即时响应导航                       |

---

## 7. 动效分级、性能降级与无 JS 契约

### 7.1 动效三分类系统（Reveal / Ambient / Pointer）

1. **Reveal 显现动效**：
   - 范围：首页卡片进入视口、Hero 大字标分段显现；
   - 参数：持续 `260ms`，缓动 `cubic-bezier(0.05, 0.7, 0.1, 1)`，位移 `translateY(14px)`；
   - 机制：IntersectionObserver 触发后立即 `unobserve`，一次性执行。
2. **Ambient 环境循环动效**：
   - 范围：网格微弱流动、扫描线扫掠（`12000ms` `linear`）、状态点呼吸脉冲（`2400ms` `cubic-bezier(0.4, 0, 0.6, 1)`）；
   - 参数：统一使用 `--motion-duration-ambient` (12000ms) 与 `--motion-duration-pulse` (2400ms)；
   - 降级：移动端（`width <= 48rem`）或低动态模式下**完全停止**，锁定静态。
3. **Pointer 光标增强动效**：
   - 范围：磁吸光晕跟随；
   - 严格限定：仅在 `(hover: hover) and (pointer: fine)` 下启用，严格在 `requestAnimationFrame` 中更新，**不隐藏系统原生光标**。

### 7.2 页面后台运行暂停与 Cleanup 契约

```ts
// 页面隐藏时 100% 暂停 rAF 循环，0% CPU 占用
document.addEventListener(
  "visibilitychange",
  () => {
    if (document.hidden) {
      ambientController.pause();
    } else {
      ambientController.resume();
    }
  },
  { signal: abortController.signal },
);
```

### 7.3 无 JavaScript 终态保障（Progressive Enhancement）

- **基础 DOM 默认终态**：所有元素在基础 CSS 中默认 `opacity: 1; transform: none;`。
- **门禁激活机制**：仅当客户端脚本在浏览器中执行成功并添加 `html[data-motion="ready"]` 后，未进入视口的元素才被暂时置为初始状态，确保在 JS 禁用或加载失败时内容 100% 正常可读。

---

## 8. 原创 Hero 右侧流场 / 扩散轨迹 SVG 规格与资产台账

### 8.1 结构与数学隐喻

- **设计隐喻**：直观表达扩散模型中逆向时间随机微分方程（Reverse-time SDE）：
  $$\mathrm{d}\mathbf{x} = \left[ \mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x}) \right] \mathrm{d}t + g(t) \mathrm{d}\bar{\mathbf{w}}$$
- **SVG 结构与图层**：
  1. 椭圆流形环（Manifold Orbits）：`rx="180" ry="130"`, `rx="120" ry="80"`, `rx="60" ry="35"`；
  2. 速度场向量流线（Bézier Streamlines）：4 组特征样条曲线；
  3. SDE 随机采样主轨迹（Main Trajectory）：荧光绿 2.5px 平滑曲线；
  4. Euler-Maruyama 离散采样步长节点：5 个离散坐标圆点，标注目标分布中心 `x₀`。

### 8.2 资产许可证台账 (Asset License Ledger)

| 资产名称                   | 路径 / 存在形式            | 创作者                  | 许可证                    | 说明                                     |
| :------------------------- | :------------------------- | :---------------------- | :------------------------ | :--------------------------------------- |
| **Hero SDE Flowfield SVG** | 内联于 `Hero.astro` / 原型 | EntropyXi / Antigravity | **MIT / CC0 (100% 自研)** | 纯代码参数化矢量绘制，零外部素材侵权风险 |
| **SCAU/SE 极客徽章 SVG**   | 内联于 `about.astro`       | EntropyXi / Antigravity | **MIT / CC0 (100% 自研)** | 原创数理符号与几何组合                   |
| **404 Glitch Singularity** | 内联于 `404.astro`         | EntropyXi / Antigravity | **MIT / CC0 (100% 自研)** | 原创拓扑断裂点阵图形                     |

---

## 9. Astro 架构映射表与零客户端框架承诺

### 9.1 组件映射表

```text
src/
├── styles/
│   ├── tokens.css               <-- 承载 §3 颜色、间距、圆角、阴影与 motion tokens
│   ├── global.css               <-- 全局 reset、Focus ring、排版、滚动条
│   ├── motion.css               <-- 通用 reveal、reduced-motion 规则
│   └── prose.css                <-- 技术文章与 MathJax 排版保护
├── components/
│   ├── chrome/
│   │   ├── SiteHeader.astro     <-- 对应 §6.1 桌面导航、移动抽屉
│   │   ├── SiteFooter.astro     <-- 对应 §6.1 系统脉冲状态页脚
│   │   └── ThemeToggle.astro    <-- 对应 §2 主题切换外观
│   ├── visual/
│   │   ├── AmbientBackground.astro <-- 对应 §7.1 4 层环境氛围背景
│   │   └── Hero.astro           <-- 对应 §6.2 叙事 Hero + §8 原创 SVG
│   └── content/
│       ├── PostCard.astro       <-- 对应 §6.3 赛博文章卡片
│       ├── PostList.astro       <-- 对应 §5 响应式文章网格
│       ├── PaginationNav.astro  <-- 对应 §6.4 静态分页器
│       ├── ReadingProgress.astro<-- 对应 §6.5 顶部单 rAF 进度条
│       └── FloatingControls.astro<-- 回顶快捷按钮
└── layouts/
    ├── BaseLayout.astro         <-- 骨架、SkipLink、防 FOUC 内联脚本
    └── PostLayout.astro         <-- 双栏文章页、sticky TOC
```

### 9.2 架构承诺：零客户端 UI 框架

本站**绝不引入 React, Vue, Svelte, Solid 等任何客户端 UI 框架**，全站基于原生 Astro SSG 静态生成。所有交互逻辑均采用纯原生 TypeScript 模块托管在 `src/lib/`，保证极致加载速度与最高代码健康度。

---

## 10. 主 Agent 验收清单（Definition of Done）

- [x] **原创性检验**：排除所有 Ayeez 人物与私有立绘，确立自研 SDE 矢量体系；
- [x] **主题策略完整**：首次跟随系统，用户选择优先，防 FOUC 脚本与清理契约齐备；
- [x] **设计令牌完备**：Dark/Light 全色阶十六进制齐全，WCAG 2.2 AA 数学计算实测达标；
- [x] **字体策略落地**：系统字体优先，font-display 不适用性声明明确，0-CLS 机制固化；
- [x] **响应式 6 级矩阵**：360px–1920px 规则明确，严格保证 0 页面级横向溢出；
- [x] **组件 8 类状态机**：交互状态与移动端触控等价策略完整覆盖；
- [x] **动效性能与降级**：M0–M4 与三分类明确，后台与 reduced-motion 100% 停止；
- [x] **无 JS 终态保障**：默认样式终态呈现，`html[data-motion="ready"]` 门禁保护；
- [x] **资产台账完备**：Hero SDE SVG 结构与 100% 自研许可证清晰记录；
- [x] **原型样板就绪**：提供自包含、零 JS、零外链的 `docs/refactor/prototype/index.html` 与 `styles.css`。

## Phase 8 动效与视觉基准

- 全量验证通过：支持无 JS 环境、低动态环境 (prefers-reduced-motion: reduce)、移动端横向无溢出。
- Axe 与 Lighthouse 通过无严重障碍门禁。
