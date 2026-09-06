# 重构自动质量门禁

## 适用范围

本文件冻结阶段 2 起的自动门禁。门禁约束 Astro SSG 产物和渐进增强客户端；不得通过降低阈值、跳过页面或隐藏内容来换取通过。

## 浏览器矩阵

| Playwright 项目       | 环境                             | 固定断言                           |
| --------------------- | -------------------------------- | ---------------------------------- |
| `desktop-chromium`    | Desktop Chrome                   | 全量 E2E、行为与 axe               |
| `mobile-390`          | 390×844、touch                   | 无横向溢出、抽屉焦点               |
| `mobile-360`          | 360×800、touch                   | 无横向溢出、抽屉焦点               |
| `mobile-320`          | 320×568、touch                   | 无横向溢出、抽屉焦点、卡片滚动可见 |
| `reduced-motion`      | `prefers-reduced-motion: reduce` | 根运动偏好为 reduced、内容可见     |
| `javascript-disabled` | JavaScript disabled              | Hero、卡片与文章静态内容可见       |
| `zoom-200`            | Desktop Chrome、页面 200% 缩放   | 核心内容可见且不产生页面级横向溢出 |
| `mobile-safari`       | Playwright WebKit（iOS 替代）    | 客户端基础环境契约                 |

iOS Safari 由主 Agent 在阶段 3、6、8 分别抽查一次；最低页面集为首页、移动菜单、搜索和一篇复杂公式文章。无真实设备时，以 Playwright WebKit（`mobile-safari` 项目）记录替代结果，并在最终验收中明确标注"WebKit 替代，非真实 iOS 设备"。当前阶段 8 即采用此方案。动效栈（ADR 0003）补充：`client-matrix.spec.ts` 的门控用例在全部 8 个项目上断言 `data-gsap-active` / `data-lenis-active` / `.lenis` 标记与环境一致（触屏与 reduced-motion 无 Lenis、reduced-motion 无 GSAP inline transform、zoom-200 禁用 Lenis），并用 view-transitions.spec 验证持久化壁纸的节点 identity。抽屉在移动端另有绘制契约断言（遮罩覆盖全视口、面板不透明）。桌面项目另有 `mobile-overflow.spec.ts`：34 页样本 × 360/320 视口断言页面级零横向溢出（分类页 2026-09-06 溢出修复的回归门禁）。

## 可访问性

- 首页、归档、搜索和 DDIM 文章的 axe `serious` / `critical` 必须为 0；
- 键盘测试必须覆盖移动菜单打开、焦点圈定、Escape 关闭与焦点返回；
- 站点自 2026-08-23 起永久锁定暗色主题（`data-theme="dark"`），不得引入主题切换按钮
  （E2E 断言 `.theme-toggle-btn` 数量为 0）；
- 无 JavaScript 时，首页、文章正文与构建期公式不得隐藏；
- `scripts/audit-content.ts` 在源码层检测 display 定界符 `$$` 直接邻接 CJK 正文（正文被公式吞入）与空公式 `$$$$`，防止不可访问的空公式与正文混用。

## 资源预算

- `dist/_astro/*.js` 第一方客户端脚本 gzip 总量上限：80 KiB（ADR 0003，原 24 KiB，为 GSAP + Lenis 动效栈上调）；
- 任一第一方客户端脚本 gzip 上限：32 KiB（原 8 KiB；分 chunk 后最大单文件为 gsap core ≈28 KiB）；
- Pagefind 搜索索引和构建期 MathJax SVG 不计入首屏 JavaScript，但继续由产物审计单独检查；
- 原「新增首屏 JavaScript ≤15 KiB gzip」子目标自 ADR 0003 起废除，由 80 KiB 总预算与 Lighthouse TBT / Long Task 门禁替代；
- GSAP/Lenis 必须经能力门控动态加载：reduced-motion 与无 JS 环境不得产生对应资源请求（E2E 以 `data-gsap-active` / `data-lenis-active` 标记与 resource entries 断言）。

`npm run audit:bundle` 在构建后生成实际 raw/gzip 总数并以非零状态阻断超限。

## 视觉与性能

- Lighthouse 固定页面四分类定义：
  - **home**：`/`，首页，无公式；
  - **search**：`/search/`，Pagefind 搜索页，无公式；
  - **article-normal**：无 MathJax 公式的普通正文文章（`math: false` 或无 `math` 字段）。当前选用：`/2026/08/05/技术随笔/WSL2安装失败排查：从DISM_COM损坏到NetCfg残留/`；
  - **article-ddim**：包含 MathJax 公式的复杂数学文章，代表性选取 DDIM：`/2026/05/10/深度学习/流匹配与扩散模型/DDIM/`。
- 四分类均 ≥90，Accessibility ≥95；
- CLS 目标 ≤0.1，本站脚本不得制造重复的 >50 ms Long Task；
- 主要视觉阶段保留 1440×900、390×844 的暗色证据，以及 reduced-motion、键盘焦点和无脚本终态
  （亮色证据随 2026-08-23 永久暗色化停止采集，历史 light 截图仅作阶段证据保留）；
- 高频动画只使用 `transform` / `opacity`。布局或绘制属性动画必须在审查记录中解释。

## 公式计数演变说明

| 阶段          | 计数      | 原因                                                                                       |
| ------------- | --------- | ------------------------------------------------------------------------------------------ |
| 阶段 0.5      | 1289/1289 | 初始基准                                                                                   |
| 阶段 8 前次审 | 1288/1288 | 一处公式定界符修复导致计数变化（具体提交未留说明）                                         |
| 阶段 8 本轮   | 1291/1291 | B-M1 修复：共轭梯度法文章第 141 行 `$$正文` 混用拆分为独立块 + 正文                        |
| 2026-08-23    | 1351/1351 | 数学格式化修复（提交 `47b5281`、`774ace9`）：统一行内/块级定界符与防断行拆分，改变捕获计数 |

计数变化均属格式修正，无公式内容删减，可接受。

## CI 运行策略

单个 Ubuntu job 顺序执行格式、静态检查、类型、内容审计、单测、Playwright 全矩阵、bundle 与产物审计。当前矩阵规模无需分片；若 CI 连续三次超过 20 分钟，再按 desktop/full 与 environment-matrix 两组分片，阈值不变。

阶段 2 本地实测：29 项 Playwright 全矩阵复用预览时耗时 6.3 秒，从干净状态自行构建和启动预览时耗时 14.2 秒，完整质量流水线约 35 秒；CI 目标冻结为 20 分钟以内，当前不分片。`scripts/run-preview.mjs` 强制测试预览保持前台并在退出时回收，避免 AI 终端自动后台化影响本地/CI 一致性。

CI 真实耗时（GitHub Actions 运行记录回填，2026-08-22 至 2026-09-04）：

- Deploy to GitHub Pages（完整 check + Pages 发布）：成功运行 1m55s – 2m19s；
- Quality（PR 门禁，同一 `npm run check`）：成功运行 2m06s – 3m51s。

均远低于 20 分钟分片阈值，维持单 job 不分片。

2026-09-06 增补（`mobile-overflow.spec.ts` 新增 68 个用例、desktop-chromium
测试数 37 → 105 后）：Deploy 成功运行 2m12s – 2m46s，增幅约 +20–30s，
与预估一致；维持单 job 不分片。
