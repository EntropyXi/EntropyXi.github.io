# 重构自动质量门禁

## 适用范围

本文件冻结阶段 2 起的自动门禁。门禁约束 Astro SSG 产物和渐进增强客户端；不得通过降低阈值、跳过页面或隐藏内容来换取通过。

## 浏览器矩阵

| Playwright 项目       | 环境                             | 固定断言                           |
| --------------------- | -------------------------------- | ---------------------------------- |
| `desktop-chromium`    | Desktop Chrome                   | 全量 E2E、行为与 axe               |
| `mobile-390`          | 390×844、touch                   | 无横向溢出、抽屉焦点               |
| `mobile-360`          | 360×800、touch                   | 无横向溢出、抽屉焦点               |
| `reduced-motion`      | `prefers-reduced-motion: reduce` | 根运动偏好为 reduced、内容可见     |
| `javascript-disabled` | JavaScript disabled              | Hero、卡片与文章静态内容可见       |
| `zoom-200`            | Desktop Chrome、页面 200% 缩放   | 核心内容可见且不产生页面级横向溢出 |

iOS Safari 由主 Agent在阶段 3、6、8 分别抽查一次；最低页面集为首页、移动菜单、搜索和一篇复杂公式文章。若没有真实设备或远程设备能力，用 Playwright WebKit 记录替代结果并在最终验收中明确标注。

## 可访问性

- 首页、归档、搜索和 DDIM 文章的 axe `serious` / `critical` 必须为 0；
- 键盘测试必须覆盖移动菜单打开、焦点圈定、Escape 关闭与焦点返回；
- 主题按钮不得产生重复 ID，桌面和移动可见实例都能操作；
- 无 JavaScript 时，首页、文章正文与构建期公式不得隐藏。

## 资源预算

- `dist/_astro/*.js` 第一方客户端脚本 gzip 总量上限：24 KiB；
- 任一第一方客户端脚本 gzip 上限：8 KiB；
- Pagefind 搜索索引和构建期 MathJax SVG 不计入首屏 JavaScript，但继续由产物审计单独检查；
- 相对阶段 0 的新增首屏 JavaScript仍以 ≤15 KiB gzip 为目标。触顶时必须先拆分按页加载；超过需要 ADR。

`npm run audit:bundle` 在构建后生成实际 raw/gzip 总数并以非零状态阻断超限。

## 视觉与性能

- Lighthouse 固定页面：首页、搜索、普通文章、DDIM；四分类均 ≥90，Accessibility ≥95；
- CLS 目标 ≤0.1，本站脚本不得制造重复的 >50 ms Long Task；
- 主要视觉阶段保留 1440×900、390×844 的暗/亮证据，以及 reduced-motion、键盘焦点和无脚本终态；
- 高频动画只使用 `transform` / `opacity`。布局或绘制属性动画必须在审查记录中解释。

## CI 运行策略

单个 Ubuntu job 顺序执行格式、静态检查、类型、内容审计、单测、Playwright 全矩阵、bundle 与产物审计。当前矩阵规模无需分片；若 CI 连续三次超过 20 分钟，再按 desktop/full 与 environment-matrix 两组分片，阈值不变。

阶段 2 本地实测：29 项 Playwright 全矩阵复用预览时耗时 6.3 秒，从干净状态自行构建和启动预览时耗时 14.2 秒，完整质量流水线约 35 秒；CI 目标冻结为 20 分钟以内，当前不分片。`scripts/run-preview.mjs` 强制测试预览保持前台并在退出时回收，避免 AI 终端自动后台化影响本地/CI 一致性。CI 的真实耗时在阶段 8 GitHub Actions 终验时回填。
