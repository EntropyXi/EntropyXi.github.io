# 博客导航栏与首屏排版精细化调整方案 (Hero & Navbar Refinement Plan)

**日期**：2026-08-23  
**状态**：计划就绪（按指令无需审查，直接执行）  
**执行环境约束**：所有 CLI 命令严格使用 Git Bash (`bash -c "..."`)，禁止使用 pwsh。

---

## 1. 需求与视觉对照清单

| 序号      | 需求项目                              | 用户参考截图                                                         | 现状分析                                                                                    | 目标设计与实现规范                                                                                                                                                                                                                                                                                |
| --------- | ------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **REQ-1** | 导航栏左侧 Brand 靠左移与下方字体平行 | `屏幕截图 2026-08-23 002312.png`                                     | 导航栏内边距为 `1.5rem`，与 Hero 文本的 `clamp(1.5rem, 5.5vw, 5.5rem)` 存在偏移，未共线对齐 | 将 `SiteHeader.astro` 的 `.site-header-inner` 水平 padding 调整为 `0 clamp(1.5rem, 5.5vw, 5.5rem)`，使导航栏 Logo 品牌文字与下方首屏标题及叙述文本严格在同一垂直参考线上共线平行                                                                                                                  |
| **REQ-2** | 首屏字体、结构、距离参考目标效果      | `屏幕截图 2026-08-23 002337.png`<br>`屏幕截图 2026-08-23 002433.png` | 标题与下方文本距离过近（`0.5rem`），正文字号偏大偏厚重，缺少呼吸感与科技装饰线              | 1. 扩大标题与说明文本的纵向安全间距（`margin-top: clamp(3rem, 6vh, 4.5rem)`）；<br>2. 正文字号精炼为 `clamp(0.92rem, 1.1vw, 1.02rem)`，行高放宽至 `2.0`，字间距 `0.05em`；<br>3. 文本下方增加科技感渐变边线（`narrative-tech-bracket`）；<br>4. 整体结构与截图 002433 的留白比例达到 1:1 精确复刻 |
| **REQ-3** | 导航栏与 Favicon 图标换成指定头像图片 | `屏幕截图 2026-08-23 002537.png`<br>`屏幕截图 2026-08-23 002525.png` | 当前为 Next.js 黑底六边形 "N" 图标                                                          | 提取并裁剪高清水仙头像，生成 `avatar-suisen.webp`、`favicon-32x32-next.png`、`favicon-16x16-next.png`、`apple-touch-icon-next.png` 及 `favicon.ico`，在导航栏 Brand 左侧使用圆形微发光水仙头像，同步更新浏览器标签页 Favicon                                                                      |

---

## 2. 详细执行计划

### 步骤 1：处理并部署水仙头像图标资产（Favicon & Navbar Brand Icon）

1. 编写 Python 图像处理脚本，以 `"C:\Users\entropy\Pictures\Screenshots\屏幕截图 2026-08-23 002525.png"` 为源，中心裁切正方形，输出：
   - `astro-public/images/avatar-suisen.webp` (128x128 高清 WebP)
   - `astro-public/images/favicon-32x32-next.png` (32x32 PNG)
   - `astro-public/images/favicon-16x16-next.png` (16x16 PNG)
   - `astro-public/images/apple-touch-icon-next.png` (180x180 PNG)
   - `astro-public/favicon.ico` (32x32 ICO)

### 步骤 2：重构 `src/components/chrome/SiteHeader.astro`

1. 将 `.site-logo-symbol` 替换为带有微金边和轻微阴影的圆形水仙头像 `<img>`：
   ```astro
   <img
     src="/images/avatar-suisen.webp"
     alt="EntropyXi Logo"
     class="site-logo-avatar"
     width="28"
     height="28"
     loading="eager"
     decoding="async"
   />
   ```
2. 调整 `.site-header-inner` 的 padding，使其与 Hero 容器完全对齐：
   ```css
   .site-header-inner {
     width: 100%;
     max-width: var(--max-width-wide);
     margin: 0 auto;
     padding: 0 clamp(1.5rem, 5.5vw, 5.5rem);
   }
   ```

### 步骤 3：精细化重构 `src/components/visual/Hero.astro`

1. 调整结构间距：
   - `.hero-narrative-block` 增加 `margin-top: clamp(2.8rem, 5.5vh, 4.2rem);`
   - 文本行高调整为 `2.0`，字号调整为 `clamp(0.92rem, 1.1vw, 1.02rem)`，字间距 `0.04em`
   - 在叙述文本底部添加夕阳金科技装饰线（`.narrative-tech-bracket`）
2. 保持标题紧凑窄长大字体（`Impact, Arial Narrow, Haettenschweiler, DIN Alternate, Roboto Condensed`）与两行不折行契约。

### 步骤 4：更新与补充自动化测试套件

1. `tests/e2e/hero-contrast.spec.ts`：更新对叙述文字、排版间距及指示器的断言；
2. `tests/e2e/client-behavior.spec.ts`：验证导航栏 Logo 头像正常渲染；
3. 执行 `npm run check` 9 步全量门禁流水线。

### 步骤 5：Git 提交与远程分支同步推送

1. 在 `main` 分支完成规范提交并合并至 `source`；
2. 推送至远程仓库触发 GitHub Actions 部署。
