# 移动端内容溢出修复计划（分类页布局 / 行内公式缩放 / 显示公式移动端字号 / 全站溢出门禁）R2 修订版

- 状态：双高精度审查完成（R1 渲染/CSS：1 P0 / 3 P1 / 4 P2；R2 测试门禁/CI：
  1 P0 / 2 P1 / 5 P2；两审查独立命中同一 P0），14 条意见全部闭合（§7），可执行
- 日期：2026-09-06
- 证据：`audit-screenshots/mobile-audit/full-overflow-report.json`（30 篇文章 +
  4 个聚合页 × 360/320 全量扫描，54 个组合存在超宽元素，其中 2 个为页面级溢出，
  均为 `/categories/`）

## 1. 背景与实测数据（R1/R2 修订后口径）

全站扫描（Playwright 触屏仿真，元素级测量；报告为每页截断 10 条的最宽
offender 抽样，含站点 chrome）：

- **页面级横向溢出（布局破坏）仅 1 处**：`/categories/` 在 360 与 320 视口下
  `scrollWidth=385 > 视口`。
  **根因（v1 判断有误，双审查独立证伪）**：`≤40rem` 视口下网格早已被
  `@media (width <= 40rem) { .category-grid { grid-template-columns: 1fr } }`
  覆盖为单列（该规则自 2026-08-22 `cffff5c` 存在），`minmax(20rem, 1fr)` 在
  窄屏从不参与。真实机制：`1fr` 等价 `minmax(auto, 1fr)`，轨道 auto 最小值
  被**最宽卡片的 min-content** 撑大——最长分类名
  「深度学习/流匹配与扩散模型/超分辨率」（nowrap ≈256px）+ 卡片 padding 40 +
  图标 16 + 间隙 ≈21.6 + count 徽标 49（flex-shrink:0）≈ 369px → 整条轨道
  369px，同轨道所有卡片（align stretch）均 369px，页宽 385 = 16 padding + 369
  - 余量。v1 的 `minmax(min(20rem,100%),1fr)` 方案被证明是 no-op（媒体查询
    整体覆盖）且混淆了"轨道最小值"与"条目最小尺寸"两个机制。
- **容器内裁切（体感"看不到"）**：全站 dist 测算，320 视口下**超宽显示公式
  229 条**（最宽 93.63ex ≈ 732px，可用宽仅 254px；ex=8.5px 由 82.956ex→705px
  实测锚定）。MathJax SVG 固定自然宽度、永不换行，由 `mjx-container` 的
  `overflow: auto hidden` 容忍（页面级不溢出），真机上嵌套横滑难发现难操作，
  行内公式尤甚。
- 代码块（最长 591px）与表格（710px）已有 `overflow: auto` /
  `display: block; overflow-x: auto` 容器（prose.css:218-222、275-283），属
  标准横滑，本次不改。
- E2E 矩阵仅覆盖 5 篇数学文章 + 首页/归档/搜索，以上问题全部落在断言盲区。

## 2. 修复项

- REQ-1（按双审查修正）：分类页单列档 `1fr` → `minmax(0, 1fr)`，并给
  `.category-atlas-card` 加 `min-width: 0`——解除轨道/条目最小尺寸劫持，让
  既有 `.category-atlas-left`（overflow:hidden）与 `.category-atlas-name`
  （ellipsis）裁剪链生效；`min-width: 0` 同时保护 >40rem 双列区间
  （641–736px 视口）的 369px 卡片溢出重叠。
- REQ-2：行内公式等比缩放——`mjx-container:not([display="true"]) > svg`
  （global.css:174-178）的 `max-width: none` 改为 `max-width: 100% +
height: auto`（SVG 带 viewBox，实测确认等比缩放成立）。定位：修复全站唯一
  真实行内溢出实例（`体系/1. 从SDE开始`，39.99ex）+ 作为防御性不变量；
  缩放后 `scrollWidth == clientWidth`，`math-scroll-hint` 的 `≤1px` 容差
  吸收取整误差、自动不再打标（已核实，无需改 JS）。
- REQ-3：窄屏显示公式字号降档——`global.css` mjx 段（全部 mjx 规则集中处）
  增加 `@media (width <= 30rem) { .prose mjx-container[display="true"] {
font-size: 0.92em } }`。量化收益（dist 全量测算）：320 视口溢出门槛
  29.88ex → 32.48ex，回收 (29.88, 32.48]ex 区间 **32 条**（@360 回收 25 条）；
  **197 条尾部公式（>32.48ex，最宽 93.63ex）仍超宽**，继续由容器横滑 +
  scroll-hint 兜底——0.92em 是"减少横滑次数"的增量优化而非根治，此为确定
  结论而非风险。副作用留档：391–480px 区间原本恰好适排的公式（如 38ex≈323px
  @390）也会被缩小。
- REQ-4：全站溢出门禁 spec `tests/e2e/mobile-overflow.spec.ts`：
  - 覆盖 **34 页样本集**（baseline 30 篇 article + about/categories/tags/
    archives；不含 16 个 tags/_、14 个 categories/_ 详情与 16 个月度归档页，
    措辞为样本集而非"全站"）；
  - **68 个参数化 case**（34 页 × 360×800 / 320×568 两视口），`test.use`
    显式 `{ hasTouch: true, isMobile: true }`；断言
    `expect.poll(() => scrollWidth <= clientWidth + 1)`（对齐 client-matrix
    惯例，消除定值等待）；失败信息附带 offender 明细（`getAttribute("class")`
    取类名、过滤 `position: fixed` 装饰层、无条数上限）；
  - 归属 desktop-chromium（该项目无 testMatch 限制；其余 8 项目均限定
    client-matrix，已核实不会误捕获），desktop 测试数 37 → 105；
  - 容差口径与 `article-layout.spec.ts:30`（无容差）并存：新 spec 用
    `+1` 并注明差异。
- REQ-5：不改变既有契约——构建期 MathJax 管线（ADR 0001）、公式可访问名
  （aria-label 在 svg 上，`math-accessibility.ts:94`；无 mjx-assistive）、
  `article-layout.spec` svg 计数与 computed display 断言、九步门禁链
  （新 spec 落在既有 test:e2e 步骤内）全部保持；motion.css 与字号降档无
  级联交互（已核实 motion.css:249-284 仅 duration/transform）。

## 3. 实施位置

| 修复  | 文件                                                             | 改动                                                                        |
| ----- | ---------------------------------------------------------------- | --------------------------------------------------------------------------- |
| REQ-1 | `src/pages/categories/index.astro`                               | 媒体查询 `1fr` → `minmax(0, 1fr)`；`.category-atlas-card` 增 `min-width: 0` |
| REQ-2 | `src/styles/global.css:174-178`                                  | inline svg `max-width: none` → `max-width: 100%` + `height: auto`           |
| REQ-3 | `src/styles/global.css` mjx 段                                   | 新增 `@media (width <= 30rem)` 字号降档块                                   |
| REQ-4 | `tests/e2e/mobile-overflow.spec.ts`（新）                        | 68 case 参数化扫描                                                          |
| 文档  | `docs/contributing/testing.md`、`docs/refactor/quality-gates.md` | E2E 覆盖描述、desktop 测试数；CI 时长数据部署后回填                         |

## 4. 验收标准

- [ ] 全量 `npm run check` 绿（desktop-chromium 37→105 用例）；
- [ ] 重跑 34 页样本扫描：`pageOverflow=true` 组合 2 → **0**（360 与 320，
      `/categories/` 修复实证）；
- [ ] 行内公式缩放实证：`体系/1. 从SDE开始` 页 360 视口下无行内 svg offender；
- [ ] 显示公式横滑 + scroll-hint 机制抽检正常（DDIM 或卷积文）；
- [ ] `article-layout.spec` / `article.spec` / axe 既有断言全绿；
- [ ] `mobile-overflow.spec` 全绿且失败路径可输出 offender 明细（临时注入
      验证后移除，或 code review 确认）。

## 5. 风险

| 风险                                | 概率 | 影响 | 缓解                                                    | 阻塞 |
| ----------------------------------- | ---- | ---- | ------------------------------------------------------- | ---- |
| 行内公式缩放后基线视觉偏移          | 中   | 低   | MathJax inline vertical-align 保留；肉眼抽检            | 否   |
| 0.92em 仅回收 32/229 条，尾部仍横滑 | 确定 | 低   | 横滑 + scroll-hint 为兜底设计；量化已写入 §1/§2         | 否   |
| 391–480px 已适排公式被缩小          | 确定 | 低   | 接受并留档（可读性优先于零横滑）                        | 否   |
| 新 spec 拖慢 CI（desktop 37→105）   | 确定 | 低   | 预估 +35–70s；quality-gates CI 时长部署后回填           | 否   |
| `minmax(0,1fr)` 改变桌面网格行为    | 低   | 低   | base 规则 `minmax(20rem,1fr)` 保持不动，仅改 ≤40rem 档  | 否   |
| min-width:0 后长分类名被省略号截断  | 确定 | 低   | ellipsis 链既有设计；title 属性可考虑（不做，保持现状） | 否   |

## 6. 回滚

四项均为独立小改动（2 个 CSS 块 + 1 个组件内两行 + 1 个新 spec），可单独
revert；不涉及依赖与管线。

## 7. 审查对账闭环表

| #   | 来源  | 级别 | 修订项                                                                                   | 处置                                                                                            | 状态 |
| --- | ----- | ---- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ---- |
| 1   | R1+R2 | P0   | REQ-1 根因错误：≤40rem 单列媒体查询使 minmax(20rem) 不生效，min() 方案为 no-op           | 根因重写为 `1fr` auto 最小值被最宽卡片 min-content 撑大；修复改 `minmax(0,1fr)` + `min-width:0` | ✅   |
| 2   | R1    | P1   | §1 数据口径：240 条 offender 为截断抽样含站点 chrome，非"200 处公式/28 篇/每页 20–30 个" | 改用 dist 全量测算口径（229 条显示公式 @320、24 篇有 svg offender）                             | ✅   |
| 3   | R1    | P1   | REQ-2 验收对 DDIM 恒真                                                                   | 目标页改 `体系/1. 从SDE开始`（全站唯一真实行内溢出）；定位改防御性不变量                        | ✅   |
| 4   | R1    | P1   | REQ-3 收益未量化                                                                         | 写入 32/229 回收、197 尾部横滑兜底的确定结论                                                    | ✅   |
| 5   | R1    | P2   | REQ-3 实施位置与断点                                                                     | global.css mjx 段；`width <= 30rem`；391–480px 副作用留档                                       | ✅   |
| 6   | R1    | P2   | motion.css 交互核对                                                                      | 已核实无交互，写入 REQ-5                                                                        | ✅   |
| 7   | R1    | P2   | 容差口径与 fixed 装饰层过滤                                                              | 新 spec `+1` 容差注明差异；offender 明细过滤 fixed 层                                           | ✅   |
| 8   | R1    | P2   | 补充三条已核实论据（aria 在 svg 上/无 assistive/1px 容差吸收）                           | 写入 §2 REQ-2                                                                                   | ✅   |
| 9   | R2    | P1   | spec 结构：68 个参数化 case + expect.poll + offender 进断言消息                          | §2 REQ-4                                                                                        | ✅   |
| 10  | R2    | P1   | desktop 37→105 耗时口径 + CI 时长回填义务                                                | §3/§5/§8                                                                                        | ✅   |
| 11  | R2    | P2   | "全站"措辞过强（81 页 baseline 只测 34 页）                                              | 措辞改为 34 页样本集                                                                            | ✅   |
| 12  | R2    | P2   | probe 三缺陷（SVGAnimatedString/10 条上限/svg snippet 空）                               | 新 spec 从零实现并修复                                                                          | ✅   |
| 13  | R2    | P2   | 删除"复用 probe 代码"表述                                                                | §2 REQ-4 改为从零实现                                                                           | ✅   |
| 14  | R2    | P2   | 文档同步缺口（testing.md/quality-gates.md）+ test.use 显式仿真 + 视口统一 360×800        | §3 实施位置表 + REQ-4                                                                           | ✅   |

## 8. DoD

- [x] 双审查意见全部闭合（§7，R1+R2 合计 14 条）；
- [x] §4 验收全过（pageOverflow 2→0、SDE 页行内 offender 归零、CNN 用例 3 轮稳定、51+68 e2e 全绿）；
- [x] 完整 `npm run check` 绿并提交（bce3a28）；
- [x] push → GitHub Actions 部署成功（2m46s）→ `smoke:production` 通过 → quality-gates CI 时长已回填。
