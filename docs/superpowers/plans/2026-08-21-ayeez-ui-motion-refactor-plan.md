# EntropyXi Blog：AyeezBlog 风格界面与动效二次重构计划

> 状态：阶段 0、0.5、1 与 2 已通过主 Agent 验收；按用户要求暂停在阶段 3 之前
> 计划日期：2026-08-21  
> 计划范围：以展示层、交互层、动效层为主；允许对 permalink、分页、聚合路由、Astro SSG、数学、搜索和部署链路做有证据、可兼容、可回滚的优化或替换  
> 明确前置：上一轮 Hexo → Astro 底层框架重构已经完成，本计划不重复迁移工作  
> 参考对象：[AyeezBlog 仓库](https://github.com/Ayeez757/AyeezBlog)、[AyeezBlog 线上站点](https://blog.ayeez.cn/)  
> 规范来源：[Google TypeScript Style Guide](https://google.github.io/styleguide/tsguide.html)、[Google HTML/CSS Style Guide](https://google.github.io/styleguide/htmlcssguide)、[Google Engineering Practices](https://google.github.io/eng-practices/review/)、[web.dev 动画性能指南](https://web.dev/articles/animations-guide)、[web.dev Motion Accessibility](https://web.dev/learn/accessibility/motion)

## 0. 执行摘要

本项目不是第二次技术栈迁移，而是在已经稳定运行的 Astro 7 静态博客之上进行一次有边界的视觉和交互重构。目标是吸收 AyeezBlog 的深色赛博气质、荧光绿视觉语言、全屏叙事首屏、扫描/网格氛围层、卡片文章流和富有节奏的微交互，同时保留 EntropyXi 自己的品牌、技术内容、信息架构和静态站优势。

重构必须满足五个原则：

1. **继承迁移成果，允许受控演进**：Astro SSG、Content Collections、30 篇文章和现有发布能力是当前可靠基线，不做无理由重写；但 permalink、分页、聚合路由、SSG、数学、搜索和部署链路在 ADR、兼容策略、迁移测试与回滚方案齐备时可以优化或替换。
2. **参考风格，不复制作品**：不复制 AyeezBlog 的人物图、背景图、Logo、文案、私有素材或逐像素布局；只提炼视觉语法、信息层级和动效节奏，并用本站自有资产重新表达。
3. **服务端优先，渐进增强**：正文、导航、文章列表和所有核心链接必须在无 JavaScript 时可读可用；JavaScript 只负责增强交互和动效。
4. **动效有预算、有降级、有测试**：所有动效必须尊重 `prefers-reduced-motion`，优先使用 `transform` 和 `opacity`，不允许以持续主线程工作换取装饰效果。
5. **所有外部 Agent 产出必须由主 Agent 审查**：外部 Agent 没有最终放行权；未完成主 Agent 的差异审查、测试和浏览器验收，不得视为完成。

## 1. 已核实的项目基线

### 1.1 当前技术基线

- 框架：Astro 7，`output: "static"`，Node.js `>=22.12.0`。
- 内容：Astro Content Collections，当前 30 篇技术文章。
- 渲染：生产文章通过 `remark-math` + `rehype-mathjax` 在构建期输出 `mjx-container`，浏览器端无需加载 MathJax；客户端 MathJax 只保留在 `/dev/math-spike/` 参考页。文章正文和基础阅读不依赖 React、Vue 或其他客户端框架。
- 搜索：Pagefind 中文静态搜索。
- 质量工具：Prettier、ESLint、Stylelint、Astro Check、Vitest、Playwright、内容审计和产物审计。
- 部署：GitHub Actions → GitHub Pages，生产域名为 `https://entropyxi.github.io`。
- 当前主结构：
  - `src/pages/`：路由与 XML endpoint；
  - `src/layouts/`：`BaseLayout`、`PostLayout`；
  - `src/components/chrome/`：页头、页脚、主题切换；
  - `src/components/content/`：文章卡片、分页、阅读进度和浮动控件；
  - `src/components/visual/`：Hero、CyberGrid；
  - `src/lib/`：内容、日期、路由和 SEO 纯逻辑；
  - `src/styles/`：tokens、global、home、prose。

### 1.2 当前实现的真实定位

当前站点已经有第一版轻量赛博视觉：深色背景、绿色强调色、网格、Hero、卡片和主题切换。这证明底层组件和内容页面已经可用，但与目标参考相比仍有以下差距：

- 首屏叙事力度不足：当前 Hero 是常规技术站布局，缺少 AyeezBlog 式全屏空间感、视觉重心和明确滚动引导。
- 氛围层较弱：网格存在，但扫描线、故障纹理、流光路径、局部辉光和景深层次尚未形成统一系统。
- 页头与导航节奏偏静态：缺少品牌字标逐段入场、活动项荧光条、滚动状态和移动菜单的完整运动语言。
- 卡片信息层级偏通用：需要形成更具辨识度的边缘光、图片/抽象封面策略、状态标记与 hover/focus 联动。
- 动效仍是零散脚本：阅读进度、回顶、菜单、主题和未来的入场/光标效果需要统一生命周期与监听器清理规范。
- 当前页面已经接近目标色相，因此本次重构重点不是“换成绿色”，而是建立完整视觉系统、动效系统和可验证的体验规则。

### 1.3 AyeezBlog 参考特征（2026-08-21 Chrome 实测）

线上参考页呈现出的可借鉴模式：

- 固定深色页头，品牌标题和导航活动项使用荧光绿线条或光晕反馈；
- 首屏接近完整视口高度，大号压缩体英文标题，左右分栏，右侧为高对比人物线稿；
- 背景由照片/纹理、扫描线、数字噪声、SVG 流线和局部绿色辉光叠加；
- 页面存在自定义磁吸光标、标题字符延迟入场、导航项分批进入、持续脉冲光条；
- 文章卡片带图像、状态标签、摘要和更新时间，整体为深色玻璃/面板语言；
- 移动端提供汉堡菜单，布局和交互做响应式收敛。

明确禁止直接复制：参考站人物图片、背景图片、站名、个人头像、社交信息、文章封面、文案和实现源码。若未来希望使用任何第三方素材，必须先记录来源、许可证和可再分发依据。

## 2. 范围与非范围

### 2.1 本次范围

- 全站设计令牌、色彩、字体、间距、阴影、边框和层级系统重构；
- 页头、页脚、全局背景和主题系统重构；
- 首页 Hero、公告/导览模块、文章卡片、分页和空状态重构；
- 归档、分类、标签、搜索、关于和 404 的统一视觉收敛；
- 文章页标题区、正文容器、目录、代码块、公式、图片、前后篇和浮动控件视觉优化；
- 统一动效内核与组件级动效；
- 移动端、触控、键盘、低动态偏好、打印和无 JavaScript 降级；
- 与视觉重构直接相关的单元、E2E、可访问性、视觉回归和性能门禁；
- 开发文档、ADR、Agent 派工和审查规则。
- 对 permalink、分页、聚合路由、Astro SSG、数学、搜索和部署链路进行问题驱动的专项优化；每个专项必须与视觉批次解耦，并提供旧行为清单、收益证据、迁移方案和回滚路径。

### 2.2 明确不在范围

- 没有问题陈述、量化收益和 ADR 的框架/生成链路替换；
- 与用户目标无关的文章批量改写；
- 新增后端、数据库、账户、管理端、实时评论或服务端搜索；
- 复制 AyeezBlog 的具体素材、页面源码或产品功能；
- 为动效引入重量级通用动画框架，除非 ADR 和实测证明原生 Web API 无法满足需求；
- 把所有页面改成客户端路由；
- 为了视觉效果牺牲数学公式、代码块、SEO、可访问性或静态可读性。

## 3. 成功标准与硬约束

### 3.1 功能兼容与受控迁移

- 默认要求 30/30 篇文章均能构建并从原 pathname 访问；若批准优化 permalink，必须提供完整旧→新映射、永久重定向能力证明、canonical 更新、外链/评论影响分析和全量迁移测试。
- 现有首页分页、归档年月分页、分类、标签、搜索、关于、Atom、Sitemap、robots 和 404 默认保持兼容；允许在独立 ADR 中重构信息架构或实现链路。
- `trailingSlash`、canonical 和历史中文 URL 编码属于可变但高风险的公共契约；任何改动都必须先验证 GitHub Pages 对重定向/404 的实际支持，不能只依赖本地开发服务器。
- MathJax 公式、代码复制、Pagefind 中文搜索、主题切换、移动菜单、目录、阅读进度和回顶全部可用。
- 无 JavaScript 时仍可阅读正文、使用主导航和访问所有内容页面。

### 3.2 视觉成功标准

- 形成独立于参考站素材的 EntropyXi 赛博视觉身份。
- 桌面首屏具有明确的标题、研究方向、行动入口和右侧原创抽象视觉重心。
- 首页、列表页、文章页、搜索页和 404 共享同一套令牌与运动语言。
- 暗色为主体验；亮色不是简单反相，必须保留层级、对比度和品牌色可读性。
- 360、390、768、1024、1440、1920 CSS px 下无页面级横向溢出。

### 3.3 动效成功标准

- 动效按“环境氛围、首屏叙事、滚动显现、交互反馈、状态切换”五类管理。
- 持续动画只允许出现在少量纯装饰层；离开视口、页面隐藏或低动态模式时暂停。
- 入场动效不得阻塞内容首次可见；脚本失败时元素保持可见。
- hover 效果必须有等价 `:focus-visible` 表达，触控设备不依赖 hover 才能读到信息。
- `prefers-reduced-motion: reduce` 下移除视差、磁吸、循环脉冲、长距离移动和错峰入场，只保留必要的即时状态变化。

### 3.4 性能与体验预算

- Core Web Vitals 目标：LCP ≤ 2.5 s、INP ≤ 200 ms、CLS ≤ 0.1。阶段 0 必须按首页/列表与聚合页/普通文章/复杂公式文章四类分别记录，不得只用首页代表全站。
- 统一实验条件：生产或等价静态预览、Chromium 稳定版、390×844、DPR 1、至少 4× CPU slowdown 与 Slow 4G 等价网络；具体版本和参数写入 `docs/refactor/baseline.md`，前后对比必须使用同一条件。
- 首页、列表、聚合、普通文章和复杂公式文章都以目标值为硬门；复杂公式文章额外记录构建后 HTML/CSS 体积、`mjx-container` 数量、公式横向滚动和 CLS。生产页已经是构建期 MathJax，不得误把 `/dev/math-spike/` 的客户端 vendor 资源计入文章页预算；若阶段 0 实测发现公式页基线缺陷，按现有 ADR 的备选方案另开阶段 0.5 专项，不得在阶段 8 临时放宽指标。
- Lighthouse 关键页固定为首页、搜索页、普通文章和复杂公式文章各一页；Performance、Accessibility、Best Practices、SEO 均 ≥ 90，Accessibility 目标 ≥ 95。其余必测页至少运行 axe、资源与产物审计。
- `axe` 在首页、文章页、搜索页、归档页无 serious/critical 违规。
- 所有高频动画优先只改变 `transform` 和 `opacity`；必须改变 paint/layout 属性时记录理由和性能轨迹。
- 动效期间不允许因本站脚本产生重复的 >50 ms Long Task。
- 首屏客户端 JavaScript 以基线实测为起点；新增 gzip 体积目标 ≤ 15 KiB，超过必须提交 ADR 和收益证明。
- 首页新增自有装饰图像总传输预算目标 ≤ 350 KiB（现代格式）；移动端不得下载仅桌面显示的大图。
- CLS 关键防线：所有媒体给出尺寸或 `aspect-ratio`，字体加载不得造成可见大幅重排。
- 英文展示字体必须自托管或来自已有可信链路，设置 `font-display: swap`；若与回退字体度量差异明显，使用经过实测的 `size-adjust`/font metric overrides，并以首屏 CLS 截图与 trace 证明不会跳版。
- 在 `width <= 48rem`、`(hover: none)` 或资源受限策略命中时，默认关闭循环扫描线和磁吸光标，减少常驻 glow/filter，避免大面积 `backdrop-filter`；移动端视觉层级通过静态颜色、边框和渐变保持，而不是依赖高成本动画。

## 4. 目标信息架构与页面体验

### 4.1 全局页面骨架

```text
html[data-theme][data-motion]
└── body
    ├── SkipLink
    ├── AmbientBackground（纯装饰，aria-hidden）
    ├── SiteHeader
    ├── main#main-content
    │   └── 页面内容
    ├── SiteFooter
    └── 全局渐进增强控件（按页面需要）
```

`AmbientBackground` 只负责装饰层，不承载内容和交互。内容层永远高于背景层；背景层 `pointer-events: none`；任何 Canvas/SVG 故障都不能影响主内容。

### 4.2 首页

1. **Hero**：接近视口高度，但使用 `min-height` 而不是固定高度；左侧为品牌叙事，右侧为原创数学/生成模型主题 SVG 视觉。
2. **系统状态/公告**：短小、可关闭与否由产品决策决定；第一阶段只做静态信息，不引入持久化状态。
3. **研究主题轨道**：以标签/节点形式展示扩散模型、流匹配、SDE/ODE、RAG、数值分析；链接到现有分类或标签。
4. **最新文章流**：保持现有分页与语义列表，卡片加入封面占位图形、分类、标题、摘要、日期和标签。
5. **分页**：保留静态链接；动画仅用于焦点/hover，不拦截导航。

### 4.3 文章页

- 标题区强调分类、标题、发布日期、更新时间和标签，不复制参考站文章数据。
- 桌面使用主栏 + sticky TOC；移动使用现有 `details` TOC。
- 正文可读性优先：字符宽度、行高、公式滚动、代码块和图片不能被赛博装饰干扰。
- 阅读进度为状态动效，不使用循环动画。
- 复制成功必须有文字反馈；失败也要反馈，不能静默。
- 浮动控件按键盘焦点顺序合理排列，并在小屏避免遮挡正文。

### 4.4 聚合与工具页

- 归档：时间轴可采用荧光节点，但 DOM 保持自然阅读顺序。
- 分类/标签：使用同一面板系统，数量信息不能只靠颜色表达。
- 搜索：输入、加载、无结果、错误、结果列表四种状态均有完整样式和可访问文本。
- 关于：可以使用时间线和技术栈面板，但禁止虚构经历或项目数据。
- 404：保留核心导航和返回首页，装饰故障效果不得闪烁。

## 5. 动效系统设计

### 5.1 动效分级

| 等级 | 用途         | 允许形式                       | 默认时长    | reduced-motion |
| ---- | ------------ | ------------------------------ | ----------- | -------------- |
| M0   | 必要状态反馈 | opacity、颜色、即时图标/文本   | 80–180 ms   | 保留但缩短     |
| M1   | 微交互       | 2–6 px translate、scale ≤ 1.02 | 120–220 ms  | 取消位移       |
| M2   | 内容显现     | opacity + 8–24 px translate    | 280–520 ms  | 立即显示       |
| M3   | 首屏叙事     | 字段分段、SVG 路径、局部辉光   | 450–1000 ms | 静态终态       |
| M4   | 环境循环     | 光线、噪声、扫描、轻微漂移     | ≥ 2 s       | 完全停止       |

### 5.2 统一动效令牌

在 `tokens.css` 中定义数值令牌，在 `motion.css` 中定义通用状态契约和 keyframes：

- `--motion-duration-instant/fast/normal/slow/hero`；
- `--motion-ease-standard/emphasized/enter/exit/spring-like`；
- `--motion-distance-xs/sm/md/lg`；
- `--motion-stagger-xs/sm`；
- `--motion-enabled` 或根元素数据属性，用于 CSS/JS 同步。

禁止在组件内散落无解释的 `0.37s`、`23px`、自定义 cubic-bezier。必要的例外必须带“为什么”注释，并在设计令牌清单登记。

### 5.3 动效内核职责

新增轻量原生模块，建议结构：

```text
src/lib/motion/
├── motion-preferences.ts   # 媒体查询与页面可见性
├── reveal-controller.ts    # IntersectionObserver 显现
├── pointer-controller.ts   # fine pointer 下的磁吸/光标增强
├── ambient-controller.ts   # 环境动画暂停与恢复
└── lifecycle.ts            # 幂等初始化、AbortController、销毁
```

约束：

- 不建立常驻全局可变对象；模块公开小型 `init(): Cleanup` 或 controller API。
- 每个初始化函数必须幂等，重复执行不会重复绑定监听器或插入 DOM。
- 所有监听器统一通过 `AbortController` 或显式 cleanup 清理。
- `requestAnimationFrame` 同一效果最多一个循环；页面隐藏时取消，恢复时按需重启。
- 指针移动只读取一次位置，并在 rAF 中写入 transform；禁止在每个 `pointermove` 中交替读写布局。
- `IntersectionObserver` 负责一次性 reveal；完成后 `unobserve`。
- 不使用 `setInterval` 驱动视觉动画。
- 不修改内建原型，不使用 `eval`、`Function` 或不安全 `innerHTML`。

### 5.4 CSS 终态优先

基础 DOM 默认展示终态。只有在 JavaScript 确认初始化成功后，根元素才添加 `data-motion="ready"`，CSS 才允许把未显现元素置为初始隐藏态。这样脚本被阻止、加载失败或执行异常时，内容不会永久不可见。

### 5.5 磁吸光标规则

- 仅在 `(hover: hover) and (pointer: fine)` 且非 reduced-motion 时启用。
- 默认系统光标不全局隐藏；第一阶段只实现跟随光晕，不替换基本 cursor。
- 首版白名单仅包含 Logo、主导航、主要 CTA 和文章卡片；正文、普通文本链接、代码、公式、输入框和可选择文本区域不得磁吸。
- 光标层 `aria-hidden="true"`、`pointer-events: none`，不得遮挡选择和点击。
- 输入框、代码选择、文本链接上自动弱化；窗口失焦和离开页面时隐藏。
- 性能不达标或造成可用性问题时可以独立关闭，不影响其他动效。

## 6. 整体架构规则（硬性）

以下规则借鉴 Google “可读性、简单性、代码健康持续改善、小变更和测试同行”的原则，并针对 Astro 静态博客落地。`必须` 为阻塞规则，`应该` 为默认规则，偏离时必须在审查说明中给出依据。

### 6.1 分层与依赖方向

允许的依赖方向：

```text
pages → layouts → components → lib/data
             └──→ styles/tokens
tests → public contracts / rendered behavior
scripts → content/schema/build output
```

- `src/pages/` 必须只负责编排路由数据和页面组合，不放通用视觉实现或复杂领域逻辑。
- `src/layouts/` 必须负责跨页面结构、metadata 和 slot，不读取页面特有 DOM。
- `src/components/` 必须按职责拆分；组件不得直接反向导入 page 或 layout。
- `src/lib/` 必须保持 UI 无关；不得导入 `.astro`、全局 DOM 或组件 CSS。
- `src/data/` 只存站点配置和稳定内容数据，不存运行时状态；当前 `src/data/site.ts` 保存站点 URL、标题、描述与语言等常量。
- `src/styles/` 负责全局令牌、基础排版和跨组件层；组件私有样式优先放组件内。
- `scripts/` 只做构建/审计任务，不能成为浏览器运行时依赖。
- 禁止循环依赖；新增跨层例外必须先写 ADR。

### 6.2 服务端与客户端边界

- 默认使用 `.astro` 服务端渲染和静态 HTML。
- 不为只需少量 DOM 交互的组件引入客户端框架。
- 当前方案明确不采用 React、Vue、Preact、Svelte 等 UI 运行时；AyeezBlog 的 Vue 3 技术栈只作为参考站实现背景，不进入本站架构。新增任何客户端框架必须通过 §6.6 ADR、体积/维护/静态降级证据和用户批准。
- 客户端模块只处理：菜单、主题、搜索、复制、进度、reveal、指针和必要状态反馈。
- 核心信息不能只由 CSS `content`、Canvas 或客户端脚本生成。
- 所有核心链接必须是真实 `<a href>`；不使用 click handler 模拟导航。
- 按钮只处理动作，链接只处理导航，禁止混用语义。

### 6.3 组件规则

- 单个组件只承担一个清晰职责；若模板、样式和脚本需要同时理解多个无关子系统，应拆分。
- Props 使用显式 `interface Props`；复杂共享类型放 `src/lib` 相应领域目录。
- 不使用布尔参数堆叠制造“万能组件”；超过两个互相影响的视觉 variant 时使用可枚举 union。
- 组件公开契约通过 Props、slot 和 `data-*` 属性表达，不依赖外部 CSS 猜测内部层级。
- 交互组件必须定义：初始状态、成功状态、失败状态、禁用状态、键盘行为和无 JS 行为。
- 视觉装饰 SVG 必须 `aria-hidden="true"`；有信息含义的图形必须提供文本等价物。

### 6.4 CSS 架构规则

建议层次：

```text
tokens.css          # 颜色、尺寸、字体、阴影、z-index、motion tokens
global.css          # reset、body、链接、focus、全局容器
motion.css          # 通用 reveal、reduced-motion、环境动画契约
home.css            # 首页页面组合，不定义跨页面 token
prose.css           # 文章正文专用
component styles    # 组件内部样式
```

- 颜色、间距、圆角、阴影、z-index 和时长必须优先使用 token。
- token 名称表达语义，如 `--color-surface-elevated`，不使用 `--dark-2` 等呈现式命名。
- 类名表达职责而非外观，如 `.post-card__meta`，不使用 `.green-box`。
- 选择器最多三层；禁止依赖 DOM 偶然结构的长选择器。
- 禁止 `!important`，唯一默认例外是经过验证的 reduced-motion 全局保护；例外必须注释。
- 不使用 ID 选择器做样式。
- 组件不得重定义全局 token；主题值统一在主题作用域维护。
- `z-index` 必须来自层级 token：background、content、sticky、overlay、modal；禁止随意使用四位数。
- 任何 hover 样式必须同时评估 `:focus-visible` 和触控行为。
- CSS 动画 keyframe 使用功能命名并集中管理；只被单组件使用的 keyframe 可留在组件内。

### 6.5 资源与版权规则

- 新视觉资产优先使用原创 SVG、CSS 生成图形或明确授权的自有图片。
- 所有图片必须记录来源、许可证/所有权、用途和优化方式。
- 不直接热链参考站资源。
- 位图优先 AVIF/WebP，并保留合理 fallback；SVG 需清理元数据和无用节点。
- 装饰图按断点提供尺寸策略，禁止移动端下载后再 `display: none` 的大资源。
- 所有非装饰图片必须有准确 `alt`；装饰图片使用空 `alt` 或 `aria-hidden`。

### 6.6 ADR 触发条件

出现以下任一情况必须先写 ADR，主 Agent 放行后实施；这里是“允许优化的决策门”，不是永久冻结：

- 引入新的运行时依赖或客户端框架；
- 修改 permalink、分页、聚合路由、Content Collection schema、Astro SSG、Markdown/数学渲染、搜索或部署链路；
- 引入 Canvas/WebGL、大型动画库或第三方统计/评论脚本；
- 超过任一性能预算；
- 改变主题存储键或页面初始化顺序；
- 改变部署、构建产物或 GitHub Pages 行为；
- 为参考站素材主张可复用许可证。

## 7. 编码规则（参考 Google 规范并本地化）

### 7.1 通用可读性

- 代码为读者而写；优先简单、直接、易检索的实现。
- 一个函数只做一件事；复杂函数先通过命名清晰的纯函数拆分，不用注释掩盖复杂度。
- 注释解释“为什么、约束和取舍”，不复述代码“做了什么”。
- 删除失效注释、死代码和被注释掉的旧实现；历史由 Git 保存。
- 不在功能变更中夹带大面积无关格式化或重命名。
- 新行为必须有测试；纯重构必须由现有或新增测试证明行为不变。

### 7.2 TypeScript

- 必须通过 strictest Astro/TypeScript 检查，不新增 `any`。
- 不使用 `@ts-ignore`、`@ts-nocheck`；测试中的特殊类型断言也应优先通过显式构造解决。
- 对显而易见的局部值使用类型推断；对模块边界、公共 Props、复杂返回值使用显式类型。
- 优先使用最简单的类型构造；不为减少少量重复引入难读的条件/映射类型。
- 变量和函数使用 `camelCase`，类型/接口/组件使用 `PascalCase`，常量只在真正不可变的模块级常量时使用 `UPPER_SNAKE_CASE`。
- 使用命名导出；默认导出仅限框架配置等现有约定。
- import 顺序：平台/第三方、空行、`@/` 内部模块、相对模块；由格式化和 lint 维持一致。
- 不扩展全局对象，不修改内建原型，不使用动态代码执行。
- DOM 查询后先收窄类型和处理空值；不使用非空断言掩盖生命周期问题。
- Promise 必须被 `await`、`return` 或显式 `void` 处理；异步失败提供用户可见或可诊断路径。

### 7.3 Astro

- frontmatter 只做数据准备，不直接操作 DOM。
- 页面和布局的 Props 必须显式定义。
- 动态 class 使用 `class:list`，避免手工拼接难读字符串。
- 内联脚本仅用于首屏必须同步执行且极小的逻辑，例如主题 FOUC 防护；其他脚本模块化。
- 不默认启用 ClientRouter；若未来启用必须做脚本重复初始化、焦点、滚动恢复和 View Transition 专项 ADR。
- scoped style 与全局 style 的边界明确；只有确实跨 slot/Markdown 边界时使用 `:global()`。

### 7.4 HTML 与可访问性

- 使用合法、语义化 HTML；标题层级连续，每页只有一个主 `h1`。
- 所有输入有可见或程序化 label；错误信息说明问题和修复方式。
- 所有交互可通过键盘完成；焦点样式清晰且不被遮挡。
- 不用颜色、位置、大小或动画作为唯一信息载体。
- 文本与背景对比达到 WCAG 2.2 AA：正文至少 4.5:1，大文本至少 3:1。
- 页面必须提供跳到主内容链接。
- 移动菜单打开时管理 `aria-expanded`、`aria-hidden`、焦点进入/返回和 Escape；背景滚动锁不能破坏页面位置。
- 复制、搜索和主题切换等状态变化使用可感知文本；必要时用克制的 `aria-live`。
- 外链使用 HTTPS，`target="_blank"` 同时使用 `rel="noopener noreferrer"`。

### 7.5 CSS

- 必须通过 Stylelint；格式由 Prettier 决定，不进行个人偏好争论。
- 使用有效 CSS 和有意义类名；属性按现有格式化结果排列，不手工维护风格冲突。
- 采用逻辑属性时保持一致，不在同一组件混用相互覆盖的物理属性。
- 响应式采用内容驱动断点；断点统一登记，组件不随意新增相近断点。
- 使用 `clamp()` 时给出可读上下限，不能造成极小或极大屏失控。
- 动画中优先 `transform`/`opacity`，慎用 filter、box-shadow、background-position 的持续变化。
- `will-change` 只在性能记录证明需要时短期启用，结束后移除。
- 主题必须使用语义 token，不在组件中直接判断暗色后重复整套规则。

### 7.6 客户端脚本

- 事件监听器默认 passive（能 passive 时）；滚动处理必须经 rAF 节流。
- 初始化和销毁对称；重复初始化不得重复注册。
- 读取布局和写入样式分批处理，避免 layout thrashing。
- 只在需要时查询 DOM，缓存稳定引用；不缓存跨页面已失效节点。
- Web API 不支持时提供静态降级，不因装饰能力缺失抛出致命错误。
- 不记录用户输入、浏览历史或其他不必要数据。
- localStorage 只存非敏感展示偏好，并捕获禁用/配额异常。

### 7.7 测试代码

- 测试验证用户可见行为和公共契约，不绑定私有实现细节。
- 每个 bug 修复先补能失败的回归测试，再修复。
- 测试名称描述条件和预期结果。
- 不用任意长 `waitForTimeout` 掩盖竞态；优先等可观察状态。
- 视觉测试冻结动画、时间和随机性；动态区域明确 mask 或进入稳定终态。
- reduced-motion、键盘和无 JS 不是可选手工检查，必须进入自动化覆盖。

## 8. 项目总体规则

### 8.1 变更单元与 Git

- 每个变更批次只解决一个可独立审查的问题，参考 Google Small CL 原则。
- 基础链路专项和 UI/动效批次必须使用不同 ADR、不同文件租约和不同变更单元；禁止在视觉批次中夹带 permalink、路由、SSG、数学、搜索或部署替换。
- 目标规模通常 100–400 行人工变更；超过约 800 行或跨 15 个以上文件时，主 Agent 必须先拆分。若因原子迁移确实无法拆分，必须在变更说明中记录不可拆理由、分段 review 顺序和额外回归范围，未记录则不进入 review。
- 纯重命名/搬迁与功能修改分开提交。
- 每个提交必须构建可用，不允许“中间提交会破坏主分支”。
- 分支使用 `codex/` 前缀，除非用户另行指定。
- 不修改或清理用户无关改动，不使用破坏性 reset/checkout。
- 提交信息采用 `type(scope): summary`，类型至少包括 `feat`、`fix`、`refactor`、`test`、`docs`、`perf`、`style`、`chore`。
- `agy` 的首次启动按用户要求从 PowerShell 发起，固定入口为 `C:\Users\entropy\AppData\Local\agy\bin\agy.exe`；该步骤之后，本项目由主 Agent、DSH 和 Antigravity 发起的全部项目终端命令必须在 Git Bash 中执行。Antigravity CLI 的交互输入继续写入其已启动会话，不视为另开 PowerShell 项目命令；Read/Glob/Grep、浏览器检查和非终端文件补丁不属于终端命令。若某审查执行器无法选择 Git Bash，则该会话禁止使用其 Shell/Pwsh 工具，由主 Agent 在 Git Bash 中完成所需核验。

### 8.2 完成定义（Definition of Done）

单个批次只有同时满足以下条件才算完成：

1. 计划条目和验收条件已实现；
2. 主 Agent 已逐文件审查外部 Agent 的 diff；
3. 格式、lint、类型、相关单测/E2E 通过；
4. Chrome 在桌面、移动、暗色、亮色和 reduced-motion 下验收；
5. 无新增 serious/critical 可访问性问题；
6. 性能和资源预算未回退，或已有批准 ADR；
7. 文档、测试和代码同步；
8. 没有未解释的 TODO、禁用测试或类型抑制；
9. 主 Agent 给出“通过/需修改/拒绝”结论；
10. 若影响生产路径，回滚点和 smoke check 已记录。
11. 对应的产物审计、URL/HTML/资源检查已通过；主要可视批次的 §12.5 截图索引已由主 Agent 采集并记录。

### 8.3 审查标准

主 Agent 按 Google Engineering Practices 的维度审查：设计、功能、复杂度、测试、命名、注释、风格、文档和代码健康。

审查严重级别：

- `blocking`：破坏路由、内容、数学、构建、部署、安全或可访问性底线，必须修复。
- `major`：明显增加复杂度、性能风险或维护负担，默认阻塞。
- `minor`：局部质量问题，可在有明确后续项时放行。
- `nit`：纯建议，不得以个人风格偏好阻塞。

### 8.4 依赖与安全

- 原生 Web API 和 CSS 能满足时不新增依赖。
- 新依赖必须说明用途、体积、许可证、维护状态、替代方案和移除成本。
- lockfile 必须同步，依赖升级独立成批。
- 不提交密钥、Token、账号、个人路径或生产日志。
- 第三方脚本默认拒绝；必须通过 CSP/隐私/失败降级评估。
- 用户可控内容不得直接进入 `set:html`/`innerHTML`；现有 JSON-LD 和受控 Markdown 管线保持审计。

### 8.5 文档规则

- 架构决策写 ADR，实施步骤写本计划或阶段记录，用户操作写 README。
- 文档使用与代码一致的真实命令和路径。
- 每个阶段结束更新状态、实测结果、已知偏差和下一阶段前置。
- 不把临时聊天结论当作持久规范；关键决策必须入库。

## 9. Agent 调用、所有权与监督规则

### 9.1 主 Agent（唯一调度者与最终审批者）

主 Agent 职责：

- 维护本计划、架构边界、任务拆分、依赖顺序和风险登记；
- 编写复杂逻辑、关键核心代码和跨组件基础设施；
- 设计动效生命周期、状态机、性能策略、测试策略和回滚方案；
- 为两个子 Agent 写清楚输入、允许修改文件、禁止项、验收条件和测试命令；
- 监督共享工作区，避免并发修改同一文件；
- 对所有子 Agent 产出做逐文件 diff review、浏览器 review 和测试；
- 对 `blocking/major` 问题要求返工，最终只由主 Agent宣布通过。

主 Agent 禁止事项：

- 不直接编写或优化主要前端可视界面；此类任务必须派给第二子 Agent；
- 不因子 Agent 声称“测试通过”就跳过本地复核；
- 不把复杂核心逻辑下放后完全失去所有权；
- 不允许两个子 Agent 同时写同一文件或相互覆盖未审查变更。

### 9.2 第一子 Agent：DeepSeek Harness

固定调用环境：

- 在 `Blog_file` 工作区使用 `dsh --profile web`；
- 模型固定为 `DeepSeek-V4-Flash`；
- 思考程度固定为 `Max`；
- 审查任务使用 `Read Only`；编码任务只在明确文件白名单下临时使用 `Workspace Write`。
- `Read Only` 表示不得修改源码、文档、配置、锁文件和其他版本控制内容；审查会话不负责运行会写入 `dist/`、缓存或截图的测试。需要构建或测试时，由主 Agent 在 Git Bash 中复跑并把结果作为审查输入。

允许任务：

- 简单逻辑、纯函数、普通编码、测试夹具、审计脚本和机械性文档更新；
- 对计划、架构、测试、性能和安全进行独立审查；
- 分析主 Agent 提供的有限 diff，提出问题和测试建议。

禁止任务：

- 主要前端可视界面、视觉令牌、布局、组件样式、素材和动画表现的实现或优化；
- 决定架构方向、修改路由/内容 schema/部署、引入依赖；
- 未授权编辑计划外文件；
- 自行提交、合并或宣布任务通过。

交付协议：

1. 主 Agent 提供单一、边界明确的任务书；
2. 子 Agent 报告修改文件、设计说明、测试和未决问题；
3. 主 Agent 检查 diff、运行相关测试、必要时要求修改；
4. 只有主 Agent 明确记录 `APPROVED` 才进入下一任务。

### 9.3 第二子 Agent：Antigravity CLI（主要前端实现者）

固定调用环境：

- 在 `D:\Blog_file` 文件夹打开 Git Bash；
- `agy`（antigravity-cli）的首次进程由 PowerShell 使用绝对路径 `C:\Users\entropy\AppData\Local\agy\bin\agy.exe` 在 `D:\Blog_file` 启动；此后所有独立项目终端命令均使用 Git Bash，Antigravity 提示词继续发送到该已启动会话；
- 模型固定为 `gemini3.7flash`；
- 思考程度固定为 `high`。

强制所有权：

- 所有“主要且涉及前端可视界面”的编写与优化必须派给该子 Agent，不能由主 Agent或 DeepSeek 子 Agent直接完成；
- 包括 Astro 可视组件模板、CSS、设计令牌的视觉值、响应式布局、Hero、导航外观、卡片、装饰 SVG、动效表现参数和视觉微调；
- 也可以承担简单逻辑或普通编码，但这是次要用途。

主 Agent仍保留：

- 组件契约、目录与依赖方向；
- 动效控制器、生命周期、安全和性能核心逻辑；
- 测试门禁、代码审查、浏览器验收和最终放行。
- 为消除写租约冲突，阶段 2 的结构性脚本拆分属于主 Agent 基础设施例外：主 Agent 只可修改相关 `.astro` 的 `<script>` 块和最薄初始化桥接，不改可视标记、样式或视觉参数；完成行为等价回归并释放租约后，组件才交给 Antigravity。

第二子 Agent交付必须包含：

- 改动前目标与参考特征；
- 修改文件清单；
- 桌面/移动、暗色/亮色、reduced-motion 的视觉断言清单，以及其能力范围内可提供的截图；§12.5 固定截图由主 Agent 使用 Chrome 统一采集、核验和署名，不能用子 Agent 的自证代替；
- 未使用参考站受限素材的声明；
- 本地执行的格式、lint、类型和相关测试；
- 已知差异和可回退点。

当前验证状态：2026-08-21 已通过 PowerShell 绝对路径成功启动 Antigravity CLI 1.1.17，工作区为 `D:/Blog_file`，已登录 Google AI Pro，模型显示 `Gemini 3.7 Flash (High)`。CLI 不可用不再是实施阻塞项；后续统一复用该入口，不依赖自动化进程未及时刷新的 PATH。

### 9.4 共享工作区写入租约

- 任一时刻同一文件只能由一个 Agent 持有写入租约。
- 主 Agent 派工时列出“允许修改”和“只读参考”文件。
- 子 Agent 完成后停止写入，主 Agent 审查期间其他 Agent 不得改这些文件。
- 需要返工时重新授予租约；禁止边审查边并发重写。
- 若出现意外重叠，立即停止相关 Agent，以 `git diff` 为证据人工合并，不接受整文件覆盖。
- 每批外部产出先存为未提交工作树变更；主 Agent批准前不提交。

### 9.5 派工决策表

| 任务                    | 主 Agent                            | DeepSeek 子 Agent | Antigravity 子 Agent |
| ----------------------- | ----------------------------------- | ----------------- | -------------------- |
| 架构、ADR、阶段拆分     | 主责                                | 审查              | 提供可视实现反馈     |
| 复杂动效生命周期/状态机 | 主责                                | 可审查            | 对接并实现视觉层     |
| 主要 Astro UI/CSS/SVG   | 禁止直接实现，负责设计契约与 review | 禁止              | 强制主责             |
| 简单纯函数/审计脚本     | 可做                                | 可主责            | 可做（次要）         |
| 单元/E2E/性能门禁       | 主责                                | 可协助普通用例    | 补视觉行为夹具       |
| 浏览器视觉调优          | 监督与最终验收                      | 不负责            | 主责                 |
| 最终 diff review        | 唯一主责                            | 可提供第二意见    | 自检但无放行权       |

## 10. 分阶段实施计划

严格顺序为：阶段 0 → 阶段 0.5（只做专项决策，获批专项另开批次）→ 阶段 1 → 阶段 2 → 阶段 3 → 阶段 4A → 阶段 4B → 阶段 5 → 阶段 6 → 阶段 7 → 阶段 8。未满足前一阶段退出条件不得用后续视觉成果“倒逼放行”。基础链路专项即使在阶段 0.5 获批，也必须与 UI 阶段串行或使用互不重叠文件租约，并单独完成构建、兼容与回滚验证。

### 阶段 0：环境、基线与冻结（主 Agent主责）

目标：证明本轮从一个可用 Astro 基线开始，并让两个子 Agent 的调用路径可复现。

任务：

- [x] 记录 `git status`、Node/npm/Astro 版本、当前构建和全量 `npm run check` 结果。
- [x] 冻结当前生产 URL/产物清单，与上一轮迁移 manifest 对照。
- [x] 为首页、文章页、归档、分类、标签、搜索、关于、404 采集 1440×900 与 390×844 的暗/亮基线。
- [x] 记录当前 CSS/JS gzip 体积、Lighthouse、axe、LCP/CLS 和动画性能基线。
- [x] 确认 DSH：Blog_file、DeepSeek-V4-Flash、Max、Read Only。
- [x] 从 PowerShell 绝对路径启动 `agy`，核实工作区 `D:/Blog_file`、`Gemini 3.7 Flash` + `High` 与登录状态；后续独立项目命令固定使用 Git Bash。
- [x] 建立 `docs/refactor/` 下的基线报告、视觉矩阵和 Agent 运行记录。
- [x] 创建 `audit-screenshots/phase-0/`，保存基线截图及对应 viewport、主题、运动偏好、commit 和采集命令元数据；二进制截图是否入库由仓库体积规则决定，但索引和结论必须入库。
- [x] 新建 refactor 分支和可恢复标签/commit（不更改生产部署）。
- [x] 建立“基础链路优化候选表”：分别记录 permalink/分页/聚合路由、SSG、数学、搜索、部署的现状痛点、候选方案、预期收益、兼容成本和是否值得单独立项。
- [x] 核实 `src/pages/dev/math-spike.astro` 是否进入生产产物；若只是迁移期实验页，记录保留/排除/删除的独立决策，不在 UI 批次顺手处理。
- [x] 记录未跟踪 Hexo 遗留目录 `.deploy_git/` 的用途、体积和清理责任；本计划不自动删除用户数据。
- [x] 记录当前 `audit-screenshots/` 顶层旧截图，建立“迁入 legacy 索引 / 保留原位 / 用户批准后清理”的处理结论，避免与 `phase-N/` 证据混淆。
- [x] 在 Agent 派工模板中注明 Astro 实际静态资源目录为 `astro-public/`（由 `astro.config.ts` 的 `publicDir` 定义），不得按惯例误写到不存在的 `public/`。

退出条件：全量基线绿色；URL/内容冻结；两个子 Agent 均能按指定模型启动；浏览器基线和性能数据可复现。

回滚：本阶段不改生产代码，只删除新增报告即可。

### 阶段 0.5：基础链路专项决策（主 Agent主责，按需执行）

目标：允许优化底层能力，但阻止视觉重构顺手夹带高风险迁移。

每个候选专项必须独立完成以下决策包：

1. 问题陈述：当前实现的可复现缺陷、维护成本或量化瓶颈；
2. 方案比较：至少包含“保持现状”和一个替代方案；
3. 公共契约：旧 URL、canonical、feed、索引、数学语法、搜索索引、部署产物和回滚点；
4. 迁移计划：兼容层、双写/双构建（如适用）、数据与内容转换、停止条件；
5. 测试证据：本地与 GitHub Pages 等价环境，不能只依赖 dev server；
6. ADR：收益大于复杂度且主 Agent批准后才进入实现。

专项规则：

- permalink/路由优化必须先证明生产托管层能正确处理旧 URL；否则保留旧 pathname 或生成兼容静态页面。
- 分页/聚合优化必须保留可抓取 `<a href>`，不得改为仅客户端无限滚动。
- SSG 替换必须保持静态可部署、构建可重复、内容全量审计和预览一致性；如改为其他渲染模式，需单独获得用户批准。
- 数学链路替换必须覆盖 30 篇扫描、5 篇复杂公式视觉基线、`\tag`、长公式、行内基线和打印。
- 搜索替换必须覆盖中文、英文、无结果、增量索引、离线静态部署、索引体积和可访问性。
- 部署替换必须具备 staging、artifact 审计、生产 smoke、权限最小化和一键回滚。

退出条件：每个候选被明确标记为“保持现状 / 进入独立实施 / 延后”，不允许模糊地混入视觉阶段。

获批专项的实现门：由主 Agent 单独派工、单独 ADR/变更单元、单独文件租约、独立回滚点，并完整遵守 §8.2 DoD。UI 阶段进行中不得并发写入该阶段租约文件；若专项改变 UI 依赖契约，必须先暂停后续 UI 阶段、完成专项回归并重新批准 design specs。

执行结果（2026-08-21）：permalink、分页、聚合路由、Astro SSG、Pagefind 与 GitHub Pages 保持现状；数学可访问性与迁移期客户端 vendor 移除作为唯一专项完成独立实施。30/30 文章、1289/1289 个公式、5 篇复杂公式双视口、Chrome DOM 与 Lighthouse `svg-img-alt` 均通过，证据与回滚见 `docs/refactor/foundation-math-implementation.md`。

### 阶段 1：视觉方向与组件规格（Antigravity 主责，主 Agent审批）

目标：先形成可审查的设计系统和低成本样板，不直接全站铺开。

任务：

- [x] Antigravity 产出首页桌面/移动视觉方案，明确与 AyeezBlog 的借鉴点和原创差异。
- [x] 定义暗/亮主题色板、字体、网格、间距、圆角、边框、阴影、辉光、z-index 和 motion tokens。
- [x] 明确首访主题策略：选择“暗色为品牌默认”或“首次跟随系统、用户选择覆盖”；不得继续保留含糊的“若支持”。若选择系统跟随，阶段 2 必须覆盖媒体查询变化、持久化优先级、FOUC 和 cleanup。
- [x] 记录展示字体的名称、来源、许可证、子集策略、回退字体、度量数据来源、`font-display` 和 CLS 验证方法。
- [x] 定义原创 Hero 右侧视觉：优先数学符号/流场/扩散轨迹 SVG，不使用参考站人物素材。
- [x] 定义 Header、Hero、PostCard、SectionHeader、Pagination、Footer、TOC、SearchResult 的状态规格。
- [x] 为每个交互组件列出 default、hover、focus-visible、active、disabled、loading、error（适用时）。
- [x] 定义断点矩阵和各断点信息保留/折叠规则。
- [x] 定义 reduced-motion 静态终态。
- [x] 交付 `docs/refactor/design-specs.md`，至少包含：语义色/token 表、WCAG 2.2 AA 对比度表、字体加载与回退方案、断点矩阵、原创 SVG 规格、组件全状态、运动分级、移动端性能降级规则和资产许可证台账。
- [x] 在 `audit-screenshots/phase-1/` 交付 1440×900 暗/亮、390×844 暗/亮以及 reduced-motion 静态终态样板；所有样板标注所用 commit 与设计规格版本。
- [x] 主 Agent检查语义、可达性、复杂度、资产许可证和 Astro 可实现性。

退出条件：`docs/refactor/design-specs.md`、固定截图矩阵、设计令牌表、组件状态表、响应式规格、动效分级和原创资产方向全部通过主 Agent review；尚不修改全站页面。

执行结果（2026-08-22）：Antigravity（Gemini 3.7 Flash / High）完成视觉规格和零依赖静态原型；主 Agent要求并验收 360/390 移动端菜单及溢出修正、键盘焦点、亮暗主题、reduced-motion、原创资产和许可台账。8 组固定浏览器证据全部通过，整库 `npm run check` 全绿，阶段 1 放行。

### 阶段 2：架构地基与自动门禁（主 Agent主责，DeepSeek 协助）

目标：在视觉批量改动前建立稳定的动效契约和测试支架。

任务：

- [x] 主 Agent盘点现有 `.astro` 内联客户端脚本；把会与 Antigravity 主要可视文件发生写租约冲突的主题、菜单、阅读进度、TOC、复制、回顶等核心行为拆入 `src/lib/` 对应 TypeScript 模块，只在组件保留最薄初始化桥接。纯首屏主题防闪脚本可按 §7.3 例外保留。
- [x] 在拆脚本前先补当前行为基线测试，再建立拆分后的等价回归：DOM 契约、无 JS、重复初始化、导航/主题/文章交互均不得变化；该批次必须在任何主要视觉写入前由主 Agent独立完成并放行。
- [x] 主 Agent只定义 motion token 的语义名称、类型/取值边界、`data-motion` 契约、生命周期和安全降级默认值；阶段 1 冻结并经批准的时长、距离、easing、辉光等视觉数值由 Antigravity 落值。若视觉规格与生命周期/性能契约冲突，不自动以任一方为准：暂停派工，由 Antigravity 提交视觉替代、主 Agent评估技术边界并作最终裁决，更新 `design-specs.md` 后继续。
- [x] 主 Agent实现 `motion-preferences`、页面可见性、幂等初始化和 cleanup。
- [x] DeepSeek 可协助编写简单纯函数、媒体查询 mock、单测和 bundle audit 脚本。
- [x] 新增 reduced-motion、无 JS、键盘、重复初始化和监听器清理测试。
- [x] 在 `playwright.config.ts` 或等价配置中建立明确项目：desktop Chromium、390×844 touch、360px overflow、reduced-motion、JavaScript disabled、200% zoom；固定主题、动画终态、字体和时间。iOS Safari 以 WebKit/真实设备/远程设备抽查，记录负责人、频率和最小页面集。
- [x] 把视觉阈值、Lighthouse 配置、bundle 预算和产物审计阈值固化到版本控制配置或 `docs/refactor/quality-gates.md`；CI 全矩阵目标总时长和分片策略在阶段 2 实测后冻结。
- [x] 建立资源与 bundle 预算报告。
- [x] 记录是否需要新增 ADR；默认不引入动画依赖。
- [x] 明确终态优先契约：仅当初始化成功后设置 `html[data-motion="ready"]`，任何异常都保持内容可见；无 JS E2E 必须断言 Hero、卡片和正文均未被隐藏。

退出条件：客户端核心行为已从主要可视租约中解耦且行为等价；新基础设施本身不改变现有视觉；单测通过；重复初始化无副作用；无 JS 内容可见；预算报告可在 CI 运行。

执行结果（2026-08-22）：主 Agent完成生命周期、主题、移动抽屉、阅读进度、回顶、复制、搜索和运动环境模块拆分，并补齐存储不可用、系统主题变化、重复初始化、cleanup、键盘、真实 Pagefind、剪贴板成功/失败及无 JS 回归。DeepSeek（DeepSeek‑V4‑Flash / Max）仅协助简单纯函数单测，主 Agent复核并收紧大小写无效值断言；Antigravity（Gemini 3.7 Flash / High）仅负责无 JS 搜索提示和对比度可视修正，主 Agent拒绝其越界修改冻结原型并纠正不实的全站样式结论。完整门禁通过：21 项单测、29 项 E2E、4 个关键页 axe serious/critical 为 0、6 种浏览器环境、5,140/24,576 B gzip 客户端预算及 1,289/1,289 条可访问公式。未引入动画依赖，无需新增 ADR；按用户要求暂停在阶段 3 之前。

### 阶段 3：全局外壳（Antigravity 强制主责）

目标：统一背景、页头、页脚、主题和页面容器。

允许修改建议：

- `src/styles/tokens.css`
- `src/styles/global.css`
- `src/styles/motion.css`（新增）
- `src/components/visual/AmbientBackground.astro`（新增或由 CyberGrid 演进）
- `src/components/chrome/SiteHeader.astro`
- `src/components/chrome/SiteFooter.astro`
- `src/components/chrome/ThemeToggle.astro`
- `src/layouts/BaseLayout.astro`（只在主 Agent预先批准的 slot/契约范围）

任务：

- [ ] 实现原创背景层：网格、扫描线、局部光晕、低密度 SVG 流线；每层可独立关闭。
- [ ] 重构固定/粘性页头、品牌字标、活动项、滚动压缩状态和焦点态。
- [ ] 完成移动抽屉开合视觉、焦点管理契约和无 JS 导航兜底。
- [ ] 统一主题 token，防止 FOUC；亮色保持足够对比。
- [ ] 页脚采用轻量系统状态风格，不虚构实时数据。
- [ ] 增加 SkipLink 和稳定的主内容焦点入口。

验收：桌面/移动无溢出；键盘完整；背景不捕获指针；reduced-motion 停止循环；页面滚动无明显 paint 抖动。

阶段截图：在 `audit-screenshots/phase-3/` 保存 Header/背景/页脚的 1440×900 暗/亮、390×844 暗/亮、reduced-motion 证明；移动截图同时证明 `width <= 48rem`/`hover: none` 下扫描循环、磁吸和高成本滤镜已关闭或降级。

### 阶段 4A：首页 Hero 与首屏叙事（Antigravity 强制主责）

目标：先单独完成本轮最关键的首屏叙事，避免 Hero 与文章卡片在同一大批次中相互掩盖性能和视觉问题。

任务：

- [ ] Hero 改为大视口叙事结构，保留真实研究主题和现有链接。
- [ ] 制作原创右侧 SVG/图形资产，包含静态 fallback。
- [ ] 实现标题分段入场、局部流光和滚动提示；内容默认终态可见。
- [ ] 增加简洁公告/导览模块；不添加虚构运营数据。
- [ ] 增加研究主题轨道，并确保所有主题是现有分类/标签或明确的非链接说明。
- [ ] 验证展示字体 `font-display: swap`、回退度量和首屏 CLS；移动端不加载仅桌面需要的资产。
- [ ] 在 `audit-screenshots/phase-4a/` 保存 1440×900 暗/亮、390×844 暗/亮、reduced-motion、无 JS 首屏可见证明。

验收：Hero 与参考站有相近的赛博气质和首屏冲击力，但品牌、SVG、文案和结构具有明显原创性；无 JS 时直接显示终态；LCP/CLS、字体和资源预算通过。

### 阶段 4B：首页文章流与分页（Antigravity 强制主责）

目标：在 Hero 已独立验收后重构内容发现路径，保持静态语义与分页契约。

任务：

- [ ] 重构 PostCard、PostList、SectionHeader 和 Pagination。
- [ ] 卡片 hover 与 focus-visible 同等清晰；移动端不依赖悬浮。
- [ ] 摘要截断采用 CSS/内容策略，不通过客户端脚本测高截断。
- [ ] 卡片磁吸仅在白名单、fine pointer 和预算允许时启用；文字选择、链接点击和触控不受影响。
- [ ] 保留静态 `<a href>`、列表语义、`/page/N/` 输出及可抓取性，不用客户端无限滚动替代分页。
- [ ] 在 `audit-screenshots/phase-4b/` 保存 1440×900 暗/亮、390×844 暗/亮、reduced-motion 与键盘焦点状态。

验收：卡片和分页的 hover/focus/touch 行为一致，原分页与内容契约无回归；阶段 4A+4B 合并后的 LCP/CLS 和 bundle 预算仍通过。

### 阶段 5：聚合页与搜索（Antigravity 主责，DeepSeek 可协助普通逻辑）

目标：让全站而非只有首页使用同一视觉系统。

任务：

- [ ] 归档页时间轴、年份/月分组和分页视觉统一。
- [ ] 分类/标签索引和详情页使用统一统计面板、列表与空状态。
- [ ] 搜索框、加载、无结果、错误、结果卡片和快捷键提示完整。
- [ ] JavaScript 禁用时显示明确静态说明，并提供分类、标签、归档等替代内容入口；无 JS E2E 断言说明和替代链接可见。
- [ ] 复审 Pagefind excerpt 进入 `innerHTML` 的受控边界与消毒/来源假设，记录安全结论；状态计数使用克制的 `aria-live`，结果列表本身不在每次输入时整表重播。
- [ ] 评估移动端搜索 `autofocus` 是否会无意弹出软键盘；默认移除，除非可用性测试证明保留更好。
- [ ] 关于页和 404 采用同一背景、面板、标题和 CTA 语言。
- [ ] DeepSeek 只可协助搜索状态纯逻辑、分页工具测试等非主要可视代码。
- [ ] 主 Agent验证所有原路由和 Pagefind 行为。
- [ ] 保留现有 `data-pagefind-body` 与 JSON-LD 输出契约；只有经阶段 0.5 独立 ADR 批准的专项变更才可调整。
- [ ] 在 `audit-screenshots/phase-5/` 交付聚合页、搜索、关于和 404 的桌面/移动、暗/亮、reduced-motion 代表截图及加载/无结果/错误状态。

退出条件：所有聚合页功能、URL 和键盘行为无回归；跨页面视觉一致。

### 阶段 6：文章阅读体验（Antigravity 负责可视层，主 Agent负责核心逻辑）

目标：在不影响数学和长文阅读的前提下完成文章页视觉重构。

Antigravity 任务：

- [ ] 重构文章标题区、meta、标签、正文面板、TOC、前后篇和浮动按钮外观。
- [ ] 优化代码块、表格、引用、图片、脚注和公式容器在暗/亮主题的表现。
- [ ] 设计目录活动项、复制反馈和阅读进度的视觉状态。
- [ ] 处理 360px 长公式、长标题、长 URL 和宽表格。

主 Agent任务：

- [ ] 重构复制、阅读进度、TOC 高亮和回顶的生命周期核心；处理失败路径。
- [ ] 保证多个初始化事件不会重复绑定。
- [ ] 保证 MathJax 和 Pagefind 标记不被视觉组件破坏。
- [ ] 保留 `mjx-container[display="true"]` 的移动横向滚动与最大宽度保护；移动 TOC 继续使用原生 `details/summary` 行为，动画不得延迟切换、焦点或 reduced-motion 静态状态。
- [ ] 补足 5 篇复杂公式文章的桌面/移动视觉回归。
- [ ] 在 `audit-screenshots/phase-6/` 保存普通文章与 5 篇复杂公式样本的桌面/移动、暗/亮、reduced-motion 代表截图。

退出条件：30 篇文章构建；5 篇复杂公式文章无回归；正文在打印、无 JS、亮/暗和 reduced-motion 下可读。

### 阶段 7：高级动效与磁吸增强（主 Agent核心，Antigravity 可视实现）

目标：在核心页面稳定后再添加可独立撤销的高级效果。

任务：

- [ ] 主 Agent完成 reveal、ambient、pointer controller 的生命周期和性能保护。
- [ ] Antigravity 调整具体入场距离、时长、层级、光晕和磁吸视觉。
- [ ] 只在 fine pointer 启用磁吸光晕，触控和 reduced-motion 不初始化。
- [ ] 页面隐藏、组件离开视口时暂停环境动画。
- [ ] 用 Chrome Performance 验证无 layout thrashing、无持续 Long Task、合成层数量合理。
- [ ] 为每种高级效果提供单独 feature flag/数据属性，便于快速回退。
- [ ] 在 `audit-screenshots/phase-7/` 保存 fine-pointer 开启态、触控禁用态、reduced-motion 禁用态与各 feature flag 回退态。

退出条件：性能预算通过；关闭任一高级效果不影响布局与功能；低动态模式无非必要运动。

### 阶段 8：全量验证、文档与上线准备（主 Agent主责）

任务：

- [ ] 运行 `npm run check` 并保存结果。
- [ ] 对关键页面运行桌面/移动、暗/亮、reduced-motion、无 JS E2E。
- [ ] 运行 axe、Lighthouse、bundle audit、资源 audit、HTML 输出审计。
- [ ] 对生产 URL manifest 做本地产物全量比对。
- [ ] Chrome 人工巡检：首页、两类文章、归档、分类、标签、搜索、关于、404。
- [ ] 检查键盘、触控、缩放 200%、打印和慢网。
- [ ] 主 Agent做最终完整 diff review；DeepSeek 做发布前只读第二意见。
- [ ] 更新 README、架构文档、视觉规范、动效规范、Agent 日志和回滚文档。
- [ ] 生成 `docs/refactor/final-validation.md`，索引 `audit-screenshots/phase-*`、测试结果、性能对比、已接受偏差和回滚 commit。
- [ ] 生成上线 smoke 清单，但未经用户明确要求不自动部署。
- [ ] smoke 清单至少列出首页、分页、中文编码文章 URL、普通/公式文章、归档、分类、标签、搜索、Atom、Sitemap、robots 和 404，并为每个 URL 写明 HTTP/静态文件、canonical、核心内容、资源与控制台断言。

退出条件：最终验收清单全绿，所有 major/blocking 审查项关闭，存在可定位的回滚 commit。

## 11. 组件级工作清单

| 组件/区域          | 当前动作            | 目标                                     | 可视实现者               | 核心审查者                   |
| ------------------ | ------------------- | ---------------------------------------- | ------------------------ | ---------------------------- |
| `BaseLayout`       | 保留 SEO/主题初始化 | 加 SkipLink、全局动效契约                | Antigravity 只做可视标记 | 主 Agent                     |
| `CyberGrid`        | 评估演进/替换       | 多层可关闭 AmbientBackground             | Antigravity              | 主 Agent                     |
| `SiteHeader`       | 重构                | 固定深色、荧光活动项、滚动态、移动抽屉   | Antigravity              | 主 Agent                     |
| `ThemeToggle`      | 优化                | 无 FOUC、清晰状态、键盘与 reduced-motion | Antigravity 外观         | 主 Agent逻辑                 |
| `Hero`             | 重点重构            | 全屏叙事、原创数学 SVG、分段入场         | Antigravity              | 主 Agent                     |
| `PostCard`         | 重点重构            | 视觉分层、分类/标签/日期、焦点联动       | Antigravity              | 主 Agent                     |
| `PostList`         | 调整组合            | 响应式网格、稳定列表语义                 | Antigravity              | 主 Agent                     |
| `PaginationNav`    | 视觉收敛            | 静态链接、状态一致                       | Antigravity              | 主 Agent                     |
| `PostLayout`       | 分步重构            | 阅读优先的标题/正文/TOC                  | Antigravity 外观         | 主 Agent逻辑                 |
| `ReadingProgress`  | 核心重构            | 单 rAF、正确比例、可禁用                 | Antigravity 外观         | 主 Agent                     |
| `FloatingControls` | 优化                | 不遮挡、触控友好、键盘完整               | Antigravity 外观         | 主 Agent逻辑                 |
| Search             | 状态完善            | 加载/无结果/错误/结果一致                | Antigravity 外观         | 主 Agent + DeepSeek 普通逻辑 |
| 聚合页面           | 视觉统一            | 时间线/面板/空状态                       | Antigravity              | 主 Agent                     |

## 12. 测试与验收矩阵

### 12.1 自动化层级

1. **单元测试**：motion preference、分页、路径、标题、纯状态映射、cleanup。
2. **组件/DOM 行为 E2E**：菜单、主题、搜索、复制、回顶、目录、分页链接。
3. **无 JS E2E**：首页/文章正文与导航仍可访问。
4. **reduced-motion E2E**：根状态正确、隐藏元素立即可见、循环动画停止、光标增强未创建。
5. **视觉回归**：关键 viewport + 主题；冻结动效到稳定终态。
6. **可访问性**：axe + 键盘脚本 + 人工读序检查。
7. **性能**：Lighthouse、资源体积、Chrome trace、Core Web Vitals 目标。
8. **产物审计**：URL、canonical、资源、MathJax、Pagefind、XML endpoint。

### 12.2 必测页面

- `/`
- `/page/2/`
- 一篇普通文章
- DDPM/DDIM/SDE/ResShift/SR3 中至少 5 篇复杂公式文章
- `/archives/`
- 一个年月归档分页
- 一条包含中文字符及其百分号编码形式的历史文章 URL，断言 pathname、canonical、站内链接和生产托管访问一致
- `/categories/` 与一个多层分类详情
- `/tags/` 与一个标签分页
- `/search/`
- `/about/`
- `/404.html` 或实际 Pages 404 入口

### 12.3 交互用例

- Tab 顺序从 SkipLink 到页头、主内容、页脚合理。
- 移动菜单：打开、Escape、背景点击、选择链接、焦点返回。
- 主题：刷新保持、localStorage 不可用，以及阶段 1 已批准的首访策略；若选择系统跟随，则覆盖媒体查询变化和用户显式选择的优先级。
- 搜索：空输入、中文、英文、无结果、快速连续输入、Pagefind 加载失败。
- 复制：成功、Clipboard API 失败、快速重复点击。
- 文章目录：点击跳转、活动项、长目录、移动 details。
- 回顶：滚动阈值、键盘触发、reduced-motion 使用即时滚动。
- 页面隐藏/恢复：循环动效暂停/恢复且不重复启动。
- 搜索无 JavaScript：静态说明与分类/标签/归档替代入口可见，输入框不会伪装成可工作的客户端搜索。

### 12.4 浏览器/设备矩阵

- Chromium 最新稳定版为自动化主基线；
- Firefox/Safari 兼容性以标准 Web API 和 BrowserStack/真实设备抽查为补充；
- Windows 桌面 fine pointer；
- Android 390px 触控；
- 360px Chromium 溢出与键盘基线；
- iOS 390px Safari 抽查；
- 200% 页面缩放；
- `prefers-reduced-motion: reduce`；
- `prefers-color-scheme` 暗/亮；
- JavaScript disabled。

### 12.5 固定视觉证据协议

每个涉及主要可视界面的阶段都必须在 `audit-screenshots/phase-N/` 形成可追溯证据，而不是只在 Agent 回复中声称“看起来正确”。最低集合：

1. 桌面 1440×900：暗色、亮色；
2. 移动 390×844：暗色、亮色；
3. `prefers-reduced-motion: reduce` 的稳定终态；
4. 该阶段关键交互的 focus-visible 或状态截图；
5. 页面包含循环/入场动效时，额外提供禁用 JavaScript 或终态优先证明；
6. 截图索引记录 commit、URL、viewport、DPR、主题、运动偏好、浏览器版本、是否关闭动画和已 mask 区域。

视觉差异只用于发现问题，不代替 DOM、axe、键盘、性能和路由测试。随机噪声、时间和持续动画必须冻结或 mask；不得通过扩大阈值掩盖真实布局回归。

## 13. 风险登记与缓解

| 风险                         | 概率 | 影响 | 缓解                                              | 阻塞上线   |
| ---------------------------- | ---: | ---: | ------------------------------------------------- | ---------- |
| 视觉重构误改历史 URL/路由    |   低 | 极高 | 文件租约、路由冻结、manifest 全量审计             | 是         |
| 动效隐藏内容且脚本失败       |   中 |   高 | 终态优先、`data-motion=ready` 后才隐藏、无 JS E2E | 是         |
| 背景/滤镜导致低端设备卡顿    |   中 |   高 | 分层开关、transform/opacity、移动降级、trace      | 是         |
| 自定义光标降低可用性         |   中 |   中 | 不隐藏系统光标、fine pointer 限定、独立 flag      | 否，可移除 |
| CSS 令牌重构破坏亮色/正文    |   中 |   高 | 语义 token、页面矩阵、对比度和视觉回归            | 是         |
| Hero 素材侵权或热链失效      |   低 | 极高 | 原创资产、license 台账、禁止热链                  | 是         |
| 子 Agent 覆盖彼此改动        |   中 |   高 | 单文件租约、串行写入、主 Agent diff review        | 是         |
| Agent 产出看似可用但测试不足 |   中 |   高 | 主 Agent复跑测试、浏览器验收、无自证放行          | 是         |
| 主 Agent 成为审查和截图瓶颈  |   中 |   中 | 小 CL、固定模板、批次抽查与阶段级全矩阵分层       | 否         |
| 视觉规格与性能契约脱节       |   中 |   高 | 阶段 1 冻结、阶段 2 冲突回转、裁决后同步规格      | 是         |
| 动画截图不稳定               |   高 |   中 | 稳定终态、禁用随机、mask 动态层                   | 是         |
| 视觉增强推高 LCP/CLS         |   中 |   高 | 资源预算、尺寸占位、断点加载、Lighthouse          | 是         |
| 脚本重复初始化/内存泄漏      |   中 |   高 | 幂等 init、AbortController、生命周期测试          | 是         |
| 文章公式/代码可读性下降      |   中 | 极高 | prose 边界、5 篇公式基线、移动横向滚动            | 是         |

## 14. 回滚策略

- 每个阶段形成独立、可构建的小批次；外壳、首页、聚合页、文章页和高级动效分开。
- 高级背景、reveal、pointer 使用独立数据属性/feature flag；出现性能或可用性问题时优先关闭单项。
- UI/动效批次不改内容、URL 或部署，回滚优先只回退对应展示层提交；若阶段 0.5 批准独立基础链路专项，则按该专项 ADR 的兼容层、旧→新映射和独立回滚点处理，禁止与展示层一起回退。
- 上线前保留最后稳定 Astro 视觉版本的 tag/commit。
- 若生产出现 P0：回退到上一个稳定 commit，重新构建并部署，不在生产分支临时堆补丁。
- 回滚后保留失败截图、trace、Lighthouse、console 和审计结果用于独立修复。

## 15. 最终验收清单

### 架构与功能

- [ ] Astro SSG、Content Collections、Pagefind、MathJax 和 GitHub Pages 若保持现状则零回归；若被替换则对应 ADR、兼容清单、迁移测试和回滚全部通过。
- [ ] 30/30 文章通过；全部保留 URL 或批准后的旧→新兼容映射通过生产等价验证。
- [ ] 无新增客户端框架或未批准依赖。
- [ ] 分层、依赖方向和组件契约符合 §6。
- [ ] 动效模块幂等、可清理、可独立关闭。

### 视觉与交互

- [ ] 首页具有 AyeezBlog 风格的赛博氛围，但无复制素材和逐像素复刻。
- [ ] Header、Hero、卡片、聚合页、文章页、搜索和 404 风格统一。
- [ ] 暗/亮主题完整，360–1920px 无页面级溢出。
- [ ] hover、focus、touch 状态齐全。
- [ ] reduced-motion 下无非必要运动，内容无隐藏。

### 质量

- [ ] `npm run check` 全绿。
- [ ] axe serious/critical 为 0。
- [ ] Lighthouse 和 Core Web Vitals 达到 §3.4。
- [ ] 新增 JS/资源体积在预算内。
- [ ] 视觉回归、无 JS、键盘、移动和公式用例通过。

### Agent 治理

- [ ] 所有主要可视界面均由 Antigravity 子 Agent实施。
- [ ] DeepSeek 子 Agent只在允许范围工作。
- [ ] 每批子 Agent产出均有主 Agent审查记录。
- [ ] 所有 blocking/major 项关闭。
- [ ] 最终由主 Agent明确放行。

## 16. DeepSeek 双高精度审查记录

### 16.0 Antigravity 前置可实施性审查（已完成）

运行事实：2026-08-21，Antigravity CLI 1.1.17；工作区 `D:/Blog_file`；Google AI Pro 已登录；模型 `Gemini 3.7 Flash`；思考程度 `High`；只读审查，未授权其修改计划或代码。

审查原始结论：**有条件可执行**。主 Agent 对建议逐项核验后完成以下吸收：

- 把终态优先的 `html[data-motion="ready"]` 契约和无 JS 内容可见测试提升为阶段 2 阻塞条件；
- 在主要可视文件交给 Antigravity 前，由主 Agent先拆离与视觉模板混写的客户端核心行为并做等价回归，消除共享 `.astro` 文件的所有权冲突；
- 将原阶段 4 拆成 4A Hero/首屏和 4B 卡片/文章流/分页两个可独立 review 的批次；
- 将 `docs/refactor/design-specs.md`、WCAG 对比度表、断点矩阵、原创 SVG 规格和组件状态设为阶段 1 固定交付；
- 为每个 UI 阶段增加桌面/移动、暗/亮、reduced-motion 的固定截图证据；
- 增加 `width <= 48rem`/`hover: none` 的扫描线、辉光、backdrop filter 和光标降级规则；
- 增加展示字体防 CLS、磁吸白名单、移动 TOC 原生行为、公式容器、`data-pagefind-body` 和 JSON-LD 契约保护；
- 明确基础链路专项与 UI 批次必须拆为独立 ADR/变更单元。

Antigravity 启动能力已经实测，不是阻塞项。它提出的历史 PATH 可见性只作为运行环境事实处理，不进入风险或放行阻塞清单。

### 16.1 固定配置

- 工具：`dsh --profile web`
- 工作区：`Blog_file`
- 模型：`DeepSeek-V4-Flash`
- 思考程度：`Max`
- 权限：`Read Only`
- 方式：Chrome 中运行两次独立会话；不允许审查 Agent 修改计划或工作区。

### 16.2 第一轮：架构、编码与项目规则审查

审查重点：边界是否忠于“Hexo → Astro 已完成”的事实；分层和依赖方向；Google 规则是否被具体化；Agent 权限是否无冲突；阶段依赖、风险、回滚和完成定义是否可执行。

实际结论：**有条件可执行；无 Blocking，6 项 Major**。模型核实了 Astro 7、30 篇文章、Pagefind、MathJax、GitHub Pages、路由和迁移文档等基线。第一会话曾自行调用一次 Pwsh 做只读 git 核验；主 Agent 当场停止生成并明确该行为违反 Git Bash 规则，其输出不计入任何合规证据。会话随后只基于已经完成的 Read/Glob/Grep 给出结论。

主 Agent裁决与吸收：

1. **motion token 所有权：接受并细化。** 主 Agent只管语义契约、取值边界、生命周期和安全默认；视觉数值由 Antigravity 冻结和落值。
2. **脚本拆分例外：接受。** §9.3 已加入只允许主 Agent 修改 `<script>` 和最薄桥接、禁止触碰可视标记/样式的基础设施例外。
3. **Git Bash 规则：拒绝模型的放宽建议。** 用户规则优先；现改为所有 Agent 的项目终端命令都必须走 Git Bash。无法选择 Git Bash 的审查器不得调用 Shell，由主 Agent代为核验。
4. **DSH Read Only：接受但不开放产物写入。** Read Only 审查只读源码和现有证据；会写缓存/产物的测试统一由主 Agent在 Git Bash 中复跑。
5. **阶段 1/2 衔接：接受并修正为双向冲突回转。** 视觉规格与核心契约冲突时暂停派工，由 Antigravity 给替代方案、主 Agent裁决并同步规格，不简单规定任一方永远优先。
6. **阶段 0.5 专项门：接受。** 已补独立派工、DoD、租约、回滚和暂停 UI 的规则。

同时吸收其可验证的 Minor：统一 `tokens.css`/`motion.css` 职责、强化大 CL 处置、增加中文 URL 测试、阶段 2 固化质量阈值、检查 `src/pages/dev/math-spike.astro` 与未跟踪 `.deploy_git/`、记录 `astro-public/` 资源目录、补主 Agent瓶颈与规格脱节风险。其“仓库不存在 `src/data/`”判断经主 Agent核对为错误，实际存在 `src/data/site.ts`，未采纳。

### 16.3 第二轮：UI、动效、测试与执行可行性审查

审查重点：Ayeez 风格映射是否完整且避免复制；动效终态、reduced-motion、触控和性能；阶段拆分是否可验收；视觉测试和浏览器矩阵；上线与回滚。

实际结论：**有条件可执行；2 项计划门定义 Blocking、3 项 Major**。第二会话为全新独立会话，明确禁止 Shell/Pwsh/Bash，只使用 Read/Glob/Grep；它核对了现有 Playwright、BaseLayout、Search、MathJax、字体和资源结构，没有参考第一轮结论。

主 Agent裁决与吸收：

1. **公式页性能门：部分接受。** 审查模型把 `/dev/math-spike/` 的客户端 MathJax 误认为生产文章链路；主 Agent依据已接受 ADR、内容配置、E2E 和产物审计纠正为构建期 MathJax。§3.4 仍保留分页面测量和固定移动节流参数，但不把客户端 vendor/typeset 计入生产文章预算；公式页额外核对静态 HTML/CSS、`mjx-container`、溢出与 CLS。
2. **无 JS 搜索降级：接受。** 阶段 5 增加静态说明和分类/标签/归档替代入口，并进入 E2E。
3. **首访主题策略：接受。** 阶段 1 必须明确“暗色品牌默认”或“首次跟随系统”，不再保留“若支持”的模糊状态。
4. **截图职责：接受。** Antigravity 交付视觉断言，Chrome 固定证据由主 Agent统一采集与署名。
5. **移动/触控自动化：接受。** 阶段 2 明确 desktop、390 touch、360 overflow、reduced-motion、无 JS、200% zoom 项目和 iOS 抽查责任。
6. **其他 Minor：接受。** 补字体许可证与度量、Pagefind excerpt `innerHTML` 复审计、脚本拆分前行为基线、搜索 `aria-live`/`autofocus`、旧截图归档和具体 smoke URL 断言。

两项 Blocking 均属于计划定义缺口，已经通过上述修订关闭，不是当前代码缺陷已被“假定修复”。代码层状态仍需阶段 0/2/5 的实际测试证明。

### 16.4 放行判断

当前放行判断：**允许启动阶段 0；尚不允许跳过阶段 0 直接进入阶段 1。** 两轮独立高精度审查已完成，进入阶段 1 前必须同时满足：

1. 当前全量质量门禁绿色或已有明确的基线缺陷记录；
2. URL/内容/数学/搜索/部署边界冻结；
3. [已满足] `agy` 已从 PowerShell 在 `D:\Blog_file` 成功启动，后续独立项目命令在 Git Bash 中执行，并确认 `Gemini 3.7 Flash` + `High`；
4. [已满足] DSH 已在两个独立会话复现 `DeepSeek-V4-Flash` + `Max` + `Read Only`；
5. 每次派工任务书重申文件租约、Git Bash 规则、禁止项和主 Agent最终审批权；
6. 阶段 0 中发现的基线缺陷均被修复或形成有负责人、门槛和回滚的明确记录。

## 17. 首个实施批次建议

首批只执行阶段 0，不改 UI。建议交付：

1. `docs/refactor/baseline.md`；
2. 桌面/移动、暗/亮关键页面截图；
3. URL/内容/公式/搜索冻结清单；
4. Lighthouse、axe、bundle、资源与动画 trace 基线；
5. DSH 与 Antigravity 的可复现启动记录；
6. 视觉改造分支和回滚点。

只有这六项通过，才把阶段 1 的视觉规格派给 Antigravity 子 Agent。这样可以保证本轮始终是一场可测量、可分批、可回滚的展示层重构，而不是对已经完成的 Astro 平台迁移进行二次扰动。
