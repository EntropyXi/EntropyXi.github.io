# Hexo 到 Astro 的详细重构计划

本文定义 EntropyXi 个人博客从 Hexo 8 + NexT 迁移到 Astro 的完整实施方案。迁移目标是保留全部内容、历史 URL、数学公式、评论与 GitHub Pages 部署能力，同时建立一套适合赛博动态视觉的新前端架构。

计划版本：2026-08-20 双审修订稿  
项目根目录：`D:\Blog_file`  
当前生产地址：`https://entropyxi.github.io`  
目标渲染模式：Astro 静态站点生成（SSG）  
目标托管平台：GitHub Pages

## 1. 决策摘要

### 1.1 最终技术决策

- 使用 Astro 作为唯一站点生成器，移除 Hexo、NexT、Pandoc 渲染链和 Stylus。
- 使用 Astro Content Collections 管理 Markdown 文章，并通过严格 schema 校验 frontmatter。
- 默认只生成静态 HTML；不引入 Node 服务、数据库、SSR adapter 或运行时内容 API。
- 默认使用 Astro 组件、语义 HTML、CSS 和少量原生 TypeScript 实现交互。
- 初始迁移阶段不引入 React、Vue、Svelte 等 UI 框架。只有当原生脚本明显增加复杂度，并且通过架构决策记录（ADR）后，才允许引入单一框架集成。
- 数学渲染先以“兼容当前 30 篇文章”为最高优先级，通过技术验证后在构建期 MathJax 和按页加载 MathJax 两种方案中选定一种；不得在未完成全量公式回归前改用纯 KaTeX。
- 使用 Pagefind 在构建完成后生成中文静态搜索索引。
- 使用 Giscus 继续提供评论，保持现有仓库、分类和 `pathname` 映射。
- 使用 GitHub Actions 构建并部署 `dist/`；共存期保留 Hexo 的 `public/` 构建目录，并把 Astro 静态源目录显式配置为 `astro-public/`，避免两套工具互相覆盖。
- 迁移和视觉重构分两条验收线：先证明内容与 URL 等价，再启用完整赛博视觉。

### 1.2 为什么不在本次迁移中保留 NexT

NexT 的模板、Stylus 变量、注入点和运行时脚本是 Hexo 主题体系的一部分。继续保留其结构会让 Astro 只充当外壳，导致双重模板体系、样式耦合和难以测试的脚本生命周期。本次迁移应重建视觉组件，但不复制 NexT 的内部实现。

迁移中间态也不复刻 NexT 动画：阶段 1 至阶段 4 只提供无动画依赖、可稳定阅读的静态页面，优先验证内容、URL、公式、目录和响应式排版。只有内容等价验收通过后，阶段 5 才单独实现类 Ayeez 的视觉语言；NexT 截图仅用于确认信息与布局未丢失，不作为最终像素复刻目标。

### 1.3 成功标准

迁移只有在以下条件全部满足时才算完成：

1. 30 篇文章全部生成成功，且每篇都有标题、描述、日期、标签、分类和规范 URL。
2. 旧站全部文章 URL 在本地生产产物和部署后的生产域名均返回 200，路径、编码形式与尾斜杠保持不变。
3. 数学公式在浏览器完成渲染后没有可见的 `$$`、`\(...\)`、`\[...\]` 或破损环境；自动检查按最终选定的构建期或客户端方案分别定义。
4. 首页、文章、归档、分类、标签、关于、搜索、RSS、Sitemap 和 404 页面全部可用。
5. Giscus 对相同 pathname 继续挂载，不创建不必要的新讨论线程。
6. 键盘、触屏、窄屏和 `prefers-reduced-motion` 场景均可操作。
7. CI 中类型检查、Lint、格式检查、内容审计、单元测试、构建验证和端到端测试全部通过。
8. GitHub Pages 成功部署，生产域名、canonical、RSS 和 Sitemap 指向正确地址。
9. README、架构文档、内容写作指南和迁移记录与代码同步更新。
10. 旧 Hexo 实现只在迁移确认后的独立清理变更中移除，清理前具备可验证的回滚点。

## 2. 当前项目基线

### 2.1 已确认的现状

- 生成器：Hexo 8.1.1。
- 主题：NexT 8.27.0，Pisces 双栏布局。
- Markdown 渲染：Pandoc，经 `hexo-renderer-pandoc` 调用。
- 数学：MathJax，文章通过 `mathjax` frontmatter 控制。
- 评论：Giscus，映射方式为 `pathname`。
- 搜索：`hexo-generator-searchdb` 生成本地搜索 XML。
- 部署：`source` 分支触发 GitHub Actions，Node 20 + Pandoc 构建并部署 `public/`。
- 文章：30 篇 Markdown。
- 内容目录：`source/_posts/`。
- 独立页面：关于、分类、标签。
- 非 Markdown 静态资源：`source/images/eigenvalue-error.png` 和文章目录内的 `体系框架图.jpg`。
- 隐藏内容：`source/_posts/.obsidian/` 必须排除，不能进入内容集合或构建产物。

### 2.2 迁移前已发现的问题

以下问题必须显式处理，不得在迁移时静默忽略：

- 所有文章都包含 `<!-- more -->`。新站使用 `description` 作为摘要后，该标记应由迁移脚本删除或忽略。
- `mathjax` 同时存在布尔值 `true` 和字符串 `"true"`，迁移后统一为布尔字段 `math: true`。
- 至少 6 处 Obsidian Wiki 图片引用（`![[...]]`）没有对应资源文件，当前不能作为有效 Markdown 图片发布。
- 一篇文章使用带内联 `style` 的原生 `<img>`。迁移后应改成普通 Markdown 图片或受控图片组件，不保留文章级表现样式。
- `体系框架图.jpg` 存在于文章目录，但当前未发现有效引用，需要确认是补回文章、迁移到资源目录还是作为未使用资源删除。
- 公式包含 `\tag`、`equation`、`align`、`aligned`、`cases`、矩阵、`\mathbb`、`\boldsymbol` 等高级语法；不能只用少量示例判断数学兼容性。
- 现有永久链接由日期和 Markdown 在 `_posts` 下的相对路径组成，包含中文、空格、全角标点和多层目录。不能简单改成 ASCII slug。
- 当前构建验证针对 `public/` 和 Hexo HTML 结构，迁移后需要重写为 `dist/` 与 Astro 结构。

### 2.3 必须冻结的行为

- 生产站根地址：`https://entropyxi.github.io/`。
- 文章路径格式：`/{YYYY}/{MM}/{DD}/{原 _posts 相对路径}/`。
- 全站尾斜杠策略：始终保留。
- 语言：`zh-CN`。
- 时区：明确配置为 `Asia/Shanghai`，不得依赖 CI 主机时区。
- 首页文章排序：按发布日期倒序。
- RSS 路径：优先继续使用 `/atom.xml`；如同时提供 `/rss.xml`，必须保留 `/atom.xml` 兼容入口。
- Sitemap 路径：继续冻结为 `/sitemap.xml`。由于 Astro 官方 Sitemap 集成默认生成 `/sitemap-index.xml` 与分片文件，本项目为不足 50,000 URL 的站点实现单文件 `src/pages/sitemap.xml.ts`，不静默改变旧路径。
- 评论映射：Giscus `pathname`。
- 外链默认新窗口行为不得通过全局脚本粗暴改写；应在组件或 Markdown 渲染规则中受控实现并添加 `rel="noopener noreferrer"`。

## 3. 项目总体规则

本节是项目级硬规则。除非 ADR 明确说明原因、替代方案和回滚方法，否则实施过程中不得偏离。

### 3.1 范围规则

- 第一目标是生成器迁移和行为等价；视觉重构不能阻塞内容正确性验证。
- 不在同一个提交中同时进行大规模格式化、内容改写、生成器迁移和视觉功能开发。
- 不引入后台管理、数据库、用户账户、运行时 API 或服务器端评论聚合。
- 不顺便改写文章观点、公式、数据或引用；内容修复必须单独记录。
- 不复制 AyeezBlog 的作者头像、品牌文案、私人链接或受第三方权利约束的图片。
- 若复用 AyeezBlog 的 Apache 2.0 代码，应保留许可证与必要版权说明，并在 `THIRD_PARTY_NOTICES.md` 记录来源和修改范围。

### 3.2 简洁性规则

- 优先使用 Astro 静态组件；交互必须证明静态 HTML + CSS 无法合理完成。
- 优先使用平台标准 API，不使用非标准浏览器 API，不修改内建原型，不使用 `eval` 或 `new Function`。
- 每新增一个生产依赖，PR 描述必须说明用途、替代方案、包体或构建影响和维护风险。
- 不为只有一次调用的简单逻辑建立抽象层；重复三次或存在明确领域概念时再提取。
- 不创建“万能组件”“全局工具箱”或无边界的 `utils.ts`；工具按领域放置。
- 页面必须在 JavaScript 加载失败时仍能阅读和导航；搜索、菜单增强和动画可以降级。

### 3.3 变更管理规则

- 每个提交只处理一个可描述的主题，并保持可构建或明确标记为迁移脚手架提交。
- 重构提交与行为变更提交分开。
- 生成文件不得与源文件混杂提交；`dist/`、`.astro/`、Pagefind 生成索引和测试报告必须忽略。
- 文档与代码在同一变更中更新。失效文档应修正或删除，不得保留误导性说明。
- 所有破坏 URL、内容 schema、公共组件接口或部署路径的决策必须写 ADR。
- 在删除 Hexo 前创建可恢复标签或分支，并保存最终 Hexo 构建的 URL 清单和验证报告。

### 3.4 质量门规则

- 禁止以 `@ts-ignore`、`@ts-nocheck`、宽泛 `any` 或关闭 Lint 规则逃避问题。
- `@ts-expect-error` 仅允许在专门验证错误输入的测试中使用，并必须说明预期错误。
- 所有 CI 警告应视作待处理事项；已知例外必须有注释、负责人和移除条件。
- 组件不得依赖 DOM 查询猜测全局结构；交互脚本只操作自身组件根节点或显式 data 属性。
- 任何修复都应先增加能复现问题的自动化检查，无法自动化时记录可复现的人工验证步骤。

### 3.5 安全与隐私规则

- 不在仓库、前端 bundle、日志或 GitHub Actions 中放置私钥、令牌或私人 API key。
- 所有第三方脚本必须使用 HTTPS，记录用途、加载页面、数据影响和移除方式。
- 第三方前端脚本优先自托管；若使用 CDN，必须固定版本、记录完整性与失效降级方案，不得以 CDN 可用作为正文可读的前提。
- 默认不增加用户追踪、指纹、广告或不可审计统计脚本。
- 使用 `set:html` 或等价原始 HTML 注入前，必须确认内容只来自受信任的仓库 Markdown，并在架构文档中说明信任边界。
- 外链必须防止 opener 攻击；用户生成内容不得直接进入页面 HTML。
- Giscus 是唯一允许的用户输入入口，博客自身不处理评论正文。

### 3.6 可访问性规则

- 语义元素优先于带点击事件的 `div`；链接使用 `<a>`，操作使用 `<button>`。
- 所有交互必须支持键盘，具备可见焦点，不得只依赖 hover。
- 每页只能有一个主 `h1`，标题层级不得跳级。
- 图片必须有准确 `alt`；纯装饰图片使用空 `alt` 或 `aria-hidden="true"`。
- 动画、流光、磁吸光标和视图切换必须尊重 `prefers-reduced-motion`。
- 文字与背景对比度、焦点对比度和触控目标尺寸以 WCAG 2.2 AA 为最低目标。
- 自定义搜索弹窗、移动菜单和目录抽屉必须处理焦点进入、焦点返回、Escape 关闭和背景滚动锁定。

### 3.7 性能规则

- 默认不为静态组件发送客户端 JavaScript。
- 首屏赛博背景优先 CSS/SVG，避免持续运行的大面积 Canvas；必要时限制帧率并在后台标签页暂停。
- 动画只使用 `transform` 和 `opacity` 等合成友好属性，避免高频布局抖动。
- 图片必须声明宽高，非首屏图片默认懒加载，封面使用合适尺寸与格式。
- 字体优先系统 CJK 字体栈；如使用自托管字体，仅包含需要的字重并设置 `font-display: swap`。
- 首屏不预加载搜索索引、评论脚本或非当前文章资源。
- 性能预算使用固定环境的 Lighthouse CI 连续运行 3 次取中位数：首页和代表文章页 Performance 分数不低于 90，LCP 不高于 2.5 秒、CLS 不高于 0.1、TBT 不高于 200 毫秒。生产存在足够真实用户数据时，INP 的 75 分位目标不高于 200 毫秒；实验室 Lighthouse 不伪装成 INP 测量。

## 4. Google 规范基线与项目化取舍

### 4.1 采用的公开规范

- Google TypeScript Style Guide：<https://google.github.io/styleguide/tsguide.html>
- Google HTML/CSS Style Guide：<https://google.github.io/styleguide/htmlcssguide.html>
- Google Markdown Style Guide：<https://google.github.io/styleguide/docguide/style.html>
- Google Documentation Best Practices：<https://google.github.io/styleguide/docguide/best_practices.html>
- Google Developer Documentation Style Guide：<https://developers.google.com/style/>
- Google Engineering Practices - Code Review：<https://google.github.io/eng-practices/review/>
- Astro TypeScript 指南：<https://docs.astro.build/en/guides/typescript/>
- Astro 内容集合指南：<https://docs.astro.build/en/guides/content-collections/>
- Astro 测试指南：<https://docs.astro.build/en/guides/testing/>
- Astro 配置参考（含可配置 `publicDir`）：<https://docs.astro.build/en/reference/configuration-reference/#publicdir>
- Astro Sitemap 集成：<https://docs.astro.build/en/guides/integrations-guide/sitemap/>
- Pagefind 多语言与中文分词：<https://pagefind.app/docs/multilingual/>

### 4.2 取舍原则

Google 规范面向其内部工程环境，不能机械复制。本项目采用以下优先级：

1. 本文明确的项目规则。
2. Astro 官方约束与当前工具自动格式化结果。
3. Google TypeScript、HTML/CSS、Markdown 和工程审查规范中可执行的部分。
4. 当规则冲突时，以可访问性、正确性、可维护性和自动化一致性为先。

明确不采用 Google 规范中与本项目生态或规模不匹配的部分：不强制每个公开 API 写 JSDoc、不采用 `snake_case` 文件名、不要求所有常量使用 `UPPER_SNAKE_CASE`、不为每个源文件添加许可证头。原因是 Astro/现代 TypeScript 生态惯例与单人博客维护成本；项目自己的 `kebab-case`、文档和许可证规则优先。

### 4.3 自动化优先

Google 工程实践强调让审查者关注设计和正确性，而不是争论空格。项目中可机械判断的规则必须由工具执行：

- Prettier + Astro 插件负责 `.astro`、TypeScript、JSON、YAML、CSS 和工程 Markdown 的基础格式；迁移文章正文由 `.prettierignore` 排除，避免无关内容改写。
- ESLint + TypeScript ESLint + Astro 插件负责程序规则。
- Stylelint 配合 Astro/HTML 语法支持，覆盖 `src/**/*.{css,astro}` 中的独立 CSS 和组件 `<style>`。
- `astro check` 负责 `.astro` 与 TypeScript 类型检查。
- 内容审计脚本负责 frontmatter、URL、资源、日期和 Markdown 禁止模式。
- Vitest 负责纯函数与组件输出测试。
- Playwright 负责生产构建端到端验证。
- axe-core 负责关键页面的自动可访问性扫描；Lighthouse CI 负责固定环境性能回归；依赖与密钥扫描负责安全门禁。
- 人工审查只处理设计、复杂度、用户体验、命名、注释、测试充分性和文档正确性。

## 5. 目标架构

### 5.1 目录结构

```text
Blog_file/
├── .github/
│   └── workflows/
│       └── deploy.yml
├── docs/
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── content-model.md
│   │   ├── rendering-and-interactivity.md
│   │   └── adr/
│   ├── contributing/
│   │   ├── coding-style.md
│   │   ├── writing-content.md
│   │   └── testing.md
│   └── superpowers/plans/
├── astro-public/             # astro.config.ts 的 publicDir；与 Hexo public/ 隔离
│   ├── favicon/
│   ├── images/
│   ├── robots.txt
│   └── CNAME                 # 仅使用自定义域名时存在
├── scripts/
│   ├── audit-content.ts
│   ├── audit-output.ts
│   ├── capture-url-manifest.ts
│   ├── migrate-content.ts
│   └── new-post.ts
├── src/
│   ├── components/
│   │   ├── chrome/           # 页头、页脚、导航、菜单
│   │   ├── content/          # 文章卡片、元信息、目录、分页
│   │   ├── feedback/         # Giscus、加载/错误状态
│   │   ├── search/           # 搜索入口和弹窗
│   │   └── visual/           # 赛博背景、流光、装饰线
│   ├── content/
│   │   └── blog/             # 迁移后的 Markdown 与文章资源
│   ├── data/
│   │   ├── navigation.ts
│   │   ├── site.ts
│   │   └── social.ts
│   ├── layouts/
│   │   ├── BaseLayout.astro
│   │   ├── ListingLayout.astro
│   │   └── PostLayout.astro
│   ├── lib/
│   │   ├── content/
│   │   ├── routing/
│   │   ├── seo/
│   │   └── shared/
│   ├── pages/
│   │   ├── [...permalink].astro
│   │   ├── 404.astro
│   │   ├── about.astro
│   │   ├── archives/         # 含旧站实际存在的分页/年月路由
│   │   ├── atom.xml.ts
│   │   ├── categories/       # 含详情与分页 catch-all
│   │   ├── index.astro
│   │   ├── page/[page].astro
│   │   ├── search.astro
│   │   ├── search.xml.ts     # 旧 hexo-generator-searchdb URL 兼容入口
│   │   ├── sitemap.xml.ts
│   │   └── tags/             # 含详情与分页 catch-all
│   ├── styles/
│   │   ├── global.css
│   │   ├── prose.css
│   │   ├── tokens.css
│   │   └── utilities.css
│   ├── content.config.ts
│   └── env.d.ts
├── tests/
│   ├── e2e/
│   ├── fixtures/
│   ├── unit/
│   └── visual/
├── astro.config.ts
├── eslint.config.js
├── package.json
├── pagefind.yml
├── playwright.config.ts
├── prettier.config.mjs
├── stylelint.config.mjs
├── tsconfig.json
└── vitest.config.ts
```

### 5.2 分层规则

#### 页面层 `src/pages`

- 只负责路由、构建期数据查询、页面级 SEO 参数与 layout 组合。
- 不包含复杂样式和可复用业务逻辑。
- 动态路由必须通过 `getStaticPaths()` 在构建期完全展开。
- 页面不得直接访问散落的环境变量；统一从站点配置模块读取。

#### 布局层 `src/layouts`

- 负责文档骨架、公共 head、页头页脚、主内容区域和结构插槽。
- `BaseLayout` 是唯一输出 `<html>`、`<head>` 和 `<body>` 的组件。
- Layout 不查询文章集合；数据由页面层以显式 props 传入。

#### 组件层 `src/components`

- 组件按领域分组，不按 Astro/Vue/JS 文件类型分组。
- 组件只接收完成渲染所需的最小 props。
- 组件不得直接读取全局内容集合或修改全局状态。
- 视觉装饰组件不得影响语义文档顺序。
- 客户端交互组件必须有无 JavaScript 降级状态。

#### 领域逻辑层 `src/lib`

- 只包含无 UI 的可测试函数，如排序、分组、URL 构建、SEO 元数据与 TOC 处理。
- `content`、`routing`、`seo` 之间不得形成循环依赖。
- `shared` 只允许真正跨领域且语义明确的函数；禁止成为杂物目录。

#### 内容层 `src/content`

- Markdown 只存内容与元数据，不存页面布局 CSS、脚本或第三方 iframe 逻辑。
- 内容文件不得导入组件；如未来需要 MDX，必须先写 ADR 并限制可用组件白名单。
- 为保持历史资源 URL，本次迁移把已发布文章图片复制到 `astro-public/<旧资源 pathname>` 并重写引用，输出 URL 必须与旧站一致。
- 新文章可以使用 Astro 支持的同目录图片资产；它不能反向改变已冻结的历史图片 URL。全站复用资源放 `astro-public/images/`。

### 5.3 依赖方向

```text
pages -> layouts -> components
  |          |           |
  +--------> lib <-------+
  |
  +--------> content/data

content/data 不得反向依赖 pages、layouts 或 components。
lib 不得依赖 pages 或 layouts。
```

### 5.4 渲染与交互边界

- 首页文章卡片、归档、标签、分类、目录和 SEO 全部构建期生成。
- 移动导航、搜索弹窗、复制代码、回到顶部、阅读进度和可选阅读模式使用原生 TypeScript 渐进增强。
- Giscus 仅在文章页滚动接近评论区域或用户主动展开评论时加载。
- Pagefind 索引只在搜索输入获得焦点或打开搜索页时加载。
- 磁吸光标仅在精细指针且未启用 reduced motion 时初始化；触屏与键盘环境完全不加载。
- 视图过渡应优先使用浏览器原生跨文档 View Transitions；如使用 Astro ClientRouter，必须验证脚本重复初始化、Giscus、滚动恢复和无 JS 导航。

### 5.5 面向对象、模块化与复用规则

- 以高内聚、低耦合、单一职责、依赖倒置和组合优先为核心，不为套用面向对象形式而制造无状态 class。
- 页面、布局和组件依赖领域接口与只读数据，不直接依赖文件系统、环境变量、第三方 SDK 或具体存储实现。
- 领域计算写成无 UI、无 I/O 的纯模块；文件读取、Astro Content Collections、浏览器 API、Giscus 和 Pagefind 放在边界适配层。
- 模块只导出调用方需要的最小契约；内部辅助函数默认不导出，不允许跨层访问内部文件形成隐式公共 API。
- 有可替换策略时先定义窄接口，例如 `MathRenderer`、`SearchAdapter`、`ContentRepository`；生产实现和测试替身都遵守同一契约。
- class 只用于确有实例状态、资源生命周期或多实现多态的对象；构造函数不得执行网络、DOM 或文件 I/O，依赖通过构造参数显式注入。
- 禁止深继承树、静态全局服务定位器和可变单例；优先小对象组合、只读值对象和显式工厂函数。
- 可复用不等于提前抽象：出现两个以上稳定调用方或明确领域边界后才提取公共模块，重复但仍在变化的代码允许暂时并置。
- 每个公共模块必须能脱离页面独立测试；测试验证公开行为和契约，不锁死内部实现。
- 跨模块数据使用不可变、可序列化的领域类型；禁止在多个层之间传递 Astro 页面上下文、DOM 节点或无约束对象。

## 6. 内容模型与 URL 规则

### 6.1 文章 schema

目标 frontmatter 逻辑模型：

```yaml
---
title: DDPM
description: 系统推导去噪扩散概率模型的核心原理……
date: 2026-03-15T21:40:00+08:00
updated: 2026-03-15T21:40:00+08:00
tags:
  - 深度学习
  - 扩散模型
categories:
  - 深度学习
  - 流匹配与扩散模型
permalink: 2026/03/15/深度学习/流匹配与扩散模型/DDPM
math: true
draft: false
cover: null
---
```

Schema 规则：

- `title`：必填非空字符串。
- `description`：必填，去除首尾空白后非空，用于卡片、SEO 和 RSS。
- `date`：必填，迁移时显式补 `+08:00`，避免 CI 时区漂移。
- `updated`：可选；缺失时等于 `date`，不得从文件 mtime 推导。
- `tags`：必填字符串数组，去重后至少一项。
- `categories`：必填字符串数组，保留从大类到子类的顺序。
- `permalink`：必填且全站唯一，不含首尾斜杠，由迁移脚本冻结。
- `math`：严格布尔值，默认 `false`。
- `draft`：严格布尔值，默认 `false`；生产构建排除草稿。
- `cover`：可选；只允许站内资源或 HTTPS URL，并提供对应 alt 数据策略。
- 未在 schema 定义的字段默认报错，避免拼写错误静默进入生产。

### 6.2 URL 生成规则

- 文章路由通过 catch-all 静态路由读取 `permalink`。
- 输出 pathname 始终为 `/${permalink}/`。
- `permalink` 一旦发布不得因改标题、移动文件或调整分类而改变。
- 新文章由 `npm run new:post -- --title "标题" --category "分类"` 创建并冻结 permalink。
- 分类和标签显示名使用原始中文；路由 slug 必须经过统一、可逆或显式映射，不允许每个页面自行编码。
- 生成 `tests/fixtures/legacy-urls.json`，保存迁移前全部 HTML URL。
- manifest 同时冻结首页分页、归档分页/年月页、分类详情及分页、标签详情及分页、静态资源和旧搜索 XML 的实际 URL；聚合页 slug 以旧产物为准，不重新猜测。
- 构建验证逐项确认清单中的路径存在，并检查 canonical 与 pathname 一致；上线后用同一清单全量检查生产 URL。
- 如未来主动修改 URL，必须生成永久重定向；GitHub Pages 无服务端重定向时使用保留旧 HTML 页面加 canonical 和即时可访问链接，不使用纯 JavaScript 空白跳转。

### 6.3 摘要与排序规则

- 首页卡片摘要只使用 `description`，不从正文截断。
- `<!-- more -->` 不再具有运行时意义，迁移脚本从新内容中移除。
- 文章按 `date` 倒序；日期相同则按 `permalink` 做稳定排序。
- 归档按年月分组；分类按 `categories` 每一级生成聚合；标签按精确字符串聚合。

### 6.4 Markdown 规则

- 使用 UTF-8 和 LF；仓库用 `.gitattributes` 固定文本行尾。
- 只使用 ATX 标题（`#`），正文从 `##` 开始，文章标题由 layout 输出。
- 标题层级连续，每个页面只有 layout 输出的一个 `h1`。
- 代码块必须使用 fenced code block 并声明语言。
- 优先 Markdown，避免内联 HTML；确需 HTML 时只允许白名单标签且不得含 `style` 或脚本事件属性。
- 图片使用标准 Markdown 或受控组件生成的语义 HTML，不使用 Obsidian Wiki 图片语法。
- 链接文本必须说明目标，不使用孤立的“点击这里”。
- 数学块前后保留空行；禁止把公式放进 HTML 标签属性。
- `src/content/blog/**` 写入 `.prettierignore`，不让 Prettier 重排历史正文；内容审计负责结构规则。迁移哈希在统一 LF、移除 frontmatter 与 `<!-- more -->` 后比较，任何其他正文差异都阻塞迁移并输出结构化差异报告。

## 7. 编码规则

### 7.1 TypeScript 规则

- 使用 `astro/tsconfigs/strictest`；若第三方类型导致阻塞，可先用 `strict`，但必须记录回到 `strictest` 的条件。
- 使用 ESM，禁止 CommonJS 新代码。
- 默认 `const`，确需重新赋值时使用 `let`，禁止 `var`。
- 每条声明只声明一个变量，变量尽量靠近首次使用处。
- 使用显式 `import type` 与 `export type`。
- 公共函数边界、组件 props、配置对象和领域模型必须有明确类型。
- 对局部明显字面量允许类型推断，禁止冗余的 `const enabled: boolean = true`。
- 使用 `unknown` 接收未经验证的外部数据，并通过 schema 或类型守卫收窄。
- 禁止非空断言作为常规手段；只有前置条件已由同一函数证明时可使用，并加简短原因。
- 禁止 TypeScript `enum`，使用 `as const` 对象或字面量联合。
- 禁止 namespace、装饰器、动态代码求值和修改全局对象。
- 优先接口描述对象结构；联合类型、函数类型和映射使用 `type`，避免复杂条件类型。
- 异步函数必须处理失败路径；不允许悬空 Promise。
- 错误信息必须包含操作与上下文，但不得包含密钥或用户敏感数据。
- 注释解释“为什么”和约束，不逐行复述代码。
- 遵守 SOLID 中适用于本项目的部分：单一职责、开放封闭、里氏替换、接口隔离和依赖倒置；发现规则与 Astro 的静态组件模型冲突时，以简单、可测试的组合设计为准并记录 ADR。
- 领域接口按能力命名且保持窄小；实现类不得在名称中无意义添加 `Manager`、`Helper`、`Processor` 或 `Util`。
- 公共方法不暴露可变内部集合；返回只读数组、只读对象或明确值对象。
- 一个文件只承载一个主要领域概念；相关的小型类型可同文件共置，避免“一类一文件”的机械拆分。

### 7.2 命名规则

- Astro 组件：`PascalCase.astro`。
- TypeScript 模块：`kebab-case.ts`；测试使用同名 `.test.ts`。
- 变量与函数：`camelCase`。
- 类型与接口：`PascalCase`，不使用 `I` 前缀。
- 常量：模块内不可变值仍用 `camelCase`；真正跨模块常量可用 `UPPER_SNAKE_CASE`。
- 布尔值使用 `is`、`has`、`can`、`should` 前缀。
- 事件处理函数使用 `handleX`，传入组件的回调 prop 使用 `onX`。
- CSS 类名使用小写连字符，并表达结构或角色，如 `.post-card`，不使用 `.green-box`、`.left2` 等表现型名称。
- `data-*` 属性用于脚本钩子时以领域命名，如 `data-search-dialog`，禁止用 CSS 类作为脚本契约。

### 7.3 Astro 组件规则

- frontmatter 区按“类型导入、值导入、Props、解构、派生数据”排序。
- 所有非平凡组件定义 `Props` 接口。
- 不在模板表达式中编写复杂排序、过滤或多层条件；在 frontmatter 中先计算。
- 组件 scoped 样式只处理局部结构；设计 token、排版和通用状态放全局层。
- 避免过深 slot 嵌套；公共 layout slot 名称必须写入架构文档。
- 客户端 `<script>` 必须幂等，可安全处理页面恢复或视图过渡后的重复初始化。
- 不使用 `set:html` 渲染不可信字符串。
- Astro 页面和组件中不得直接硬编码站点域名、作者信息或社交地址。

### 7.4 HTML 规则

- 使用 HTML5 doctype、`lang="zh-CN"`、正确 viewport 和 UTF-8 声明。
- 使用有效、语义化 HTML；不使用已废弃元素。
- 属性使用双引号，与 Google HTML/CSS 规范和 Prettier 输出保持一致。
- 所有表单控件有 label；图标按钮有可访问名称。
- 不使用正 tabindex；仅允许 `0` 和必要时 `-1`。
- DOM 顺序必须与阅读和键盘顺序一致，CSS 不得制造语义顺序错乱。
- 装饰 SVG 不可被辅助技术重复朗读。
- 外部资源全部 HTTPS。

### 7.5 CSS 规则

- 使用原生 CSS，不引入 Sass、Less、Stylus、CSS-in-JS 或 utility-first 框架。
- `tokens.css` 是颜色、间距、圆角、阴影、层级、动效时间和排版尺度的唯一设计 token 来源。
- 使用 CSS 自定义属性表示主题值，禁止在多个组件重复魔法颜色。
- 选择器保持低特异性，优先单类选择器和 `:where()`；禁止 ID 选择器参与样式。
- 嵌套最多两层，不依赖页面祖先链定位局部组件。
- 每个声明单独一行，规则之间空一行，删除无效或重复声明。
- 类名表达用途而非表现，禁止无意义缩写。
- 响应式采用移动优先；断点来自 token，不在组件中随意创造相近断点。
- 必须提供 light/dark 颜色对；主题以 `data-theme` 和系统偏好驱动。
- 所有关键动画提供 reduced-motion 静态替代。
- `!important` 默认禁止；仅第三方嵌入覆盖层允许，并需在同一规则注释原因。
- 不使用远程 CSS `@import`；字体和样式由本地构建管理。

### 7.6 JavaScript 与浏览器脚本规则

- 新脚本使用 TypeScript，由构建器输出模块。
- 事件监听器必须具备清理或幂等重复绑定策略。
- 滚动监听使用 passive listener；高频更新使用 `requestAnimationFrame` 合并。
- 优先 IntersectionObserver 和 ResizeObserver，不在滚动事件中遍历并测量大量元素。
- localStorage 只保存非敏感用户偏好，键名带站点前缀和版本。
- 查询 DOM 后必须处理元素不存在的情况。
- 动态生成 HTML 时使用 DOM API 或组件模板，不拼接不可信 HTML 字符串。
- 功能检测优先于浏览器识别。

### 7.7 JSON、YAML 与配置规则

- JSON key 使用 `camelCase`。
- 配置必须有 schema 或 TypeScript 类型，不接受无约束对象。
- 环境差异通过明确配置表达，不在代码中根据主机名散落判断。
- `package-lock.json` 必须提交；CI 使用 `npm ci`。
- 依赖版本升级单独提交，并记录构建与视觉回归结果。

### 7.8 文档规则

- 每篇工程文档只有一个 H1，开头用 1 至 3 句说明目的和适用范围。
- 标题具体、唯一、层级连续；中文标题避免无意义的“相关内容”“其他”。
- 命令、文件名、字段名和代码标识符使用行内代码格式。
- 示例命令必须可复制，并说明运行目录与预期结果。
- README 只承担项目入口，详细架构、测试和写作说明放到 `docs/` 并由 README 链接。
- 代码行为改变时同步更新文档；过期文档删除而不是加“可能已失效”。

## 8. 视觉与页面设计规划

### 8.1 视觉原则

- 借鉴目标站的暗色赛博气质、绿色流光、固定导航、欢迎首屏和卡片动效，但建立 EntropyXi 自有品牌。
- 内容阅读优先于装饰；文章页的视觉噪声显著低于首页。
- 所有装饰层设置 `pointer-events: none`，不得遮挡选择文本、链接或公式滚动。
- 绿色只作为强调和状态色，正文保持高可读中性色。
- 桌面、平板和手机使用同一信息架构，不通过隐藏重要内容“适配”。

### 8.2 首页

- 固定页头：站点名称、首页、归档、分类、标签、关于、搜索、GitHub。
- 首屏 Hero：EntropyXi 品牌标题、技术方向简介、CSS/SVG 流光背景、向下引导。
- 公告卡片：静态配置驱动，可关闭或隐藏。
- 文章卡片：桌面 3 列、平板 2 列、手机 1 列；显示标题、描述、日期、分类、标签和可选封面。
- 卡片进入动画由 IntersectionObserver 添加状态类；无 JS 时全部直接可见。
- 分页使用真实链接，不做必须依赖 JavaScript 的无限滚动。

### 8.3 文章页

- 主栏承载标题、元信息、正文、上下篇和评论。
- 桌面右侧为 sticky TOC；手机使用可关闭抽屉。
- TOC 数据来自 Astro Markdown 渲染返回的 headings，不重新扫描并解析正文 HTML。
- 顶部显示阅读进度；右下角提供目录、评论和回顶按钮。
- 阅读模式属于第二阶段增强；必须保留可复制 URL、刷新和浏览器返回行为。
- 代码块支持语言标识、横向滚动和复制；复制失败显示可访问反馈。
- 长公式容器可横向滚动，不改变行内公式基线。

### 8.4 列表页面

- 归档：按年月分组的时间轴，支持纯静态年份锚点。
- 分类：展示层级面包屑、分类数量和文章列表。
- 标签：标签云只作为补充，同时提供按字典序排列的可访问列表。
- 搜索：独立 `/search/` 页面与全局弹窗共用一个搜索组件。
- 404：提供返回首页、搜索和近期文章，不自动跳转。

### 8.5 主题和动效

- 默认遵循系统主题，用户选择写入本地偏好。
- 主题切换在首屏绘制前应用，避免闪烁。
- reduced motion 下移除文字逐字动画、磁吸光标、视图滑动和循环流光，仅保留必要状态变化。
- 动画持续时间和缓动函数来自 token；禁止每个组件自定义相近数值。

## 9. 数学渲染迁移专项

### 9.1 技术验证

建立包含以下语法的公式夹具：

- 行内 `$...$` 与显示 `$$...$$`。
- `\(...\)` 与 `\[...\]` 分隔符；若扫描确认旧文未使用，也必须保留负向夹具证明审计不会误报。
- `aligned`、`align`、`equation`、`cases`。
- `bmatrix`、`pmatrix`。
- `\tag`、`\text`、`\operatorname`。
- `\mathbb`、`\boldsymbol`、`\underbrace`。
- 中文文本、长公式和嵌套上下标。

比较两个候选方案：

1. Unified Markdown processor + 构建期 MathJax 输出。
2. Markdown 保留数学节点 + 仅 `math: true` 页面按需加载 MathJax 3。

选择标准按顺序为：渲染正确性、可访问性、HTML 体积、首屏性能、实现复杂度。数学 spike 和分隔符策略必须在阶段 2 正文冻结前完成并写 ADR；任何一项导致文章公式变化都不能直接上线。

### 9.2 公式验收

- 30 篇文章逐篇检查渲染完成后无可见的原始分隔符泄漏。
- 为公式最复杂的至少 5 篇文章建立固定视口截图基线。
- 检查行内公式基线、中文标点间距、显示公式编号和矩阵布局。
- 检查 360px 宽屏下长公式可滚动且页面本身不横向溢出。
- 检查暗色模式下公式颜色、选中、复制和打印。
- 若选择构建期 MathJax：静态扫描构建 HTML，断言无 TeX 分隔符、未处理环境或 MathJax 错误标记。
- 若选择按页客户端 MathJax：静态扫描只断言 `math: false` 页面无 TeX；`math: true` 页面由 Playwright 等待 MathJax 渲染 Promise 完成，再断言无可见原始分隔符和错误，并在同一时点采集截图。

## 10. 搜索、评论、SEO 与发布能力

### 10.1 Pagefind 搜索

- `npm run build` 顺序为 `astro check`、`astro build`、Pagefind 索引、输出审计。
- 文章正文容器标记 `data-pagefind-body`，导航、TOC、页脚和 Giscus 不进入索引。
- 标题、分类、标签通过 Pagefind metadata/filter 标记暴露。
- `<html lang="zh-CN">` 必须存在，用于 Pagefind 语言检测、索引隔离和中文 UI。npm 提供的 Pagefind extended 支持无空格语言分词；仍以真实中文查询 E2E 判断结果质量。
- 搜索组件延迟导入索引，输入防抖，支持键盘上下选择、Enter 打开和 Escape 关闭。
- E2E 至少验证中文完整词、英文术语、无结果和特殊字符查询。
- 旧 `/search.xml` 作为兼容入口保留一个迁移周期，由静态 endpoint 生成；新 UI 不依赖它。若阶段 0 证实生产路径不同，以线上 manifest 路径为准。

### 10.2 Giscus

- 把现有 Giscus 配置集中到 `src/data/site.ts`，不散落在模板。
- 继续使用 `data-mapping="pathname"`，路径验证通过后再测试评论。
- 根据主题变化向 Giscus iframe 发送官方主题切换消息。
- 评论脚本仅文章页加载；列表页和搜索页不得加载。
- 评论加载失败不影响正文，不进行无限重试。

### 10.3 SEO

- `BaseLayout` 统一生成 title、description、canonical、Open Graph、Twitter Card 和语言信息。
- 文章页使用 `BlogPosting` JSON-LD；结构化数据来自已校验 schema。
- canonical 使用 `Astro.site` 与规范 pathname 构造，禁止字符串拼接域名。
- 自动生成 Sitemap，并排除草稿、测试和内部页面。
- RSS/Atom 内容至少包括标题、描述、发布日期和规范链接。
- `robots.txt` 明确指向 Sitemap。
- `/sitemap.xml` 由 `src/pages/sitemap.xml.ts` 生成单文件；Atom 同样由 `atom.xml.ts` 明确实现并用 XML 解析器测试，不假设官方 RSS 集成自动生成 Atom。
- 404、分页和聚合页 title/description 不得重复。

## 11. 测试架构

### 11.1 静态检查

- `npm run format:check`：Prettier。
- `npm run lint`：ESLint 与 Stylelint。
- `npm run check:types`：`astro check`。
- `npm run audit:content`：frontmatter、日期、URL 唯一性、资源和禁止语法。
- `npm run check`：串联全部快速质量门。

### 11.2 单元测试

使用 Vitest 覆盖：

- `buildPostPath()` 对中文、空格、括号、冒号和多层目录的处理。
- URL 编码/解码往返，包括空格、`%20`、全角标点、中文和保留字符。
- 日期解析和 Asia/Shanghai 格式化。
- 文章稳定排序。
- 分类树和标签聚合。
- canonical、Open Graph 和 JSON-LD 生成。
- Atom 与 Sitemap XML 的结构、规范 URL 和稳定快照。
- 摘要、阅读时间与 TOC 纯函数。
- 内容迁移脚本的幂等性。

### 11.3 构建输出审计

重写 `scripts/verify-build.js` 为 TypeScript 输出审计，检查：

- `dist/index.html` 存在。
- 生成文章数等于非草稿文章数。
- 历史 URL manifest 中每个文章及聚合 pathname 都对应正确的 `index.html`，资源 pathname 对应文件。
- 每篇文章只有一个 H1、非空 description 和 canonical。
- title、description、渲染日期、分类、标签与阶段 0 manifest 一致；分页和聚合页 metadata 不重复。
- 不发布 `.obsidian`、README、源 Markdown、草稿或密钥样式文件。
- HTML 不包含 `undefined`、`[object Object]` 或本机绝对路径；数学检查按 §9.2 所选方案执行。
- 所有站内 `href` 和 `src` 对应输出资源。
- `/sitemap.xml`、Atom/RSS、robots 和 404 存在且 XML 可解析。
- Pagefind 索引存在，正文索引结果集合与非草稿文章集合一致；导航、TOC、页脚和 Giscus 不进入正文索引。

### 11.4 端到端测试

Playwright 在构建后的 `astro preview` 上运行，覆盖 Chromium、Firefox 和 WebKit 的关键路径：

- 首页加载、导航、分页和卡片链接。
- 历史中文 URL 直接访问。
- 文章目录跳转、回顶、代码复制和评论占位。
- 分类、标签、归档和搜索。
- 主题切换持久化。
- 移动菜单、移动 TOC 和横向公式滚动。
- 键盘导航和焦点返回。
- reduced motion 行为。
- 404 页面。
- axe-core 扫描首页、文章、搜索、分类和 404；不得出现 serious 或 critical 级违规，其他级别必须有记录和处理结论。

### 11.5 视觉与人工测试

- 视觉截图只覆盖稳定页面和稳定区域，不把 Giscus、日期波动内容或循环动画纳入像素比较。
- 视口至少包括 360×800、768×1024、1440×900。
- 使用 Playwright `toHaveScreenshot` 管理基线；基线只通过带原因的独立变更更新，CI 不自动接受新图。固定字体、主题、时区和动画状态，数学页按 §9.2 等待渲染完成。
- 人工检查 Chrome、Firefox、Safari/WebKit 等价行为。
- 发布前使用真实生产构建检查断网降级、慢速网络、字体失败和第三方脚本失败。

## 12. CI/CD 设计

### 12.1 GitHub Actions 流程

`source` 分支继续作为部署触发分支，直到用户主动调整分支策略。建立两个 workflow：PR 的 `quality.yml` 只验证不部署；`source` push 的 `deploy.yml` 复用同一质量命令后部署。新流程：

1. Checkout，`fetch-depth: 0`。
2. Setup Node，版本由 `.nvmrc` 或 `package.json#engines` 明确指定。
3. `npm ci`。
4. `npm run check`：格式、Lint、类型、内容、单元、构建、Pagefind 和输出审计各执行一次。
5. `npm audit --audit-level=high`，并运行密钥扫描；有明确误报时只能通过有负责人、原因和到期日的 allowlist 处理。
6. 安装 Playwright 浏览器并运行 E2E、axe 与视觉测试；PR 跑 Chromium/Firefox/WebKit，部署分支至少跑 Chromium，其他浏览器结果必须来自同一 commit 的已通过 PR。
7. 运行 Lighthouse CI 固定页面与阈值。
8. 上传 `dist/` 为 GitHub Pages artifact。
9. Deploy Pages，并明确确认仓库 Pages Source 为 GitHub Actions。
10. 以 manifest 为输入，对生产域名的 30/30 文章和全部冻结聚合 URL 检查 HTTP 200、最终 pathname 与 canonical；同时检查首页、Sitemap、Atom、搜索资源和一篇数学代表页。

### 12.2 构建脚本建议

```json
{
  "scripts": {
    "dev": "astro dev",
    "preview": "astro preview",
    "legacy:clean": "hexo clean",
    "legacy:build": "hexo generate",
    "legacy:check": "npm run legacy:clean && npm run legacy:build && node scripts/verify-build.js",
    "check:types": "astro check",
    "audit:content": "tsx scripts/audit-content.ts",
    "build:site": "astro build",
    "build:search": "pagefind --site dist",
    "audit:output": "tsx scripts/audit-output.ts",
    "build": "npm run build:site && npm run build:search && npm run audit:output",
    "test:unit": "vitest",
    "test:e2e": "playwright test",
    "lint": "eslint . && stylelint \"src/**/*.{css,astro}\"",
    "format": "prettier --write .",
    "format:check": "prettier --check .",
    "migrate:content": "tsx scripts/migrate-content.ts",
    "new:post": "tsx scripts/new-post.ts",
    "check": "npm run format:check && npm run lint && npm run check:types && npm run audit:content && npm run test:unit -- --run && npm run build"
  }
}
```

最终脚本以实际依赖和运行时间调整，但命令职责必须保持清晰，禁止一个名字模糊的脚本隐式修改源文件。

## 13. 分阶段实施计划

### 阶段 0：创建迁移安全网

目标：获得可比较、可回滚的 Hexo 基准。

任务：

- [x] 确认工作树干净并记录当前 commit。
- [x] 创建 `codex/astro-migration` 分支。
- [x] 运行现有 `npm run check`，保存通过结果。
- [x] 生成旧站 manifest：全部文章、分页、归档年月、分类、标签、静态资源、`search.xml`、RSS 与 Sitemap URL，以及每页 canonical、title、description、渲染日期、分类和标签。
- [x] 用生产站爬取结果与本地 Hexo manifest 交叉核对，冻结真实的百分号编码、空格、全角标点、大小写、重定向和尾斜杠；不一致时以生产可访问行为为准并记录原因。
- [x] 审计 30 篇 frontmatter 是否具备 title、description、date、tags、categories；缺字段不得静默合成，必须生成逐篇人工确认清单。
- [x] 扫描全部数学分隔符和环境，确认 `\(...\)`、`\[...\]` 的真实使用情况并生成公式夹具输入。
- [x] 选取首页、归档、分类、标签、关于和 5 篇复杂公式文章，保存基准截图。
- [x] 记录当前 Giscus 配置、搜索、RSS、Sitemap 与部署行为。
- [x] 标记 Hexo 可回滚点，例如 `pre-astro-migration`。
- [x] 在本阶段不得改变生产部署。

退出条件：旧站可重复构建，URL 清单与截图基准已入库，回滚点可定位。

### 阶段 1：建立 Astro 骨架

目标：建立严格、可测试的空 Astro 静态项目。

任务：

- [x] 安装 Astro、TypeScript 支持和必要官方集成。
- [x] 建立 `astro.config.ts`，配置 `site`、`trailingSlash: 'always'`、静态输出、`publicDir: 'astro-public'` 和 Markdown processor 候选；Hexo 继续独占其生成目录 `public/`。
- [x] 建立 `tsconfig.json` 严格配置与路径别名。
- [x] 建立 ESLint、可解析 `.astro` 的 Stylelint、Prettier、`.prettierignore`、EditorConfig 和固定 LF 的 `.gitattributes`。
- [x] 建立基础目录、`BaseLayout`、空首页和 404。
- [x] 建立 Vitest 与 Playwright 最小可运行测试。
- [x] 更新 `.gitignore` 排除 Astro、dist、Pagefind 和测试产物。
- [x] 按 §12.2 显式保留 `legacy:clean`、`legacy:build`、`legacy:check`；新旧 `build`/`check` 的语义写入 README，禁止同名覆盖后仍按旧含义验收。
- [x] 在真实文章扫描结果上完成数学渲染 spike，决定分隔符和构建期/客户端策略，写 ADR 后才能进入阶段 2。
- [x] 验证 `/sitemap.xml`、`/atom.xml`、旧 `search.xml` endpoint 的最小输出与 XML 解析测试。

退出条件：`npm run check:types`、Lint、单元测试、Astro build 和最小 E2E 全部通过。

### 阶段 2：内容 schema 与自动迁移

目标：无损迁移 30 篇文章和静态资源。

任务：

- [x] 定义内容 collection schema。
- [x] 编写幂等 `migrate-content.ts`，只从 `source/_posts/**/*.md` 读取，排除隐藏目录。
- [x] 统一日期为带 `+08:00` 的 ISO 8601。
- [x] 逐篇比较旧站渲染日期与新值，证明补 `+08:00` 符合旧站语义；差异逐篇记录，不凭假设批量修正。
- [x] 统一 `mathjax` 为 `math` 布尔值。
- [x] 为每篇文章生成并冻结 legacy permalink。
- [x] 删除或忽略 `<!-- more -->`，保留 `description`。
- [x] 保持 Markdown 正文、公式和引用内容不变；只有数学 ADR 明确批准的分隔符归一化可加入差异白名单。
- [x] 把现有有效图片迁到 `astro-public/<旧资源 pathname>` 并修正引用，保持资源 URL 等价。
- [x] 对缺失 Obsidian 图片生成阻塞报告，逐项补资源或明确删除引用。
- [x] 确认 `体系框架图.jpg` 的归属。
- [x] 建立 content audit 并让错误信息定位到文件和字段。
- [x] 比较统一 LF、剔除 frontmatter 与 `<!-- more -->` 后的正文哈希；若数学 ADR 批准分隔符归一化，脚本必须把该类差异单独列出，其他差异一律阻塞。

退出条件：30 篇内容全部通过 schema，资源无悬空引用，permalink 全局唯一且等于旧 URL manifest。

### 阶段 3：最小静态文章页与数学兼容

目标：完成内容等价、无动画依赖、桌面和移动端均可稳定阅读的静态文章页面；本阶段不复刻 NexT 视觉，也不提前实现 Ayeez 风格。

任务：

- [x] 实现 catch-all 文章静态路由。
- [x] 实现 `PostLayout`、标题、日期、分类、标签与正文排版。
- [x] 集成阶段 1 已由 ADR 选定的数学方案，并通过完整公式夹具。
- [x] 实现 TOC 数据、桌面 sticky 目录和移动抽屉。
- [x] 实现代码高亮、语言标签、复制按钮和横向滚动。
- [x] 实现图片基础样式、灯箱的必要性评估和无 JS 降级。
- [x] 迁移 Giscus，保持 pathname。（用户决策：本阶段不引入 Giscus，文章页不加载评论脚本）
- [x] 实现上一篇/下一篇。
- [x] 对 30 篇文章运行输出审计。
- [x] 对 5 篇复杂公式文章运行视觉回归。（本阶段为桌面/移动布局烟雾测试，像素基线阶段 5 补）

退出条件：所有历史文章 URL 200，公式与正文无已知回归，评论路径保持一致。

### 阶段 4：聚合页面与站点能力

目标：替代 Hexo 生成器插件提供的全部静态能力。

任务：

- [x] 首页文章列表与静态分页，pathname 与 manifest 逐字节一致。
- [x] 归档页及 manifest 中存在的年月/分页路由。
- [x] 分类索引、分类详情、层级展示和旧分页路由。
- [x] 标签索引、标签详情和旧分页路由。
- [x] 关于页迁移。
- [x] Pagefind 构建和中文搜索 UI。
- [x] 保留 manifest 中的旧 `search.xml` 兼容 URL，但新搜索不依赖该 XML。
- [x] Atom/RSS feed。
- [x] Sitemap、robots 和 canonical。
- [x] 结构化数据与社交分享 metadata。
- [x] 完整 404 页面。
- [x] 用新实现替代旧 verify-build 的全部能力。

退出条件：站点功能矩阵完成，搜索、RSS、Sitemap 和聚合页通过自动化测试。

### 阶段 5：赛博视觉系统

目标：在不改变内容、URL 与静态可读基线的前提下完成类 Ayeez 的目标风格；借鉴信息层级、暗色赛博气质和交互节奏，但使用本站自有品牌、组件和素材。

任务：

- [ ] 定义设计 token 和 light/dark 主题。
- [ ] 建立 Ayeez 参考页面的功能/视觉矩阵，区分可借鉴模式与不可复制的品牌、素材和内容。
- [ ] 实现固定页头、桌面导航与移动菜单。
- [ ] 实现首页 Hero、品牌文案、SVG/CSS 流光与装饰线。
- [ ] 实现公告卡片和文章卡片网格。
- [ ] 实现滚动显现、hover/focus 状态与 reduced-motion 版本。
- [ ] 实现阅读进度、回顶和目录浮动按钮。
- [ ] 评估并实现可选阅读模式。
- [ ] 实现主题切换无闪烁。
- [ ] 完成移动端、触控和键盘细节。
- [ ] 运行 Lighthouse、可访问性和视觉回归。
- [ ] Lighthouse CI、axe 和视觉基线均按 §3.6、§3.7、§11.4、§11.5 的量化标准通过。

退出条件：视觉验收通过，性能和可访问性不低于预算，禁用 JavaScript 后仍可完整阅读导航。

### 阶段 6：切换 CI/CD

目标：让 Astro 成为唯一生产构建。

任务：

- [ ] 更新 GitHub Actions，移除 Pandoc 安装。
- [ ] 将 Pages artifact 从 `public/` 改为 `dist/`。
- [ ] 在 PR 或临时 Pages 环境验证完整产物。
- [ ] 检查 GitHub Pages 设置、base 路径和站点根地址。
- [ ] 明确把 GitHub Pages Source 设置为 GitHub Actions，并把该设置的旧值与恢复步骤写入回滚文档。
- [ ] 执行生产前 URL manifest 对比。
- [ ] 部署后以 manifest 对生产域名运行全量 URL、canonical 和资源 smoke check，不只抽查代表文章。
- [ ] 观察评论、搜索、404、RSS 和公式错误 7 天；出现任一历史文章 404、canonical/pathname 漂移、Giscus 新线程映射或已确认公式回归即回滚。
- [ ] 保留回滚到最后 Hexo commit 的明确操作说明。

退出条件：生产站由 Astro 构建，全量 smoke check 通过，并完成 7 天观察期且未触发回滚条件。

### 阶段 7：清理 Hexo

目标：删除双重实现并收敛文档。

任务：

- [ ] 删除 Hexo、NexT、Pandoc、Stylus 和旧生成器依赖。
- [ ] 删除 `_config.yml`、`_config.next.yml`、`scaffolds/`、旧注入模板和废弃脚本。
- [ ] 删除迁移期临时兼容代码和无用截图。
- [ ] 更新 README、贡献指南、写作指南、部署和故障排查。
- [ ] 保留 URL manifest、迁移 ADR 和第三方许可证记录。
- [ ] 运行全量 CI 和一次干净环境 `npm ci && npm run check`。

退出条件：仓库只含 Astro 生产路径，无已引用的 Hexo 文件或依赖，文档与实际命令一致。

## 14. 代码审查规则

每个 PR 按 Google Engineering Practices 的关注点审查：设计、功能、复杂度、测试、命名、注释、风格和文档。

### 14.1 作者清单

- [ ] PR 只包含一个清晰目标。
- [ ] 描述用户可见变化、技术方案、风险和回滚。
- [ ] 关联计划阶段和验收条件。
- [ ] 新行为有测试，修复有回归测试。
- [ ] 不夹带无关格式化。
- [ ] 文档与代码同步。
- [ ] 本地运行与 CI 等价的命令。
- [ ] 提供必要的桌面、移动、暗色和 reduced-motion 截图。

### 14.2 审查者清单

- [ ] 设计是否符合分层和依赖方向。
- [ ] 是否存在更简单的平台能力或 Astro 原生方案。
- [ ] URL、内容和评论兼容是否被破坏。
- [ ] 失败、空数据、无 JS、慢网络和第三方失败路径是否合理。
- [ ] 类型是否表达真实领域约束。
- [ ] 可访问性和键盘行为是否完整。
- [ ] 性能是否引入不必要 JavaScript 或持续动画。
- [ ] 测试是否验证行为而非实现细节。
- [ ] 注释是否解释原因，文档是否准确。
- [ ] 审查意见标注严重程度：`blocking`、`major`、`minor`、`nit`。

### 14.3 严重程度与单人维护规则

- `blocking`：会破坏内容、历史 URL、评论映射、部署、安全或已冻结验收标准，合并前必须解决。
- `major`：明显影响架构、可维护性、可访问性或性能；默认阻塞，只有记录负责人、原因和移除日期后才能延期。
- `minor`：非阻塞的局部质量问题，应创建明确后续项。
- `nit`：纯建议，不得伪装成阻塞项。
- 涉及 URL、数学、内容迁移或部署切换的 PR 优先邀请第二人审查。单人维护时，作者必须先让全部自动门禁完成，再脱离编辑上下文进行一次完整自审，并在 PR 中逐项填写作者与审查者清单；不能以“只有一个人”为理由跳过门禁。

## 15. 风险登记表

| 风险                                   |   概率 | 影响 | 缓解措施                                                   | 阻塞上线           |
| -------------------------------------- | -----: | ---: | ---------------------------------------------------------- | ------------------ |
| 中文多层旧 URL 改变                    |     中 | 极高 | permalink 冻结、manifest 全量比较、E2E 直访                | 是                 |
| 旧分页/聚合 URL 被 catch-all 遗漏      |     中 |   高 | 线上 manifest、显式分页路由、输出与生产全量检查            | 是                 |
| MathJax 高级公式不兼容                 |     高 | 极高 | 公式夹具、30 篇扫描、5 篇视觉基线、保留客户端 MathJax 备选 | 是                 |
| Giscus 因 pathname 变化产生新线程      |     中 |   高 | 路径先验收，保持 trailing slash 和 mapping                 | 是                 |
| Obsidian 图片缺失                      | 已发生 |   中 | 资源审计阻塞、逐项补齐或明确处理                           | 是                 |
| Astro/Markdown processor 改变原始 HTML |     中 |   高 | 内容哈希、代表文章 DOM 对比、禁止未审计 raw HTML           | 是                 |
| Hexo `public/` 覆盖 Astro 静态源       |     高 |   高 | Astro `publicDir: 'astro-public'`，共存期目录隔离          | 是                 |
| 动画导致低端设备卡顿                   |     中 |   中 | CSS/SVG 优先、减少持续动画、性能预算、reduced motion       | 否，但阻塞视觉阶段 |
| ClientRouter 导致脚本重复初始化        |     中 |   高 | 默认使用原生导航，启用前做生命周期专项测试                 | 是                 |
| Pagefind 中文结果质量不足              |     中 |   中 | extended build、中文 E2E、保留标签/分类导航                | 否                 |
| GitHub Pages base/canonical 配错       |     低 |   高 | `site` 固定、无 base、部署 smoke test                      | 是                 |
| 一次性大改难以回滚                     |     中 |   高 | 阶段提交、Hexo 基线标签、迁移与视觉分离                    | 是                 |
| 第三方依赖升级破坏构建                 |     中 |   中 | lockfile、Dependabot/手动单独升级、CI 全量验证             | 否                 |

## 16. 回滚策略

### 16.1 切换前

- 生产仍由 Hexo workflow 部署。
- Astro workflow 使用不同名称，未验证前不授予部署步骤。
- Astro 静态源使用 `astro-public/`，Hexo 继续生成到 `public/`；每个阶段均可删除 Astro 新文件而不影响 Hexo 构建。

### 16.2 切换后

- 保留最后成功 Hexo commit/tag。
- 如发现 P0 问题，回退部署 workflow 和源 commit，不在生产上紧急拼补复杂迁移代码。
- 恢复阶段 6 记录的 GitHub Pages Source 设置和最后成功 Hexo workflow；回滚完成后再次执行旧站 manifest 抽查。
- 回滚后保留失败产物、Actions 日志、URL/公式审计报告，用独立修复分支处理。
- 评论数据在 GitHub Discussions，不随前端回滚丢失；关键是恢复相同 pathname。

## 17. 最终验收清单

### 内容与链接

- [ ] 30/30 文章生成。
- [ ] 历史文章 URL 在本地产物和生产域名均为 30/30 返回 200，pathname 编码、尾斜杠和 canonical 一致。
- [ ] manifest 中旧分页、归档、分类、标签、搜索 XML 和静态资源 URL 均得到明确保留或书面处置。
- [ ] 标题、描述、日期、分类、标签与旧站一致。
- [ ] 所有图片存在且 alt 合理。
- [ ] 无 Obsidian Wiki 图片语法残留。
- [ ] 无 `.obsidian` 文件发布。

### 数学与排版

- [ ] 按 §9.2 所选方案验证，浏览器完成渲染后无可见原始数学分隔符泄漏。
- [ ] 高级环境和 `\tag` 正常。
- [ ] 行内公式基线正常。
- [ ] 长公式移动端可滚动。
- [ ] 暗色和打印模式正常。

### 功能

- [ ] 首页、分页、文章、归档、分类、标签、关于正常。
- [ ] 中文和英文搜索正常。
- [ ] Giscus 正常且 pathname 未变化。
- [ ] RSS/Atom、Sitemap、robots、404 正常。
- [ ] 主题、移动菜单、TOC、复制、回顶正常。

### 工程质量

- [ ] 格式、Lint、类型、内容审计通过。
- [ ] 单元、构建审计、E2E、视觉回归通过。
- [ ] axe 关键页面无 serious/critical 违规，Lighthouse CI 达到性能预算。
- [ ] 依赖审计与密钥扫描通过，无未批准的高危依赖或泄露配置。
- [ ] README、架构、测试、写作和部署文档准确。
- [ ] 干净环境可以仅凭文档完成安装、构建、预览和部署。

### 用户体验

- [ ] 360px 到宽屏布局无横向页面溢出。
- [ ] 键盘可完成全部操作。
- [ ] reduced motion 下无强制大幅动画。
- [ ] JavaScript 失败时正文和导航可用。
- [ ] 性能预算和 WCAG 2.2 AA 目标满足。

## 18. 明确不在本次范围内

- 后台文章管理系统。
- 用户注册登录。
- 数据库和 Spring Boot 后端。
- 实时说说、朋友圈或评论聚合 API。
- 服务端站内搜索。
- 自动 AI 摘要或封面生成。
- 在内容等价尚未完成前重写全部文章。

这些功能如未来需要，必须作为新的产品需求和 ADR 单独设计，不能污染本次静态迁移边界。

## 19. 建议的首个实施批次

首个批次只执行阶段 0 和阶段 1，预期交付：

1. Hexo 基线 URL 与截图清单。
2. 可独立运行的 Astro 空骨架。
3. 严格 TypeScript、Lint、格式与测试工具链。
4. 一页不带复杂视觉的静态首页和 404。
5. 不修改生产 workflow，不迁移正文，不删除 Hexo。

首批通过后再执行内容迁移，可把最大风险控制在 URL 和数学两条可独立验证的链路内。

## 20. DeepSeek 双精度审查记录

### 20.1 审查配置

- 调用方式：`dsh --profile web`，通过 Chrome 打开本地 DeepSeek Harness。
- 模型：`DeepSeek-V4-Flash`。
- 思考程度：`Max`。
- 权限：`Read Only`；提示词明确禁止读取或修改网页当前工作区，只允许审查粘贴的计划文本。
- 第一轮：迁移架构、内容、URL、数学、Giscus、Pagefind、GitHub Pages 与阶段依赖审查；约 3 分 25 秒。
- 第二轮：Google 规范映射、规则可执行性、质量门禁、测试、安全、性能、可访问性与回滚审查；使用新会话独立完成，约 5 分 31 秒。
- 两轮结论均为“有条件可执行”；初稿问题已在本修订稿中闭环，实施仍须逐阶段满足退出条件。

### 20.2 已吸收的主要意见

- 隔离 Hexo 输出 `public/` 与 Astro 静态源，避免共存期互相删除或污染。
- 把旧分页、归档、分类、标签、搜索 XML 和资源 URL 纳入线上 manifest，而不只验证 30 篇文章。
- 把数学 spike 前置到正文冻结前，并让构建期/客户端 MathJax 使用不同且可执行的验收方法。
- 固化历史图片 URL、正文 Prettier 边界、行尾归一化、时区逐篇核对和脚本命名映射。
- 增加部署后全量生产 URL 检查、Stylelint 的 `.astro` 覆盖、axe、Lighthouse CI、依赖/密钥门禁和 7 天观察期。
- 明确 `/sitemap.xml` 的兼容实现、视觉基线更新流程、严重程度定义和单人维护审查方式。

### 20.3 经官方资料复核后未照抄的意见

- 第一轮声称 Astro 不能修改 `publicDir`。官方配置参考明确允许设置自定义目录；因此保留其指出的“目录冲突”风险，但采用 `publicDir: 'astro-public'`，不采用错误前提。
- 第二轮声称 Pagefind 不做中文分词。Pagefind 官方文档说明 npm 默认 extended 版本支持中文等无空格语言的 segmentation；因此保留真实中文 E2E 和质量兜底，不把功能描述降级成错误结论。
- 审查建议把所有工具都设为硬门禁。本计划只保留能对应明确风险的门禁，并通过单次 `npm run check` 去重，避免为 30 篇静态博客重复执行相同检查。

### 20.4 修订后放行判断

本计划现在可以启动阶段 0 和阶段 1。进入阶段 2 前仍有四项不可跳过的实证条件：线上/本地 manifest 已交叉核对、30 篇 frontmatter 审计完成、数学 ADR 已选定、`astro-public/` 隔离与新旧脚本映射已由实际构建验证。
